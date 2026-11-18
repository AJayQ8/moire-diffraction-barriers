"""Replay the fixed 30-degree elastic-frame control on public MoS2 diffraction."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import map_coordinates

from author_detector_replay import detector_intensity
from author_release import load_author_release
from bilayer_relaxation import make_grid, sample_disregistry, wavevectors
from observe_bilayers import read_input_metadata
from run_heterobilayer_sparse_geometry import solve_from_geometry
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_camera import normalize_camera
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_constrained_equilibrium import score_fields
from run_mos2_frame_sensitivity import fold_masks
from run_mos2_intensity_only_replay import disk_images
from run_mos2_local_intensity_geometry import geometric_folds
from run_mos2_nonlinear_controls import minimize_all
from run_mos2_observation_controls import (
    HALF_WINDOW, INPUT_NAMES, NOMINAL, gaussian_gain, raster_coordinates, render_traces, sample_cuts, score,
)
from run_mos2_uniaxial_check import decompose
from run_mos2_wall_landscape import get_observer, width_pair
from run_mos2_wall_width_transfer import DISTANCE, geometry_sections
from run_registry_prediction import rotation
from run_wall_heterostrain import SOURCE
from shared_compute import atomic_json, checkpoint_guard


TARGETS = ("DS13", "DS14")
CONTROLS = ("DS1", "DS2")
V005 = "mos2_geometry_probe_v005.json"
LOCAL_SCORES = "moire_longitudinal_transfer_replay_v01.json"
CAMERAS = ("transferred_even", "separate_disjoint")
ANCHOR = ("MoS2", "MoS2", "P")


def variants(scan, gradient):
    """Rotate the domain axes of the measured registry gradient without rescaling it."""
    native = np.eye(2)
    minus = rotation(-np.pi / 6)
    plus = minus.T
    _, controls = decompose(gradient)
    models = {
        "nominal": (gradient, native),
        "remove_mean_strain": (controls["no_mean_dilation"], native),
        "rotate_minus_30": (gradient @ minus, native),
    }
    if scan in TARGETS:
        alternate = rotation(-np.pi / 3)
        converted = alternate @ gradient @ plus
        models["alternate_nominal"] = (alternate @ gradient, alternate)
        models["alternate_rotate_plus_30"] = (converted, alternate)
        np.testing.assert_allclose(converted, alternate @ models["rotate_minus_30"][0] @ alternate.T,
                                   rtol=0, atol=1e-14)
    else:
        models["rotate_plus_30"] = (gradient @ plus, native)
    for label, (g, _) in models.items():
        if "rotate" in label:
            np.testing.assert_allclose(np.linalg.svd(g, compute_uv=False),
                                       np.linalg.svd(gradient, compute_uv=False), atol=1e-14, rtol=0)
    return models


def to_native(motion, basis, frame):
    """Transport an equivalent registry indexing back onto the original periodic grid."""
    transform = np.linalg.solve(basis, frame @ basis)
    permutation = np.rint(transform).astype(int)
    np.testing.assert_allclose(transform, permutation, rtol=0, atol=1e-12)
    n = motion.shape[0]
    ij = np.indices((n, n)).transpose(1, 2, 0)
    i, j = np.moveaxis((ij @ permutation.T) % n, -1, 0)
    return motion[i, j] @ frame


def spectra_and_solvers(scan, original, context, calibrations):
    basis, gradient, q = (context[key] for key in ("basis", "gradient", "reciprocal"))
    phys = original["physics"]
    n, stiffness = phys["grid"], phys["stiffness_eV_per_cell"]
    adhesion = phys["stacking_coefficients_meV_per_cell"]["anchor"]
    initial = make_grid(gradient, basis, n, *stiffness)
    registry = initial.fractional @ basis.T
    spectra = {camera: {} for camera in calibrations}
    checks = {}
    for name, (g, transform) in variants(scan, gradient).items():
        checkpoint_guard()
        grid = make_grid(g, basis, n, *stiffness)
        if name == "rotate_minus_30":
            np.testing.assert_allclose(grid.k, initial.k @ rotation(-np.pi / 6), atol=1e-12, rtol=0)
        motion, diagnostic = minimize_all(grid, adhesion)
        local = registry + to_native(motion, basis, transform)
        strain, _ = decompose(g)
        checks[name] = {**diagnostic, **strain, "polar_mean_percent": float(np.mean(strain["principal_strains_percent"])),
                        "cell_area_nm2": float(abs(np.linalg.det(grid.cell)))}
        for camera, calibration in calibrations.items():
            signal = detector_intensity(local, q, calibration)
            spectra[camera][name] = np.fft.fft2(signal, axes=(0, 1))
    return spectra, checks


def held_pixel_scores(context, images, spectra):
    """Score second-order pixels with the transferred detector and fold-only normalization."""
    basis, registry, second = (context["basis"], context["rigid"], context["rings"]["second"])
    n = next(iter(spectra.values())).shape[0]
    fit = np.asarray(context["receipt"]["first_ring_affine_fit"])
    cell = np.linalg.solve(fit[:2].T, basis)
    freq2 = np.sum(wavevectors(cell, n)**2, axis=-1)
    gain = gaussian_gain(freq2, NOMINAL)
    index = n * np.moveaxis(registry @ np.linalg.inv(basis).T, -1, 0)
    observed = images[..., second]
    fields = {}
    for name, spectrum in spectra.items():
        images_model = np.fft.ifft2(spectrum[..., second] * gain[..., None], axes=(0, 1)).real
        fields[name] = np.stack([
            map_coordinates(channel, index, order=3, mode="grid-wrap")
            for channel in np.moveaxis(images_model, -1, 0)
        ], axis=-1)
    results = []
    for train, test in fold_masks(context["support"], context["probe"]):
        observed_fold = normalize_camera(observed, train)[test]
        errors = {name: float(np.sqrt(np.mean((pixels[test] - observed_fold)**2)))
                  for name, pixels in fields.items()}
        results.append({"training_pixels": int(train.sum()), "scored_pixels": int(test.sum()), "rmse": errors})
    return results


def score_walls(context, images, modes, sections, original, previous, spectra, calibration):
    """Replay the original cuts and selected diffraction channels at nominal probe width."""
    basis, q, rings = (context[k] for k in ("basis", "reciprocal", "rings"))
    n = original["physics"]["grid"]
    modes_by_index = {tuple(mode): k for k, mode in enumerate(modes)}
    walls = []
    for i, section in enumerate(sections):
        saved = original["walls"][i]
        if saved["centers"] != section["centers"]:
            raise ValueError("The observed wall positions changed")
        if not saved["original_cohort"] or section["reach_nm"] < HALF_WINDOW:
            continue
        checkpoint_guard()
        ref = previous["walls"][len(walls)]
        if ref["centers"] != section["centers"]:
            raise ValueError("The nominal scoring cohort changed")
        channels = [np.asarray([modes_by_index[tuple(m)] for m in group]) for group in saved["miller_indices"]]
        view = {**section, "keep": abs(DISTANCE) <= HALF_WINDOW}
        reader = get_observer(view, q, calibration["transferred_even"], rings, NOMINAL, channels)
        observed = sample_cuts(images, section["coordinates"]).mean(axis=0)
        target = reader.shape(observed)
        measured = width_pair(reader, observed)
        freq2, cut_ix, corner_ix, weights = raster_coordinates(section, basis, original["pixel_nm"], n)
        gain = gaussian_gain(freq2, NOMINAL)
        row = {"centers": section["centers"], "midpoint_xy_nm": saved["midpoint_xy_nm"],
               "spacing_nm": section["spacing_nm"], "measured": measured, "arms": {}}
        for cam, models in spectra.items():
            result = {}
            for name, spectrum in models.items():
                _, trace = render_traces(spectrum, gain, cut_ix, corner_ix, weights)
                result[name] = score(reader, target, trace)
            frozen = ref["nominal"][cam]
            for measured_fit, previous_fit in ((measured, frozen["measured"]),
                                                (result["nominal"], frozen["models"]["anchor"]["native"]),
                                                (result["remove_mean_strain"], frozen["models"]["0_no_mean"]["native"])):
                if measured_fit["qualified"] != previous_fit["qualified"]:
                    raise ValueError("Frozen width qualification failed to replay")
                for key in ("widths_nm", "transfer_error"):
                    if key in previous_fit:
                        np.testing.assert_allclose(measured_fit[key], previous_fit[key], atol=1e-5, rtol=0)
            row["arms"][cam] = result
        walls.append(row)
    if len(walls) != len(previous["walls"]):
        raise ValueError("Some frozen wall sections were skipped")
    return walls


def local_replay(scan, context, images, angle, previous):
    """Run the same local second-ring folds as the earlier mean-strain comparison."""
    receipt = context["receipt"]
    basis = context["basis"]
    yy, xx = np.indices(images.shape[:2])
    points = np.stack((xx, yy), axis=-1) * receipt["pixel_nm"]
    base_fit = np.asarray(receipt["first_ring_affine_fit"])
    centers = np.asarray(receipt["centers_rc_px"])
    frozen = {fold["center_group"]: fold["scores"] for fold in previous["local_scores"]}
    all_scores = []
    for group, _, warp, fit, mask, _ in geometric_folds(
        centers, points, receipt["pixel_nm"], basis, base_fit, context["support"],
    ):
        if group not in frozen:
            continue
        checkpoint_guard()
        gradient = fit[:2].T @ rotation(-angle)
        models = variants(scan, gradient)
        observed_cell = np.linalg.solve(gradient, basis)
        predicted = {}
        for name in ("nominal", "remove_mean_strain", "rotate_minus_30"):
            grid, motion, _ = solve_from_geometry(models[name][0], basis, ANCHOR)
            predicted[name] = sample_disregistry(grid, motion, warp, observed_cell)
        scores = score_fields(predicted, context, images, mask)
        for recent, old in zip(scores, frozen[group], strict=True):
            if recent["pixels"] != old["pixels"]:
                raise ValueError("The old local diffraction pixels changed")
            for name, original_name in (("nominal", "full"), ("remove_mean_strain", "no_mean_dilation")):
                np.testing.assert_allclose(recent["rmse"][name], old["rmse"][original_name], atol=1e-12, rtol=0)
        all_scores.extend(scores)
    return all_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "release", "dependencies", "evidence", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    start = perf_counter()
    files = (SOURCE, BENCHMARK, SETTINGS, *(args.evidence / name for name in INPUT_NAMES),
             args.evidence / V005, args.evidence / LOCAL_SCORES)
    source, benchmark, settings, primary, control, optics, frozen, earlier = (
        json.loads(path.read_text()) for path in files
    )
    library = ("run_mos2_observation_controls.py", "bilayer_relaxation.py", "run_mos2_uniaxial_check.py",
               "run_mos2_constrained_equilibrium.py", "run_mos2_camera_transfer.py", "author_detector_replay.py",
               "run_mos2_wall_width_transfer.py", "run_mos2_nonlinear_controls.py")
    hashes = {p.name: digest(p) for p in (Path(__file__), *files, *(PROJECT / k for k in library))}
    scans = (*TARGETS, *CONTROLS)
    result = {
        "input_sha256": hashes, "requested_scans": scans,
        "scope": "Test fixed ±30 degree mechanical-frame changes at the original 1.25 nm probe. "
                 "The archived registry, raster, width observer, energy and camera coefficients are held fixed "
                 "for the width and transferred-calibration holdout comparisons. The earlier local intensity "
                 "folds also retain their original train/test detector-fit procedure.",
        "scans": {},
    }
    elapsed = 0.
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        result = json.loads(args.output.read_text())
        if result["input_sha256"] != hashes or result["requested_scans"] != list(scans):
            raise ValueError("The source freeze changed since the checkpoint")
        elapsed = result["seconds"]
    author = load_author_release(args.release, args.dependencies)
    for scan in scans:
        if scan in result["scans"]:
            continue
        checkpoint_guard()
        metadata = read_input_metadata(scan, args.archive)
        images, modes, checksum = disk_images(args.archive, scan, author)
        ref = source["scans"][scan]
        if checksum != ref["diskset_sha256"] or metadata["archive_md5"] != benchmark["scans"][scan]["archive_md5"]:
            raise ValueError(f"The archived diffraction changed for {scan}")
        meta = metadata["metadata_fields"]
        context = centers_and_context(images, modes, meta, settings, tuple(ref["interface"]))
        if context["receipt"] != ref["geometry"]:
            raise ValueError(f"The first-ring geometry changed for {scan}")
        angle = np.deg2rad(float(meta["SampleRotation"]))
        sections = geometry_sections(context, rotation(angle))
        optical = control["scans"][scan]["camera_diagnostic"]["by_model"]["anchor"]
        even = optics["frozen_common_coefficients"]
        calib = {"transferred_even": np.broadcast_to([even[0], 0., even[1]], (len(modes), 3)),
                 "separate_disjoint": np.asarray(optical["coefficients"])}
        original = primary["scans"][scan]
        spectra, solvers = spectra_and_solvers(scan, original, context, calib)
        walls = score_walls(context, images, modes, sections, original, frozen["scans"][scan], spectra, calib)
        result["scans"][scan] = {
            "solvers": solvers, "walls": walls,
            "held_second_order": held_pixel_scores(context, images, spectra["transferred_even"]),
            "earlier_local_second_order": local_replay(scan, context, images, angle, earlier["scans"][scan]),
        }
        result["seconds"] = elapsed + perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, result)
        print(scan, len(walls), round(result["seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
