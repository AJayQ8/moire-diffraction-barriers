"""Test a 10 meV sliding saddle at the published curvature and nominal probe."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from author_detector_replay import detector_intensity
from author_release import load_author_release
from bilayer_relaxation import MODES, make_grid, stack_energy
from observe_bilayers import read_input_metadata
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_elastic_frame import CAMERAS, TARGETS, V005, variants
from run_mos2_intensity_only_replay import disk_images
from run_mos2_nonlinear_controls import minimize_all
from run_mos2_observation_controls import (
    HALF_WINDOW, INPUT_NAMES, NOMINAL, gaussian_gain, raster_coordinates, render_traces, sample_cuts, score,
)
from run_mos2_wall_landscape import get_observer, isolated_width
from run_mos2_wall_width_transfer import DISTANCE, geometry_sections
from run_registry_prediction import rotation
from run_wall_heterostrain import SOURCE
from shared_compute import atomic_json, checkpoint_guard


def check_landscape(basis, anchor, alternative):
    ab, aa, sp = np.array([[1/3, 1/3], [0, 0], [.5, 0]]) @ basis.T
    reciprocal = 2 * np.pi * MODES @ np.linalg.inv(basis)

    def hessian(point, coefficients):
        terms = np.repeat(coefficients[1:4], 3) / 1000
        return (reciprocal.T * (-terms * np.cos(reciprocal @ point))) @ reciprocal

    np.testing.assert_allclose(hessian(ab, anchor), hessian(ab, alternative), atol=1e-12, rtol=0)
    positions = np.stack([ab, aa, sp])
    va, _ = stack_energy(positions, basis, anchor)
    vb, force = stack_energy(positions, basis, alternative)
    np.testing.assert_allclose(vb[:2], va[:2], atol=1e-12, rtol=0)
    np.testing.assert_allclose(force, 0., atol=1e-12, rtol=0)
    saddle = np.linalg.eigvalsh(hessian(sp, alternative))
    if not saddle[0] < 0 < saddle[1]:
        raise ValueError("The tested path does not cross an index-one saddle")
    barrier = 1000 * isolated_width(basis, basis[:, 0], alternative)[1]
    np.testing.assert_allclose(barrier, 10., atol=1e-10, rtol=0)
    return {"coefficients_meV_per_cell": alternative.tolist(), "saddle_meV_per_cell": barrier,
            "stable_curvature_eV_nm2": float(np.trace(hessian(ab, alternative)) / 2),
            "AA_minus_AB_meV_per_cell": float(1000 * (vb[1] - vb[0]))}


def evaluate(scan, context, images, modes, metadata, original, camera, optics, previous, corrected):
    basis, gradient, q, rings = (context[k] for k in ("basis", "gradient", "reciprocal", "rings"))
    angle = np.deg2rad(float(metadata["SampleRotation"]))
    sections = geometry_sections(context, rotation(angle))
    if [s["centers"] for s in sections] != [w["centers"] for w in original["walls"]]:
        raise ValueError("The measured section identities changed")
    physics = original["physics"]
    n = physics["grid"]
    anchor = np.asarray(physics["stacking_coefficients_meV_per_cell"]["anchor"])
    if np.any(anchor[4:]):
        raise ValueError("The saddle perturbation assumes a symmetric parallel bilayer")
    # Preserve the stable and AA energies and the AB Hessian. Linear conditions
    # on the three even shells then fix the -3.2 meV change at the saddle.
    alternative = anchor + [2.4, 1.6, -.8, -1.6, 0., 0.]
    landscape = check_landscape(basis, anchor, alternative)
    terms = optics["frozen_common_coefficients"]
    detector = np.broadcast_to([terms[0], 0., terms[1]], (len(q), 3))
    disjoint = np.asarray(camera["camera_diagnostic"]["by_model"]["anchor"]["coefficients"])
    frames = {"nominal": gradient}
    if scan in TARGETS:
        frames["rotated"] = variants(scan, gradient)["rotate_minus_30"][0]
    spectra, solvers = {}, {}
    for label, g in frames.items():
        checkpoint_guard()
        grid = make_grid(g, basis, n, *physics["stiffness_eV_per_cell"])
        motion, solvers[label] = minimize_all(grid, alternative)
        registry = grid.fractional @ basis.T + motion
        spectra[label] = {
            arm: np.fft.fft2(detector_intensity(registry, q, coeff), axes=(0, 1))
            for arm, coeff in zip(CAMERAS, (detector, disjoint), strict=True)
        }
    indices = {tuple(mode): i for i, mode in enumerate(modes)}
    walls = []
    for section, saved in zip(sections, original["walls"], strict=True):
        if not saved["original_cohort"] or section["reach_nm"] < HALF_WINDOW:
            continue
        checkpoint_guard()
        ref = previous["walls"][len(walls)]
        if ref["centers"] != section["centers"]:
            raise ValueError("The frozen wall cutoff or order changed")
        channels = [np.asarray([indices[tuple(m)] for m in group]) for group in saved["miller_indices"]]
        view = {**section, "keep": abs(DISTANCE) <= HALF_WINDOW}
        reader = get_observer(view, q, detector, rings, NOMINAL, channels)
        measured = sample_cuts(images, section["coordinates"]).mean(axis=0)
        target = reader.shape(measured)
        f2, cuts, corners, weights = raster_coordinates(section, basis, original["pixel_nm"], n)
        gain = gaussian_gain(f2, NOMINAL)
        row = {"centers": ref["centers"], "midpoint_xy_nm": ref["midpoint_xy_nm"],
               "spacing_nm": section["spacing_nm"], "measured": ref["nominal"]["transferred_even"]["measured"],
               "arms": {}}
        for arm in CAMERAS:
            comparisons = {}
            for label, frame_spectrum in spectra.items():
                _, trace = render_traces(frame_spectrum[arm], gain, cuts, corners, weights)
                if label == "nominal":
                    baseline = ref["nominal"][arm]["models"]["anchor"]["native"]
                else:
                    control = corrected["walls"][len(walls)]
                    if control["centers"] != ref["centers"]:
                        raise ValueError("The corrected wall identities changed")
                    baseline = control["arms"][arm]["rotate_minus_30"]
                comparisons[label] = {"13.2": baseline, "10": score(reader, target, trace)}
            row["arms"][arm] = comparisons
        walls.append(row)
    if len(walls) != len(previous["walls"]):
        raise ValueError("Did not score every archived wall")
    return {"landscape": landscape, "solvers": solvers, "walls": walls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "release", "dependencies", "evidence", "frame_result", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    start = perf_counter()
    files = (SOURCE, BENCHMARK, SETTINGS, *(args.evidence / name for name in INPUT_NAMES),
             args.evidence / V005, args.frame_result)
    source, benchmark, settings, primary, cameras, optics, prior, rotated = (json.loads(f.read_text()) for f in files)
    libraries = ("run_mos2_elastic_frame.py", "run_mos2_observation_controls.py", "bilayer_relaxation.py",
                 "run_mos2_wall_landscape.py", "run_mos2_wall_width_transfer.py", "run_mos2_nonlinear_controls.py")
    hashes = {f.name: digest(f) for f in (Path(__file__), *files, *(PROJECT / name for name in libraries))}
    if rotated["input_sha256"][V005] != hashes[V005] or not set(TARGETS) <= set(rotated["scans"]):
        raise ValueError("The frame check does not match the frozen reference")
    names = primary["protocol"]["scans"]
    result = {"requested_scans": names, "input_sha256": hashes,
              "scope": "One retrospective fixed-material sensitivity. Compare 10 and 13.2 meV with the published "
                       "stable-stacking curvature, AA energy and stiffness. Use the original nominal probe and "
                       "detector responses. Both frames are tried in DS13 and DS14; other maps use the original frame.",
              "scans": {}}
    elapsed = 0.
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        result = json.loads(args.output.read_text())
        if result["requested_scans"] != names or result["input_sha256"] != hashes:
            raise ValueError("The processed dataset or model changed since the checkpoint")
        elapsed = result["seconds"]
    author = load_author_release(args.release, args.dependencies)
    for scan in names:
        if scan in result["scans"]:
            continue
        checkpoint_guard()
        meta = read_input_metadata(scan, args.archive)
        images, modes, sha = disk_images(args.archive, scan, author)
        ref = source["scans"][scan]
        if sha != ref["diskset_sha256"] or meta["archive_md5"] != benchmark["scans"][scan]["archive_md5"]:
            raise ValueError(f"The raw intensities changed for {scan}")
        fields = meta["metadata_fields"]
        context = centers_and_context(images, modes, fields, settings, tuple(ref["interface"]))
        if context["receipt"] != ref["geometry"]:
            raise ValueError(f"The geometry changed for {scan}")
        result["scans"][scan] = evaluate(scan, context, images, modes, fields, primary["scans"][scan],
                                         cameras["scans"][scan], optics, prior["scans"][scan],
                                         rotated["scans"].get(scan))
        result["seconds"] = elapsed + perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, result)
        print(scan, len(result["scans"][scan]["walls"]), round(result["seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
