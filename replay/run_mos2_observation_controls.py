"""Replay MoS2 wall cuts, scan raster sampling, and a bounded probe extension."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import map_coordinates, spline_filter

from author_detector_replay import detector_intensity
from author_release import load_author_release
from bilayer_relaxation import make_grid, wavevectors
from observe_bilayers import read_input_metadata
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_directional_diffraction import wall_sectors
from run_mos2_intensity_only_replay import disk_images
from run_mos2_nonlinear_controls import minimize_all
from run_mos2_wall_landscape import CROP, RELAXED, get_observer, width_pair
from run_mos2_wall_profile_closure import PROFILE_DISTANCE, transfer
from run_mos2_wall_width_transfer import DISTANCE, geometry_sections
from run_registry_prediction import rotation
from run_wall_heterostrain import SOURCE
from shared_compute import atomic_json, checkpoint_guard


HALF_WINDOW = 4.25
NOMINAL = 1.25
EXTENSION = (1.75, 1.80, 1.85, 1.90, 1.95, 2.00, 2.10, 2.20)
RASTER_PROBES = (1.25, 1.45, 1.50, 1.55, 1.60, 1.65, 1.75)
RAMAN = ("raman_shape", "raman_scale")
DETECTORS = ("transferred_even", "separate_disjoint")
INPUT_NAMES = (
    "mos2_wall_landscape_v01.json",
    "mos2_wall_landscape_camera_controls_v01.json",
    "mos2_camera_transfer_v01.json",
)


def sample_cuts(images, coordinates):
    """Sample each recorded channel from native scan pixels."""
    points = np.moveaxis(coordinates, -1, 0)
    return np.stack([
        map_coordinates(image, points, order=1, mode="nearest")
        for image in np.moveaxis(images, -1, 0)
    ], axis=-1)


def fit_cut_averaging(observer, images, coords, cuts):
    individual = [width_pair(observer, cut) for cut in cuts]
    aligned = None
    if all(row["qualified"][0] for row in individual):
        centers = np.array([row["centers_nm"][0] for row in individual])
        direction = (coords[:, 1] - coords[:, 0]) / (DISTANCE[1] - DISTANCE[0])
        shifted = coords + (centers - centers.mean())[:, None, None] * direction[:, None, :]
        aligned = width_pair(observer, sample_cuts(images, shifted).mean(axis=0))
    return {"individual": individual, "unaligned": width_pair(observer, cuts.mean(axis=0)),
            "first_order_centers_aligned": aligned}


def opposite_spot_errors(observer, signal, modes):
    """Same-wall disagreement for opposite spots selected by the width reader."""
    errors = []
    for channels, shape in zip(observer.channels, observer.shape(signal), strict=True):
        modes_in_order = {tuple(modes[idx]): j for j, idx in enumerate(channels)}
        differences = []
        for mode, j in modes_in_order.items():
            k = modes_in_order.get(tuple(-h for h in mode))
            if k is not None and j < k:
                differences.append(float(np.linalg.norm(shape[:, j] - shape[:, k])**2 * len(channels)))
        errors.append({"pairs": len(differences),
                       "rms": float(np.sqrt(np.mean(differences))) if differences else None})
    return errors


def raster_coordinates(section, basis, pixel_nm, n):
    """Prolong the local affine registry to the needed native raster corners."""
    coords = section["coordinates"]
    design = np.column_stack((coords[..., ::-1].reshape(-1, 2) * pixel_nm, np.ones(coords.size // 2)))
    phase = section["phase"].reshape(-1, 2)
    transform = np.linalg.lstsq(design, phase, rcond=None)[0]
    np.testing.assert_allclose(design @ transform, phase, atol=1e-10, rtol=0)
    cell = np.linalg.solve(transform[:2].T, basis)
    frequency_squared = np.sum(wavevectors(cell, n)**2, axis=-1)
    center = len(DISTANCE) // 2
    scan_step = (coords[:, center + 1] - coords[:, center]) / (DISTANCE[1] - DISTANCE[0])
    cuts = coords[:, center, None, :] + PROFILE_DISTANCE[None, :, None] * scan_step[:, None, :]
    np.testing.assert_allclose(cuts[:, CROP], coords, atol=1e-10, rtol=0)
    lower = np.floor(cuts)
    fraction = cuts - lower
    corners, weights = [], []
    for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1)):
        corners.append(lower + [dr, dc])
        weights.append((fraction[..., 0] if dr else 1 - fraction[..., 0])
                       * (fraction[..., 1] if dc else 1 - fraction[..., 1]))
    inverse_basis = np.linalg.inv(basis)

    def grid_index(rc):
        registry = (rc[..., ::-1] * pixel_nm) @ transform[:2] + transform[2]
        return n * np.moveaxis(registry @ inverse_basis.T, -1, 0)

    return frequency_squared, grid_index(cuts), grid_index(np.asarray(corners)), np.asarray(weights)


def render_traces(spectrum, gain, cut_index, corner_index, weights):
    field = np.fft.ifft2(spectrum * gain[..., None], axes=(0, 1)).real
    continuous, raster = [], []
    for plane in np.moveaxis(field, -1, 0):
        spline = spline_filter(plane, order=3, mode="grid-wrap")
        values = map_coordinates(spline, cut_index, order=3, mode="grid-wrap", prefilter=False)
        continuous.append(values.mean(axis=0))
        if corner_index is not None:
            pixels = map_coordinates(spline, corner_index, order=3, mode="grid-wrap", prefilter=False)
            raster.append(np.sum(pixels * weights, axis=0).mean(axis=0))
    return np.column_stack(continuous), np.column_stack(raster) if raster else None


def score(observer, target, trace):
    result = width_pair(observer, trace[CROP])
    registered = transfer(observer, target, trace)
    result["transfer_error"] = [row["error_by_ring"][1 - i] for i, row in enumerate(registered)]
    return result


def gaussian_gain(frequency_squared, probe):
    return np.exp(-frequency_squared * probe**2 / (16 * np.log(2)))


def baseline_qualified(ref):
    candidate = ref["readouts"]["compact"]["1.25"]["4.25"]
    return all(candidate["measured"]["qualified"]) and all(
        all(candidate["models"][model]["qualified"]) for model in RELAXED
    )


def examine(context, images, modes, metadata, original, control, optical):
    basis, gradient, q, rings = (context[key] for key in ("basis", "gradient", "reciprocal", "rings"))
    scan_angle = np.deg2rad(float(metadata["SampleRotation"]))
    sections = geometry_sections(context, rotation(scan_angle))
    if [section["centers"] for section in sections] != [row["centers"] for row in original["walls"]]:
        raise ValueError("Wall identities changed relative to the frozen study")
    physics = original["physics"]
    grid = make_grid(gradient, basis, physics["grid"], *physics["stiffness_eV_per_cell"])
    n = len(grid.fractional)
    compact = np.asarray(original["cameras"]["compact"]["channels"])
    wall_mask = np.logical_or.reduce(tuple(wall_sectors(context["rigid"], basis).values()))
    training = context["support"] & ~wall_mask
    if training.sum() != original["cameras"]["compact"]["training_pixels"]:
        raise ValueError("The compact calibration mask changed")
    compact_images = images / np.ptp(images[training], axis=0)
    even = optical["frozen_common_coefficients"]
    nominal_detector = np.broadcast_to([even[0], 0., even[1]], (len(q), 3))
    models = {
        "anchor": np.asarray(physics["stacking_coefficients_meV_per_cell"]["anchor"]),
        **{model: np.asarray(control["raman_constraint"]["models"][model]["coefficients_meV_per_cell"])
           for model in RAMAN},
    }
    detector = {
        "compact": {"anchor": compact},
        "transferred_even": {model: nominal_detector for model in RAMAN},
        "separate_disjoint": {
            model: np.asarray(control["camera_diagnostic"]["by_model"][model]["coefficients"]) for model in RAMAN
        },
    }
    np.testing.assert_allclose(q @ basis, np.rint(q @ basis), atol=1e-12, rtol=0)
    spectra = {}
    solvers = {}
    for model, coefficients in models.items():
        motion, solvers[model] = minimize_all(grid, coefficients)
        registry = grid.fractional @ basis.T + motion
        for name, coefficients_by_model in detector.items():
            if model in coefficients_by_model:
                image = detector_intensity(registry, q, coefficients_by_model[model])
                spectra[name, model] = np.fft.fft2(image, axes=(0, 1))
    mode_keys = {tuple(mode): i for i, mode in enumerate(modes)}
    output = []
    for section, ref in zip(sections, original["walls"], strict=True):
        if not ref["original_cohort"] or section["reach_nm"] < HALF_WINDOW:
            continue
        checkpoint_guard()
        view = {**section, "keep": abs(DISTANCE) <= HALF_WINDOW}
        channels = [np.array([mode_keys[tuple(mode)] for mode in group], dtype=int)
                    for group in ref["miller_indices"]]
        coords = section["coordinates"]
        cuts = sample_cuts(images, coords)
        measured = cuts.mean(axis=0)
        observer = get_observer(view, q, compact, rings, NOMINAL, channels)
        frequency, cut_index, corner_index, weights = raster_coordinates(
            section, basis, original["pixel_nm"], n
        )
        record = {"centers": ref["centers"], "midpoint_xy_nm": ref["midpoint_xy_nm"],
                  "adjacent_cut_spacing_nm": float(0.08 * section["length_nm"]),
                  "in_fig2_width_cohort": baseline_qualified(ref),
                  "opposite_spots": opposite_spot_errors(observer, measured, modes)}
        if record["in_fig2_width_cohort"]:
            compact_cuts = sample_cuts(compact_images, coords)
            fits = fit_cut_averaging(observer, compact_images, coords, compact_cuts)
            reference = ref["readouts"]["compact"]["1.25"]["4.25"]["measured"]
            np.testing.assert_allclose(fits["unaligned"]["widths_nm"], reference["widths_nm"],
                                       atol=1e-5, rtol=0)
            anchor = spectra["compact", "anchor"]
            direct, raster = render_traces(anchor, gaussian_gain(frequency, NOMINAL), cut_index,
                                           corner_index, weights)
            target = observer.shape(compact_cuts.mean(axis=0))
            record["cut_averaging"] = fits
            record["anchor_raster"] = {"continuous": score(observer, target, direct),
                                       "native_raster": score(observer, target, raster)}
        record["extension"] = {}
        record["raman_raster"] = {}
        for probe in sorted(set(EXTENSION) | set(RASTER_PROBES)):
            reader = get_observer(view, q, nominal_detector, rings, probe, channels)
            observed = width_pair(reader, measured)
            target = reader.shape(measured)
            gain = gaussian_gain(frequency, probe)
            if probe in EXTENSION:
                record["extension"][f"{probe:g}"] = {}
            if probe in RASTER_PROBES:
                record["raman_raster"][f"{probe:g}"] = {}
            for detector_name in DETECTORS:
                use_raster = probe in RASTER_PROBES and detector_name == "transferred_even"
                if probe not in EXTENSION and not use_raster:
                    continue
                direct_models, raster_models = {}, {}
                for model in RAMAN:
                    direct, raster = render_traces(spectra[detector_name, model], gain, cut_index,
                                                   corner_index if use_raster else None, weights)
                    direct_models[model] = score(reader, target, direct)
                    if use_raster:
                        raster_models[model] = score(reader, target, raster)
                if probe in EXTENSION:
                    record["extension"][f"{probe:g}"][detector_name] = {
                        "measured": observed, "models": direct_models,
                    }
                if use_raster:
                    record["raman_raster"][f"{probe:g}"] = {
                        "measured": observed, "continuous": direct_models, "native_raster": raster_models,
                    }
        output.append(record)
    if len(output) != sum(wall["original_cohort"] for wall in original["walls"]):
        raise ValueError("The original wall-section set is incomplete")
    return {"twist_deg": original["inferred_twist_deg"], "pixel_nm": original["pixel_nm"],
            "solvers": solvers, "walls": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scans", nargs="+", help="Limit a timing pilot to named maps.")
    args = parser.parse_args()
    start = perf_counter()
    inputs = (SOURCE, BENCHMARK, SETTINGS, *(args.evidence / name for name in INPUT_NAMES))
    source, benchmark, settings, primary, control, optical = (json.loads(p.read_text()) for p in inputs)
    library = ("bilayer_relaxation.py", "author_detector_replay.py", "run_mos2_wall_landscape.py",
               "run_mos2_wall_width_transfer.py", "run_mos2_wall_profile_closure.py")
    hashes = {path.name: digest(path) for path in (Path(__file__), *inputs, *(PROJECT / x for x in library))}
    if control["input_sha256"][INPUT_NAMES[0]] != hashes[INPUT_NAMES[0]]:
        raise ValueError("The calibration and geometry use different frozen studies")
    order = primary["protocol"]["scans"]
    if args.scans and (set(args.scans) - set(order) or len(set(args.scans)) != len(args.scans)):
        raise ValueError("The chosen maps must be unique members of the frozen set")
    report = {"input_sha256": hashes, "expected_scans": order, "extended_probes_nm": EXTENSION,
              "raster_probes_nm": RASTER_PROBES, "scans": {}}
    elapsed = 0.
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        report = json.loads(args.output.read_text())
        if report["input_sha256"] != hashes or report["expected_scans"] != order:
            raise ValueError("The source freeze changed since the checkpoint")
        elapsed = report["seconds"]
    release = load_author_release(args.release, args.dependencies)
    for name in args.scans or order:
        if name in report["scans"]:
            continue
        checkpoint_guard()
        metadata = read_input_metadata(name, args.archive)
        images, modes, checksum = disk_images(args.archive, name, release)
        ref = source["scans"][name]
        if checksum != ref["diskset_sha256"] or metadata["archive_md5"] != benchmark["scans"][name]["archive_md5"]:
            raise ValueError(f"The diffraction input changed for {name}")
        context = centers_and_context(images, modes, metadata["metadata_fields"], settings, tuple(ref["interface"]))
        if context["receipt"] != ref["geometry"]:
            raise ValueError(f"The inferred geometry changed for {name}")
        report["scans"][name] = examine(context, images, modes, metadata["metadata_fields"], primary["scans"][name],
                                        control["scans"][name], optical)
        report["seconds"] = elapsed + perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, report)
        print(name, len(report["scans"][name]["walls"]), round(report["seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
