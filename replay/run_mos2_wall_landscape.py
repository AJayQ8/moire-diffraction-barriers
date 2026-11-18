"""Compare material inputs on the same MoS2 diffraction profiles and windows."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from author_release import load_author_release
from bilayer_relaxation import MODES, STACKING, elastic_parameters_per_reference_cell, sample_disregistry
from observe_bilayers import read_input_metadata
from run_heterobilayer_sparse_geometry import solve_from_geometry
from run_heterobilayer_template_transfer import design_matrix
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_directional_diffraction import wall_sectors
from run_mos2_intensity_only_replay import disk_images
from run_mos2_nonlinear_controls import alternative_inputs, minimize_all
from run_mos2_slow_readout_transfer import calibrate, render
from run_mos2_wall_period_scaling import half_maximum
from run_mos2_wall_profile_closure import PROFILE_DISTANCE, fields_on_chord, render_profiles, transfer
from run_mos2_wall_width_transfer import DISTANCE, REDUCED, Profiles, actual_profiles, geometry_sections, isolated_width
from run_registry_prediction import rotation
from run_wall_heterostrain import PROTOCOL, SOURCE
from shared_compute import atomic_json, checkpoint_guard


TEST = PROJECT / "mos2_wall_landscape_protocol_v01.json"
ARCHIVE_SCORES = PROJECT / "moire_wall_energy_readout_v01.json"
ARCHIVE_WIDTHS = PROJECT / "mos2_wall_width_transfer_v01.json"
CROP = abs(PROFILE_DISTANCE) <= DISTANCE[-1]
RELAXED = ("anchor", "barrier_075", "barrier_125", "shape_enaldiev")
NOMINAL_PROBE = 1.25


def equilibria(context, interface):
    gradient, basis = context["gradient"], context["basis"]
    grid, anchor_motion, solver = solve_from_geometry(gradient, basis, interface)
    anchor = np.asarray(STACKING[interface])
    stiffness = elastic_parameters_per_reference_cell(*interface[:2])
    alternate = np.array(alternative_inputs(basis, "P")[0], copy=True)
    edge = basis[:, 0]
    barrier = isolated_width(basis, edge, anchor)[1]
    alternate *= barrier / isolated_width(basis, edge, alternate)[1]
    landscapes = {
        "anchor": anchor,
        "barrier_075": 0.75 * anchor,
        "barrier_125": 1.25 * anchor,
        "shape_enaldiev": alternate,
    }
    motions = {"anchor": anchor_motion}
    receipts = {"anchor": solver}
    for name in RELAXED[1:]:
        motions[name], receipts[name] = minimize_all(grid, landscapes[name])
    barriers = {name: 1000 * isolated_width(basis, edge, coeff)[1] for name, coeff in landscapes.items()}
    expected = 1000 * barrier * np.array([1.0, 0.75, 1.25, 1.0])
    np.testing.assert_allclose(list(barriers.values()), expected, atol=1e-10, rtol=0)
    return grid, motions, (anchor, stiffness), {
        "grid": int(grid.fractional.shape[0]),
        "barriers_meV_per_cell": barriers,
        "stacking_coefficients_meV_per_cell": {key: val.tolist() for key, val in landscapes.items()},
        "stiffness_eV_per_cell": list(stiffness),
        "solvers": receipts,
    }


def capture_footprint(sections, shape):
    """Mark the source pixels used by any bilinear width-profile sample."""
    footprint = np.zeros(shape, dtype=bool)
    for section in sections:
        lower = np.floor(section["coordinates"][:, section["keep"]]).astype(int).reshape(-1, 2)
        for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1)):
            footprint[lower[:, 0] + dr, lower[:, 1] + dc] = True
    return footprint


def calibration(images, context, compact, sections):
    rigid, basis, q = (context[k] for k in ("rigid", "basis", "reciprocal"))
    train = context["support"] & ~np.logical_or.reduce(tuple(wall_sectors(rigid, basis).values()))
    footprint = capture_footprint(sections, train.shape)
    field = rigid + design_matrix(rigid, basis, MODES) @ compact
    result = {}
    for name, phase in (("compact", field), ("rigid", rigid)):
        signals = render(phase, np.ones(train.shape), q, context["probe"])
        normalized, coeff = calibrate(signals, images, train)
        delta = np.einsum("xycp,cp->xyc", signals, coeff) - normalized
        receipt = {
            "channels": coeff.tolist(),
            "training_pixels": int(train.sum()),
            "training_pixels_in_any_cut": int((train & footprint).sum()),
            "training_channel_rmse": np.sqrt(np.mean(delta[train]**2, axis=0)).tolist(),
            "cut_pixels_outside_training": int((footprint & ~train).sum()),
            "outside_training_cut_channel_rmse": np.sqrt(np.mean(delta[footprint & ~train]**2, axis=0)).tolist(),
        }
        result[name] = (normalized, coeff, receipt)
    return result


def make_states(section, context, angle, grid, motions, compact, material, probe):
    states, weights = fields_on_chord(section, context, angle, grid, motions["anchor"],
                                      grid, motions["shape_enaldiev"], compact, {"anchor": material}, probe)
    phase = states["rigid"]
    fields = {"anchor": states["periodic_anchor"], "shape_enaldiev": states["periodic_alternate"],
              "rigid": phase}
    for name in ("barrier_075", "barrier_125"):
        fields[name] = sample_disregistry(grid, motions[name], phase, grid.cell)
    return fields, weights


def midpoint_widths(section, grid, motions):
    center = len(DISTANCE) // 2
    phase = section["phase"][1]
    slope = (phase[center + 1] - phase[center]) / (DISTANCE[center + 1] - DISTANCE[center])
    edge = section["jump"]
    slip = np.array((-edge[1], edge[0])) / np.linalg.norm(edge)
    radius = 0.49 * section["spacing_nm"]
    fine = np.arange(-int(radius / .025), int(radius / .025) + 1) * .025
    query = phase[center] + fine[:, None] * slope
    result = {}
    for name, motion in motions.items():
        field = sample_disregistry(grid, motion, query, grid.cell)
        gradient = np.gradient(field @ slip, fine, edge_order=2)
        left = half_maximum(fine, gradient[::-1])
        right = half_maximum(fine, gradient)
        result[name] = None if left is None or right is None else 0.5 * (left + right)
    return result


def width_pair(observer, signal):
    target = observer.shape(signal)
    fits = [observer.fit(target, [ring]) for ring in (0, 1)]
    return {
        "widths_nm": [fit["width_nm"] for fit in fits],
        "qualified": [fit["qualified"] for fit in fits],
        "centers_nm": [fit["center_nm"] for fit in fits],
        "fit_error": [fit["error_by_ring"][ring] for ring, fit in enumerate(fits)],
    }


def get_observer(section, q, camera, rings, probe, channels):
    observer = Profiles(section, q, camera, rings, probe, "sine_gordon")
    observer.channels = channels
    return observer


def inspect_wall(section, geometrical, grid, motions, material, cameras, observed, compact, protocol):
    context, angle, modes = geometrical
    q, rings = (context[k] for k in ("reciprocal", "rings"))
    reference = Profiles(section, q, cameras["compact"][1], rings, NOMINAL_PROBE, "sine_gordon")
    channels = reference.channels
    windows = {"adaptive": section}
    for name in protocol["fit_windows_nm"][1:]:
        radius = float(name)
        if section["reach_nm"] >= radius:
            windows[name] = {**section, "keep": abs(DISTANCE) <= radius}
    readouts = {name: {} for name in cameras}
    short_blur = {}
    for probe in protocol["probe_fwhm_nm"]:
        fields, weights = make_states(section, context, angle, grid, motions, compact, material, probe)
        for cam, (_, coeff, _) in cameras.items():
            rendered = render_profiles(fields, q, coeff, weights, probe)
            readouts[cam][f"{probe:g}"] = {}
            for window, reduced in windows.items():
                observer = get_observer(reduced, q, coeff, rings, probe, channels)
                measured = width_pair(observer, observed[cam])
                shape = observer.shape(observed[cam])
                models = {}
                for name, signal in rendered.items():
                    fit = width_pair(observer, signal[CROP])
                    aligned = transfer(observer, shape, signal)
                    fit["transfer_error"] = [row["error_by_ring"][1 - i] for i, row in enumerate(aligned)]
                    fit["source_error"] = [row["error_by_ring"][i] for i, row in enumerate(aligned)]
                    fit["alignment_at_bound"] = [row["at_position_bound"] for row in aligned]
                    models[name] = fit
                readouts[cam][f"{probe:g}"][window] = {"measured": measured, "models": models}
            if cam == "compact":
                observer = get_observer(windows["2.50"], q, coeff, rings, NOMINAL_PROBE, channels)
                short_blur[f"{probe:g}"] = width_pair(observer, rendered["anchor"][CROP])
    return {"spacing_nm": section["spacing_nm"], "reach_nm": section["reach_nm"],
            "orientation_factor": section["k_factor"],
            "miller_indices": [modes[ix].tolist() for ix in channels],
            "model_core_widths_nm": midpoint_widths(section, grid, motions),
            "short_window_anchor_read_as_1.25_nm": short_blur, "readouts": readouts}


def reproduce_archived(wall, score):
    row = wall["readouts"]["compact"]["1.25"]["adaptive"]
    for fit, name in ((row["measured"], "measured"), (row["models"]["anchor"], "periodic_anchor")):
        if not all(fit["qualified"]):
            raise ValueError(f"The original qualified width at {wall['centers']} failed replay")
        np.testing.assert_allclose(fit["widths_nm"], score["widths_nm"]["1.25"][name], atol=1e-5, rtol=0)


def analyze_scan(context, images, modes, metadata, interface, compact, archive_scores, archive_widths, protocol):
    angle = np.deg2rad(float(metadata["SampleRotation"]))
    sections = geometry_sections(context, rotation(angle))
    if [s["centers"] for s in sections] != [w["centers"] for w in archive_widths["walls"]]:
        raise ValueError("The measured wall chords differ from the archived data")
    cameras = calibration(images, context, compact, sections)
    grid, motions, material, methods = equilibria(context, interface)
    qualified = {tuple(row["centers"]): row for row in archive_scores["chords"]}
    locations = np.asarray(context["receipt"]["centers_rc_px"])
    pixel = context["receipt"]["pixel_nm"]
    walls = []
    for section in sections:
        checkpoint_guard()
        indices = section["centers"]
        observed = {key: actual_profiles(val[0], section["coordinates"]) for key, val in cameras.items()}
        wall = inspect_wall(section, (context, angle, modes), grid, motions, material, cameras, observed,
                            compact, protocol)
        chord = tuple(indices)
        wall.update(centers=indices, midpoint_xy_nm=(locations[indices].mean(axis=0)[::-1] * pixel).tolist(),
                    original_cohort=chord in qualified)
        if wall["original_cohort"]:
            reproduce_archived(wall, qualified[chord])
        walls.append(wall)
    return {"inferred_twist_deg": archive_scores["twist_deg"], "pixel_nm": pixel, "centers_rc_px": locations.tolist(),
            "cameras": {key: val[2] for key, val in cameras.items()}, "physics": methods, "walls": walls}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("release", type=Path)
    cli.add_argument("dependencies", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--resume", action="store_true", help="Reuse completed scans from an identical source freeze.")
    args = cli.parse_args()
    start = perf_counter()
    paths = (TEST, SOURCE, BENCHMARK, SETTINGS, REDUCED, PROTOCOL, ARCHIVE_SCORES, ARCHIVE_WIDTHS)
    protocol, source, benchmark, settings, reduced, _, scores, widths = (json.loads(p.read_text()) for p in paths)
    author = load_author_release(args.release, args.dependencies)
    code = tuple(PROJECT / name for name in (
        "bilayer_relaxation.py", "author_detector_replay.py", "run_mos2_wall_width_transfer.py",
        "run_mos2_wall_profile_closure.py", "run_mos2_wall_period_scaling.py", "run_mos2_nonlinear_controls.py",
        "run_mos2_slow_readout_transfer.py", "run_moire_common_intensity.py", "wall_registry_fit.py",
    ))
    hashes = {p.name: digest(p) for p in (Path(__file__), *paths, *code)}
    report = {"protocol": protocol, "input_sha256": hashes, "scans": {}}
    elapsed = 0.
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        report = json.loads(args.output.read_text())
        if report["input_sha256"] != hashes or not set(report["scans"]) <= set(protocol["scans"]):
            raise ValueError("The saved checkpoint used a different input freeze")
        elapsed = report["seconds"]
    for scan in protocol["scans"]:
        if scan in report["scans"]:
            continue
        checkpoint_guard()
        saved = source["scans"][scan]
        metadata = read_input_metadata(scan, args.archive)
        images, modes, image_hash = disk_images(args.archive, scan, author)
        if image_hash != saved["diskset_sha256"] or metadata["archive_md5"] != benchmark["scans"][scan]["archive_md5"]:
            raise ValueError(f"The archived scan {scan} changed")
        interface = tuple(saved["interface"])
        context = centers_and_context(images, modes, metadata["metadata_fields"], settings, interface)
        if context["receipt"] != saved["geometry"]:
            raise ValueError(f"The first-ring geometry of {scan} changed")
        compact = np.array(reduced["scans"][scan]["coefficients_nm"]["three_stars"])
        result = analyze_scan(context, images, modes, metadata["metadata_fields"], interface, compact,
                              scores["scans"][scan], widths["scans"][scan], protocol)
        report["scans"][scan] = result
        report["seconds"] = elapsed + perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, report)
        print(scan, len(result["walls"]), round(report["seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
