"""Replay absolute wall-width errors on pairs of opposite MoS2 diffraction spots."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from author_detector_replay import detector_intensity
from author_release import load_author_release
from bilayer_relaxation import make_grid
from observe_bilayers import read_input_metadata
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_intensity_only_replay import disk_images
from run_mos2_nonlinear_controls import minimize_all
from run_mos2_observation_controls import (
    HALF_WINDOW, INPUT_NAMES, NOMINAL, gaussian_gain, raster_coordinates, render_traces, sample_cuts,
)
from run_mos2_wall_landscape import CROP, get_observer, width_pair
from run_mos2_wall_profile_closure import PROFILE_DISTANCE, transfer
from run_mos2_wall_width_transfer import DISTANCE, geometry_sections
from run_registry_prediction import rotation
from run_wall_heterostrain import SOURCE
from shared_compute import atomic_json, checkpoint_guard


WINDOWS = (HALF_WINDOW, 2.50)
EVEN = "transferred_even"
DISJOINT = "separate_disjoint"
NATIVE = "mos2_geometry_probe_v005.json"


def paired_spots(modes, rings, selected):
    """Mirror the frozen channel groups, using both sides of each retained pair."""
    partitions = [[], []]
    for ring, channels in zip(("first", "second"), selected, strict=True):
        lookup = {tuple(modes[i]): int(i) for i in np.flatnonzero(rings[ring])}
        keys = set()
        for i in channels:
            pair = tuple(int(v) for v in modes[i])
            other = tuple(-v for v in pair)
            if other not in lookup:
                raise ValueError(f"The selected {ring}-order spot {pair} lacks its opposite")
            keys.add(max(pair, other))
        canonical = sorted(keys)
        for side, direction in enumerate((1, -1)):
            partitions[side].append(np.array([lookup[tuple(direction * v for v in k)] for k in canonical]))
    return partitions


def check_frozen(current, reference):
    if current["qualified"] != reference["qualified"]:
        raise ValueError("Width-fit qualification differs from the frozen reference")
    np.testing.assert_allclose(current["widths_nm"], reference["widths_nm"], atol=1e-5, rtol=0)


def shape_errors(reader, target, trace, pairs):
    """Compare shape residuals after registering the model on the other order."""
    alignment = transfer(reader, reader.shape(target), trace)
    measured = [p.shape(target) for p in pairs]
    errors = []
    for ring in (0, 1):
        location = alignment[1-ring]["center_nm"]
        shifted = np.column_stack([np.interp(DISTANCE-location, PROFILE_DISTANCE, c) for c in trace.T])
        a, b = [shape[ring] - p.shape(shifted)[ring] for p, shape in zip(pairs, measured, strict=True)]
        common, difference = (a+b)/2, (a-b)/2
        errors.append({
            "common_squared": float(np.sum(common**2)), "difference_squared": float(np.sum(difference**2)),
            "plus_squared": float(np.sum(a**2)), "minus_squared": float(np.sum(b**2)),
            "source_center_nm": location, "source_center_at_bound": alignment[1-ring]["at_position_bound"],
        })
    return errors


def examine(context, images, modes, angle, original, calibration, previous):
    basis, grad, q, rings = (context[k] for k in ("basis", "gradient", "reciprocal", "rings"))
    sections = geometry_sections(context, rotation(angle))
    physics = original["physics"]
    n = physics["grid"]
    grid = make_grid(grad, basis, n, *physics["stiffness_eV_per_cell"])
    motion, solver = minimize_all(grid, physics["stacking_coefficients_meV_per_cell"]["anchor"])
    registry = grid.fractional @ basis.T + motion
    spectra = {name: np.fft.fft2(detector_intensity(registry, q, coeff), axes=(0, 1))
               for name, coeff in calibration.items()}
    archived = {tuple(row["centers"]): row for row in previous["walls"]}
    modes_by_index = {tuple(mode): i for i, mode in enumerate(modes)}
    receipts = []
    for section, ref in zip(sections, original["walls"], strict=True):
        if section["centers"] != ref["centers"]:
            raise ValueError("The archived wall identities changed")
        if not ref["original_cohort"] or section["reach_nm"] < HALF_WINDOW:
            continue
        checkpoint_guard()
        frozen = archived[tuple(section["centers"])]
        selected = [np.asarray([modes_by_index[tuple(m)] for m in group]) for group in ref["miller_indices"]]
        halves = paired_spots(modes, rings, selected)
        coords = section["coordinates"]
        observed = sample_cuts(images, coords).mean(axis=0)
        freq2, cut, corners, weights = raster_coordinates(section, basis, original["pixel_nm"], n)
        gain = gaussian_gain(freq2, NOMINAL)
        predictions = {name: render_traces(spectrum, gain, cut, corners, weights)[1]
                       for name, spectrum in spectra.items()}
        record = {k: frozen[k] for k in ("centers", "midpoint_xy_nm", "spacing_nm", "character", "scan_angle_deg")}
        record.update(
            half_modes=[[modes[order].tolist() for order in side] for side in halves],
            windows={},
        )
        for width in WINDOWS:
            view = {**section, "keep": abs(DISTANCE) <= width}
            reader = get_observer(view, q, calibration[EVEN], rings, NOMINAL, selected)
            paired = [get_observer(view, q, calibration[EVEN], rings, NOMINAL, group) for group in halves]
            observed_halves = [width_pair(half, observed) for half in paired]
            models = {}
            if width == HALF_WINDOW:
                check_frozen(width_pair(reader, observed), frozen["nominal"][EVEN]["measured"])
            for camera, trace in predictions.items():
                if width == HALF_WINDOW:
                    check_frozen(width_pair(reader, trace[CROP]), frozen["nominal"][camera]["models"]["anchor"]["native"])
                models[camera] = {
                    "widths": [width_pair(half, trace[CROP]) for half in paired],
                    "profile": shape_errors(reader, observed, trace, paired),
                }
            record["windows"][f"{width:g}"] = {"measured": observed_halves, "models": models}
        receipts.append(record)
    if len(receipts) != len(archived):
        raise ValueError("The frozen wall cohort is incomplete")
    return {"solver": solver, "walls": receipts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("archive", "release", "dependencies", "evidence", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    paths = (SOURCE, BENCHMARK, SETTINGS, *(args.evidence / name for name in INPUT_NAMES), args.evidence / NATIVE)
    source, bench, settings, primary, control, optical, frozen = [json.loads(p.read_bytes()) for p in paths]
    libraries = ("run_mos2_observation_controls.py", "run_mos2_wall_landscape.py",
                 "run_mos2_wall_width_transfer.py", "run_mos2_wall_profile_closure.py",
                 "run_mos2_nonlinear_controls.py", "bilayer_relaxation.py", "author_detector_replay.py")
    hashes = {p.name: digest(p) for p in (Path(__file__), *paths, *(PROJECT / name for name in libraries))}
    scans = primary["protocol"]["scans"]
    output = {"scope": "Absolute width and registered shape errors on paired opposite spots at the fixed "
                       "archived alignment, anchor energy, and nominal probe. Both readings share a raster. "
                       "All ten v005 scans were examined previously.",
              "half_windows_nm": WINDOWS, "gaussian_probe_fwhm_nm": NOMINAL,
              "input_sha256": hashes, "requested_scans": scans, "scans": {}}
    elapsed = 0.
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        output = json.loads(args.output.read_bytes())
        if output["input_sha256"] != hashes or output["requested_scans"] != scans:
            raise ValueError("Saved checkpoint belongs to a different comparison")
        elapsed = output["seconds"]
    author = load_author_release(args.release, args.dependencies)
    even = optical["frozen_common_coefficients"]
    for name in scans:
        if name in output["scans"]:
            continue
        checkpoint_guard()
        metadata = read_input_metadata(name, args.archive)
        images, modes, checksum = disk_images(args.archive, name, author)
        ref = source["scans"][name]
        if checksum != ref["diskset_sha256"] or metadata["archive_md5"] != bench["scans"][name]["archive_md5"]:
            raise ValueError(f"The archived input changed for {name}")
        fields = metadata["metadata_fields"]
        context = centers_and_context(images, modes, fields, settings, tuple(ref["interface"]))
        if context["receipt"] != ref["geometry"]:
            raise ValueError(f"The frozen geometry changed for {name}")
        angle = np.deg2rad(float(fields["SampleRotation"]))
        disjoint = control["scans"][name]["camera_diagnostic"]["by_model"]["anchor"]["coefficients"]
        cameras = {EVEN: np.broadcast_to([even[0], 0., even[1]], (len(modes), 3)), DISJOINT: np.asarray(disjoint)}
        output["scans"][name] = examine(context, images, modes, angle, primary["scans"][name], cameras,
                                        frozen["scans"][name])
        output["seconds"] = elapsed + perf_counter() - started
        checkpoint_guard()
        atomic_json(args.output, output)
        print(name, len(output["scans"][name]["walls"]), round(output["seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
