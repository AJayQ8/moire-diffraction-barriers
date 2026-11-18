"""Test whether solving each wall's measured local cell improves its prediction."""

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
from run_mos2_opposite_widths import DISJOINT, EVEN, check_frozen, paired_spots, shape_errors
from run_mos2_wall_landscape import CROP, get_observer, width_pair
from run_mos2_wall_width_transfer import DISTANCE, geometry_sections
from run_registry_prediction import rotation
from run_wall_heterostrain import SOURCE
from shared_compute import atomic_json, checkpoint_guard


SCANS = ("DS13", "DS14", "DS10")


def local_gradient(section, pixel_nm, scan_angle):
    points = section["coordinates"][..., ::-1].reshape(-1, 2) * pixel_nm
    fit = np.column_stack((points, np.ones(len(points))))
    phase = section["phase"].reshape(-1, 2)
    transform = np.linalg.lstsq(fit, phase, rcond=None)[0]
    np.testing.assert_allclose(fit @ transform, phase, atol=1e-10, rtol=0)
    return transform[:2].T @ rotation(-scan_angle)


def evaluate(context, images, modes, angle, original, cameras, comparison):
    pixel = original["pixel_nm"]
    basis, q, rings = (context[k] for k in ("basis", "reciprocal", "rings"))
    sections = geometry_sections(context, rotation(angle))
    old_walls = {tuple(w["centers"]): w for w in comparison["walls"]}
    physics = original["physics"]
    baseline_n = physics["grid"]
    adhesion = physics["stacking_coefficients_meV_per_cell"]["anchor"]
    global_area = abs(np.linalg.det(np.linalg.solve(context["gradient"], basis)))
    by_mode = {tuple(mode): i for i, mode in enumerate(modes)}
    measured = []
    for section, ref in zip(sections, original["walls"], strict=True):
        if section["centers"] != ref["centers"]:
            raise ValueError("The fixed width cohort changed")
        if tuple(section["centers"]) not in old_walls:
            continue
        checkpoint_guard()
        old = old_walls[tuple(section["centers"])]
        selected = [np.asarray([by_mode[tuple(m)] for m in group]) for group in ref["miller_indices"]]
        halves = paired_spots(modes, rings, selected)
        if [[modes[order].tolist() for order in side] for side in halves] != old["half_modes"]:
            raise ValueError("The opposite diffraction spots changed")
        gradient = local_gradient(section, pixel, angle)
        cell = np.linalg.solve(gradient, basis)
        mesh_nodes = np.max(np.linalg.norm(cell, axis=0)) / .30
        n = max(baseline_n, 2**int(np.ceil(np.log2(mesh_nodes))))
        if n > 512:
            raise ValueError(f"Local cell exceeds the archived grid limit at {section['centers']}: {n}")
        grid = make_grid(gradient, basis, n, *physics["stiffness_eV_per_cell"])
        motion, solver = minimize_all(grid, adhesion)
        registry = grid.fractional @ basis.T + motion
        freq2, cut, corners, weights = raster_coordinates(section, basis, pixel, n)
        np.testing.assert_allclose(np.sum(grid.k**2, axis=-1), freq2, atol=1e-9, rtol=1e-11)
        gain = gaussian_gain(freq2, NOMINAL)
        view = {**section, "keep": abs(DISTANCE) <= HALF_WINDOW}
        reader = get_observer(view, q, cameras[EVEN], rings, NOMINAL, selected)
        paired = [get_observer(view, q, cameras[EVEN], rings, NOMINAL, group) for group in halves]
        observation = sample_cuts(images, section["coordinates"]).mean(axis=0)
        scores = old["windows"][f"{HALF_WINDOW:g}"]
        for fit, frozen in zip([width_pair(half, observation) for half in paired], scores["measured"], strict=True):
            check_frozen(fit, frozen)
        report = {}
        for camera, coeff in cameras.items():
            spectrum = np.fft.fft2(detector_intensity(registry, q, coeff), axes=(0, 1))
            _, trace = render_traces(spectrum, gain, cut, corners, weights)
            report[camera] = {
                "widths": [width_pair(half, trace[CROP]) for half in paired],
                "profile": shape_errors(reader, observation, trace, paired),
            }
        measured.append({
            "centers": section["centers"], "local_gradient": gradient.tolist(), "grid": n,
            "local_to_global_cell_area": float(abs(np.linalg.det(cell)) / global_area),
            "solver": solver, "local": report,
        })
    if len(measured) != len(old_walls):
        raise ValueError("Some original wall sections were skipped")
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "release", "dependencies", "evidence", "global_result", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    paths = (SOURCE, BENCHMARK, SETTINGS, *(args.evidence / name for name in INPUT_NAMES), args.global_result)
    source, bench, settings, primary, control, optical, global_scores = [json.loads(p.read_bytes()) for p in paths]
    libraries = ("run_mos2_opposite_widths.py", "run_mos2_observation_controls.py", "run_mos2_wall_landscape.py",
                 "run_mos2_wall_width_transfer.py", "run_mos2_wall_profile_closure.py", "bilayer_relaxation.py",
                 "run_mos2_nonlinear_controls.py", "author_detector_replay.py")
    hashes = {p.name: digest(p) for p in (Path(__file__), *paths, *(PROJECT / name for name in libraries))}
    output = {"scope": "The four-center geometry is already used in the observation. Switch only the elastic "
                       "equilibrium from the full-map geometry to the local four-center cell. Use the nominal "
                       "alignment, stacking energy, probe, detector coefficients and +/-4.25 nm window.",
              "input_sha256": hashes, "requested_scans": list(SCANS), "scans": {}}
    elapsed = 0.
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        output = json.loads(args.output.read_bytes())
        if output["input_sha256"] != hashes or output["requested_scans"] != list(SCANS):
            raise ValueError("Resume requires the same source and wall cohort")
        elapsed = output["seconds"]
    author = load_author_release(args.release, args.dependencies)
    even = optical["frozen_common_coefficients"]
    for name in SCANS:
        if name in output["scans"]:
            continue
        checkpoint_guard()
        metadata = read_input_metadata(name, args.archive)
        images, modes, checksum = disk_images(args.archive, name, author)
        ref = source["scans"][name]
        if checksum != ref["diskset_sha256"] or metadata["archive_md5"] != bench["scans"][name]["archive_md5"]:
            raise ValueError(f"The archived data changed for {name}")
        fields = metadata["metadata_fields"]
        context = centers_and_context(images, modes, fields, settings, tuple(ref["interface"]))
        if context["receipt"] != ref["geometry"]:
            raise ValueError(f"The first-order geometry changed for {name}")
        angle = np.deg2rad(float(fields["SampleRotation"]))
        disjoint = control["scans"][name]["camera_diagnostic"]["by_model"]["anchor"]["coefficients"]
        cameras = {EVEN: np.broadcast_to([even[0], 0., even[1]], (len(modes), 3)), DISJOINT: np.asarray(disjoint)}
        output["scans"][name] = evaluate(context, images, modes, angle, primary["scans"][name], cameras,
                                         global_scores["scans"][name])
        output["seconds"] = elapsed + perf_counter() - started
        checkpoint_guard()
        atomic_json(args.output, output)
        print(name, len(output["scans"][name]), round(output["seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
