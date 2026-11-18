"""Locate the Gaussian probe crossover for the two Raman-matched MoS2 models."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from author_release import load_author_release
from bilayer_relaxation import make_grid
from observe_bilayers import read_input_metadata
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_intensity_only_replay import disk_images
from run_mos2_nonlinear_controls import minimize_all
from run_mos2_wall_landscape import CROP, get_observer, width_pair
from run_mos2_wall_profile_closure import fields_on_chord, render_profiles, transfer
from run_mos2_wall_width_transfer import DISTANCE, REDUCED, actual_profiles, geometry_sections
from run_registry_prediction import rotation
from run_wall_heterostrain import SOURCE
from shared_compute import atomic_json, checkpoint_guard


CAMERA = PROJECT / "mos2_camera_transfer_v01.json"
PRIMARY = PROJECT / "mos2_wall_landscape_v01.json"
CONTROL = PROJECT / "mos2_wall_landscape_camera_controls_v01.json"
MODELS = ("raman_shape", "raman_scale")
CONDITIONS = (1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60, 1.65, 1.70)
METHOD = "Reuse the frozen materials, calibrated camera coefficients, 4.25 nm half-window and wall locations. Fill in the Gaussian probe interval bracketed by the nominal 1.25 and alternate 1.75 nm results. Keep stiffness, camera, reflections and profile scoring fixed."


def score_scan(context, images, modes, meta, original, control, compact, even_terms):
    basis, gradient, q, rings = (context[key] for key in ("basis", "gradient", "reciprocal", "rings"))
    angle = np.deg2rad(float(meta["SampleRotation"]))
    sections = geometry_sections(context, rotation(angle))
    if [s["centers"] for s in sections] != [w["centers"] for w in original["walls"]]:
        raise ValueError("A wall section changed")
    physics = original["physics"]
    anchor = np.asarray(physics["stacking_coefficients_meV_per_cell"]["anchor"])
    stiffness = physics["stiffness_eV_per_cell"]
    grid = make_grid(gradient, basis, physics["grid"], *stiffness)
    motion, solvers = {}, {}
    for name in MODELS:
        coefficients = np.asarray(control["raman_constraint"]["models"][name]["coefficients_meV_per_cell"])
        motion[name], solvers[name] = minimize_all(grid, coefficients)
    nominal = np.broadcast_to(np.array([even_terms[0], 0., even_terms[1]]), (len(q), 3))
    own = {name: np.asarray(control["camera_diagnostic"]["by_model"][name]["coefficients"]) for name in MODELS}

    walls = []
    mode_keys = {tuple(row): i for i, row in enumerate(modes)}
    for section, ref in zip(sections, original["walls"], strict=True):
        if section["reach_nm"] < 4.25 or not ref["original_cohort"]:
            continue
        checkpoint_guard()
        subset = {**section, "keep": abs(DISTANCE) <= 4.25}
        actual = actual_profiles(images, section["coordinates"])
        channels = [np.asarray([mode_keys[tuple(channel)] for channel in group], dtype=int)
                    for group in ref["miller_indices"]]
        profiles = {}
        for probe in CONDITIONS:
            fields, weights = fields_on_chord(
                section, context, angle, grid, motion["raman_shape"], grid, motion["raman_scale"],
                compact, {"anchor": (anchor, stiffness)}, probe)
            fields = {"raman_shape": fields["periodic_anchor"], "raman_scale": fields["periodic_alternate"]}
            observer = get_observer(subset, q, nominal, rings, probe, channels)
            measured = width_pair(observer, actual)
            observed_shape = observer.shape(actual)
            profiles[f"{probe:g}"] = {}
            for cam, coefficients in (("transferred_even", {name: nominal for name in MODELS}),
                                      ("separate_disjoint", own)):
                fits = {}
                for name, field in fields.items():
                    response = render_profiles({name: field}, q, coefficients[name], weights, probe)[name]
                    fit = width_pair(observer, response[CROP])
                    results = transfer(observer, observed_shape, response)
                    fit["transfer_error"] = [row["error_by_ring"][1-i] for i, row in enumerate(results)]
                    fits[name] = fit
                profiles[f"{probe:g}"][cam] = {"measured": measured, "models": fits}
        walls.append({"centers": ref["centers"], "readouts": profiles})
    return {"solvers": solvers, "walls": walls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    start = perf_counter()
    inputs = SOURCE, BENCHMARK, SETTINGS, REDUCED, CAMERA, PRIMARY, CONTROL
    source, benchmark, settings, reduced, optics, primary, control = (json.loads(p.read_text()) for p in inputs)
    paths = (Path(__file__), *inputs, PROJECT / "run_mos2_wall_landscape_camera_controls.py",
             PROJECT / "run_mos2_raman_probe_sensitivity.py")
    hashes = {p.name: digest(p) for p in paths}
    if digest(PRIMARY) != control["input_sha256"][PRIMARY.name]:
        raise ValueError("The input camera analysis used a different set of primary scores")
    report = {"method": METHOD, "probes_nm": CONDITIONS, "input_sha256": hashes, "scans": {}}
    elapsed = 0.
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        report = json.loads(args.output.read_text())
        if report["input_sha256"] != hashes:
            raise ValueError("The probe sensitivity input freeze changed")
        elapsed = report["seconds"]
    order = primary["protocol"]["scans"]
    if set(order) != set(control["scans"]) or set(order) != set(primary["scans"]):
        raise ValueError("The nominal experiments are incomplete")
    author = load_author_release(args.release, args.dependencies)
    for scan in order:
        if scan in report["scans"]:
            continue
        checkpoint_guard()
        meta = read_input_metadata(scan, args.archive)
        images, modes, file_hash = disk_images(args.archive, scan, author)
        saved = source["scans"][scan]
        if file_hash != saved["diskset_sha256"] or meta["archive_md5"] != benchmark["scans"][scan]["archive_md5"]:
            raise ValueError(f"A source archive changed in {scan}")
        interface = tuple(saved["interface"])
        view = centers_and_context(images, modes, meta["metadata_fields"], settings, interface)
        if view["receipt"] != saved["geometry"]:
            raise ValueError(f"The geometry changed in {scan}")
        compact = np.asarray(reduced["scans"][scan]["coefficients_nm"]["three_stars"])
        report["scans"][scan] = score_scan(view, images, modes, meta["metadata_fields"], primary["scans"][scan],
                                          control["scans"][scan], compact, optics["frozen_common_coefficients"])
        report["seconds"] = elapsed + perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, report)
        print(scan, len(report["scans"][scan]["walls"]), round(report["seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
