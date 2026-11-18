"""Test directional stretch in the energy at fixed measured moiré geometry."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import polar
from scipy.ndimage import gaussian_filter

from author_detector_replay import detector_intensity, rms
from author_release import load_author_release
from bilayer_relaxation import sample_disregistry
from observe_bilayers import load_dataset
from run_heterobilayer_sparse_geometry import solve_from_geometry
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_regime_benchmark import SETTINGS, detector_data
from run_mos2_template_transfer import ANTIPARALLEL, BENCHMARK, PARALLEL, make_context
from shared_compute import atomic_json, checkpoint_guard


def camera(field, context):
    values = detector_intensity(field, context["reciprocal"], context["coefficients"])
    width = context["probe_pixels"]
    return gaussian_filter(values, (width, width, 0), mode="nearest")


def compare(full, isotropic, context, support):
    measured = context["measured"]
    results = {}
    for name, ring in context["rings"].items():
        target = measured[support][:, ring]
        base = isotropic[support][:, ring]
        revised = full[support][:, ring]
        step = (revised-base).ravel()
        remaining = (target-base).ravel()
        power = step @ step
        results[name] = {
            "full_rmse": rms(target, revised), "isotropic_relaxation_rmse": rms(target, base),
            "full_minus_isotropic_signal_rms": float(np.sqrt(power/len(step))),
            "observed_amplitude_along_directional_correction": float((step @ remaining)/power) if power > 1e-15 else None,
        }
    return results


def wall_sectors(background, basis):
    """Select the three saddle-point regions using the common geometry alone."""
    fractional = (background @ np.linalg.inv(basis).T + 0.5) % 1 - 0.5
    targets = np.array(((0.5, 0), (0, 0.5), (0.5, -0.5)))
    # The scan disregistry uses the 60° primitive lattice. The third
    # direction is the difference of its two primitive vectors.
    delta = (fractional[..., None, :] - targets + 0.5) % 1 - 0.5
    offsets = np.array(((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)))
    distance = np.full(fractional.shape[:2] + (3,), np.inf)
    for offset in offsets:
        distance = np.minimum(distance, np.linalg.norm((delta + offset) @ basis.T, axis=-1))
    lattice = np.linalg.norm(basis[:, 0])
    nearest = np.argmin(distance, axis=-1)
    return {str(i+1): (nearest == i) & (distance[..., i] <= 0.13*lattice) for i in range(3)}


def evaluate(context, reference):
    interface = tuple(reference["interface"])
    gradient = np.asarray(reference["affine_geometry"]["gradient_crystal"])
    basis = context["basis"]
    angle, stretch = polar(np.eye(2) + gradient)
    isotropic_gradient = np.sqrt(np.linalg.det(stretch)) * angle - np.eye(2)
    determinant = np.linalg.det(np.eye(2) + gradient)
    if abs(np.linalg.det(np.eye(2) + isotropic_gradient) - determinant) > 1e-12:
        raise ValueError("Removing directional stretch changed the fitted area ratio")
    phase_origin = basis @ np.full(2, 2/3 if interface[-1] == "AP" else 0)
    rigid = context["fields"]["rigid"]
    grid, motion, solver = solve_from_geometry(isotropic_gradient, basis, interface)
    observed_cell = np.linalg.solve(gradient, basis)
    isotropic_field = sample_disregistry(grid, motion, phase_origin+rigid, observed_cell) - phase_origin
    full, iso = (camera(field, context) for field in (context["fields"]["full"], isotropic_field))
    support = context["support"]
    rows = {"all_pixels": compare(full, iso, context, support)}
    for key, values in rows["all_pixels"].items():
        if abs(values["full_rmse"] - reference["models"]["warped"]["full"]["diffraction_rmse"][key]) > 1e-10:
            raise ValueError("The common full prediction changed")
    if interface[-1] == "P":
        sectors = wall_sectors(rigid, basis)
        for key, wall in sectors.items():
            selected = support & wall
            if selected.sum() >= 25:
                rows[f"wall_sector_{key}"] = compare(full, iso, context, selected)
        any_wall = np.logical_or.reduce(tuple(sectors.values()))
        if (support & any_wall).sum() >= 75:
            rows["wall_regions"] = compare(full, iso, context, support & any_wall)
    return {"interface": interface, "score_pixels": int(support.sum()),
            "directional_stretch_percent": float(100*np.ptp(np.linalg.eigvalsh(stretch))),
            "isotropic_relaxation": solver, "readouts": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    benchmark = json.loads(BENCHMARK.read_text())
    for name, expected in benchmark["input_sha256"].items():
        if digest(PROJECT / name) != expected:
            raise ValueError(f"The common benchmark input {name} changed")
    settings = json.loads(SETTINGS.read_text())
    p = json.loads(PARALLEL.read_text())["batches"]
    ap = json.loads(ANTIPARALLEL.read_text())
    cases = p["development"] + p["held_out"] + ap["development"] + ap["replay"]
    author = load_author_release(args.release, args.dependencies)
    result = {
        "scope": "Compare relaxation on measured affine strain to relaxation computed without directional strain, evaluated on the identical nonuniform measured geometry. Preserve mean twist and area. Predict normalized integrated diffraction for all 24 unique MoS2 maps, and on three preassigned geometry-only saddle-point regions for the parallel maps. The projected directional signal amplitude is a diagnostic and is never fed back into either model.",
        "input_sha256": {f.name: digest(f) for f in (Path(__file__), BENCHMARK, SETTINGS, PARALLEL, ANTIPARALLEL)},
        "scans": {},
    }
    for scan in cases:
        checkpoint_guard()
        data = load_dataset(scan, args.archive, allowed_ids=cases, allow_author_crop=True)
        reference = benchmark["scans"][scan]
        if data.archive_md5 != reference["archive_md5"]:
            raise ValueError("The archive changed")
        detector = detector_data(args.archive, scan, author)
        context = make_context(data, detector, reference, settings)
        row = evaluate(context, reference)
        result["scans"][scan] = row
        result["wall_seconds"] = perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, result)
        print(scan, row["directional_stretch_percent"], row["readouts"]["all_pixels"]["both"], flush=True)


if __name__ == "__main__":
    main()
