"""Replay fixed DS31 strain readouts on unused rows of the same acquisition."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from author_detector_replay import read_detector_fit
from author_release import load_author_release
from run_heterobilayer_raw_arc import read_trace, select_arcs, solve_profiles, take_profiles
from run_heterobilayer_raw_ds31 import (
    FIRST_COLUMN, ROWS_PER_CHUNK, SCAN, WINDOW, fit_patch, load_counts, model_trace_comparison,
)
from run_heterobilayer_raw_injection import HIGH, RAW
from run_heterobilayer_raw_ring_controls import block_counts, higher_orders
from run_heterobilayer_transfer import PROJECT, digest
from shared_compute import atomic_json, checkpoint_guard


ORIGINAL_ARCS = PROJECT / "heterobilayer_raw_arc_v01.json"
REPLAY_ROWS = (15, 22, 29, 36, 43, 50, 58, 66, 74, 82, 89)
INPUTS = (Path(__file__), RAW, HIGH, ORIGINAL_ARCS,
          *(PROJECT / name for name in ("run_heterobilayer_raw_ds31.py", "run_heterobilayer_raw_arc.py",
                                       "run_heterobilayer_raw_ring_controls.py", "run_heterobilayer_raw_injection.py")))


def freeze_receipt():
    return {
        "dataset": "DS31 Diffraction SI.dm4",
        "original_rows": json.loads(RAW.read_text())["source"]["sample_rows"],
        "replay_rows": REPLAY_ROWS,
        "columns": (FIRST_COLUMN, FIRST_COLUMN + ROWS_PER_CHUNK - 1),
        "blocks": "Ten neighboring scan positions on the same row. Run both readouts on the squared reciprocal order 9.",
        "geometry": "Use the existing full periodic model transported onto the measured stacking-center geometry. Fit arc templates and reciprocal starting parameters only on the original 44 blocks.",
        "primary_comparison": "On shared interior blocks report centered trace RMSE, score of a constant trace, measured and predicted variability, and correlation. Both readouts must favor the model for a positive replay. Report channel disagreement and alternating-pixel repeatability.",
        "scope": "The new raw pixels are spatially disjoint from the original raw edge samples. Both batches share an acquisition and the original archived geometry.",
        "input_sha256": {p.name: digest(p) for p in INPUTS},
    }


def summarize(used):
    predicted = np.array([b["matched_model_dilations_percent"]["warped_relaxed"] for b in used])
    result = {"blocks": len(used), "model_variability_pp": float(predicted.std()), "readouts": {}}
    observed = {}
    for name in ("paired", "arcs"):
        center = np.array([b[name]["mean"] for b in used])
        even = np.array([b[name]["even"] for b in used])
        odd = np.array([b[name]["odd"] for b in used])
        result["readouts"][name] = {
            "mean_trace_percent": float(center.mean()),
            "trace_variability_pp": float(center.std()),
            "model_centered_rmse_pp": float((predicted - center).std()),
            "constant_centered_rmse_pp": float(center.std()),
            "model_correlation": float(np.corrcoef(predicted, center)[0, 1]),
            "even_odd_rms_disagreement_pp": float(np.sqrt(np.mean((even - odd)**2))),
        }
        observed[name] = center
    result["readout_agreement"] = {
        "centered_disagreement_pp": float((observed["paired"] - observed["arcs"]).std()),
        "correlation": float(np.corrcoef(observed["paired"], observed["arcs"])[0, 1]),
    }
    result["model_beats_constant_on_both"] = all(
        v["model_centered_rmse_pp"] < v["constant_centered_rmse_pp"] for v in result["readouts"].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("freeze", type=Path)
    parser.add_argument("--write-freeze", action="store_true")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--dependencies", type=Path)
    parser.add_argument("--original-ranges", type=Path)
    parser.add_argument("--replay-ranges", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint_guard()
    receipt = freeze_receipt()
    if args.write_freeze:
        if args.freeze.exists():
            raise ValueError("The original spatial replay freeze must remain intact")
        atomic_json(args.freeze, receipt)
        return
    if any(getattr(args, path) is None for path in ("archive", "release", "dependencies", "original_ranges",
                                                   "replay_ranges", "output")):
        parser.error("Supply both ranges, archive and analysis paths for the frozen replay")
    if json.loads(json.dumps(receipt)) != json.loads(args.freeze.read_text()):
        raise ValueError("The spatial replay protocol or inputs changed")
    start = perf_counter()
    raw = json.loads(RAW.read_text())
    high = json.loads(HIGH.read_text())
    if set(REPLAY_ROWS) & set(raw["source"]["sample_rows"]):
        raise ValueError("The replay contains an original calibration row")
    for name, expected in raw["range_sha256"].items():
        checkpoint_guard()
        if digest(args.original_ranges / name) != expected:
            raise ValueError(f"The original raw DS31 data changed in {name}")
    author = load_author_release(args.release, args.dependencies)
    disk, _, _, fit_hash = read_detector_fit(args.archive, SCAN, "fit", author)
    if high["input_sha256"]["DS31_fit.pkl"] != fit_hash:
        raise ValueError("The native reciprocal geometry changed")
    mask = disk._in_use.astype(bool)
    positions = np.column_stack((disk._xvec[mask], disk._yvec[mask]))
    modes = disk.clean_normgset(sanity_plot=False)
    source, _, alignment, raw_hashes = load_counts(
        args.replay_ranges, positions, disk._rvec[mask], disk.df_set())
    expected_source = {**raw["source"], "sample_rows": list(REPLAY_ROWS)}
    if source != expected_source:
        raise ValueError("The unused raw samples do not match the frozen coordinates")
    high_modes, nominal = higher_orders(modes, positions)
    h, k = high_modes.T
    outer = h*h + h*k + k*k == 9
    high_modes, nominal = high_modes[outer], nominal[outer]
    centers, original = block_counts(args.original_ranges, nominal, raw["source"])
    _, held = block_counts(args.replay_ranges, nominal, source)
    grand = np.mean([b["mean"] for b in original], axis=0)
    setup = high["aggregate"]["higher_order_9"]
    theta = np.asarray(setup["geometry_fit_parameters"])
    bound = setup["reciprocal_matrix_bound_pixels"]
    radius = 1.71 / source["detector_spacing_mrad"]
    arcs = select_arcs(grand, nominal, centers, high_modes, radius, theta)
    prior_arcs = json.loads(ORIGINAL_ARCS.read_text())["groups"]["higher_order_9"]
    if arcs[-1].tolist() != prior_arcs["arcs_per_layer"]:
        raise ValueError("The frozen arc template count changed")
    output = {
        "scope": receipt["scope"], "freeze_sha256": digest(args.freeze), "source": source,
        "range_sha256": raw_hashes, "arcs_per_layer": arcs[-1].tolist(),
        "alignment": [{k: v for k, v in row.items() if k != "counts"} for row in alignment],
        "blocks": [],
    }
    for block in held:
        checkpoint_guard()
        row = {k: block[k] for k in ("row", "column_start", "column_count")}
        row.update(paired={}, arcs={}, arc_diagnostics={}, paired_diagnostics={})
        for phase in ("mean", "even", "odd"):
            counts = block[phase]
            _, reading = fit_patch(counts, high_modes, nominal, centers, radius, theta, matrix_bound=bound)
            edge, diagnostics = solve_profiles(take_profiles(counts, arcs), arcs)
            row["paired"][phase] = reading["relative_trace_percent"]
            row["arcs"][phase] = read_trace(theta + edge, arcs[6])
            row["paired_diagnostics"][phase] = {k: reading[k] for k in (
                "converged", "poisson_deviance_per_dof", "reciprocal_matrix_parameters_at_bound")}
            row["arc_diagnostics"][phase] = diagnostics
        row["both_all"] = {"relative_trace_percent": row["paired"]["mean"]}
        output["blocks"].append(row)
    output["paired_model_comparison"] = model_trace_comparison(args.archive, output["blocks"])
    used = [r for r in output["blocks"] if "matched_model_dilations_percent" in r]
    if len(used) < 20:
        raise ValueError("Too few frozen spatial blocks match the original interior model")
    output["summary"] = summarize(used)
    output["wall_seconds"] = perf_counter() - start
    checkpoint_guard()
    atomic_json(args.output, output)
    print(output["summary"], flush=True)


if __name__ == "__main__":
    main()
