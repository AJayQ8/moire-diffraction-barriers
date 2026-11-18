"""Recalculate the supporting manuscript comparisons from case-level results."""

import argparse
import json
from math import isclose
from pathlib import Path
from statistics import mean, median

from build_tables import load


def high_twist_scans(readouts):
    return {name: scan for name, scan in readouts["scans"].items() if scan["twist_deg"] >= 1.0}


def profile_transfer(readouts, observed, predicted):
    """Compare one source-order fit per model on the other order."""
    by_map = []
    for name in high_twist_scans(readouts):
        cases = []
        for measured, periodic in zip(observed["scans"][name]["walls"], predicted["scans"][name]["walls"],
                                      strict=True):
            if measured["centers"] != periodic["centers"]:
                raise ValueError(f"{name}: wall IDs changed")
            observed_fits = measured["fits"]["1.25"]["separate"]
            estimated = periodic["probe"]["1.25"]["periodic_anchor"]["transfer"]
            cases.append([fit[source]["error_by_ring"][1 - source]
                          for source in (0, 1) for fit in (estimated, observed_fits)])
        by_map.append([mean(values) for values in zip(*cases)])
    mean_errors = [mean(values) for values in zip(*by_map)]
    return [1 - mean_errors[i] / mean_errors[i + 1] for i in (0, 2)]


def displacement_substitutions(readouts, substitutions):
    """Evaluate core and surrounding displacement on identical matched walls."""
    kinds = ("surroundings_1.5nm", "core_1.5nm", "full")
    errors = {name: [] for name in kinds}
    for name, scan in high_twist_scans(readouts).items():
        candidates = {tuple(row["centers"]): row for row in substitutions["scans"][name]["walls"]}
        by_model = {kind: [] for kind in kinds}
        for chord in scan["chords"]:
            measured = chord["widths_nm"]["1.25"]["measured"]
            gap = measured[0] - measured[1]
            comparison = candidates[tuple(chord["centers"])]
            if not isclose(gap, comparison["measured_width_difference_nm"], abs_tol=1e-10):
                raise ValueError(f"{name}: measured widths changed")
            for kind in kinds:
                result = comparison["models"][kind]
                if not result["widths_qualified"]:
                    raise ValueError(f"{name}: {kind} lacks a matched width")
                by_model[kind].append(result["width_difference_nm"] - gap)
        for kind in kinds:
            errors[kind].append(abs(mean(by_model[kind])))
    return {name: mean(values) for name, values in errors.items()}


def repeatability(records, window):
    map_scores = [row[window] for row in records["summary"].values()]
    repeats = []
    for scan in records["scans"].values():
        for chord in scan["walls"]:
            if window in chord["fits"]:
                left, right = chord["fits"][window]["opposite"]
                if left["qualified"] and right["qualified"]:
                    repeats.append(abs(left["gap_nm"] - right["gap_nm"]))
    return {
        "per_map_rmse_nm": mean(row["within_scan_rmse_nm"] for row in map_scores),
        "constant_rmse_nm": mean(row["constant_rmse_nm"] for row in map_scores),
        "median_repeat_disagreement_nm": median(repeats),
        "map_wins": sum(row["within_scan_rmse_nm"] < row["constant_rmse_nm"] for row in map_scores),
        "qualified_repeat_pairs": len(repeats),
    }


def supplied_probe(recovery, probe):
    if recovery["images_sha256"] != probe["images_sha256"]:
        raise ValueError("Supplied-probe fits use different images")
    if recovery["truth_sha256"] != probe["sealed_truth_sha256"]:
        raise ValueError("Supplied-probe fits use a different truth key")
    bins = {}
    for name, result in probe["samples"].items():
        sample = recovery["samples"][name]
        if sample["condition"] != "variable_probe_and_landscape":
            continue
        barrier = sample["truth"]["barrier_meV_per_cell"]
        rows = bins.setdefault(sample["spacing_nm"], [])
        for trial in result["fits"]:
            fit = trial["profile_shapes"]
            low, high = fit["barrier_interval"]
            rows.append((abs(fit["barrier_meV_per_cell"] / barrier - 1), (high - low) / barrier))
    return {f"{spacing:.2f}": {
        "median_absolute_relative_error": median(error for error, _ in fits),
        "median_relative_interval_width": median(width for _, width in fits),
        "within_20_percent": sum(error <= .2 for error, _ in fits),
        "captures": len(fits),
    } for spacing, fits in sorted(bins.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path,
                        default=Path(__file__).resolve().parents[2] / "evidence")
    args = parser.parse_args()
    evidence = args.evidence
    primary = load(evidence, "moire_wall_energy_readout_v01")
    repeat = load(evidence, "mos2_wall_signal_repeatability_v01")
    checks = {
        "fractional_profile_error_reduction": profile_transfer(
            primary, load(evidence, "mos2_wall_width_transfer_v01"), load(evidence, "mos2_wall_profile_closure_v01")),
        "matched_displacement_substitution_mae_nm": displacement_substitutions(
            primary, load(evidence, "mos2_wall_surroundings_v01")),
        "adaptive_window_repeatability": repeatability(repeat, "adaptive"),
        "fixed_window_repeatability": repeatability(repeat, "fixed_4.25_nm"),
        "known_probe_combined_uncertainty": supplied_probe(
            load(evidence, "mos2_barrier_recovery_v01"), load(evidence, "mos2_barrier_probe_control_v01")),
    }
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
