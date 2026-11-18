#!/usr/bin/env python3
"""Summarize cross-order barrier agreement in the frozen synthetic fits.

This is a read-only reanalysis of saved fits and revealed simulation truths. It
does not regenerate images, refit profiles, interpolate barriers, or extrapolate
beyond the 18 saved scenarios.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PAPER = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PAPER / "evidence" / "mos2_barrier_recovery_v01.json"
DEFAULT_OUTPUT = PAPER / "evidence" / "mos2_order_agreement_v01.json"
MODELS = ("isolated", "periodic")
ORDERS = ("first", "second", "both")
FIT_STYLE = "profile_shapes"
AGREEMENT_THRESHOLDS = (1.10, 1.20, 1.30)
ERROR_FRACTION = 0.20
REALIZATIONS_PER_SCENARIO = 12


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def qualified_point_fit(fit: dict[str, Any]) -> bool:
    """Keep finite interior barrier fits; intervals are reported, not filtered."""
    return (
        fit.get("hit_barrier_bound") is False
        and finite_positive(fit.get("barrier_meV_per_cell"))
    )


def valid_interval(fit: dict[str, Any]) -> bool:
    interval = fit.get("barrier_interval")
    if not isinstance(interval, list) or len(interval) != 2:
        return False
    low, high = interval
    return (
        finite_positive(low)
        and finite_positive(high)
        and float(low) <= float(high)
    )


def fit_record(fit: dict[str, Any], truth: float) -> dict[str, Any]:
    estimate = float(fit["barrier_meV_per_cell"])
    interval = [float(value) for value in fit["barrier_interval"]]
    return {
        "barrier_meV_per_cell": estimate,
        "barrier_interval_meV_per_cell": interval,
        "absolute_relative_error": abs(estimate / truth - 1.0),
        "error_gt_20_percent": abs(estimate / truth - 1.0) > ERROR_FRACTION,
        "interval_contains_truth": interval[0] <= truth <= interval[1],
        "qualified_interior_fit": qualified_point_fit(fit),
        "hit_barrier_bound": fit.get("hit_barrier_bound"),
        "probe_fwhm_nm": fit.get("probe_fwhm_nm"),
        "landscape_mixture": fit.get("landscape_mixture"),
        "shift_nm": fit.get("shift_nm"),
        "reduced_noise_loss": fit.get("reduced_noise_loss"),
    }


def validate_input(data: dict[str, Any]) -> None:
    samples = data.get("samples")
    if not isinstance(samples, dict) or len(samples) != 18:
        raise ValueError("Expected exactly 18 saved synthetic scenarios")
    grouped: dict[tuple[float, str], list[str]] = defaultdict(list)
    for sample_id, sample in samples.items():
        spacing = float(sample["spacing_nm"])
        condition = str(sample["condition"])
        truth = sample["truth"].get("barrier_meV_per_cell")
        if not finite_positive(truth):
            raise ValueError(f"{sample_id}: missing positive known barrier truth")
        grouped[(spacing, condition)].append(sample_id)
        for model in MODELS:
            if model not in sample.get("fits", {}):
                raise ValueError(f"{sample_id}: missing model {model}")
            for order in ORDERS:
                fits = sample["fits"][model].get(order, {}).get(FIT_STYLE)
                if not isinstance(fits, list) or len(fits) != REALIZATIONS_PER_SCENARIO:
                    raise ValueError(f"{sample_id}/{model}/{order}: expected 12 saved profile-shape fits")
    if len(grouped) != 6 or any(len(ids) != 3 for ids in grouped.values()):
        raise ValueError("Expected three truth scenarios for each spacing/condition group")


def summarize_group(
    sample_ids: list[str], samples: dict[str, Any], model: str
) -> dict[str, Any]:
    candidates = []
    scenario_rows = []
    for sample_id in sorted(sample_ids):
        sample = samples[sample_id]
        truth = float(sample["truth"]["barrier_meV_per_cell"])
        first_fits = sample["fits"][model]["first"][FIT_STYLE]
        second_fits = sample["fits"][model]["second"][FIT_STYLE]
        both_fits = sample["fits"][model]["both"][FIT_STYLE]
        eligible_indices = []
        exclusions = {"first_bound_or_invalid": 0, "second_bound_or_invalid": 0}
        if len(first_fits) != len(second_fits):
            raise ValueError(f"{sample_id}/{model}: first/second realization counts differ")
        for index, (first, second) in enumerate(zip(first_fits, second_fits)):
            first_ok = qualified_point_fit(first)
            second_ok = qualified_point_fit(second)
            if not first_ok:
                exclusions["first_bound_or_invalid"] += 1
            if not second_ok:
                exclusions["second_bound_or_invalid"] += 1
            if not (first_ok and second_ok):
                continue
            eligible_indices.append(index)
            candidates.append({
                "sample_id": sample_id,
                "realization_index_zero_based": index,
                "truth": truth,
                "first": float(first["barrier_meV_per_cell"]),
                "second": float(second["barrier_meV_per_cell"]),
            })

        scenario_rows.append({
            "sample_id": sample_id,
            "truth_barrier_meV_per_cell": truth,
            "candidate_pairs": len(first_fits),
            "qualified_pairs": len(eligible_indices),
            "excluded_first_fits": exclusions["first_bound_or_invalid"],
            "excluded_second_fits": exclusions["second_bound_or_invalid"],
            "both_order_fit_qualification": {
                "qualified_count": sum(qualified_point_fit(fit) for fit in both_fits),
                "candidate_count": len(both_fits),
            },
        })

    threshold_rows: dict[str, Any] = {}
    for threshold in AGREEMENT_THRESHOLDS:
        agreed = [
            row for row in candidates
            if max(row["first"], row["second"]) / min(row["first"], row["second"]) <= threshold
        ]
        miss_flags = [
            (
                abs(row["first"] / row["truth"] - 1.0) > ERROR_FRACTION,
                abs(row["second"] / row["truth"] - 1.0) > ERROR_FRACTION,
            )
            for row in agreed
        ]
        either = sum(first_miss or second_miss for first_miss, second_miss in miss_flags)
        both = sum(first_miss and second_miss for first_miss, second_miss in miss_flags)
        first_misses = sum(first_miss for first_miss, _ in miss_flags)
        second_misses = sum(second_miss for _, second_miss in miss_flags)
        threshold_rows[f"{threshold:.2f}"] = {
            "agreement_factor_threshold": threshold,
            "agreement_definition": "max(first, second) / min(first, second) <= threshold",
            "agree_pairs": len(agreed),
            "agree_fraction_of_qualified_pairs": len(agreed) / len(candidates) if candidates else None,
            "among_agreeing_pairs": {
                "first_order_misses_gt_20_percent": first_misses,
                "second_order_misses_gt_20_percent": second_misses,
                "either_order_misses_gt_20_percent": either,
                "both_orders_miss_gt_20_percent": both,
                "exactly_one_order_misses_gt_20_percent": either - both,
                "neither_order_misses_gt_20_percent": len(agreed) - either,
            },
        }

    all_first_miss = sum(abs(row["first"] / row["truth"] - 1.0) > ERROR_FRACTION for row in candidates)
    all_second_miss = sum(abs(row["second"] / row["truth"] - 1.0) > ERROR_FRACTION for row in candidates)
    all_both_miss = sum(
        abs(float(fit["barrier_meV_per_cell"]) / float(samples[row["sample_id"]]["truth"]["barrier_meV_per_cell"]) - 1.0)
        > ERROR_FRACTION
        for row in candidates
        for fit in [samples[row["sample_id"]]["fits"][model]["both"][FIT_STYLE][row["realization_index_zero_based"]]]
        if qualified_point_fit(fit)
    )
    # The 1.20 threshold is the prespecified primary comparison requested for
    # this summary; the adjacent thresholds expose agreement-cutoff sensitivity.
    primary = threshold_rows["1.20"]
    total_pairs = len(sample_ids) * REALIZATIONS_PER_SCENARIO
    return {
        "candidate_order_pairs": total_pairs,
        "qualified_matched_order_pairs": len(candidates),
        "excluded_pairs_due_to_bound_or_invalid_single_order_fit": total_pairs - len(candidates),
        "qualified_both_order_fits": sum(row["both_order_fit_qualification"]["qualified_count"] for row in scenario_rows),
        "candidate_both_order_fits": total_pairs,
        "truth_error_gt_20_percent_all_qualified_pairs": {
            "first_order": all_first_miss,
            "second_order": all_second_miss,
            "both_order_fit": all_both_miss,
        },
        "agreement_sensitivity": threshold_rows,
        "primary_factor_1_20": {
            "agree_pairs": primary["agree_pairs"],
            "either_order_misses_gt_20_percent_among_agreeing": primary["among_agreeing_pairs"]["either_order_misses_gt_20_percent"],
            "both_orders_miss_gt_20_percent_among_agreeing": primary["among_agreeing_pairs"]["both_orders_miss_gt_20_percent"],
        },
        "scenario_breakdown": scenario_rows,
    }


def example_case(samples: dict[str, Any]) -> dict[str, Any]:
    sample_id = "sample_06"
    realization = 4
    sample = samples[sample_id]
    truth = float(sample["truth"]["barrier_meV_per_cell"])
    model = "periodic"
    orders = {
        order: fit_record(sample["fits"][model][order][FIT_STYLE][realization], truth)
        for order in ORDERS
    }
    first, second = orders["first"], orders["second"]
    agreement_factor = max(first["barrier_meV_per_cell"], second["barrier_meV_per_cell"]) / min(
        first["barrier_meV_per_cell"], second["barrier_meV_per_cell"]
    )
    return {
        "sample_id": sample_id,
        "realization_index_zero_based": realization,
        "spacing_nm": float(sample["spacing_nm"]),
        "condition": sample["condition"],
        "model": model,
        "fit_style": FIT_STYLE,
        "truth_barrier_meV_per_cell": truth,
        "first_second_agreement_factor": agreement_factor,
        "agrees_at_primary_factor_1_20": agreement_factor <= 1.20,
        "fits": orders,
    }


def build_summary(data: dict[str, Any], input_bytes: bytes, script_path: Path) -> dict[str, Any]:
    validate_input(data)
    grouped: dict[tuple[float, str], list[str]] = defaultdict(list)
    for sample_id, sample in data["samples"].items():
        grouped[(float(sample["spacing_nm"]), str(sample["condition"]))].append(sample_id)

    groups = []
    for (spacing, condition), sample_ids in sorted(grouped.items()):
        models = {model: summarize_group(sample_ids, data["samples"], model) for model in MODELS}
        groups.append({
            "spacing_nm": spacing,
            "condition": condition,
            "scenario_count": len(sample_ids),
            "realizations_per_scenario": REALIZATIONS_PER_SCENARIO,
            "known_truth_barriers_meV_per_cell": sorted(
                float(data["samples"][sample_id]["truth"]["barrier_meV_per_cell"])
                for sample_id in sample_ids
            ),
            "models": models,
        })

    target_group = next(
        group for group in groups
        if math.isclose(group["spacing_nm"], 28.6, abs_tol=0.1)
        and group["condition"] == "variable_probe"
    )
    periodic_primary = target_group["models"]["periodic"]["primary_factor_1_20"]
    if (
        periodic_primary["agree_pairs"] != 15
        or periodic_primary["either_order_misses_gt_20_percent_among_agreeing"] != 6
        or periodic_primary["both_orders_miss_gt_20_percent_among_agreeing"] != 5
    ):
        raise ValueError(f"Expected the saved 5/15 agreement example; observed {periodic_primary}")

    example = example_case(data["samples"])
    if not (
        math.isclose(example["truth_barrier_meV_per_cell"], 19.972020319440855, abs_tol=1e-10)
        and math.isclose(example["fits"]["first"]["barrier_meV_per_cell"], 32.82339825390523, abs_tol=1e-10)
        and math.isclose(example["fits"]["second"]["barrier_meV_per_cell"], 32.82339825390523, abs_tol=1e-10)
    ):
        raise ValueError("The sample_06 realization-4 reference example changed")

    return {
        "schema": "mos2_order_agreement_v01",
        "scope": (
            "Read-only reanalysis of the frozen known-barrier recovery fits. No images were regenerated, "
            "no profiles were refit, no truth values were inferred, and no interpolation or extrapolation was used."
        ),
        "provenance": {
            "input_file": "evidence/mos2_barrier_recovery_v01.json",
            "input_sha256": sha256_bytes(input_bytes),
            "fit_before_reveal_sha256": data.get("fit_before_reveal_sha256"),
            "truth_sha256": data.get("truth_sha256"),
            "images_sha256": data.get("images_sha256"),
            "source_sha256": data.get("source_sha256"),
            "simulation_seed": data.get("simulation_seed"),
            "analysis_script": "analysis/summarize_mos2_order_agreement.py",
            "analysis_script_sha256": sha256_bytes(script_path.read_bytes()),
        },
        "method": {
            "fit_style": FIT_STYLE,
            "models": list(MODELS),
            "reported_fit_orders": list(ORDERS),
            "agreement_pair": ["first", "second"],
            "qualification": (
                "A first/second pair is eligible only when both saved profile_shapes point estimates are finite, "
                "positive, and not flagged hit_barrier_bound. There is no additional fit-success field in the "
                "sealed output. Barrier intervals are reported but do not qualify or reject a point estimate."
            ),
            "matched_cohort": (
                "Pair first and second estimates only within the same saved sample scenario and zero-based "
                "realization index. Each spacing/condition group contains three barrier scenarios with twelve "
                "paired realizations each."
            ),
            "agreement_definition": "max(first, second) / min(first, second) <= factor threshold",
            "primary_agreement_factor": 1.20,
            "sensitivity_agreement_factors": [1.10, 1.30],
            "truth_miss_definition": "absolute relative error abs(estimate / known_truth - 1) > 0.20",
            "agreement_conditioned_miss_counts": (
                "either miss means at least one of the two order estimates is >20% from truth; both miss means "
                "each is >20% from truth. These counts are among pairs passing the agreement factor."
            ),
            "truth_source": "samples[sample_id].truth.barrier_meV_per_cell in the frozen recovery record",
            "sensitivity_limits": (
                "Counts describe this finite synthetic design and its saved nuisance realizations, not a probability "
                "for experimental scans or a calibrated real-data uncertainty."
            ),
        },
        "input_counts": {
            "scenarios": len(data["samples"]),
            "spacing_condition_groups": len(groups),
            "model_families": len(MODELS),
            "realizations_per_scenario": REALIZATIONS_PER_SCENARIO,
            "synthetic_images": len(data["samples"]) * REALIZATIONS_PER_SCENARIO,
            "saved_fits_per_scenario": len(MODELS) * len(ORDERS) * 2 * REALIZATIONS_PER_SCENARIO,
            "fit_style_analyzed": FIT_STYLE,
        },
        "groups": groups,
        "case_example": example,
        "checks": {
            "scenario_count_18": True,
            "216_synthetic_images": True,
            "three_scenarios_per_spacing_condition": True,
            "periodic_variable_probe_28_6_primary_5_of_15": True,
            "sample_06_realization_4_truth_and_point_estimates_match": True,
            "all_group_counts_recomputed_from_saved_fits": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_bytes = args.input.read_bytes()
    data = json.loads(input_bytes)
    output = build_summary(data, input_bytes, Path(__file__).resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256_bytes(args.output.read_bytes()),
        "scenario_count": output["input_counts"]["scenarios"],
        "image_count": output["input_counts"]["synthetic_images"],
        "primary_5_of_15": output["checks"]["periodic_variable_probe_28_6_primary_5_of_15"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
