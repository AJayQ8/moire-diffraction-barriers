#!/usr/bin/env python3
"""Reproduce map-level spacing and eight-map barrier-score checks from saved fits.

From the extracted paper root, run the relative command:

    python3 analysis/summarize_mos2_spacing_and_weighting.py

The script reads only archived JSON evidence. It writes one evidence JSON next
to the source records and performs reference-total, cohort, and score checks.
It launches no simulations and has no third-party Python dependencies.
"""

from __future__ import annotations

import json
import math
import statistics
from hashlib import sha256
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = PAPER_DIR / "evidence"
OUTPUT_PATH = EVIDENCE_DIR / "mos2_spacing_and_weighting_v01.json"

PRIMARY_MAPS = ("DS3", "DS4", "DS6", "DS9", "DS10", "DS12")
OMITTED_MAPS = {"DS13", "DS14"}
CAMERAS = ("transferred_even", "separate_disjoint")
FAMILIES = (
    ("fixed_AA_energy", "mos2_saddle_sweep_v01.json"),
    ("fixed_AA_curvature", "mos2_energy_family_v01.json"),
)
HIGHER_TWIST_MAPS = ("DS3", "DS4", "DS6", "DS9", "DS10", "DS12")
LOWER_TWIST_RETAINED_MAPS = ("DS1", "DS2")
TOL = 1e-12

INPUT_NAMES = (
    "mos2_matched_wall_models_v01.json",
    "mos2_matched_wall_models_summary_v01.json",
    "mos2_saddle_sweep_v01.json",
    "mos2_saddle_sweep_summary_v01.json",
    "mos2_energy_family_v01.json",
    "mos2_energy_family_summary_v01.json",
    "mos2_eight_map_check_v019.json",
)


def digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return math.fsum(values) / len(values)


def rms(values: list[float]) -> float:
    return math.sqrt(mean([value * value for value in values]))


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=TOL):
        raise AssertionError(f"{label}: {actual:.17g} != {expected:.17g}")


def has_two_qualified_orders(record: dict) -> bool:
    qualified = record.get("qualified", [])
    return len(qualified) == 2 and all(qualified)


def width_gap(widths: list[float]) -> float:
    if len(widths) != 2:
        raise ValueError(f"Expected two diffraction-order widths, got {widths!r}")
    return widths[0] - widths[1]


def read_inputs() -> tuple[dict[str, bytes], dict[str, dict]]:
    payloads: dict[str, bytes] = {}
    records: dict[str, dict] = {}
    for name in INPUT_NAMES:
        payload = (EVIDENCE_DIR / name).read_bytes()
        payloads[name] = payload
        records[name] = json.loads(payload.decode("utf-8"))
    return payloads, records


def primary_spacing_and_ols(records: dict[str, dict]) -> dict:
    cases = records["mos2_matched_wall_models_v01.json"]
    summary = records["mos2_matched_wall_models_summary_v01.json"]
    if summary["case_sha256"] != digest(
        (EVIDENCE_DIR / "mos2_matched_wall_models_v01.json").read_bytes()
    ):
        raise AssertionError("Matched-model summary does not reference these saved cases")
    if tuple(summary["groups"]["higher_twist"]["scan_grid"]["scans"]) != PRIMARY_MAPS:
        raise AssertionError("Saved close-group map order differs from the declared cohort")

    rows: list[dict] = []
    for scan in PRIMARY_MAPS:
        walls = cases["scans"][scan]["walls"]
        eligible = []
        for index, wall in enumerate(walls):
            models = wall["models"]
            if not has_two_qualified_orders(wall["measured"]):
                continue
            if not has_two_qualified_orders(models["network"]["scan_grid"]):
                continue
            if not has_two_qualified_orders(models["isolated"]["scan_grid"]):
                continue
            eligible.append((index, wall))

        expected_map = summary["maps"][scan]
        expected_count = expected_map["models"]["scan_grid"]["matched_sections"]
        if len(eligible) != expected_count:
            raise AssertionError(
                f"{scan}: common scan-grid cohort has {len(eligible)} walls, "
                f"reference has {expected_count}"
            )

        measured_gaps = [width_gap(wall["measured"]["widths_nm"]) for _, wall in eligible]
        network_gaps = [
            width_gap(wall["models"]["network"]["scan_grid"]["widths_nm"])
            for _, wall in eligible
        ]
        isolated_gaps = [
            width_gap(wall["models"]["isolated"]["scan_grid"]["widths_nm"])
            for _, wall in eligible
        ]
        map_values = {
            "measured": mean(measured_gaps),
            "network": mean(network_gaps),
            "isolated": mean(isolated_gaps),
        }
        reference_values = expected_map["models"]["scan_grid"]["mean_first_minus_second_nm"]
        for key in map_values:
            assert_close(map_values[key], reference_values[key], f"{scan} {key} map-mean gap")

        rows.append(
            {
                "map": scan,
                "eligible_sections": len(eligible),
                "median_wall_spacing_nm": statistics.median(
                    [wall["spacing_nm"] for _, wall in eligible]
                ),
                "mean_first_minus_second_width_nm": map_values,
                "map_mean_gap_absolute_error_nm": {
                    "network": abs(map_values["network"] - map_values["measured"]),
                    "isolated": abs(map_values["isolated"] - map_values["measured"]),
                },
                "eligible_source_indices_zero_based": [index for index, _ in eligible],
            }
        )

    total_sections = sum(row["eligible_sections"] for row in rows)
    if total_sections != summary["groups"]["higher_twist"]["scan_grid"]["matched_sections"]:
        raise AssertionError("Close-group section total differs from the saved reference")

    model_metrics = {}
    for model in ("network", "isolated"):
        errors = [row["map_mean_gap_absolute_error_nm"][model] for row in rows]
        model_metrics[model] = {
            "map_equal_mae_nm": mean(errors),
            "map_equal_rmse_nm": rms(errors),
            "per_map_absolute_errors_nm": {
                row["map"]: row["map_mean_gap_absolute_error_nm"][model] for row in rows
            },
        }
        expected = summary["groups"]["higher_twist"]["scan_grid"]
        assert_close(
            model_metrics[model]["map_equal_mae_nm"],
            expected["map_mean_gap_absolute_error_nm"][model],
            f"close-group {model} MAE",
        )

    five_map_gains = []
    for omitted in rows:
        retained = [row for row in rows if row["map"] != omitted["map"]]
        network_mae = mean(
            [row["map_mean_gap_absolute_error_nm"]["network"] for row in retained]
        )
        isolated_mae = mean(
            [row["map_mean_gap_absolute_error_nm"]["isolated"] for row in retained]
        )
        five_map_gains.append(
            {
                "omitted_map": omitted["map"],
                "retained_maps": [row["map"] for row in retained],
                "retained_map_count": len(retained),
                "network_map_equal_mae_nm": network_mae,
                "isolated_map_equal_mae_nm": isolated_mae,
                "network_mae_advantage_over_isolated_nm": isolated_mae - network_mae,
            }
        )
    five_map_gain_values = [
        row["network_mae_advantage_over_isolated_nm"] for row in five_map_gains
    ]

    loo_rows = []
    for held_out in rows:
        training = [row for row in rows if row["map"] != held_out["map"]]
        x = [row["median_wall_spacing_nm"] for row in training]
        y = [row["mean_first_minus_second_width_nm"]["measured"] for row in training]
        x_bar = mean(x)
        y_bar = mean(y)
        denominator = math.fsum((value - x_bar) ** 2 for value in x)
        if denominator == 0.0:
            raise AssertionError("OLS training spacings have zero variance")
        slope = math.fsum(
            (x_value - x_bar) * (y_value - y_bar) for x_value, y_value in zip(x, y)
        ) / denominator
        intercept = y_bar - slope * x_bar
        observed = held_out["mean_first_minus_second_width_nm"]["measured"]
        predicted = intercept + slope * held_out["median_wall_spacing_nm"]
        loo_rows.append(
            {
                "held_out_map": held_out["map"],
                "training_maps": [row["map"] for row in training],
                "training_map_count": len(training),
                "slope_nm_per_nm": slope,
                "intercept_nm": intercept,
                "median_wall_spacing_nm": held_out["median_wall_spacing_nm"],
                "observed_mean_gap_nm": observed,
                "predicted_mean_gap_nm": predicted,
                "prediction_minus_observed_nm": predicted - observed,
                "absolute_prediction_error_nm": abs(predicted - observed),
                "network_fixed_model_absolute_error_nm": held_out[
                    "map_mean_gap_absolute_error_nm"
                ]["network"],
                "network_beats_ols_on_this_held_out_map": (
                    held_out["map_mean_gap_absolute_error_nm"]["network"]
                    < abs(predicted - observed)
                ),
            }
        )

    loo_errors = [row["prediction_minus_observed_nm"] for row in loo_rows]
    loo_mae = mean([abs(value) for value in loo_errors])
    loo_rmse = rms(loo_errors)
    network_mae = model_metrics["network"]["map_equal_mae_nm"]
    isolated_mae = model_metrics["isolated"]["map_equal_mae_nm"]
    network_rmse = model_metrics["network"]["map_equal_rmse_nm"]
    isolated_rmse = model_metrics["isolated"]["map_equal_rmse_nm"]
    network_beats_ols = [
        row["held_out_map"]
        for row in loo_rows
        if row["network_beats_ols_on_this_held_out_map"]
    ]
    if len(network_beats_ols) != 5 or len(loo_rows) != 6:
        raise AssertionError("Expected the fixed network model to beat the LOO OLS line on 5/6 maps")
    five_map_gain_min = min(five_map_gain_values)
    five_map_gain_max = max(five_map_gain_values)
    if round(five_map_gain_min, 2) != 1.08 or round(five_map_gain_max, 2) != 1.35:
        raise AssertionError("Five-map gains no longer reproduce the SI 1.08--1.35 nm range")

    return {
        "cohort": {
            "maps": list(PRIMARY_MAPS),
            "eligible_sections": total_sections,
            "eligibility": (
                "A source wall is retained only when measured, periodic-network, and "
                "isolated-wall scan-grid fits qualify in both diffraction orders."
            ),
            "map_rows": rows,
        },
        "physical_model_errors": model_metrics,
        "network_mae_advantage_over_isolated_nm": isolated_mae - network_mae,
        "fixed_model_five_map_gains_after_dropping_one_map": {
            "definition": (
                "For each omitted map, recompute equal-map MAE for the fixed network and isolated "
                "predictions on the other five maps; gain is isolated MAE minus network MAE."
            ),
            "per_omitted_map": five_map_gains,
            "minimum_gain_nm": five_map_gain_min,
            "maximum_gain_nm": five_map_gain_max,
        },
        "leave_one_map_out_ols": {
            "predictor": "Median retained wall-row spacing per map (nm)",
            "response": "Map-mean measured first-order minus second-order fitted width (nm)",
            "model": "Ordinary least squares with one linear predictor and an intercept.",
            "held_out_prediction_error_convention": "Predicted minus observed map-mean gap.",
            "map_equal_mae_nm": loo_mae,
            "map_equal_rmse_nm": loo_rmse,
            "network_fixed_model_advantage_over_ols_loo_mae_nm": loo_mae - network_mae,
            "network_fixed_model_advantage_over_ols_loo_rmse_nm": loo_rmse - network_rmse,
            "isolated_fixed_model_advantage_over_ols_loo_mae_nm": loo_mae - isolated_mae,
            "held_out_maps_where_network_beats_ols_count": len(network_beats_ols),
            "held_out_map_count": len(loo_rows),
            "held_out_maps_where_network_beats_ols": network_beats_ols,
            "per_map_predictions": loo_rows,
            "comparison_note": (
                "The OLS errors are held-out by map. Network and isolated scores are fixed-physics "
                "map-mean errors on the same six maps; those physical predictions are not refit to "
                "the measured map-mean width gaps. A positive advantage means lower error than OLS."
            ),
        },
        "reference_checks": {
            "matched_section_total_matches_saved_summary": True,
            "per_map_measured_and_model_gap_means_match_saved_summary": True,
            "network_map_equal_mae_matches_saved_summary": True,
            "isolated_map_equal_mae_matches_saved_summary": True,
        },
    }


def two_family_common_wall_indices(
    scan: str,
    energy_scan: dict,
    curvature_scan: dict,
    settings: list[str],
) -> list[int]:
    energy_walls = energy_scan["walls"]
    curvature_walls = curvature_scan["walls"]
    if len(energy_walls) != len(curvature_walls):
        raise AssertionError(f"{scan}: energy-family source wall counts differ")

    eligible_indices = []
    for index, (energy_wall, curvature_wall) in enumerate(zip(energy_walls, curvature_walls)):
        for key in ("centers", "spacing_nm"):
            if energy_wall[key] != curvature_wall[key]:
                raise AssertionError(f"{scan} wall {index}: family source geometry differs in {key}")
        if energy_wall["measured"]["widths_nm"] != curvature_wall["measured"]["widths_nm"]:
            raise AssertionError(f"{scan} wall {index}: measured widths differ between families")
        if energy_wall["measured"]["qualified"] != curvature_wall["measured"]["qualified"]:
            raise AssertionError(f"{scan} wall {index}: measured qualification differs between families")
        if not has_two_qualified_orders(energy_wall["measured"]):
            continue

        all_fits_qualify = True
        for wall in (energy_wall, curvature_wall):
            for camera in CAMERAS:
                arm = wall["arms"][camera]["nominal"]
                for setting in settings:
                    if setting not in arm or not has_two_qualified_orders(arm[setting]):
                        all_fits_qualify = False
                        break
                if not all_fits_qualify:
                    break
            if not all_fits_qualify:
                break
        if all_fits_qualify:
            eligible_indices.append(index)
    return eligible_indices


def map_curve_for_family_camera(
    walls: list[dict], indices: list[int], settings: list[str], camera: str
) -> list[float]:
    scores = []
    for setting in settings:
        order_mean_residuals = [[], []]
        for index in indices:
            wall = walls[index]
            prediction = wall["arms"][camera]["nominal"][setting]
            measured_widths = wall["measured"]["widths_nm"]
            predicted_widths = prediction["widths_nm"]
            for order in (0, 1):
                order_mean_residuals[order].append(
                    predicted_widths[order] - measured_widths[order]
                )
        per_order_absolute_map_errors = [
            abs(mean(residuals)) for residuals in order_mean_residuals
        ]
        scores.append(mean(per_order_absolute_map_errors))
    return scores


def aggregate_curve(
    maps: tuple[str, ...],
    map_curves: dict[str, list[float]],
    counts: dict[str, int],
    settings: list[str],
) -> dict:
    equal_map_curve = []
    section_weighted_curve = []
    total_sections = sum(counts[scan] for scan in maps)
    for position, _setting in enumerate(settings):
        equal_map_curve.append(mean([map_curves[scan][position] for scan in maps]))
        section_weighted_curve.append(
            math.fsum(counts[scan] * map_curves[scan][position] for scan in maps)
            / total_sections
        )

    def summarize(curve: list[float]) -> dict:
        best_index = min(range(len(curve)), key=curve.__getitem__)
        return {
            "score_curve_nm": curve,
            "best_sampled_barrier_meV_per_cell": float(settings[best_index]),
            "minimum_score_nm": curve[best_index],
        }

    return {
        "maps": list(maps),
        "eligible_sections": total_sections,
        "map_equal": summarize(equal_map_curve),
        "eligible_section_weighted": summarize(section_weighted_curve),
    }


def eight_map_results(records: dict[str, dict], payloads: dict[str, bytes]) -> dict:
    reference = records["mos2_eight_map_check_v019.json"]
    sweep = records["mos2_saddle_sweep_v01.json"]
    curvature = records["mos2_energy_family_v01.json"]
    sweep_summary = records["mos2_saddle_sweep_summary_v01.json"]
    curvature_summary = records["mos2_energy_family_summary_v01.json"]

    if sweep["settings_meV_per_cell"] != curvature["settings_meV_per_cell"]:
        raise AssertionError("Fixed-energy and fixed-curvature barrier grids differ")
    if sweep["requested_scans"] != curvature["requested_scans"]:
        raise AssertionError("Fixed-energy and fixed-curvature map lists differ")
    if sweep["half_window_nm"] != curvature["half_window_nm"]:
        raise AssertionError("Fixed-energy and fixed-curvature windows differ")
    if sweep["probe_fwhm_nm"] != curvature["probe_fwhm_nm"]:
        raise AssertionError("Fixed-energy and fixed-curvature probes differ")
    if curvature["input_sha256"]["mos2_saddle_sweep_v01.json"] != digest(
        payloads["mos2_saddle_sweep_v01.json"]
    ):
        raise AssertionError("Curvature record does not reference this exact sweep input")

    for source_name in (
        "mos2_saddle_sweep_v01.json",
        "mos2_energy_family_v01.json",
        "mos2_saddle_sweep_summary_v01.json",
        "mos2_energy_family_summary_v01.json",
    ):
        expected_hash = reference["inputs_sha256"].get(source_name)
        if expected_hash != digest(payloads[source_name]):
            raise AssertionError(f"Eight-map reference does not hash-match {source_name}")

    settings = sweep["settings_meV_per_cell"]
    retained_maps = tuple(reference["scope"]["retained_maps_in_saved_rng_order"])
    if set(retained_maps) != set(sweep["requested_scans"]) - OMITTED_MAPS:
        raise AssertionError("The saved eight-map selection differs from the declared omission")

    common_indices: dict[str, list[int]] = {}
    counts: dict[str, int] = {}
    family_raw: dict[str, dict] = {}
    for family, source_name in FAMILIES:
        family_raw[family] = records[source_name]

    for scan in retained_maps:
        indices = two_family_common_wall_indices(
            scan,
            sweep["scans"][scan],
            curvature["scans"][scan],
            settings,
        )
        expected_coverage = reference["map_coverage"][scan]
        if indices != expected_coverage["common_source_indices_zero_based"]:
            raise AssertionError(f"{scan}: reconstructed common source IDs differ from v019")
        if len(sweep["scans"][scan]["walls"]) != expected_coverage["source_sections"]:
            raise AssertionError(f"{scan}: source wall total differs from v019")
        if len(indices) != expected_coverage["common_sections_across_both_families_and_all_fits"]:
            raise AssertionError(f"{scan}: common wall count differs from v019")
        common_indices[scan] = indices
        counts[scan] = len(indices)

    if sum(counts.values()) != 175:
        raise AssertionError(f"Eight-map matched total is {sum(counts.values())}, expected 175")

    groups = {
        "all_eight_maps": retained_maps,
        "six_higher_twist_maps": tuple(scan for scan in HIGHER_TWIST_MAPS if scan in retained_maps),
        "remaining_two_lower_twist_maps": tuple(
            scan for scan in LOWER_TWIST_RETAINED_MAPS if scan in retained_maps
        ),
    }
    if len(groups["six_higher_twist_maps"]) != 6 or len(groups["remaining_two_lower_twist_maps"]) != 2:
        raise AssertionError("Eight-map high/low subgroup sizes differ from the stated design")

    computed_curves: dict[str, dict[str, dict[str, list[float]]]] = {}
    family_results: dict[str, dict] = {}
    reference_curve_checks = {}
    max_reference_delta = 0.0

    for family, source_name in FAMILIES:
        computed_curves[family] = {}
        family_results[family] = {"intensity_fits": {}}
        for camera in CAMERAS:
            per_map_curves = {}
            for scan in retained_maps:
                raw_scan = family_raw[family]["scans"][scan]
                per_map_curves[scan] = map_curve_for_family_camera(
                    raw_scan["walls"], common_indices[scan], settings, camera
                )
            computed_curves[family][camera] = per_map_curves

            group_results = {}
            for group_name, group_maps in groups.items():
                group_results[group_name] = aggregate_curve(
                    group_maps, per_map_curves, counts, settings
                )
            family_results[family]["intensity_fits"][camera] = {
                "per_map_error_curves_nm": per_map_curves,
                "groups": group_results,
            }

            ref_frame_curves = reference["families"][family]["cameras"][camera]["frames"]
            ref_group_curve = group_results["all_eight_maps"]["map_equal"]["score_curve_nm"]
            for frame in ("archived", "rotate_ds13_ds14"):
                expected_curve = ref_frame_curves[frame]["map_mean_width_error_curve_nm"]
                if len(expected_curve) != len(settings):
                    raise AssertionError(f"{family}/{camera}/{frame}: reference curve length differs")
                differences = [abs(a - b) for a, b in zip(ref_group_curve, expected_curve)]
                frame_max = max(differences)
                max_reference_delta = max(max_reference_delta, frame_max)
                if frame_max > TOL:
                    raise AssertionError(
                        f"{family}/{camera}/{frame}: map-equal curve differs by {frame_max:.3g}"
                    )
                reference_curve_checks[f"{family}/{camera}/{frame}"] = {
                    "maximum_absolute_difference_nm": frame_max,
                    "matches_saved_v019_curve": True,
                }

    expected_totals = reference["families"]["fixed_AA_energy"]["matched_sections"]
    if expected_totals != 175:
        raise AssertionError(f"v019 fixed-energy section total is {expected_totals}, not 175")

    branch_check = reference.get("archived_rotated_branch_check", {})
    if not branch_check.get("verified") or branch_check.get("maximum_absolute_group_score_difference_nm") != 0.0:
        raise AssertionError("Saved v019 archived/rotated retained-branch verification failed")

    return {
        "cohort": {
            "omitted_maps": sorted(OMITTED_MAPS),
            "maps_in_reference_order": list(retained_maps),
            "eligible_sections_total": sum(counts.values()),
            "eligible_sections_by_map": counts,
            "common_source_indices_zero_based_by_map": common_indices,
            "subgroups": {
                "six_higher_twist_maps": {
                    "maps": list(groups["six_higher_twist_maps"]),
                    "eligible_sections": sum(counts[scan] for scan in groups["six_higher_twist_maps"]),
                },
                "remaining_two_lower_twist_maps": {
                    "maps": list(groups["remaining_two_lower_twist_maps"]),
                    "eligible_sections": sum(counts[scan] for scan in groups["remaining_two_lower_twist_maps"]),
                },
            },
            "nominal_probe_fwhm_nm": sweep["probe_fwhm_nm"],
            "half_window_nm": sweep["half_window_nm"],
            "barriers_meV_per_cell": [float(setting) for setting in settings],
        },
        "score_definition": {
            "intensity_fits": {
                "transferred_even": "Primary coefficients transferred from the four other maps.",
                "separate_disjoint": "Target-map coefficients fitted with wall-cut pixels and buffer excluded.",
            },
            "map_error": (
                "For each map, barrier, family and intensity fit, calculate predicted-minus-measured "
                "width residuals over the same common sections. Average signed residuals separately "
                "by diffraction order; take the absolute value of each order mean; average the two "
                "order errors."
            ),
            "equal_map": "Arithmetic mean of the per-map absolute map-mean errors; every map has equal weight.",
            "eligible_section_weighted": (
                "Sum each per-map absolute map-mean error multiplied by that map's common eligible-section "
                "count, divided by the total eligible-section count. This changes only the weights on "
                "within-map absolute errors. It is not the per-wall absolute-error average, not a new fit, "
                "and not an estimate of between-map or device uncertainty."
            ),
        },
        "families": family_results,
        "reference_checks": {
            "common_source_ids_match_v019_by_map": True,
            "175_section_total_matches_v019": True,
            "fixed_energy_and_curvature_equal_map_curves_match_v019": True,
            "maximum_equal_map_curve_difference_nm": max_reference_delta,
            "equal_map_curve_checks": reference_curve_checks,
            "retained_archived_rotated_branches_equal_per_v019": {
                "verified": True,
                "max_group_score_difference_nm": 0.0,
                "basis": branch_check.get("basis"),
                "retained_fit_width_arrays_compared": branch_check.get(
                    "exact_fit_width_arrays_compared_across_both_families_cameras_and_all_grid_barriers"
                ),
            },
            "fixed_energy_raw_input_hash_matches_fixed_curvature_record": True,
            "reference_summary_hashes_match": True,
        },
    }


def main() -> None:
    payloads, records = read_inputs()
    spacing = primary_spacing_and_ols(records)
    eight = eight_map_results(records, payloads)

    result = {
        "schema": "mos2-spacing-and-weighting-v023",
        "analysis": "Saved-score aggregation only; no simulations or physical calculations were run.",
        "reproduction_command": (
            "From the extracted paper root: python3 analysis/summarize_mos2_spacing_and_weighting.py"
        ),
        "analysis_sha256": digest(Path(__file__).read_bytes()),
        "inputs_sha256": {name: digest(payloads[name]) for name in sorted(payloads)},
        "primary_close_group_spacing_and_ols": spacing,
        "eight_map_fixed_probe_weighting": eight,
        "limitation": (
            "All results condition on the saved scans, material records, probe settings, common fit-qualified "
            "walls and specified aggregation. The OLS diagnostic has six map-level observations. Spatial "
            "sections are not independent maps or devices."
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT_PATH.relative_to(PAPER_DIR)}")
    print("Checks passed: close-group reference totals; OLS inputs; eight-map IDs/counts; all equal-map curves.")


if __name__ == "__main__":
    main()
