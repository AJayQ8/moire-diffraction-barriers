"""Export clean CSV result tables from the saved v021 evidence; no model is run."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
RESULTS = ROOT / "results"


def load(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def write_csv(name, columns, rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def export_headline_scores():
    matched = load("mos2_matched_wall_models_summary_v01.json")
    groups = matched["groups"]
    specs = (
        ("Closer spacing, six maps", "higher_twist"),
        ("Wider spacing, four maps", "lower_twist"),
        ("All ten maps", "all_maps"),
    )
    rows = []
    for label, key in specs:
        result = groups[key]["scan_grid"]
        errors = result["map_mean_gap_absolute_error_nm"]
        rows.append({
            "comparison": label,
            "maps": ";".join(result["scans"]),
            "matched_wall_segments": result["matched_sections"],
            "isolated_wall_mean_error_nm": errors["isolated"],
            "periodic_network_mean_error_nm": errors["network"],
        })
    write_csv(
        "matched_width_errors.csv",
        ("comparison", "maps", "matched_wall_segments",
         "isolated_wall_mean_error_nm", "periodic_network_mean_error_nm"),
        rows,
    )

    recovery = load("mos2_barrier_recovery_v01.json")
    rows = []
    for condition, summary in recovery["summary"].items():
        if not condition.startswith("1.85/"):
            continue
        for model in ("isolated", "periodic"):
            result = summary[f"{model}/both/profile_shapes"]
            rows.append({
                "nominal_spacing_nm": 8.5,
                "synthetic_condition": condition.split("/", 1)[1],
                "model": model,
                "median_absolute_relative_barrier_error_percent":
                    100 * result["median_absolute_relative_error"],
                "within_20_percent_fraction": result["fraction_within_20_percent"],
                "realizations": result["realizations"],
            })
    write_csv(
        "synthetic_barrier_recovery.csv",
        ("nominal_spacing_nm", "synthetic_condition", "model",
         "median_absolute_relative_barrier_error_percent",
         "within_20_percent_fraction", "realizations"),
        rows,
    )


def export_eight_map_sensitivity():
    record = load("mos2_eight_map_check_v019.json")
    rows = []
    for family, family_record in record["families"].items():
        for fit, fit_record in family_record["cameras"].items():
            for frame, result in fit_record["frames"].items():
                interval = result["reselected_minimum_error_p05_p95_nm"]
                barrier_interval = result["reselected_best_barrier_p05_p95_meV_per_cell"]
                rows.append({
                    "energy_family": family,
                    "intensity_fit": fit,
                    "registry_frame": frame,
                    "omitted_maps": ";".join(record["scope"]["omitted_maps"]),
                    "retained_maps": ";".join(record["scope"]["retained_maps_in_saved_rng_order"]),
                    "nominal_probe_fwhm_nm": record["scope"]["nominal_probe_fwhm_nm"],
                    "best_grid_barrier_mev_per_cell": float(result["best_grid_barrier_meV_per_cell"]),
                    "minimum_map_mean_width_error_nm": result["minimum_map_mean_width_error_nm"],
                    "reselected_minimum_error_p05_nm": interval[0],
                    "reselected_minimum_error_p95_nm": interval[1],
                    "reselected_best_barrier_p05_mev_per_cell": barrier_interval[0],
                    "reselected_best_barrier_p95_mev_per_cell": barrier_interval[1],
                    "archived_rotated_equal": record["archived_rotated_branch_check"]["verified"],
                })
    write_csv(
        "eight_map_saddle_sensitivity.csv",
        ("energy_family", "intensity_fit", "registry_frame", "omitted_maps",
         "retained_maps", "nominal_probe_fwhm_nm", "best_grid_barrier_mev_per_cell",
         "minimum_map_mean_width_error_nm", "reselected_minimum_error_p05_nm",
         "reselected_minimum_error_p95_nm", "reselected_best_barrier_p05_mev_per_cell",
         "reselected_best_barrier_p95_mev_per_cell", "archived_rotated_equal"),
        rows,
    )


def export_representative_measurements():
    cases = load("mos2_matched_wall_models_v01.json")
    rows = []
    for scan_name in ("DS10", "DS14"):
        scan = cases["scans"][scan_name]
        for index, wall in enumerate(scan["walls"]):
            measured = wall["measured"]
            isolated = wall["models"]["isolated"]["scan_grid"]
            network = wall["models"]["network"]["scan_grid"]
            if not all(measured["qualified"] + isolated["qualified"] + network["qualified"]):
                continue
            rows.append({
                "scan": scan_name,
                "wall_index_zero_based": index,
                "center_indices": "-".join(str(value) for value in wall["centers"]),
                "row_spacing_nm": wall["spacing_nm"],
                "measured_first_order_width_nm": measured["widths_nm"][0],
                "measured_second_order_width_nm": measured["widths_nm"][1],
                "isolated_first_order_width_nm": isolated["widths_nm"][0],
                "isolated_second_order_width_nm": isolated["widths_nm"][1],
                "network_first_order_width_nm": network["widths_nm"][0],
                "network_second_order_width_nm": network["widths_nm"][1],
            })
    write_csv(
        "representative_mos2_wall_widths.csv",
        ("scan", "wall_index_zero_based", "center_indices", "row_spacing_nm",
         "measured_first_order_width_nm", "measured_second_order_width_nm",
         "isolated_first_order_width_nm", "isolated_second_order_width_nm",
         "network_first_order_width_nm", "network_second_order_width_nm"),
        rows,
    )


def main():
    export_headline_scores()
    export_eight_map_sensitivity()
    export_representative_measurements()
    print(f"Wrote saved-score CSVs to {RESULTS}")


if __name__ == "__main__":
    main()
