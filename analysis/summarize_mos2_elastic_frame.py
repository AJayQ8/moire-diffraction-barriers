"""Summarize the 30-degree frame control using matched walls and pixels."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from summarize_mos2_geometry_probe import block_draws, error_stats, interval, qualified


def wall_summary(walls, fits, baseline_name, rng):
    names = tuple(fits[0])
    selected = [i for i, (wall, result) in enumerate(zip(walls, fits, strict=True))
                if qualified(wall["measured"], *(result[name] for name in names))]
    observed = np.array([wall["measured"]["widths_nm"] for wall in walls])
    profile_draw, _ = block_draws(walls, range(len(walls)), rng)
    profile_mean = {
        name: np.mean([estimate[name]["transfer_error"] for estimate in fits], axis=1)
        for name in names
    }
    scores = {}
    for name, values in profile_mean.items():
        delta = values - profile_mean[baseline_name]
        scores[name] = {"mean": float(np.mean(values)), "delta_vs_nominal": float(np.mean(delta)),
                        "delta_p05_p95": interval(profile_draw @ delta)}
    output = {"shape_sections": len(walls), "qualified_common_width_sections": len(selected),
              "profile_error": scores}
    if not selected:
        return output
    draw, support = block_draws(walls, selected, rng)
    actual = observed[selected]
    means = actual.mean(axis=0)
    sample_means = draw @ actual
    output.update(width_block_support=support, widths_nm={"measured": means.tolist()})
    predictions = {name: np.array([fits[i][name]["widths_nm"] for i in selected]) for name in names}
    baseline = error_stats(sample_means, draw @ predictions[baseline_name])
    for name, prediction in predictions.items():
        avg = prediction.mean(axis=0)
        delta = avg - means
        output["widths_nm"][name] = {
            "pair": avg.tolist(), "predicted_minus_measured": delta.tolist(),
            "mean_absolute_error": float(error_stats(means, avg)),
            "mean_absolute_per_wall_error": float(np.mean(np.abs(prediction - actual))),
            "delta_error_vs_nominal_p05_p95": interval(error_stats(sample_means, draw @ prediction) - baseline),
        }
    if "alternate_rotate_plus_30" in predictions:
        output["reindexing_max_abs_difference_nm"] = float(
            np.max(abs(predictions["rotate_minus_30"] - predictions["alternate_rotate_plus_30"]))
        )
    return output


def pixel_summary(rows):
    if not rows:
        return {"folds": 0, "rmse": {}}
    nominal = np.array([fold["rmse"]["nominal"] for fold in rows])
    summary = {}
    for name in rows[0]["rmse"]:
        values = np.array([fold["rmse"][name] for fold in rows])
        summary[name] = {"mean": float(np.mean(values)), "improved_folds": int(np.sum(values < nominal)),
                         "relative_change_percent": float(100 * (values.mean() / nominal.mean() - 1))}
    return {"folds": len(rows), "rmse": summary}


def summarize(source, saddle):
    if set(source["scans"]) != set(source["requested_scans"]):
        raise ValueError("Expected all target and control maps")
    rng = np.random.default_rng(20261116)
    result = {}
    for scan in source["requested_scans"]:
        row = source["scans"][scan]
        walls = row["walls"]
        reports = {}
        for camera in ("transferred_even", "separate_disjoint"):
            fits = [wall["arms"][camera] for wall in walls]
            if saddle:
                reports[camera] = {
                    frame: wall_summary(walls, [fit[frame] for fit in fits], "13.2", rng)
                    for frame in fits[0]
                }
            else:
                reports[camera] = wall_summary(walls, fits, "nominal", rng)
        result[scan] = {"solvers": row["solvers"], "wall_scores": reports}
        if saddle:
            result[scan]["landscape"] = row["landscape"]
        else:
            result[scan]["held_pixels"] = pixel_summary(row["held_second_order"])
            result[scan]["local_replay"] = pixel_summary(row["earlier_local_second_order"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--saddle", action="store_true")
    args = parser.parse_args()
    payload = args.input.read_bytes()
    result = {"input_sha256": sha256(payload).hexdigest(), "scans": summarize(json.loads(payload), args.saddle)}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
