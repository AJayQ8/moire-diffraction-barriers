"""Summarize the fixed MoS2 wall comparison from its saved measured and predicted fits."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from summarize_mos2_geometry_probe import GEOMETRY_SEED, block_draws, interval, qualified


MODELS = ("network", "isolated")
WIDTHS = ("measured", *MODELS)
SAMPLING = ("scan_grid", "direct_at_same_settings", "earlier_adaptive_window")


def width_summary(walls, take, widths, rng):
    mean_widths = widths.mean(axis=0)
    gaps = widths[..., 0] - widths[..., 1]
    mean_gaps = gaps.mean(axis=0)
    error = abs(mean_gaps[1:] - mean_gaps[0])
    draw, blocks = block_draws(walls, take, rng)
    paired = draw @ (gaps[:, 1:] - gaps[:, :1])
    advantage = abs(paired[:, 0]) - abs(paired[:, 1])
    return {
        "matched_sections": len(take),
        "mean_widths_nm": dict(zip(WIDTHS, mean_widths.tolist(), strict=True)),
        "mean_first_minus_second_nm": dict(zip(WIDTHS, mean_gaps.tolist(), strict=True)),
        "map_mean_gap_absolute_error_nm": dict(zip(MODELS, error.tolist(), strict=True)),
        "mean_absolute_error_of_the_two_map_mean_widths_nm": dict(zip(
            MODELS, abs(mean_widths[1:] - mean_widths[0]).mean(axis=1).tolist(), strict=True)),
        "mean_absolute_wall_gap_error_nm": dict(zip(
            MODELS, abs(gaps[:, 1:] - gaps[:, :1]).mean(axis=0).tolist(), strict=True)),
        "network_minus_isolated_error_p05_p95_nm": interval(advantage),
        "spatial_blocks": blocks,
    }, advantage


def summarize_map(scan, rng):
    walls = scan["walls"]
    receipt = {
        "candidate_sections": len(walls),
        "twist_deg": scan["twist_deg"],
        "models": {},
        "profile_error_on_all_candidate_sections": {
            model: float(np.mean([w["models"][model]["scan_grid"]["transfer_error"] for w in walls]))
            for model in MODELS
        },
    }
    takes = {
        "scan_grid": [i for i, w in enumerate(walls) if qualified(
            w["measured"], *(w["models"][m]["scan_grid"] for m in MODELS)
        )],
        "earlier_adaptive_window": list(range(len(walls))),
    }
    takes["direct_at_same_settings"] = [i for i in takes["scan_grid"] if qualified(
        *(walls[i]["models"][m]["direct"] for m in MODELS)
    )]
    receipt["excluded_from_scan_grid_comparison"] = [
        w["centers"] for i, w in enumerate(walls) if i not in takes["scan_grid"]
    ]
    draws = {}
    for sampling in SAMPLING:
        take = takes[sampling]
        if not take:
            continue
        if sampling == "earlier_adaptive_window":
            widths = [[walls[i]["legacy_widths_nm"][name] for name in WIDTHS] for i in take]
        else:
            key = "scan_grid" if sampling == "scan_grid" else "direct"
            widths = [[walls[i]["measured"]["widths_nm"],
                       *(walls[i]["models"][m][key]["widths_nm"] for m in MODELS)] for i in take]
        receipt["models"][sampling], draws[sampling] = width_summary(walls, take, np.asarray(widths), rng)
    return receipt, draws


def summarize_group(by_map, draws, names, sampling):
    used = [n for n in names if sampling in by_map[n]["models"]]
    if not used:
        return {"scans_without_matched_fits": names, "matched_sections": 0}
    rows = [by_map[n]["models"][sampling] for n in used]
    value = "map_mean_gap_absolute_error_nm"
    summary = {
        "scans": used,
        "scans_without_matched_fits": [n for n in names if n not in used],
        "matched_sections": sum(r["matched_sections"] for r in rows),
        "network_wins": sum(r[value]["network"] < r[value]["isolated"] for r in rows),
        "network_minus_isolated_error_p05_p95_nm": interval(
            np.mean([draws[n][sampling] for n in used], axis=0)),
    }
    for metric in (value, "mean_absolute_error_of_the_two_map_mean_widths_nm", "mean_absolute_wall_gap_error_nm"):
        summary[metric] = {m: float(np.mean([r[metric][m] for r in rows])) for m in MODELS}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = args.cases.read_bytes()
    report = json.loads(payload)
    names, primary = report["requested_scans"], report["primary_scans"]
    if set(report["scans"]) != set(names):
        raise ValueError("The model comparison has not finished on all requested scans")
    rng = np.random.default_rng(GEOMETRY_SEED)
    by_map, draws = {}, {}
    for name in names:
        by_map[name], draws[name] = summarize_map(report["scans"][name], rng)
    groups = {
        "higher_twist": primary,
        "lower_twist": [n for n in names if n not in primary],
        "all_maps": names,
    }
    summaries = {group: {sampling: summarize_group(by_map, draws, scans, sampling)
                         for sampling in SAMPLING} for group, scans in groups.items()}
    old = summaries["higher_twist"]["earlier_adaptive_window"]
    old_errors = old["map_mean_gap_absolute_error_nm"]
    if (old["matched_sections"], round(old_errors["network"], 3), round(old_errors["isolated"], 3)) != (168, .169, 1.735):
        raise ValueError("The saved earlier comparison does not reproduce")
    result = {
        "schema": "mos2-matched-scan-grid-wall-models-summary-v01",
        "case_sha256": sha256(payload).hexdigest(),
        "primary": report["primary"],
        "uncertainty": ("Fifth and ninety-fifth percentiles from paired spatial-block resampling within "
                        "the same maps, conditional on the probe, geometry, material and selected scans."),
        "groups": summaries,
        "maps": by_map,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
