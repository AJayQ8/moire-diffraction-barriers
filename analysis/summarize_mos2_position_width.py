"""Compare position-corrected wall widths on paired sections and spatial blocks."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from summarize_mos2_directional_probe import CAMERAS, centered_rms
from summarize_mos2_geometry_probe import block_draws, interval


def width_metrics(walls, camera, cases, order, rng, cohort):
    def qualified(wall, name):
        fits = [*wall["measured"], *wall["models"][camera][name]["widths"]]
        return all(fit["qualified"][order] for fit in fits)

    n_qualified = {name: sum(qualified(walls[i], name) for i in cohort) for name in cases}
    retained = [i for i in cohort if all(qualified(walls[i], name) for name in cases)]
    if len(retained) < 4:
        return {"candidate_counts": n_qualified, "matched_walls": len(retained)}
    draws, blocks = block_draws(walls, retained, rng)
    observed = np.array([[half["widths_nm"][order] for half in walls[i]["measured"]] for i in retained])
    predictions = {
        name: np.array([[half["widths_nm"][order] for half in walls[i]["models"][camera][name]["widths"]]
                        for i in retained]) for name in cases
    }
    results = {}
    ref = observed - predictions["nominal"]
    absolute_ref = abs(ref).mean(axis=1)
    centered_ref = centered_rms(ref, draws)
    for name, predicted in predictions.items():
        residual = observed - predicted
        absolute = abs(residual).mean(axis=1)
        results[name] = {
            "mean_absolute_error_nm": float(absolute.mean()),
            "centered_error_nm": centered_rms(residual),
            "mean_measured_minus_predicted_nm": float(residual.mean()),
            "paired_absolute_error_change_nm": interval(draws @ (absolute - absolute_ref)),
            "paired_centered_error_change_nm": interval(centered_rms(residual, draws) - centered_ref),
            "width_change_vs_nominal_rms_nm": float(np.sqrt(np.mean((predicted - predictions["nominal"])**2))),
        }
    return {"candidate_counts": n_qualified, "matched_walls": len(retained),
            "wall_centers": [walls[i]["centers"] for i in retained], "spatial_blocks": blocks,
            "opposite_spot_mean_absolute_difference_nm": float(abs(observed[:, 0] - observed[:, 1]).mean()),
            "constant_width_centered_error_nm": centered_rms(observed), "results": results}


def summarize(raw):
    if set(raw["scans"]) != set(raw["requested_scans"]):
        raise ValueError("Incomplete position-control result")
    rng = np.random.default_rng(20261118)
    cases = raw["cases"]
    scans = {}
    for scan in raw["requested_scans"]:
        data = raw["scans"][scan]
        walls = data["walls"]
        jointly_supported = [i for i, wall in enumerate(walls) if wall["within_fit_hull"]]
        draws, blocks = block_draws(walls, range(len(walls)), rng)
        scans[scan] = {"wall_count": len(walls), "within_geometry_support": len(jointly_supported),
                       "domain_intensity_fits": data["warp_fits"], "detectors": {}}
        for camera in CAMERAS:
            profiles = {
                name: np.array([wall["models"][camera][name]["registration"]["error_by_ring"][1] for wall in walls])
                for name in cases
            }
            scans[scan]["detectors"][camera] = {
                "widths": [width_metrics(walls, camera, cases, order, rng, range(len(walls))) for order in (0, 1)],
                "second_order_widths_inside_geometry_support": width_metrics(walls, camera, cases, 1, rng,
                                                                           jointly_supported),
                "second_order_profile_spatial_blocks": blocks,
                "second_order_profiles": {
                    name: {"error": float(y.mean()), "paired_change": interval(draws @ (y - profiles["nominal"]))}
                    for name, y in profiles.items()
                },
            }
    return {"scans": scans, "cases": cases}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cases", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    contents = args.cases.read_bytes()
    result = summarize(json.loads(contents))
    result["input_sha256"] = sha256(contents).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
