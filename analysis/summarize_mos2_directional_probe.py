"""Select blur on first-order data and score directional width changes in the second order."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from summarize_mos2_geometry_probe import block_draws, interval


CAMERAS = ("transferred_even", "separate_disjoint")
SOURCES = ("DS13", "DS14")


def centered_rms(values, draw=None):
    if draw is None:
        return float(np.sqrt(np.mean((values - values.mean(axis=0))**2)))
    means = draw @ values
    variance = np.maximum(draw @ (values**2) - means**2, 0.)
    return np.sqrt(np.mean(variance, axis=1))


def scores_for_order(walls, camera, probes, order, rng):
    valid = []
    for i, wall in enumerate(walls):
        fits = [*wall["measured"], *(f for v in wall["models"][camera].values() for f in v["widths"])]
        if all(fit["qualified"][order] for fit in fits):
            valid.append(i)
    if not valid:
        raise ValueError(f"No sections qualified across the optical controls in order {order+1}")
    draws, support = block_draws(walls, valid, rng)
    measured = np.array([[fit["widths_nm"][order] for fit in walls[i]["measured"]] for i in valid])
    prediction, metrics, resamples = {}, {}, {}
    for name in probes:
        predicted = np.array([[fit["widths_nm"][order] for fit in walls[i]["models"][camera][name]["widths"]]
                              for i in valid])
        prediction[name] = predicted
        residual = measured - predicted
        metrics[name] = {"centered_error_nm": centered_rms(residual),
                         "mean_absolute_error_nm": float(np.mean(np.abs(residual))),
                         "mean_measured_minus_predicted_nm": float(residual.mean())}
        resamples[name] = centered_rms(residual, draws)
    spread = {name: centered_rms(prediction[name] - prediction[case["equal_trace_circle"]])
              for name, case in probes.items() if case["kind"] == "elliptical"}
    return {"qualified_walls": len(valid), "wall_centers": [walls[i]["centers"] for i in valid],
            "spatial_blocks": support, "probe_width_scores": metrics, "width_draws": resamples,
            "directional_width_effect_vs_circle_nm": spread}


def scan_statistics(walls, camera, probes, rng):
    sample, _ = block_draws(walls, range(len(walls)), rng)
    profile = {
        name: np.array([wall["models"][camera][name]["first_order_registration"]["error_by_ring"]
                        for wall in walls]) for name in probes
    }
    return {
        "profile_walls": len(walls), "profile_scores": {k: v.mean(axis=0).tolist() for k, v in profile.items()},
        "profile_draws": {k: sample @ v for k, v in profile.items()},
        "width_by_order": [scores_for_order(walls, camera, probes, j, rng) for j in (0, 1)],
    }


def test_transfer(target, ellipse, circle):
    row = target["width_by_order"][1]
    return {
        "ellipse": ellipse, "equal_trace_circle": circle, "width_walls": row["qualified_walls"],
        "profile_walls": target["profile_walls"],
        "second_order": {name: {**row["probe_width_scores"][name],
                                "profile_error": target["profile_scores"][name][1]}
                         for name in ("nominal", ellipse, circle)},
        "paired_ranges_ellipse_minus_control": {
            name: {
                "centered_width_error_nm": interval(row["width_draws"][ellipse] - row["width_draws"][name]),
                "profile_error": interval(target["profile_draws"][ellipse][:, 1] - target["profile_draws"][name][:, 1]),
            } for name in ("nominal", circle)
        },
    }


def summarize(raw):
    if set(raw["scans"]) != set(raw["requested_scans"]):
        raise ValueError("The probe comparison is incomplete")
    rng = np.random.default_rng(20261118)
    probes = raw["probes"]
    ellipses = [name for name, case in probes.items() if case["kind"] == "elliptical"]
    answers = {}
    for camera in CAMERAS:
        stats = {s: scan_statistics(raw["scans"][s], camera, probes, rng) for s in raw["requested_scans"]}
        configurations = {}
        for source in SOURCES:
            widths = stats[source]["width_by_order"][0]["probe_width_scores"]
            profiles = stats[source]["profile_scores"]
            choices = {
                "first_order_centered_width": min(ellipses, key=lambda name: widths[name]["centered_error_nm"]),
                "first_order_profile": min(ellipses, key=lambda name: profiles[name][0]),
            }
            configurations[source] = {
                metric: {target: test_transfer(value, name, probes[name]["equal_trace_circle"])
                         for target, value in stats.items()}
                for metric, name in choices.items()
            }
        answers[camera] = {
            "first_order_selected_ellipses": configurations,
            "scans": {scan: {
                "profile_walls": configuration["profile_walls"],
                "profile_scores": configuration["profile_scores"],
                "width_by_order": [{k: v for k, v in row.items() if k != "width_draws"}
                                   for row in configuration["width_by_order"]],
            } for scan, configuration in stats.items()},
        }
    return {"probes": probes, "cameras": answers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_scores", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.case_scores.read_bytes()
    result = summarize(json.loads(source))
    result["input_sha256"] = sha256(source).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
