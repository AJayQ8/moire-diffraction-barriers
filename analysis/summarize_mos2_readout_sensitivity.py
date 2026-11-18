"""Summarize matched reflection and optical sensitivity on the fixed MoS2 maps."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from summarize_mos2_geometry_probe import CAMERAS, FIXED, block_draws, interval, qualified


NOMINAL = "1.25"
TRANSFERRED = "transferred_even"


def array_of_widths(walls, selector):
    return np.asarray([selector(wall)["widths_nm"] for wall in walls])


def width_errors(observed, predicted, draws):
    mean_errors = draws @ (predicted - observed)
    return {"width_error_nm": abs(mean_errors).mean(axis=1),
            "gap_error_nm": abs(mean_errors[:, 0] - mean_errors[:, 1])}


def paired_draws(walls, retained, rng):
    draws, support = block_draws(walls, retained, rng)
    return np.vstack((np.full((1, len(retained)), 1 / len(retained)), draws)), support


def cutoff_fits(row):
    return [row["measured"], *(fits[model] for fits in row["models"].values() for model in FIXED)]


def selection_results(raw, rng):
    cutoffs = list(map(str, raw["cutoffs_percent"]))
    probes = [f"{probe:g}" for probe in raw["gaussian_fwhm_nm"]]
    scans, totals = {}, {}
    for scan in raw["requested_scans"]:
        walls = raw["scans"][scan]["walls"]
        eligible = {(cut, p): [i for i, w in enumerate(walls) if qualified(*cutoff_fits(w["cutoff"][cut][p]))]
                    for cut in cutoffs for p in probes}
        retained = sorted(set.intersection(*(set(row) for row in eligible.values())))
        if not retained:
            raise ValueError(f"{scan} has no matched widths for the cutoff comparison")
        chosen = [walls[i] for i in retained]
        draws, support = paired_draws(walls, retained, rng)
        totals[scan] = {}
        scans[scan] = {"profile_walls": len(walls), "width_walls": len(chosen), "spatial_blocks": support,
                       "matched_centers": [wall["centers"] for wall in chosen], "cutoffs": {}}
        for cut in cutoffs:
            channels = [sum(w["miller_indices"][cut][j] != w["miller_indices"]["30"][j] for w in walls)
                        for j in (0, 1)]
            report = {"wall_lists_changed_by_order": channels,
                      "width_counts_at_each_probe_before_matching": {p: len(eligible[cut, p]) for p in probes},
                      "detectors": {}}
            totals[scan][cut] = {}
            for camera in CAMERAS:
                detector = {model: float(np.mean([w["cutoff"][cut][NOMINAL]["models"][camera][model]["transfer_error"]
                                                   for w in walls])) for model in ("anchor", "rigid")}
                width = {}
                for p in probes:
                    obs = array_of_widths(chosen, lambda w: w["cutoff"][cut][p]["measured"])
                    candidates = {model: width_errors(obs, array_of_widths(
                        chosen, lambda w: w["cutoff"][cut][p]["models"][camera][model]), draws) for model in FIXED}
                    width[p] = candidates
                totals[scan][cut][camera] = (detector, width)
                report["detectors"][camera] = {"profile_error": detector, "width_errors": {
                    p: {name: {metric: float(value[0]) for metric, value in fits.items()}
                        for name, fits in part.items()} for p, part in width.items()}}
            scans[scan]["cutoffs"][cut] = report
    groups = {}
    low = set(raw["lower_twist_scans"])
    for group, names in (("all", list(scans)), ("low", [name for name in scans if name in low]),
                         ("high", [name for name in scans if name not in low])):
        result = {}
        for cut in cutoffs:
            result[cut] = {}
            for camera in CAMERAS:
                scores = [totals[name][cut][camera] for name in names]
                profile = {model: float(np.mean([row[0][model] for row in scores])) for model in ("anchor", "rigid")}
                widths = {}
                for p in probes:
                    errors = {
                        model: {metric: np.mean([row[1][p][model][metric] for row in scores], axis=0)
                                for metric in ("width_error_nm", "gap_error_nm")} for model in FIXED}
                    contrast = errors["raman_shape"]["width_error_nm"] - errors["raman_scale"]["width_error_nm"]
                    widths[p] = {"models": {name: {metric: float(v[0]) for metric, v in row.items()}
                                            for name, row in errors.items()},
                                 "raman_error_difference_nm": float(contrast[0]),
                                 "raman_paired_range_nm": interval(contrast[1:])}
                result[cut][camera] = {"profile_error": profile,
                                       "relaxed_profile_wins": sum(a["anchor"] < a["rigid"] for a, _ in scores),
                                       "widths": widths}
        groups[group] = {"scans": names, "width_walls": sum(scans[n]["width_walls"] for n in names),
                         "cutoffs": result}
    return {"scans": scans, "groups": groups}


def pupil_results(raw, rng):
    settings = [*map(str, raw["defocus_nm"]), *raw["frequency_bands"]]
    trials, counts = {}, {}
    for name in raw["requested_scans"]:
        if name in raw["lower_twist_scans"]:
            continue
        walls = raw["scans"][name]["walls"]
        retained = [i for i, w in enumerate(walls) if qualified(
            w["cutoff"]["30"][NOMINAL]["measured"], *(fit for row in w["pupil"].values() for fit in row.values()))]
        if not retained:
            raise ValueError(f"{name} has no matched widths for the optical comparison")
        chosen = [walls[i] for i in retained]
        draws, support = paired_draws(walls, retained, rng)
        obs = array_of_widths(chosen, lambda w: w["cutoff"]["30"][NOMINAL]["measured"])
        trials[name] = {setting: {
            model: width_errors(obs, array_of_widths(chosen, lambda w: w["pupil"][setting][model]), draws)["width_error_nm"]
            for model in FIXED} for setting in settings}
        counts[name] = {"width_walls": len(chosen), "centers": [w["centers"] for w in chosen], "spatial_blocks": support}
    result = {}
    for setting in settings:
        errors = {model: np.mean([case[setting][model] for case in trials.values()], axis=0) for model in FIXED}
        diff = errors["raman_shape"] - errors["raman_scale"]
        lower, upper = interval(diff[1:])
        result[setting] = {
            "model_width_errors_nm": {name: float(values[0]) for name, values in errors.items()},
            "raman_error_difference_nm": float(diff[0]), "raman_paired_range_nm": [lower, upper],
            "transfer_at_0.2_cycles_per_nm": raw["pupil_transfer_at_0.2_cycles_per_nm"][setting],
            "preference": "13.2" if upper < 0 else "18.87" if lower > 0 else "unresolved",
        }
    return {"counts": counts, "width_walls": sum(row["width_walls"] for row in counts.values()),
            "defocus_settings_nm": raw["defocus_nm"], "bands": raw["frequency_bands"], "settings": result,
            "band_edges_cycles_nm": raw["band_edges_cycles_nm"],
            "band_transition_half_width_cycles_nm": raw["band_transition_half_width_cycles_nm"],
            "band_target_defocus_nm": raw["band_target_defocus_nm"]}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cases", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    contents = args.cases.read_bytes()
    raw = json.loads(contents)
    if set(raw["scans"]) != set(raw["requested_scans"]):
        raise ValueError("Incomplete cutoff and probe comparison")
    rng = np.random.default_rng(20261119)
    result = {"input_sha256": sha256(contents).hexdigest(), "cutoffs": selection_results(raw, rng),
              "pupil": pupil_results(raw, rng)}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
