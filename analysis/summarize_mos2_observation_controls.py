"""Summarize the fixed-cohort probe extension and scan-observation controls."""

import argparse
import json
from pathlib import Path

import numpy as np

from run_mos2_observation_controls import DETECTORS, RAMAN
from summarize_mos2_probe_crossover import crossings, joined_cameras, map_shape_loss, map_width_loss, qualified


def read(directory, name):
    return json.loads((directory / name).read_text())


def score_curve(rows, probes, metric):
    array = np.array([[metric(setting) for setting in settings] for settings in rows.values()])
    trajectory = array.mean(axis=0)
    delta = trajectory[:, 1] - trajectory[:, 0]
    minima = {}
    for j, model in enumerate(RAMAN):
        i = int(np.argmin(trajectory[:, j]))
        minima[model] = {"probe_nm": probes[i], "error": float(trajectory[i, j])}
    return {
        "model_errors": trajectory.tolist(), "minima": minima, "crossings": crossings(probes, delta),
        "leave_one_map_out_crossings": {
            name: crossings(probes, np.diff(np.delete(array, i, axis=0).mean(axis=0), axis=-1)[:, 0])
            for i, name in enumerate(rows)
        },
    }


def extension_summary(records, predecessor, old_probes, new_probes):
    probes = old_probes + [p for p in new_probes if p > old_probes[-1]]
    joined = {}
    roundtrips = {"widths_nm": [], "profile": []}
    for scan, previous in predecessor.items():
        current = {tuple(row["centers"]): row["extension"] for row in records[scan]["walls"]}
        assert set(current) == set(previous)
        values = {}
        for key, wall in previous.items():
            values[key] = {}
            for arm in DETECTORS:
                old_endpoint, new_endpoint = wall[arm][-1], current[key]["1.75"][arm]
                for model in RAMAN:
                    old, new = old_endpoint["models"][model], new_endpoint["models"][model]
                    roundtrips["profile"].extend(abs(np.array(old["transfer_error"]) - new["transfer_error"]))
                    if all(old["qualified"]) and all(new["qualified"]):
                        roundtrips["widths_nm"].extend(abs(np.array(old["widths_nm"]) - new["widths_nm"]))
                values[key][arm] = wall[arm] + [current[key][f"{p:g}"][arm] for p in probes[len(old_probes):]]
        joined[scan] = list(values.values())
    profiles = {}
    for arm in DETECTORS:
        rows = {scan: list(zip(*(wall[arm] for wall in walls), strict=True)) for scan, walls in joined.items()}
        profiles[arm] = score_curve(rows, probes, map_shape_loss)
    width_comparisons = {}
    for stop in new_probes:
        used_probes = [p for p in probes if p <= stop]
        k = len(used_probes)
        cohort = {scan: [row for row in walls if all(qualified(r) for v in row.values() for r in v[:k])]
                  for scan, walls in joined.items()}
        counts = {scan: len(walls) for scan, walls in cohort.items()}
        comparison = {"probes_nm": used_probes, "width_counts": counts, "width_walls": sum(counts.values())}
        if all(counts.values()):
            comparison["detectors"] = {}
            for arm in DETECTORS:
                rows = {scan: list(zip(*(wall[arm][:k] for wall in walls), strict=True))
                        for scan, walls in cohort.items()}
                comparison["detectors"][arm] = score_curve(rows, used_probes, map_width_loss)
        width_comparisons[f"{stop:g}"] = comparison
    agreement = {key: {"median": float(np.median(v)), "max": float(max(v))} for key, v in roundtrips.items()}
    return {"probes_nm": probes, "width_comparisons": width_comparisons,
            "profile_walls": sum(map(len, predecessor.values())), "roundtrip_at_1.75_nm": agreement,
            "profiles": profiles}


def raster_summary(records, probes):
    kinds = ("continuous", "native_raster")
    scores = {key: {"widths": {}, "profiles": {}} for key in kinds}
    counts = {}
    for scan, value in records.items():
        walls = [
            [{name: {"measured": row["measured"], "models": row[name]} for name in kinds}
             for p in probes for row in [wall["raman_raster"][f"{p:g}"]]]
            for wall in value["walls"]
        ]
        common = [entry for entry in walls if all(qualified(row[name]) for row in entry for name in kinds)]
        counts[scan] = len(common)
        for name in kinds:
            scores[name]["widths"][scan] = [[wall[i][name] for wall in common] for i in range(len(probes))]
            scores[name]["profiles"][scan] = [[wall[i][name] for wall in walls] for i in range(len(probes))]
    return {"probes_nm": probes, "width_counts": counts, "width_walls": sum(counts.values()),
            "detectors": {
                name: {"widths": score_curve(v["widths"], probes, map_width_loss),
                       "profiles": score_curve(v["profiles"], probes, map_shape_loss)}
                for name, v in scores.items()
            }}


def difference_summary(records, primary, control):
    by_scan = {}
    for scan, row in records.items():
        cuts, raster, repeats = [], [], []
        refs = {tuple(wall["centers"]): wall for wall in primary["scans"][scan]["walls"]}
        controls = {tuple(wall["centers"]): wall for wall in control["scans"][scan]["walls"]}
        for wall in row["walls"]:
            chord = tuple(wall["centers"])
            models = controls[chord]["readouts"]["transferred_even"]["1.25"]["4.25"]["models"]
            for ring, pair in enumerate(wall["opposite_spots"]):
                if pair["pairs"]:
                    repeats.append([pair["rms"], models["anchor"]["transfer_error"][1-ring],
                                    models["rigid"]["transfer_error"][1-ring]])
            if not wall["in_fig2_width_cohort"]:
                continue
            fits = wall["cut_averaging"]
            aligned = fits["first_order_centers_aligned"]
            reading = wall["anchor_raster"]
            raw = np.asarray(fits["unaligned"]["widths_nm"])
            prediction = refs[chord]["readouts"]["compact"]["1.25"]["4.25"]["models"]["anchor"]["widths_nm"]
            if aligned and all(aligned["qualified"]) and all(all(f["qualified"]) for f in fits["individual"]):
                re_read = np.array(aligned["widths_nm"])
                raw_read = np.mean([fit["widths_nm"] for fit in fits["individual"]], axis=0)
                span = np.ptp([fit["centers_nm"][0] for fit in fits["individual"]])
                cuts.append([raw, raw_read, re_read, prediction, [span, span]])
            if all(all(v["qualified"]) for v in reading.values()):
                raster.append([raw, reading["continuous"]["widths_nm"], reading["native_raster"]["widths_nm"]])
        if not all((cuts, raster, repeats)):
            raise ValueError(f"A matched control has no comparable sections on {scan}")
        group = np.array(cuts)
        raw, singles, aligned, predicted, _ = group.mean(axis=0)
        rendering = np.array(raster)
        raster_means = rendering.mean(axis=0)
        by_scan[scan] = {
            "twist_deg": row["twist_deg"],
            "adjacent_cut_spacing_nm_median": float(np.median([w["adjacent_cut_spacing_nm"] for w in row["walls"]])),
            "cut_sections": len(cuts), "cut_center_span_nm_median": float(np.median(group[:, 4, 0])),
            "unaligned_width_nm": raw.tolist(), "single_cut_width_nm": singles.tolist(),
            "aligned_width_nm": aligned.tolist(), "frozen_anchor_width_nm": predicted.tolist(),
            "width_errors_unaligned_individual_aligned_nm": np.mean(abs(np.array([raw, singles, aligned])-predicted),
                                                                    axis=1).tolist(),
            "raster_sections": len(raster),
            "raster_continuous_and_native_width_shift_nm": (raster_means[2] - raster_means[1]).tolist(),
            "raster_errors_continuous_native_nm": np.mean(abs(raster_means[1:] - raster_means[0]), axis=1).tolist(),
            "opposite_order_scores": np.mean(repeats, axis=0).tolist(), "opposite_order_directions": len(repeats),
        }
    ordered = sorted(by_scan.values(), key=lambda v: v["twist_deg"])
    groups = {}
    for name, group in (("all", ordered), ("low_four", ordered[:4]), ("high_six", ordered[4:])):
        groups[name] = {
            key: np.mean([v[key] for v in group], axis=0).tolist()
            for key in ("width_errors_unaligned_individual_aligned_nm", "raster_errors_continuous_native_nm",
                        "raster_continuous_and_native_width_shift_nm", "opposite_order_scores")
        }
        for field in ("cut_sections", "raster_sections", "opposite_order_directions"):
            groups[name][field] = sum(v[field] for v in group)
    return {"scans": by_scan, "groups": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("controls", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    control = read(args.evidence, "mos2_wall_landscape_camera_controls_v01.json")
    primary = read(args.evidence, "mos2_wall_landscape_v01.json")
    original = read(args.evidence, "mos2_probe_crossover_summary_v01.json")
    endpoints = read(args.evidence, "mos2_raman_probe_sensitivity_v01.json")
    sweep = read(args.evidence, "mos2_raman_probe_crossover_v01.json")
    data = json.loads(args.controls.read_text())
    records = data["scans"]
    if set(records) != set(primary["scans"]):
        raise ValueError("The ten-map numerical replay is incomplete")
    previous = joined_cameras(control, endpoints, sweep, original["probes_nm"])
    result = {
        "extensions": extension_summary(records, previous, original["probes_nm"], data["extended_probes_nm"]),
        "raster": raster_summary(records, data["raster_probes_nm"]),
        "observation": difference_summary(records, primary, control),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
