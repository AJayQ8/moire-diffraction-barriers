"""Compare the extended low-twist scan with the original small crop."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_graphene_wall_signal import summarize
from run_heterobilayer_transfer import digest
from shared_compute import atomic_json, checkpoint_guard


def midpoint_and_direction(wall, scan, origin, pixel):
    midpoint = (np.asarray(wall["midpoint_rc"]) + origin) * pixel
    centers = np.asarray(scan["first_ring_geometry"]["centers_rc_px"])
    start, finish = centers[wall["centers"]]
    direction = finish - start
    return midpoint, direction / np.linalg.norm(direction)


def additional_walls(rows, scan, previous, origin, previous_origin, pixel, tolerance):
    old_geometry = [midpoint_and_direction(wall, previous, previous_origin, pixel)
                    for wall in previous["walls"][0]]
    result = []
    for wall in rows:
        midpoint, direction = midpoint_and_direction(wall, scan, origin, pixel)
        matches = any(np.linalg.norm(midpoint - center) <= tolerance
                      and abs(direction @ tangent) >= np.cos(np.deg2rad(20))
                      for center, tangent in old_geometry)
        if not matches:
            result.append(wall)
    return result


def report(rows):
    summary = summarize(rows)
    if summary.get("common_model_pairs", 0):
        measured = summary["observed_gap_on_common_pairs_nm_median"]
        predicted = summary["models"]["periodic"]["gap_nm_median"]
        repeat = summary["opposite_gap_med_abs_difference_nm"]
        error = abs(predicted - measured)
        resolved = (summary["common_model_pairs"] >= 6 and summary["opposite_triplet_pairs"] >= 3
                    and repeat is not None and error > max(.5, repeat))
        summary.update({"absolute_median_gap_error_nm": error, "magnitude_vs_repeatability":
                        "resolved mismatch" if resolved else "unresolved"})
    if rows:
        summary["profile_errors_all_geometry_chords"] = {}
        for model in ("periodic", "isolated"):
            errors = [[wall["models"][model]["transfer"][i]["error_by_ring"][1-i] for i in (0, 1)]
                      for wall in rows]
            summary["profile_errors_all_geometry_chords"][model] = np.mean(errors, axis=0).tolist()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("prior_score", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    checkpoint_guard()
    scan = json.loads(args.score.read_text())["scans"]["s1-10"]
    previous = json.loads(args.prior_score.read_text())["scans"]["s1-10"]
    protocol = json.loads(args.protocol.read_text())
    geometry = protocol["scans"]["s1-10"]
    if problem := scan.get("qualification_failure") or scan.get("reason"):
        atomic_json(args.output, {"qualification_failure": problem})
        return
    origin = np.asarray(geometry["region_rc"])[:, 0]
    lower, upper = np.asarray(geometry["prior_region_rc"]).T
    pixel = protocol["inputs"]["raster_pixel_nm"]
    comparisons = []
    for walls in scan["walls"]:
        outside = []
        for wall in walls:
            midpoint = np.asarray(wall["midpoint_rc"]) + origin
            margin = np.maximum(lower - midpoint, midpoint - (upper - 1)).max() * pixel
            if margin >= 5.:
                outside.append(wall)
        comparisons.append({
            "full_field": report(walls),
            "new_wall_positions": report(outside),
            "unmatched_to_original_chords": {
                str(tolerance): report(additional_walls(walls, scan, previous, origin, lower, pixel, tolerance))
                for tolerance in (1.5, 3.)
            },
        })
    for first, second in zip(*scan["walls"], strict=True):
        if first["centers"] != second["centers"]:
            raise ValueError("The two apertures do not share the same wall geometry")
    common_indices = [i for i, (first, second) in enumerate(zip(*scan["walls"], strict=True))
                      if all(fit["qualified"] for wall in (first, second) for fit in wall["measured"])]
    for comparison, walls in zip(comparisons, scan["walls"], strict=True):
        comparison["common_aperture_walls"] = report([walls[i] for i in common_indices])
    atomic_json(args.output, {
        "scope": "The frozen new-position set uses centers at least 5 nm outside the old rectangle. The retrospective "
                 "chord audit compares geometrically matched observations. New chords can fall inside the rectangle "
                 "because the larger image includes the stacking centers that qualify their geometry.",
        "apertures": comparisons,
        "inputs_sha256": {p.name: digest(p) for p in (args.score, args.protocol, args.prior_score, Path(__file__))},
    })


if __name__ == "__main__":
    main()
