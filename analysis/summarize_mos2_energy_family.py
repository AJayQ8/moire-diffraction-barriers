"""Compare two stacking-energy families on the same measured wall sections."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from summarize_mos2_geometry_probe import CAMERAS, GEOMETRY_SEED, block_draws, qualified
from summarize_mos2_saddle_sweep import FRAMES, group_scores, wall_fit


FAMILIES = ("fixed_AA_energy", "fixed_AA_curvature")
LOW_TWIST = {"DS1", "DS2", "DS13", "DS14"}


def score_map(scan, reference, alternative, energies, rng):
    original, new = reference["walls"], alternative["walls"]
    if [w["centers"] for w in original] != [w["centers"] for w in new]:
        raise ValueError(f"{scan}: the two families do not use the same wall sections")
    matched, errors = [], []
    for index, walls in enumerate(zip(original, new, strict=True)):
        observed = walls[0]["measured"]
        if observed != walls[1]["measured"]:
            raise ValueError(f"{scan}: a measured width changed")
        fits = [wall_fit(wall, camera, frame, energy) for wall in walls for energy in energies
                for camera in CAMERAS for frame in FRAMES]
        if qualified(observed, *fits):
            matched.append(index)
            errors.append(np.asarray([fit["widths_nm"] for fit in fits]) - observed["widths_nm"])
    if not matched:
        raise ValueError(f"{scan}: no sections qualify for all fits")
    differences = np.asarray(errors).reshape(len(matched), len(FAMILIES), len(energies),
                                             len(CAMERAS), len(FRAMES), 2)
    draws, blocks = block_draws(original, np.asarray(matched), rng)
    point = abs(differences.mean(axis=0)).mean(axis=-1)
    paired = (draws @ differences.reshape(len(matched), -1)).reshape(len(draws), *differences.shape[1:])
    coverage = {"source_sections": len(original), "common_sections": len(matched), "block": blocks,
                "included_centers": [original[i]["centers"] for i in matched],
                "map_mean_width_error_nm": point.tolist()}
    return point, abs(paired).mean(axis=-1), coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "alternative", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    original_bytes, alternative_bytes = (p.read_bytes() for p in (args.reference, args.alternative))
    original, alternative = (json.loads(payload) for payload in (original_bytes, alternative_bytes))
    energies = original["settings_meV_per_cell"]
    scans = original["requested_scans"]
    for key in ("requested_scans", "settings_meV_per_cell", "half_window_nm", "probe_fwhm_nm"):
        if original[key] != alternative[key]:
            raise ValueError(f"The two families use different {key}")
    if any(set(result["scans"]) != set(scans) for result in (original, alternative)):
        raise ValueError("Both sweeps must finish on every requested scan")
    original_hash = sha256(original_bytes).hexdigest()
    if alternative["input_sha256"][args.reference.name] != original_hash:
        raise ValueError("This alternative used a different reference sweep")
    rng = np.random.default_rng(GEOMETRY_SEED)
    points, draws, coverage = {}, {}, {}
    for scan in scans:
        points[scan], draws[scan], coverage[scan] = score_map(
            scan, original["scans"][scan], alternative["scans"][scan], energies, rng,
        )
    subsets = {
        "ten_maps": scans,
        "lower_twist_maps": [scan for scan in scans if scan in LOW_TWIST],
        "higher_twist_maps": [scan for scan in scans if scan not in LOW_TWIST],
        "omit_DS13": [scan for scan in scans if scan != "DS13"],
        "omit_DS14": [scan for scan in scans if scan != "DS14"],
    }
    results = {}
    for index, name in enumerate(FAMILIES):
        family_points = {scan: points[scan][index] for scan in scans}
        family_draws = {scan: draws[scan][:, index] for scan in scans}
        results[name] = {
            subset: group_scores(maps, family_points, family_draws, coverage, energies)
            for subset, maps in subsets.items()
        }
    report = {
        "schema": "mos2-matched-energy-families-summary-v01",
        "source_sha256": {args.reference.name: original_hash,
                          args.alternative.name: sha256(alternative_bytes).hexdigest()},
        "scope": ("Map-equal absolute error of map-mean fitted widths. Paired spatial draws use identical "
                  "sections and blocks in both families. Ranges condition on these scans and the fixed probe."),
        "settings_meV_per_cell": energies,
        "map_error_axes": [list(FAMILIES), "energy", list(CAMERAS), list(FRAMES)],
        "map_coverage": coverage,
        "families": results,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
