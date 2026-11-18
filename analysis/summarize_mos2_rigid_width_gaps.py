"""Compare fitted diffraction-order width differences on sections with all three model fits."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from summarize_mos2_geometry_probe import GEOMETRY_SEED, block_draws, interval, qualified


MODELS = ("rigid", "isolated", "network")
READOUTS = ("measured", *MODELS)


def scan_scores(name, walls, frozen_walls, rng):
    archived = {tuple(wall["centers"]): wall for wall in frozen_walls}
    if len(archived) != len(frozen_walls) or set(archived) != {tuple(w["centers"]) for w in walls}:
        raise ValueError(f"{name}: archived wall identifiers differ from the matched comparison")

    paired_count = 0
    indices, gaps, excluded = [], [], []
    for index, wall in enumerate(walls):
        nominal = archived[tuple(wall["centers"])]["nominal"]["transferred_even"]
        periodic = nominal["models"]["anchor"]["native"]
        rigid = nominal["models"]["rigid"]["native"]
        network = wall["models"]["network"]["scan_grid"]
        isolated = wall["models"]["isolated"]["scan_grid"]
        measured = wall["measured"]
        if measured != nominal["measured"] or network["qualified"] != periodic["qualified"]:
            raise ValueError(f"{name}: the saved measured or periodic fits do not match")
        np.testing.assert_allclose(network["widths_nm"], periodic["widths_nm"], rtol=0, atol=1e-8)
        if not qualified(measured, isolated, network):
            continue
        paired_count += 1
        if not qualified(rigid):
            excluded.append(wall["centers"])
            continue
        indices.append(index)
        gaps.append([fit["widths_nm"][0] - fit["widths_nm"][1]
                     for fit in (measured, rigid, isolated, network)])

    row = {"pair_sections": paired_count, "three_model_sections": len(indices),
           "excluded_rigid_wall_ids": excluded,
           "common_wall_ids": [walls[i]["centers"] for i in indices]}
    if not indices:
        return row, None
    gaps = np.array(gaps)
    mean_gaps = gaps.mean(axis=0)
    residuals = gaps[:, 1:] - gaps[:, :1]
    errors = abs(residuals.mean(axis=0))
    weights, blocks = block_draws(walls, indices, rng)
    row.update(
        mean_first_minus_second_nm=dict(zip(READOUTS, mean_gaps.tolist(), strict=True)),
        map_mean_gap_absolute_error_nm=dict(zip(MODELS, errors.tolist(), strict=True)),
        spatial_blocks=blocks,
    )
    return row, abs(weights @ residuals)


def summarize_group(maps, draws, names):
    kept = [name for name in names if maps[name]["three_model_sections"]]
    group = {
        "pair_maps": names,
        "three_model_maps": kept,
        "pair_sections": sum(maps[name]["pair_sections"] for name in names),
        "three_model_sections": sum(maps[name]["three_model_sections"] for name in names),
    }
    if not kept:
        return group
    values = np.array([[maps[name]["map_mean_gap_absolute_error_nm"][model] for model in MODELS]
                       for name in kept]).mean(axis=0)
    errors = np.mean([draws[name] for name in kept], axis=0)
    group["map_mean_gap_absolute_error_nm"] = dict(zip(MODELS, values.tolist(), strict=True))
    group["network_error_reduction_p05_p95_nm"] = {
        model: interval(errors[:, i] - errors[:, MODELS.index("network")])
        for i, model in enumerate(MODELS) if model != "network"
    }
    return group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matched_cases", type=Path)
    parser.add_argument("archived_geometry_cases", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    match_bytes = args.matched_cases.read_bytes()
    frozen_bytes = args.archived_geometry_cases.read_bytes()
    match = json.loads(match_bytes)
    frozen = json.loads(frozen_bytes)
    frozen_hash = sha256(frozen_bytes).hexdigest()
    if frozen_hash != match["input_sha256"]["mos2_geometry_probe_v005.json"]:
        raise ValueError("The matched comparison used a different archived geometry record")

    rng = np.random.default_rng(GEOMETRY_SEED)
    names = match["requested_scans"]
    maps, draws = {}, {}
    for name in names:
        maps[name], draws[name] = scan_scores(
            name, match["scans"][name]["walls"], frozen["scans"][name]["walls"], rng,
        )
    high = match["primary_scans"]
    low = [name for name in names if name not in high]
    report = {
        "schema": "mos2-rigid-width-gap-comparison-v01",
        "case_sha256": sha256(match_bytes).hexdigest(),
        "geometry_case_sha256": frozen_hash,
        "probe_fwhm_nm": match["probe_fwhm_nm"],
        "half_window_nm": match["half_window_nm"],
        "selection": "Successful interior fits in both orders for measured, rigid, isolated and periodic profiles.",
        "uncertainty": "Fifth and ninety-fifth percentiles of paired spatial-block resampling within the selected maps.",
        "groups": {group: summarize_group(maps, draws, subset)
                   for group, subset in (("higher_twist", high), ("lower_twist", low))},
        "maps": maps,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
