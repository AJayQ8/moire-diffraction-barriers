"""Summarize archived saddle-sweep scores and conditional spatial-block ranges."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from summarize_mos2_geometry_probe import (
    CAMERAS,
    DRAW_COUNT,
    GEOMETRY_SEED,
    LOW,
    PHASES,
    block_draws,
    interval,
)
from summarize_mos2_saddle_sweep import FRAMES, group_scores, map_scores, wall_fit


ENERGIES = ("13.2", "15")
MAP_POINT_SCANS = ("DS1", "DS2", "DS13", "DS14")


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def order_residuals(scan, record, settings, indices, draw):
    """Return absolute mean residuals and paired block draws by diffraction order."""
    walls = record["walls"]
    take = [walls[i] for i in indices]
    residual = np.empty((len(take), len(settings), len(CAMERAS), len(FRAMES), 2), dtype=float)
    for wi, wall in enumerate(take):
        for ei, setting in enumerate(settings):
            for ci, camera in enumerate(CAMERAS):
                for fi, frame in enumerate(FRAMES):
                    residual[wi, ei, ci, fi] = (
                        np.asarray(wall_fit(wall, camera, frame, setting)["widths_nm"], dtype=float)
                        - np.asarray(wall["measured"]["widths_nm"], dtype=float)
                    )
    points = np.abs(residual.mean(axis=0))
    sampled = np.abs((draw @ residual.reshape(len(take), -1)).reshape(
        len(draw), len(settings), len(CAMERAS), len(FRAMES), 2,
    ))
    return points, sampled


def group_comparison(selected, points, sampled, order_points, order_sampled, settings):
    """Summarize archived 13.2/15 errors, equally averaging maps."""
    group_point = np.mean([points[name] for name in selected], axis=0)
    group_sampled = np.mean([sampled[name] for name in selected], axis=0)
    group_order_point = np.mean([order_points[name] for name in selected], axis=0)
    group_order_sampled = np.mean([order_sampled[name] for name in selected], axis=0)
    out = {}
    for ci, camera in enumerate(CAMERAS):
        row = {"score_nm": {}, "score_p05_p95_nm": {},
               "absolute_order_residual_nm": {}, "absolute_order_residual_p05_p95_nm": {}}
        values = {}
        draws = {}
        for energy in ENERGIES:
            ei = settings.index(energy)
            values[energy] = group_point[ei, ci, 0]
            draws[energy] = group_sampled[:, ei, ci, 0]
            row["score_nm"][energy] = float(values[energy])
            row["score_p05_p95_nm"][energy] = interval(draws[energy])
            row["absolute_order_residual_nm"][energy] = group_order_point[ei, ci, 0].tolist()
            row["absolute_order_residual_p05_p95_nm"][energy] = np.asarray(interval(
                group_order_sampled[:, ei, ci, 0, :],
            )).T.tolist()
        row["score_nm"]["delta_15_minus_13.2"] = float(values["15"] - values["13.2"])
        row["score_delta_p05_p95_nm"] = interval(draws["15"] - draws["13.2"])
        out[camera] = row
    return out


def summarize(data, orientation_summary):
    names = data["requested_scans"]
    settings = data["settings_meV_per_cell"]
    if set(names) != set(data["scans"]):
        raise ValueError("The saved sweep must contain every requested scan")
    if set(names) != set(orientation_summary["width_comparison"]["per_map"]):
        raise ValueError("The matched orientation summary is incomplete")

    rng = np.random.default_rng(GEOMETRY_SEED)
    order_rng = np.random.default_rng(GEOMETRY_SEED)
    points, sampled, coverage = {}, {}, {}
    order_points, order_sampled = {}, {}
    for name in names:
        points[name], sampled[name], coverage[name] = map_scores(
            name,
            data["scans"][name],
            settings,
            orientation_summary["width_comparison"]["per_map"][name],
            rng,
        )
        indices = coverage[name]["common_indices"]
        draw, blocks = block_draws(data["scans"][name]["walls"], indices, order_rng)
        if blocks != coverage[name]["block"]:
            raise ValueError(f"{name}: paired order draws do not reproduce the saved block design")
        order_points[name], order_sampled[name] = order_residuals(
            name, data["scans"][name], settings, indices, draw,
        )
        np.testing.assert_allclose(
            order_sampled[name].mean(axis=-1), sampled[name], rtol=0, atol=1e-12,
        )

    low = [name for name in LOW if name in names]
    high = [name for name in names if name not in low]
    groups = {"all_maps": names, "low_four": low, "high_six": high}
    group_drawn = {}
    optima = {}
    for group_name, selected in groups.items():
        result = group_scores(selected, points, sampled, coverage, settings)
        group_drawn[group_name] = np.mean([sampled[name] for name in selected], axis=0)
        frames = {}
        for ci, camera in enumerate(CAMERAS):
            frames[camera] = {}
            for fi, frame in enumerate(FRAMES):
                curve = result["cameras"][camera]["curves"][frame]
                draw_scores = group_drawn[group_name][:, :, ci, fi]
                frames[camera][frame] = {
                    "best_setting_meV_per_cell": curve["best_setting_meV_per_cell"],
                    "minimum_score_nm": curve["sampled_minimum_nm"],
                    "sampled_minimum_score_p05_p95_nm": interval(np.min(draw_scores, axis=1)),
                    "sampled_best_setting_p05_p95_meV_per_cell":
                        curve["sampled_best_setting_p05_p95_meV_per_cell"],
                    "minimum_at_grid_boundary": curve["minimum_at_grid_boundary"],
                }
        optima[group_name] = {
            "maps": selected,
            "matched_sections": sum(coverage[name]["common_sections"] for name in selected),
            "cameras": {camera: {"frames": frames[camera]} for camera in CAMERAS},
        }

    comparison = {
        group_name: group_comparison(selected, points, sampled, order_points, order_sampled, settings)
        for group_name, selected in groups.items()
    }

    selected_maps = {}
    for name in MAP_POINT_SCANS:
        if name not in names:
            continue
        per_camera = {}
        for ci, camera in enumerate(CAMERAS):
            per_frame = {}
            for fi, frame in enumerate(FRAMES):
                scores, order_errors = {}, {}
                for energy in ENERGIES:
                    ei = settings.index(energy)
                    scores[energy] = float(points[name][ei, ci, fi])
                    order_errors[energy] = order_points[name][ei, ci, fi].tolist()
                per_frame[frame] = {
                    "score_nm": scores,
                    "score_delta_15_minus_13.2_nm": float(scores["15"] - scores["13.2"]),
                    "absolute_order_residual_nm": order_errors,
                }
            per_camera[camera] = per_frame
        selected_maps[name] = {
            "common_sweep_sections": coverage[name]["common_sections"],
            "source_sections": coverage[name]["source_sections"],
            "block_side_nm": coverage[name]["block"]["side_nm"],
            "occupied_blocks_by_origin": coverage[name]["block"]["occupied_blocks"],
            "frames_by_camera": per_camera,
        }

    return {
        "schema": "mos2-saddle-reporting-v01",
        "source_sha256": {},
        "energy_grid_meV_per_cell": [float(x) for x in settings],
        "half_window_nm": data["half_window_nm"],
        "probe_fwhm_nm": data["probe_fwhm_nm"],
        "score_definition": (
            "For each map, energy, frame, and detector calibration, average signed predicted-minus-measured "
            "width residuals over the common all-energy/all-frame/all-calibration wall cohort separately "
            "for the two diffraction orders. The width score is the mean of the two absolute residuals, "
            "in nm. Group scores average maps equally."
        ),
        "cohorts": {
            "resampling_map_order": names,
            "all_maps": names,
            "low_four": low,
            "high_six": high,
            "common_sweep_sections": {name: coverage[name]["common_sections"] for name in names},
            "total_common_sweep_sections": sum(coverage[name]["common_sections"] for name in names),
        },
        "paired_spatial_block_resampling": {
            "seed": GEOMETRY_SEED,
            "origins": [list(x) for x in PHASES],
            "draws_per_origin": DRAW_COUNT,
            "total_draws_per_map": DRAW_COUNT * len(PHASES),
            "scope": (
                "Within-map resampling conditional on the fixed selected scans and common qualified "
                "all-sweep sections; the same block draw is paired across energy candidates, frames, "
                "detector calibrations, and diffraction orders. Groups average each map equally. "
                "Intervals do not represent between-map or device-level sampling uncertainty."
            ),
        },
        "archived_width_comparison": {
            "settings_meV_per_cell": list(ENERGIES),
            "all_maps": comparison["all_maps"],
            "groups": {"low_four": comparison["low_four"], "high_six": comparison["high_six"]},
        },
        "optima": {"groups": optima},
        "selected_map_points": selected_maps,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=root / "evidence/mos2_saddle_sweep_v01.json")
    parser.add_argument("--orientation-summary", type=Path,
                        default=root / "evidence/mos2_orientation_candidates_v01.json")
    parser.add_argument("--output", type=Path, default=root / "evidence/mos2_saddle_reporting_v01.json")
    args = parser.parse_args()
    sweep_bytes = args.sweep.read_bytes()
    orientation_bytes = args.orientation_summary.read_bytes()
    result = summarize(json.loads(sweep_bytes), json.loads(orientation_bytes))
    result["source_sha256"] = {
        args.sweep.name: sha256(sweep_bytes).hexdigest(),
        args.orientation_summary.name: sha256(orientation_bytes).hexdigest(),
        Path(__file__).name: sha(Path(__file__)),
        "summarize_mos2_saddle_sweep.py": sha(Path(__file__).with_name("summarize_mos2_saddle_sweep.py")),
        "summarize_mos2_geometry_probe.py": sha(Path(__file__).with_name("summarize_mos2_geometry_probe.py")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
