"""Compare the fixed-input MoS2 saddle curves on one common width cohort."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from summarize_mos2_geometry_probe import CAMERAS, DRAW_COUNT, GEOMETRY_SEED, PHASES, block_draws, interval, qualified


FRAMES = ("archived", "rotate_ds13_ds14")
ENDPOINTS = ("10", "13.2")
ROTATED_MAPS = {"DS13", "DS14"}


def wall_fit(wall, camera, frame, setting):
    key = "nominal" if frame == "archived" else "rotated"
    available = wall["arms"][camera]
    return available.get(key, available["nominal"])[setting]


def map_scores(scan, record, settings, reference, rng):
    walls = record["walls"]
    settings_qualification = []
    pairs = []
    for wall in walls:
        by_setting = [
            [wall_fit(wall, camera, frame, key) for camera in CAMERAS for frame in FRAMES]
            for key in settings
        ]
        settings_qualification.append([qualified(wall["measured"], *fits) for fits in by_setting])
        pairs.append([[fit["widths_nm"] for fit in fits] for fits in by_setting])
        for camera in CAMERAS:
            if ("rotated" in wall["arms"][camera]) != (scan in ROTATED_MAPS):
                raise ValueError(f"{scan}: the selective rotation has the wrong map scope")
    eligible = np.asarray(settings_qualification)
    endpoint_cohort = np.flatnonzero(eligible[:, [settings.index(key) for key in ENDPOINTS]].all(axis=1))
    if endpoint_cohort.tolist() != reference["common_qualified_source_indices_zero_based"]:
        raise ValueError(f"{scan}: the v008 endpoint cohort does not reproduce")
    retained = np.flatnonzero(eligible.all(axis=1))
    if not retained.size:
        raise ValueError(f"{scan}: no matched fit qualifies across the complete sweep")

    measures = np.asarray([wall["measured"]["widths_nm"] for wall in walls])
    predictions = np.asarray(pairs).reshape(len(walls), len(settings), len(CAMERAS), len(FRAMES), 2)
    differences = predictions - measures[:, None, None, None, :]
    point = abs(differences[retained].mean(axis=0)).mean(axis=-1)
    reproduce = abs(differences[endpoint_cohort].mean(axis=0)).mean(axis=-1)
    for j, camera in enumerate(CAMERAS):
        candidates = reference["cameras"][camera]["candidates"]
        prior = [[candidates[name]["equal_order_map_score_nm"] for name in
                  (f"archive_{key}", f"two_rotated_{key}")] for key in ENDPOINTS]
        np.testing.assert_allclose(reproduce[[settings.index(k) for k in ENDPOINTS], j, :], prior,
                                   rtol=0, atol=1e-12)

    draw, blocks = block_draws(walls, retained, rng)
    resampled = (draw @ differences[retained].reshape(len(retained), -1)).reshape(
        len(draw), len(settings), len(CAMERAS), len(FRAMES), 2,
    )
    shape = {
        "source_sections": len(walls),
        "endpoint_sections": int(len(endpoint_cohort)),
        "common_sections": int(len(retained)),
        "common_indices": retained.tolist(),
        "qualified_per_setting_all_frames_and_detectors": eligible.sum(axis=0).tolist(),
        "included_centers": [walls[k]["centers"] for k in retained],
        "excluded_since_endpoint_centers": [walls[k]["centers"] for k in sorted(set(endpoint_cohort) - set(retained))],
        "block": blocks,
        "map_mean_width_error_nm": point.tolist(),
    }
    return point, abs(resampled).mean(axis=-1), shape


def group_scores(selected, points, resamples, coverage, settings):
    scores = np.mean([points[scan] for scan in selected], axis=0)
    draws = np.mean([resamples[scan] for scan in selected], axis=0)
    energy = np.asarray([float(key) for key in settings])
    cameras = {}
    for j, camera in enumerate(CAMERAS):
        curves = {}
        best_indices = np.argmin(scores[:, j, :], axis=0)
        sampled_best_indices = np.argmin(draws[:, :, j, :], axis=1)
        sampled_best_energies = energy[sampled_best_indices]
        for k, frame in enumerate(FRAMES):
            winner = int(best_indices[k])
            curves[frame] = {
                "mean_curve_nm": scores[:, j, k].tolist(),
                "pointwise_p05_p95_nm": interval(draws[:, :, j, k]),
                "sampled_minimum_nm": float(scores[winner, j, k]),
                "best_setting_meV_per_cell": settings[winner],
                "sampled_best_setting_p05_p95_meV_per_cell": interval(sampled_best_energies[:, k]),
                "minimum_at_grid_boundary": winner in (0, len(settings)-1),
            }
        fixed_gap = draws[:, best_indices[1], j, 1] - draws[:, best_indices[0], j, 0]
        delta = scores[best_indices[1], j, 1] - scores[best_indices[0], j, 0]
        pointwise_contrast = draws[:, :, j, 1] - draws[:, :, j, 0]
        cameras[camera] = {
            "curves": curves,
            "best_setting_difference_rotated_minus_archived_meV_per_cell": float(energy[best_indices[1]] - energy[best_indices[0]]),
            "best_setting_difference_p05_p95_meV_per_cell": interval(
                sampled_best_energies[:, 1] - sampled_best_energies[:, 0],
            ),
            "best_settings_score_difference_rotated_minus_archived_nm": float(delta),
            "paired_range_nm_at_fixed_best_settings": interval(fixed_gap),
            "pointwise_rotated_minus_archived_p05_p95_nm": interval(pointwise_contrast),
        }
    return {"maps": selected, "sections": sum(coverage[scan]["common_sections"] for scan in selected),
            "cameras": cameras}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sweep", "orientation_summary", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    case_data, prior_data = (p.read_bytes() for p in (args.sweep, args.orientation_summary))
    raw, prior = (json.loads(payload) for payload in (case_data, prior_data))
    names = raw["requested_scans"]
    settings = raw["settings_meV_per_cell"]
    if set(names) != set(raw["scans"]) or set(names) != set(prior["width_comparison"]["per_map"]):
        raise ValueError("The full paired ten-map result is required")
    points, resamples, coverage = {}, {}, {}
    rng = np.random.default_rng(GEOMETRY_SEED)
    for scan in names:
        points[scan], resamples[scan], coverage[scan] = map_scores(
            scan, raw["scans"][scan], settings, prior["width_comparison"]["per_map"][scan], rng,
        )
    full = group_scores(names, points, resamples, coverage, settings)
    omitted = {scan: group_scores([name for name in names if name != scan], points, resamples, coverage, settings)
               for scan in names}
    out = {
        "schema": "mos2-fixed-probe-saddle-sweep-summary-v01",
        "source_sha256": {
            args.sweep.name: sha256(case_data).hexdigest(),
            args.orientation_summary.name: sha256(prior_data).hexdigest(),
            Path(__file__).name: sha256(Path(__file__).read_bytes()).hexdigest(),
            "summarize_mos2_geometry_probe.py": sha256(Path(__file__).with_name("summarize_mos2_geometry_probe.py").read_bytes()).hexdigest(),
        },
        "settings_meV_per_cell": settings,
        "half_window_nm": raw["half_window_nm"],
        "probe_fwhm_nm": raw["probe_fwhm_nm"],
        "order_of_map_score_array": ["barrier", list(CAMERAS), list(FRAMES)],
        "map_coverage": coverage,
        "spatial_blocks": {"seed": GEOMETRY_SEED, "origins": PHASES, "draws_per_origin": DRAW_COUNT,
                           "scope": "Paired within each fixed map, conditional on selected public scans and inputs."},
        "ten_maps": full,
        "leave_one_map_out": omitted,
    }
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
