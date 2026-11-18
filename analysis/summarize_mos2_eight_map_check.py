"""Reaggregate the saved MoS2 saddle scores after omitting DS13 and DS14."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from summarize_mos2_energy_family import FAMILIES, score_map
from summarize_mos2_geometry_probe import (
    CAMERAS,
    DRAW_COUNT,
    GEOMETRY_SEED,
    PHASES,
    interval,
)
from summarize_mos2_saddle_sweep import FRAMES, group_scores, map_scores


PAPER_DIR = Path(__file__).resolve().parents[1]
OMITTED = {"DS13", "DS14"}
ENDPOINTS = ("10", "13.2")


def digest(payload):
    return sha256(payload).hexdigest()


def assert_curve_replay(actual, expected, label):
    """Check the prior saved full-map result against this replay."""
    for camera in CAMERAS:
        for frame in FRAMES:
            got = actual["cameras"][camera]["curves"][frame]
            want = expected["cameras"][camera]["curves"][frame]
            for key in ("mean_curve_nm", "pointwise_p05_p95_nm"):
                np.testing.assert_allclose(got[key], want[key], rtol=0, atol=1e-12,
                                           err_msg=f"{label} {camera} {frame} {key}")
            for key in ("sampled_minimum_nm", "best_setting_meV_per_cell"):
                if got[key] != want[key]:
                    raise AssertionError(f"{label} {camera} {frame} {key} differs")
            np.testing.assert_allclose(
                got["sampled_best_setting_p05_p95_meV_per_cell"],
                want["sampled_best_setting_p05_p95_meV_per_cell"],
                rtol=0, atol=1e-12,
                err_msg=f"{label} {camera} {frame} sampled best barrier interval",
            )


def verify_sweep_replay(sweep, orientation, saved_summary):
    """Reproduce the original all-grid cohort and the orientation endpoint cohort."""
    names = sweep["requested_scans"]
    if set(names) != set(sweep["scans"]):
        raise ValueError("The saddle sweep is incomplete")
    if set(names) != set(orientation["width_comparison"]["per_map"]):
        raise ValueError("The orientation summary does not cover the sweep maps")

    rng = np.random.default_rng(GEOMETRY_SEED)
    points, draws, coverage = {}, {}, {}
    for scan in names:
        points[scan], draws[scan], coverage[scan] = map_scores(
            scan,
            sweep["scans"][scan],
            sweep["settings_meV_per_cell"],
            orientation["width_comparison"]["per_map"][scan],
            rng,
        )
    replay = group_scores(names, points, draws, coverage, sweep["settings_meV_per_cell"])
    assert_curve_replay(replay, saved_summary["ten_maps"], "original sweep replay")
    kept = [scan for scan in names if scan not in OMITTED]
    return {
        "full_ten_map_score_curves_reproduced": True,
        "orientation_endpoint_cohorts_reproduced_for_retained_maps": kept,
        "original_sweep_common_sections_by_retained_map": {
            scan: coverage[scan]["common_sections"] for scan in kept
        },
    }


def source_indices(scan_record, included_centers):
    wanted = {tuple(center) for center in included_centers}
    found = [i for i, wall in enumerate(scan_record["walls"])
             if tuple(wall["centers"]) in wanted]
    if len(found) != len(wanted):
        raise ValueError("Could not map the family cohort back to its saved source indices")
    return found


def verify_family_replay(names, points, draws, coverage, saved_summary, settings):
    """Reproduce both full-map energy-family curves from the saved score record."""
    for scan in names:
        for key in ("source_sections", "common_sections", "block", "included_centers"):
            if coverage[scan][key] != saved_summary["map_coverage"][scan][key]:
                raise AssertionError(f"{scan}: saved family cohort or block design differs in {key}")
        np.testing.assert_allclose(
            coverage[scan]["map_mean_width_error_nm"],
            saved_summary["map_coverage"][scan]["map_mean_width_error_nm"],
            rtol=0, atol=1e-12,
            err_msg=f"{scan}: saved per-map family score curves",
        )
    for family_index, family in enumerate(FAMILIES):
        family_points = {scan: points[scan][family_index] for scan in names}
        family_draws = {scan: draws[scan][:, family_index] for scan in names}
        replay = group_scores(names, family_points, family_draws, coverage, settings)
        expected = saved_summary["families"][family]["ten_maps"]
        assert_curve_replay(replay, expected, f"{family} full-family replay")
    return True


def verify_archived_rotated_equality(scan, reference_record, alternative_record, settings):
    """The selective-rotation branch must fall back to archived rows off DS13/DS14."""
    fit_rows = 0
    for family_record in (reference_record, alternative_record):
        walls = family_record["walls"]
        for wall in walls:
            for camera in CAMERAS:
                arms = wall["arms"][camera]
                if "rotated" in arms:
                    raise AssertionError(f"{scan}: unexpected rotated rows on the retained set")
                for setting in settings:
                    archived = arms["nominal"][setting]["widths_nm"]
                    rotated = arms.get("rotated", arms["nominal"])[setting]["widths_nm"]
                    np.testing.assert_array_equal(archived, rotated)
                    fit_rows += 1
    return fit_rows


def conditional_group(selected, points, draws, coverage, settings):
    """Return a map-equal curve and reselected-minimum block-resampling ranges."""
    summary = group_scores(selected, points, draws, coverage, settings)
    group_draws = np.mean([draws[scan] for scan in selected], axis=0)
    cameras = {}
    for camera_index, camera in enumerate(CAMERAS):
        cameras[camera] = {"frames": {}}
        for frame_index, frame in enumerate(FRAMES):
            curve = summary["cameras"][camera]["curves"][frame]
            draw_curve = group_draws[:, :, camera_index, frame_index]
            sampled_minima = np.min(draw_curve, axis=1)
            cameras[camera]["frames"][frame] = {
                "map_mean_width_error_curve_nm": curve["mean_curve_nm"],
                "paired_within_map_p05_p95_by_barrier_nm": curve["pointwise_p05_p95_nm"],
                "best_grid_barrier_meV_per_cell": curve["best_setting_meV_per_cell"],
                "reselected_best_barrier_p05_p95_meV_per_cell":
                    curve["sampled_best_setting_p05_p95_meV_per_cell"],
                "minimum_map_mean_width_error_nm": curve["sampled_minimum_nm"],
                "reselected_minimum_error_p05_p95_nm": interval(sampled_minima),
                "minimum_at_grid_boundary": curve["minimum_at_grid_boundary"],
            }
    return group_draws, cameras


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path,
                        default=PAPER_DIR / "evidence/mos2_saddle_sweep_v01.json")
    parser.add_argument("--curvature", type=Path,
                        default=PAPER_DIR / "evidence/mos2_energy_family_v01.json")
    parser.add_argument("--orientation-summary", type=Path,
                        default=PAPER_DIR / "evidence/mos2_orientation_candidates_v01.json")
    parser.add_argument("--sweep-summary", type=Path,
                        default=PAPER_DIR / "evidence/mos2_saddle_sweep_summary_v01.json")
    parser.add_argument("--family-summary", type=Path,
                        default=PAPER_DIR / "evidence/mos2_energy_family_summary_v01.json")
    parser.add_argument("--output", type=Path,
                        default=PAPER_DIR / "evidence/mos2_eight_map_check_v019.json")
    parser.add_argument("--tex-output", type=Path)
    args = parser.parse_args()

    payloads = {path.name: path.read_bytes() for path in (
        args.sweep, args.curvature, args.orientation_summary,
        args.sweep_summary, args.family_summary,
    )}
    sweep = json.loads(payloads[args.sweep.name])
    curvature = json.loads(payloads[args.curvature.name])
    orientation = json.loads(payloads[args.orientation_summary.name])
    saved_sweep = json.loads(payloads[args.sweep_summary.name])
    saved_families = json.loads(payloads[args.family_summary.name])

    names = sweep["requested_scans"]
    settings = sweep["settings_meV_per_cell"]
    if set(names) != set(curvature["requested_scans"]):
        raise ValueError("The reference and fixed-AA-curvature records use different map sets")
    for key in ("requested_scans", "settings_meV_per_cell", "half_window_nm", "probe_fwhm_nm"):
        if sweep[key] != curvature[key]:
            raise ValueError(f"The two family records use different {key}")
    if curvature["input_sha256"][args.sweep.name] != digest(payloads[args.sweep.name]):
        raise ValueError("The curvature record does not reference this exact saddle sweep")

    sweep_replay = verify_sweep_replay(sweep, orientation, saved_sweep)

    # Preserve the established full-ten-map RNG sequence before selecting eight maps.
    rng = np.random.default_rng(GEOMETRY_SEED)
    points, draws, coverage = {}, {}, {}
    for scan in names:
        points[scan], draws[scan], coverage[scan] = score_map(
            scan, sweep["scans"][scan], curvature["scans"][scan], settings, rng,
        )
    family_replay = verify_family_replay(names, points, draws, coverage, saved_families, settings)
    selected = [scan for scan in names if scan not in OMITTED]
    if len(selected) != 8 or OMITTED.intersection(selected):
        raise AssertionError("The requested eight-map selection was not formed")

    retained_coverage = {}
    branch_fit_rows = 0
    for scan in selected:
        indices = source_indices(sweep["scans"][scan], coverage[scan]["included_centers"])
        retained_coverage[scan] = {
            "source_sections": coverage[scan]["source_sections"],
            "common_sections_across_both_families_and_all_fits": coverage[scan]["common_sections"],
            "common_source_indices_zero_based": indices,
            "block": coverage[scan]["block"],
        }
        branch_fit_rows += verify_archived_rotated_equality(
            scan, sweep["scans"][scan], curvature["scans"][scan], settings,
        )
        np.testing.assert_array_equal(points[scan][..., 0], points[scan][..., 1])
        np.testing.assert_array_equal(draws[scan][..., 0], draws[scan][..., 1])

    family_results = {}
    family_draws = {}
    for family_index, family in enumerate(FAMILIES):
        family_points = {scan: points[scan][family_index] for scan in names}
        family_sampled = {scan: draws[scan][:, family_index] for scan in names}
        group_draw, curves = conditional_group(
            selected, family_points, family_sampled, coverage, settings,
        )
        family_draws[family] = group_draw
        family_results[family] = {
            "maps": selected,
            "matched_sections": sum(coverage[scan]["common_sections"] for scan in selected),
            "cameras": curves,
        }

    paired_family_comparison = {}
    for camera_index, camera in enumerate(CAMERAS):
        frame_rows = {}
        for frame_index, frame in enumerate(FRAMES):
            energy_draws = family_draws["fixed_AA_energy"][:, :, camera_index, frame_index]
            curvature_draws = family_draws["fixed_AA_curvature"][:, :, camera_index, frame_index]
            energy_minima = np.min(energy_draws, axis=1)
            curvature_minima = np.min(curvature_draws, axis=1)
            energy_barriers = np.asarray(settings, dtype=float)[np.argmin(energy_draws, axis=1)]
            curvature_barriers = np.asarray(settings, dtype=float)[np.argmin(curvature_draws, axis=1)]
            energy_point = family_results["fixed_AA_energy"]["cameras"][camera]["frames"][frame]
            curvature_point = family_results["fixed_AA_curvature"]["cameras"][camera]["frames"][frame]
            frame_rows[frame] = {
                "curvature_minus_energy_minimum_error_nm": (
                    curvature_point["minimum_map_mean_width_error_nm"]
                    - energy_point["minimum_map_mean_width_error_nm"]
                ),
                "paired_reselected_minimum_error_difference_p05_p95_nm": interval(
                    curvature_minima - energy_minima,
                ),
                "curvature_minus_energy_best_barrier_meV_per_cell": (
                    float(curvature_point["best_grid_barrier_meV_per_cell"])
                    - float(energy_point["best_grid_barrier_meV_per_cell"])
                ),
                "paired_reselected_best_barrier_difference_p05_p95_meV_per_cell": interval(
                    curvature_barriers - energy_barriers,
                ),
            }
        paired_family_comparison[camera] = {"frames": frame_rows}

    sensitivity = {}
    for family_index, family in enumerate(FAMILIES):
        sensitivity[family] = {}
        for camera_index, camera in enumerate(CAMERAS):
            rows = {}
            for scan in selected:
                archived = points[scan][family_index, :, camera_index, 0]
                rows[scan] = {
                    "score_at_10_meV_per_cell_nm": float(archived[settings.index(ENDPOINTS[0])]),
                    "score_at_13_2_meV_per_cell_nm": float(archived[settings.index(ENDPOINTS[1])]),
                    "score_change_13_2_minus_10_nm": float(
                        archived[settings.index(ENDPOINTS[1])] - archived[settings.index(ENDPOINTS[0])]
                    ),
                    "best_grid_barrier_meV_per_cell": settings[int(np.argmin(archived))],
                    "minimum_score_nm": float(np.min(archived)),
                }
            sensitivity[family][camera] = rows

    result = {
        "schema": "mos2-eight-map-saddle-check-v019",
        "inputs_sha256": {
            name: digest(payload) for name, payload in payloads.items()
        },
        "analysis_sha256": {
            Path(__file__).name: digest(Path(__file__).read_bytes()),
            "summarize_mos2_saddle_sweep.py": digest(Path(__file__).with_name("summarize_mos2_saddle_sweep.py").read_bytes()),
            "summarize_mos2_energy_family.py": digest(Path(__file__).with_name("summarize_mos2_energy_family.py").read_bytes()),
            "summarize_mos2_geometry_probe.py": digest(Path(__file__).with_name("summarize_mos2_geometry_probe.py").read_bytes()),
        },
        "scope": {
            "omitted_maps": sorted(OMITTED),
            "retained_maps_in_saved_rng_order": selected,
            "settings_meV_per_cell": settings,
            "half_window_nm": sweep["half_window_nm"],
            "nominal_probe_fwhm_nm": sweep["probe_fwhm_nm"],
            "intensity_fits": list(CAMERAS),
            "frames": list(FRAMES),
            "common_eligibility": (
                "A section qualifies only if its measured widths and every saved width fit from both "
                "families, every grid barrier, both frames, and both intensity fits qualify."
            ),
            "score": (
                "For each map, barrier, family, frame, and intensity fit, average signed width residuals "
                "over the common sections separately by diffraction order, take the absolute values, "
                "then average the two orders. Group scores average maps equally."
            ),
            "resampling": {
                "seed": GEOMETRY_SEED,
                "origins": [list(phase) for phase in PHASES],
                "draws_per_origin_per_map": DRAW_COUNT,
                "block_side_rule": "sqrt(2/sqrt(3)) times the median retained row spacing per map",
                "rng_map_order": names,
                "scope": (
                    "Paired within-map spatial block resampling, conditional on the selected maps, "
                    "saved probe, common section cohorts, and material records. Draw rows remain paired "
                    "across both families, barriers, frames, intensity fits, and diffraction orders. "
                    "Each map has equal group weight; intervals do not represent between-map or "
                    "device-level uncertainty."
                ),
            },
        },
        "replay_verification": {
            "original_sweep": sweep_replay,
            "energy_family_full_ten_map_curves_reproduced": family_replay,
            "fixed_curvature_reference_sha256_matches": True,
        },
        "map_coverage": retained_coverage,
        "archived_rotated_branch_check": {
            "verified": True,
            "basis": "The retained maps have no saved rotated arm rows; the established wall_fit helper therefore reuses their archived nominal rows.",
            "retained_maps": selected,
            "exact_fit_width_arrays_compared_across_both_families_cameras_and_all_grid_barriers": branch_fit_rows,
            "maximum_absolute_group_score_difference_nm": 0.0,
        },
        "families": family_results,
        "paired_family_comparison": paired_family_comparison,
        "barrier_sensitivity_screen": {
            "metric": "Per-map existing map-mean width error at 13.2 meV/cell minus that at 10 meV/cell; no binary sensitivity threshold is imposed.",
            "families": sensitivity,
        },
        "limitation": (
            "These results are conditional on the saved simulation records, fixed 1.25 nm nominal probe, "
            "selected eight maps, and common fit-qualified sections; they do not quantify between-map, "
            "device-level, or broader material-parameter uncertainty."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")

    if args.tex_output:
        lines = [
            "% Generated by summarize_mos2_eight_map_check.py from saved v01 evidence.",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Energy family & Intensity fit & Best barrier & Barrier range & Width error & Error range \\",
            r" & & (meV/cell) & (meV/cell) & (nm) & (nm) \\",
            r"\midrule",
        ]
        display = {
            "fixed_AA_energy": "Fixed AA energy",
            "fixed_AA_curvature": "Fixed AA curvature",
            "transferred_even": "Four other maps",
            "separate_disjoint": "Cuts excluded",
        }
        for family in FAMILIES:
            for camera in CAMERAS:
                row = result["families"][family]["cameras"][camera]["frames"]["archived"]
                barrier_range = row["reselected_best_barrier_p05_p95_meV_per_cell"]
                error_range = row["reselected_minimum_error_p05_p95_nm"]
                lines.append(
                    f"{display[family]} & {display[camera]} & "
                    f"{float(row['best_grid_barrier_meV_per_cell']):.3g} & "
                    f"[{barrier_range[0]:.3g}, {barrier_range[1]:.3g}] & "
                    f"{row['minimum_map_mean_width_error_nm']:.3f} & "
                    f"[{error_range[0]:.3f}, {error_range[1]:.3f}] \\\\"
                )
        lines.extend([r"\bottomrule", r"\end{tabular}", ""])
        args.tex_output.parent.mkdir(parents=True, exist_ok=True)
        args.tex_output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
