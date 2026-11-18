"""Summarize conditional spatial-block uncertainty in the frozen graphene cases.

The input is the read-only v004 source bundle. This script performs no diffraction
replay: it resamples paired, already-fitted wall records from the three published
scans and records the analysis lineage in a small JSON file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
DEFAULT_ARCHIVE = PAPER / "moire_v004_sources.zip"
DEFAULT_OUTPUT = PAPER / "evidence" / "graphene_map_uncertainty_v005.json"
BLOCK_FACTORS = (0.5, 1.0, 1.5)
ORIGIN_PHASES = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))
SPACING_TO_SQRT_CELL_AREA = math.sqrt(2.0 / math.sqrt(3.0))
PIXEL_NM = 0.5
DRAWS_PER_ORIGIN = 1000
SEED = 20261115


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(zf: zipfile.ZipFile, member: str):
    content = zf.read(member)
    return json.loads(content), content


def checksum_manifest(zf: zipfile.ZipFile):
    raw = zf.read("evidence/SHA256SUMS").decode()
    result = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        checksum, member = line.split(maxsplit=1)
        member = member.removeprefix("./")
        result[f"evidence/{member}"] = checksum
    return result


def fitted_gap(fits):
    if fits is None or len(fits) != 2 or not all(x.get("qualified") is True for x in fits):
        return None
    return float(fits[0]["width_nm"]) - float(fits[1]["width_nm"])


def coordinates_nm(row, crop_origin_rc=(0.0, 0.0)):
    if row.get("midpoint_nm") is not None:
        return [float(x) for x in row["midpoint_nm"]]
    if row.get("midpoint_rc") is not None:
        # A row/column swap is only a rigid coordinate rotation, so it preserves
        # Euclidean distances used by the square block partition.
        return [PIXEL_NM * (float(x) + float(offset))
                for x, offset in zip(row["midpoint_rc"], crop_origin_rc)]
    raise ValueError("A frozen wall record has no saved midpoint coordinates")


def geometry_record(row, window=None, crop_origin_rc=(0.0, 0.0)):
    if window is None:
        source = row
    else:
        source = row.get("windows", {}).get(window)
        if source is None:
            raise ValueError(f"Missing geometry window {window}")
    return {
        "midpoint_nm": coordinates_nm(source, crop_origin_rc),
        "spacing_nm": float(source["spacing_nm"]),
    }


def normalize_repeat_wall(row, crop_origin_rc=(0.0, 0.0)):
    measured = row.get("measured")
    if measured is None or len(measured) != 2:
        return None
    measured_gap = fitted_gap(measured)
    if measured_gap is None:
        return None
    models = {}
    for name in ("periodic", "isolated"):
        model = row.get("models", {}).get(name, {})
        widths = model.get("widths")
        gap = fitted_gap(widths)
        if gap is None:
            return None
        models[name] = {"gap_nm": gap, "widths_nm": [float(x["width_nm"]) for x in widths]}
    opposite = [fitted_gap(x) for x in row.get("opposite_triplets", [])]
    opposite_disagreement = None
    if len(opposite) == 2 and None not in opposite:
        opposite_disagreement = abs(opposite[0] - opposite[1])
    return {
        "midpoint_nm": coordinates_nm(row, crop_origin_rc),
        "spacing_nm": float(row["spacing_nm"]),
        "measured_widths_nm": [float(x["width_nm"]) for x in measured],
        "measured_gap_nm": measured_gap,
        "models": models,
        "opposite_gap_disagreement_nm": opposite_disagreement,
    }


def normalize_s15_wall(row, window, crop_origin_rc=(0.0, 0.0)):
    case = row.get("windows", {}).get(window)
    if case is None:
        return None
    measured = case.get("measured")
    measured_gap = fitted_gap(measured)
    if measured_gap is None:
        return None
    models = {}
    for name in ("periodic", "isolated"):
        widths = case.get("models", {}).get(name, {}).get("widths")
        gap = fitted_gap(widths)
        if gap is None:
            return None
        models[name] = {"gap_nm": gap, "widths_nm": [float(x["width_nm"]) for x in widths]}
    opposite = [fitted_gap(x) for x in case.get("opposite_triplets", [])]
    opposite_disagreement = None
    if len(opposite) == 2 and None not in opposite:
        opposite_disagreement = abs(opposite[0] - opposite[1])
    return {
        "midpoint_nm": coordinates_nm(case, crop_origin_rc),
        "spacing_nm": float(case["spacing_nm"]),
        "measured_widths_nm": [float(x["width_nm"]) for x in measured],
        "measured_gap_nm": measured_gap,
        "models": models,
        "opposite_gap_disagreement_nm": opposite_disagreement,
    }


def make_cases_from_repeat(rows):
    # The runner's common-pair set requires both measured orders plus both model
    # orders to qualify, precisely the cohort represented by normalize_*.
    output = []
    for row in rows:
        case = normalize_repeat_wall(row)
        if case is not None:
            output.append(case)
    return output


def make_cases_s15(rows, window, require_opposite, crop_origin_rc=(0.0, 0.0)):
    output = []
    for row in rows:
        case = normalize_s15_wall(row, window, crop_origin_rc)
        if case is None:
            continue
        if require_opposite and case["opposite_gap_disagreement_nm"] is None:
            continue
        output.append(case)
    return output


def median_summary(values):
    values = np.asarray(values, dtype=float)
    return float(np.median(values))


def percentiles(values):
    values = np.asarray(values, dtype=float)
    return {
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
    }


def occupied_blocks(cases, all_geometry, side_nm, phase):
    xy0 = np.asarray([r["midpoint_nm"] for r in all_geometry], dtype=float)
    origin = np.min(xy0, axis=0)
    dx, dy = phase
    buckets = {}
    for i, row in enumerate(cases):
        x, y = row["midpoint_nm"]
        key = (math.floor((x - origin[0]) / side_nm + dx),
               math.floor((y - origin[1]) / side_nm + dy))
        buckets.setdefault(key, []).append(i)
    return list(buckets.values())


def values_for_case(case):
    out = {
        "measured_width1_nm": case["measured_widths_nm"][0],
        "measured_width2_nm": case["measured_widths_nm"][1],
        "measured_gap_nm": case["measured_gap_nm"],
    }
    for model, row in case["models"].items():
        out[f"{model}_gap_nm"] = row["gap_nm"]
        out[f"{model}_width1_nm"] = row["widths_nm"][0]
        out[f"{model}_width2_nm"] = row["widths_nm"][1]
    return out


def resample_condition(cases, all_geometry, factor, draws, seed):
    if not cases:
        raise ValueError("Cannot resample an empty wall cohort")
    median_spacing_nm = statistics.median(float(row["spacing_nm"]) for row in cases)
    side_nm = factor * SPACING_TO_SQRT_CELL_AREA * median_spacing_nm
    point_rows = [values_for_case(case) for case in cases]
    metrics = sorted(point_rows[0])
    points = {metric: median_summary([row[metric] for row in point_rows]) for metric in metrics}
    difference_metrics = [
        f"{model}_minus_experiment_median_gap_nm" for model in ("periodic", "isolated")
    ]
    for model in ("periodic", "isolated"):
        points[f"{model}_minus_experiment_median_gap_nm"] = (
            points[f"{model}_gap_nm"] - points["measured_gap_nm"]
        )
    all_metrics = [*metrics, *difference_metrics]
    observed_opposite = [
        case["opposite_gap_disagreement_nm"] for case in cases
        if case["opposite_gap_disagreement_nm"] is not None
    ]
    point_repeatability = (median_summary(observed_opposite) if observed_opposite else None)
    by_metric = {key: [] for key in all_metrics}
    block_counts = []
    rng = np.random.default_rng(seed)
    for phase in ORIGIN_PHASES:
        blocks = occupied_blocks(cases, all_geometry, side_nm, phase)
        block_counts.append(len(blocks))
        if not blocks:
            continue
        for _ in range(draws):
            draw = rng.integers(0, len(blocks), size=len(blocks))
            selected = [cases[index] for block_id in draw for index in blocks[int(block_id)]]
            selected_rows = [values_for_case(case) for case in selected]
            sampled = {key: median_summary([row[key] for row in selected_rows]) for key in metrics}
            for model in ("periodic", "isolated"):
                sampled[f"{model}_minus_experiment_median_gap_nm"] = (
                    sampled[f"{model}_gap_nm"] - sampled["measured_gap_nm"]
                )
            # Avoid presenting a spot-repeatability distribution as a median
            # CI. This statistic is reported only as the source's separate
            # per-wall readout disagreement yardstick.
            for key in all_metrics:
                by_metric[key].append(sampled[key])

    support = {
        "block_side_nm": float(side_nm),
        "block_factor": factor,
        "geometry_candidate_walls_for_origin": len(all_geometry),
        "block_spacing_reference_walls": len(cases),
        "median_spacing_nm": float(median_spacing_nm),
        "paired_width_walls": len(cases),
        "occupied_blocks_by_origin": block_counts,
        "minimum_occupied_blocks": min(block_counts) if block_counts else 0,
        "single_block_origins": sum(value == 1 for value in block_counts),
        "zero_block_origins": sum(value == 0 for value in block_counts),
        "bootstrap_draws_per_origin": draws,
        "origins": len(ORIGIN_PHASES),
    }
    return {
        "point": points,
        "block_percentiles": {key: percentiles(value) for key, value in by_metric.items()},
        "repeatability_yardstick": {
            "matched_opposite_spot_pairs": len(observed_opposite),
            "median_absolute_opposite_gap_disagreement_nm": point_repeatability,
            "interpretation": "Median absolute per-wall disagreement between two opposite-spot gap readouts. This is a within-wall repeatability scale, not a confidence interval for the scan median.",
        },
        "support": support,
    }


def summarize_cohort(name, description, cases, geometry, draws, base_seed):
    if not cases:
        raise ValueError(f"Empty cohort: {name}")
    scale_rows = {}
    for index, factor in enumerate(BLOCK_FACTORS):
        scale_rows[str(factor)] = resample_condition(cases, geometry, factor, draws, base_seed + index * 1021)
    return {
        "name": name,
        "description": description,
        "candidate_geometry_walls": len(geometry),
        "qualified_paired_width_walls": len(cases),
        "point": scale_rows["1.0"]["point"],
        "primary_block_scale": scale_rows["1.0"],
        "block_scale_sensitivity": {
            factor: {
                "support": result["support"],
                "block_percentiles": result["block_percentiles"],
            }
            for factor, result in scale_rows.items()
        },
        "limits": [
            "Percentile ranges are conditional within-scan spatial resampling sensitivity for these fixed saved walls, with per-wall measured and model widths paired inside each sampled block.",
            "No acquisition-level or independent-specimen uncertainty is estimated; the public records do not provide an independent-device crosswalk.",
            "The block length and origin phase are plausible spatial-clustering sensitivity choices, not empirically measured correlation lengths.",
        ],
    }


def paired_window_change(cases_adaptive, cases_fixed, all_geometry, factor, draws, seed):
    """Paired fixed-minus-adaptive changes on the identical wall subset."""
    if len(cases_adaptive) != len(cases_fixed) or not cases_adaptive:
        raise ValueError("Paired window cohorts must contain the same nonempty wall list")
    for a, f in zip(cases_adaptive, cases_fixed):
        if a["midpoint_nm"] != f["midpoint_nm"]:
            raise ValueError("Adaptive/fixed wall order differs in the frozen cases")
    median_spacing_nm = statistics.median(float(row["spacing_nm"]) for row in cases_adaptive)
    side_nm = factor * SPACING_TO_SQRT_CELL_AREA * median_spacing_nm
    points = {}
    rows_a = [values_for_case(case) for case in cases_adaptive]
    rows_f = [values_for_case(case) for case in cases_fixed]
    for statistic in ("mean", "median"):
        for metric in ("measured_gap_nm", "periodic_gap_nm"):
            va = np.asarray([row[metric] for row in rows_a], dtype=float)
            vf = np.asarray([row[metric] for row in rows_f], dtype=float)
            op = np.mean if statistic == "mean" else np.median
            points[f"{statistic}_{metric}_fixed_minus_adaptive_nm"] = float(op(vf) - op(va))
        points[f"{statistic}_periodic_minus_experiment_residual_fixed_minus_adaptive_nm"] = (
            points[f"{statistic}_periodic_gap_nm_fixed_minus_adaptive_nm"]
            - points[f"{statistic}_measured_gap_nm_fixed_minus_adaptive_nm"]
        )

    rng = np.random.default_rng(seed)
    samples = {key: [] for key in points}
    phase_counts = []
    for phase in ORIGIN_PHASES:
        blocks = occupied_blocks(cases_adaptive, all_geometry, side_nm, phase)
        phase_counts.append(len(blocks))
        for _ in range(draws):
            picks = rng.integers(0, len(blocks), size=len(blocks))
            selected = [idx for block_id in picks for idx in blocks[int(block_id)]]
            a = [rows_a[idx] for idx in selected]
            f = [rows_f[idx] for idx in selected]
            for statistic in ("mean", "median"):
                op = np.mean if statistic == "mean" else np.median
                d_measured = float(op([row["measured_gap_nm"] for row in f])
                                   - op([row["measured_gap_nm"] for row in a]))
                d_periodic = float(op([row["periodic_gap_nm"] for row in f])
                                   - op([row["periodic_gap_nm"] for row in a]))
                samples[f"{statistic}_measured_gap_nm_fixed_minus_adaptive_nm"].append(d_measured)
                samples[f"{statistic}_periodic_gap_nm_fixed_minus_adaptive_nm"].append(d_periodic)
                samples[f"{statistic}_periodic_minus_experiment_residual_fixed_minus_adaptive_nm"].append(
                    d_periodic - d_measured
                )
    return {
        "cohort_walls": len(cases_adaptive),
        "statistic": "Fixed-window minus adaptive-window; both windows are resampled with the same sampled wall blocks.",
        "block_side_nm": float(side_nm),
        "median_spacing_nm": float(median_spacing_nm),
        "block_spacing_reference_walls": len(cases_adaptive),
        "geometry_candidate_walls_for_origin": len(all_geometry),
        "occupied_blocks_by_origin": phase_counts,
        "point": points,
        "block_percentiles": {key: percentiles(value) for key, value in samples.items()},
        "draws_per_origin": draws,
        "origins": len(ORIGIN_PHASES),
    }


def summary_checks(repeat_summary, low_summary, s15_saved):
    checks = {}
    primary6 = repeat_summary["apertures"][0]["all"]["summary"]
    cases6 = make_cases_from_repeat(repeat_summary["walls"][0])
    checks["s1-6_primary"] = {
        "case_pairs": {"expected": 52, "actual": len(cases6), "passed": len(cases6) == 52},
        "measured_gap_median_nm": {
            "saved": primary6["observed_gap_nm_median"],
            "cases": median_summary([row["measured_gap_nm"] for row in cases6]),
        },
        "periodic_gap_median_nm": {
            "saved": primary6["models"]["periodic"]["gap_nm_median"],
            "cases": median_summary([row["models"]["periodic"]["gap_nm"] for row in cases6]),
        },
        "opposite_spot_count": {"saved": 40, "cases": sum(x["opposite_gap_disagreement_nm"] is not None for x in cases6)},
    }
    checks["s1-10_primary"] = {
        "case_pairs": {"expected": 14, "actual": len(low_summary), "passed": len(low_summary) == 14},
        "measured_gap_median_nm": {
            "saved": 1.8047603239488095,
            "cases": median_summary([row["measured_gap_nm"] for row in low_summary]),
        },
        "periodic_gap_median_nm": {
            "saved": 2.4874162529056316,
            "cases": median_summary([row["models"]["periodic"]["gap_nm"] for row in low_summary]),
        },
        "opposite_spot_count": {"saved": 6, "cases": sum(x["opposite_gap_disagreement_nm"] is not None for x in low_summary)},
    }
    rows = s15_saved["apertures"][1]["walls"]
    s15_checks = {}
    expected = {"adaptive": (13, 2.975735242828248, 1.8016400755922577),
                "fixed": (12, 0.5253350674740576, 1.145398529578794)}
    for window in ("adaptive", "fixed"):
        cases = make_cases_s15(rows, window, require_opposite=True)
        mean_gap = float(np.mean([row["measured_gap_nm"] for row in cases]))
        repeat = median_summary([row["opposite_gap_disagreement_nm"] for row in cases])
        n, mean_expected, repeat_expected = expected[window]
        s15_checks[window] = {
            "matched_opposite_readouts": {"expected": n, "actual": len(cases), "passed": len(cases) == n},
            "mean_gap_nm": {"saved": mean_expected, "cases": mean_gap},
            "median_opposite_disagreement_nm": {"saved": repeat_expected, "cases": repeat},
        }
    checks["s1-5_existing_warning_summary"] = s15_checks
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draws", type=int, default=DRAWS_PER_ORIGIN)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    archive_bytes_hash = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(args.archive) as zf:
        manifest = checksum_manifest(zf)
        repeat, repeat_bytes = read_json(zf, "evidence/graphene_wall_signal_repeat_v01.json")
        low, low_bytes = read_json(zf, "evidence/graphene_wall_signal_low_twist_coverage_v01.json")
        s15, s15_bytes = read_json(zf, "evidence/graphene_wall_warning_s1-5_v01.json")

    for member, content in (
        ("evidence/graphene_wall_signal_repeat_v01.json", repeat_bytes),
        ("evidence/graphene_wall_signal_low_twist_coverage_v01.json", low_bytes),
        ("evidence/graphene_wall_warning_s1-5_v01.json", s15_bytes),
    ):
        if sha256_bytes(content) != manifest[member]:
            raise ValueError(f"Archive checksum manifest mismatch for {member}")

    s16_geometry = [geometry_record(row) for row in repeat["walls"][0]]
    # The full geometry support includes walls whose measured widths fail; the
    # saved case list already includes all geometrically selected chords.
    s16_cases = make_cases_from_repeat(repeat["walls"][0])

    s110_rows = low["scans"]["s1-10"]["walls"][0]
    s110_region = low["scans"]["s1-10"]["receipt"]["region_rc"]
    s110_origin_rc = (float(s110_region[0][0]), float(s110_region[1][0]))
    s110_geometry = [geometry_record(row, crop_origin_rc=s110_origin_rc) for row in s110_rows]
    s110_cases = [normalize_repeat_wall(row, s110_origin_rc) for row in s110_rows]
    s110_cases = [row for row in s110_cases if row is not None]

    s15_rows = s15["apertures"][1]["walls"]
    s15_region = s15["extraction"]["region_rc"]
    s15_origin_rc = (float(s15_region[0][0]), float(s15_region[1][0]))
    s15_geometry = [geometry_record(row, "adaptive", s15_origin_rc) for row in s15_rows]
    s15_adaptive_repeat = make_cases_s15(s15_rows, "adaptive", require_opposite=True, crop_origin_rc=s15_origin_rc)
    s15_fixed_repeat = make_cases_s15(s15_rows, "fixed", require_opposite=True, crop_origin_rc=s15_origin_rc)
    s15_adaptive_all = make_cases_s15(s15_rows, "adaptive", require_opposite=False, crop_origin_rc=s15_origin_rc)
    s15_fixed_all = make_cases_s15(s15_rows, "fixed", require_opposite=False, crop_origin_rc=s15_origin_rc)
    s15_common_adaptive = []
    s15_common_fixed = []
    for row in s15_rows:
        adaptive = normalize_s15_wall(row, "adaptive", s15_origin_rc)
        fixed = normalize_s15_wall(row, "fixed", s15_origin_rc)
        if (adaptive is not None and fixed is not None
                and adaptive["opposite_gap_disagreement_nm"] is not None
                and fixed["opposite_gap_disagreement_nm"] is not None):
            s15_common_adaptive.append(adaptive)
            s15_common_fixed.append(fixed)
    if len(s15_common_adaptive) != 8:
        raise ValueError(f"Expected the v004 common eight-section s1-5 cohort, got {len(s15_common_adaptive)}")

    checks = summary_checks(repeat, s110_cases, s15)
    # Ensure saved case-level medians reproduce the manuscript's primary
    # s1-6 and s1-10 values before generating uncertainty ranges.
    if not np.isclose(checks["s1-6_primary"]["measured_gap_median_nm"]["saved"],
                      checks["s1-6_primary"]["measured_gap_median_nm"]["cases"], atol=1e-12):
        raise ValueError("s1-6 case median no longer matches its frozen summary")
    if not np.isclose(checks["s1-10_primary"]["measured_gap_median_nm"]["saved"],
                      checks["s1-10_primary"]["measured_gap_median_nm"]["cases"], atol=1e-12):
        raise ValueError("s1-10 case median no longer matches its frozen summary")

    conditions = [
        summarize_cohort(
            "s1-6 author aperture 2.662 px, all matched widths",
            "Published mean twist 1.2592°. Local wall twists span 0.999–1.7399°. Full-scan primary aperture and frozen paired width/model cohort.",
            s16_cases, s16_geometry, args.draws, args.seed + 100,
        ),
        summarize_cohort(
            "s1-10 author aperture 2.466 px, all matched widths",
            "Published mean twist 0.6272°. This is the full-field extension of a previously examined central crop that yielded two width pairs.",
            s110_cases, s110_geometry, args.draws, args.seed + 200,
        ),
        summarize_cohort(
            "s1-5 aperture 3.5 px, adaptive window, opposite-spot matched",
            "Published twist 1.2343°. Uses the 13 rows with qualified primary width and both opposite-triplet gap readouts; this reproduces the saved repeatability-warning cohort.",
            s15_adaptive_repeat, s15_geometry, args.draws, args.seed + 300,
        ),
        summarize_cohort(
            "s1-5 aperture 3.5 px, fixed 4.25 nm window, opposite-spot matched",
            "Same scan and geometry; uses the 12 rows with qualified primary width and both opposite-triplet gap readouts.",
            s15_fixed_repeat, s15_geometry, args.draws, args.seed + 400,
        ),
        summarize_cohort(
            "s1-5 aperture 3.5 px, adaptive window, all qualified widths",
            "Sensitivity cohort retaining every row with qualified measured, periodic and isolated width pairs; 27 rows. The published warning statistic uses the stricter opposite-spot matched subset.",
            s15_adaptive_all, s15_geometry, args.draws, args.seed + 500,
        ),
        summarize_cohort(
            "s1-5 aperture 3.5 px, fixed 4.25 nm window, all qualified widths",
            "Sensitivity cohort retaining every row with qualified measured, periodic and isolated width pairs; 18 rows. The published warning statistic uses the stricter opposite-spot matched subset.",
            s15_fixed_all, s15_geometry, args.draws, args.seed + 600,
        ),
        summarize_cohort(
            "s1-5 aperture 3.5 px, adaptive window, common eight",
            "The exact eight sections shared with the fixed-window comparison: measured widths and both opposite-triplet gap readouts qualify in both windows.",
            s15_common_adaptive, s15_geometry, args.draws, args.seed + 700,
        ),
        summarize_cohort(
            "s1-5 aperture 3.5 px, fixed 4.25 nm window, common eight",
            "The same eight sections as the adaptive-window common cohort; paired windows preserve identical wall identities.",
            s15_common_fixed, s15_geometry, args.draws, args.seed + 800,
        ),
    ]

    common8_means = {
        "adaptive_measured_gap_nm": float(np.mean([row["measured_gap_nm"] for row in s15_common_adaptive])),
        "fixed_measured_gap_nm": float(np.mean([row["measured_gap_nm"] for row in s15_common_fixed])),
        "adaptive_periodic_gap_nm": float(np.mean([row["models"]["periodic"]["gap_nm"] for row in s15_common_adaptive])),
        "fixed_periodic_gap_nm": float(np.mean([row["models"]["periodic"]["gap_nm"] for row in s15_common_fixed])),
    }
    expected_common8 = {
        "adaptive_measured_gap_nm": 2.4832228805096204,
        "fixed_measured_gap_nm": -0.10526117568812776,
        "adaptive_periodic_gap_nm": 4.4010886386184795,
        "fixed_periodic_gap_nm": 3.043557053647832,
    }
    for key, expected in expected_common8.items():
        if not np.isclose(common8_means[key], expected, atol=1e-12):
            raise ValueError(f"s1-5 common-eight mean mismatch in {key}: {common8_means[key]} vs {expected}")
    paired_window_sensitivity = {
        str(factor): paired_window_change(
            s15_common_adaptive, s15_common_fixed, s15_geometry, factor, args.draws, args.seed + 900 + index * 1031
        )
        for index, factor in enumerate(BLOCK_FACTORS)
    }

    output = {
        "schema": "graphene_map_uncertainty_v005",
        "source": {
            "bundle": args.archive.name,
            "bundle_sha256": archive_bytes_hash,
            "input_members_sha256": {
                "evidence/graphene_wall_signal_repeat_v01.json": manifest["evidence/graphene_wall_signal_repeat_v01.json"],
                "evidence/graphene_wall_signal_low_twist_coverage_v01.json": manifest["evidence/graphene_wall_signal_low_twist_coverage_v01.json"],
                "evidence/graphene_wall_warning_s1-5_v01.json": manifest["evidence/graphene_wall_warning_s1-5_v01.json"],
            },
            "analysis_script_sha256": sha256_bytes(Path(__file__).read_bytes()),
        },
        "estimand": {
            "measured_gap_nm": "Per selected wall, first-order apparent width minus second-order apparent width. Report median over the qualified wall cohort.",
            "model_gap_nm": "Same signed difference from saved periodic or isolated model width fits on the same wall cohort.",
            "model_minus_experiment_median_gap_nm": "Median of model-predicted wall gaps minus median measured wall gap, calculated on each paired spatial-block resample.",
            "resampling_unit": "Occupied spatial blocks within each fixed scan; each sampled block carries paired measured and model width records.",
            "cohorts": "s1-6 and s1-10 use the frozen all-model-qualified paired width sets. s1-5 primary repeats the prior opposite-triplet matched subsets for its warning diagnostic; all-qualified s1-5 medians are included as sensitivity cohorts.",
        },
        "coordinate_transforms": {
            "s1-6": {"stored_coordinate": "midpoint_nm", "pixel_origin_offset_nm": [0.0, 0.0]},
            "s1-10": {
                "stored_coordinate": "crop-relative midpoint_rc",
                "region_origin_rc": list(s110_origin_rc),
                "raster_pixel_nm": PIXEL_NM,
                "global_nm_coordinate": "0.5 × (midpoint_rc + [9,9]); row/column order is a rigid coordinate rotation for the square-block geometry.",
            },
            "s1-5": {
                "stored_coordinate": "crop-relative midpoint_rc",
                "region_origin_rc": list(s15_origin_rc),
                "raster_pixel_nm": PIXEL_NM,
                "global_nm_coordinate": "0.5 × (midpoint_rc + [40,40]); row/column order is a rigid coordinate rotation for the square-block geometry.",
            },
        },
        "resampling": {
            "block_side": "factor × sqrt(2/sqrt(3)) × median same-family wall spacing on the resampled qualified cohort; triangular-cell-area approximation.",
            "block_factors": list(BLOCK_FACTORS),
            "spatial_origin": "Minimum saved wall midpoint coordinates over all geometry-selected cases; crop-relative row/column coordinates are translated by the frozen region origin and converted using the published 0.5 nm raster spacing. Four half-block origin phases are pooled.",
            "bootstrap_draws_per_origin": args.draws,
            "seed": args.seed,
            "interval": "P5–P95 percentile ranges, conditional spatial sensitivity within the fixed scan geometry.",
            "cluster_rule": "Sample the occupied blocks of the width-qualified cohort with replacement, drawing the same number of occupied blocks. Include every qualified wall in each selected block, preserving case pairing and duplicate block draws.",
        },
        "summary_validation": checks,
        "scan_conditions": conditions,
        "s1-5_common_eight_section_window_comparison": {
            "definition": "Intersect per-window width qualification and paired opposite-triplet readout availability across adaptive and fixed half-windows; use the same eight saved wall chords at both windows.",
            "walls": 8,
            "mean_point_checks_against_v004": common8_means,
            "paired_window_spatial_sensitivity": paired_window_sensitivity,
        },
        "data_status": {
            "s1-6": "The v004 protocol describes this as a fresh acquisition in the February series with the observer and aperture fixed before scoring; it remains in the same publication, instrument and collaboration, and the full scan was opened for geometry and width extraction.",
            "s1-10": "Previously examined central crop; only two qualifying width pairs were available there. The present full-field series is an extension of that same acquisition and does not constitute an independent scan.",
            "s1-5": "The warning protocol applies a threshold developed on MoS2 to this fresh graphene scan. The aperture and geometry gates were applied before inspecting the scan-level warning outcome; it is a retrospective within-study check with shared team and instrument.",
        },
        "yardstick_interpretation": {
            "existing_opposite_spot_metric": "Median absolute per-wall disagreement between two independent opposite-spot gap readouts on the matched walls.",
            "is_median_uncertainty": False,
            "recommendation": "Retain opposite-spot disagreement as an empirical within-wall repeatability scale; report it with its own paired-wall denominator. Use the spatial-block P5–P95 ranges for conditional uncertainty of scan-level medians. Neither quantity is a device-level or specimen-level confidence interval.",
        },
        "limits": [
            "The three acquisitions are separate scans but share publication, instrument and research group; no independent specimen crosswalk is available in the saved records.",
            "Spatial-block intervals condition on saved wall selection, aperture, fitted widths, and model predictions. They do not include aperture, probe, image registration, model or between-acquisition uncertainty.",
            "Block-size factors and origin phases expose sensitivity to spatial dependence assumptions; the saved records do not empirically identify a correlation length.",
            "s1-10 uses a full-field extension after a small crop was viewed. s1-5 is an additional retrospective warning test. These do not constitute blinded external validation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print("Wrote", args.output)
    for row in conditions:
        primary = row["primary_block_scale"]
        print(row["name"], "n=", row["qualified_paired_width_walls"])
        for key in ("measured_gap_nm", "periodic_minus_experiment_median_gap_nm", "isolated_minus_experiment_median_gap_nm"):
            print(" ", key, row["point"][key], primary["block_percentiles"][key])
        print(" ", "blocks", primary["support"]["occupied_blocks_by_origin"])


if __name__ == "__main__":
    main()
