"""Summarize paired wall scores and within-map spatial block sensitivity.

This reads a frozen run_mos2_wall_landscape.py result. Transfer errors are the
already-normalized Euclidean scores saved by the forward runner; this helper
does not transform them. Bootstrap intervals resample nearby wall midpoints
within each fixed map and are conditional on those maps and fitted settings.
"""

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


RELAXED = ("anchor", "barrier_075", "barrier_125", "shape_enaldiev")
MODELS = (*RELAXED, "rigid")
SPACING_TO_SQRT_CELL_AREA = math.sqrt(2.0 / math.sqrt(3.0))
ORIGIN_PHASES = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def pair(value):
    if not isinstance(value, list) or len(value) != 2:
        return (None, None)
    return tuple(finite_number(x) for x in value)


def mean(values):
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def quantile(values, p):
    vals = sorted(x for x in values if x is not None and math.isfinite(x))
    if not vals:
        return None
    point = (len(vals) - 1) * p
    low = int(math.floor(point))
    high = int(math.ceil(point))
    return vals[low] + (point - low) * (vals[high] - vals[low])


def bool_pair(value):
    if not isinstance(value, list) or len(value) != 2:
        return (False, False)
    return tuple(x is True for x in value)


def discover_conditions(scans):
    found = set()
    for scan in scans.values():
        for wall in scan.get("walls", []):
            if not wall.get("original_cohort"):
                continue
            for camera, probes in wall.get("readouts", {}).items():
                for probe, windows in probes.items():
                    for window in windows:
                        found.add((str(camera), str(probe), str(window)))
    def sort_key(row):
        camera, probe, window = row
        window_value = -1.0 if window == "adaptive" else float(window)
        return camera, float(probe), window_value
    return sorted(found, key=sort_key)


def wall_record(wall, wall_index, condition):
    camera, probe, window = condition
    readout = (wall.get("readouts", {}).get(camera, {}).get(probe, {}).get(window, {}))
    scores = {}
    widths = {}
    for model in MODELS:
        row = readout.get("models", {}).get(model, {})
        scores[model] = pair(row.get("transfer_error"))
        widths[model] = {
            "values": pair(row.get("widths_nm")),
            "qualified": bool_pair(row.get("qualified")),
        }
    measured = readout.get("measured", {})
    widths["measured"] = {
        "values": pair(measured.get("widths_nm")),
        "qualified": bool_pair(measured.get("qualified")),
    }
    xy = pair(wall.get("midpoint_xy_nm"))
    return {
        "index": wall_index,
        "xy": xy,
        "spacing_nm": finite_number(wall.get("spacing_nm")),
        "scores": scores,
        "widths": widths,
    }


def score_rows(scan, condition, model, require_all_models=False):
    result = []
    for i, wall in enumerate(scan.get("walls", [])):
        if not wall.get("original_cohort"):
            continue
        row = wall_record(wall, i, condition)
        names = ("anchor", model) if not require_all_models else MODELS
        if all(all(x is not None for x in row["scores"][name]) for name in names):
            result.append(row)
    return result


def per_map_score(records, model):
    vals = [(r["scores"][model][0] + r["scores"][model][1]) / 2 for r in records]
    directions = [r["scores"][model] for r in records]
    return {
        "combined": mean(vals),
        "first_to_second": mean(x[0] for x in directions),
        "second_to_first": mean(x[1] for x in directions),
        "walls": len(records),
    }


def condition_scores(scans, condition):
    model_map_values = {m: {} for m in MODELS}
    model_maps = {m: {} for m in MODELS}
    comparisons = {}
    coverage = {}
    for model in MODELS:
        candidate_per_scan = {}
        anchor_per_scan = {}
        wins = 0
        paired_walls = 0
        map_deltas = []
        for scan_name, scan in scans.items():
            rows = score_rows(scan, condition, model)
            if not rows:
                continue
            candidate = per_map_score(rows, model)
            anchor = per_map_score(rows, "anchor")
            model_map_values[model][scan_name] = candidate
            model_maps[model][scan_name] = candidate["combined"]
            if model != "anchor":
                candidate_per_scan[scan_name] = candidate["combined"]
                anchor_per_scan[scan_name] = anchor["combined"]
                delta = candidate["combined"] - anchor["combined"]
                map_deltas.append(delta)
                wins += sum(
                    (r["scores"][model][0] + r["scores"][model][1])
                    < (r["scores"]["anchor"][0] + r["scores"]["anchor"][1])
                    for r in rows
                )
                paired_walls += len(rows)
        if model != "anchor":
            cmean = mean(candidate_per_scan.values())
            amean = mean(anchor_per_scan.values())
            comparisons[model] = {
                "paired_equal_map_mean_error": cmean,
                "anchor_on_same_walls_equal_map_mean_error": amean,
                "delta_error_candidate_minus_anchor": None if cmean is None else cmean - amean,
                "gain_percent_vs_anchor": None if not amean else 100 * (1 - cmean / amean),
                "maps": len(candidate_per_scan),
                "map_win_count": sum(candidate_per_scan[s] < anchor_per_scan[s] for s in candidate_per_scan),
                "paired_original_walls": paired_walls,
                "wall_win_fraction_pooled": None if not paired_walls else wins / paired_walls,
                "map_delta_min": min(map_deltas) if map_deltas else None,
                "map_delta_max": max(map_deltas) if map_deltas else None,
            }
    for scan_name, scan in scans.items():
        walls = [w for w in scan.get("walls", []) if w.get("original_cohort")]
        if not walls:
            continue
        all_rows = [wall_record(w, i, condition) for i, w in enumerate(scan.get("walls", [])) if w.get("original_cohort")]
        coverage[scan_name] = {
            "original_walls": len(walls),
            "finite_all_model_score_walls": sum(
                all(all(x is not None for x in r["scores"][m]) for m in MODELS) for r in all_rows
            ),
        }
    return model_map_values, comparisons, coverage


def width_summary(scans, condition):
    metrics = ("ring1", "ring2", "signed_gap_ring1_minus_ring2")
    by_model = {m: {key: {} for key in metrics} for m in (*RELAXED, "measured")}
    rigid_subset = {m: {key: {} for key in metrics} for m in (*MODELS, "measured")}
    qualifications = {m: {"ring1": 0, "ring2": 0, "both": 0} for m in (*MODELS, "measured")}
    common_per_map = {}
    rigid_per_map = {}
    common_records = {}
    for scan_name, scan in scans.items():
        rows = []
        for i, wall in enumerate(scan.get("walls", [])):
            if not wall.get("original_cohort"):
                continue
            rec = wall_record(wall, i, condition)
            for model in (*MODELS, "measured"):
                q = rec["widths"][model]["qualified"]
                qualifications[model]["ring1"] += int(q[0])
                qualifications[model]["ring2"] += int(q[1])
                qualifications[model]["both"] += int(q[0] and q[1])
            if all(all(rec["widths"][m]["qualified"]) for m in (*RELAXED, "measured")):
                rows.append(rec)
        common_records[scan_name] = rows
        common_per_map[scan_name] = len(rows)
        rigid_rows = [r for r in rows if all(r["widths"]["rigid"]["qualified"])]
        rigid_per_map[scan_name] = len(rigid_rows)
        for model in (*RELAXED, "measured"):
            for ring in (0, 1):
                value = mean(r["widths"][model]["values"][ring] for r in rows)
                if value is not None:
                    by_model[model][f"ring{ring+1}"][scan_name] = value
            gap = mean(r["widths"][model]["values"][0] - r["widths"][model]["values"][1]
                       for r in rows if None not in r["widths"][model]["values"])
            if gap is not None:
                by_model[model]["signed_gap_ring1_minus_ring2"][scan_name] = gap
        for model in (*MODELS, "measured"):
            for ring in (0, 1):
                value = mean(r["widths"][model]["values"][ring] for r in rigid_rows)
                if value is not None:
                    rigid_subset[model][f"ring{ring+1}"][scan_name] = value
            gap = mean(r["widths"][model]["values"][0] - r["widths"][model]["values"][1]
                       for r in rigid_rows if None not in r["widths"][model]["values"])
            if gap is not None:
                rigid_subset[model]["signed_gap_ring1_minus_ring2"][scan_name] = gap
    equal_map = {
        model: {metric: mean(values.values()) for metric, values in data.items()}
        for model, data in by_model.items()
    }
    rigid_equal_map = {
        model: {metric: mean(values.values()) for metric, values in data.items()}
        for model, data in rigid_subset.items()
    }
    n_common_maps = sum(n > 0 for n in common_per_map.values())
    n_rigid_maps = sum(n > 0 for n in rigid_per_map.values())
    return {
        "qualification_counts_over_v002_walls": qualifications,
        "common_both_ring_width_cohort": {
            "definition": "Measured plus anchor, barrier_075, barrier_125 and shape_enaldiev must each qualify in both rings; rigid is summarized separately.",
            "walls_by_map": common_per_map,
            "walls_total": sum(common_per_map.values()),
            "maps_with_common_walls": n_common_maps,
            "equal_map_means_nm": equal_map,
            "rigid_both_ring_qualified_subset": {
                "definition": "Subset of the common measured-plus-four-relaxed width cohort that also has an interior, qualified rigid fit in both rings.",
                "walls_by_map": rigid_per_map,
                "walls_total": sum(rigid_per_map.values()),
                "maps_with_walls": n_rigid_maps,
                "equal_map_means_nm": rigid_equal_map,
            },
        },
    }


def combine_condition(scans, condition):
    map_values, comparisons, coverage = condition_scores(scans, condition)
    model_means = {}
    for model in MODELS:
        model_means[model] = {
            metric: mean(row[metric] for row in map_values[model].values())
            for metric in ("combined", "first_to_second", "second_to_first")
        }
        model_means[model]["maps"] = len(map_values[model])
    ranked = sorted(
        ((model_means[m]["combined"], m) for m in MODELS if model_means[m]["combined"] is not None)
    )
    return {
        "camera": condition[0], "probe_fwhm_nm": float(condition[1]),
        "window_half_width_nm": condition[2],
        "equal_map_mean_transfer_error": model_means,
        "paired_vs_anchor": comparisons,
        "rank_by_combined_transfer_error": [name for _, name in ranked],
        "score_coverage": coverage,
        "widths": width_summary(scans, condition),
    }


def same_wall_across_conditions(scans, conditions):
    common = {}
    for scan_name, scan in scans.items():
        eligible = []
        for i, wall in enumerate(scan.get("walls", [])):
            if not wall.get("original_cohort"):
                continue
            good = True
            for condition in conditions:
                row = wall_record(wall, i, condition)
                if any(any(x is None for x in row["scores"][m]) for m in MODELS):
                    good = False
                    break
            if good:
                eligible.append(i)
        common[scan_name] = eligible
    by_condition = []
    for condition in conditions:
        models = {m: [] for m in MODELS}
        wins = {m: 0 for m in MODELS if m != "anchor"}
        for scan_name, scan in scans.items():
            rows = [wall_record(scan["walls"][i], i, condition) for i in common[scan_name]]
            if not rows:
                continue
            for m in MODELS:
                models[m].append(per_map_score(rows, m)["combined"])
            for m in wins:
                wins[m] += sum(
                    sum(r["scores"][m]) < sum(r["scores"]["anchor"]) for r in rows
                )
        means = {m: mean(vals) for m, vals in models.items()}
        contrasts = {}
        for m in wins:
            a, b = means["anchor"], means[m]
            contrasts[m] = {
                "delta_error_candidate_minus_anchor": None if a is None or b is None else b - a,
                "gain_percent_vs_anchor": None if not a else 100 * (1 - b / a),
                "map_win_count": sum(models[m][i] < models["anchor"][i] for i in range(min(len(models[m]), len(models["anchor"])))),
                "wall_win_count": wins[m],
            }
        by_condition.append({
            "camera": condition[0], "probe_fwhm_nm": float(condition[1]), "window_half_width_nm": condition[2],
            "equal_map_mean_transfer_error": means, "paired_vs_anchor": contrasts,
        })
    return {
        "definition": "Original-cohort wall midpoints with finite two-direction scores for anchor and every candidate in every available camera/probe/window condition; identical walls are used within each map for all settings.",
        "common_wall_counts_by_map": {name: len(ix) for name, ix in common.items()},
        "total_walls": sum(map(len, common.values())),
        "maps_with_common_walls": sum(bool(ix) for ix in common.values()),
        "condition_sensitivity": by_condition,
    }


def blocks_for(records, model, side, phase):
    valid = []
    for row in records:
        xy = row["xy"]
        if any(x is None for x in xy):
            continue
        anchor = row["scores"]["anchor"]
        candidate = row["scores"][model]
        if any(x is None for x in anchor + candidate):
            continue
        valid.append((row, xy))
    if not valid:
        return []
    xmin = min(xy[0] for _, xy in valid)
    ymin = min(xy[1] for _, xy in valid)
    grouped = defaultdict(list)
    dx, dy = phase
    for row, xy in valid:
        key = (math.floor((xy[0] - xmin) / side + dx), math.floor((xy[1] - ymin) / side + dy))
        grouped[key].append(row)
    result = []
    for rows in grouped.values():
        anchor_vals = [(r["scores"]["anchor"][0] + r["scores"]["anchor"][1]) / 2 for r in rows]
        model_vals = [(r["scores"][model][0] + r["scores"][model][1]) / 2 for r in rows]
        result.append((sum(anchor_vals), sum(model_vals), len(rows)))
    return result


def bootstrap_model(scans, condition, model, factor, draws, seed):
    rng = random.Random(seed)
    coverage = {}
    phase_parts = []
    per_phase = []
    for phase in ORIGIN_PHASES:
        maps = {}
        for scan_name, scan in scans.items():
            records = score_rows(scan, condition, model)
            if not records:
                continue
            spacings = [r["spacing_nm"] for r in records if r["spacing_nm"] is not None]
            if not spacings:
                continue
            side = factor * SPACING_TO_SQRT_CELL_AREA * statistics.median(spacings)
            blocks = blocks_for(records, model, side, phase)
            if blocks:
                maps[scan_name] = blocks
        phase_parts.append(maps)
        per_phase.append({name: len(blocks) for name, blocks in maps.items()})
    all_counts = defaultdict(list)
    for summary in per_phase:
        for name, n in summary.items():
            all_counts[name].append(n)
    for scan_name, scan in scans.items():
        rows = score_rows(scan, condition, model)
        if rows:
            spacings = [r["spacing_nm"] for r in rows if r["spacing_nm"] is not None]
            if spacings:
                side = factor * SPACING_TO_SQRT_CELL_AREA * statistics.median(spacings)
                coverage[scan_name] = {"side_nm": side, "blocks_by_origin": all_counts.get(scan_name, [])}
    samples = {"anchor": [], "candidate": [], "delta": [], "gain_percent": []}
    for maps in phase_parts:
        for _ in range(draws):
            map_a, map_m = [], []
            for blocks in maps.values():
                chosen = [blocks[rng.randrange(len(blocks))] for _ in range(len(blocks))]
                total_n = sum(b[2] for b in chosen)
                if total_n:
                    map_a.append(sum(b[0] for b in chosen) / total_n)
                    map_m.append(sum(b[1] for b in chosen) / total_n)
            a, m = mean(map_a), mean(map_m)
            if a is None or m is None:
                continue
            samples["anchor"].append(a)
            samples["candidate"].append(m)
            samples["delta"].append(m - a)
            samples["gain_percent"].append(100 * (1 - m / a) if a else None)
    return {
        "block_side_factor_of_sqrt_cell_area": factor,
        "block_side_definition": "factor × sqrt(2/sqrt(3)) × scan median same-family wall-row spacing; triangular-cell area approximation.",
        "map_block_support": coverage,
        "blocks_fewer_than_three_at_any_origin": sorted(
            name for name, counts in all_counts.items() if counts and min(counts) < 3
        ),
        "percentile_ranges_5_50_95": {
            key: [quantile(values, p) for p in (0.05, 0.50, 0.95)]
            for key, values in samples.items()
        },
        "probability_candidate_lower_error": mean([1.0 if d < 0 else 0.0 for d in samples["delta"]]),
        "bootstrap_draws_per_origin": draws,
        "origins": len(ORIGIN_PHASES),
    }


def spatial_sensitivity(scans, condition, draws, seed):
    result = {}
    # Evaluate every model at three triangular-cell block scales. The anchor
    # row is the paired baseline on each model's finite common walls.
    for factor_index, factor in enumerate((0.5, 1.0, 1.5)):
        result[str(factor)] = {}
        for model_index, model in enumerate(m for m in MODELS if m != "anchor"):
            result[str(factor)][model] = bootstrap_model(
                scans, condition, model, factor, draws,
                seed + factor_index * 1009 + model_index * 9176,
            )
    return {
        "condition": {"camera": condition[0], "probe_fwhm_nm": float(condition[1]), "window_half_width_nm": condition[2]},
        "scope": "Resample square clusters of wall midpoint scores within each fixed scan, pairing anchor and candidate score on each resampled wall; average map means with equal scan weights. Four half-block origin phases are combined.",
        "blocks": result,
        "limits": "Percentile ranges describe conditional within-map spatial sensitivity with fitted registration and optical calibration fixed. Scans may share specimens/devices; no device crosswalk is available, so these are not device-level confidence intervals and do not include acquisition, camera, probe or model uncertainty.",
    }


def short_window_summary(scans):
    probes = set()
    for scan in scans.values():
        for wall in scan.get("walls", []):
            if wall.get("original_cohort"):
                probes.update(wall.get("short_window_anchor_read_as_1.25_nm", {}).keys())
    fits = {}
    for probe in sorted(probes, key=float):
        map_values = {"ring1": {}, "ring2": {}, "ring1_error_vs_intrinsic": {}, "ring2_error_vs_intrinsic": {}}
        quals = {"ring1": 0, "ring2": 0, "both": 0}
        wall_count = 0
        for scan_name, scan in scans.items():
            wall_data = []
            for wall in scan.get("walls", []):
                if not wall.get("original_cohort"):
                    continue
                fit = wall.get("short_window_anchor_read_as_1.25_nm", {}).get(probe, {})
                widths = pair(fit.get("widths_nm"))
                q = bool_pair(fit.get("qualified"))
                intrinsic = finite_number(wall.get("model_core_widths_nm", {}).get("anchor"))
                quals["ring1"] += int(q[0]); quals["ring2"] += int(q[1]); quals["both"] += int(q[0] and q[1])
                if all(x is not None for x in widths) and intrinsic is not None:
                    wall_data.append((widths, intrinsic))
            if wall_data:
                map_values["ring1"][scan_name] = mean(x[0][0] for x in wall_data)
                map_values["ring2"][scan_name] = mean(x[0][1] for x in wall_data)
                map_values["ring1_error_vs_intrinsic"][scan_name] = mean(x[0][0] - x[1] for x in wall_data)
                map_values["ring2_error_vs_intrinsic"][scan_name] = mean(x[0][1] - x[1] for x in wall_data)
                wall_count += len(wall_data)
        fits[probe] = {
            "equal_map_mean_nm": {key: mean(values.values()) for key, values in map_values.items()},
            "maps": len(map_values["ring1"]), "qualified_v002_wall_count": quals,
            "paired_intrinsic_width_walls": wall_count,
        }
    return {
        "synthetic_anchor_render": {
            "observer_window_half_width_nm": 2.5,
            "observer_probe_fwhm_nm": 1.25,
            "true_rendered_probe_fwhm_nm": fits,
            "interpretation": "These are noiseless forward-rendered anchor profiles measured with the short-window observer, compared with the saved intrinsic midpoint slip-gradient FWHM. They isolate window/probe response; they are not repeated experimental measurements or a barrier recovery test.",
        }
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=Path("mos2_wall_landscape_v01.json"))
    parser.add_argument("output", type=Path, nargs="?", default=Path("mos2_wall_landscape_uncertainty_v01.json"))
    parser.add_argument("--draws", type=int, default=1000, help="Block bootstrap draws for each tile-origin phase.")
    parser.add_argument("--seed", type=int, default=642117, help="Deterministic bootstrap seed.")
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    scans = data["scans"]
    conditions = discover_conditions(scans)
    primary = ("compact", "1.25", "4.25")
    if primary not in conditions:
        raise ValueError(f"Expected primary condition {primary}; found {conditions}")
    condition_results = [combine_condition(scans, condition) for condition in conditions]
    summary = {
        "source": str(args.input.resolve()),
        "source_sha256": digest(args.input),
        "helper_sha256": digest(Path(__file__).resolve()),
        "score_definition": "transfer_error is the saved Euclidean norm of the selected per-reflection detrended unit shapes, with ring channel-count normalization, after fitting position in the source direction. Values are used directly; they are dimensionless.",
        "population": {
            "maps": len(scans),
            "original_cohort_walls": sum(bool(w.get("original_cohort")) for s in scans.values() for w in s.get("walls", [])),
            "all_geometry_walls": sum(len(s.get("walls", [])) for s in scans.values()),
            "primary_condition": {"camera": primary[0], "probe_fwhm_nm": 1.25, "window_half_width_nm": 4.25},
            "v002_width_cohort": "Width summaries use measured and all four relaxed candidates qualified in both rings. Rigid qualification is counted separately. Full-profile score coverage keeps every finite original-cohort wall regardless of width qualification.",
        },
        "conditions": condition_results,
        "same_wall_across_settings": same_wall_across_conditions(scans, conditions),
        "primary_spatial_block_sensitivity": spatial_sensitivity(scans, primary, args.draws, args.seed),
        "short_window_response": short_window_summary(scans),
        "limits": [
            "All ten maps were previously examined; map-level means do not create an independent specimen sample.",
            "The device crosswalk is unavailable. Spatial intervals are conditional within-map resampling ranges, not device-level confidence intervals.",
            "Block side is based on the isotropic triangular moire-cell area approximation from each map's median same-family wall-row spacing; it is a sensitivity scale, not a measured spatial correlation length.",
            "Camera, probe and window checks are sensitivity analyses on the same walls; they are not independent replications.",
        ],
    }
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.output} from {args.input}; {len(conditions)} conditions, {summary['population']['original_cohort_walls']} original-cohort walls")


if __name__ == "__main__":
    main()
