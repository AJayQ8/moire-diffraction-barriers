"""Plot matched experimental wall widths and the probe-dependent model ranking."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

import matplotlib.pyplot as plt
import numpy as np


FIGURES = Path(__file__).resolve().parent
EVIDENCE = FIGURES.parents[1] / "evidence"
MODELS = ("anchor", "barrier_075", "barrier_125", "shape_enaldiev")
WINDOWS = ("2.50", "4.25", "adaptive")
BLUE = "#245e8b"
ORANGE = "#a64d31"
GRAY = "#687582"
WIDTH = 17.7 / 2.54
NOMINAL_WINDOW = "4.25"
COMMON_WINDOWS = ("2.50", "4.25", "adaptive")
SPACING_TO_SQRT_CELL_AREA = math.sqrt(2.0 / math.sqrt(3.0))
ORIGIN_PHASES = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))
BLOCK_FACTORS = (0.5, 1.0, 1.5)
LOW_MAPS = ("DS1", "DS2", "DS14", "DS13")
HIGH_MAPS = ("DS12", "DS9", "DS10", "DS6", "DS4", "DS3")


def article_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.edgecolor": "#82909a",
        "axes.labelcolor": "#273640",
        "axes.titlecolor": "#273640",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.color": "#273640",
        "ytick.color": "#273640",
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })


def map_rows(scan):
    return [row["readouts"]["compact"]["1.25"] for row in scan["walls"] if row["original_cohort"]]


def qualified(row):
    return all(row["measured"]["qualified"]) and all(
        all(row["models"][model]["qualified"]) for model in MODELS
    )


def map_widths(rows):
    return np.mean([row["measured"]["widths_nm"] for row in rows], axis=0), np.mean(
        [row["models"]["anchor"]["widths_nm"] for row in rows], axis=0
    )


def qualified_wall_rows(scan, window):
    """Return matched measured/relaxed width pairs for one map and window."""
    rows = []
    for wall in scan["walls"]:
        if not wall["original_cohort"]:
            continue
        condition = wall["readouts"]["compact"]["1.25"].get(window)
        if condition is not None and qualified(condition):
            rows.append((wall, condition))
    return rows


def percentile_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
    }


def map_blocks(records, origin_reference, side_nm, phase):
    xy = np.asarray([wall["midpoint_xy_nm"] for wall, _ in records], dtype=float)
    origin_xy = np.asarray([wall["midpoint_xy_nm"] for wall, _ in origin_reference], dtype=float)
    xmin, ymin = np.min(origin_xy, axis=0)
    dx, dy = phase
    grouped = {}
    for index, (x, y) in enumerate(xy):
        key = (math.floor((x - xmin) / side_nm + dx),
               math.floor((y - ymin) / side_nm + dy))
        grouped.setdefault(key, []).append(index)
    return list(grouped.values())


def bootstrap_one_map(records, spacing_reference, factor, draws, rng):
    """Paired block resampling of both measured and anchor widths in one map."""
    measured = np.asarray([row["measured"]["widths_nm"] for _, row in records], dtype=float)
    anchor = np.asarray([row["models"]["anchor"]["widths_nm"] for _, row in records], dtype=float)
    spacings = [float(wall["spacing_nm"]) for wall, _ in spacing_reference]
    side_nm = factor * SPACING_TO_SQRT_CELL_AREA * statistics.median(spacings)
    per_origin = []
    block_counts = []
    for phase in ORIGIN_PHASES:
        blocks = map_blocks(records, spacing_reference, side_nm, phase)
        block_counts.append(len(blocks))
        block_measured = np.asarray([measured[ix].sum(axis=0) for ix in blocks])
        block_anchor = np.asarray([anchor[ix].sum(axis=0) for ix in blocks])
        block_sizes = np.asarray([len(ix) for ix in blocks], dtype=float)
        draw_counts = rng.integers(0, len(blocks), size=(draws, len(blocks)))
        selected = np.eye(len(blocks), dtype=float)[draw_counts].sum(axis=1)
        total_walls = selected @ block_sizes
        sampled_measured = (selected @ block_measured) / total_walls[:, None]
        sampled_anchor = (selected @ block_anchor) / total_walls[:, None]
        # Absolute residual of each map-mean width, averaged equally over orders.
        map_abs_error = np.mean(np.abs(sampled_measured - sampled_anchor), axis=1)
        per_origin.append({
            "measured": sampled_measured,
            "anchor": sampled_anchor,
            "map_abs_error": map_abs_error,
        })
    observed_measured = measured.mean(axis=0)
    observed_anchor = anchor.mean(axis=0)
    observed_error = float(np.mean(np.abs(observed_measured - observed_anchor)))
    bootstrap = {
        name: np.concatenate([origin[name] for origin in per_origin], axis=0)
        for name in ("measured", "anchor", "map_abs_error")
    }
    support = {
        "side_nm": float(side_nm),
        "wall_pairs": len(records),
        "geometry_walls_for_block_scale": len(spacing_reference),
        "median_spacing_for_block_scale_nm": float(statistics.median(spacings)),
        "blocks_by_origin": block_counts,
        "minimum_occupied_blocks": min(block_counts),
        "single_block_origins": sum(count == 1 for count in block_counts),
        "one_block_at_every_origin": all(count == 1 for count in block_counts),
    }
    summary = {
        "measured_width_nm": {
            f"order{order}": percentile_summary(bootstrap["measured"][:, order - 1])
            for order in (1, 2)
        },
        "anchor_width_nm": {
            f"order{order}": percentile_summary(bootstrap["anchor"][:, order - 1])
            for order in (1, 2)
        },
        "paired_absolute_width_error_nm": percentile_summary(bootstrap["map_abs_error"]),
    }
    return {
        "point": {
            "measured_width_nm": observed_measured.tolist(),
            "anchor_width_nm": observed_anchor.tolist(),
            "paired_absolute_width_error_nm": observed_error,
        },
        "bootstrap": bootstrap,
        "percentiles": summary,
        "support": support,
    }


def spatial_width_uncertainty(data, raw_source, archive_path, draws=1000, seed=20261115):
    """Compute equal-map summaries with paired spatial block draws per map."""
    ordered = sorted(data["scans"].items(), key=lambda item: item[1]["inferred_twist_deg"])
    map_names = [name for name, _ in ordered]
    map_angles = {name: float(scan["inferred_twist_deg"]) for name, scan in ordered}
    map_rows_by_name = {
        name: qualified_wall_rows(scan, NOMINAL_WINDOW) for name, scan in ordered
    }
    geometry_rows_by_name = {
        name: [(wall, None) for wall in scan["walls"] if wall["original_cohort"]]
        for name, scan in ordered
    }
    geometry_counts = {name: len(rows) for name, rows in geometry_rows_by_name.items()}
    primary_counts = {name: len(rows) for name, rows in map_rows_by_name.items()}
    common_counts = {}
    for name, scan in ordered:
        count = 0
        for wall in scan["walls"]:
            if not wall["original_cohort"]:
                continue
            readouts = wall["readouts"]["compact"]["1.25"]
            if all(key in readouts and qualified(readouts[key]) for key in COMMON_WINDOWS):
                count += 1
        common_counts[name] = count

    factor_results = {}
    for factor_index, factor in enumerate(BLOCK_FACTORS):
        rng = np.random.default_rng(seed + factor_index * 1009)
        map_results = {}
        for name in map_names:
            map_results[name] = bootstrap_one_map(
                map_rows_by_name[name], geometry_rows_by_name[name], factor, draws, rng
            )
        group_samples = {}
        for group, names in (("low_four", LOW_MAPS), ("high_six", HIGH_MAPS)):
            group_samples[group] = np.mean(
                np.stack([map_results[name]["bootstrap"]["map_abs_error"] for name in names], axis=1),
                axis=1,
            )
        group_samples["low_minus_high"] = group_samples["low_four"] - group_samples["high_six"]
        point_groups = {
            group: float(np.mean([
                map_results[name]["point"]["paired_absolute_width_error_nm"] for name in names
            ]))
            for group, names in (("low_four", LOW_MAPS), ("high_six", HIGH_MAPS))
        }
        point_groups["low_minus_high"] = point_groups["low_four"] - point_groups["high_six"]
        factor_results[str(factor)] = {
            "map_results": {
                name: {
                    "point": result["point"],
                    "percentiles": result["percentiles"],
                    "support": result["support"],
                    "interval_plotted": result["support"]["minimum_occupied_blocks"] >= 2,
                }
                for name, result in map_results.items()
            },
            "groups": {
                group: {
                    "maps": list(LOW_MAPS if group == "low_four" else HIGH_MAPS if group == "high_six" else []),
                    "point_nm": point_groups[group],
                    "block_percentiles_nm": percentile_summary(group_samples[group]),
                }
                for group in ("low_four", "high_six", "low_minus_high")
            },
        }

    nominal_total = sum(primary_counts.values())
    common_total = sum(common_counts.values())
    if (nominal_total != 239 or common_total != 175 or len(map_names) != 10
            or sum(geometry_counts.values()) != 243
            or tuple(map_names[:4]) != LOW_MAPS or tuple(map_names[4:]) != HIGH_MAPS):
        raise ValueError(
            f"Unexpected Fig. 2 cohorts: {nominal_total} nominal, {common_total} common, "
            f"{len(map_names)} maps, {sum(geometry_counts.values())} geometry walls"
        )
    primary = factor_results["1.0"]
    low_point = primary["groups"]["low_four"]["point_nm"]
    high_point = primary["groups"]["high_six"]["point_nm"]
    if not np.isclose(low_point, 0.430218939, atol=5e-10) or not np.isclose(high_point, 0.050250295, atol=5e-10):
        raise ValueError(f"Fig. 2 paired absolute-width errors changed: {low_point}, {high_point}")

    source_bytes = raw_source["bytes"]
    prior_uncertainty_path = EVIDENCE / "mos2_wall_landscape_uncertainty_v01.json"
    prior_uncertainty_hash = sha256_file(prior_uncertainty_path) if prior_uncertainty_path.exists() else None
    archive_used = raw_source.get("member") is not None
    archive_hash = sha256_file(archive_path) if archive_used and archive_path.exists() else None
    return {
        "schema": "mos2_map_width_uncertainty_v004",
        "source": {
            "record": archive_path.name if archive_used else Path(raw_source["description"]).name,
            "member": raw_source.get("member"),
            "raw_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "archive_path": archive_path.name if archive_used else None,
            "archive_sha256": archive_hash,
            "prior_uncertainty_record": "evidence/mos2_wall_landscape_uncertainty_v01.json",
            "prior_uncertainty_sha256": prior_uncertainty_hash,
            "plot_script_sha256": sha256_file(Path(__file__)),
        },
        "score_definition": (
            "Within each map, average measured and anchor widths separately across the matched qualified wall pairs "
            "for each diffraction order. Paired absolute-width error is the mean absolute difference between those "
            "two map-mean widths, averaged equally over orders. Group summaries give each map equal weight."
        ),
        "cohorts": {
            "nominal_fixed_window": {
                "camera": "compact", "probe_fwhm_nm": 1.25, "window_half_width_nm": NOMINAL_WINDOW,
                "walls": nominal_total, "maps": len(map_names), "counts_by_map": primary_counts,
                "geometry_walls_for_block_scale_by_map": geometry_counts,
                "definition": "Original-cohort sections with measured and all four relaxed model width pairs qualified in both orders.",
            },
            "all_three_windows": {
                "camera": "compact", "probe_fwhm_nm": 1.25,
                "windows_half_width_nm": list(COMMON_WINDOWS), "walls": common_total,
                "maps": len(map_names), "counts_by_map": common_counts,
                "definition": "Same eligibility rule intersected across 2.50, 4.25 and adaptive half-windows; used in Fig. 2b.",
            },
        },
        "resampling": {
            "primary_block_factor": 1.0,
            "sensitivity_block_factors": list(BLOCK_FACTORS),
            "block_side_definition": "factor × sqrt(2/sqrt(3)) × map median spacing_nm; triangular-cell-area approximation.",
            "block_side_spacing_support": "All original-cohort wall geometries per map, matching the released 243-wall score bootstrap; width values are resampled only on the qualified 239-pair cohort.",
            "origin_phases": [list(x) for x in ORIGIN_PHASES],
            "draws_per_origin_per_map": draws,
            "seed": seed,
            "draw_rule": "Sample the occupied spatial blocks with replacement, drawing the same number of blocks as occupied; include all wall pairs in each selected block.",
            "pairing": "Each resampled block jointly carries measured and anchor widths in both orders; maps are held fixed and averaged equally.",
            "interval": "P5–P95 percentile range (90% conditional within-map spatial block resampling range).",
            "support_note": "Map bars are omitted when any origin phase has fewer than two occupied blocks; dagger marks that limited support. One-block phases contribute no within-map spread to the group bootstrap.",
        },
        "group_contrast": {
            "low_four_maps": list(LOW_MAPS), "high_six_maps": list(HIGH_MAPS),
            "low_minus_high_definition": "Equal-map mean of per-map paired absolute-width errors in low four maps minus the equal-map mean in high six maps.",
            "primary_block_scale": primary["groups"],
            "block_scale_sensitivity": {
                factor: result["groups"] for factor, result in factor_results.items()
            },
        },
        "maps": {
            name: {
                "inferred_twist_deg": map_angles[name],
                "nominal_wall_pairs": primary_counts[name],
                "common_three_window_pairs": common_counts[name],
                "primary": primary["map_results"][name],
                "block_scale_sensitivity": {
                    factor: result["map_results"][name]
                    for factor, result in factor_results.items() if factor != "1.0"
                },
            }
            for name in map_names
        },
        "limits": [
            "Intervals describe conditional spatial sensitivity within these ten previously examined maps with fitted physics and optical responses fixed; maps are not resampled.",
            "No independent device crosswalk is available, so these ranges are not device-level confidence intervals and do not include acquisition, detector, probe or model uncertainty.",
            "At the primary block scale DS1 has one occupied block at all four origins; its group-bootstrap contribution is fixed, and its spatial range cannot be estimated.",
        ],
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_landscape(evidence, archive, explicit_path=None):
    if explicit_path is not None:
        raw_bytes = explicit_path.read_bytes()
        return json.loads(raw_bytes), {"bytes": raw_bytes, "description": str(explicit_path)}
    evidence_path = evidence / "mos2_wall_landscape_v01.json"
    if evidence_path.exists():
        raw_bytes = evidence_path.read_bytes()
        return json.loads(raw_bytes), {"bytes": raw_bytes, "description": str(evidence_path)}
    if archive.exists():
        member = "evidence/mos2_wall_landscape_v01.json"
        with zipfile.ZipFile(archive) as source_zip:
            raw_bytes = source_zip.read(member)
        return json.loads(raw_bytes), {
            "bytes": raw_bytes,
            "description": str(archive),
            "member": member,
        }
    raise FileNotFoundError(f"No landscape JSON at {evidence_path} or in {archive}")


def scores(data):
    ordered = sorted(data["scans"].values(), key=lambda row: row["inferred_twist_deg"])
    angles, nominal, gaps = [], [], []
    nominal_count, window_count = 0, 0
    for scan in ordered:
        rows = map_rows(scan)
        standard = [row["4.25"] for row in rows if "4.25" in row and qualified(row["4.25"])]
        matched = [row for row in rows if all(key in row and qualified(row[key]) for key in WINDOWS)]
        angles.append(scan["inferred_twist_deg"])
        nominal.append(map_widths(standard))
        gaps.append([[(widths[0] - widths[1]) for widths in map_widths([row[key] for row in matched])]
                     for key in WINDOWS])
        nominal_count += len(standard)
        window_count += len(matched)
    return np.asarray(angles), np.asarray(nominal), np.asarray(gaps), nominal_count, window_count


def render_widths(data, output, uncertainty, uncertainty_output):
    angles, nominal, gaps, first_n, common_n = scores(data)
    uncertainty_output.parent.mkdir(parents=True, exist_ok=True)
    uncertainty_output.write_text(json.dumps(uncertainty, indent=2) + "\n")
    fig, (left, right) = plt.subplots(ncols=2, figsize=(WIDTH, 2.48), layout="constrained")
    left.set_title("(a) Absolute fitted widths", loc="left")
    names = [name for name, _ in sorted(data["scans"].items(), key=lambda item: item[1]["inferred_twist_deg"])]
    primary_maps = uncertainty["group_contrast"]["primary_block_scale"]
    map_results = uncertainty["maps"]
    for i, (order, color) in enumerate((("First order", BLUE), ("Second order", ORANGE))):
        for series in ("measured_width_nm", "anchor_width_nm"):
            eligible = [j for j, name in enumerate(names) if map_results[name]["primary"]["interval_plotted"]]
            intervals = [
                map_results[names[j]]["primary"]["percentiles"][series][f"order{i + 1}"]
                for j in eligible
            ]
            x = angles[eligible]
            lower = np.asarray([item["p05"] for item in intervals])
            upper = np.asarray([item["p95"] for item in intervals])
            left.vlines(x, lower, upper, color=color, linewidth=.65, alpha=.72, zorder=1)
            left.hlines(lower, x - .009, x + .009, color=color, linewidth=.65, alpha=.72, zorder=1)
            left.hlines(upper, x - .009, x + .009, color=color, linewidth=.65, alpha=.72, zorder=1)
        left.scatter(angles, nominal[:, 0, i], s=16, color=color, label=f"{order}, measured")
        left.scatter(angles, nominal[:, 1, i], s=16, facecolors="white", edgecolors=color,
                     linewidth=.85, label=f"{order}, periodic")
    for j, name in enumerate(names):
        support = map_results[name]["primary"]["support"]
        if support["minimum_occupied_blocks"] < 2:
            left.annotate("†", (angles[j], min(5.58, float(np.max(nominal[j])) + .11)),
                          ha="center", va="bottom", fontsize=7, color="#354550")
    left.text(.02, .025, "Bars: P5–P95 spatial-block range\n† one-block origin: no map interval",
              transform=left.transAxes, ha="left", va="bottom", fontsize=6.0, color="#43535f",
              bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86, "pad": 1.8})
    left.set(xlabel="Twist from measured geometry (°)", ylabel="Map mean fitted width (nm)", ylim=(1.85, 5.7))
    left.legend(fontsize=6.7, loc="upper left", handletextpad=.25, labelspacing=.24, borderaxespad=.30,
                columnspacing=.65, ncols=2)

    right.set_title("(b) Width difference across fit windows", loc="left")
    for idx, (label, color, mark) in enumerate((("Measured", BLUE, "o"), ("Periodic", ORANGE, "s"))):
        for row in gaps[:, :, idx]:
            right.plot(range(3), row, color=color, alpha=.21, lw=.65)
        right.plot(range(3), gaps[:, :, idx].mean(axis=0), color=color, linewidth=1.5, marker=mark,
                   markersize=4, label=f"{label}, map mean")
    right.set(xlabel="Profile half-window", ylabel="First minus second width (nm)",
              xticks=[0, 1, 2], xticklabels=["2.50 nm", "4.25 nm", "Adaptive"], xlim=(-.1, 2.1))
    right.legend(fontsize=6.7, loc="upper left", borderaxespad=.3)
    for ax in (left, right):
        ax.grid(axis="y", color="#dce4e9", lw=.6)
        ax.set_axisbelow(True)
    fig.savefig(output)
    plt.close(fig)
    print(f"{len(angles)} maps, {first_n} absolute-width sections, {common_n} shared across windows")
    print("Window mean gaps (measured / periodic, nm):", np.round(gaps.mean(axis=0), 4))
    for key in ("low_four", "high_six", "low_minus_high"):
        print(key, primary_maps[key])


def render_crossover(data, output):
    extension = data["extensions"]
    comparison = extension["width_comparisons"]["1.9"]
    fig, axes = plt.subplots(ncols=2, figsize=(WIDTH, 2.48), layout="constrained")
    labels = (("13.2 meV", BLUE), ("18.87 meV", ORANGE))
    left, right = axes
    left.set_title("(a) Paired width error", loc="left")
    right.set_title("(b) Diffraction shape error", loc="left")
    models = ("raman_shape", "raman_scale")
    for j, (label, color) in enumerate(labels):
        for arm, line in (("transferred_even", "-"), ("separate_disjoint", "--")):
            widths = comparison["detectors"][arm]
            left.plot(comparison["probes_nm"], np.asarray(widths["model_errors"])[:, j],
                      line, lw=1.25, color=color, marker="o", markersize=2.2,
                      label=label if arm == "transferred_even" else None)
        minimum = comparison["detectors"]["transferred_even"]["minima"][models[j]]
        left.scatter(minimum["probe_nm"], minimum["error"], marker="D", s=16,
                     color=color, edgecolors="white", linewidth=.45, zorder=4)
        profiles = extension["profiles"]["transferred_even"]
        right.plot(extension["probes_nm"], np.asarray(profiles["model_errors"])[:, j],
                   "-", lw=1.25, color=color, marker="o", markersize=2.2)
    left.set(xlim=(1.225, 1.923), xticks=[1.25, 1.40, 1.55, 1.70, 1.85],
             ylabel="Map-mean width error (nm)")
    right.set(xlim=(1.224, 2.233), xticks=[1.25, 1.45, 1.65, 1.85, 2.05, 2.20],
              ylabel="Normalized profile error")
    for ax in axes:
        ax.set_xlabel("Assumed Gaussian probe FWHM (nm)")
        ax.axvline(1.25, color="#8795a0", lw=.8, ls=":", zorder=0)
        ax.grid(axis="y", color="#dce4e9", lw=.6)
        ax.set_axisbelow(True)
    left.legend(loc="upper right", fontsize=7, borderaxespad=.4, handlelength=1.6)
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--output", type=Path, default=FIGURES)
    parser.add_argument("--landscape", type=Path, help="Optional standalone frozen landscape JSON.")
    parser.add_argument("--source-archive", type=Path, default=FIGURES.parents[1] / "moire_v003_sources.zip")
    parser.add_argument("--uncertainty-output", type=Path,
                        default=FIGURES.parents[1] / "results" / "mos2_map_width_uncertainty_replay_v004.json")
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--fig2-only", action="store_true",
                        help="Render Fig. 2 only and leave the crossover figure untouched.")
    args = parser.parse_args()
    article_style()
    args.output.mkdir(parents=True, exist_ok=True)
    measured, raw_source = load_landscape(args.evidence, args.source_archive, args.landscape)
    uncertainty = spatial_width_uncertainty(
        measured, raw_source, args.source_archive, draws=args.bootstrap_draws, seed=args.seed
    )
    render_widths(
        measured, args.output / "measured_map_widths.pdf", uncertainty, args.uncertainty_output
    )
    if not args.fig2_only:
        crossover = json.loads((args.evidence / "mos2_observation_summary_v004.json").read_text())
        render_crossover(crossover, args.output / "probe_crossover.pdf")


if __name__ == "__main__":
    main()
