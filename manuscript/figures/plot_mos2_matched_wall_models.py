"""Build the matched wall-model comparison figure and manuscript values."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np

from plot_moire_barrier_revision import BLUE, EVIDENCE, FIGURES, ORANGE, WIDTH, article_style


GROUPS = (("higher_twist", "Close"), ("lower_twist", "Wide"))
MODELS = (
    ("measured", "Measured", "#273640", "o", 0.),
    ("network", "Periodic network", BLUE, "s", .12),
    ("isolated", "Isolated wall", ORANGE, "x", -.12),
)


def map_points(cases, summary):
    points = []
    for name, row in summary["maps"].items():
        comparison = row["models"]["scan_grid"]
        excluded = row["excluded_from_scan_grid_comparison"]
        selected = [wall for wall in cases["scans"][name]["walls"] if wall["centers"] not in excluded]
        if len(selected) != comparison["matched_sections"]:
            raise ValueError(f"{name}: matched-section count differs from the case record")
        points.append({
            "name": name,
            "spacing": median(wall["spacing_nm"] for wall in selected),
            "sections": len(selected),
            "gap": comparison["mean_first_minus_second_nm"],
        })
    return sorted(points, key=lambda row: row["spacing"])


def write_values(points, summary, summary_hash, output):
    values = {}
    for name, label in GROUPS:
        group = summary["groups"][name]["scan_grid"]
        error = group["map_mean_gap_absolute_error_nm"]
        spacing = [row["spacing"] for row in points if row["name"] in group["scans"]]
        low, high = group["network_minus_isolated_error_p05_p95_nm"]
        values.update({
            f"{label}Count": group["matched_sections"],
            f"{label}Spacing": rf"{min(spacing):.1f}\text{{--}}{max(spacing):.1f}",
            f"{label}Isolated": f"{error['isolated']:.3f}",
            f"{label}Network": f"{error['network']:.3f}",
            f"{label}GainRange": f"[{-high:.3f}, {-low:.3f}]",
        })
    full = summary["groups"]["all_maps"]["scan_grid"]
    values["Count"] = full["matched_sections"]
    error = summary["groups"]["higher_twist"]["scan_grid"]["map_mean_gap_absolute_error_nm"]
    values["ClosePercentage"] = round(100 * (1 - error["network"] / error["isolated"]))
    body = [f"% mos2_matched_wall_models_summary_v01.json sha256: {summary_hash}",
            f"% mos2_matched_wall_models_v01.json sha256: {summary['case_sha256']}"]
    for name, value in values.items():
        body.append(rf"\newcommand{{\Matched{name}}}{{{value}}}")
    body.append(r"\newcommand{\MatchedMapRows}{%")
    for row in points:
        body.append(" & ".join([row["name"], str(row["sections"]), f"{row['spacing']:.2f}",
                               *(f"{row['gap'][name]:+.3f}" for name, *_ in MODELS)]) + r" \\")
    body.append("}")
    output.write_text("\n".join(body) + "\n")


def write_rigid_values(report, digest, output):
    close = report["groups"]["higher_twist"]
    wide = report["groups"]["lower_twist"]
    error = close["map_mean_gap_absolute_error_nm"]
    advantage = close["network_error_reduction_p05_p95_nm"]["rigid"]
    values = {
        "CloseCount": close["three_model_sections"],
        "CloseMaps": len(close["three_model_maps"]),
        "WideCount": wide["three_model_sections"],
        "GapError": f"{error['rigid']:.2f}",
        "IsolatedGapError": f"{error['isolated']:.2f}",
        "NetworkGapError": f"{error['network']:.2f}",
        "NetworkAdvantage": f"[{advantage[0]:.2f}, {advantage[1]:.2f}]",
    }
    body = [f"% mos2_rigid_width_gaps_v01.json sha256: {digest}"]
    body.extend(rf"\newcommand{{\Rigid{name}}}{{{value}}}" for name, value in values.items())
    body.append(r"\newcommand{\RigidFitRows}{%")
    for name, row in report["maps"].items():
        body.append(f"{name} & {row['pair_sections']} & {row['three_model_sections']}" + r" \\")
    body.append("}")
    output.write_text("\n".join(body) + "\n")


def plot_rigid_comparison(ax, close):
    models = ("rigid", "isolated", "network")
    errors = [close["map_mean_gap_absolute_error_nm"][model] for model in models]
    top = max(errors)
    colors = ("#596a72", ORANGE, BLUE)
    ax.bar(range(3), errors, color=colors, width=.63, zorder=2)
    for index, (value, color) in enumerate(zip(errors, colors, strict=True)):
        ax.text(index, value + .025 * top, f"{value:.2f}", color=color, fontsize=7.2,
                ha="center", va="bottom")
    ax.set_title("(c) Common subset", loc="left")
    ax.set(xlim=(-.55, 2.55), ylim=(0, 1.17*top), ylabel="Mean absolute error (nm)",
           xticks=range(3), xticklabels=("Rigid", "Isolated", "Network"))
    ax.set_xlabel(f"{close['three_model_sections']} sections in {len(close['three_model_maps'])} maps")
    ax.grid(axis="y", color="#dbe0e4", lw=.5, zorder=0)


def plot_comparison(points, summary, output, rigid=None):
    article_style()
    ratios = [1.48, 1.] if rigid is None else [1.40, 1.03, .98]
    fig, axes = plt.subplots(1, len(ratios), figsize=(WIDTH, 2.82 if rigid else 2.65),
                              gridspec_kw={"width_ratios": ratios}, layout="constrained")
    left, right = axes[:2]
    if rigid:
        plot_rigid_comparison(axes[2], rigid["groups"]["higher_twist"])
    spacing = np.array([row["spacing"] for row in points])
    left.axhline(0.0, color="#aab3b9", lw=.7, zorder=0)
    for row in points:
        left.plot([row["spacing"], row["spacing"] + .12],
                  [row["gap"]["measured"], row["gap"]["network"]], color="#b9c8d3", lw=.85, zorder=1)
    for name, label, color, marker, shift in MODELS:
        gaps = [row["gap"][name] for row in points]
        left.plot(spacing + shift, gaps, linestyle="none", marker=marker, ms=5.0,
                  color=color, markerfacecolor="white" if name == "network" else color,
                  markeredgewidth=1.05, label=label, zorder=3)
    left.set(xlabel="Median spacing between wall rows (nm)",
             ylabel=r"Map-mean width difference $w_1-w_2$ (nm)",
             xlim=(6.5, 32.8), ylim=(-.90, 2.16), xticks=[8, 12, 16, 20, 24, 28, 32])
    left.set_title("(a) Width difference in each map", loc="left")
    right.set_title("(b) Error by spacing group" if rigid else "(b) Model error by spacing group", loc="left")
    labels = []
    for index, (group, _) in enumerate(GROUPS):
        row = summary["groups"][group]["scan_grid"]
        labels.append(("Closer rows" if index == 0 else "Wider rows") + f"\n{row['matched_sections']} sections")
        for model, color, offset in (("isolated", ORANGE, -.16), ("network", BLUE, .16)):
            value = row["map_mean_gap_absolute_error_nm"][model]
            right.bar(index + offset, value, width=.26, color=color, zorder=2)
            label = f"{value:.2f}" if rigid else f"{value:.3f}"
            right.text(index + offset, value + .040, label, color=color, fontsize=7.2,
                       ha="center", va="bottom")
    right.set(xlim=(-.55, 1.55), ylim=(0, 1.53), ylabel="Mean absolute error (nm)",
              xticks=[0, 1], xticklabels=labels, yticks=[0., .4, .8, 1.2])
    right.grid(axis="y", color="#dbe0e4", lw=.5, zorder=0)
    fig.legend(*left.get_legend_handles_labels(), loc="outside upper center", ncol=3,
               frameon=False, handletextpad=.32, columnspacing=1.7)
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument("--tables", type=Path, default=FIGURES.parent / "tables")
    parser.add_argument("--rigid", action="store_true", help="Include the common subset with rigid-layer fits.")
    args = parser.parse_args()
    case_bytes = (args.evidence / "mos2_matched_wall_models_v01.json").read_bytes()
    summary_bytes = (args.evidence / "mos2_matched_wall_models_summary_v01.json").read_bytes()
    cases, summary = json.loads(case_bytes), json.loads(summary_bytes)
    if sha256(case_bytes).hexdigest() != summary["case_sha256"]:
        raise ValueError("The matched comparison summary refers to a different case record")
    points = map_points(cases, summary)
    if args.rigid:
        rigid_bytes = (args.evidence / "mos2_rigid_width_gaps_v01.json").read_bytes()
        rigid = json.loads(rigid_bytes)
        if rigid["case_sha256"] != summary["case_sha256"]:
            raise ValueError("The rigid comparison refers to a different case record")
        for name, _ in GROUPS:
            if rigid["groups"][name]["pair_sections"] != summary["groups"][name]["scan_grid"]["matched_sections"]:
                raise ValueError("The two-model cohort differs between the comparisons")
        write_rigid_values(rigid, sha256(rigid_bytes).hexdigest(), args.tables / "v019_rigid_values.tex")
        plot_comparison(points, summary, args.figures / "matched_wall_widths_v019.pdf", rigid)
    else:
        write_values(points, summary, sha256(summary_bytes).hexdigest(), args.tables / "v017_matched_wall_values.tex")
        plot_comparison(points, summary, args.figures / "matched_wall_widths_v017.pdf")


if __name__ == "__main__":
    main()
