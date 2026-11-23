"""Plot synthetic barrier recovery under the combined nuisance condition.

The plotted statistic is the recovered/input barrier ratio for each fit
realization. Points show the median of the 12 noisy readouts at each known
barrier level; whiskers show the empirical 10th--90th percentile range.
These are Monte Carlo repeat spreads, not profile-likelihood intervals.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from panel_style import panel_heading


INK = "#23303c"
BLUE = "#23689b"
ORANGE = "#b4552a"
GRAY = "#6a717b"
GRID = "#e1e6ea"
CONDITION = "variable_probe_and_landscape"
PANELS = ((0.55, 28.6, "a"), (1.85, 8.5, "b"))
METHODS = (
    ("periodic_free", "Periodic, fitted probe width", BLUE, "o"),
    ("isolated", "Exact isolated wall", ORANGE, "s"),
    ("periodic_known", "Periodic, supplied probe width", GRAY, "^"),
)
X_OFFSETS = {"periodic_free": -0.20, "isolated": 0.0, "periodic_known": 0.20}


def rows_for_spacing(recovery, twist, spacing_nm):
    rows = [
        (sample_id, sample)
        for sample_id, sample in recovery["samples"].items()
        if sample["condition"] == CONDITION
        and abs(sample["twist_degrees"] - twist) < 1e-8
        and abs(sample["spacing_nm"] - spacing_nm) < 0.1
    ]
    rows.sort(key=lambda item: item[1]["truth"]["barrier_meV_per_cell"])
    if len(rows) != 3:
        raise ValueError(f"Expected three barrier levels at {spacing_nm:g} nm; found {len(rows)}")
    return rows


def estimates(sample_id, sample, probe_control):
    truth = float(sample["truth"]["barrier_meV_per_cell"])
    profile_fits = sample["fits"]
    fitted_probe_rows = profile_fits["periodic"]["both"]["profile_shapes"]
    isolated_rows = profile_fits["isolated"]["both"]["profile_shapes"]
    known_probe_rows = probe_control["samples"][sample_id]["fits"]
    if not (len(isolated_rows) == len(fitted_probe_rows) == len(known_probe_rows) == 12):
        raise ValueError(f"{sample_id}: expected 12 matching noisy readouts per method")

    values = {
        "isolated": np.array([row["barrier_meV_per_cell"] for row in isolated_rows], dtype=float),
        "periodic_free": np.array([row["barrier_meV_per_cell"] for row in fitted_probe_rows], dtype=float),
        "periodic_known": np.array([row["profile_shapes"]["barrier_meV_per_cell"]
                                     for row in known_probe_rows], dtype=float),
    }
    return {
        method: np.percentile(series / truth, (10, 50, 90))
        for method, series in values.items()
    }


def build_figure(recovery, probe_control, output):
    if recovery["images_sha256"] != probe_control["images_sha256"]:
        raise ValueError("The probe control uses different synthetic images")
    if recovery["truth_sha256"] != probe_control["sealed_truth_sha256"]:
        raise ValueError("The probe control is not paired with the recovery truth key")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.linewidth": 0.7,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.fontsize": 7.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.08, 3.55), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.90, bottom=0.33, wspace=0.24)

    for ax, (twist, spacing, panel) in zip(axes, PANELS, strict=True):
        cases = rows_for_spacing(recovery, twist, spacing)
        levels = np.array([case[1]["truth"]["barrier_meV_per_cell"] for case in cases], dtype=float)
        values = {name: [] for name, *_ in METHODS}
        for sample_id, sample in cases:
            summary = estimates(sample_id, sample, probe_control)
            for method, quantiles in summary.items():
                values[method].append(quantiles)

        ax.axhline(1.0, color="#89949d", lw=0.85, ls=(0, (3, 2)), zorder=0)
        for method, label, color, marker in METHODS:
            quantiles = np.asarray(values[method])
            low, median, high = quantiles.T
            x = levels + X_OFFSETS[method]
            ax.errorbar(
                x,
                median,
                yerr=np.vstack((median - low, high - median)),
                fmt=f"{marker}-",
                color=color,
                markersize=4.0,
                markeredgewidth=0.65,
                linewidth=1.0,
                elinewidth=0.9,
                capsize=2.2,
                capthick=0.8,
                zorder=3,
                label=label,
            )

        panel_heading(ax, panel, f"{spacing:.1f} nm wall spacing")
        ax.set_xlim(8.7, 21.7)
        ax.set_ylim(0.65, 2.08)
        ax.set_xticks(levels, [f"{value:.2f}" for value in levels])
        ax.set_yticks((0.75, 1.0, 1.25, 1.5, 1.75, 2.0))
        ax.set_xlabel("Input barrier (meV/cell)")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, lw=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#a2adb7")
        ax.tick_params(length=2.8, width=0.65, pad=2.5)

    axes[0].set_ylabel("Recovered / input barrier")
    handles = [Line2D([0], [0], color=color, marker=marker, lw=1.0, markersize=4.0,
                      markeredgewidth=0.65, label=label)
               for _, label, color, marker in METHODS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.075),
               ncol=3, frameon=False, handletextpad=0.45, columnspacing=1.4)
    fig.savefig(output, facecolor="white", metadata={
        "Title": "Synthetic MoS2 barrier recovery under probe and landscape mismatch",
        "Subject": "Two-ring barrier recovery for two moire spacings",
        "Creator": "plot_barrier_recovery.py",
    })
    plt.close(fig)


def print_statistics(recovery, probe_control):
    for twist, spacing, _ in PANELS:
        print(f"{twist:.2f} deg / {spacing:.1f} nm")
        for sample_id, sample in rows_for_spacing(recovery, twist, spacing):
            truth = sample["truth"]["barrier_meV_per_cell"]
            stats = estimates(sample_id, sample, probe_control)
            line = ", ".join(
                f"{method}={q[1]:.3f} [{q[0]:.3f}, {q[2]:.3f}]"
                for method, q in stats.items()
            )
            print(f"  {sample_id}: truth={truth:.3f}; {line}")


def main():
    directory = Path(__file__).resolve().parent
    evidence = directory.parents[1] / "evidence"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery", type=Path, default=evidence / "mos2_barrier_recovery_v01.json")
    parser.add_argument("--probe-control", type=Path, default=evidence / "mos2_barrier_probe_control_v01.json")
    parser.add_argument("--output", type=Path, default=directory / "barrier_recovery_v027.pdf")
    args = parser.parse_args()

    recovery = json.loads(args.recovery.read_text())
    probe_control = json.loads(args.probe_control.read_text())
    build_figure(recovery, probe_control, args.output)
    print_statistics(recovery, probe_control)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
