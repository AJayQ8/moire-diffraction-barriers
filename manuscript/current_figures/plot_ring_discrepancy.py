"""Render the measured ring discrepancy and its fixed-energy spacing control."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from panel_style import panel_heading


INK = "#23303c"
BLUE = "#23689b"
ORANGE = "#b4552a"
GRAY = "#6a717b"
GRID = "#e1e6ea"


def observed(ax, evidence):
    rows = sorted(evidence["scans"].values(), key=lambda row: row["twist_deg"])
    angles = [row["twist_deg"] for row in rows]
    ratios = np.array([[row["typical_barrier_ratio"][probe]["measured"] for row in rows]
                       for probe in ("0.75", "1.25", "1.75")])
    periodic = [row["typical_barrier_ratio"]["1.25"]["periodic_anchor"] for row in rows]
    isolated = [row["typical_barrier_ratio"]["1.25"]["isolated_anchor"] for row in rows]
    ax.axhline(1, lw=.85, color="#bcc5cd", zorder=0)
    ax.errorbar(angles, ratios[1], yerr=(ratios[1]-ratios.min(axis=0), ratios.max(axis=0)-ratios[1]),
                fmt="o", color=INK, markersize=4.1, capsize=2.3, lw=.9, zorder=4,
                label="Experiment")
    ax.scatter(angles, periodic, facecolors="white", edgecolors=BLUE, s=26,
               linewidths=1.1, zorder=3, label="Periodic model")
    ax.scatter(angles, isolated, color=GRAY, marker="x", s=19, linewidths=1.,
               zorder=2, label="Isolated wall")
    ax.set_yscale("log", base=2)
    ax.set_yticks((1, 2, 4, 8), ("1×", "2×", "4×", "8×"))
    ax.set(ylim=(.83, 8.6), xlim=(.32, 1.98), xlabel="Twist from first-order reflections (°)",
           ylabel="Apparent barrier ratio, second / first")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=True))
    order = ("Experiment", "Periodic model", "Isolated wall")
    ax.legend([by_label[label] for label in order], order, loc="upper left",
              frameon=False, handletextpad=.45, borderaxespad=.5)


def controlled(ax, evidence):
    sweep = sorted((row for label, row in evidence["cases"].items() if label.startswith("pure_twist/")),
                   key=lambda row: row["row_spacing_nm"])
    spacing = [row["row_spacing_nm"] for row in sweep]
    predicted = np.array([[fit["width_nm"] for fit in row["reconstruction_widths"]["periodic"]]
                          for row in sweep])
    actual = [row["true_periodic_width_nm"] for row in sweep]
    ax.plot(spacing, predicted[:, 0], "o-", color=BLUE, ms=3.4, lw=1.2, label="First-order fit")
    ax.plot(spacing, predicted[:, 1], "s-", color=ORANGE, ms=3.35, lw=1.2, label="Second-order fit")
    ax.plot(spacing, actual, "d--", color=INK, ms=3.2, lw=1.1, label="Midpoint slip-gradient width")
    ax.set(xlim=(7.1, 41), ylim=(1.5, 5.65), xlabel="Spacing between wall rows (nm)",
           ylabel="Width (nm)")
    ax.set_xticks((8, 16, 24, 32, 40))
    ax.legend(loc="upper right", frameon=False, handletextpad=.55, borderaxespad=.4)


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=directory.parents[1]/"evidence")
    parser.add_argument("--output", type=Path, default=directory/"diffraction_order_widths_v027.pdf")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "text.color": INK, "axes.labelcolor": INK,
                         "axes.titlecolor": INK, "axes.linewidth": .7,
                         "xtick.color": INK, "ytick.color": INK,
                         "legend.fontsize": 7.35, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.85))
    fig.subplots_adjust(left=.115, right=.983, top=.867, bottom=.195, wspace=.37)
    rings = json.loads((args.evidence/"moire_wall_energy_readout_v01.json").read_text())
    sweep = json.loads((args.evidence/"mos2_wall_period_scaling_v01.json").read_text())
    observed(axes[0], rings)
    controlled(axes[1], sweep)
    for ax, key, title in zip(axes, "ab", ("Parallel MoS₂ maps", "Fixed-energy simulation"), strict=True):
        panel_heading(ax, key, title)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, lw=.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#a2adb7")
        ax.tick_params(length=2.8, width=.65)
    fig.savefig(args.output, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
