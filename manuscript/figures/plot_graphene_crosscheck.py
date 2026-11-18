"""Compare the same wall-width observable in two archived graphene scans."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INK = "#23303c"
BLUE = "#23689b"
ORANGE = "#b4552a"
GRAY = "#6a717b"
GRID = "#e1e6ea"


def primary_scores(evidence):
    high = json.loads((evidence / "graphene_wall_signal_repeat_v01.json").read_text())
    low = json.loads((evidence / "graphene_wall_signal_low_twist_coverage_v01.json").read_text())
    return (high["apertures"][0]["all"]["summary"], low["scans"]["s1-10"]["apertures"][0])


def plot(scores, output):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "text.color": INK,
        "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.linewidth": .7,
        "xtick.color": INK, "ytick.color": INK, "legend.fontsize": 7.1,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 3.02), gridspec_kw={"width_ratios": (1, 1)})
    fig.subplots_adjust(left=.105, right=.983, top=.875, bottom=.205, wspace=.38)
    positions = np.arange(len(scores))
    observed = np.array([score["observed_gap_on_common_pairs_nm_median"] for score in scores])
    periodic = np.array([score["models"]["periodic"]["gap_nm_median"] for score in scores])
    isolated = np.array([score["models"]["isolated"]["gap_nm_median"] for score in scores])
    repeat = np.array([score["opposite_gap_med_abs_difference_nm"] for score in scores])
    counts = [score["common_model_pairs"] for score in scores]
    ticks = [f"{angle}°" + "\n" + f"{count} sections"
             for angle, count in zip(("1.26", "0.63"), counts, strict=True)]

    ax = axes[0]
    ax.scatter(positions - .13, observed, s=25, color=INK, label="Experiment", zorder=4)
    ax.scatter(positions, periodic, s=31, edgecolors=BLUE, facecolors="white",
               linewidths=1.15, label="Periodic model", zorder=3)
    ax.scatter(positions + .13, isolated, s=26, color=GRAY, marker="x",
               linewidths=1.15, label="Isolated wall", zorder=3)
    ax.set(ylabel="Median first-minus-second width (nm)", ylim=(-.22, 5.15))
    ax.legend(loc="upper right", frameon=False, handletextpad=.3, labelspacing=.48)

    ax = axes[1]
    ax.bar(positions - .15, abs(observed - periodic), .24, color=BLUE,
           label="Periodic model error", zorder=3)
    ax.bar(positions + .15, repeat, .24, color=ORANGE,
           label="Opposite-spot disagreement", zorder=3)
    for x, error, disagreement in zip(positions, abs(observed - periodic), repeat, strict=True):
        ax.text(x - .15, error + .06, f"{error:.2f}", ha="center", fontsize=7)
        ax.text(x + .15, disagreement + .06, f"{disagreement:.2f}", ha="center", fontsize=7)
    ax.set(ylabel="Difference (nm)", ylim=(0, 3.1))
    ax.legend(loc="upper left", frameon=False, handlelength=1.1, handletextpad=.45)

    for ax, letter, title in zip(axes, "ab", ("Width difference by graphene scan", "Prediction and readout precision"),
                                 strict=True):
        ax.set_title(title, loc="left", fontsize=8.8, pad=8)
        ax.text(-.145, 1.052, f"({letter})", transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom")
        ax.set_xticks(positions, ticks)
        ax.set_xlim(-.43, 1.43)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, lw=.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#a2adb7")
        ax.tick_params(length=2.8, width=.65)
    fig.savefig(output, facecolor="white", metadata={"Title": "Cross-check of diffraction width differences in graphene"})
    plt.close(fig)


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=directory.parents[1] / "evidence")
    parser.add_argument("--output", type=Path, default=directory / "graphene_crosscheck.pdf")
    args = parser.parse_args()
    plot(primary_scores(args.evidence), args.output)


if __name__ == "__main__":
    main()
