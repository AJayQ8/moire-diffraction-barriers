"""Plot paired width errors across the tested stacking-energy surfaces."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from panel_style import panel_heading


TRANSFERRED = "transferred_even"
STYLES = (
    ("archived", "Archived crystal orientation", "#23689b", "o", "-"),
    ("rotate_ds13_ds14", "Rotated in DS13 and DS14", "#ab4c2a", "s", "--"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = json.loads(args.summary.read_text())
    energies = np.asarray([float(key) for key in result["settings_meV_per_cell"]])

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#687984",
        "axes.labelcolor": "#273746",
        "text.color": "#273746",
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7.0, 2.95))
    fig.subplots_adjust(left=.105, right=.98, bottom=.205, top=.80, wspace=.22)
    if "families" in result:
        families = result["families"]
        cases = (
            ("Fixed AA–AB energy difference", families["fixed_AA_energy"]["ten_maps"]),
            ("Fixed AA curvature", families["fixed_AA_curvature"]["ten_maps"]),
        )
    else:
        cases = (
            ("All ten maps", result["ten_maps"]),
            ("DS13 omitted", result["leave_one_map_out"]["DS13"]),
        )
    for axis, letter, (label, group) in zip(axes, "ab", cases, strict=True):
        panel_heading(axis, letter, label)
        curves = group["cameras"][TRANSFERRED]["curves"]
        for frame, legend, color, marker, style in STYLES:
            row = curves[frame]
            low, high = row["pointwise_p05_p95_nm"]
            axis.fill_between(energies, low, high, color=color, alpha=0.13, linewidth=0)
            axis.plot(energies, row["mean_curve_nm"], ls=style, marker=marker, ms=3.2, lw=1.3,
                      color=color, label=legend)
            axis.scatter(float(row["best_setting_meV_per_cell"]), row["sampled_minimum_nm"],
                         s=44, marker=marker, facecolors="white", edgecolors=color, linewidths=1.3, zorder=5)
        axis.set_xticks(np.arange(energies[0], energies[-1] + 0.1, 2))
        axis.grid(axis="y", color="#d2dbe1", lw=0.45)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Map-mean width error (nm)")
    fig.supxlabel("Sliding barrier (meV per cell)", fontsize=8, y=.045, color="#273746")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .985),
               ncol=2, frameon=False, handlelength=2.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
