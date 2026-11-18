#!/usr/bin/env python3
"""Render measured-wall examples from their evidence JSON.

The default scan order preserves the earlier DS14-then-DS10 layout.  Use
``--scan-order DS10 DS14`` for the v015 close-then-wide layout.
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE.parent.parent / "evidence" / "measured_wall_examples_v01.json"
DEFAULT_OUTPUT = HERE / "measured_wall_examples.pdf"
DEFAULT_SCAN_ORDER = ("DS14", "DS10")

BLUE = "#23689b"
ORANGE = "#b4552a"
INK = "#222222"
GREY = "#777777"


def panel_label(ax, label):
    ax.text(
        0.025, 0.975, f"({label})", transform=ax.transAxes,
        ha="left", va="top", fontsize=8.5, fontweight="bold",
        color="white" if label in ("a", "d") else INK,
        bbox={"facecolor": "black" if label in ("a", "d") else "white",
              "alpha": 0.68, "edgecolor": "none", "pad": 1.5},
        zorder=20,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--scan-order", nargs=2, default=DEFAULT_SCAN_ORDER,
        metavar=("TOP", "BOTTOM"),
        help="scan IDs to show in the top and bottom rows (default: DS14 DS10)",
    )
    args = parser.parse_args()
    record = json.loads(args.input.read_text())
    missing = [name for name in args.scan_order if name not in record["scans"]]
    if missing:
        parser.error(f"scan(s) absent from evidence input: {', '.join(missing)}")
    native = record.get("schema") == "mos2-native-wall-examples-v01"
    comparison = ("rigid", ORANGE, "--", "Rigid layers") if native else (
        "isolated_anchor", ORANGE, "--", "Isolated wall",
    )
    model_styles = (("periodic_anchor", BLUE, "-", "Periodic network"), comparison)
    widths_key = "measured_widths_nm" if native else "widths_from_archived_separate_ring_fits_nm"

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.3,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })

    scans = [record["scans"][name] for name in args.scan_order]
    labels = ("a", "b", "c", "d", "e", "f")
    fig, axes = plt.subplots(
        2, 3, figsize=(7.05, 4.72),
        gridspec_kw={"width_ratios": (1.03, 1.0, 1.0)},
    )
    all_shapes = []

    for row_index, (scan_name, scan) in enumerate(zip(args.scan_order, scans)):
        image_record = scan["first_ring_image"]
        image = np.asarray(image_record["raw_integrated_count_crop"], dtype=float)
        row0, row1, col0, col1 = image_record["crop_bounds_rc_px_half_open"]
        pixel = float(scan["pixel_size_nm"])
        origin_x, origin_y = image_record["display_origin_xy_nm"]
        edge_offset = 0.5 if native else 0.0
        extent = (
            (col0 - edge_offset) * pixel - origin_x,
            (col1 - edge_offset) * pixel - origin_x,
            (row1 - edge_offset) * pixel - origin_y,
            (row0 - edge_offset) * pixel - origin_y,
        )
        lo, hi = image_record["display_quantiles_raw_counts"]
        ax_map = axes[row_index, 0]
        im = ax_map.imshow(
            image, origin="upper", extent=extent, cmap="gray_r",
            vmin=lo, vmax=hi, interpolation="nearest", aspect="equal",
        )
        chord_xy = np.asarray(image_record["chord_xy_nm"]) - np.array([origin_x, origin_y])
        cut_xy = np.asarray(image_record["midpoint_cut_xy_nm"]) - np.array([origin_x, origin_y])
        cut_distance = np.asarray(image_record["midpoint_cut_distance_nm_full"], dtype=float)
        cut_indices = np.asarray(image_record["midpoint_cut_displayed_indices"], dtype=int)
        if len(cut_distance) != len(cut_xy) or not np.array_equal(
            cut_indices, np.flatnonzero(
                (cut_distance >= scan["profile_distance_nm"][0] - 1e-9)
                & (cut_distance <= scan["profile_distance_nm"][-1] + 1e-9)
            )
        ):
            raise ValueError(f"{scan_name}: displayed cut does not match the profile window")
        ax_map.plot(
            chord_xy[:, 0], chord_xy[:, 1],
            color="#16a085", lw=1.7, ls="--", solid_capstyle="round", zorder=5,
        )
        ax_map.scatter(
            chord_xy[:, 0], chord_xy[:, 1], s=11, color="#16a085",
            edgecolors="white", linewidths=0.55, zorder=7,
        )
        ax_map.plot(cut_xy[cut_indices, 0], cut_xy[cut_indices, 1], color="#c23b32", lw=1.15, zorder=6)
        ax_map.set_title(
            f"{scan_name}  {scan['twist_deg']:.3f}°\n{scan['row_spacing_nm']:.2f} nm wall spacing",
            loc="left", pad=3,
        )
        ax_map.set_xlabel("Image x (nm)")
        ax_map.set_ylabel("Image y (nm)")
        ax_map.tick_params(length=2.2, pad=1.6)
        ax_map.set_xlim(extent[0], extent[1])
        ax_map.set_ylim(extent[2], extent[3])
        cbar = fig.colorbar(im, ax=ax_map, fraction=0.047, pad=0.018)
        cbar.set_label("Integrated counts", fontsize=6.3, labelpad=2)
        cbar.ax.tick_params(labelsize=6.2, length=2, pad=1)
        panel_label(ax_map, labels[row_index * 3])

        distance = np.asarray(scan["profile_distance_nm"], dtype=float)
        for ring_index, (ring_key, panel_index) in enumerate((("first", 1), ("second", 2))):
            ax = axes[row_index, panel_index]
            ring = scan["profile_channels"][ring_key]
            mode_hk = ring["representative_mode_hk"]
            observed = np.asarray(ring["plotted_measured_unit_l2_shape"], dtype=float)
            all_shapes.append(observed)
            ax.plot(
                distance, observed, color=INK, lw=0.75, marker="o",
                markersize=2.25, markevery=2, markerfacecolor="white",
                markeredgewidth=0.7, zorder=4,
                label="Measured channel" if row_index == 0 and ring_index == 0 else None,
            )
            for model_name, color, linestyle, label in model_styles:
                prediction = np.asarray(
                    scan["model_curves"][model_name]["unit_l2_shape_by_ring"][ring_key],
                    dtype=float,
                )
                all_shapes.append(prediction)
                ax.plot(
                    distance, prediction, color=color, ls=linestyle,
                    label=label if row_index == 0 and ring_index == 0 else None,
                )
            order = "first" if ring_key == "first" else "second"
            ax.set_title(f"{order} order  ({mode_hk[0]:g}, {mode_hk[1]:g})", loc="left", pad=3)
            ax.set_xlim(distance.min(), distance.max())
            ax.axhline(0, color=GREY, lw=0.55, zorder=0)
            ax.set_xlabel("Distance across wall (nm)")
            ax.set_ylabel("Normalized profile")
            ax.tick_params(length=2.2, pad=1.6)
            apparent_width = scan[widths_key][ring_key]
            ax.text(
                0.97, 0.96,
                rf"$w_{ring_index + 1} = {apparent_width:.2f}\,\mathrm{{nm}}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.1,
                color=INK,
                bbox={"facecolor": "white", "edgecolor": "#aaaaaa",
                      "linewidth": 0.35, "alpha": 0.88, "pad": 1.3},
            )
            panel_label(ax, labels[row_index * 3 + panel_index])

    ymax = max(float(np.max(np.abs(curve))) for curve in all_shapes) * (1.24 if native else 1.10)
    for ax in axes[:, 1:].flat:
        ax.set_ylim(-ymax, ymax)

    profile_legend_handles = [
        Line2D([0], [0], color=INK, lw=0.75, marker="o", markersize=2.25,
               markerfacecolor="white", markeredgewidth=0.7,
               label="Measured profile"),
        *(Line2D([0], [0], color=color, lw=1.45, ls=style, label=label)
          for _, color, style, label in model_styles),
    ]
    map_legend_handles = [
        Line2D([0], [0], color="#16a085", lw=1.7, ls="--", marker="o",
               markersize=3.0, markerfacecolor="#16a085", markeredgecolor="white",
               markeredgewidth=0.55, label="Wall segment"),
        Line2D([0], [0], color="#c23b32", lw=1.35,
               label="Midpoint cut (±4.25 nm)"),
    ]
    # Matplotlib fills legends by column; interleave handles so the first row
    # presents the three profile curves and the second row presents map marks.
    legend_handles = [
        profile_legend_handles[0], map_legend_handles[0],
        profile_legend_handles[1], map_legend_handles[1],
        profile_legend_handles[2],
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=3, frameon=False, handlelength=1.8, columnspacing=1.05,
    )
    fig.text(
        0.5, 0.89,
        "Each width fit uses all selected reflections. Each plot shows one reflection.",
        ha="center", va="center", fontsize=7.0,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.105, top=0.81, wspace=0.68, hspace=0.54)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf")
    plt.close(fig)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
