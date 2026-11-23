"""Draw the overview, matched model comparisons and local example from saved fits."""
import argparse
import json
from hashlib import sha256
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from panel_style import panel_heading
from plot_graphical_abstract_preview import domain_network, sheet
from plot_mos2_matched_wall_models import GROUPS, MODELS, map_points, plot_rigid_comparison
from plot_v021_overview import BLUE, INK, ORANGE, draw_profile, draw_wall_image, style

FIGURES = Path(__file__).resolve().parent
EVIDENCE = FIGURES.parents[1] / "evidence"
FIGURE_WIDTH = 7.02
WALL = "#16a085"
CUT = "#c23b32"
SPACING = "#79538c"


def axes_box(fig, left, bottom, width, height):
    """Specify a panel's dimensions in inches, independent of the figure height."""
    w, h = fig.get_size_inches()
    return fig.add_axes([left / w, bottom / h, width / w, height / h])


def line_key(ax, handles, y, columns):
    return ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
                     ncol=columns, frameon=False, borderaxespad=0, handlelength=1.65,
                     columnspacing=0.95, handletextpad=0.4, fontsize=7.4)


def schematics(fig):
    bilayer = axes_box(fig, 0.68, 5.95, 2.19, 1.01)
    bilayer.set_axis_off()
    panel_heading(bilayer, "a", "Twisted MoS₂ bilayer")
    for bottom, angle, muted in ((0.01, -32, True), (0.51, -12, False)):
        layer = bilayer.inset_axes([0.07, bottom, 0.86, 0.49])
        layer.set(xlim=(-126,126), ylim=(62,-62), aspect="equal")
        layer.set_axis_off()
        sheet(layer, (0,0), angle, muted=muted)

    geometry = axes_box(fig, 4.19, 5.95, 2.65, 1.01)
    geometry.set_axis_off()
    panel_heading(geometry, "b", "Moiré domains and wall cut")
    diagram = geometry.inset_axes([0.005,0.04,0.52,0.93])
    diagram.set(xlim=(-71,71), ylim=(71,-71), aspect="equal")
    diagram.set_axis_off()
    domain_network(diagram, 0, 0, 68)
    tangent = np.array([32, -7.3])
    midpoint = np.array([-10, 2]) + tangent / 2
    normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
    ends = midpoint + np.array([-23,23])[:, None] * normal
    diagram.plot(ends[:,0], ends[:,1], color=CUT, lw=1.5, zorder=9)
    for y, color, text in ((0.61, "#046d71", "Wall segment"), (0.38, CUT, "Transverse cut")):
        geometry.plot([0.56,0.63], [y,y], color=color, lw=1.7, transform=geometry.transAxes)
        geometry.text(0.66, y, text, va="center", fontsize=8.0, transform=geometry.transAxes)


def image_and_profiles(fig, scan, map_name, labels, top):
    row0,row1,col0,col1 = scan["first_ring_image"]["crop_bounds_rc_px_half_open"]
    aspect = (row1-row0)/(col1-col0)
    image_width = min(2.19, 2.22 / aspect)
    image_height = image_width * aspect
    left = 0.68
    ax = axes_box(fig, left, top-image_height, image_width, image_height)
    image = draw_wall_image(ax, scan)
    panel_heading(ax, labels[0], f"{map_name} diffraction map")
    cb = axes_box(fig, left+image_width+0.09, top-image_height, 0.075, image_height)
    bar = fig.colorbar(image, cax=cb)
    bar.ax.set_title("Counts", fontsize=7.3, pad=7)
    bar.ax.tick_params(labelsize=7, length=2, pad=2)
    line_key(ax, [
        Line2D([], [], color=WALL, ls="--", marker="o", ms=3, mew=.45, mec="white", label="Wall segment"),
        Line2D([], [], color=CUT, label="Transverse cut")], -0.46 / image_height, 2)

    profiles = []
    maxima = []
    for index,(key,letter) in enumerate(zip(("first","second"), labels[1:])):
        axis = axes_box(fig, 4.19, top-0.98-1.40*index, 2.65, 0.98)
        maxima.append(draw_profile(axis, scan, key))
        hk = scan["profile_channels"][key]["representative_mode_hk"]
        order = "First" if index == 0 else "Second"
        panel_heading(axis, letter, f"{order} order ({hk[0]:g}, {hk[1]:g})")
        axis.set_ylabel("Normalized profile", labelpad=3)
        if index == 0:
            axis.set_xlabel("")
            axis.tick_params(labelbottom=False)
        profiles.append(axis)
    for axis in profiles:
        axis.set_ylim(-1.24*max(maxima), 1.24*max(maxima))
    line_key(profiles[1], [
        Line2D([], [], color=INK, lw=.75, marker="o", mfc="white", mew=.7, ms=2.6, label="Measured"),
        Line2D([], [], color=BLUE, label="Periodic network"),
        Line2D([], [], color=ORANGE, ls="--", label="Rigid layers")], -0.49 / 0.98, 3)


def map_gaps(ax, points, group):
    for row in points:
        d, gaps = row["spacing"], row["gap"]
        ax.plot([d,d+.12], [gaps["measured"],gaps["network"]], color="#b9c8d3", lw=.85, zorder=1)
    for key,name,color,marker,offset in MODELS:
        ax.plot([r["spacing"]+offset for r in points], [r["gap"][key] for r in points],
                ls="none", marker=marker, ms=5, color=color, mfc="white" if key=="network" else color,
                mew=1.05, label=name, zorder=3)
    ax.axhline(0, color="#aab3b9", lw=.7, zorder=0)
    ax.set(xlabel="Spacing between parallel wall rows (nm)", ylabel=r"Mean $w_1-w_2$ (nm)",
           xlim=(6.5,32.8), ylim=(-.9,2.65), xticks=[8,12,16,20,24,28,32])
    panel_heading(ax, "f", f"Width difference across the ten {group} maps")
    ax.legend(loc="upper right", ncol=3, frameon=False, handlelength=1.1,
              handletextpad=.4, columnspacing=1.2, fontsize=7.4)


def matched_errors(fig, summary, rigid, additional, group):
    ax = axes_box(fig, .70, 2.85, 2.47, 1.45)
    for index,(spacing_group,_) in enumerate(GROUPS):
        errors = summary["groups"][spacing_group]["scan_grid"]["map_mean_gap_absolute_error_nm"]
        for key,label,color,offset in (("isolated","Isolated wall",ORANGE,-.16),
                                        ("network","Periodic network",BLUE,.16)):
            value = errors[key]
            ax.bar(index+offset, value, width=.26, color=color, zorder=2, label=label if index==0 else None)
            ax.text(index+offset,value+.04,f"{value:.2f}",color=color,ha="center",va="bottom",fontsize=8)
    ax.set(xlim=(-.55,1.55), ylim=(0,1.68), ylabel="Mean absolute error (nm)",
           xticks=[0,1], xticklabels=["Closer spacing\n168 walls, 6 maps","Wider spacing\n74 walls, 4 maps"],
           yticks=[0,.4,.8,1.2,1.6])
    ax.grid(axis="y", color="#dbe0e4", lw=.5, zorder=0)
    panel_heading(ax, "a", f"{group.capitalize()} maps")
    ax.legend(loc="upper right", frameon=False, fontsize=7.2, borderaxespad=.2)

    ax = axes_box(fig, 4.18, 2.85, 2.62, 1.45)
    plot_rigid_comparison(ax, rigid["groups"]["higher_twist"])
    ax.set_title("", loc="left")
    ax.set_xlabel("112 walls in 5 closer-spaced maps", labelpad=3)
    panel_heading(ax, "b", "Walls with fits for all three models")

    ax = axes_box(fig, .70, .69, 6.11, 1.42)
    maps = sorted(additional["maps"], key=lambda name: additional["maps"][name]["primary"]["median_spacing_nm"])
    positions = [0,1,2,3,4.45]
    for key,label,color,marker,offset in (("isolated","Isolated wall",ORANGE,"x",-.16),
                                         ("network","Periodic network",BLUE,"s",0),
                                         ("spacing_only","Spacing alone",SPACING,"D",.16)):
        errors = [additional["maps"][m]["primary"]["absolute_gap_error_nm"][key] for m in maps]
        errors.append(additional["primary"]["equal_map_mae_nm"][key])
        ax.plot(np.array(positions)+offset, errors, ls="none", marker=marker, ms=5.6,
                color=color, mfc="white" if key!="isolated" else color, mew=1.1, label=label, zorder=4)
    ax.set(xticks=positions, xticklabels=maps+["Four-map mean"], xlim=(-.48,4.95),
           ylim=(-.07,2.13), yticks=[0,.5,1,1.5,2], ylabel="Absolute error (nm)")
    ax.axvline(3.73, color="#a8b1b6", lw=.8, ls=":")
    ax.grid(axis="y", color="#dbe0e4", lw=.5, zorder=0)
    panel_heading(ax, "c", "Four additional closer-spaced maps")
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.22), frameon=False, ncol=3,
              handlelength=1.1, handletextpad=.35, columnspacing=1.1, fontsize=7.4)


def save(fig, stem, title, edition):
    fig.savefig(FIGURES / f"{stem}_{edition}.pdf", facecolor="white",
                metadata={"Title":title, "Author":"Ayas Alzanki", "CreationDate":None, "ModDate":None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edition", choices=("v027", "v028"), default="v027")
    edition = parser.parse_args().edition
    group = "primary" if edition == "v028" else "original"
    style()
    def read(name):
        return json.loads((EVIDENCE / name).read_text())
    examples = read("mos2_native_wall_examples_v01.json")["scans"]
    path = EVIDENCE / "mos2_matched_wall_models_v01.json"
    summary = read("mos2_matched_wall_models_summary_v01.json")
    rigid = read("mos2_rigid_width_gaps_v01.json")
    assert sha256(path.read_bytes()).hexdigest() == summary["case_sha256"] == rigid["case_sha256"]
    points = map_points(json.loads(path.read_bytes()), summary)
    additional = read("mos2_additional_parallel_map_widths_v01.json")

    fig = plt.figure(figsize=(FIGURE_WIDTH,7.26))
    schematics(fig)
    image_and_profiles(fig, examples["DS10"], "DS10", "cde", top=5.47)
    map_gaps(axes_box(fig, .68, .53, 6.16, 1.40), points, group)
    save(fig, "interface_overview", "One physical wall gives two diffraction-derived widths", edition)

    fig = plt.figure(figsize=(FIGURE_WIDTH,4.62))
    matched_errors(fig, summary, rigid, additional, group)
    save(fig, "model_errors", "Width-difference errors across maps", edition)

    fig = plt.figure(figsize=(FIGURE_WIDTH,3.45))
    image_and_profiles(fig, examples["DS14"], "DS14", "abc", top=3.16)
    save(fig, "local_width_outlier", "Local width discrepancy in DS14", "v027")


if __name__ == "__main__":
    main()
