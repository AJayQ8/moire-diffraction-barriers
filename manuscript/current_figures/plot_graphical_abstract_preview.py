"""Compose the stand-alone graphical abstract from the saved diffraction records."""

import argparse
from hashlib import sha256
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Polygon, Rectangle
import numpy as np


PAPER = Path(__file__).resolve().parents[2]
OUT = PAPER / "submission" / "graphical_abstract_preview"
DATA_DOI = "https://doi.org/10.5281/zenodo.7865970"
DATA_LICENSE = "https://creativecommons.org/licenses/by/4.0/"
INK, BODY, MUTE = "#132D37", "#2D4750", "#49616A"
TEAL, CORAL, RULE = "#046D71", "#BB471F", "#CDDBDC"
W, H = 1200, 480
PT = 72 / 120
BASE = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])
S_OFFSET = BASE.sum(axis=0) / 3


def color_blend(start, end, weight):
    return np.array(to_rgb(start)) * (1 - weight) + np.array(to_rgb(end)) * weight


def text(ax, x, y, value, size, *, color=INK, weight="normal", **kwargs):
    return ax.text(x, y, value, fontsize=size * PT, color=color, fontweight=weight,
                   ha=kwargs.pop("ha", "left"), va=kwargs.pop("va", "center"),
                   zorder=20, **kwargs)


def line(ax, points, color, width, **kwargs):
    points = np.array(points)
    return ax.plot(points[:, 0], points[:, 1], color=color, lw=width * PT,
                   solid_capstyle="round", **kwargs)[0]


def sheet(ax, center, degrees, muted=False):
    # Trigonal-prismatic monolayer. Both sulfur planes have the same in-plane
    # sites; each Mo bonds to three sites in each plane. Projection is oblique.
    theta = np.deg2rad(degrees)
    rotation = np.array([[np.cos(theta), np.sin(theta)],
                         [-np.sin(theta), np.cos(theta)]])
    pitch = 23.0
    def project(pos, height):
        x, y = pos @ rotation
        return np.array([center[0] + pitch * x,
                         center[1] + pitch * (0.44 * y - 0.90 * height)])

    mo, sulfur = {}, {}
    for i in range(-5, 6):
        for j in range(-5, 6):
            if max(abs(i), abs(j), abs(i+j)) <= 5:
                mo[i, j] = np.array([i, j]) @ BASE
                sulfur[i, j] = mo[i, j] + S_OFFSET
    mo_color = "#667F87" if muted else "#1A6470"
    s_color = "#D1B87E" if muted else "#C99543"
    bond_color = "#9BAAAD" if muted else "#768A85"
    # Sort all elements in viewing depth, including the sulfur subplanes.
    objects = []
    for key, pos in mo.items():
        xy = project(pos, 0)
        depth = (pos @ rotation)[1]
        objects.append((0.90 * depth, "atom", xy, 5.05, mo_color))
        for offset in ((0, 0), (-1, 0), (0, -1)):
            site = (key[0]+offset[0], key[1]+offset[1])
            if site not in sulfur:
                continue
            for height in (-0.46, 0.46):
                endpoint = project(sulfur[site], height)
                mid_depth = 0.90 * (depth + (sulfur[site] @ rotation)[1]) / 2 + 0.44 * height / 2
                objects.append((mid_depth - 0.015, "bond", xy, endpoint, bond_color))
    for pos in sulfur.values():
        for height in (-0.46, 0.46):
            depth = 0.90 * (pos @ rotation)[1] + 0.44 * height
            c = color_blend(s_color, "#657883", .20) if height < 0 else s_color
            objects.append((depth, "atom", project(pos, height), 3.85, c))
    for _, kind, xy, other, col in sorted(objects, key=lambda x: x[0]):
        if kind == "bond":
            line(ax, (xy, other), col, 2.15, zorder=3)
        else:
            radius = other
            ax.add_patch(Circle(xy, radius, fc=color_blend(col, "#142E37", .14), ec="none", zorder=3))
            ax.add_patch(Circle(xy+[-.12*radius, -.15*radius], .81*radius, fc=col, ec="none", zorder=3))
            ax.add_patch(Circle(xy+[-.29*radius, -.33*radius], .37*radius,
                                fc=color_blend(col, "white", .37), ec="none", zorder=3))


def domain_network(ax, x, y, radius):
    # In parallel homobilayers, equivalent low-energy stackings alternate in
    # triangular domains. This sketch depicts wall topology, not a measured ROI.
    clip = Circle((x, y), radius, fc="white", ec="none", zorder=5)
    ax.add_patch(clip)
    a = np.array([32.0, -7.3])
    b = np.array([22.3, 24.0])
    origin = np.array([x-10, y+2])
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = origin + i*a + j*b
            for coords, fill in (([p, p+a, p+b], "#D2E9E4"),
                                 ([p+a, p+a+b, p+b], "#EBF3F1")):
                poly = Polygon(coords, fc=fill, ec="#5D918E", lw=1.55*PT, zorder=6)
                poly.set_clip_path(clip)
                ax.add_patch(poly)
    for i in range(-3, 4):
        for j in range(-3, 4):
            site = origin+i*a+j*b
            if np.linalg.norm(site-[x,y]) < radius-2:
                ax.add_patch(Circle(site, 2.0, fc="#427974", ec="none", zorder=7))
    segment = line(ax, [origin, origin+a], TEAL, 4.2, zorder=7)
    segment.set_clip_path(clip)
    ax.add_patch(Circle((x,y), radius, fc="none", ec="#5C8686", lw=1.2*PT, zorder=8))


def twist_arrow(ax):
    angles = np.deg2rad(np.linspace(147, 116, 30))
    points = np.column_stack((165 + 141*np.cos(angles), 190 + 68*np.sin(angles)))
    line(ax, points, TEAL, 1.9, zorder=8)
    direction = points[-1] - points[-3]
    direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    tip = points[-1] + 1.8*direction
    base = tip - 8*direction
    ax.add_patch(Polygon([tip,base+3.5*normal,base-3.5*normal],fc=TEAL,ec="none",zorder=8))


def measured_wall(ax, scan):
    record = scan["first_ring_image"]
    counts = np.asarray(record["raw_integrated_count_crop"])
    row0, row1, col0, col1 = record["crop_bounds_rc_px_half_open"]
    assert counts.shape == (row1-row0, col1-col0)
    left, top, width = 467.0, 154.0, 285.0
    magnification = width / (col1-col0)
    height = (row1-row0)*magnification
    right, bottom = left+width, top+height
    lo, hi = record["display_quantiles_raw_counts"]
    ax.imshow(counts, cmap="gray_r", vmin=lo, vmax=hi, origin="upper",
              extent=(left, right, bottom, top), interpolation="nearest", zorder=2)
    ax.add_patch(Rectangle((left,top),width,height,fc="none",ec="#B6C3C3",lw=.9*PT,zorder=4))
    pixel_nm = scan["pixel_size_nm"]
    indices = record["midpoint_cut_displayed_indices"]
    cut = np.array(record["midpoint_cut_xy_nm"])[indices]
    cut_xy = np.column_stack((left+(cut[:,0]/pixel_nm-col0+.5)*magnification,
                              top+(cut[:,1]/pixel_nm-row0+.5)*magnification))
    line(ax, cut_xy, "white", 5.5, zorder=6)
    line(ax, cut_xy, "#D94F22", 3.25, zorder=7)
    line(ax, [(left,top-18),(left+25,top-18)], "#D94F22", 3.25, zorder=7)
    text(ax, left+35, top-18, "Profile across wall", 17.5, color=CORAL)
    # Solid backing is a measurement annotation. Its line length is five nm.
    scale = 5 / pixel_nm * magnification
    bx, by = left+11, bottom-11
    ax.add_patch(Rectangle((bx-4.5,by-20),scale+9,24,fc="#FFFFFF",alpha=.92,ec="none",zorder=8))
    line(ax, [(bx,by),(bx+scale,by)], INK, 2.9, zorder=9)
    text(ax, bx+scale/2, by-11, "5 nm", 14.0, weight="bold", ha="center")
    return bottom


def error_chart(ax, scores, n_maps, n_walls):
    x, limit, max_nm = 816, 331, 1.5
    for y, key, label, color in ((207, "isolated", "Isolated wall", "#799094"),
                                 (299, "network", "Relaxed network", TEAL)):
        score = scores[key]
        text(ax, x, y-30, label, 20, weight="bold", color=BODY)
        text(ax, x+limit, y-34, f"{score:.2f} nm", 32, weight="bold", ha="right",
             color=TEAL if key=="network" else INK)
        ax.add_patch(Rectangle((x,y),limit*score/max_nm,10,fc=color,ec="none",zorder=3))
    line(ax, [(x,331),(x+limit,331)], RULE, 1.1)
    for tick in (0, 0.5, 1.0, 1.5):
        dx = x + limit*tick/max_nm
        line(ax, [(dx,331),(dx,335)], "#8CA3A4", 1.0)
        text(ax, dx, 347, f"{tick:g}", 14.5, color=MUTE, ha="center")
    text(ax, x, 383, f"{n_maps} maps with closely spaced walls", 18.5, weight="bold", color=BODY)
    text(ax, x, 408, f"{n_walls} matched wall segments", 18, color=MUTE)


def draw(output=OUT, include_additional=False):
    examples = PAPER / "evidence/mos2_native_wall_examples_v01.json"
    comparisons = PAPER / "evidence/mos2_matched_wall_models_summary_v01.json"
    scan = json.loads(examples.read_text())["scans"]["DS10"]
    matched = json.loads(comparisons.read_text())
    group = matched["groups"]["higher_twist"]["scan_grid"]
    assert group["matched_sections"]==168 and len(group["scans"])==6
    scores = group["map_mean_gap_absolute_error_nm"]
    scans = list(group["scans"])
    n_walls = group["matched_sections"]
    inputs = [examples, comparisons]
    if include_additional:
        inputs.append(PAPER / "evidence/mos2_additional_parallel_map_widths_v01.json")
        extra = json.loads(inputs[-1].read_text())
        errors = [matched["maps"][name]["models"]["scan_grid"]["map_mean_gap_absolute_error_nm"]
                  for name in scans]
        assert set(scans).isdisjoint(extra["maps"])
        errors.extend(row["primary"]["absolute_gap_error_nm"] for row in extra["maps"].values())
        scores = {key: float(np.mean([row[key] for row in errors])) for key in ("isolated", "network")}
        scans.extend(extra["maps"])
        n_walls += extra["primary"]["walls"]
    widths = scan["measured_widths_nm"]
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Liberation Sans","DejaVu Sans"],
                         "pdf.fonttype":42,"svg.fonttype":"none","svg.hashsalt":"moire-wall-abstract-v1",
                         "figure.facecolor":"white","savefig.facecolor":"white"})
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_axes((0,0,1,1))
    ax.set(xlim=(0,W),ylim=(H,0),aspect="equal")
    ax.set_axis_off()
    text(ax, 38, 42, "One domain wall, two apparent widths", 43, weight="bold")
    line(ax, [(39,77),(1160,77)], RULE, 1.3)
    text(ax, 39, 105, r"Twisted bilayer MoS$_{2}$", 25, weight="bold")
    text(ax, 466, 105, "First-order diffraction", 25, weight="bold")
    text(ax, 816, 105, "Error in the width difference", 24.5, weight="bold")
    text(ax, 816, 136, "Mean absolute error across map averages", 16.5, color=MUTE)
    # A separate scale schematic shows the real-space array of walls formed
    # by the layers. The leader does not designate a point in the measured crop.
    line(ax, [(235,190),(272,170),(316,170)], "#839D9C", 1.25, zorder=1)
    sheet(ax,(167,316),-32,muted=True)
    sheet(ax,(165,190),-12)
    twist_arrow(ax)
    domain_network(ax,354,195,49)
    text(ax,354,258,"Wall network",17.5,weight="bold",ha="center",color=BODY)
    for x, material, col in [(44,"Mo","#1A6470"),(109,"S","#C99543")]:
        ax.add_patch(Circle((x,394),5,fc=col,ec="none",zorder=8))
        text(ax,x+11,394,material,17.5,weight="bold",color=BODY)
    text(ax,40,417,"Schematic with enlarged twist and separation",15.5,color=MUTE)
    bottom=measured_wall(ax,scan)
    for x, label, value in ((467,"First-order width",widths["first"]),(622,"Second-order width",widths["second"])):
        text(ax,x,bottom+23,label,17.5,color=MUTE)
        text(ax,x,bottom+49,f"{value:.2f} nm",27.0,weight="bold",color=BODY)
    error_chart(ax,scores,len(scans),n_walls)
    line(ax, [(39,438),(1160,438)], RULE, 1.2)
    credit=text(ax,39,459,"4D-STEM data adapted from Van Winkle et al. (2023)",15.0,color=MUTE)
    credit.set_url(DATA_DOI)
    license_label=text(ax,1160,459,"CC BY 4.0",15.0,color=MUTE,ha="right")
    license_label.set_url(DATA_LICENSE)
    output.mkdir(parents=True,exist_ok=True)
    name=output/"Graphical_abstract"
    description=("Original schematics of twisted parallel MoS2 layers and their domain-wall network. "
                 "The DS10 image displays integrated first-order diffraction counts. The orange line is the "
                 "central transverse measurement across the wall. Its first- and second-order fitted widths "
                 f"are 4.16 and 2.83 nm. Across {len(scans)} closely spaced maps, the mean absolute error of the "
                 f"map-average first-minus-second fitted width is {scores['isolated']:.2f} nm for isolated walls and {scores['network']:.2f} nm "
                 "for the relaxed periodic network. DS10 crop adapted from Van Winkle et al., "
                 "doi:10.5281/zenodo.7865970, CC BY 4.0.")
    fig.savefig(name.with_suffix(".pdf"), dpi=300, metadata={"Author":"Ayas Alzanki",
                "Title":"One domain wall, two apparent widths","Subject":description,
                "CreationDate":None,"ModDate":None})
    fig.savefig(name.with_suffix(".svg"),metadata={"Title":"One domain wall, two apparent widths",
                "Creator":"Ayas Alzanki","Description":description,"Date":None})
    svg = name.with_suffix(".svg")
    svg.write_text("\n".join(row.rstrip() for row in svg.read_text().splitlines()) + "\n")
    fig.savefig(name.with_suffix(".png"),dpi=300,metadata={"Title":"One domain wall, two apparent widths",
                "Author":"Ayas Alzanki","Description":description})
    plt.close(fig)
    provenance = {
        "inputs_sha256": {p.name:sha256(p.read_bytes()).hexdigest() for p in inputs},
        "schematic": "Idealized parallel MoS2 layers in an exploded view; relative twist and separation enlarged. Separate triangular wall-network sketch at another scale.",
        "experimental_image": {"source_doi":DATA_DOI,"license":DATA_LICENSE,"scan":"DS10",
              "channel":"First-order integrated diffraction counts","pixel_size_nm":scan["pixel_size_nm"],
              "profile":"Central transverse cut of the three averaged profiles, plotted inside the fitted window.",
              "first_and_second_order_fitted_widths_nm":widths},
        "comparison": {"statistic":"Mean across maps of absolute error in each map's mean first-minus-second fitted width.",
              "scans":scans,"matched_wall_segments":n_walls,"error_nm":scores},
    }
    (output / "Graphical_abstract_inputs.json").write_text(json.dumps(provenance,indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--include-additional", action="store_true", help="Include the four additional closely spaced maps")
    args = parser.parse_args()
    draw(args.output, args.include_additional)
