"""Make the scan-grid geometry and probe figures from saved wall scores."""

import argparse
import json
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import numpy as np

from plot_moire_barrier_revision import BLUE, FIGURES, GRAY, ORANGE, WIDTH, article_style


EVEN = "transferred_even"
CURVES = {
    "anchor": ("Published", GRAY, "o"),
    "raman_shape": ("Raman, 13.2 meV", BLUE, "s"),
    "raman_scale": ("Raman, 18.9 meV", ORANGE, "^"),
}


def read(path, member=None):
    if member is None:
        return json.loads(path.read_text())
    with zipfile.ZipFile(path) as package:
        return json.loads(package.read(member))


def plot_nominal(data, output, saddle=None):
    fig, (left, right) = plt.subplots(ncols=2, figsize=(WIDTH, 2.7), layout="constrained")
    maps = data["native"][EVEN]["maps"]
    order = sorted(maps, key=lambda name: data["scans"][name]["twist_deg"])
    twist = np.array([data["scans"][name]["twist_deg"] for name in order])
    left.set_title("(a) Absolute fitted widths", loc="left")
    for i, color in enumerate((BLUE, ORANGE)):
        for measured, (shift, marker) in zip((True, False), ((-.009, "o"), (.009, "s")), strict=True):
            series = [maps[name] if measured else maps[name]["models"]["anchor"] for name in order]
            key = "measured_nm" if measured else "width_nm"
            bounds = "measured_p05_p95_nm" if measured else "width_p05_p95_nm"
            values = [row[key][i] for row in series]
            mask = [j for j, name in enumerate(order) if min(maps[name]["occupied_blocks"]) >= 2]
            limits = np.array([series[j][bounds] for j in mask])[:, :, i]
            left.vlines(twist[mask]+shift, limits[:, 0], limits[:, 1], color=color, linewidth=.8, alpha=.55)
            left.scatter(twist+shift, values, s=18, c=color if measured else "white", edgecolor=color,
                         marker=marker, linewidth=.85, zorder=3,
                         label=f"Order {i+1}, {'measured' if measured else 'predicted'}")
    left.set(xlabel="Inferred twist (°)", ylabel="Map-mean width (nm)")
    left.legend(loc="upper right", fontsize=6.9, ncol=2, handletextpad=.28, columnspacing=.64)
    left.set_ylim(top=max(left.get_ylim()[1], 5.65))
    if saddle is None:
        plot_projected_strains(right, maps, order)
    else:
        plot_rotated_width_errors(right, saddle)
    fig.savefig(output)
    plt.close(fig)


def plot_projected_strains(right, maps, order):
    right.set_title("(b) Strain supplied to relaxation", loc="left")
    low = [name for name in order if "0_no_mean" in maps[name]["models"]]
    positions = np.arange(len(low))
    modes = (("anchor", "Inferred", "o", GRAY), ("0_no_mean", "Zero mean", "s", BLUE),
             ("0_uniaxial", "Uniaxial", "^", ORANGE))
    for i, (key, label, marker, color) in enumerate(modes):
        for branch in (0, 5):
            tag = key if branch == 0 else ("5_full" if key == "anchor" else f"5_{key[2:]}")
            for j, name in enumerate(low):
                model = maps[name]["models"].get(tag)
                if model is None:
                    continue
                x = j + (i - 1) * .20 + (branch == 5) * .065
                if min(maps[name]["occupied_blocks"]) >= 2:
                    right.vlines(x, *model["mean_absolute_width_error_p05_p95_nm"], color=color,
                                 linewidth=.8, alpha=.55)
                right.scatter(x, model["mean_absolute_width_error_nm"], s=20, marker=marker,
                              c=color if branch == 0 else "white", edgecolors=color, linewidth=.85, zorder=3,
                              label=label if branch == 0 and j == 0 else None)
    labels = [name + (r"$^\dagger$" if min(maps[name]["occupied_blocks"]) < 2 else "") for name in low]
    right.set_xticks(positions, labels)
    right.set(ylabel="Error of map-mean widths (nm)")
    right.set_ylim(bottom=0, top=max(right.get_ylim()[1] + .1, 1.7))
    right.legend(loc="upper left", ncol=3, fontsize=6.9, handletextpad=.26, columnspacing=.56)


def plot_rotated_width_errors(axis, saddle):
    cases = (("nominal", "13.2"), ("nominal", "10"), ("rotated", "13.2"), ("rotated", "10"))
    positions = np.arange(len(cases))
    axis.set_title("(b) Low-twist width residuals", loc="left")
    axis.axhline(0, c=GRAY, lw=.7, zorder=1)
    for (map_name, marker, offset) in (("DS13", "o", -.045), ("DS14", "s", .045)):
        scores = saddle["scans"][map_name]["wall_scores"][EVEN]
        for order, (color, shift) in enumerate(((BLUE, -.105), (ORANGE, .105))):
            residuals = [scores[frame]["widths_nm"][energy]["predicted_minus_measured"][order]
                         for frame, energy in cases]
            axis.scatter(positions + offset + shift, residuals, c=color, marker=marker,
                         edgecolors="white", linewidths=.35, s=26, zorder=3)
        axis.scatter([], [], c=GRAY, marker=marker, s=24, label=map_name)
    axis.set_xticks(positions, ["13.2\nArchived", "10\nArchived", "13.2\nRotated", "10\nRotated"])
    axis.set(xlabel="Sliding barrier (meV per cell)", ylabel="Predicted − measured (nm)",
             xlim=(-.35, 3.35), ylim=(-.86, 1.43))
    axis.legend(loc="upper center", fontsize=7.0, ncol=2, handletextpad=.3, columnspacing=.8)


def plot_probes(data, output):
    fig, axes = plt.subplots(ncols=2, figsize=(WIDTH, 2.58), layout="constrained", sharey=True)
    sweep = data["gaussian"]["1.9"]
    for name, letter, axis in zip(("low", "high"), ("a", "b"), axes, strict=True):
        row = sweep["groups"][name]
        probes = sweep["settings"]
        axis.set_title(f"({letter}) {name.capitalize()}er twist, {row['sections']} wall sections", loc="left")
        for model, (label, color, marker) in CURVES.items():
            case = row["curves"][f"{EVEN}/{model}"]
            lo, hi = case["pointwise_p05_p95_nm"]
            axis.fill_between(probes, lo, hi, color=color, alpha=.13, linewidth=0)
            axis.plot(probes, case["mean_curve_nm"], color=color, linewidth=1.3, marker=marker,
                      markersize=3.3, label=label)
        axis.set(xlabel="Assumed Gaussian probe FWHM (nm)", xlim=(1.22, 1.93),
                 xticks=(1.25, 1.45, 1.65, 1.85))
    axes[0].set_ylabel("Error of map-mean widths (nm)")
    axes[0].legend(loc="upper left", fontsize=7., ncol=1, handlelength=1.7)
    fig.savefig(output)
    plt.close(fig)


def plot_raman_grid(data, output):
    fig, axes = plt.subplots(ncols=2, figsize=(WIDTH, 2.65), layout="constrained", sharey=True)
    grid = data["raman_family"]
    materials = data["scans"]["DS14"]["barriers_meV_per_cell"]
    candidates = sorted((name for name in materials if name != "anchor"), key=materials.get)
    energies = [materials[name] for name in candidates]
    values = [np.asarray([grid["groups"][group]["curves"][f"{EVEN}/{name}"]["mean_curve_nm"]
                          for name in candidates]) for group in ("low", "high")]
    vmax = max(np.max(scores) for scores in values)
    for axis, name, letter, scores in zip(axes, ("low", "high"), ("a", "b"), values, strict=True):
        probes = grid["settings"]
        image = axis.pcolormesh(probes, energies, scores, vmin=0., vmax=vmax, cmap="cividis", shading="nearest",
                                rasterized=True)
        axis.scatter(np.repeat(probes, len(energies)), np.tile(energies, len(probes)), s=2.5, c="white", alpha=.55)
        ix, iy = np.unravel_index(np.argmin(scores), scores.shape)
        axis.scatter(probes[iy], energies[ix], marker="+", s=40, c="#d0faff", linewidths=1.3)
        axis.set_title(f"({letter}) {name.capitalize()}er twist, {grid['groups'][name]['sections']} sections", loc="left")
        axis.set(xlabel="Assumed Gaussian probe FWHM (nm)", xticks=(1.25, 1.45, 1.65, 1.85))
    axes[0].set_ylabel("Saddle barrier (meV per cell)")
    fig.colorbar(image, ax=axes, label="Width error (nm)", fraction=.03)
    fig.savefig(output)
    plt.close(fig)


def plot_section_residuals(data, detailed, output):
    fig, axes = plt.subplots(ncols=3, figsize=(WIDTH, 2.58), layout="constrained")
    maps = data["native"][EVEN]["maps"]
    for name, scan in detailed["scans"].items():
        color = BLUE if scan["twist_deg"] < 1.0 else ORANGE
        walls = [scan["walls"][i] for i in maps[name]["width_cohort"]]
        for i, marker in enumerate(("o", "^")):
            measured = np.array([wall["nominal"][EVEN]["measured"]["widths_nm"][i] for wall in walls])
            predicted = np.array([wall["nominal"][EVEN]["models"]["anchor"]["native"]["widths_nm"][i] for wall in walls])
            axes[0].scatter(measured, predicted, color=color, marker=marker, s=8, alpha=.54, linewidth=0)
            axes[1].scatter([row["character"] for row in walls], predicted-measured,
                            color=color, marker=marker, s=8, alpha=.54, linewidth=0)
            theta = [min(row["scan_angle_deg"], 180-row["scan_angle_deg"]) for row in walls]
            axes[2].scatter(theta, predicted-measured, color=color, marker=marker, s=8, alpha=.54, linewidth=0)
    lo = min(axes[0].get_xlim()[0], axes[0].get_ylim()[0])
    hi = max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])
    axes[0].plot([lo, hi], [lo, hi], linewidth=.7, color=GRAY)
    axes[0].set(xlabel="Measured width (nm)", ylabel="Predicted width (nm)", xlim=(lo, hi), ylim=(lo, hi))
    axes[1].set(xlabel=r"Wall character $(\hat{\mathbf{s}} \cdot \mathbf{n})^2$", ylabel="Width residual (nm)")
    axes[2].set(xlabel="Wall angle to scan x (°)", ylabel="Width residual (nm)")
    for k, axis in enumerate(axes):
        axis.set_title(f"({chr(97+k)})", loc="left")
        if k:
            axis.axhline(0, linewidth=.7, color=GRAY)
    for color, label in ((BLUE, "Lower twist"), (ORANGE, "Higher twist")):
        axes[0].scatter([], [], c=color, s=14, label=label)
    for marker, label in (("o", "First order"), ("^", "Second order")):
        axes[2].scatter([], [], c=GRAY, s=14, marker=marker, label=label)
    axes[0].legend(fontsize=6.6, loc="upper left")
    axes[2].legend(fontsize=6.6, loc="upper center")
    fig.savefig(output)
    plt.close(fig)


def plot_geometry_and_energy(output):
    fig, (left, right) = plt.subplots(ncols=2, figsize=(WIDTH, 2.55), layout="constrained")
    left.set_title("(a) Wall cuts on the scan grid", loc="left")
    px, py = np.meshgrid(np.arange(0, 7.1, .5), np.arange(0, 5.1, .5))
    left.scatter(px.ravel(), py.ravel(), color="#c0c7cc", s=4, linewidth=0)
    centers = np.array([[.65, 1.40], [6.35, 3.60]])
    along = centers[1] - centers[0]
    normal = np.array([-along[1], along[0]]) / np.linalg.norm(along)
    left.plot(*centers.T, color=GRAY, linewidth=.8, linestyle="--")
    left.scatter(*centers.T, color=GRAY, s=18, label="Stacking centers")
    for fraction in (.42, .50, .58):
        mid = centers[0] + fraction * along
        cut = mid + np.array([-2.1, 2.1])[:, None] * normal
        left.plot(*cut.T, color=BLUE, linewidth=1, alpha=.82)
    point = centers.mean(axis=0) + .8 * normal
    square = np.floor(point / .5) * .5 + .5 * np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    left.plot(*square.T, color=ORANGE, linewidth=1.05)
    left.scatter(*point, marker="x", color=ORANGE, s=20, linewidth=.95)
    left.set(xlabel="Scan x (nm)", ylabel="Scan y (nm)", aspect="equal")
    left.legend(loc="upper left", fontsize=7.)

    right.set_title("(b) Sliding path through the saddle", loc="left")
    slip = np.linspace(-.5, .5, 241)
    x, y = .5 - slip/3, 2*slip/3
    xy = 2*np.pi*np.column_stack([x, y])
    h, k = xy.T
    shells = np.stack((np.cos(h)+np.cos(k)+np.cos(h+k),
                       np.cos(h+2*k)+np.cos(h-k)+np.cos(2*h+k),
                       np.cos(2*h)+np.cos(2*k)+np.cos(2*h+2*k)), axis=1)
    for label, color, linestyle, coefficients in (
        ("Published", GRAY, "-", (14.020, -2.542, -.884)),
        ("Raman, 13.2 meV", BLUE, "--", (17.703568, -6.225568, -4.567568)),
        ("Raman, 18.9 meV", ORANGE, "-", (20.040007, -3.633502, -1.263578)),
        ("Enaldiev, converted", "#496d59", ":", (13.604611, -3.456655, -1.728328)),
    ):
        energy = shells @ coefficients
        right.plot(slip, energy-energy[0], color=color, linestyle=linestyle, linewidth=1.3, label=label)
    right.set(xlabel="Fraction of partial slip from the saddle", ylabel="Relative energy (meV per cell)",
              xticks=(-.5, -.25, 0., .25, .5))
    right.legend(loc="upper left", fontsize=6.6, handlelength=1.4)
    right.set_ylim(0., 25.5)
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("walls", type=Path)
    parser.add_argument("--saddle", type=Path)
    parser.add_argument("--output", type=Path, default=FIGURES)
    args = parser.parse_args()
    data = read(args.summary)
    walls = read(args.walls, "evidence/mos2_geometry_probe_v005.json" if args.walls.suffix == ".zip" else None)
    article_style()
    if args.saddle:
        plot_nominal(data, args.output / "scan_grid_wall_widths_v016.pdf", read(args.saddle))
    else:
        plot_nominal(data, args.output / "native_wall_widths.pdf")
    plot_probes(data, args.output / "native_probe_curves.pdf")
    plot_raman_grid(data, args.output / "raman_barrier_probe_grid.pdf")
    plot_section_residuals(data, walls, args.output / "wall_width_residuals_v016.pdf")
    plot_geometry_and_energy(args.output / "wall_geometry_energy_v016.pdf")


if __name__ == "__main__":
    main()
