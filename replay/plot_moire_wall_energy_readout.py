"""Check how isolated-wall fits to different diffraction orders change the inferred barrier."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parent
WIDTHS = PROJECT / "mos2_wall_width_transfer_v01.json"
PROFILES = PROJECT / "mos2_wall_profile_closure_v01.json"
SWEEP = PROJECT / "mos2_wall_period_scaling_v01.json"
PROBES = ("0.75", "1.25", "1.75")
NOMINAL = "1.25"
OBSERVED = "measured"
PERIODIC = "periodic_anchor"
ISOLATED = "isolated_anchor"
KINDS = (OBSERVED, PERIODIC, ISOLATED)
INK = "#172632"
BLUE = "#24638f"
ORANGE = "#a75428"
GRAY = "#656c78"


def load(path):
    return json.loads(path.read_text())


def scan_ratios(scans, forecasts):
    result = {}
    for scan, measured in scans.items():
        if measured["group"] != "parallel_comparison":
            continue
        fitted = forecasts[scan]
        if fitted["group"] != measured["group"]:
            raise ValueError(f"Comparison group changed in {scan}")
        retained = []
        for raw, prediction in zip(measured["walls"], fitted["walls"], strict=True):
            if raw["centers"] != prediction["centers"] or not np.isclose(raw["spacing_nm"], prediction["spacing_nm"]):
                raise ValueError(f"Unmatched wall chord in {scan}")
            widths = {}
            for probe in PROBES:
                pairs = {OBSERVED: raw["fits"][probe]["separate"],
                         **{name: prediction["probe"][probe][name]["apparent_widths"]
                            for name in (PERIODIC, ISOLATED)}}
                if all(len(pair) == 2 and all(p["qualified"] for p in pair) for pair in pairs.values()):
                    widths[probe] = {name: [p["width_nm"] for p in pair] for name, pair in pairs.items()}
            if len(widths) == len(PROBES):
                retained.append({"centers": raw["centers"], "spacing_nm": raw["spacing_nm"], "widths_nm": widths})
        if not retained:
            raise ValueError(f"No matched, qualified wall chords in {scan}")
        summary = {}
        for probe in PROBES:
            summary[probe] = {}
            for name in KINDS:
                pairs = np.array([row["widths_nm"][probe][name] for row in retained])
                # Fixed slip and stiffness give width proportional to barrier**(-1/2)
                # for the isolated wall profile used in the readout.
                ratio = np.exp(np.median(2*np.log(pairs[:, 0]/pairs[:, 1])))
                summary[probe][name] = float(ratio)
        result[scan] = {"twist_deg": measured["inferred_twist_deg"], "total_wall_candidates": len(measured["walls"]),
                        "qualified_wall_count": len(retained), "typical_barrier_ratio": summary, "chords": retained}
    return result


def group_summary(rows, probe):
    ratios = np.array([[row["typical_barrier_ratio"][probe][name] for name in KINDS] for row in rows])
    distances = abs(np.log2(ratios[:, 1:]/ratios[:, :1]))
    return {"maps": len(rows), "measured_ratio_range": [float(ratios[:, 0].min()), float(ratios[:, 0].max())],
            "mean_absolute_log2_ratio_error": dict(zip(KINDS[1:], distances.mean(axis=0).tolist(), strict=True)),
            "periodic_wins": int(np.sum(distances[:, 0] < distances[:, 1]))}


def observed_panel(ax, rows):
    angles = [row["twist_deg"] for row in rows]
    nominal = {name: [row["typical_barrier_ratio"][NOMINAL][name] for row in rows] for name in KINDS}
    measured = np.array(nominal[OBSERVED])
    probes = np.array([[row["typical_barrier_ratio"][probe][OBSERVED] for probe in PROBES] for row in rows])
    uncertainty = (measured-probes.min(axis=1), probes.max(axis=1)-measured)
    ax.axhline(1., linestyle=(0, (3, 3)), color="#bac2c9", linewidth=1)
    ax.errorbar(angles, measured, yerr=uncertainty, fmt="o", ms=5.6, linewidth=1.1, capsize=2.5,
                color=INK, ecolor="#8c98a3", label="Measured diffraction", zorder=5)
    ax.scatter(angles, nominal[PERIODIC], s=47, facecolor="white", edgecolor=BLUE, linewidths=1.55,
               label="Model with neighboring walls", zorder=4)
    ax.scatter(angles, nominal[ISOLATED], marker="x", s=36, color=GRAY, linewidths=1.25,
               label="Isolated-wall model", zorder=4)
    ax.set_yscale("log", base=2)
    ax.set_yticks([1, 2, 4, 8], ["1×", "2×", "4×", "8×"])
    ax.set_ylim(.83, 10.25)
    ax.set_xlabel("Twist angle (°)")
    ax.set_ylabel("Inferred barrier ratio\n(second ring ÷ first ring)")
    ax.set_title("a   Same walls, different inferred barriers", loc="left")
    ax.legend(loc="upper left", fontsize=8.8, frameon=False)


def controlled_panel(ax, sweep):
    spacing = [row["row_spacing_nm"] for row in sweep]
    true_width = [row["true_periodic_width_nm"] for row in sweep]
    widths = np.array([[fit["width_nm"] for fit in row["reconstruction_widths"]["periodic"]] for row in sweep])
    first, = ax.plot(spacing, widths[:, 0], "o-", color=BLUE, markersize=4.3, linewidth=1.6,
                     label="Apparent width, first ring")
    second, = ax.plot(spacing, widths[:, 1], "s-", color=ORANGE, markersize=4.2, linewidth=1.6,
                      label="Apparent width, second ring")
    true, = ax.plot(spacing, true_width, "o--", color=INK, markersize=4.1, linewidth=1.2,
                    label="Width in the simulated structure")
    ax.set_xlabel("Spacing between wall rows (nm)")
    ax.set_ylabel("Wall width (nm)")
    ax.set_ylim(1.45, 6.28)
    ax.set_title("b   Fixed-material simulation", loc="left")
    ax.legend(handles=(true, first, second), loc="upper right", frameon=False, fontsize=8.8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", type=Path)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    comparisons = scan_ratios(load(WIDTHS)["scans"], load(PROFILES)["scans"])
    rows = sorted(comparisons.values(), key=lambda row: row["twist_deg"])
    sweep = [row for key, row in load(SWEEP)["cases"].items() if key.startswith("pure_twist/")]
    sweep.sort(key=lambda row: row["row_spacing_nm"])
    if any(not all(v["qualified"] for v in row["reconstruction_widths"]["periodic"]) for row in sweep):
        raise ValueError("An ideal probe readout is outside the fitted width range")
    report = {"quantity": "Ratio of effective isolated-wall barrier scales inferred from the same chord on the second and first diffraction rings. Median log ratio per map, exponentiated.",
              "window_and_matching": "Use the saved adaptive wall windows. Keep only matched measured, periodic and isolated fits qualified at all three probe sizes. All comparison maps had been examined previously.",
              "limits": "The isolated-wall scaling holds slip, stiffness and normalized interaction shape fixed. Ratios quantify disagreement of inferred scales. Diffraction orders share the acquisition. Probe variation shows sensitivity and the simulation sweep describes an ideal microscope.",
              "input_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in (WIDTHS, PROFILES, SWEEP, Path(__file__))},
              "scans": comparisons,
              "lower_twist_maps": {probe: group_summary([row for row in rows if row["twist_deg"] < 1.], probe)
                                    for probe in PROBES},
              "higher_twist_maps": {probe: group_summary([row for row in rows if row["twist_deg"] >= 1.], probe)
                                     for probe in PROBES}}
    args.results.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    plt.rcParams.update({"font.size": 10.5, "axes.titleweight": "semibold", "axes.titlecolor": INK,
                         "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 5.35), layout="constrained")
    fig.suptitle("Wall spacing changes the interpretation of diffraction", fontsize=17, weight="semibold")
    fig.supxlabel("MoS₂ comparison maps. Whiskers span assumed probe diameters from 0.75 to 1.75 nm.",
                  fontsize=9.1, color="#4b5966")
    observed_panel(axes[0], rows)
    controlled_panel(axes[1], sweep)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#b5bec5")
        ax.grid(axis="y", color="#e8ecef", linewidth=.8)
        ax.set_axisbelow(True)
    fig.savefig(args.figure, dpi=210, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
