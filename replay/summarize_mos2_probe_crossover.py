"""Join the frozen wall scores and report which probe gives equal model error."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parent
MODELS = ("raman_shape", "raman_scale")
COLORS = {"transferred_even": "#286787", "separate_disjoint": "#b35937"}
LABELS = {"transferred_even": "Fixed camera response", "separate_disjoint": "Disjoint camera fits"}


def read(filename):
    return json.loads((PROJECT / filename).read_text())


def qualified(row):
    return all(row["measured"]["qualified"]) and all(all(row["models"][model]["qualified"]) for model in MODELS)


def map_width_loss(rows):
    observed = np.mean([r["measured"]["widths_nm"] for r in rows], axis=0)
    return np.array([np.mean(abs(np.mean([r["models"][m]["widths_nm"] for r in rows], axis=0)-observed))
                     for m in MODELS])


def map_shape_loss(rows):
    return np.array([np.mean([r["models"][m]["transfer_error"] for r in rows]) for m in MODELS])


def crossings(probes, difference):
    found = []
    for i in range(len(probes)-1):
        a, b = probes[i:i+2]
        fa, fb = difference[i:i+2]
        if fa == 0:
            found.append({"bracket_nm": [a, a], "linear_estimate_nm": a})
        elif fa * fb < 0:
            found.append({"bracket_nm": [a, b], "linear_estimate_nm": float(a + (b-a)*fa/(fa-fb))})
    return found


def joined_cameras(cameras, initial, sweep, probes):
    """Keep the known wall identities and use the same sections at every probe."""
    joined = {}
    for scan, item in sweep["scans"].items():
        previous = {tuple(w["centers"]): w["readouts"] for w in initial["scans"][scan]["walls"]}
        nominal = {tuple(w["centers"]): w["readouts"] for w in cameras["scans"][scan]["walls"]}
        rows = {}
        for wall in item["walls"]:
            key = tuple(wall["centers"])
            rows[key] = {
                camera: [
                    nominal[key][camera]["1.25"]["4.25"] if probe == 1.25 else
                    previous[key]["1.75"][camera] if probe == 1.75 else
                    wall["readouts"][f"{probe:g}"][camera]
                    for probe in probes
                ]
                for camera in COLORS
            }
        assert set(rows) == set(previous)
        joined[scan] = rows
    return joined


def summarize(joined, optics, probes):
    total_widths, total_profiles = 0, 0
    by_camera = {camera: {} for camera in COLORS}
    pupil_controls = {name: {camera: {} for camera in COLORS} for name in optics["ideal_pupils"]}
    # Use one qualification intersection for both cameras and all probe
    # widths. The full-profile comparison retains every original wall.
    for scan, data in joined.items():
        common = [key for key, r in data.items() if all(qualified(v) for rows in r.values() for v in rows)]
        total_widths += len(common)
        total_profiles += len(data)
        circular = {tuple(x["centers"]): x["optical_scores"] for x in optics["scans"][scan]["walls"]}
        assert set(circular) == set(data)
        pupil_common = [key for key in common if all(
            qualified(row) for styles in circular[key].values() for row in styles.values())]
        for camera in COLORS:
            by_camera[camera][scan] = {
                "widths": [map_width_loss([data[key][camera][i] for key in common]).tolist()
                           for i in range(len(probes))],
                "profiles": [map_shape_loss([row[camera][i] for row in data.values()]).tolist()
                             for i in range(len(probes))],
                "width_count": len(common), "profile_count": len(data),
            }
            for name in pupil_controls:
                pupil_controls[name][camera][scan] = {
                    "widths": map_width_loss([circular[key][name][camera] for key in pupil_common]).tolist(),
                    "profiles": map_shape_loss([circular[key][name][camera] for key in data]).tolist(),
                    "width_count": len(pupil_common), "profile_count": len(data),
                    "gaussian_nominal_widths": map_width_loss([data[key][camera][0] for key in pupil_common]).tolist(),
                    "gaussian_nominal_profiles": map_shape_loss([row[camera][0] for row in data.values()]).tolist(),
                }
    aggregate = {}
    for camera, scans in by_camera.items():
        aggregate[camera] = {}
        for key in ("widths", "profiles"):
            array = np.array([row[key] for row in scans.values()])
            series = array.mean(axis=0)
            difference = series[:, 1] - series[:, 0]
            aggregate[camera][key] = {
                "model_errors": series.tolist(), "higher_minus_lower_error": difference.tolist(),
                "crossings": crossings(probes, difference),
                "leave_one_map_out_crossings": {
                    scan: crossings(probes, np.diff(np.delete(array, i, axis=0).mean(axis=0), axis=-1)[:, 0])
                    for i, scan in enumerate(scans)
                }
            }
    pupil_summary = {}
    for kernel, cameras in pupil_controls.items():
        pupil_summary[kernel] = {}
        for camera, scans in cameras.items():
            pupil_summary[kernel][camera] = {key: np.mean([row[key] for row in scans.values()], axis=0).tolist()
                                            for key in ("widths","profiles","gaussian_nominal_widths",
                                                        "gaussian_nominal_profiles")}
        pupil_summary[kernel]["maps"] = len(scans)
        pupil_summary[kernel]["width_count"] = sum(row["width_count"] for row in scans.values())
        pupil_summary[kernel]["profile_count"] = sum(row["profile_count"] for row in scans.values())
    return {"probes_nm": probes, "model_order": MODELS, "maps": len(joined), "matched_width_walls": total_widths,
            "profile_walls": total_profiles, "gaussian": aggregate, "circular": pupil_summary,
            "ideal_pupils": optics["ideal_pupils"], "map_scores": by_camera}


def draw(summary, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.right": False,
                         "axes.spines.top": False, "axes.edgecolor": "#778690", "axes.labelcolor": "#273640",
                         "xtick.color": "#273640", "ytick.color": "#273640", "legend.frameon": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), layout="constrained")
    probes = summary["probes_nm"]
    titles = {
        "widths": f"(a) Wall widths · {summary['matched_width_walls']} matched sections",
        "profiles": f"(b) Full diffraction profiles · {summary['profile_walls']} sections"
    }
    units = {"widths": "Higher minus lower barrier error (nm)", "profiles": "Higher minus lower barrier error"}
    for ax, key in zip(axes, titles):
        ax.set_title(titles[key], loc="left", fontsize=12)
        ax.axhline(0., color="#64717a", linewidth=.9)
        ax.axvline(1.25, color="#93a2ac", linewidth=.8, linestyle="--")
        ax.axvspan(*((1.55, 1.60) if key == "widths" else (1.45, 1.55)), color="#e5edf2", zorder=0)
        for (camera, row), marker in zip(summary["gaussian"].items(), ("o","s")):
            ax.plot(probes, row[key]["higher_minus_lower_error"], marker=marker, markersize=3.5,
                    color=COLORS[camera], label=LABELS[camera])
        ax.set(xlim=(1.225,1.775), xlabel="Assumed Gaussian probe FWHM (nm)", ylabel=units[key],
               xticks=[1.25,1.35,1.45,1.55,1.65,1.75])
        ax.grid(axis="y", color="#dfe6e9", lw=.6)
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower left", fontsize=8.5)
    axes[1].text(.975, .95, "Positive values favor 13.2 meV\nNegative values favor 19.1 meV",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=8.5, color="#273640")
    fig.suptitle("The inferred MoS₂ barrier changes with probe blur", fontsize=15, color="#273640")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=PROJECT / "mos2_probe_crossover_summary_v01.json")
    cli.add_argument("--figure", type=Path, default=PROJECT / "mos2_probe_crossover_v01.png")
    args = cli.parse_args()
    controls = read("mos2_wall_landscape_camera_controls_v01.json")
    earlier = read("mos2_raman_probe_sensitivity_v01.json")
    sweep = read("mos2_raman_probe_crossover_v01.json")
    optics = read("mos2_circular_probe_check_v01.json")
    assert controls["scans"].keys() == earlier["scans"].keys() == sweep["scans"].keys() == optics["scans"].keys()
    probes = [1.25, *sweep["probes_nm"], 1.75]
    result = summarize(joined_cameras(controls, earlier, sweep, probes), optics, probes)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    draw(result, args.figure)


if __name__ == "__main__":
    main()
