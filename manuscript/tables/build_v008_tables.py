"""Generate the v008 values and tables from the two frozen score records."""

from hashlib import sha256
import json
import numpy as np

from build_v007_tables import EVIDENCE, TABLES, macro, row, rows, signed_range


TRANSFERRED = "transferred_even"
DISJOINT = "separate_disjoint"
PRIMARY = "archive_13.2"
ALTERNATIVE = "two_rotated_10"
LOW = {"DS1", "DS2", "DS13", "DS14"}
ORDER_NAMES = ("first", "second")


def two_values(values):
    return r"\ \text{and}\ ".join(f"{value:.2f}" for value in values)


def three_values(values):
    first, second, third = values
    return rf"{first:.2f},\ {second:.2f}\ \text{{and}}\ {third:.2f}"


def figure_values(examples):
    definitions = []
    for scan, label in (("DS14", "Wide"), ("DS10", "Close")):
        result = examples[scan]
        measured = [result["measured_widths_nm"][key] for key in ORDER_NAMES]
        predicted = [result["periodic_widths_nm"][key] for key in ORDER_NAMES]
        prefix = "Example" + label
        definitions.extend([
            macro(prefix + "Measured", two_values(measured)),
            macro(prefix + "Periodic", two_values(predicted)),
            macro(prefix + "BarrierRatio", f"{(measured[0] / measured[1])**2:.2f}"),
        ])
        if label == "Close":
            curves = result["model_curves"]
            bias = (predicted[0] - predicted[1]) - (measured[0] - measured[1])
            definitions.extend([
                macro("ExampleCloseGapBias", f"{bias:.2f}"),
                macro("ExampleCloseRigidShape", f"{curves['rigid']['position_transfer'][0]['error_by_ring'][1]:.2f}"),
                macro("ExampleClosePeriodicShape", f"{curves['periodic_anchor']['position_transfer'][0]['error_by_ring'][1]:.2f}"),
            ])
    return definitions


def window_values(data):
    variants = data["windows"]
    fit_maps = data["window_fits"]
    selected = {
        name: [wall["windows"] for wall in result["walls"]
               if all(all(fit["qualified"]) for window in wall["windows"].values()
                      for fit in (window["measured"], *window["models"].values()))]
        for name, result in fit_maps.items()
    }
    if len(fit_maps) != 10 or sum(len(item["walls"]) for item in fit_maps.values()) != 243:
        raise ValueError("The frozen ten-map native-raster section set is incomplete")
    if not all(selected.values()):
        raise ValueError("Every map must contribute qualified sections in all windows")
    by_window = {}
    centered = {key: [] for key in ("measured", TRANSFERRED, DISJOINT)}
    for name in variants:
        means = {}
        for scan, records in selected.items():
            gaps = {}
            for camera in centered:
                widths = np.array([(record[name]["measured"] if camera == "measured" else
                                    record[name]["models"][camera])["widths_nm"] for record in records])
                gaps[camera] = widths[:, 0] - widths[:, 1]
                if name == "4.25":
                    centered[camera].extend(gaps[camera] - gaps[camera].mean())
            means[scan] = {key: float(value.mean()) for key, value in gaps.items()}
        by_window[name] = [
            *[float(np.mean([record[key] for record in means.values()])) for key in centered],
            *[float(np.mean([mean["measured"] for scan, mean in means.items() if scan in group]))
              for group in (LOW, set(means) - LOW)],
        ]
    observed = np.asarray(centered["measured"])
    rmse = {cam: np.sqrt(np.mean((observed - centered[cam])**2)) for cam in (TRANSFERRED, DISJOINT)}
    baseline = np.sqrt(np.mean(observed**2))
    return [
        macro("NativeWindowCount", sum(map(len, selected.values()))),
        macro("NativeWindowMeasured", three_values([by_window[key][0] for key in variants])),
        macro("NativeWindowPredicted", three_values([by_window[key][1] for key in variants])),
        macro("NativeWindowCenteredRMSE", f"{rmse[TRANSFERRED]:.3f}"),
        macro("NativeWindowDisjointRMSE", f"{rmse[DISJOINT]:.3f}"),
        macro("NativeWindowBaselineRMSE", f"{baseline:.3f}"),
        *rows("NativeWindowRows", [row(name.capitalize(), *[f"{value:.3f}" for value in by_window[name]])
                                   for name in variants]),
        *rows("NativeWindowCounts", [row(scan, len(fit_maps[scan]["walls"]), len(selected[scan]))
                                     for scan in sorted(selected, key=lambda key: int(key[2:]))]),
    ]


def orientation_values(data):
    test = data["width_comparison"]
    even, disjoint = (test["overall"][camera] for camera in (TRANSFERRED, DISJOINT))
    def score(record, candidate):
        return record["equal_map_scores"][candidate]["equal_map_score_nm"]

    walls = test["per_map"]
    table = []
    for case in test["candidate_definitions"]:
        name = case["name"]
        axes = "Rotated on DS13, DS14" if name.startswith("two_rotated") else "Archived"
        table.append(row(axes, f"{case['barrier_meV_per_cell']:.1f}", f"{score(even, name):.3f}",
                         f"{score(disjoint, name):.3f}"))
    change = "alternate_minus_primary_equal_map_score_p05_p95_nm"
    omit = "leave_one_map_out_alternate_minus_primary_nm"
    audit = data["bragg_radius_audit"]
    groups = audit["group_comparison"]
    reduction = groups["DS13_DS14_minus_other_eight"]

    def wall_error(scan):
        values = walls[scan]["cameras"][TRANSFERRED]["candidates"]
        a, b = (values[name]["wall_order_mae_nm"] for name in (PRIMARY, ALTERNATIVE))
        return rf"{a:.2f}\ \text{{to}}\ {b:.2f}"

    return [
        macro("JointWidthCount", test["common_qualified_sections"]),
        macro("JointArchiveError", f"{score(even, PRIMARY):.3f}"),
        macro("JointRotatedError", f"{score(even, ALTERNATIVE):.3f}"),
        macro("JointDisjointArchiveError", f"{score(disjoint, PRIMARY):.3f}"),
        macro("JointDisjointRotatedError", f"{score(disjoint, ALTERNATIVE):.3f}"),
        macro("JointChangeRange", signed_range(even["primary_vs_alternate"][change])),
        macro("JointDisjointChangeRange", signed_range(disjoint["primary_vs_alternate"][change])),
        macro("JointOmitDSEven", f"{even['primary_vs_alternate'][omit]['DS13']:+.3f}"),
        macro("JointOmitDSDisjoint", f"{disjoint['primary_vs_alternate'][omit]['DS13']:+.3f}"),
        macro("JointWallDSOneThree", wall_error("DS13")),
        macro("JointWallDSOneFour", wall_error("DS14")),
        macro("RadiusFirstReduction", f"{-reduction['first']['percent_of_other_eight_mean']:.1f}"),
        macro("RadiusSecondReduction", f"{-reduction['second']['percent_of_other_eight_mean']:.1f}"),
        *rows("JointOrientationRows", table),
        *rows("SpotRadiusRows", [
            row(scan, *[f"{record['rings'][order]['median_detector_px']:.2f}" for order in ORDER_NAMES])
            for scan, record in sorted(audit["scans"].items(), key=lambda pair: int(pair[0][2:]))
        ]),
    ]


def main():
    paths = [EVIDENCE / file for file in ("mos2_native_wall_examples_v01.json",
                                           "mos2_orientation_candidates_v01.json")]
    raw = [path.read_bytes() for path in paths]
    native, orientation = map(json.loads, raw)
    report = [f"% {path.name} sha256: {sha256(payload).hexdigest()}"
              for path, payload in zip(paths, raw, strict=True)]
    report.extend(figure_values(native["scans"]))
    report.extend(window_values(native))
    report.extend(orientation_values(orientation))
    (TABLES / "v008_revision_values.tex").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
