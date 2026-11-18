"""Build the profile-fit sensitivity and agreement tables from the saved evidence."""

import json
from hashlib import sha256
from pathlib import Path
from statistics import mean


TABLES = Path(__file__).resolve().parent
EVIDENCE = TABLES.parents[1] / "evidence"
EVEN = "transferred_even"
DISJOINT = "separate_disjoint"
CONDITIONS = {
    "nominal": "Known nominal probe",
    "variable_probe": "Variable probe and noise",
    "variable_probe_and_landscape": "Variable probe, noise and landscape",
}
BANDS = {
    "below_0.10": r"Below \(0.10\,\mathrm{nm}^{-1}\)",
    "0.10_to_0.30": r"\(0.10\)--\(0.30\,\mathrm{nm}^{-1}\)",
    "above_0.30": r"Above \(0.30\,\mathrm{nm}^{-1}\)",
}


def row(*values):
    return " & ".join(map(str, values)) + r" \\"


def macro(name, value):
    return rf"\newcommand{{\{name}}}{{{value}}}"


def rows(name, values):
    return [macro(name, "%\n" + "\n".join(values) + "\n")]


def signed_range(values):
    return f"[{values[0]:+.3f}, {values[1]:+.3f}]"


def agreement_tables(data):
    cases = []
    for group in data["groups"]:
        model = group["models"]["periodic"]
        entry = model["agreement_sensitivity"]["1.20"]
        misses = entry["among_agreeing_pairs"]
        cases.append(row(f"{group['spacing_nm']:.1f}", CONDITIONS[group["condition"]],
                         model["qualified_matched_order_pairs"], entry["agree_pairs"],
                         misses["either_order_misses_gt_20_percent"], misses["both_orders_miss_gt_20_percent"]))
    wide = next(group["models"]["periodic"] for group in data["groups"]
                if group["spacing_nm"] > 20 and group["condition"] == "variable_probe")
    sensitivity = wide["agreement_sensitivity"]
    primary = sensitivity["1.20"]
    ratios = [f"{sensitivity[s]['among_agreeing_pairs']['both_orders_miss_gt_20_percent']}/"
              f"{sensitivity[s]['agree_pairs']}" for s in ("1.10", "1.30")]
    example = data["case_example"]
    intervals = [example["fits"][side]["barrier_interval_meV_per_cell"] for side in ("first", "second")]
    return [*rows("AgreementRows", cases),
            macro("AgreementWideCount", primary["agree_pairs"]),
            macro("AgreementWideBothMiss", primary["among_agreeing_pairs"]["both_orders_miss_gt_20_percent"]),
            macro("AgreementWideSensitivity", " and ".join(r"\(" + ratio + r"\)" for ratio in ratios)),
            macro("AgreementExampleEstimate", f"{example['fits']['first']['barrier_meV_per_cell']:.1f}"),
            macro("AgreementExampleTruth", f"{example['truth_barrier_meV_per_cell']:.1f}"),
            macro("AgreementFirstInterval", f"[{intervals[0][0]:.1f}, {intervals[0][1]:.1f}]"),
            macro("AgreementSecondInterval", f"[{intervals[1][0]:.1f}, {intervals[1][1]:.1f}]")]


def cutoff_tables(data):
    all_maps, high = (data["groups"][name] for name in ("all", "high"))
    cases = []
    for cut, item in all_maps["cutoffs"].items():
        changes = sum(scan["cutoffs"][cut]["wall_lists_changed_by_order"][0] for scan in data["scans"].values())
        for camera, label in ((EVEN, "Other maps"), (DISJOINT, "Cuts excluded")):
            report = item[camera]
            assert report["relaxed_profile_wins"] == len(all_maps["scans"])
            shape = " / ".join(f"{report['profile_error'][name]:.3f}" for name in ("anchor", "rigid"))
            nominal = report["widths"]["1.25"]["models"]["anchor"]["width_error_nm"]
            high_scores = high["cutoffs"][cut][camera]["widths"]
            contrasts = [f"{high_scores[p]['raman_error_difference_nm']:+.3f} "
                         f"{signed_range(high_scores[p]['raman_paired_range_nm'])}" for p in ("1.25", "1.75")]
            cases.append(row(cut, label, changes, shape, f"{nominal:.3f}", *contrasts))
    return [*rows("ReflectionCutoffRows", cases), macro("CutoffCommonWalls", all_maps["width_walls"]),
            macro("CutoffHighWalls", high["width_walls"])]


def transfer_tables(data):
    cases = []
    for setting in (*map(str, data["defocus_settings_nm"]), *data["bands"]):
        test = data["settings"][setting]
        label = BANDS.get(setting, f"{setting} nm")
        cases.append(row(label, f"{test['transfer_at_0.2_cycles_per_nm']:.3f}",
                         f"{test['raman_error_difference_nm']:+.3f}", signed_range(test["raman_paired_range_nm"]),
                         "Unresolved" if test["preference"] == "unresolved" else test["preference"]))
    return [*rows("OpticalTransferRows", cases), macro("OpticalTransferWalls", data["width_walls"])]


def position_tables(data):
    cases = []
    for scan in ("DS13", "DS14", "DS10"):
        record = data["scans"][scan]
        folds = record["domain_intensity_fits"]["guarded"]["wall_intensity"]
        scores = record["detectors"][DISJOINT]["widths"][1]
        reference, guarded, physical = (scores["results"][name] for name in (
            "nominal", "guarded_residual", "guarded_physical"))
        before, after = (mean(fold[key] for fold in folds) for key in ("nominal", "warped"))
        cases.append(row(scan, scores["matched_walls"], f"{before:.3f} / {after:.3f}",
                         *(f"{record['mean_absolute_error_nm']:.3f}" for record in (reference, guarded, physical)),
                         signed_range(guarded["paired_absolute_error_change_nm"])))
    return rows("LowTwistPositionRows", cases)


def main():
    files = ("mos2_order_agreement_v01.json", "mos2_readout_sensitivity_summary_v01.json",
             "mos2_position_width_summary_v01.json")
    values = [(EVIDENCE / name).read_bytes() for name in files]
    agreement, readout, position = map(json.loads, values)
    header = [f"% {name} SHA-256 {sha256(data).hexdigest()}" for name, data in zip(files, values, strict=True)]
    content = [*header, *agreement_tables(agreement), *cutoff_tables(readout["cutoffs"]),
               *transfer_tables(readout["pupil"]), *position_tables(position)]
    (TABLES / "v007_readout_tables.tex").write_text("\n".join(content) + "\n")


if __name__ == "__main__":
    main()
