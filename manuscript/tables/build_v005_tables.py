"""Generate manuscript values and supplementary rows from the native-raster replay."""

from hashlib import sha256
import json
from pathlib import Path

TABLES = Path(__file__).resolve().parent
EVIDENCE = TABLES.parents[1] / "evidence"
EVEN, DISJOINT = "transferred_even", "separate_disjoint"
ORDER = ("DS1", "DS2", "DS14", "DS13", "DS12", "DS9", "DS10", "DS6", "DS4", "DS3")
ARCHIVE_GEOMETRY = {
    "DS1": (.369, .451), "DS2": (.523, .138), "DS3": (1.849, .546),
    "DS4": (1.814, .453), "DS6": (1.721, .849), "DS9": (1.353, .597),
    "DS10": (1.597, .930), "DS12": (1.193, .581), "DS13": (.856, .619), "DS14": (.818, .319),
}


def row(*entries):
    return " & ".join(map(str, entries)) + r" \\"


def build(s):
    even = s["native"][EVEN]
    disjoint = s["native"][DISJOINT]
    errors = even["width_error"]
    scores = even["pooled_profiles"]
    if scores["anchor_native_map_wins"] != 10 or disjoint["pooled_profiles"]["anchor_native_map_wins"] != 10:
        raise ValueError("The stated profile comparison changed")
    curves = s["gaussian"]["1.9"]["groups"]["high"]["curves"]
    out = {
        "NativeWidthSections": errors["all"]["walls"],
        "NativeAllError": f'{errors["all"]["mean_error_nm"]:.3f}',
        "NativeLowError": f'{errors["low"]["mean_error_nm"]:.3f}',
        "NativeHighError": f'{errors["high"]["mean_error_nm"]:.3f}',
        "NativeTwistContrast": f'{errors["low_minus_high_nm"]:.3f}',
        "NativeTwistInterval": r"\text{--}".join(f"{v:.3f}" for v in errors["low_minus_high_p05_p95_nm"]),
        "NativeWithoutDSOne": f'{errors["low"]["leave_one_map_out_nm"]["DS1"]:.3f}',
        "NativeDisjointLow": f'{disjoint["width_error"]["low"]["mean_error_nm"]:.3f}',
        "NativeDisjointHigh": f'{disjoint["width_error"]["high"]["mean_error_nm"]:.3f}',
        "NativeAnchorShape": f'{scores["anchor"]["native"]:.3f}',
        "NativeRigidShape": f'{scores["rigid"]["native"]:.3f}',
        "NativeProbeSections": s["gaussian"]["1.9"]["groups"]["all"]["sections"],
    }
    for model, tag in (("anchor", "Ref"), ("raman_shape", "RamanLower"), ("raman_scale", "RamanHigher")):
        c = curves[f"{EVEN}/{model}"]
        out[f"NativeHigh{tag}{'Best' if model == 'anchor' else ''}"] = f'{c["sampled_minimum_nm"]:.3f}'
        out[f"NativeHigh{tag}Probe"] = f'{c["best_setting"]:.2f}'
    macros = "\n".join(r"\newcommand{\%s}{%s}" % pair for pair in out.items())

    geometry = []
    for name in ORDER:
        inferred = s["scans"][name]
        published = ARCHIVE_GEOMETRY[name]
        mm = even["maps"][name]
        retained = [s["gaussian"][key]["coverage"][name]["common"] for key in ("1.75", "1.9", "2.2")]
        geometry.append(row(
            name, f'{inferred["twist_deg"]:.3f}', f"{published[0]:.3f}", f"{published[1]:.3f}",
            f'{inferred["polar_mean_strain_percent"]:+.3f}', mm["eligible_walls"], mm["walls"], *retained,
        ))
    totals = [s["gaussian"][key]["groups"]["all"]["sections"] for key in ("1.75", "1.9", "2.2")]
    geometry.extend([r"\midrule", row("Total", "", "", "", "", 243, errors["all"]["walls"], *totals)])

    minima = []
    for group in ("low", "high", "all"):
        case = s["gaussian"]["1.9"]["groups"][group]
        for camera, title in ((EVEN, "Other maps"), (DISJOINT, "Cuts excluded")):
            cells = []
            for model in ("anchor", "raman_shape", "raman_scale"):
                c = case["curves"][f"{camera}/{model}"]
                cells.append(f'{c["sampled_minimum_nm"]:.3f} ({c["best_setting"]:.2f})')
            contrast = case["contrasts"][camera]
            lo, hi = contrast["raman_best_setting_error_difference_p05_p95_nm"]
            minima.append(row(group.capitalize(), title, case["sections"], *cells, f"[{lo:+.3f}, {hi:+.3f}]",
                              ", ".join(f"{v:.2f}" for v in contrast["interpolated_crossings"])))

    pupil = s["ideal_pupil"]
    pupil_rows = []
    for i, z in enumerate(pupil["settings"]):
        errors_by_group = [pupil["groups"][k]["curves"][m]["mean_curve_nm"][i]
                           for k in ("low", "high") for m in ("anchor", "raman_shape", "raman_scale")]
        pupil_rows.append(row(z, *(f"{value:.3f}" for value in errors_by_group)))

    strain = []
    for name in ORDER[:4]:
        m = even["maps"][name]
        for model, values in m["models"].items():
            if model == "anchor_continuous":
                continue
            if model == "anchor":
                branch, case = "0", "full"
            else:
                branch, case = model.split("_", maxsplit=1)
            case = {"full": "Inferred", "no_mean": "Zero mean", "pure_twist": "Pure twist", "uniaxial": "Uniaxial"}[case]
            width, delta = (values[k] for k in ("width_nm", "predicted_minus_measured_nm"))
            strain.append(row(name, branch, case, m["walls"], *(f"{w:.2f}" for w in width),
                              *(f"{d:+.2f}" for d in delta), f'{values["mean_absolute_width_error_nm"]:.3f}'))
    return {
        "v005_native_values": macros,
        "v005_native_rows": "\n\n".join(
            r"\newcommand{\%s}{%%" % name + "\n" + "\n".join(lines) + "\n}"
            for name, lines in (("NativeMapRows", geometry), ("NativeProbeRows", minima),
                                ("NativePupilRows", pupil_rows), ("NativeStrainRows", strain))
        ),
    }


def main():
    payload = (EVIDENCE / "mos2_geometry_probe_summary_v005.json").read_bytes()
    checksum = sha256(payload).hexdigest()
    for name, contents in build(json.loads(payload)).items():
        (TABLES / f"{name}.tex").write_text(f"% mos2_geometry_probe_summary_v005.json SHA-256 {checksum}\n{contents}\n")


if __name__ == "__main__":
    main()
