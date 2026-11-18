"""Build the v009 saddle comparison from archived paired scores."""

from hashlib import sha256
import json

from build_v007_tables import EVIDENCE, TABLES, macro, row, rows, signed_range


CAMERAS = ("transferred_even", "separate_disjoint")
FRAMES = ("archived", "rotate_ds13_ds14")
FRAME_LABELS = ("Archived orientation", "Rotated on DS13 and DS14")


def setting_range(values):
    return f"[{values[0]:.1f}, {values[1]:.1f}]"


def build(sweep, orientation):
    settings = sweep["settings_meV_per_cell"]
    groups = (sweep["ten_maps"], sweep["leave_one_map_out"]["DS13"])
    original = orientation["width_comparison"]
    scores = original["overall"][CAMERAS[0]]["equal_map_scores"]
    table = [
        row(label, *[f"{scores[prefix + k]['equal_map_score_nm']:.3f}" for k in ("10", "13.2")])
        for label, prefix in zip(FRAME_LABELS, ("archive_", "two_rotated_"), strict=True)
    ]
    all_settings = [
        row(key, *[f"{group['cameras'][camera]['curves'][frame]['mean_curve_nm'][index]:.3f}"
                   for group in groups for camera in CAMERAS for frame in FRAMES])
        for index, key in enumerate(settings)
    ]
    fit_summary = []
    main = []
    for group, label, prefix in zip(groups, ("All ten maps", "DS13 omitted"), ("SweepAll", "SweepOmit"),
                                    strict=True):
        main.append(macro(prefix + "Sections", group["sections"]))
        for camera, detector in zip(CAMERAS, ("Other maps", "Cuts excluded"), strict=True):
            scores = group["cameras"][camera]
            archived, rotated = (scores["curves"][frame] for frame in FRAMES)
            interval = scores["best_setting_difference_p05_p95_meV_per_cell"]
            fit_summary.append(row(label, detector, archived["best_setting_meV_per_cell"],
                                   rotated["best_setting_meV_per_cell"], setting_range(interval),
                                   signed_range(scores["paired_range_nm_at_fixed_best_settings"])))
            if camera == CAMERAS[0]:
                main.extend([
                    macro(prefix + "Archived", archived["best_setting_meV_per_cell"]),
                    macro(prefix + "Rotated", rotated["best_setting_meV_per_cell"]),
                ])
                if prefix == "SweepAll":
                    main.extend([
                        macro("SweepAllArchivedError", f"{archived['sampled_minimum_nm']:.3f}"),
                        macro("SweepAllRotatedError", f"{rotated['sampled_minimum_nm']:.3f}"),
                        macro("SweepAllShiftRange", setting_range(interval)),
                        macro("SweepAllBestErrorRange", signed_range(scores["paired_range_nm_at_fixed_best_settings"])),
                    ])
    return [
        macro("SweepSettingCount", len(settings)),
        *main,
        *rows("EndpointComparisonRows", table),
        *rows("SweepCurveRows", all_settings),
        *rows("SweepMinimumRows", fit_summary),
    ]


def main():
    paths = [EVIDENCE / name for name in ("mos2_saddle_sweep_summary_v01.json",
                                         "mos2_orientation_candidates_v01.json")]
    inputs = [path.read_bytes() for path in paths]
    result = [f"% {path.name} sha256: {sha256(payload).hexdigest()}"
              for path, payload in zip(paths, inputs, strict=True)]
    result.extend(build(*map(json.loads, inputs)))
    (TABLES / "v009_saddle_tables.tex").write_text("\n".join(result) + "\n")


if __name__ == "__main__":
    main()
