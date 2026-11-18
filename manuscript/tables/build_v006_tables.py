"""Build the frame, saddle and paired probe comparisons from the saved scores."""

import cmath
from hashlib import sha256
import json
from math import radians
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np


TABLES = Path(__file__).resolve().parent
PAPER = TABLES.parents[1]
sys.path.insert(0, str(PAPER / "analysis"))
from summarize_mos2_geometry_probe import (  # noqa: E402
    CAMERAS, FIXED, GEOMETRY_SEED, LOW, block_draws, error_stats, interval, qualified,
)


EVEN = "transferred_even"


def load(name):
    member = f"evidence/{name}.json"
    path = PAPER / member
    if path.is_file():
        return path.read_bytes()
    with ZipFile(PAPER / "moire_v005_sources.zip") as archive:
        return archive.read(member)


def row(*values):
    return " & ".join(map(str, values)) + r" \\"


def frame_rows(scans):
    for name in LOW:
        r = scans[name]
        pixel = r["held_pixels"]["rmse"]
        profile = r["wall_scores"][EVEN]["profile_error"]
        native, rotated = "nominal", "rotate_minus_30"
        yield row(
            name, r["wall_scores"][EVEN]["shape_sections"],
            f"{pixel[native]['mean']:.3f}", f"{pixel[rotated]['mean']:.3f}",
            f"{pixel[rotated]['relative_change_percent']:+.1f}",
            f"{profile[native]['mean']:.3f}", f"{profile[rotated]['mean']:.3f}",
        )


def saddle_rows(scans):
    for camera, calibration in zip(CAMERAS, ("Other maps", "Cuts excluded"), strict=True):
        for name in ("DS13", "DS14"):
            scores = scans[name]["wall_scores"][camera]
            for frame, label in (("nominal", "Archived"), ("rotated", "Rotated")):
                case = scores[frame]
                baseline, lower = [case["widths_nm"][energy] for energy in ("13.2", "10")]
                lo, hi = lower["delta_error_vs_nominal_p05_p95"]
                yield row(
                    name, calibration, label, case["qualified_common_width_sections"],
                    f"{baseline['mean_absolute_error']:.3f}", f"{lower['mean_absolute_error']:.3f}",
                    f"[{lo:+.3f}, {hi:+.3f}]",
                    f"{baseline['mean_absolute_per_wall_error']:.3f}",
                    f"{lower['mean_absolute_per_wall_error']:.3f}",
                )


def small_angle_shift(scans):
    shifts = []
    for name in ("DS13", "DS14"):
        model = scans[name]["solvers"]["nominal"]
        stretch = 1 + model["polar_mean_percent"] / 100
        gradient = cmath.rect(stretch, radians(model["twist_degrees"])) - 1
        for angle in (-.3, .3):
            shifted = abs(1 + gradient * cmath.rect(1, radians(angle)))
            shifts.append(100 * abs(shifted - stretch))
    return max(shifts)


def paired_high_twist(geometry):
    """Use the common width cohort in main Figure 3 and fixed probe settings."""
    scans = sorted(geometry["scans"].items(), key=lambda item: item[1]["twist_deg"])
    keys = [f"{value:g}" for value in geometry["gaussian_fwhm_nm"] if value <= 1.9]
    rng = np.random.default_rng(GEOMETRY_SEED)
    contrasts = {cam: [] for cam in CAMERAS}
    sections = 0
    for name, scan in scans:
        keep = []
        walls = scan["walls"]
        for idx, wall in enumerate(walls):
            fits = [wall["gaussian"][key] for key in keys]
            if all(qualified(r["measured"], *(r["models"][cam][k] for cam in CAMERAS for k in FIXED))
                   for r in fits):
                keep.append(idx)
        draw, _ = block_draws(walls, keep, rng)
        if name in LOW:
            continue
        sections += len(keep)
        for camera in CAMERAS:
            pair = []
            for probe, model in (("1.25", "anchor"), ("1.75", "raman_scale")):
                fits = [walls[i]["gaussian"][probe] for i in keep]
                measured = np.asarray([r["measured"]["widths_nm"] for r in fits])
                predicted = np.asarray([r["models"][camera][model]["widths_nm"] for r in fits])
                pair.append(error_stats(draw @ measured, draw @ predicted))
            contrasts[camera].append(pair[1] - pair[0])
    if sections != 94:
        raise ValueError(f"The published 94-section higher-twist cohort changed to {sections}")
    return {cam: interval(np.mean(draws, axis=0)) for cam, draws in contrasts.items()}


def main():
    names = ("mos2_elastic_frame_summary_v01", "mos2_saddle_control_summary_v01", "mos2_geometry_probe_v005")
    payloads = [load(name) for name in names]
    frame, saddle, native = [json.loads(payload) for payload in payloads]
    bounds = paired_high_twist(native)
    lines = [f"% {name}.json SHA-256 {sha256(payload).hexdigest()}"
             for name, payload in zip(names, payloads, strict=True)]
    for macro, rows in (
        ("FrameRows", frame_rows(frame["scans"])),
        ("SaddleRows", saddle_rows(saddle["scans"])),
    ):
        lines.extend([rf"\newcommand{{\{macro}}}{{%", *rows, "}"])
    lines.append(r"\newcommand{\SmallFrameEffect}{%.3f}" % small_angle_shift(frame["scans"]))
    for name, camera in zip(("Transferred", "Disjoint"), CAMERAS, strict=True):
        lo, hi = bounds[camera]
        lines.append(r"\newcommand{\FrameProbeRange%s}{[%+.3f,\ %+.3f]}" % (name, lo, hi))
    (TABLES / "v006_geometry_rows.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
