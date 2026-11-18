"""Test a frozen wall-width warning on new graphene diffraction profiles."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import QhullError

from bilayer_relaxation import make_grid
from run_graphene_wall_signal import assess_wall, gap, reconstruct_geometry
from run_graphene_wall_signal_repeat import repeatability
from run_heterobilayer_transfer import digest
from run_mos2_nonlinear_controls import minimize_all
from run_mos2_wall_width_transfer import DISTANCE
from shared_compute import atomic_json, checkpoint_guard


PROJECT = Path(__file__).resolve().parent
PROTOCOL = PROJECT / "wall_width_warning_protocol_v01.json"
SAMPLING = PROJECT / "wall_width_warning_sampling_v01.json"
WINDOWS = ("adaptive", "fixed")


def warning_statistics(pairs, chords, minimum):
    """Compare the signed ring gap with paired opposite-spot repeatability."""
    output = {"chords": chords, "matched_readouts": len(pairs), "minimum": minimum}
    if pairs:
        values = np.asarray(pairs)
        signed = float(values[:, 0].mean())
        disagreement = float(np.median(values[:, 1]))
        output.update({"mean_signed_gap_nm": signed, "median_paired_repeat_disagreement_nm": disagreement,
                       "gap_exceeds_repeatability": signed > disagreement if len(pairs) >= minimum else None})
    return output


def width_warning(window, minimum):
    pairs = []
    for row in window:
        observed = gap(row["measured"])
        opposite_a, opposite_b = map(gap, row["opposite_triplets"])
        if None not in (observed, opposite_a, opposite_b):
            pairs.append((observed, abs(opposite_a - opposite_b)))
    return warning_statistics(pairs, len(window), minimum)


def prediction_score(walls, minimum):
    """Use one ring for a local centering fit and score the other ring's shape."""
    eligible = [row for row in walls if gap(row["measured"]) is not None]
    output = {"chords": len(walls), "qualified_chords": len(eligible), "minimum": minimum}
    if not eligible:
        return output
    periodic = np.asarray([[row["models"]["periodic"]["transfer"][i]["error_by_ring"][1 - i] for i in (0, 1)]
                           for row in eligible]).mean(axis=0)
    isolated = np.asarray([[row["measured"][i]["error_by_ring"][1 - i] for i in (0, 1)]
                           for row in eligible]).mean(axis=0)
    output.update({"periodic_error": periodic.tolist(), "source_fitted_isolated_error": isolated.tolist(),
                   "fractional_error_reduction": [(float(1 - p/b) if b > 0. else None)
                                                  for p, b in zip(periodic, isolated, strict=True)],
                   "periodic_wins_both_ways": bool(np.all(periodic < isolated)) if len(eligible) >= minimum else None})
    return output


def evaluate(rows, config):
    outcomes = {}
    for name, diagnostic, test, minimum in (
            ("whole_scan", None, None, config["minimum_whole_scan_chords"]),
            ("left_to_right", "left", "right", config["minimum_half_scan_chords"]),
            ("right_to_left", "right", "left", config["minimum_half_scan_chords"])):
        warning = {window: width_warning([x["windows"][window] for x in rows
                                          if window in x["windows"] and (diagnostic is None or x["half"] == diagnostic)],
                                         minimum) for window in WINDOWS}
        comparisons = [metrics.get("gap_exceeds_repeatability") for metrics in warning.values()]
        target = [x["windows"]["adaptive"] for x in rows if test is None or x["half"] == test]
        verdict = all(comparisons) if None not in comparisons else None
        outcomes[name] = {"warning": verdict, "measurements": warning, "prediction": prediction_score(target, minimum)}
    return outcomes


def chord_side(section, width):
    # Only the samples within the adaptive fit window contribute to the score.
    # One native column clears the bilinear interpolation stencil on either side.
    columns = section["coordinates"][:, section["keep"], 1]
    center = (width - 1) / 2
    if columns.max() <= center - 1:
        return "left"
    if columns.min() >= center + 1:
        return "right"
    return "crosses_split"


def reliable_rings(signals, settings):
    return {name: len(values) >= settings["minimum_opposite_pairs_per_ring"]
            and bool(np.median(values) >= settings["median_opposite_correlation_min"])
            for name, values in signals.items()}


def measure_and_score(intensities, receipt, scan, rule, inputs):
    available = np.asarray(receipt["usable"], dtype=bool)
    correlation = [repeatability(aperture, available) for aperture in intensities]
    ring_quality = [reliable_rings(item, rule["instrument"]) for item in correlation]
    output = {"extraction": receipt,
              "apertures": [{"radius_px": radius, "opposite_intensity_correlation": corr,
                             "qualified_rings": quality}
                            for radius, corr, quality in zip(scan["aperture_radii_px"], correlation, ring_quality,
                                                            strict=True)]}
    if not ring_quality[0]["first"]:
        output["qualification_failure"] = "Primary first-ring repeatability is too weak for the geometry fit"
        return output
    pixel = scan["raster_pixel_nm"]
    try:
        context, sections, candidates = reconstruct_geometry(intensities[0], pixel, inputs["lattice_nm"],
                                                             scan["published_twist_degrees"], available)
    except (ValueError, QhullError) as error:
        output["qualification_failure"] = str(error)
        return output
    output["first_ring_geometry"] = context["receipt"]
    width = intensities.shape[2]
    output["candidate_chords"] = candidates
    output["selected_chords"] = len(sections)
    output["center_column_in_crop"] = (width - 1) / 2
    if not sections:
        output["qualification_failure"] = "First-ring geometry supplies no complete interior wall chord"
        return output
    if not any(all(entry.values()) for entry in ring_quality):
        output["qualification_failure"] = "Neither detector aperture resolves both diffraction orders"
        return output
    adhesion = np.r_[inputs["adhesion_even_coefficients_meV_per_cell"], 0., 0.]
    moduli = np.asarray(inputs["single_layer_bulk_shear_eV_per_cell"])/2
    grid = make_grid(context["gradient"], context["basis"], 96, *moduli)
    motion, output["equilibrium"] = minimize_all(grid, adhesion)
    half_window = rule["warning"]["fixed_half_window_nm"]
    for aperture, report in zip(intensities, output["apertures"], strict=True):
        if not all(report["qualified_rings"].values()):
            report["qualification_failure"] = "At least one diffraction order failed the opposite-spot check"
            continue
        rows = []
        for section in sections:
            checkpoint_guard()
            windows = {"adaptive": section}
            if section["reach_nm"] >= half_window:
                windows["fixed"] = {**section, "keep": abs(DISTANCE) <= half_window, "reach_nm": half_window}
            assessed = {name: assess_wall(chord, aperture, available, context, grid, motion, adhesion, moduli,
                                           scan["probe_fwhm_nm"], receipt["row_stride"])
                        for name, chord in windows.items()}
            rows.append({"half": chord_side(section, width), "windows": assessed})
        report["warnings_and_scores"] = evaluate(rows, rule["warning"])
        report["walls"] = rows
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan")
    parser.add_argument("extraction", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replay-protocol", type=Path, help="Explicit archival replay for a development scan")
    args = parser.parse_args()
    rule = json.loads(PROTOCOL.read_text())
    source_protocol = args.replay_protocol or PROTOCOL
    scan = json.loads(source_protocol.read_text())["scans"][args.scan]
    array_file = args.extraction / "intensity.npz"
    receipt_file = args.extraction / "receipt.json"
    receipt = json.loads(receipt_file.read_text())
    expected_url = f'https://zenodo.org/records/{scan["record"]}/files/{scan["file"]}?download=1'
    if (receipt["protocol_sha256"] != digest(source_protocol) or receipt["region_rc"] != scan["region_rc"]
            or receipt["input_url"] != expected_url or receipt["row_stride"] != scan.get("row_stride", 2)):
        raise ValueError("The extracted diffraction lineage changed")
    (r0, r1), (c0, c1) = receipt["region_rc"]
    scan_shape = (len(scan["aperture_radii_px"]), len(range(r0, r1, receipt["row_stride"])), c1 - c0, 12)
    with np.load(array_file) as saved:
        data = saved["intensity"]
    if data.shape != scan_shape or len(receipt["byte_ranges"]) != scan_shape[1]:
        raise ValueError("Incomplete aperture-integrated raster")
    settings = PROJECT / rule["graphene_inputs"]
    fixed_inputs = json.loads(settings.read_text())["inputs"]
    sampling = ({"raster_pixel_nm": fixed_inputs["raster_pixel_nm"]} if args.replay_protocol
                else json.loads(SAMPLING.read_text())[args.scan])
    scan = {**scan, **sampling}
    result = measure_and_score(data, receipt, scan, rule, fixed_inputs)
    paths = [Path(__file__), PROTOCOL, source_protocol, array_file, receipt_file, settings,
             PROJECT / "run_graphene_wall_signal.py", PROJECT / "run_mos2_wall_width_transfer.py",
             PROJECT / "run_graphene_wall_signal_repeat.py"]
    if args.replay_protocol is None:
        paths.append(SAMPLING)
    references = {p.name: digest(p) for p in paths}
    output = {"scan": args.scan, "archival_replay": args.replay_protocol is not None,
              "sampling": sampling, "input_sha256": references, **result}
    atomic_json(args.output, output)
    print(f"Scored {output.get('selected_chords', 0)} geometry-selected wall chords in {args.scan}", flush=True)


if __name__ == "__main__":
    main()
