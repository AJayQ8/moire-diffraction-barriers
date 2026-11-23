"""Extend the published map-mean diffraction width comparison to the four calibration maps."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from shared_compute import atomic_json, checkpoint_guard


TEST_MAPS = ("DS5", "DS7", "DS8", "DS11")


def detector_from_other_maps(calibration, held):
    training = {key: row for key, row in calibration["development_map_coefficients"].items() if key != held}
    by_model = [np.median([row[key] for row in training.values()], axis=0)
                for key in next(iter(training.values()))]
    coefficients = np.median(by_model, axis=0)
    if held not in TEST_MAPS:
        np.testing.assert_allclose(coefficients, calibration["frozen_common_coefficients"], atol=1e-12)
    if coefficients[0] <= 0:
        raise ValueError("The shared even detector gain must have positive sign")
    return coefficients, sorted(training)


def summarize(rows, methods, line):
    selected = [row for row in rows if all(all(row["fits"][key]["qualified"]) for key in methods)]
    if not selected:
        return {"walls": 0}
    gaps = {}
    absolute_widths = {}
    for key in methods:
        widths = np.array([row["fits"][key]["widths_nm"] for row in selected])
        absolute_widths[key] = widths.mean(axis=0).tolist()
        gaps[key] = float(np.mean(widths[:, 0]-widths[:, 1]))
    spacing = float(np.median([r["spacing_nm"] for r in selected]))
    gaps["spacing_only"] = float(np.polyval(line, spacing))
    return {"walls": len(selected), "median_spacing_nm": spacing,
            "first_minus_second_map_mean_nm": gaps, "map_mean_widths_nm": absolute_widths,
            "absolute_gap_error_nm": {k: abs(v-gaps["measured"]) for k,v in gaps.items() if k!="measured"}}


def score_map(name, directory, author, common, benchmark, settings, calibration, line):
    from author_detector_replay import detector_intensity
    from bilayer_relaxation import STACKING, elastic_parameters_per_reference_cell, make_grid
    from observe_bilayers import read_input_metadata
    from run_moire_common_intensity import centers_and_context
    from run_mos2_intensity_only_replay import disk_images
    from run_mos2_nonlinear_controls import minimize_all
    from run_mos2_matched_wall_models import BLUR_STEP_NM, corner_distances, isolated_profiles
    from run_mos2_observation_controls import CROP, HALF_WINDOW, NOMINAL, gaussian_gain, raster_coordinates, render_traces, sample_cuts
    from run_mos2_wall_landscape import width_pair
    from run_mos2_wall_width_transfer import DISTANCE, Profiles, geometry_sections
    from run_registry_prediction import rotation

    meta = read_input_metadata(name, directory)
    images, modes, payload_hash = disk_images(directory, name, author)
    saved = common["scans"][name]
    if meta["archive_md5"] != benchmark["scans"][name]["archive_md5"] or payload_hash != saved["diskset_sha256"]:
        raise ValueError(f"Source diffraction or archive checksum mismatch on {name}")
    interface = tuple(saved["interface"])
    if interface != ("MoS2", "MoS2", "P"):
        raise ValueError("Unexpected material or stacking")
    fields = meta["metadata_fields"]
    ctx = centers_and_context(images, modes, fields, settings, interface)
    if ctx["receipt"] != saved["geometry"]:
        raise ValueError(f"First-order junctions differ from the original geometry on {name}")
    basis, q = ctx["basis"], ctx["reciprocal"]
    material = (STACKING[interface], elastic_parameters_per_reference_cell(*interface[:2]))
    cell = np.linalg.solve(ctx["gradient"], basis)
    n = max(64, 2**int(np.ceil(np.log2(np.linalg.norm(cell, axis=0).max()/.3))))
    if n > 512:
        raise ValueError("Geometry exceeds the released grid range")
    grid = make_grid(ctx["gradient"], basis, n, *material[1])
    motion, solver = minimize_all(grid, material[0])
    coefficients, training = detector_from_other_maps(calibration, name)
    gain, background = coefficients
    camera = np.broadcast_to((gain, 0., background), (len(q), 3))
    rigid = grid.fractional @ basis.T
    spectra = {key: np.fft.fft2(detector_intensity(value, q, camera), axes=(0,1))
               for key, value in (("rigid", rigid), ("network", rigid + motion))}
    sections = geometry_sections(ctx, rotation(np.deg2rad(float(fields["SampleRotation"]))))
    rows = []
    for section in sections:
        checkpoint_guard()
        if section["reach_nm"] < HALF_WINDOW:
            continue
        observer = Profiles({**section, "keep": abs(DISTANCE)<=HALF_WINDOW}, q, camera, ctx["rings"],
                            NOMINAL, "sine_gordon")
        fits = {"measured": width_pair(observer, sample_cuts(images, section["coordinates"]).mean(axis=0))}
        freq, cuts, corners, weights = raster_coordinates(section, basis, ctx["receipt"]["pixel_nm"], n)
        blur = gaussian_gain(freq, NOMINAL)
        for key, spectrum in spectra.items():
            raster = render_traces(spectrum, blur, cuts, corners, weights)[1]
            fits[key] = width_pair(observer, raster[CROP])
        normal = corner_distances(section, basis, n, corners, weights)
        profile = isolated_profiles(section, q, camera, material, basis, normal, weights, BLUR_STEP_NM)[1]
        fits["isolated"] = width_pair(observer, profile[CROP])
        rows.append({"centers": section["centers"], "spacing_nm": section["spacing_nm"],
                     "miller_indices": [modes[j].tolist() for j in observer.channels], "fits": fits})
    return {"archive_md5": meta["archive_md5"], "integrated_data_sha256": payload_hash,
            "detector_training_maps": training, "even_detector_gain_and_background": coefficients.tolist(),
            "geometry_candidates": len(sections), "geometry_eligible": len(rows), "walls": rows,
            "twist_degrees": float(np.rad2deg(abs(ctx["gradient"][1,0]-ctx["gradient"][0,1])/2)),
            "solve": solver,
            "qualified_by_method": {key: sum(all(w["fits"][key]["qualified"]) for w in rows)
                                    for key in ("measured","network","isolated","rigid")},
            "primary": summarize(rows, ("measured","network","isolated"), line),
            "also_require_rigid": summarize(rows, ("measured","network","isolated","rigid"), line)}


def check_control(replay, original, landscape):
    original_by_wall = {tuple(w["centers"]): w for w in original["scans"]["DS10"]["walls"]}
    historical_channels = {tuple(w["centers"]): w["miller_indices"]
                           for w in landscape["scans"]["DS10"]["walls"]}
    matched = 0
    largest_difference = 0.
    for row in replay["walls"]:
        edge = tuple(row["centers"])
        if edge not in original_by_wall or row["miller_indices"] != historical_channels[edge]:
            continue
        prior = original_by_wall[edge]
        for key in ("measured", "network", "isolated"):
            old = prior["measured"] if key=="measured" else prior["models"][key]["scan_grid"]
            new = row["fits"][key]
            if new["qualified"] != old["qualified"]:
                raise ValueError("The common DS10 fits no longer have the same qualification")
            shift = np.max(np.abs(np.array(new["widths_nm"])-old["widths_nm"]))
            largest_difference = max(largest_difference, float(shift))
        matched += 1
    if not matched or largest_difference > .002:
        raise ValueError(f"DS10 replay failed: {matched} matching walls, {largest_difference} nm difference")
    return {"matched_reference_walls": matched, "max_absolute_width_difference_nm": largest_difference,
            "original_wall_count": len(original_by_wall)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("replay","archive","reader","dependencies","released_evidence","paper_evidence","output"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.replay.resolve()))
    from author_release import load_author_release
    names = {
        "calibration": args.paper_evidence/"mos2_camera_transfer_v01.json",
        "spacing": args.paper_evidence/"mos2_spacing_and_weighting_v01.json",
        "original": args.paper_evidence/"mos2_matched_wall_models_v01.json",
        "landscape": args.released_evidence/"mos2_wall_landscape_v01.json",
        "common": args.replay/"moire_common_intensity_replay_v01.json",
        "benchmark": args.replay/"moire_regime_benchmark_v02.json",
        "settings": args.replay/"registry_protocol_v01.json"}
    records = {key: json.loads(path.read_bytes()) for key, path in names.items()}
    train = records["spacing"]["primary_close_group_spacing_and_ols"]["cohort"]["map_rows"]
    line = np.polyfit([r["median_wall_spacing_nm"] for r in train],
                      [r["mean_first_minus_second_width_nm"]["measured"] for r in train], 1)
    result = {"input_sha256": {v.name: sha256(v.read_bytes()).hexdigest() for v in names.values()},
              "spacing_baseline": {"training_maps": [r["map"] for r in train],
                                   "slope_and_intercept": line.tolist()}, "maps": {}}
    reader = load_author_release(args.reader, args.dependencies)
    for name in ("DS10", *TEST_MAPS):
        checkpoint_guard()
        row = score_map(name, args.archive, reader, records["common"], records["benchmark"],
                        records["settings"], records["calibration"], line)
        if name == "DS10":
            result["historical_replay_DS10"] = check_control(row, records["original"], records["landscape"])
        else:
            result["maps"][name] = row
        atomic_json(args.output, result)
    for cohort in ("primary", "also_require_rigid"):
        rows = [v[cohort] for v in result["maps"].values() if v[cohort]["walls"]]
        if rows:
            result[cohort] = {
                "maps": len(rows), "walls": sum(r["walls"] for r in rows),
                "equal_map_mae_nm": {key: float(np.mean([r["absolute_gap_error_nm"][key] for r in rows]))
                                     for key in rows[0]["absolute_gap_error_nm"]}}
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
