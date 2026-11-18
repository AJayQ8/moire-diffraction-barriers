"""Check Fourier-grid convergence for one frozen low-twist MoS2 wall."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


PAPER = Path(__file__).resolve().parents[1]
EVIDENCE = PAPER / "evidence"
SCAN = "DS14"
CENTERS = [3, 7]
ENERGIES = ("13.2", "19")
TRANSFERRED = "transferred_even"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--practice-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--release-src", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    practice = args.practice_dir.resolve()
    if not (practice / "bilayer_relaxation.py").is_file():
        raise FileNotFoundError(f"No pinned practice sources under {practice}")
    sys.path.insert(0, str(PAPER / "analysis"))
    sys.path.insert(0, str(practice))

    from author_detector_replay import detector_intensity
    from author_release import load_author_release
    from bilayer_relaxation import make_grid
    from observe_bilayers import read_input_metadata
    from run_heterobilayer_transfer import digest as source_digest
    from run_moire_common_intensity import BENCHMARK, centers_and_context
    from run_moire_regime_benchmark import SETTINGS
    from run_mos2_intensity_only_replay import disk_images
    from run_mos2_nonlinear_controls import minimize_all
    from run_mos2_observation_controls import (
        CROP, HALF_WINDOW, NOMINAL, gaussian_gain, raster_coordinates, render_traces, sample_cuts, score,
    )
    from run_mos2_wall_landscape import get_observer, isolated_width, width_pair
    from run_mos2_wall_width_transfer import DISTANCE, geometry_sections
    from run_registry_prediction import rotation
    from run_wall_heterostrain import SOURCE
    from shared_compute import atomic_json, checkpoint_guard

    start = perf_counter()
    example_path = EVIDENCE / "measured_wall_examples_v01.json"
    display_path = EVIDENCE / "mos2_native_wall_examples_v01.json"
    sweep_path = EVIDENCE / "mos2_saddle_sweep_v01.json"
    camera_path = EVIDENCE / "mos2_camera_transfer_v01.json"
    example = json.loads(example_path.read_text())["scans"][SCAN]
    display = json.loads(display_path.read_text())["scans"][SCAN]
    sweep = json.loads(sweep_path.read_text())
    cameras = json.loads(camera_path.read_text())
    source_data = json.loads(Path(SOURCE).read_text())
    settings = json.loads(Path(SETTINGS).read_text())
    archive = args.archive_dir.resolve() / f"{SCAN}.zip"
    metadata = read_input_metadata(SCAN, args.archive_dir)
    author = load_author_release(args.release_src, args.dependencies)
    images, modes, diskset_hash = disk_images(args.archive_dir, SCAN, author)
    source_record = source_data["scans"][SCAN]
    if metadata["archive_md5"] != example["archive_md5"]:
        raise ValueError("The DS14 archive does not match the frozen featured-wall record")
    if diskset_hash != example["diskset_payload_sha256"] or diskset_hash != source_record["diskset_sha256"]:
        raise ValueError("The DS14 integrated disk data differ from the frozen record")
    fields = metadata["metadata_fields"]
    if fields["SampleRotation"] != example["metadata_fields"]["SampleRotation"]:
        raise ValueError("The DS14 raster orientation differs from the frozen record")
    context = centers_and_context(images, modes, fields, settings, tuple(source_record["interface"]))
    if context["receipt"] != source_record["geometry"]:
        raise ValueError("The first-ring geometry differs from the pinned source record")
    basis, q, rings = (context[name] for name in ("basis", "reciprocal", "rings"))
    angle = np.deg2rad(float(fields["SampleRotation"]))
    sections = geometry_sections(context, rotation(angle))
    matching = [section for section in sections if section["centers"] == CENTERS]
    if len(matching) != 1:
        raise ValueError("The frozen representative chord is not unique")
    section = matching[0]
    if abs(section["spacing_nm"] - example["row_spacing_nm"]) > 1e-8:
        raise ValueError("The representative chord spacing changed")
    if display["selected_center_indices"] != CENTERS:
        raise ValueError("The display record points to a different wall")

    mode_index = {tuple(int(v) for v in mode): index for index, mode in enumerate(modes)}
    channels = [np.asarray([mode_index[tuple(mode)] for mode in display["profile_channels"][order]["all_mode_hk"]])
                for order in ("first", "second")]
    nominal_coefficients = np.asarray(cameras["frozen_common_coefficients"], dtype=float)
    detector_coefficients = np.broadcast_to(
        [nominal_coefficients[0], 0.0, nominal_coefficients[1]], (len(q), 3)
    )
    reader = get_observer({**section, "keep": abs(DISTANCE) <= HALF_WINDOW}, q,
                          detector_coefficients, rings, NOMINAL, channels)
    measured_trace = sample_cuts(images, section["coordinates"]).mean(axis=0)
    measured_fit = width_pair(reader, measured_trace)

    grid_refinement = json.loads(example_path.read_text())["grid_refinement"]
    normal_n = int(example["base_relaxation"]["grid_n"])
    if normal_n != int(grid_refinement["grid_n_refined"]):
        raise ValueError("The representative normal grid disagrees with the saved refinement record")
    saved_wall = next(row for row in sweep["scans"][SCAN]["walls"] if row["centers"] == CENTERS)
    measured = saved_wall["measured"]
    if measured_fit["qualified"] != measured["qualified"]:
        raise ValueError("The width observer qualification changed")
    np.testing.assert_allclose(measured_fit["widths_nm"], measured["widths_nm"], atol=1e-5, rtol=0)
    physics = example["fixed_material_inputs"]
    stiffness = physics["moduli_bulk_shear_eV_per_cell"]
    anchor_mev = np.asarray(sweep["scans"][SCAN]["landscapes"]["13.2"]["coefficients_meV_per_cell"])
    recorded_coefficients = np.asarray(physics["stacking_coefficients_eV_per_cell"])
    np.testing.assert_allclose(anchor_mev, recorded_coefficients, atol=1e-10, rtol=0)
    for energy in ENERGIES:
        recorded = np.asarray(sweep["scans"][SCAN]["landscapes"][energy]["coefficients_meV_per_cell"])
        calculated = anchor_mev + (13.2 - float(energy)) * np.array([.75, .5, -.25, -.5, 0., 0.])
        np.testing.assert_allclose(recorded, calculated, atol=1e-10, rtol=0)
    cell_length_nm = float(np.max(np.linalg.norm(np.linalg.solve(context["gradient"], basis), axis=0)))
    if 2 ** int(np.ceil(np.log2(cell_length_nm / .30))) > normal_n:
        raise ValueError("The recorded normal grid does not satisfy the stated 0.30 nm mesh rule")

    result_by_case = {}
    normal_grid_saved_widths = {}
    for n in (normal_n, 2 * normal_n):
        frequency_squared, cut_index, corner_index, weights = raster_coordinates(
            section, basis, context["receipt"]["pixel_nm"], n
        )
        gain = gaussian_gain(frequency_squared, NOMINAL)
        rigid_registry = None
        cases = {}
        for energy in ENERGIES:
            checkpoint_guard()
            coefficients = np.asarray(sweep["scans"][SCAN]["landscapes"][energy]["coefficients_meV_per_cell"])
            grid = make_grid(context["gradient"], basis, n, *stiffness)
            if rigid_registry is None:
                rigid_registry = grid.fractional @ basis.T
            motion, solver = minimize_all(grid, coefficients)
            spectrum = np.fft.fft2(
                detector_intensity(rigid_registry + motion, q, detector_coefficients), axes=(0, 1)
            )
            _, trace = render_traces(spectrum, gain, cut_index, corner_index, weights)
            fitted = score(reader, reader.shape(measured_trace), trace)
            if not all(fitted["qualified"]):
                raise ValueError(f"The {energy} meV fit failed at grid N={n}")
            predicted_gap = fitted["widths_nm"][0] - fitted["widths_nm"][1]
            measured_gap = measured["widths_nm"][0] - measured["widths_nm"][1]
            cases[energy] = {
                "solver": solver,
                "qualified": fitted["qualified"],
                "predicted_widths_nm": fitted["widths_nm"],
                "measured_minus_predicted_widths_nm": [m-p for m, p in zip(measured["widths_nm"], fitted["widths_nm"], strict=True)],
                "predicted_signed_gap_first_minus_second_nm": predicted_gap,
                "measured_signed_gap_first_minus_second_nm": measured_gap,
                "measured_minus_predicted_signed_gap_nm": measured_gap - predicted_gap,
                "cross_order_normalized_profile_error": fitted["transfer_error"],
                "width_fit_error": fitted["fit_error"],
            }
            if n == normal_n:
                reference = saved_wall["arms"][TRANSFERRED]["nominal"][energy]
                np.testing.assert_allclose(fitted["widths_nm"], reference["widths_nm"], atol=1e-5, rtol=0)
                normal_grid_saved_widths[energy] = reference["widths_nm"]
                if energy == "13.2":
                    np.testing.assert_allclose(fitted["transfer_error"], reference["transfer_error"], atol=1e-8, rtol=0)
            del spectrum, motion, grid
        result_by_case[str(n)] = cases

    changes = {}
    for energy in ENERGIES:
        base = result_by_case[str(normal_n)][energy]
        refined = result_by_case[str(2*normal_n)][energy]
        width_change = [b-a for a, b in zip(base["predicted_widths_nm"], refined["predicted_widths_nm"], strict=True)]
        profile_change = [b-a for a, b in zip(base["cross_order_normalized_profile_error"], refined["cross_order_normalized_profile_error"], strict=True)]
        residual = base["measured_minus_predicted_widths_nm"]
        changes[energy] = {
            "refined_minus_normal_predicted_widths_nm": width_change,
            "refined_minus_normal_predicted_signed_gap_nm": (
                refined["predicted_signed_gap_first_minus_second_nm"] - base["predicted_signed_gap_first_minus_second_nm"]
            ),
            "refined_minus_normal_cross_order_normalized_profile_error": profile_change,
            "absolute_width_change_over_normal_grid_experimental_width_residual": [
                abs(delta)/abs(error) if error else None for delta, error in zip(width_change, residual, strict=True)
            ],
            "absolute_gap_change_over_normal_grid_experimental_gap_residual": (
                abs(refined["predicted_signed_gap_first_minus_second_nm"] - base["predicted_signed_gap_first_minus_second_nm"])
                / abs(base["measured_minus_predicted_signed_gap_nm"])
            ),
        }
    paths = [
        Path(__file__), example_path, display_path, sweep_path, camera_path,
        Path(SOURCE), Path(BENCHMARK), Path(SETTINGS), archive,
        practice / "bilayer_relaxation.py", practice / "author_detector_replay.py",
        practice / "author_release.py", practice / "observe_bilayers.py",
        practice / "run_moire_common_intensity.py", practice / "run_mos2_intensity_only_replay.py",
        practice / "run_mos2_nonlinear_controls.py", practice / "run_mos2_wall_landscape.py",
        practice / "run_mos2_wall_width_transfer.py", practice / "run_registry_prediction.py",
        practice / "run_moire_regime_benchmark.py",
        PAPER / "analysis" / "run_mos2_saddle_sweep.py",
        PAPER / "analysis" / "run_mos2_observation_controls.py",
    ]
    source_hashes = {path.name: digest(path) for path in paths}
    expected_source_hash = example["source_sha256"].get("bilayer_relaxation")
    if expected_source_hash is not None and expected_source_hash != source_hashes["bilayer_relaxation.py"]:
        raise ValueError("The relaxation source differs from the archived featured-wall record")
    input_hashes = {
        **source_hashes,
        "author_release.zip": digest(args.release_src.resolve().parents[1] / "release.zip"),
    }
    output = {
        "schema": "mos2-grid-refinement-v01",
        "scope": "One frozen low-twist DS14 wall section [3,7], unchanged measured raster geometry, detector, 1.25 nm Gaussian probe, 4.25 nm fit window, channels, and profile cuts. Compare the published 13.2 meV/cell anchor with the upper tested 19 meV/cell saddle at the recorded N=128 mesh and N=256 doubled mesh.",
        "selection": {"scan": SCAN, "centers": CENTERS, "twist_deg": example["twist_deg"],
                      "spacing_nm": section["spacing_nm"], "normal_grid_n": normal_n,
                      "grid_spacing_upper_bound_nm": cell_length_nm/normal_n,
                      "high_saddle_meV_per_cell": 19.0,
                      "energy_family": "The frozen saddle-sweep direction preserves the AB Hessian and AA/AB energies."},
        "inputs": {"probe_fwhm_nm": NOMINAL, "half_window_nm": HALF_WINDOW,
                   "stiffness_bulk_shear_eV_per_cell": stiffness,
                   "frozen_stacking_coefficients_numeric_values": recorded_coefficients.tolist(),
                   "frozen_stacking_coefficients_source_field_label": "stacking_coefficients_eV_per_cell",
                   "frozen_stacking_coefficients_interpreted_units": "meV per primitive cell, as independently confirmed by the saved landscape and rigid energy",
                   "detector_coefficients_frozen_even": nominal_coefficients.tolist(),
                   "measured_widths_nm": measured["widths_nm"],
                   "measured_signed_gap_first_minus_second_nm": measured["widths_nm"][0]-measured["widths_nm"][1],
                   "measured_fit_error": measured["fit_error"]},
        "nominal_saved_score_verification": {
            "grid_n": normal_n, "energy_meV_per_cell": 13.2,
            "predicted_widths_nm": result_by_case[str(normal_n)]["13.2"]["predicted_widths_nm"],
            "saved_widths_nm": saved_wall["arms"][TRANSFERRED]["nominal"]["13.2"]["widths_nm"],
            "cross_order_normalized_profile_error": result_by_case[str(normal_n)]["13.2"]["cross_order_normalized_profile_error"],
            "saved_cross_order_normalized_profile_error": saved_wall["arms"][TRANSFERRED]["nominal"]["13.2"]["transfer_error"],
            "passed": True,
        },
        "normal_grid_saved_sweep_widths_nm": normal_grid_saved_widths,
        "cases_by_grid_n": result_by_case,
        "doubled_minus_normal": changes,
        "normal_grid_requirement": {"rule": "next power of two above longest cell-vector length / 0.30 nm, with minimum 64", "longest_cell_vector_nm": cell_length_nm, "n": normal_n},
        "interpretation_limit": "Conditional single-section numerical convergence check; it does not estimate scan-wide variability or validate the physical energy family.",
        "input_label_caveat": "The existing measured_wall_examples_v01.json uses the field name stacking_coefficients_eV_per_cell for numerical values that match the saved meV-per-cell Fourier coefficients; this runner uses the authoritative meV-per-cell arrays from the saddle-sweep evidence and does not edit the older record.",
        "input_sha256": input_hashes,
        "elapsed_seconds": perf_counter() - start,
    }
    checkpoint_guard()
    atomic_json(args.output.resolve(), output)
    print(f"{SCAN} {CENTERS}: N={normal_n}->{2*normal_n}; wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
