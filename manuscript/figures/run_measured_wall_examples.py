#!/usr/bin/env python3
"""Replay two representative measured MoS2 wall profiles for Figure 1.

This is deliberately bounded to one chord from DS14 and one from DS10, with
one nominal probe setting. A third relaxation doubles the DS10 equilibrium
grid size to report a local mesh-refinement check.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


sys.dont_write_bytecode = True


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fit_isolated_observer(state, signal, profile_distance, distance):
    central = signal[np.abs(profile_distance) <= distance[-1]]
    target = state.shape(central)
    fits = [state.fit(target, [ring]) for ring in (0, 1)]
    return {
        "widths_nm": [float(row["width_nm"]) for row in fits],
        "centers_nm": [float(row["center_nm"]) for row in fits],
        "signed_gap_first_minus_second_nm": float(fits[0]["width_nm"] - fits[1]["width_nm"]),
        "qualified": [bool(row["qualified"]) for row in fits],
    }


def choose_representative_scans(widths):
    probe_names = ("0.75", "1.25", "1.75")
    comparison = {
        scan: row for scan, row in widths["scans"].items()
        if row["group"] == "parallel_comparison"
    }
    qualified = {}
    for scan, row in comparison.items():
        selected = []
        for index, wall in enumerate(row["walls"]):
            all_probes = all(
                all(fit["qualified"] for fit in wall["fits"][probe]["separate"])
                for probe in probe_names
            )
            if all_probes:
                first, second = wall["fits"]["1.25"]["separate"]
                selected.append({
                    "index": index,
                    "centers": tuple(int(x) for x in wall["centers"]),
                    "spacing_nm": float(wall["spacing_nm"]),
                    "width_first_nm": float(first["width_nm"]),
                    "width_second_nm": float(second["width_nm"]),
                    "gap_nm": float(first["width_nm"] - second["width_nm"]),
                    "fits": wall["fits"],
                })
        qualified[scan] = selected
    count = sum(len(rows) for rows in qualified.values())
    if count != 243:
        raise ValueError(f"Expected the archived 243 all-probe-qualified chords; found {count}")

    groups = {
        "lower_twist": [
            scan for scan, row in comparison.items()
            if row["inferred_twist_deg"] < 0.8 and qualified[scan]
        ],
        "higher_twist": [
            scan for scan, row in comparison.items()
            if row["inferred_twist_deg"] >= 1.18 and qualified[scan]
        ],
    }
    if (len(groups["lower_twist"]), len(groups["higher_twist"])) != (4, 6):
        raise ValueError(f"Unexpected lower/higher comparison-map counts: {groups}")

    representatives = []
    for group, scan_ids in groups.items():
        median_twist = float(np.median([
            comparison[scan]["inferred_twist_deg"] for scan in scan_ids
        ]))
        scan = min(
            scan_ids,
            key=lambda item: (
                abs(float(comparison[item]["inferred_twist_deg"]) - median_twist),
                -len(qualified[item]),
                item,
            ),
        )
        map_rows = qualified[scan]
        median_gap = float(np.median([row["gap_nm"] for row in map_rows]))
        chord = min(
            map_rows,
            key=lambda row: (abs(row["gap_nm"] - median_gap), row["centers"]),
        )
        representatives.append({
            "group": group,
            "group_median_twist_deg": median_twist,
            "scan": scan,
            "scan_twist_deg": float(comparison[scan]["inferred_twist_deg"]),
            "qualified_chords_in_scan": len(map_rows),
            "map_median_gap_nm": median_gap,
            **chord,
        })
    if [(row["scan"], row["centers"]) for row in representatives] != [
        ("DS14", (3, 7)), ("DS10", (1, 7))
    ]:
        raise ValueError("Deterministic representative scan/chord selection changed")
    return representatives, qualified, count


def run_selected_scans(
    *, args, project, author, protocol, source, benchmark, settings, reduced,
    width_rows, closure, representatives, count,
):
    from shared_compute import atomic_json, checkpoint_guard
    from bilayer_relaxation import (
        MODES, STACKING, elastic_parameters_per_reference_cell, relax,
    )
    from run_heterobilayer_sparse_geometry import solve_from_geometry
    from run_heterobilayer_template_transfer import design_matrix
    from run_moire_common_intensity import centers_and_context
    from run_mos2_directional_diffraction import wall_sectors
    from run_mos2_intensity_only_replay import disk_images
    from run_mos2_slow_readout_transfer import calibrate, render
    from run_mos2_wall_profile_closure import (
        PROFILE_DISTANCE, fields_on_chord, render_profiles, transfer,
    )
    from run_mos2_wall_width_transfer import DISTANCE, PRIMARY_PROBE, Profiles, actual_profiles, geometry_sections
    from run_registry_prediction import rotation
    from observe_bilayers import read_input_metadata

    source_hashes = {
        "replay_helper": sha256(Path(__file__)),
        "plot_source": sha256(args.plot_source),
        "practice_runner_wall_profile_closure": sha256(project / "run_mos2_wall_profile_closure.py"),
        "practice_runner_wall_width_transfer": sha256(project / "run_mos2_wall_width_transfer.py"),
        "bilayer_relaxation": sha256(project / "bilayer_relaxation.py"),
        "author_detector_replay": sha256(project / "author_detector_replay.py"),
        "profile_width_evidence": sha256(project / "mos2_wall_width_transfer_v01.json"),
        "profile_closure_evidence": sha256(project / "mos2_wall_profile_closure_v01.json"),
        "energy_readout_evidence": sha256(project / "moire_wall_energy_readout_v01.json"),
        "reduced_equilibrium_coefficients": sha256(project / "mos2_reduced_equilibrium_v01.json"),
        "geometry_source": sha256(project / "wall_heterostrain_v01.json"),
        "geometry_protocol": sha256(project / "wall_heterostrain_protocol_v01.json"),
        "intensity_benchmark": sha256(project / "moire_common_intensity_replay_v01.json"),
        "regime_settings": sha256(project / "moire_regime_benchmark_v02.json"),
        "author_release_zip": sha256(args.release.parents[1] / "release.zip"),
    }
    result = {
        "schema": "measured-wall-examples-v01",
        "scope": (
            "Two selected real MoS2 comparison walls: DS14 and DS10, one chord "
            "per scan, nominal 1.25 nm probe. No scan-wide fit, new contrast "
            "selection, parameter tuning, or additional equilibrium sweep."
        ),
        "selection": {
            "primary_population": "Exactly the 243 comparison-map chords whose separate first- and second-ring fits qualify at all three assumed probe FWHM values (0.75, 1.25, 1.75 nm).",
            "rule": (
                "Divide the ten comparison scans into the existing lower-twist "
                "(<0.8 deg) and higher-twist (>=1.18 deg) groups. Choose the "
                "scan nearest the group's median inferred twist; ties use the "
                "larger qualified-chord count, then scan ID. Within that scan "
                "choose the qualified chord nearest the scan's median signed "
                "first-minus-second apparent-width gap; ties use the sorted "
                "first-ring center indices."
            ),
            "qualifying_chord_count": count,
            "representatives": [
                {k: v for k, v in row.items() if k != "fits"}
                for row in representatives
            ],
        },
        "profile_transform": {
            "distance_step_nm": float(DISTANCE[1] - DISTANCE[0]),
            "measurement": (
                "Average linear-interpolated integrated-disk intensity across "
                "the three established cuts at longitudinal chord fractions "
                "0.42, 0.50, and 0.58."
            ),
            "camera_normalization": (
                "For each disk, divide measured counts by their peak-to-peak "
                "range on the existing support outside the three wall sectors. "
                "Camera coefficients are least-squares fit to the fixed "
                "three-star compact field on that same non-wall support."
            ),
            "profile_shape": (
                "On the retained geometry-limited profile window, remove a "
                "least-squares linear baseline independently for each disk, "
                "then divide by that channel's L2 norm. No sign flip or "
                "amplitude fit is applied. Plot one deterministic "
                "highest-contrast retained channel per ring, rescaled to unit "
                "L2 norm; the archived scores use all retained channels."
            ),
            "model_observation": (
                "Apply the existing detector intensity law, 5-point "
                "Gauss-Hermite quadrature along the wall, and the nominal "
                "Gaussian probe across the wall; then apply the same profile "
                "detrend and L2 normalization."
            ),
        },
        "source_sha256": source_hashes,
        "scans": {},
    }

    ds10_refinement = None
    for selection in representatives:
        scan = selection["scan"]
        checkpoint_guard()
        prior = source["scans"][scan]
        metadata = read_input_metadata(scan, args.archive)
        images, modes, diskset_sha256 = disk_images(args.archive, scan, author)
        if metadata["archive_md5"] != benchmark["scans"][scan]["archive_md5"]:
            raise ValueError(f"{scan}: archive checksum differs from saved benchmark")
        if diskset_sha256 != prior["diskset_sha256"]:
            raise ValueError(f"{scan}: diskset payload differs from saved source")

        interface = tuple(prior["interface"])
        metadata_fields = metadata["metadata_fields"]
        context = centers_and_context(images, modes, metadata_fields, settings, interface)
        if context["receipt"] != prior["geometry"]:
            raise ValueError(f"{scan}: first-ring geometry differs from saved source")
        scan_angle = np.deg2rad(float(metadata_fields["SampleRotation"]))
        sections = geometry_sections(context, rotation(scan_angle))
        section = sections[selection["index"]]
        if tuple(section["centers"]) != tuple(selection["centers"]):
            raise ValueError(f"{scan}: selected chord ordering changed")

        basis, nominal = context["basis"], context["rigid"]
        q, support = context["reciprocal"], context["support"]
        compact = np.asarray(reduced["scans"][scan]["coefficients_nm"]["three_stars"])
        compact_field = nominal + design_matrix(nominal, basis, MODES) @ compact
        calibration_mask = support & ~np.logical_or.reduce(
            tuple(wall_sectors(nominal, basis).values())
        )
        normalized, camera = calibrate(
            render(compact_field, np.ones(support.shape), q, context["probe"]),
            images,
            calibration_mask,
        )
        checkpoint_guard()
        grid, motion, relaxation = solve_from_geometry(context["gradient"], basis, interface)
        grid_n = int(motion.shape[0])
        material = {
            "anchor": (
                STACKING[interface],
                elastic_parameters_per_reference_cell(*interface[:2]),
            )
        }
        states, along_wall_weights = fields_on_chord(
            section, context, scan_angle, grid, motion, grid, motion,
            compact, material, PRIMARY_PROBE,
        )
        rendered = render_profiles(states, q, camera, along_wall_weights, PRIMARY_PROBE)
        raw_integrated = actual_profiles(images, section["coordinates"])
        measured = actual_profiles(normalized, section["coordinates"])
        observer = Profiles(section, q, camera, context["rings"], PRIMARY_PROBE, "sine_gordon")
        measured_shapes = observer.shape(measured)
        detrended = measured[observer.keep] - observer.trend @ measured[observer.keep]
        contrast_norm = np.linalg.norm(detrended, axis=0)
        channels = [np.asarray(items, dtype=int) for items in observer.channels]
        representative_channels = [
            int(max(index_set, key=lambda index: contrast_norm[index]))
            for index_set in channels
        ]
        selected_profiles = {}
        models = {}
        for ring in (0, 1):
            label = "first" if ring == 0 else "second"
            ids = observer.channels[ring]
            n_ids = len(ids)
            channel = representative_channels[ring]
            col = int(np.flatnonzero(ids == channel)[0])
            selected_profiles[label] = {
                "channel_indices": [int(x) for x in ids],
                "representative_channel_index": channel,
                "representative_mode_hk": np.asarray(modes[channel], dtype=float).tolist(),
                "representative_channel_selection": (
                    "Largest detrended measured-profile L2 amplitude among the "
                    "established retained channels for this ring."
                ),
                "raw_integrated_count_profiles_by_channel": {
                    str(int(ch)): raw_integrated[observer.keep, int(ch)].astype(float).tolist()
                    for ch in ids
                },
                "camera_normalized_profiles_by_channel": {
                    str(int(ch)): measured[observer.keep, int(ch)].astype(float).tolist()
                    for ch in ids
                },
                "unit_l2_shape_profiles_by_channel": {
                    str(int(ch)): (
                        measured_shapes[ring][:, j] * np.sqrt(n_ids)
                    ).astype(float).tolist()
                    for j, ch in enumerate(ids)
                },
                "plotted_measured_unit_l2_shape": (
                    measured_shapes[ring][:, col] * np.sqrt(n_ids)
                ).astype(float).tolist(),
            }

        for name in ("isolated_anchor", "periodic_anchor"):
            signal = rendered[name]
            transfer_rows = transfer(observer, measured_shapes, signal)
            first_fit = transfer_rows[0]
            center = float(first_fit["center_nm"])
            shifted = np.column_stack([
                np.interp(DISTANCE - center, PROFILE_DISTANCE, signal[:, channel])
                for channel in range(signal.shape[1])
            ])
            transformed = observer.shape(shifted)
            intensity = {}
            shapes = {}
            for ring in (0, 1):
                label = "first" if ring == 0 else "second"
                ch = representative_channels[ring]
                col = int(np.flatnonzero(observer.channels[ring] == ch)[0])
                intensity[label] = shifted[observer.keep, ch].astype(float).tolist()
                shapes[label] = (
                    transformed[ring][:, col] * np.sqrt(len(observer.channels[ring]))
                ).astype(float).tolist()
            models[name] = {
                "source_ring_for_position_fit": "first",
                "fitted_center_nm_first_ring_only": center,
                "same_center_applied_to_both_rings": True,
                "position_transfer": transfer_rows[0],
                "camera_normalized_intensity": intensity,
                "unit_l2_shape_by_ring": shapes,
            }

        width_wall = width_rows["scans"][scan]["walls"][selection["index"]]
        if any(
            not all(fit["qualified"] for fit in width_wall["fits"][probe]["separate"])
            for probe in ("0.75", "1.25", "1.75")
        ):
            raise ValueError(f"{scan}: selected wall fails an all-probe qualification")
        archived_apparent = closure["scans"][scan]["walls"][selection["index"]][
            "probe"
        ]["1.25"]["periodic_anchor"]["apparent_widths"]
        base_apparent = fit_isolated_observer(
            observer, rendered["periodic_anchor"], PROFILE_DISTANCE, DISTANCE,
        )
        for ring, saved in enumerate(archived_apparent):
            if abs(base_apparent["widths_nm"][ring] - saved["width_nm"]) > 1e-6:
                raise ValueError(f"{scan}: base apparent width differs from archived result")
        archived_transfer = closure["scans"][scan]["walls"][selection["index"]][
            "probe"
        ]["1.25"]["periodic_anchor"]["transfer"][0]
        if abs(models["periodic_anchor"]["fitted_center_nm_first_ring_only"]
               - archived_transfer["center_nm"]) > 1e-6:
            raise ValueError(f"{scan}: first-ring fit differs from archived closure")

        centers_rc = np.asarray(context["receipt"]["centers_rc_px"], dtype=float)
        first_center = centers_rc[int(section["centers"][0])]
        second_center = centers_rc[int(section["centers"][1])]
        cut_rc = np.asarray(section["coordinates"][1], dtype=float)
        origin_rc = 0.5 * (first_center + second_center)
        padding_px = max(2.0, 2.0 / float(context["receipt"]["pixel_nm"]))
        bound_points = np.vstack((first_center, second_center, cut_rc))
        row0 = max(0, int(np.floor(bound_points[:, 0].min() - padding_px)))
        row1 = min(images.shape[0], int(np.ceil(bound_points[:, 0].max() + padding_px)) + 1)
        col0 = max(0, int(np.floor(bound_points[:, 1].min() - padding_px)))
        col1 = min(images.shape[1], int(np.ceil(bound_points[:, 1].max() + padding_px)) + 1)
        image_channel = representative_channels[0]
        image_crop = images[row0:row1, col0:col1, image_channel].astype(float)
        image_quantiles = np.quantile(image_crop, (0.02, 0.98)).astype(float).tolist()
        pixel_nm = float(context["receipt"]["pixel_nm"])
        origin_xy_nm = origin_rc[::-1] * pixel_nm
        center_positions_rc = centers_rc[[int(section["centers"][0]), int(section["centers"][1])]]
        center_positions_xy_nm = center_positions_rc[:, ::-1] * pixel_nm
        cut_xy_nm = cut_rc[:, ::-1] * pixel_nm
        selected_data_fits = width_wall["fits"]["1.25"]["separate"]
        if abs(float(section["spacing_nm"]) - float(width_wall["spacing_nm"])) > 1e-9:
            raise ValueError(f"{scan}: geometry and saved width spacing disagree")
        row = {
            "twist_deg": float(width_rows["scans"][scan]["inferred_twist_deg"]),
            "row_spacing_nm": float(section["spacing_nm"]),
            "row_spacing_source": "geometry_sections; equality checked against width-transfer row",
            "selected_center_indices": [int(x) for x in section["centers"]],
            "selected_center_positions_rc_px": center_positions_rc.tolist(),
            "selected_center_positions_xy_nm": center_positions_xy_nm.tolist(),
            "pixel_size_nm": pixel_nm,
            "profile_window_half_width_nm": float(np.max(np.abs(DISTANCE[observer.keep]))),
            "archive_md5": str(metadata["archive_md5"]),
            "archive_sha256": sha256(args.archive / f"{scan}.zip"),
            "diskset_payload_sha256": diskset_sha256,
            "metadata_fields": {
                key: metadata_fields.get(key) for key in (
                    "Material", "Orientation", "LatticeConstant", "PixelSize",
                    "SampleRotation", "AvgMoireTwist", "AvgHeteroStrain",
                )
            },
            "widths_from_archived_separate_ring_fits_nm": {
                "first": float(selected_data_fits[0]["width_nm"]),
                "second": float(selected_data_fits[1]["width_nm"]),
                "signed_gap_first_minus_second": float(
                    selected_data_fits[0]["width_nm"] - selected_data_fits[1]["width_nm"]
                ),
                "center_fit_first": float(selected_data_fits[0]["center_nm"]),
                "center_fit_second": float(selected_data_fits[1]["center_nm"]),
                "qualified_at_all_three_probes": True,
            },
            "profile_distance_nm": [float(x) for x in DISTANCE[observer.keep]],
            "profile_channels": selected_profiles,
            "camera_coefficients_by_disk": np.asarray(camera).astype(float).tolist(),
            "camera_calibration_pixel_count": int(calibration_mask.sum()),
            "probe_fwhm_nm": float(PRIMARY_PROBE),
            "model_curves": models,
            "fixed_material_inputs": {
                "interface_key": [str(x) for x in interface],
                "stacking_coefficients_eV_per_cell": np.asarray(STACKING[interface]).astype(float).tolist(),
                "moduli_bulk_shear_eV_per_cell": np.asarray(
                    elastic_parameters_per_reference_cell(*interface[:2])
                ).astype(float).tolist(),
                "isolated_anchor_intrinsic_width_nm": float(
                    width_wall["isolated"]["isolated_anchor"]["width_nm"]
                ),
                "isolated_anchor_barrier_meV_per_cell": float(
                    width_wall["isolated"]["isolated_anchor"]["barrier_meV"]
                ),
            },
            "base_relaxation": {
                "grid_n": grid_n,
                "grid_shape": list(grid.fractional.shape[:2]),
                "success": bool(relaxation["success"]),
                "iterations": int(relaxation["iterations"]),
                "force_rms_eV_per_nm": float(relaxation["force_rms_eV_per_nm"]),
                "energy_eV_per_cell": float(relaxation["energy_eV_per_cell"]),
                "rigid_energy_eV_per_cell": float(relaxation["rigid_energy_eV_per_cell"]),
                "max_relative_shift_nm": float(relaxation["max_relative_shift_nm"]),
            },
            "first_ring_image": {
                "representative_channel_index": image_channel,
                "representative_mode_hk": np.asarray(modes[image_channel], dtype=float).tolist(),
                "raw_integrated_count_crop": image_crop.tolist(),
                "crop_bounds_rc_px_half_open": [row0, row1, col0, col1],
                "display_origin_xy_nm": origin_xy_nm.tolist(),
                "first_center_rc_px": first_center.tolist(),
                "second_center_rc_px": second_center.tolist(),
                "chord_xy_nm": center_positions_xy_nm.tolist(),
                "midpoint_cut_xy_nm": cut_xy_nm.tolist(),
                "midpoint_cut_rc_px": cut_rc.tolist(),
                "display_quantiles_raw_counts": image_quantiles,
                "display_note": "Raw integrated counts from the selected first-ring disk; crop and color stretch are for display only.",
            },
            "source_sha256": source_hashes,
        }
        print(
            scan,
            section["centers"],
            "s_nm", round(float(section["spacing_nm"]), 3),
            "w_nm", [round(float(x["width_nm"]), 4) for x in selected_data_fits],
            "grid", grid_n,
            "force_rms", f"{relaxation['force_rms_eV_per_nm']:.3e}",
            flush=True,
        )

        if scan == "DS10":
            n_refined = 2 * grid_n
            checkpoint_guard()
            refined_grid, refined_motion, refined_info = relax(
                context["gradient"], basis, interface, n_refined, checkpoint_guard,
            )
            refined_states, refined_weights = fields_on_chord(
                section, context, scan_angle, refined_grid, refined_motion,
                refined_grid, refined_motion, compact, material, PRIMARY_PROBE,
            )
            refined_rendered = render_profiles(
                refined_states, q, camera, refined_weights, PRIMARY_PROBE,
            )
            refined_apparent = fit_isolated_observer(
                observer, refined_rendered["periodic_anchor"], PROFILE_DISTANCE, DISTANCE,
            )
            ds10_refinement = {
                "scan": "DS10",
                "section_center_indices": [int(x) for x in section["centers"]],
                "grid_n_base": grid_n,
                "grid_n_refined": n_refined,
                "base_apparent_widths_nm": base_apparent["widths_nm"],
                "refined_apparent_widths_nm": refined_apparent["widths_nm"],
                "base_signed_gap_first_minus_second_nm": base_apparent["signed_gap_first_minus_second_nm"],
                "refined_signed_gap_first_minus_second_nm": refined_apparent["signed_gap_first_minus_second_nm"],
                "gap_change_refined_minus_base_nm": float(
                    refined_apparent["signed_gap_first_minus_second_nm"]
                    - base_apparent["signed_gap_first_minus_second_nm"]
                ),
                "base_force_rms_eV_per_nm": float(relaxation["force_rms_eV_per_nm"]),
                "refined_force_rms_eV_per_nm": float(refined_info["force_rms_eV_per_nm"]),
                "base_success": bool(relaxation["success"]),
                "refined_success": bool(refined_info["success"]),
                "base_iterations": int(relaxation["iterations"]),
                "refined_iterations": int(refined_info["iterations"]),
                "criteria": (
                    "Same measured geometry, material inputs, camera coefficients, "
                    "nominal probe, wall section, channel sets, profile window, "
                    "detector response, and isolated-shape width observer; only "
                    "the periodic-equilibrium grid N is doubled."
                ),
            }
        result["scans"][scan] = row

    if ds10_refinement is None:
        raise ValueError("The required DS10 grid-refinement check did not run")
    result["grid_refinement"] = ds10_refinement
    checkpoint_guard()
    atomic_json(args.output, result)
    print(f"saved {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--shared-compute", type=Path, required=True)
    parser.add_argument("--plot-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    sys.path.insert(0, str(args.shared_compute.resolve()))
    sys.path.insert(0, str(project))
    from author_release import load_author_release
    from run_moire_common_intensity import BENCHMARK as benchmark_path
    from run_moire_regime_benchmark import SETTINGS as settings_path
    from run_wall_heterostrain import PROTOCOL as protocol_path, SOURCE as source_path

    args.plot_source = args.plot_source.resolve()
    widths = json.loads((project / "mos2_wall_width_transfer_v01.json").read_text())
    closure = json.loads((project / "mos2_wall_profile_closure_v01.json").read_text())
    reduced = json.loads((project / "mos2_reduced_equilibrium_v01.json").read_text())
    representatives, _, count = choose_representative_scans(widths)
    author = load_author_release(args.release, args.dependencies)

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing evidence: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_selected_scans(
        args=args,
        project=project,
        author=author,
        protocol=json.loads(protocol_path.read_text()),
        source=json.loads(source_path.read_text()),
        benchmark=json.loads(benchmark_path.read_text()),
        settings=json.loads(settings_path.read_text()),
        reduced=reduced,
        width_rows=widths,
        closure=closure,
        representatives=representatives,
        count=count,
    )


if __name__ == "__main__":
    main()
