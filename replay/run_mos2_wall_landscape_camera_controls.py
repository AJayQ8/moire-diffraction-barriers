"""Repeat the nominal-probe material comparison with independent camera controls."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import binary_dilation

from author_release import load_author_release
from bilayer_relaxation import MODES, sample_disregistry, stack_energy
from observe_bilayers import read_input_metadata
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_barrier_recovery import CURVATURE
from run_mos2_directional_diffraction import wall_sectors
from run_mos2_intensity_only_replay import disk_images
from run_mos2_nonlinear_controls import minimize_all
from run_mos2_slow_readout_transfer import calibrate, render
from run_mos2_wall_landscape import (CROP, NOMINAL_PROBE, capture_footprint, equilibria,
                                     get_observer, make_states, width_pair)
from run_mos2_wall_profile_closure import render_profiles, transfer
from run_mos2_wall_width_transfer import DISTANCE, REDUCED, actual_profiles, geometry_sections, isolated_width
from run_registry_prediction import rotation
from run_wall_heterostrain import SOURCE
from shared_compute import atomic_json, checkpoint_guard


CAMERA = PROJECT / "mos2_camera_transfer_v01.json"
METHOD = {
    "provenance": "Secondary check motivated by overlap between the spot-calibration region and the profile cuts in the first campaign. The same previously examined MoS2 maps, chords and channels are retained.",
    "transferred_even": "Keep the common even contrast learned on four separate development maps, set odd contrast to zero, and perform no optical fit on target maps. Gain and offset cancel in the detrended unit-shape score.",
    "separate_disjoint": "Fit an equally flexible per-reflection optical response for each candidate on the old camera region after excluding all pixels sampled by any wall cut and a two-probe-sigma buffer. Use the same transferred-even width observer on measured and all rendered traces. All candidates share the same fit pixels. No material parameter is estimated.",
    "raman_constraint": {
        "target_cm_inverse": 22.3,
        "source": "https://doi.org/10.1016/j.cap.2021.02.011",
        "sample": "Room-temperature Raman spectrum of 3R MoS2 on SiO2/Si. It is a separately acquired untwisted specimen, so transfer of its local curvature to the moire bilayers is an assumption.",
        "comparison": "Supply the shear-mode curvature either by uniform energy scaling or by changing the shape with both saddle and AA energy differences fixed. Evaluate both in diffraction with the same original stiffness and camera rules."
    },
    "probe_fwhm_nm": 1.25,
    "fit_windows": ["adaptive", "4.25", "2.50"]
}


def optical_checks(images, context, sections, grid, motions):
    rigid, basis, q = (context[key] for key in ("rigid", "basis", "reciprocal"))
    footprint = capture_footprint(sections, images.shape[:2])
    buffer_px = int(np.ceil(2 * context["probe"]))
    separation = binary_dilation(footprint, structure=np.ones((3, 3), dtype=bool), iterations=buffer_px)
    train = context["support"] & ~np.logical_or.reduce(tuple(wall_sectors(rigid, basis).values())) & ~separation
    receipt = {"nonoverlap_pixels": int(train.sum()), "buffer_px": buffer_px, "by_model": {}}
    if train.sum() < 30:
        receipt["failure"] = "Too few pixels outside the buffered footprints to fit the camera."
        return None, receipt
    coeff = {}
    for name in (*motions, "rigid"):
        field = rigid if name == "rigid" else sample_disregistry(grid, motions[name], rigid, grid.cell)
        terms = render(field, np.ones(train.shape), q, context["probe"])
        matrices = terms[train].transpose(1, 0, 2)
        condition = float(np.max(np.linalg.cond(matrices)))
        if not np.isfinite(condition) or condition > 30.0:
            receipt["failure"] = f"The buffered camera fit is ill-conditioned for {name}."
            return None, receipt
        normalized, coeff[name] = calibrate(terms, images, train)
        error = np.einsum("xycp,cp->xyc", terms, coeff[name]) - normalized
        receipt["by_model"][name] = {
            "condition_number_max": condition,
            "training_channel_rmse": np.sqrt(np.mean(error[train]**2, axis=0)).tolist(),
            "cut_channel_rmse": np.sqrt(np.mean(error[footprint]**2, axis=0)).tolist(),
            "coefficients": coeff[name].tolist(),
        }
    return coeff, receipt


def raman_landscapes(basis, material, physics, primary_motions):
    """Use the independent small-displacement frequency to fix AB curvature."""
    point = basis @ np.array([1/3, 1/3])
    g = 2 * np.pi * MODES @ np.linalg.inv(basis)
    relative_mass_kg = 0.5 * (95.95 + 2 * 32.06) * 1.66053906892e-27
    energy_density_unit = 0.1602176634  # eV/nm^2 to N/m
    light_cm_s = 29_979_245_800.
    frequency = METHOD["raman_constraint"]["target_cm_inverse"]
    target = (2 * np.pi * light_cm_s * frequency)**2 * relative_mass_kg / energy_density_unit

    def curvature(adhesion):
        assert np.allclose(adhesion[4:], 0., atol=1e-11)
        coefficient = -np.cos(g @ point) * np.repeat(adhesion[1:4], 3) / 1000
        matrix = (g.T * coefficient) @ g
        value = np.trace(matrix)/2
        np.testing.assert_allclose(matrix, value * np.eye(2), atol=1e-10, rtol=0.)
        return value

    anchor = material[0]
    at_anchor = curvature(anchor)
    scale = target / at_anchor
    perturbation = (target - at_anchor) / curvature(CURVATURE)
    alternatives = {"raman_scale": anchor * scale, "raman_shape": anchor + perturbation * CURVATURE}
    barrier = physics["barriers_meV_per_cell"]["anchor"]
    path = np.stack([point, basis[:, 0] / 2, np.zeros(2)])
    reference_energy = stack_energy(path, basis, anchor)[0]
    record = {"target_curvature_eV_nm2": target, "anchor_scale_to_raman": scale,
              "constant_barrier_perturbation_meV": perturbation, "models": {}}
    for name, coeff in alternatives.items():
        np.testing.assert_allclose(curvature(coeff), target, atol=1e-10, rtol=0.)
        edge = isolated_width(basis, basis[:, 0], coeff)[1] * 1000
        energy = stack_energy(path, basis, coeff)[0]
        if name == "raman_shape":
            np.testing.assert_allclose(edge, barrier, atol=1e-10, rtol=0.)
            np.testing.assert_allclose(energy - energy[0], reference_energy - reference_energy[0],
                                       atol=1e-12, rtol=0.)
        else:
            np.testing.assert_allclose(edge, scale * barrier, atol=1e-10, rtol=0.)
        record["models"][name] = {"coefficients_meV_per_cell": coeff.tolist(), "barrier_meV_per_cell": edge,
                                  "aa_to_ab_meV_per_cell": float(1000*(energy[2]-energy[0]))}

    for name in primary_motions:
        coeff = np.array(physics["stacking_coefficients_meV_per_cell"][name])
        k = curvature(coeff)
        record["models"][name] = {
            "barrier_meV_per_cell": physics["barriers_meV_per_cell"][name],
            "predicted_frequency_cm_inverse": np.sqrt(k * energy_density_unit / relative_mass_kg)/(2*np.pi*light_cm_s)
        }
    return alternatives, record


def compare_window(subsection, detector, rings, selector, nominal, observed, rendered):
    observer = get_observer(subsection, detector, nominal, rings, NOMINAL_PROBE, selector)
    measured = width_pair(observer, observed)
    target = observer.shape(observed)
    models = {}
    for name, signal in rendered.items():
        fit = width_pair(observer, signal[CROP])
        aligned = transfer(observer, target, signal)
        fit["transfer_error"] = [row["error_by_ring"][1 - i] for i, row in enumerate(aligned)]
        fit["alignment_at_bound"] = [row["at_position_bound"] for row in aligned]
        models[name] = fit
    return {"measured": measured, "models": models}


def perform_scan(context, images, modes, metadata, interface, compact, previous, even_terms):
    angle = np.deg2rad(float(metadata["SampleRotation"]))
    sections = geometry_sections(context, rotation(angle))
    if [s["centers"] for s in sections] != [w["centers"] for w in previous["walls"]]:
        raise ValueError("The frozen wall geometry differs from the first material comparison")
    grid, motions, material, physics = equilibria(context, interface)
    if physics["barriers_meV_per_cell"] != previous["physics"]["barriers_meV_per_cell"]:
        raise ValueError("A material candidate changed between camera comparisons")
    candidates, raman = raman_landscapes(context["basis"], material, physics, motions)
    for key, values in candidates.items():
        motions[key], raman["models"][key]["solver"] = minimize_all(grid, values)
    q, rings = context["reciprocal"], context["rings"]
    gain, offset = even_terms
    nominal = np.broadcast_to(np.array([gain, 0., offset]), (len(q), 3))
    own, diagnostic = optical_checks(images, context, sections, grid, motions)
    # Detrending and normalization remove any positive per-channel scale
    # and additive constant. Use the untouched integrated intensities.
    walls = []
    mode_keys = {tuple(row): i for i, row in enumerate(modes)}
    for section, source in zip(sections, previous["walls"], strict=True):
        checkpoint_guard()
        channels = [np.array([mode_keys[tuple(id_)] for id_ in ids], dtype=int) for ids in source["miller_indices"]]
        fields, weights = make_states(section, context, angle, grid, motions, compact, material, NOMINAL_PROBE)
        for name in candidates:
            fields[name] = sample_disregistry(grid, motions[name], fields["rigid"], grid.cell)
        observations = actual_profiles(images, section["coordinates"])
        predictions = {"transferred_even": render_profiles(fields, q, nominal, weights, NOMINAL_PROBE)}
        if own is not None:
            predictions["separate_disjoint"] = {
                model: render_profiles({model: field}, q, own[model], weights, NOMINAL_PROBE)[model]
                for model, field in fields.items()
            }
        readouts = {camera: {"1.25": {}} for camera in predictions}
        for window in METHOD["fit_windows"]:
            subset = section
            if window != "adaptive":
                radius = float(window)
                if section["reach_nm"] < radius:
                    continue
                subset = {**section, "keep": abs(DISTANCE) <= radius}
            for camera, profiles in predictions.items():
                readouts[camera]["1.25"][window] = compare_window(
                    subset, q, rings, channels, nominal, observations, profiles
                )
        walls.append({key: source[key] for key in ("centers", "midpoint_xy_nm", "spacing_nm", "original_cohort")}
                     | {"readouts": readouts})
    return {key: previous[key] for key in ("inferred_twist_deg", "pixel_nm", "centers_rc_px")} | {
        "camera_diagnostic": diagnostic, "physics": physics, "raman_constraint": raman, "walls": walls
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("release", type=Path)
    cli.add_argument("dependencies", type=Path)
    cli.add_argument("material_scores", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--resume", action="store_true")
    args = cli.parse_args()
    start = perf_counter()
    paths = SOURCE, BENCHMARK, SETTINGS, REDUCED, CAMERA, args.material_scores
    source, benchmark, settings, reduced, transferred, primary = (json.loads(p.read_text()) for p in paths)
    order = primary["protocol"]["scans"]
    if set(primary["scans"]) != set(order):
        raise ValueError("The initial material comparison is incomplete")
    for name, sha in primary["input_sha256"].items():
        if digest(PROJECT / name) != sha:
            raise ValueError(f"The initial material comparison input changed: {name}")
    checkpoints = {p.name: digest(p) for p in (Path(__file__), *paths, PROJECT / "run_mos2_barrier_recovery.py")}
    report = {"protocol": primary["protocol"] | {"camera_controls": METHOD},
              "input_sha256": checkpoints, "scans": {}}
    elapsed = 0.
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        report = json.loads(args.output.read_text())
        if report["input_sha256"] != checkpoints or not set(report["scans"]) <= set(order):
            raise ValueError("The camera-control checkpoint used a different source freeze")
        elapsed = report["seconds"]
    author = load_author_release(args.release, args.dependencies)
    for scan in order:
        if scan in report["scans"]:
            continue
        checkpoint_guard()
        saved = source["scans"][scan]
        meta = read_input_metadata(scan, args.archive)
        images, modes, fingerprint = disk_images(args.archive, scan, author)
        if fingerprint != saved["diskset_sha256"] or meta["archive_md5"] != benchmark["scans"][scan]["archive_md5"]:
            raise ValueError(f"The archive changed for {scan}")
        interface = tuple(saved["interface"])
        context = centers_and_context(images, modes, meta["metadata_fields"], settings, interface)
        if context["receipt"] != saved["geometry"]:
            raise ValueError(f"The first-ring geometry changed for {scan}")
        compact = np.asarray(reduced["scans"][scan]["coefficients_nm"]["three_stars"])
        report["scans"][scan] = perform_scan(
            context, images, modes, meta["metadata_fields"], interface, compact, primary["scans"][scan],
            transferred["frozen_common_coefficients"]
        )
        report["seconds"] = elapsed + perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, report)
        row = report["scans"][scan]["camera_diagnostic"]
        print(scan, "non-overlapping calibration pixels", row["nonoverlap_pixels"],
              "seconds", round(report["seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
