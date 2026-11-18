"""Sweep the sliding saddle with the v008 probe, detector and wall geometry."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from author_detector_replay import detector_intensity
from author_release import load_author_release
from bilayer_relaxation import MODES, make_grid, stack_energy
from observe_bilayers import read_input_metadata
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_elastic_frame import CAMERAS, TARGETS, variants
from run_mos2_intensity_only_replay import disk_images
from run_mos2_nonlinear_controls import minimize_all
from run_mos2_observation_controls import (
    HALF_WINDOW, INPUT_NAMES, NOMINAL, gaussian_gain, raster_coordinates, render_traces, sample_cuts,
)
from run_mos2_opposite_widths import check_frozen
from run_mos2_wall_landscape import CROP, get_observer, width_pair
from run_mos2_wall_width_transfer import DISTANCE, geometry_sections, isolated_width
from run_registry_prediction import rotation
from run_wall_heterostrain import SOURCE
from shared_compute import atomic_json, checkpoint_guard


BARRIERS = ("9", "9.5", "10", "10.5", "11", "11.5", "12", "12.5",
            "13", "13.2", "13.5", "14", "14.5", "15",
            "15.5", "16", "16.5", "17", "17.5", "18", "18.5", "19")
ENDPOINTS = ("10", "13.2")
REFERENCE_ME_V = 13.2
ENDPOINT_CHECK_SCAN = "DS13"
REPLAY_MODULES = (
    "author_detector_replay", "author_release", "bilayer_relaxation",
    "run_moire_common_intensity", "run_mos2_elastic_frame", "run_mos2_nonlinear_controls",
    "run_mos2_intensity_only_replay", "run_mos2_observation_controls", "run_mos2_opposite_widths",
    "run_mos2_wall_landscape", "run_mos2_wall_width_transfer", "run_registry_prediction",
)


def saddle_family(basis, anchor):
    """The affine family through both v008 landscapes, in meV per cell."""
    if np.any(anchor[4:]):
        raise ValueError("The sweep assumes the even stacking energy of a parallel bilayer")
    # This direction changes neither V_AB, V_AA nor the Hessian at AB.
    per_mev_reduction = np.array([.75, .5, -.25, -.5, 0., 0.])
    family = {key: anchor + (REFERENCE_ME_V - float(key)) * per_mev_reduction for key in BARRIERS}
    ab, aa, sp = np.array([[1/3, 1/3], [0, 0], [.5, 0]]) @ basis.T
    points = np.stack([ab, aa, sp])
    reciprocal = 2 * np.pi * MODES @ np.linalg.inv(basis)
    cosines = np.cos(points @ reciprocal.T)

    def hessians(coefficients):
        weights = -cosines * np.repeat(coefficients[1:4], 3) / 1000
        return (reciprocal.T * weights[:, None, :]) @ reciprocal

    reference, _ = stack_energy(points, basis, anchor)
    stable_hessian = hessians(anchor)[0]
    records = {}
    for key, coeff in family.items():
        energy, force = stack_energy(points, basis, coeff)
        hessian = hessians(coeff)
        np.testing.assert_allclose(energy[:2], reference[:2], atol=1e-12, rtol=0)
        np.testing.assert_allclose(hessian[0], stable_hessian, atol=1e-12, rtol=0)
        np.testing.assert_allclose(force, 0., atol=1e-12, rtol=0)
        eigenvalues = np.linalg.eigvalsh(hessian[2])
        if not eigenvalues[0] < 0 < eigenvalues[1]:
            raise ValueError(f"{key} meV does not retain an index-one saddle")
        barrier = 1000 * isolated_width(basis, basis[:, 0], coeff)[1]
        np.testing.assert_allclose(barrier, float(key), atol=1e-10, rtol=0)
        records[key] = {
            "saddle_meV_per_cell": barrier,
            "coefficients_meV_per_cell": coeff.tolist(),
            "AA_minus_AB_meV_per_cell": float(1000 * (energy[1] - energy[0])),
            "stable_curvature_eV_nm2": float(np.trace(hessian[0]) / 2),
            "saddle_hessian_eigenvalues_eV_nm2": eigenvalues.tolist(),
        }
    return family, records


def make_spectra(scan, context, physics, family, detectors, energies):
    basis, gradient, q = (context[key] for key in ("basis", "gradient", "reciprocal"))
    frames = {"nominal": gradient}
    if scan in TARGETS:
        frames["rotated"] = variants(scan, gradient)["rotate_minus_30"][0]
    spectra, solvers = {}, {}
    for frame, deformation in frames.items():
        grid = make_grid(deformation, basis, physics["grid"], *physics["stiffness_eV_per_cell"])
        rigid = grid.fractional @ basis.T
        spectra[frame], solvers[frame] = {}, {}
        for key in energies:
            checkpoint_guard()
            motion, solvers[frame][key] = minimize_all(grid, family[key])
            spectra[frame][key] = {
                arm: np.fft.fft2(detector_intensity(rigid + motion, q, coeff), axes=(0, 1))
                for arm, coeff in detectors.items()
            }
    return spectra, solvers


def evaluate(scan, context, images, modes, metadata, original, camera, optics, frozen, prefix):
    basis, q, rings = (context[key] for key in ("basis", "reciprocal", "rings"))
    angle = np.deg2rad(float(metadata["SampleRotation"]))
    all_sections = geometry_sections(context, rotation(angle))
    old = original["walls"]
    if [s["centers"] for s in all_sections] != [wall["centers"] for wall in old]:
        raise ValueError("The measured wall geometry changed")
    supported = [(section, ref) for section, ref in zip(all_sections, old, strict=True)
                 if ref["original_cohort"] and section["reach_nm"] >= HALF_WINDOW]
    if [section["centers"] for section, _ in supported] != [wall["centers"] for wall in frozen["walls"]]:
        raise ValueError("The fixed width-cohort identities changed")
    if prefix is not None and [wall["centers"] for wall in prefix["walls"]] != [
        wall["centers"] for wall in frozen["walls"]
    ]:
        raise ValueError("The extension must use the same sections")

    physics = original["physics"]
    anchor = np.asarray(physics["stacking_coefficients_meV_per_cell"]["anchor"])
    family, landscapes = saddle_family(basis, anchor)
    np.testing.assert_allclose(family["10"], frozen["landscape"]["coefficients_meV_per_cell"], atol=1e-12, rtol=0)
    terms = optics["frozen_common_coefficients"]
    detector = np.broadcast_to([terms[0], 0., terms[1]], (len(q), 3))
    disjoint = np.asarray(camera["camera_diagnostic"]["by_model"]["anchor"]["coefficients"])
    detectors = dict(zip(CAMERAS, (detector, disjoint), strict=True))
    if prefix is None:
        previous_walls = frozen["walls"]
        computed = BARRIERS if scan == ENDPOINT_CHECK_SCAN else tuple(k for k in BARRIERS if k not in ENDPOINTS)
    else:
        previous_walls = prefix["walls"]
        computed = tuple(k for k in BARRIERS if k not in prefix["landscapes"])
    spectra, solvers = make_spectra(scan, context, physics, family, detectors, computed)
    if prefix is not None:
        for frame, records in prefix["solvers"].items():
            solvers[frame] = {**records, **solvers[frame]}

    mode_index = {tuple(mode): i for i, mode in enumerate(modes)}
    walls = []
    for (section, ref), base in zip(supported, previous_walls, strict=True):
        checkpoint_guard()
        channels = [np.asarray([mode_index[tuple(mode)] for mode in group]) for group in ref["miller_indices"]]
        view = {**section, "keep": abs(DISTANCE) <= HALF_WINDOW}
        reader = get_observer(view, q, detector, rings, NOMINAL, channels)
        if scan == ENDPOINT_CHECK_SCAN and prefix is None:
            measured = sample_cuts(images, section["coordinates"]).mean(axis=0)
            check_frozen(width_pair(reader, measured), base["measured"])
        freq2, cuts, corners, weights = raster_coordinates(section, basis, original["pixel_nm"], physics["grid"])
        gain = gaussian_gain(freq2, NOMINAL)
        arms = {camera_name: {} for camera_name in CAMERAS}
        for frame, cases in spectra.items():
            for camera_name in CAMERAS:
                fits = dict(base["arms"][camera_name][frame])
                for key, cameras in cases.items():
                    _, trace = render_traces(cameras[camera_name], gain, cuts, corners, weights)
                    fit = width_pair(reader, trace[CROP])
                    if key in ENDPOINTS:
                        check_frozen(fit, fits[key])
                    else:
                        fits[key] = fit
                arms[camera_name][frame] = fits
        walls.append({**base, "arms": arms})
    return {"landscapes": landscapes, "solvers": solvers, "walls": walls}


def read_prefix(path, hashes, names):
    if path is None:
        return None, None
    previous = json.loads(path.read_bytes())
    grid = previous["settings_meV_per_cell"]
    if len(grid) >= len(BARRIERS) or list(BARRIERS[:len(grid)]) != grid:
        raise ValueError("The previous settings must be a strict prefix of the full sweep")
    if set(previous["scans"]) != set(names):
        raise ValueError("The previous sweep must include every source map")
    for name, checksum in hashes.items():
        if name != Path(__file__).name and previous["input_sha256"].get(name) != checksum:
            raise ValueError(f"The extension loads a different input for {name}")
    receipt = {"sha256": digest(path), "initial_settings_meV_per_cell": grid,
               "initial_runner_sha256": previous["input_sha256"][Path(__file__).name],
               "initial_seconds": previous["seconds"]}
    return previous, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "release", "dependencies", "evidence", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extend-from", type=Path, help="Reuse a completed sweep containing the initial settings")
    args = parser.parse_args()
    start = perf_counter()
    files = (SOURCE, BENCHMARK, SETTINGS, *(args.evidence / name for name in INPUT_NAMES),
             args.evidence / "mos2_saddle_control_v01.json")
    source, benchmark, settings, primary, cameras, optics, frozen = (json.loads(f.read_text()) for f in files)
    modules = tuple(PROJECT / (name + ".py") for name in REPLAY_MODULES)
    hashes = {f.name: digest(f) for f in (Path(__file__), *files, *modules)}
    hashes["release.zip"] = digest(args.release.parents[1] / "release.zip")
    for name in set(hashes) & set(frozen["input_sha256"]):
        if hashes[name] != frozen["input_sha256"][name]:
            raise ValueError(f"{name} changed relative to the frozen endpoint calculation")
    names = primary["protocol"]["scans"]
    if set(names) != set(frozen["scans"]):
        raise ValueError("The frozen endpoint calculation and sweep must use the same maps")
    prefix, extension = read_prefix(args.extend_from, hashes, names)
    report = {
        "schema": "mos2-fixed-probe-saddle-sweep-v01",
        "requested_scans": names,
        "settings_meV_per_cell": list(BARRIERS),
        "probe_fwhm_nm": NOMINAL,
        "half_window_nm": HALF_WINDOW,
        "endpoint_fits_reused_meV_per_cell": list(ENDPOINTS),
        "endpoint_replay_checked_in": ENDPOINT_CHECK_SCAN,
        "input_sha256": hashes,
        "extension": extension,
        "scans": {},
    }
    elapsed = prefix["seconds"] if prefix is not None else 0.
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        report = json.loads(args.output.read_text())
        if report["input_sha256"] != hashes or report["requested_scans"] != names or report["extension"] != extension:
            raise ValueError("The checkpoint and run inputs differ")
        elapsed = report["seconds"]
    author = load_author_release(args.release, args.dependencies)
    for scan in names:
        if scan in report["scans"]:
            continue
        checkpoint_guard()
        meta = read_input_metadata(scan, args.archive)
        images, modes, checksum = disk_images(args.archive, scan, author)
        ref = source["scans"][scan]
        if checksum != ref["diskset_sha256"] or meta["archive_md5"] != benchmark["scans"][scan]["archive_md5"]:
            raise ValueError(f"The archived intensities changed for {scan}")
        fields = meta["metadata_fields"]
        context = centers_and_context(images, modes, fields, settings, tuple(ref["interface"]))
        if context["receipt"] != ref["geometry"]:
            raise ValueError(f"The inferred geometry changed for {scan}")
        report["scans"][scan] = evaluate(scan, context, images, modes, fields, primary["scans"][scan],
                                         cameras["scans"][scan], optics, frozen["scans"][scan],
                                         None if prefix is None else prefix["scans"][scan])
        report["seconds"] = elapsed + perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, report)
        print(f"{scan}: {len(report['scans'][scan]['walls'])} sections", flush=True)


if __name__ == "__main__":
    main()
