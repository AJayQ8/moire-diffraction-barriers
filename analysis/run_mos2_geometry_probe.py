"""Test low-twist geometry and compare energies on native MoS2 scan pixels."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import polar

from author_detector_replay import detector_intensity
from author_release import load_author_release
from bilayer_relaxation import make_grid
from observe_bilayers import read_input_metadata
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_circular_probe_check import circular_transfer
from run_mos2_frame_sensitivity import frame_options
from run_mos2_intensity_only_replay import disk_images
from run_mos2_nonlinear_controls import minimize_all
from run_mos2_observation_controls import (
    HALF_WINDOW, INPUT_NAMES, NOMINAL, baseline_qualified, gaussian_gain,
    raster_coordinates, render_traces, sample_cuts, score,
)
from run_mos2_uniaxial_check import decompose
from run_mos2_wall_landscape import CROP, get_observer, width_pair
from run_mos2_wall_width_transfer import DISTANCE, geometry_sections, isolated_width
from run_registry_prediction import rotation
from run_wall_heterostrain import SOURCE
from shared_compute import atomic_json, checkpoint_guard


LOW = ("DS1", "DS2", "DS13", "DS14")
GAUSSIANS = (1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.9, 2.05, 2.2)
DEFOCUS_NM = (0, 300, 600, 900, 1200)
FIXED = ("anchor", "raman_shape", "raman_scale")
FRACTIONS = (0.25, 0.5, 0.75)


def geometry_controls(gradient, basis, n, stiffness, adhesion):
    """Vary the mechanics while preserving every observed registry and cut."""
    motions, checks = {}, {}
    integer_grid = np.indices((n, n)).transpose(1, 2, 0)
    for turns, rot, branch, diagnostics in frame_options(gradient):
        if not diagnostics["passes_original_twist_dilation_bounds"]:
            continue
        diagnostics_strain, controls = decompose(branch)
        orientation, _ = polar(np.eye(2) + branch)
        pure_twist = orientation - np.eye(2)
        lattice_map = np.linalg.solve(basis, rot @ basis)
        permutation = np.rint(lattice_map).astype(int)
        np.testing.assert_allclose(lattice_map, permutation, rtol=0, atol=1e-12)
        if round(abs(np.linalg.det(permutation))) != 1:
            raise ValueError("A lattice rotation must permute the periodic grid")
        i, j = np.moveaxis((integer_grid @ permutation.T) % n, -1, 0)
        for name, deformation in (("full", branch), ("no_mean", controls["no_mean_dilation"]),
                                  ("uniaxial", controls["uniaxial"]), ("pure_twist", pure_twist)):
            if turns == 0 and name == "full":
                continue
            tag = f"{turns}_{name}"
            grid = make_grid(deformation, basis, n, *stiffness)
            movement, checks[tag] = minimize_all(grid, adhesion)
            # Move the periodic displacement back to the original registry axes.
            # The unrelaxed registry, reciprocal vectors and raster stay fixed.
            motions[tag] = movement[i, j] @ rot
        checks[f"branch_{turns}"] = {**diagnostics, **diagnostics_strain}
    return motions, checks


def make_spectra(context, original, raman, disjoint, detector, geometry, geometry_only):
    basis, gradient, q = (context[k] for k in ("basis", "gradient", "reciprocal"))
    physics = original["physics"]
    n, stiffness = physics["grid"], physics["stiffness_eV_per_cell"]
    grid = make_grid(gradient, basis, n, *stiffness)
    rigid = grid.fractional @ basis.T
    adhesion = {"anchor": np.asarray(physics["stacking_coefficients_meV_per_cell"]["anchor"])}
    if not geometry_only:
        adhesion.update({name: np.asarray(raman[name]["coefficients_meV_per_cell"]) for name in FIXED[1:]})
        for fraction in FRACTIONS:
            adhesion[f"raman_{fraction:g}"] = (
                (1 - fraction) * adhesion["raman_shape"] + fraction * adhesion["raman_scale"]
            )
    motions, solvers = {}, {}
    for name, terms in adhesion.items():
        motions[name], solvers[name] = minimize_all(grid, terms)
    if geometry:
        controls, checks = geometry_controls(gradient, basis, n, stiffness, adhesion["anchor"])
        motions.update(controls)
        solvers.update(checks)
    motions["rigid"] = np.zeros_like(motions["anchor"])
    transferable = {name: detector for name in motions}
    alternate = {name: np.asarray(disjoint[name]["coefficients"])
                 for name in (*FIXED, "rigid") if name in motions}
    if geometry:
        alternate.update({name: alternate["anchor"] for name in controls})
    cameras = {"transferred_even": transferable, "separate_disjoint": alternate}
    spectra = {
        camera: {name: np.fft.fft2(detector_intensity(rigid + motions[name], q, coeff), axes=(0, 1))
                 for name, coeff in calibration.items()}
        for camera, calibration in cameras.items()
    }
    compact = np.asarray(original["cameras"]["compact"]["channels"])
    check = np.fft.fft2(detector_intensity(rigid + motions["anchor"], q, compact), axes=(0, 1))
    barriers = {name: 1000 * isolated_width(basis, basis[:, 0], terms)[1] for name, terms in adhesion.items()}
    if not geometry_only:
        for fraction in FRACTIONS:
            expected = (1 - fraction) * barriers["raman_shape"] + fraction * barriers["raman_scale"]
            np.testing.assert_allclose(barriers[f"raman_{fraction:g}"], expected, atol=1e-8, rtol=0)
    return spectra, check, solvers, barriers


def read_wall(section, ref, previous, images, context, detector, channels, spectra, check, compact, n, pupil):
    basis, q, rings = (context[key] for key in ("basis", "reciprocal", "rings"))
    observed = sample_cuts(images, section["coordinates"]).mean(axis=0)
    frequency, cuts, corners, weights = raster_coordinates(section, basis, context["receipt"]["pixel_nm"], n)
    view = {**section, "keep": abs(DISTANCE) <= HALF_WINDOW}
    nominal_gain = gaussian_gain(frequency, NOMINAL)
    if previous["in_fig2_width_cohort"]:
        observer = get_observer(view, q, compact, rings, NOMINAL, channels)
        direct, native = render_traces(check, nominal_gain, cuts, corners, weights)
        for key, trace in (("continuous", direct), ("native_raster", native)):
            np.testing.assert_allclose(width_pair(observer, trace[CROP])["widths_nm"],
                                       previous["anchor_raster"][key]["widths_nm"], atol=1e-5, rtol=0)
    chord = section["coordinates"][2, len(DISTANCE)//2, ::-1] - section["coordinates"][0, len(DISTANCE)//2, ::-1]
    result = {
        "centers": section["centers"], "midpoint_xy_nm": ref["midpoint_xy_nm"],
        "spacing_nm": section["spacing_nm"], "character": section["k_factor"],
        "scan_angle_deg": float(np.rad2deg(np.arctan2(chord[1], chord[0])) % 180),
        "legacy_width_cohort": previous["in_fig2_width_cohort"], "nominal": {}, "gaussian": {}, "pupil": {},
    }
    observer = get_observer(view, q, detector, rings, NOMINAL, channels)
    measured = width_pair(observer, observed)
    target = observer.shape(observed)
    for camera, models in spectra.items():
        result["nominal"][camera] = {"measured": measured, "models": {}}
        for name, spectrum in models.items():
            if name.startswith("raman_") and name not in FIXED:
                continue
            direct, native = render_traces(spectrum, nominal_gain, cuts, corners, weights)
            result["nominal"][camera]["models"][name] = {
                "native": score(observer, target, native),
            }
            if name in ("anchor", "rigid"):
                result["nominal"][camera]["models"][name]["continuous"] = score(observer, target, direct)
    if "raman_shape" not in spectra["transferred_even"]:
        return result
    for probe in GAUSSIANS:
        reader = get_observer(view, q, detector, rings, probe, channels)
        gain = gaussian_gain(frequency, probe)
        at_probe = {"measured": width_pair(reader, observed), "models": {}}
        for camera, models in spectra.items():
            fits = {}
            for name, spectrum in models.items():
                if name not in FIXED and not name.startswith("raman_"):
                    continue
                if probe == NOMINAL and name in FIXED:
                    fit = result["nominal"][camera]["models"][name]["native"]
                else:
                    _, trace = render_traces(spectrum, gain, cuts, corners, weights)
                    fit = width_pair(reader, trace[CROP])
                fits[name] = fit
            at_probe["models"][camera] = fits
        result["gaussian"][f"{probe:g}"] = at_probe
    freq = np.sqrt(frequency) / (2 * np.pi)
    table_freq, responses = pupil
    for defocus, response in responses.items():
        gain = np.interp(freq, table_freq, response, right=0.)
        fits = {}
        for name in FIXED:
            _, trace = render_traces(spectra["transferred_even"][name], gain, cuts, corners, weights)
            fits[name] = width_pair(observer, trace[CROP])
        result["pupil"][str(defocus)] = fits
    return result


def examine(name, context, images, modes, meta, original, control, optical, prior, geometry_only):
    basis, gradient, q, rings = (context[k] for k in ("basis", "gradient", "reciprocal", "rings"))
    angle = np.deg2rad(float(meta["SampleRotation"]))
    sections = geometry_sections(context, rotation(angle))
    if [s["centers"] for s in sections] != [row["centers"] for row in original["walls"]]:
        raise ValueError("Wall identities changed relative to the frozen study")
    np.testing.assert_allclose(q @ basis, np.rint(q @ basis), atol=1e-12, rtol=0)
    even = optical["frozen_common_coefficients"]
    detector = np.broadcast_to([even[0], 0., even[1]], (len(q), 3))
    spectra, check, solvers, barriers = make_spectra(
        context, original, control["raman_constraint"]["models"], control["camera_diagnostic"]["by_model"],
        detector, name in LOW, geometry_only,
    )
    modes_by_index = {tuple(mode): i for i, mode in enumerate(modes)}
    previous = {tuple(row["centers"]): row for row in prior["walls"]}
    compact = np.asarray(original["cameras"]["compact"]["channels"])
    table_freq = np.linspace(0., 2., 2501)
    pupil = (table_freq, {} if geometry_only else {z: circular_transfer(table_freq, z) for z in DEFOCUS_NM})
    walls = []
    for section, archived in zip(sections, original["walls"], strict=True):
        if not archived["original_cohort"] or section["reach_nm"] < HALF_WINDOW:
            continue
        checkpoint_guard()
        channels = [np.asarray([modes_by_index[tuple(m)] for m in group]) for group in archived["miller_indices"]]
        old = previous[tuple(section["centers"])]
        if baseline_qualified(archived) != old["in_fig2_width_cohort"]:
            raise ValueError("The archived nominal width qualification changed")
        walls.append(read_wall(section, archived, old, images, context, detector, channels, spectra,
                               check, compact, original["physics"]["grid"], pupil))
    if len(walls) != len(previous):
        raise ValueError("The original wall-section set is incomplete")
    _, stretch = polar(np.eye(2) + gradient)
    return {
        "twist_deg": original["inferred_twist_deg"], "pixel_nm": original["pixel_nm"],
        "polar_principal_strain_percent": (100 * (np.linalg.eigvalsh(stretch) - 1)).tolist(),
        "polar_mean_strain_percent": float(100 * (np.trace(stretch) / 2 - 1)),
        "barriers_meV_per_cell": barriers, "solvers": solvers, "walls": walls,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "release", "dependencies", "evidence", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--scans", nargs="+")
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    start = perf_counter()
    inputs = (SOURCE, BENCHMARK, SETTINGS, *(args.evidence / name for name in INPUT_NAMES),
              args.evidence / "mos2_observation_controls_v004.json")
    source, benchmark, settings, primary, control, optical, prior = (
        json.loads(path.read_text()) for path in inputs
    )
    libraries = ("run_mos2_observation_controls.py", "bilayer_relaxation.py", "run_mos2_wall_landscape.py",
                 "run_mos2_wall_width_transfer.py", "run_mos2_frame_sensitivity.py",
                 "run_mos2_circular_probe_check.py", "run_mos2_nonlinear_controls.py",
                 "run_mos2_uniaxial_check.py")
    hashes = {path.name: digest(path) for path in (Path(__file__), *inputs, *(PROJECT / name for name in libraries))}
    if control["input_sha256"][INPUT_NAMES[0]] != hashes[INPUT_NAMES[0]]:
        raise ValueError("The material and detector comparison use different source geometry")
    names = args.scans or (list(LOW) if args.geometry_only else primary["protocol"]["scans"])
    if len(set(names)) != len(names) or set(names) - set(primary["protocol"]["scans"]):
        raise ValueError("Expected distinct scans from the frozen wall set")
    report = {
        "input_sha256": hashes, "scans": {}, "requested_scans": names, "geometry_only": args.geometry_only,
        "gaussian_fwhm_nm": GAUSSIANS, "ideal_pupil_defocus_nm": DEFOCUS_NM,
        "raman_interpolation_fractions": FRACTIONS,
    }
    elapsed = 0.
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        report = json.loads(args.output.read_text())
        if report["input_sha256"] != hashes or report["geometry_only"] != args.geometry_only:
            raise ValueError("The checkpoint came from different inputs")
        if set(report["scans"]) - set(names):
            raise ValueError("The checkpoint contains maps outside the requested set")
        report["requested_scans"] = names
        elapsed = report["seconds"]
    release = load_author_release(args.release, args.dependencies)
    for name in names:
        if name in report["scans"]:
            continue
        checkpoint_guard()
        metadata = read_input_metadata(name, args.archive)
        images, modes, checksum = disk_images(args.archive, name, release)
        ref = source["scans"][name]
        if checksum != ref["diskset_sha256"] or metadata["archive_md5"] != benchmark["scans"][name]["archive_md5"]:
            raise ValueError(f"The diffraction input changed for {name}")
        fields = metadata["metadata_fields"]
        context = centers_and_context(images, modes, fields, settings, tuple(ref["interface"]))
        if context["receipt"] != ref["geometry"]:
            raise ValueError(f"The inferred geometry changed for {name}")
        report["scans"][name] = examine(name, context, images, modes, fields, primary["scans"][name],
                                        control["scans"][name], optical, prior["scans"][name], args.geometry_only)
        report["seconds"] = elapsed + perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, report)
        print(name, len(report["scans"][name]["walls"]), round(report["seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
