"""Test an ideal circular pupil in the existing intensity-convolution model."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.special import j0, roots_legendre

from author_detector_replay import detector_intensity
from author_release import load_author_release
from bilayer_relaxation import make_grid, wavevectors
from observe_bilayers import read_input_metadata
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_coherent_window import APERTURE, CONVERGENCE, WAVELENGTH
from run_mos2_intensity_only_replay import disk_images
from run_mos2_nonlinear_controls import minimize_all
from run_mos2_raman_probe_sensitivity import CAMERA, CONTROL, MODELS, PRIMARY
from run_mos2_wall_landscape import CROP, get_observer, width_pair
from run_mos2_wall_profile_closure import PROFILE_DISTANCE, fields_on_chord, render_profiles, transfer
from run_mos2_wall_width_transfer import DISTANCE, REDUCED, actual_profiles, geometry_sections
from run_registry_prediction import rotation
from run_wall_heterostrain import SOURCE
from shared_compute import atomic_json, checkpoint_guard


FOCUS = {"circular_focused": 0., "circular_defocus_600nm": 600.}
PROBE = 1.25
METHOD = (
    "At published 80 kV and angle, replace the Gaussian intensity-convolution kernel by the "
    "intensity point-spread function of an ideal circular pupil at focus or 600 nm illustrative "
    "defocus. Use the same frozen intensity coefficients and isolated-wall width observer to "
    "compare predictions and measurements. The aperture is specified by the paper. The transfer "
    "model makes no added claim about the phase or virtual detector windows on the individual scans."
)


def circular_transfer(frequency, defocus_nm):
    """Normalized autocorrelation of the circular pupil with a defocus phase."""
    s = np.minimum(np.asarray(frequency) / (2 * APERTURE), 1.0)
    if defocus_nm == 0:
        return (2 / np.pi) * (np.arccos(s) - s * np.sqrt(1 - s**2))
    nodes, weights = roots_legendre(24)
    bound = np.arccos(s)
    t = np.outer(bound, .5 * (nodes + 1))
    x = np.cos(t) - s[:, None]
    arg = (2 * np.pi * WAVELENGTH * defocus_nm * APERTURE) * np.asarray(frequency)[:, None] * x
    return (2 / np.pi) * bound * ((np.sin(t)**2 * np.cos(arg)) @ weights)


def pupil_receipt():
    radius = np.linspace(0., 2.5, 501)
    nodes, weights = roots_legendre(80)
    rho = .5 * (nodes + 1)
    kernel = j0(2 * np.pi * APERTURE * radius[:, None] * rho)
    out = {}
    for name, defocus in FOCUS.items():
        amplitude = kernel @ (weights * rho * np.exp(-1j*np.pi*WAVELENGTH*defocus*(APERTURE*rho)**2))
        intensity = abs(amplitude)**2
        cross = np.flatnonzero(intensity < .5 * intensity[0])[0]
        width = 2 * np.interp(.5 * intensity[0], intensity[cross-1:cross+1][::-1],
                              radius[cross-1:cross+1][::-1])
        frequency = np.array([0., .10, .20, .30, 2*APERTURE, 3*APERTURE])
        gain = circular_transfer(frequency, defocus)
        np.testing.assert_allclose(gain[[0, -2, -1]], [1, 0, 0], atol=1e-13)
        out[name] = {"defocus_nm": defocus, "central_fwhm_nm": width, "on_axis_intensity": intensity[0],
                     "cycles_per_nm": frequency.tolist(), "intensity_transfer": gain.tolist()}
    return out


def local_observation(section, basis, pixel, n):
    """Use the same local affine registry map as the Gaussian profile calculation."""
    xy = section["coordinates"][..., ::-1].reshape(-1, 2) * pixel
    rhs = section["phase"].reshape(-1, 2)
    design = np.column_stack((xy, np.ones(len(xy))))
    transform = np.linalg.lstsq(design, rhs, rcond=None)[0]
    if np.max(abs(design @ transform - rhs)) > 1e-10:
        raise ValueError("The measured section is not affine")
    cell = np.linalg.solve(transform[:2].T, basis)
    frequency = np.linalg.norm(wavevectors(cell, n), axis=-1) / (2 * np.pi)
    center = len(DISTANCE) // 2
    phase = section["phase"][:, center, None, :] + PROFILE_DISTANCE[None, :, None] * (
        section["phase"][0, center + 1] - section["phase"][0, center]) / (DISTANCE[1]-DISTANCE[0])
    coordinate = n * np.moveaxis(phase @ np.linalg.inv(basis).T, -1, 0)
    return frequency, coordinate


def sample_image(spectrum, gain, coordinate):
    image = np.fft.ifft2(spectrum * gain[..., None], axes=(0, 1)).real
    return np.column_stack([
        map_coordinates(image[..., i], coordinate, order=3, mode="grid-wrap").mean(axis=0)
        for i in range(image.shape[-1])
    ])


def profile_scores(observer, target, trace):
    fit = width_pair(observer, trace[CROP])
    results = transfer(observer, target, trace)
    fit["transfer_error"] = [row["error_by_ring"][1 - i] for i, row in enumerate(results)]
    return fit


def examine(context, images, modes, meta, original, control, compact, even_terms, frequencies, response):
    basis, gradient, q, rings = (context[key] for key in ("basis", "gradient", "reciprocal", "rings"))
    angle = np.deg2rad(float(meta["SampleRotation"]))
    sections = geometry_sections(context, rotation(angle))
    if [s["centers"] for s in sections] != [w["centers"] for w in original["walls"]]:
        raise ValueError("The wall geometry differs from the main comparison")
    physics = original["physics"]
    stiffness = physics["stiffness_eV_per_cell"]
    anchor = np.asarray(physics["stacking_coefficients_meV_per_cell"]["anchor"])
    grid = make_grid(gradient, basis, physics["grid"], *stiffness)
    motion, spectra = {}, {}
    nominal = np.broadcast_to([even_terms[0], 0., even_terms[1]], (len(q), 3))
    own = {name: np.asarray(control["camera_diagnostic"]["by_model"][name]["coefficients"]) for name in MODELS}
    cameras = {"transferred_even": {name: nominal for name in MODELS}, "separate_disjoint": own}
    np.testing.assert_allclose(q @ basis, np.rint(q @ basis), atol=1e-12, rtol=0.)
    for name in MODELS:
        coeff = np.asarray(control["raman_constraint"]["models"][name]["coefficients_meV_per_cell"])
        motion[name], _ = minimize_all(grid, coeff)
        registry = grid.fractional @ basis.T + motion[name]
        for cam, calibration in cameras.items():
            spectra[cam, name] = np.fft.fft2(detector_intensity(registry, q, calibration[name]), axes=(0, 1))

    mode_keys = {tuple(row): i for i, row in enumerate(modes)}
    records = []
    roundtrip_errors = []
    for section, ref in zip(sections, original["walls"], strict=True):
        if section["reach_nm"] < 4.25 or not ref["original_cohort"]:
            continue
        checkpoint_guard()
        subsection = {**section, "keep": abs(DISTANCE) <= 4.25}
        frequency, coordinates = local_observation(section, basis, context["receipt"]["pixel_nm"], physics["grid"])
        transfers = {name: np.interp(frequency, frequencies, values, right=0.) for name, values in response.items()}
        channels = [np.asarray([mode_keys[tuple(channel)] for channel in group], dtype=int)
                    for group in ref["miller_indices"]]
        obs = get_observer(subsection, q, nominal, rings, PROBE, channels)
        measured = actual_profiles(images, section["coordinates"])
        observed_shape = obs.shape(measured)
        measured_pair = width_pair(obs, measured)
        score = {name: {} for name in transfers}
        for cam in cameras:
            for mode, kernel in transfers.items():
                score[mode][cam] = {"measured": measured_pair, "models": {}}
                for name in MODELS:
                    trace = sample_image(spectra[cam, name], kernel, coordinates)
                    score[mode][cam]["models"][name] = profile_scores(obs, observed_shape, trace)
        # Check the fast periodic convolution on every chord against the original
        # Gaussian point and normal-quadrature implementation.
        stddev = PROBE / np.sqrt(8*np.log(2))
        old_fields, weights = fields_on_chord(section, context, angle, grid, motion["raman_shape"],
                                              grid, motion["raman_scale"], compact,
                                              {"anchor": (anchor, stiffness)}, PROBE)
        reference = render_profiles({"shape": old_fields["periodic_anchor"]}, q, nominal, weights, PROBE)["shape"]
        fast = sample_image(spectra["transferred_even", "raman_shape"],
                            np.exp(-2 * (np.pi * stddev * frequency)**2), coordinates)
        error = np.linalg.norm(np.concatenate([(a - b).ravel() for a, b in
                zip(obs.shape(reference[CROP]), obs.shape(fast[CROP]), strict=True)]))
        roundtrip_errors.append(float(error))
        records.append({"centers": ref["centers"], "optical_scores": score})
    return {"walls": records, "fft_gaussian_shape_check": {
        "maximum_norm_difference": max(roundtrip_errors),
        "mean_norm_difference": float(np.mean(roundtrip_errors))
    }}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("release", type=Path)
    cli.add_argument("dependencies", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--resume", action="store_true")
    args = cli.parse_args()
    start = perf_counter()
    inputs = SOURCE, BENCHMARK, SETTINGS, REDUCED, CAMERA, PRIMARY, CONTROL
    source, benchmark, settings, reduced, optics, primary, control = (json.loads(p.read_text()) for p in inputs)
    extra = ("run_mos2_raman_probe_sensitivity.py", "run_mos2_coherent_window.py",
             "run_mos2_wall_profile_closure.py", "run_mos2_wall_landscape.py", "bilayer_relaxation.py")
    hashes = {p.name: digest(p) for p in (Path(__file__), *inputs, *(PROJECT / name for name in extra))}
    receipt = pupil_receipt()
    frequency = np.linspace(0, 2 * APERTURE, 4097)
    response = {key: circular_transfer(frequency, defocus) for key, defocus in FOCUS.items()}
    report = {"method": METHOD, "energy_kV": 80, "semiangle_mrad": CONVERGENCE * 1000,
              "wavelength_nm": WAVELENGTH, "ideal_pupils": receipt, "input_sha256": hashes, "scans": {}}
    elapsed = 0.
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        report = json.loads(args.output.read_text())
        if report["input_sha256"] != hashes:
            raise ValueError("The pupil check source freeze changed")
        elapsed = report["seconds"]
    author = load_author_release(args.release, args.dependencies)
    for scan in primary["protocol"]["scans"]:
        if scan in report["scans"]:
            continue
        checkpoint_guard()
        meta = read_input_metadata(scan, args.archive)
        images, modes, checksum = disk_images(args.archive, scan, author)
        row = source["scans"][scan]
        if checksum != row["diskset_sha256"] or meta["archive_md5"] != benchmark["scans"][scan]["archive_md5"]:
            raise ValueError("The archived diffraction changed")
        context = centers_and_context(images, modes, meta["metadata_fields"], settings, tuple(row["interface"]))
        if context["receipt"] != row["geometry"]:
            raise ValueError("The measured geometry changed")
        compact = np.asarray(reduced["scans"][scan]["coefficients_nm"]["three_stars"])
        result = examine(context, images, modes, meta["metadata_fields"], primary["scans"][scan],
                         control["scans"][scan], compact, optics["frozen_common_coefficients"], frequency, response)
        report["scans"][scan] = result
        report["seconds"] = elapsed + perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, report)
        print(scan, result["fft_gaussian_shape_check"], round(report["seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
