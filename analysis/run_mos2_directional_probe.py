"""Test fixed elliptical probe responses on the same MoS2 wall sections."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from author_detector_replay import detector_intensity
from author_release import load_author_release
from bilayer_relaxation import make_grid
from observe_bilayers import read_input_metadata
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_intensity_only_replay import disk_images
from run_mos2_local_wall_mechanics import SCANS, local_gradient
from run_mos2_nonlinear_controls import minimize_all
from run_mos2_observation_controls import (
    HALF_WINDOW, INPUT_NAMES, NOMINAL, gaussian_gain, raster_coordinates, render_traces, sample_cuts,
)
from run_mos2_opposite_widths import DISJOINT, EVEN, check_frozen, paired_spots
from run_mos2_wall_landscape import CROP, get_observer, width_pair
from run_mos2_wall_profile_closure import transfer
from run_mos2_wall_width_transfer import DISTANCE, geometry_sections
from run_registry_prediction import rotation
from run_wall_heterostrain import SOURCE
from shared_compute import atomic_json, checkpoint_guard


MAJOR_AXES_NM = (1.75, 2.20)
RASTER_ANGLES_DEG = tuple(range(0, 180, 30))


def probe_settings():
    settings = {"nominal": {"kind": "circular", "fwhm_nm": NOMINAL}}
    for major in MAJOR_AXES_NM:
        display = f"{major:.2f}"
        equal_trace = f"circle_{display}"
        settings[equal_trace] = {"kind": "circular", "fwhm_nm": float(np.hypot(major, NOMINAL)/np.sqrt(2))}
        for angle in RASTER_ANGLES_DEG:
            settings[f"ellipse_{display}_{angle:03d}"] = {
                "kind": "elliptical", "major_fwhm_nm": major, "minor_fwhm_nm": NOMINAL,
                "major_angle_to_raster_x_deg": angle, "equal_trace_circle": equal_trace,
            }
    return settings


def probe_gains(k_raster, freq2, settings):
    np.testing.assert_allclose(np.sum(k_raster**2, axis=-1), freq2, atol=1e-9, rtol=1e-11)
    result = {}
    for name, probe in settings.items():
        if probe["kind"] == "circular":
            result[name] = gaussian_gain(freq2, probe["fwhm_nm"])
            continue
        angle = np.deg2rad(probe["major_angle_to_raster_x_deg"])
        major_k2 = (k_raster @ np.array([np.cos(angle), np.sin(angle)]))**2
        variance_increment = probe["major_fwhm_nm"]**2 - probe["minor_fwhm_nm"]**2
        exponent = probe["minor_fwhm_nm"]**2 * freq2 + variance_increment * major_k2
        result[name] = np.exp(-exponent/(16*np.log(2)))
    # Two orthogonal ellipses have the same product as two equal-trace circles.
    for major in MAJOR_AXES_NM:
        display = f"{major:.2f}"
        np.testing.assert_allclose(result[f"ellipse_{display}_000"] * result[f"ellipse_{display}_090"],
                                   result[f"circle_{display}"]**2, atol=1e-13, rtol=1e-11)
    return result


def evaluate(context, images, modes, angle, original, cameras, observed_case, local_case, settings):
    pixel = original["pixel_nm"]
    basis, q, rings = (context[k] for k in ("basis", "reciprocal", "rings"))
    sections = geometry_sections(context, rotation(angle))
    original_walls = {tuple(row["centers"]): row for row in original["walls"]}
    sampled = {tuple(section["centers"]): section for section in sections}
    by_mode = {tuple(mode): i for i, mode in enumerate(modes)}
    physics = original["physics"]
    adhesion = physics["stacking_coefficients_meV_per_cell"]["anchor"]
    if [row["centers"] for row in observed_case] != [row["centers"] for row in local_case]:
        raise ValueError("The archived geometry and spot-level scores use different walls")
    walls = []
    for previous, local in zip(observed_case, local_case, strict=True):
        checkpoint_guard()
        index = tuple(local["centers"])
        section = sampled[index]
        ref = original_walls[index]
        selected = [np.asarray([by_mode[tuple(m)] for m in group]) for group in ref["miller_indices"]]
        halves = paired_spots(modes, rings, selected)
        if [[modes[order].tolist() for order in side] for side in halves] != previous["half_modes"]:
            raise ValueError("The archived diffraction spots changed")
        gradient = local_gradient(section, pixel, angle)
        np.testing.assert_allclose(gradient, local["local_gradient"], atol=1e-12, rtol=0)
        n = local["grid"]
        grid = make_grid(gradient, basis, n, *physics["stiffness_eV_per_cell"])
        motion, _ = minimize_all(grid, adhesion)
        registry = grid.fractional @ basis.T + motion
        freq2, cut, corners, weights = raster_coordinates(section, basis, pixel, n)
        gains = probe_gains(grid.k @ rotation(angle), freq2, settings)
        view = {**section, "keep": abs(DISTANCE) <= HALF_WINDOW}
        reader = get_observer(view, q, cameras[EVEN], rings, NOMINAL, selected)
        paired = [get_observer(view, q, cameras[EVEN], rings, NOMINAL, group) for group in halves]
        observation = sample_cuts(images, section["coordinates"]).mean(axis=0)
        measured = previous["windows"][f"{HALF_WINDOW:g}"]["measured"]
        for half, frozen_fit in zip(paired, measured, strict=True):
            check_frozen(width_pair(half, observation), frozen_fit)
        target = reader.shape(observation)
        models = {}
        for camera, coeff in cameras.items():
            spectrum = np.fft.fft2(detector_intensity(registry, q, coeff), axes=(0, 1))
            reports = {}
            for name, gain in gains.items():
                _, trace = render_traces(spectrum, gain, cut, corners, weights)
                width = [width_pair(half, trace[CROP]) for half in paired]
                if name == "nominal":
                    for current, frozen in zip(width, local["local"][camera]["widths"], strict=True):
                        check_frozen(current, frozen)
                reports[name] = {"widths": width, "first_order_registration": transfer(reader, target, trace)[0]}
            models[camera] = reports
        record = {k: previous[k] for k in ("centers", "midpoint_xy_nm", "spacing_nm", "scan_angle_deg")}
        walls.append({**record, "measured": measured, "models": models})
    if len(walls) != len(local_case):
        raise ValueError("The direction test missed an archived wall")
    return walls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "release", "dependencies", "evidence", "opposite_result", "local_result", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    paths = (SOURCE, BENCHMARK, SETTINGS, *(args.evidence / name for name in INPUT_NAMES),
             args.opposite_result, args.local_result)
    source, bench, settings, primary, control, optical, opposite, local = [json.loads(p.read_bytes()) for p in paths]
    libraries = ("run_mos2_opposite_widths.py", "run_mos2_local_wall_mechanics.py",
                 "run_mos2_observation_controls.py", "run_mos2_wall_landscape.py", "run_mos2_wall_width_transfer.py",
                 "run_mos2_wall_profile_closure.py", "bilayer_relaxation.py", "author_detector_replay.py",
                 "run_mos2_nonlinear_controls.py")
    hashes = {p.name: digest(p) for p in (Path(__file__), *paths, *(PROJECT / name for name in libraries))}
    probes = probe_settings()
    output = {"scope": "Keep the measured sections, detector fits, local mechanical equilibria and width reader "
                       "fixed. Change the illumination kernel in the forward simulation. The ellipse angles are "
                       "measured from raster x. Fits use the first diffraction order for profile position and "
                       "leave the second order for scoring. The three scans have been examined previously.",
              "input_sha256": hashes, "requested_scans": list(SCANS), "probes": probes, "scans": {}}
    elapsed = 0.
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        output = json.loads(args.output.read_bytes())
        if (output["input_sha256"] != hashes or output["requested_scans"] != list(SCANS)
                or output["probes"] != probes):
            raise ValueError("Resume requires identical archived inputs and probe settings")
        elapsed = output["seconds"]
    author = load_author_release(args.release, args.dependencies)
    even = optical["frozen_common_coefficients"]
    for name in SCANS:
        if name in output["scans"]:
            continue
        checkpoint_guard()
        metadata = read_input_metadata(name, args.archive)
        images, modes, checksum = disk_images(args.archive, name, author)
        ref = source["scans"][name]
        if checksum != ref["diskset_sha256"] or metadata["archive_md5"] != bench["scans"][name]["archive_md5"]:
            raise ValueError(f"The archived scan changed for {name}")
        fields = metadata["metadata_fields"]
        context = centers_and_context(images, modes, fields, settings, tuple(ref["interface"]))
        if context["receipt"] != ref["geometry"]:
            raise ValueError(f"The measured geometry changed for {name}")
        angle = np.deg2rad(float(fields["SampleRotation"]))
        disjoint = control["scans"][name]["camera_diagnostic"]["by_model"]["anchor"]["coefficients"]
        cameras = {EVEN: np.broadcast_to([even[0], 0., even[1]], (len(modes), 3)), DISJOINT: np.asarray(disjoint)}
        output["scans"][name] = evaluate(
            context, images, modes, angle, primary["scans"][name], cameras,
            opposite["scans"][name]["walls"], local["scans"][name], probes,
        )
        output["seconds"] = elapsed + perf_counter() - started
        checkpoint_guard()
        atomic_json(args.output, output)
        print(name, len(output["scans"][name]), round(output["seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
