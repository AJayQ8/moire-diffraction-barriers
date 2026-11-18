"""Compare isolated and periodic walls with the same experimental scan sampling."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter1d

from author_detector_replay import detector_intensity
from author_release import load_author_release
from bilayer_relaxation import make_grid
from observe_bilayers import read_input_metadata
from run_heterobilayer_transfer import digest
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_intensity_only_replay import disk_images
from run_mos2_nonlinear_controls import minimize_all
from run_mos2_observation_controls import (
    CROP, HALF_WINDOW, NOMINAL, gaussian_gain, raster_coordinates, render_traces, sample_cuts, score,
)
from run_mos2_opposite_widths import check_frozen
from run_mos2_wall_landscape import get_observer, width_pair
from run_mos2_wall_profile_closure import PROFILE_DISTANCE, isolated_field, render_profiles
from run_mos2_wall_width_transfer import DISTANCE, Profiles, geometry_sections
from run_registry_prediction import rotation
from run_wall_heterostrain import SOURCE
from shared_compute import atomic_json, checkpoint_guard


INPUT_NAMES = (
    "moire_wall_energy_readout_v01.json",
    "mos2_wall_landscape_v01.json",
    "mos2_geometry_probe_v005.json",
    "mos2_camera_transfer_v01.json",
)
CAMERA = "transferred_even"
PRIMARY_SCANS = ("DS3", "DS4", "DS6", "DS9", "DS10", "DS12")
PRIMARY_SECTIONS = 168
BLUR_STEP_NM = 0.025
CENTER = len(DISTANCE) // 2


def corner_distances(section, basis, n, corners, weights):
    """Project the exact four pixel centers used by the periodic calculation."""
    phase = section["phase"][:, CENTER]
    normal_gradient = (section["phase"][0, CENTER + 1] - phase[0]) / (DISTANCE[1] - DISTANCE[0])
    # Affine registry differences along the wall are multiples of the lattice
    # jump. The second reciprocal row extracts signed distance along the cut.
    projector = np.linalg.inv(np.column_stack((section["jump"], normal_gradient)))[1]
    pixel_phase = (np.moveaxis(corners, 0, -1) / n) @ basis.T
    distance = (pixel_phase - phase[None, :, None, :]) @ projector
    average = (distance * weights).sum(axis=0)
    np.testing.assert_allclose(average, np.broadcast_to(PROFILE_DISTANCE, average.shape),
                               atol=1e-8, rtol=0)
    return distance


def isolated_profiles(section, q, detector, material, basis, corner_nm, weights, step_nm):
    """For a straight wall, the circular Gaussian reduces to its normal marginal."""
    # The padded line contains all sampled pixel centers and eight probe
    # standard deviations at each end.
    sigma_nm = NOMINAL / np.sqrt(8 * np.log(2))
    radius = int(np.ceil((np.max(np.abs(corner_nm)) + 8 * sigma_nm) / step_nm))
    distance = step_nm * np.arange(-radius, radius + 1)
    displacement = isolated_field(distance, section["jump"], material, basis,
                                  section["slip_sign"], section["k_factor"])
    intensity = detector_intensity(displacement, q, detector)
    blurred = gaussian_filter1d(intensity, sigma_nm / step_nm, axis=0, mode="nearest", truncate=8.)

    direct = np.column_stack([np.interp(PROFILE_DISTANCE, distance, channel) for channel in blurred.T])
    pixels = np.stack([np.interp(corner_nm, distance, channel) for channel in blurred.T], axis=-1)
    sampled = np.sum(pixels * weights[..., None], axis=0).mean(axis=0)
    return direct, sampled


def replay_legacy_isolated(section, q, compact, rings, material, basis, prior):
    """Verify the exact historical isolated model using its original window and optics."""
    displacement = isolated_field(PROFILE_DISTANCE, section["jump"], material, basis,
                                  section["slip_sign"], section["k_factor"])
    fields = {"isolated": np.broadcast_to(displacement, (3, 5, *displacement.shape))}
    _, hermite_weights = np.polynomial.hermite.hermgauss(5)
    intensity = render_profiles(fields, q, compact, hermite_weights / np.sqrt(np.pi), NOMINAL)["isolated"]
    fit = width_pair(Profiles(section, q, compact, rings, NOMINAL, "sine_gordon"), intensity[CROP])
    check_frozen(fit, {"qualified": [True, True], "widths_nm": prior})
    return fit["widths_nm"]


def compare_scan(context, images, modes, metadata, old, energy, frozen, optics):
    basis, q, rings = (context[key] for key in ("basis", "reciprocal", "rings"))
    sections = geometry_sections(context, rotation(np.deg2rad(float(metadata["SampleRotation"]))))
    saved_geometry = energy["walls"]
    if [s["centers"] for s in sections] != [w["centers"] for w in saved_geometry]:
        raise ValueError("Wall geometry has changed")
    eligible = [(s, w) for s, w in zip(sections, saved_geometry, strict=True)
                if w["original_cohort"] and s["reach_nm"] >= HALF_WINDOW]
    old_by_wall = {tuple(w["centers"]): w for w in old["chords"]}
    nominal_rows = frozen["walls"]
    if [s["centers"] for s, _ in eligible] != [w["centers"] for w in nominal_rows]:
        raise ValueError("The final scan-grid comparison uses a different set of wall sections")
    if {tuple(s["centers"]) for s, _ in eligible} != set(old_by_wall):
        raise ValueError("The earlier comparison uses a different set of wall sections")

    physics = energy["physics"]
    n = physics["grid"]
    stiffness = physics["stiffness_eV_per_cell"]
    terms = np.asarray(physics["stacking_coefficients_meV_per_cell"]["anchor"])
    material = (terms, stiffness)
    grid = make_grid(context["gradient"], basis, n, *stiffness)
    motion, solver = minimize_all(grid, terms)
    gain, offset = optics["frozen_common_coefficients"]
    detector = np.broadcast_to([gain, 0., offset], (len(q), 3))
    registry = grid.fractional @ basis.T + motion
    spectrum = np.fft.fft2(detector_intensity(registry, q, detector), axes=(0, 1))
    compact = np.asarray(energy["cameras"]["compact"]["channels"])
    mode_index = {tuple(mode): i for i, mode in enumerate(modes)}
    pixel_nm = context["receipt"]["pixel_nm"]

    results = []
    resolution_check = None
    for (section, original), saved in zip(eligible, nominal_rows, strict=True):
        checkpoint_guard()
        prior = old_by_wall[tuple(section["centers"])]["widths_nm"]["1.25"]
        legacy_widths = replay_legacy_isolated(section, q, compact, rings, material, basis,
                                               prior["isolated_anchor"])
        channels = [np.asarray([mode_index[tuple(mode)] for mode in ring])
                    for ring in original["miller_indices"]]
        view = {**section, "keep": abs(DISTANCE) <= HALF_WINDOW}
        reader = get_observer(view, q, detector, rings, NOMINAL, channels)
        measured_trace = sample_cuts(images, section["coordinates"]).mean(axis=0)
        measured = width_pair(reader, measured_trace)
        target = reader.shape(measured_trace)
        reference = saved["nominal"][CAMERA]
        check_frozen(measured, reference["measured"])

        frequency, cuts, corners, weights = raster_coordinates(section, basis, pixel_nm, n)
        network = render_traces(spectrum, gaussian_gain(frequency, NOMINAL), cuts, corners, weights)
        corner_nm = corner_distances(section, basis, n, corners, weights)
        isolated = isolated_profiles(section, q, detector, material, basis, corner_nm, weights, BLUR_STEP_NM)
        fits = {
            "network": dict(zip(("direct", "scan_grid"),
                                [score(reader, target, trace) for trace in network], strict=True)),
            "isolated": dict(zip(("direct", "scan_grid"),
                                 [score(reader, target, trace) for trace in isolated], strict=True)),
        }
        for sampling, key in (("direct", "continuous"), ("scan_grid", "native")):
            network_fit = fits["network"][sampling]
            stored_fit = reference["models"]["anchor"][key]
            check_frozen(network_fit, stored_fit)
            np.testing.assert_allclose(network_fit["transfer_error"], stored_fit["transfer_error"],
                                       atol=1e-8, rtol=0)

        if resolution_check is None:
            refined_trace = isolated_profiles(section, q, detector, material, basis, corner_nm, weights,
                                               BLUR_STEP_NM / 2)[1]
            refined = width_pair(reader, refined_trace[CROP])
            base = fits["isolated"]["scan_grid"]
            if refined["qualified"] != base["qualified"]:
                raise ValueError("Refining the isolated-prediction sampling changed fit qualification")
            delta = np.asarray(refined["widths_nm"]) - base["widths_nm"]
            if np.max(abs(delta)) > .002:
                raise ValueError("Refining the isolated-prediction sampling moved a fitted width")
            resolution_check = {
                "centers": section["centers"],
                "refined_minus_base_widths_nm": delta.tolist(),
                "maximum_absolute_normalized_intensity_change": float(np.max(abs(refined_trace - isolated[1]))),
            }
        results.append({
            "centers": section["centers"],
            "midpoint_xy_nm": saved["midpoint_xy_nm"],
            "spacing_nm": section["spacing_nm"],
            "measured": measured,
            "models": fits,
            "legacy_widths_nm": {
                "measured": prior["measured"],
                "network": prior["periodic_anchor"],
                "isolated": legacy_widths,
            },
        })
    return {"twist_deg": frozen["twist_deg"], "solver": solver,
            "pixel_spacing_nm": pixel_nm, "blur_grid_check": resolution_check, "walls": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "release", "dependencies", "evidence", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    start = perf_counter()
    sources = [args.evidence / name for name in INPUT_NAMES]
    earlier, landscape, final, optics = (json.loads(p.read_text()) for p in sources)
    saved_sources = [SOURCE, SETTINGS, BENCHMARK]
    source, settings, benchmark = (json.loads(p.read_text()) for p in saved_sources)
    scans = final["requested_scans"]
    if {name for name in scans if earlier["scans"][name]["twist_deg"] >= 1.0} != set(PRIMARY_SCANS):
        raise ValueError("The higher-twist comparison maps changed")
    if sum(len(earlier["scans"][name]["chords"]) for name in PRIMARY_SCANS) != PRIMARY_SECTIONS:
        raise ValueError("The primary comparison sections changed")
    libraries = ("bilayer_relaxation", "author_release", "run_moire_common_intensity",
                 "run_mos2_wall_width_transfer", "run_mos2_wall_profile_closure",
                 "run_mos2_wall_landscape", "run_mos2_observation_controls", "run_mos2_nonlinear_controls")
    paths = [Path(__file__), *sources, *saved_sources,
             *(Path(__file__).parent / f"{name}.py" for name in libraries),
             args.release.parents[1] / "release.zip", *(args.archive / f"{name}.zip" for name in scans)]
    result = {
        "schema": "mos2-matched-scan-grid-wall-models-v01",
        "primary": ("Mean over six higher-twist maps of the absolute prediction error in the map-mean "
                    "first-minus-second fitted width, on the original matched section IDs. Use only "
                    "walls where the measured and both predicted widths qualify in both orders."),
        "controls": ("Use the published 13.2 meV energy, archived crystal orientations, three cuts, "
                     "recorded scan positions, Gaussian probe, intensity coefficients from four other maps, "
                     "reflection lists and fitting procedure for both models. Also evaluate both predictions "
                     "directly at the cut positions under the same settings."),
        "limit": ("All maps were previously examined. The periodic network includes neighboring walls "
                  "and residual displacement in the domains. The comparison concerns measured diffraction "
                  "widths and profiles at the specified probe and material settings."),
        "probe_fwhm_nm": NOMINAL,
        "half_window_nm": HALF_WINDOW,
        "isolated_blur_step_nm": BLUR_STEP_NM,
        "primary_scans": list(PRIMARY_SCANS),
        "primary_sections": PRIMARY_SECTIONS,
        "requested_scans": scans,
        "input_sha256": {p.name: digest(p) for p in paths},
        "scans": {},
    }
    if args.output.is_file():
        if not args.resume:
            raise FileExistsError("Use --resume to extend an existing result")
        previous = json.loads(args.output.read_text())
        if any(previous[key] != value for key, value in result.items() if key != "scans"):
            raise ValueError("The saved checkpoint has different inputs or analysis settings")
        result = previous
    elapsed = result.get("seconds", 0.)
    reader = load_author_release(args.release, args.dependencies)
    for scan in scans:
        if scan in result["scans"]:
            continue
        checkpoint_guard()
        record = source["scans"][scan]
        metadata = read_input_metadata(scan, args.archive)
        if metadata["archive_md5"] != benchmark["scans"][scan]["archive_md5"]:
            raise ValueError(f"Public diffraction archive changed for {scan}")
        images, modes, payload_sha = disk_images(args.archive, scan, reader)
        if payload_sha != record["diskset_sha256"]:
            raise ValueError(f"Integrated diffraction intensities changed for {scan}")
        fields = metadata["metadata_fields"]
        context = centers_and_context(images, modes, fields, settings, tuple(record["interface"]))
        if context["receipt"] != record["geometry"]:
            raise ValueError(f"Archived geometry changed for {scan}")
        result["scans"][scan] = compare_scan(context, images, modes, fields, earlier["scans"][scan],
                                             landscape["scans"][scan], final["scans"][scan], optics)
        result["seconds"] = elapsed + perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, result)
        print(scan, len(result["scans"][scan]["walls"]), round(result["seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
