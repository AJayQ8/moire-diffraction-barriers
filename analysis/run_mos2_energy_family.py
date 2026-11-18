"""Repeat the MoS2 orientation comparison with fixed AB and AA curvatures."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from run_mos2_saddle_sweep import (
    BARRIERS,
    BENCHMARK,
    CAMERAS,
    CROP,
    DISTANCE,
    HALF_WINDOW,
    INPUT_NAMES,
    MODES,
    NOMINAL,
    PROJECT,
    REFERENCE_ME_V,
    REPLAY_MODULES,
    SETTINGS,
    SOURCE,
    centers_and_context,
    check_frozen,
    digest,
    disk_images,
    gaussian_gain,
    geometry_sections,
    get_observer,
    load_author_release,
    make_spectra,
    raster_coordinates,
    read_input_metadata,
    render_traces,
    rotation,
    stack_energy,
    width_pair,
)
from shared_compute import atomic_json, checkpoint_guard


def fixed_curvature_landscapes(basis, reference):
    """Preserve the AB energy and both site Hessians as the saddle varies."""
    if np.any(reference[4:]):
        raise ValueError("This family requires the even energy of a parallel bilayer")
    # In the three cosine shells the weights for the AA and AB curvatures
    # are (1, 3, 4) and (-1/2, 3, -2). The barrier weights are
    # (1/2, -4, 9/2). Solving these three constraints gives the change
    # below. The constant also keeps the reference energy of AB fixed.
    direction = np.array([-1.8, -1.6, 0., .4, 0., 0.])
    ab, aa, sp = np.array([[1/3, 1/3], [0, 0], [.5, 0]]) @ basis.T
    sites = np.stack([ab, aa, sp])
    reciprocal = 2 * np.pi * MODES @ np.linalg.inv(basis)
    cosines = np.cos(sites @ reciprocal.T)

    def hessians(coefficients):
        weights = -cosines * np.repeat(coefficients[1:4], 3) / 1000
        return (reciprocal.T * weights[:, None, :]) @ reciprocal

    reference_energies, _ = stack_energy(sites, basis, reference)
    reference_hessians = hessians(reference)
    family, descriptions = {}, {}
    for setting in BARRIERS:
        coefficients = reference + (float(setting) - REFERENCE_ME_V) * direction
        energies, gradient = stack_energy(sites, basis, coefficients)
        hessian = hessians(coefficients)
        np.testing.assert_allclose(gradient, 0., atol=1e-12, rtol=0)
        np.testing.assert_allclose(energies[0], reference_energies[0], atol=1e-12, rtol=0)
        np.testing.assert_allclose(hessian[:2], reference_hessians[:2], atol=1e-12, rtol=0)
        ab_eigenvalues, aa_eigenvalues, sp_eigenvalues = np.linalg.eigvalsh(hessian)
        if min(ab_eigenvalues) <= 0 or max(aa_eigenvalues) >= 0:
            raise ValueError(f"{setting} meV fails to preserve stable AB and a local maximum at AA")
        if not sp_eigenvalues[0] < 0 < sp_eigenvalues[1] or energies[1] <= energies[2]:
            raise ValueError(f"{setting} meV fails to preserve the saddle below AA")
        barrier = 1000 * (energies[2] - energies[0])
        np.testing.assert_allclose(barrier, float(setting), atol=1e-10, rtol=0)
        family[setting] = coefficients
        descriptions[setting] = {
            "coefficients_meV_per_cell": coefficients.tolist(),
            "saddle_meV_per_cell": float(barrier),
            "AA_minus_AB_meV_per_cell": float(1000 * (energies[1] - energies[0])),
            "AB_hessian_eigenvalues_eV_nm2": ab_eigenvalues.tolist(),
            "AA_hessian_eigenvalues_eV_nm2": aa_eigenvalues.tolist(),
            "saddle_hessian_eigenvalues_eV_nm2": sp_eigenvalues.tolist(),
        }
    return family, descriptions


def evaluate(scan, context, modes, metadata, original, camera, optics, baseline):
    basis, q, rings = (context[key] for key in ("basis", "reciprocal", "rings"))
    angle = np.deg2rad(float(metadata["SampleRotation"]))
    sections = geometry_sections(context, rotation(angle))
    source_walls = original["walls"]
    if [s["centers"] for s in sections] != [w["centers"] for w in source_walls]:
        raise ValueError(f"{scan}: the measured wall geometry changed")
    selected = [(section, source) for section, source in zip(sections, source_walls, strict=True)
                if source["original_cohort"] and section["reach_nm"] >= HALF_WINDOW]
    if [s["centers"] for s, _ in selected] != [w["centers"] for w in baseline["walls"]]:
        raise ValueError(f"{scan}: the width comparison no longer uses the same sections")

    physics = original["physics"]
    reference = np.asarray(physics["stacking_coefficients_meV_per_cell"]["anchor"])
    family, landscapes = fixed_curvature_landscapes(basis, reference)
    terms = optics["frozen_common_coefficients"]
    common_response = np.broadcast_to([terms[0], 0., terms[1]], (len(q), 3))
    disjoint = np.asarray(camera["camera_diagnostic"]["by_model"]["anchor"]["coefficients"])
    detectors = dict(zip(CAMERAS, (common_response, disjoint), strict=True))
    spectra, solvers = make_spectra(scan, context, physics, family, detectors, BARRIERS)

    mode_index = {tuple(mode): i for i, mode in enumerate(modes)}
    walls = []
    for (section, source), saved in zip(selected, baseline["walls"], strict=True):
        checkpoint_guard()
        channels = [np.asarray([mode_index[tuple(mode)] for mode in group])
                    for group in source["miller_indices"]]
        window = {**section, "keep": abs(DISTANCE) <= HALF_WINDOW}
        fit_width = get_observer(window, q, common_response, rings, NOMINAL, channels)
        freq2, cuts, corners, weights = raster_coordinates(section, basis, original["pixel_nm"], physics["grid"])
        gain = gaussian_gain(freq2, NOMINAL)
        arms = {name: {} for name in CAMERAS}
        for frame, cases in spectra.items():
            for camera_name in CAMERAS:
                fits = {}
                for setting, spectra_by_detector in cases.items():
                    _, trace = render_traces(spectra_by_detector[camera_name], gain, cuts, corners, weights)
                    fits[setting] = width_pair(fit_width, trace[CROP])
                check_frozen(fits["13.2"], saved["arms"][camera_name][frame]["13.2"])
                arms[camera_name][frame] = fits
        walls.append({**saved, "arms": arms})
    return {"landscapes": landscapes, "solvers": solvers, "walls": walls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "release", "dependencies", "evidence", "baseline", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    start = perf_counter()
    files = (SOURCE, BENCHMARK, SETTINGS, *(args.evidence / name for name in INPUT_NAMES))
    source, benchmark, settings, primary, cameras, optics = (json.loads(f.read_text()) for f in files)
    baseline = json.loads(args.baseline.read_text())
    names = primary["protocol"]["scans"]
    if list(BARRIERS) != baseline["settings_meV_per_cell"] or set(names) != set(baseline["scans"]):
        raise ValueError("The comparison requires the complete original ten-map sweep")
    replay_modules = (*REPLAY_MODULES, "run_mos2_saddle_sweep")
    hashes = {f.name: digest(f) for f in (Path(__file__), args.baseline, *files,
                                          *(PROJECT / f"{name}.py" for name in replay_modules))}
    hashes["release.zip"] = digest(args.release.parents[1] / "release.zip")
    shared_inputs = set(hashes) & set(baseline["input_sha256"])
    for name in shared_inputs:
        if hashes[name] != baseline["input_sha256"][name]:
            raise ValueError(f"{name} changed relative to the original sweep")

    report = {
        "schema": "mos2-fixed-AB-and-AA-curvatures-saddle-sweep-v01",
        "requested_scans": names,
        "settings_meV_per_cell": list(BARRIERS),
        "probe_fwhm_nm": NOMINAL,
        "half_window_nm": HALF_WINDOW,
        "scope": ("Use the original experiment and settings with a new energy family that fixes AB energy "
                  "and the AB and AA curvatures. AA minus AB energy varies. Replay 13.2 meV in each scan, "
                  "orientation and detector fit against the original sweep."),
        "input_sha256": hashes,
        "scans": {},
    }
    elapsed = 0.
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        previous = json.loads(args.output.read_text())
        if any(previous[key] != report[key] for key in report if key != "scans"):
            raise ValueError("The checkpoint uses different inputs or settings")
        report["scans"] = previous["scans"]
        elapsed = previous["seconds"]
    author = load_author_release(args.release, args.dependencies)
    for scan in names:
        if scan in report["scans"]:
            continue
        checkpoint_guard()
        meta = read_input_metadata(scan, args.archive)
        images, modes, checksum = disk_images(args.archive, scan, author)
        ref = source["scans"][scan]
        if checksum != ref["diskset_sha256"] or meta["archive_md5"] != benchmark["scans"][scan]["archive_md5"]:
            raise ValueError(f"{scan}: the archived intensities changed")
        fields = meta["metadata_fields"]
        context = centers_and_context(images, modes, fields, settings, tuple(ref["interface"]))
        if context["receipt"] != ref["geometry"]:
            raise ValueError(f"{scan}: the inferred geometry changed")
        report["scans"][scan] = evaluate(scan, context, modes, fields, primary["scans"][scan],
                                         cameras["scans"][scan], optics, baseline["scans"][scan])
        report["seconds"] = elapsed + perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, report)
        print(f"{scan}: {len(report['scans'][scan]['walls'])} wall sections checked", flush=True)


if __name__ == "__main__":
    main()
