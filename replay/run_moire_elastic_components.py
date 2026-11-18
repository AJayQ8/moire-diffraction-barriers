"""Test compressional and rotational displacement contributions separately."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull

from author_detector_replay import detector_intensity, rms
from author_release import load_author_release
from bilayer_relaxation import MONOLAYERS, sample_disregistry
from observe_bilayers import _good_interior_triangle_mask, _inside_polygon_rc, load_dataset
from run_antiparallel_signed_audit import local_geometry
from run_detector_prediction import AUTHOR_BASIS, apply_spatial
from run_heterobilayer_corrugation import BASIS, MODEL_A, ORIGIN
from run_heterobilayer_diffraction_check import read_native_detector, register_frame
from run_heterobilayer_ring_geometry import mean_rows
from run_heterobilayer_sparse_geometry import solve_from_geometry
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_regime_benchmark import SCAN_IDS, SETTINGS, detector_data
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


BENCHMARK = PROJECT / "moire_regime_benchmark_v02.json"
RING_RESULTS = tuple(PROJECT / f"heterobilayer_first_ring_seed_{batch}_v01.json" for batch in ("qualification", "replay"))
RING_FREEZE = PROJECT / "heterobilayer_first_ring_seed_freeze_v01.json"


def split_displacement(grid, full):
    """Helmholtz-decompose the relative field in the physical periodic cell."""
    k = grid.k
    squared = np.sum(k*k, axis=-1)
    spectrum = np.fft.fft2(full, axes=(0, 1))
    coefficient = np.sum(k * spectrum, axis=-1) / np.where(squared > 0, squared, 1)
    compressional = np.fft.ifft2(coefficient[..., None] * k, axes=(0, 1)).real
    rotational = full - compressional
    energies = np.array([np.mean(np.sum(u*u, axis=-1)) for u in (compressional, rotational, full)])
    if abs(energies[0] + energies[1] - energies[2]) > 1e-10:
        raise ValueError("The periodic Helmholtz components do not conserve displacement power")
    return {"rigid": np.zeros_like(full), "compression_only": compressional,
            "rotation_only": rotational, "full": full}, float(energies[0] / energies[2])


def detector_scores(grid, full, phase, origin, scale, measured, reciprocal, coefficients,
                    support, rings, sigma_pixels, orient):
    components, power = split_displacement(grid, full)
    scores = {}
    for label, displacement in components.items():
        vector = (sample_disregistry(grid, displacement, phase, grid.cell) - origin) / scale
        intensity = detector_intensity(orient(vector), reciprocal, coefficients)
        filtered = gaussian_filter(intensity, (sigma_pixels, sigma_pixels, 0), mode="nearest")
        scores[label] = {ring: rms(measured[support][:, channels], filtered[support][:, channels])
                         for ring, channels in rings.items()}
    return scores, power


def common_rings(modes):
    h, k = modes.T
    ring_index = np.rint(h*h + h*k + k*k).astype(int)
    return {"first": ring_index == 1, "second": ring_index == 3, "both": np.ones(len(h), dtype=bool)}


def check_model_score(score, original):
    if any(abs(score[key] - original[key]) > 1e-10 for key in original):
        raise ValueError("The ablation did not reproduce its full-model reference")


def same_scan_scores(data, detector, reference, settings):
    measured, fitted, modes, coefficients, _ = detector
    crop, turns, reflect, transform, alignment = register_frame(data, fitted, modes, coefficients)
    if alignment != reference["detector_alignment"]:
        raise ValueError("The benchmark calibration frame changed")
    measured = measured[crop]
    reciprocal = modes @ transform / data.lattice_nm
    rings = common_rings(modes)
    interface = tuple(reference["interface"])
    scale = MONOLAYERS["MoS2"][0] / data.lattice_nm if interface[1] != "MoS2" else 1
    basis = data.lattice_nm * np.array(((1.0, 0.5), (0.0, np.sqrt(3) / 2)))
    origin = scale * basis @ np.full(2, 2/3 if data.orientation == "ap" else 0)
    fit = np.asarray(reference["affine_geometry"]["fit_scan"])
    gradient = np.asarray(reference["affine_geometry"]["gradient_crystal"])
    shape = data.u_nm.shape[:2]
    yy, xx = np.indices(shape, dtype=float)
    points = np.stack((xx, yy), axis=-1) * data.pixel_nm
    nominal = points @ fit[:2] + fit[-1]
    correction, _ = local_geometry(data, points, basis, fit, settings["center_inlier_nm"])
    warped = nominal + correction
    valid = data.source_valid_mask
    sigma = settings["smoothing_sigma_pixels"]
    weights = gaussian_filter(valid.astype(float), sigma, mode="constant")
    interior, _ = _good_interior_triangle_mask(data)
    distance2 = np.full(shape, np.inf)
    for y, x in data.centers_rc_px:
        distance2 = np.minimum(distance2, (yy-y)**2 + (xx-x)**2)
    support = interior & valid & (weights > 0.97) & (distance2 > (2*sigma)**2)
    support[:4] = support[-4:] = support[:, :4] = support[:, -4:] = False
    if support.sum() != reference["score_pixels"]:
        raise ValueError("The common comparison support changed")
    grid, full, _ = solve_from_geometry(scale * gradient, scale * basis, interface)
    probe = settings["probe_fwhm_nm"] / (data.pixel_nm * np.sqrt(8 * np.log(2)))

    def orient(field):
        return apply_spatial(field, turns, reflect)

    results = {}
    for name, rigid in (("affine", nominal), ("warped", warped)):
        scores, power = detector_scores(grid, full, origin + scale*rigid, origin, scale, measured, reciprocal,
                                       coefficients, orient(support), rings, probe, orient)
        check_model_score(scores["full"], reference["models"][name]["full"]["diffraction_rmse"])
        check_model_score(scores["rigid"], reference["models"][name]["rigid"]["diffraction_rmse"])
        results[name] = scores
    return {"interface": interface, "compressional_displacement_power_fraction": power,
            "pixels": int(support.sum()), "scores": results}


def cross_ring_scores(data, detector, reference, settings):
    measured, _, modes, coefficients, _ = detector
    frame = reference["first_ring_frame"]
    mirror = np.diag((-1, 1)) if frame["vector_reflection"] else np.eye(2)
    transform = np.linalg.inv(AUTHOR_BASIS) @ rotation(frame["vector_rotation_60deg_steps"] * np.pi/3) @ mirror
    reciprocal = modes @ transform / data.lattice_nm
    row, col = data.metadata.get("author_unwrap_crop_origin_row_col", (0, 0))
    height, width = data.u_nm.shape[:2]
    measured = measured[row:row+height, col:col+width]
    scale = MODEL_A / data.lattice_nm
    rings = {k: v for k, v in common_rings(modes).items() if k != "both"}
    centers = np.asarray(reference["first_ring_centers_rc_px"])
    labels = np.asarray(reference["lattice_indices"])
    translations = labels @ (BASIS / scale).T
    groups = (labels[:, 0] - labels[:, 1]) % 3
    rows, columns = np.indices((height, width), dtype=float)
    points = np.stack((columns, rows), axis=-1) * data.pixel_nm
    center_points = centers[:, ::-1] * data.pixel_nm
    distance2 = np.full((height, width), np.inf)
    for y, x in centers:
        distance2 = np.minimum(distance2, (rows-y)**2 + (columns-x)**2)
    support = distance2 > (2 * settings["smoothing_sigma_pixels"])**2
    support[:4] = support[-4:] = support[:, :4] = support[:, -4:] = False
    angle = np.deg2rad(data.sample_rotation_degrees)
    probe = settings["probe_fwhm_nm"] / (data.pixel_nm * np.sqrt(8 * np.log(2)))
    folds = []
    for fold in reference["folds"]:
        checkpoint_guard()
        train = groups != fold["withheld_group"]
        inplane = np.column_stack((center_points[train], np.ones(train.sum())))
        fit = np.linalg.lstsq(inplane, translations[train], rcond=None)[0]
        nominal = points @ fit[:2] + fit[-1]
        delta = translations - (center_points @ fit[:2] + fit[-1])
        local = RBFInterpolator(center_points[train], delta[train], kernel="thin_plate_spline")
        warped = nominal + local(points.reshape(-1, 2)).reshape(points.shape)
        hull = ConvexHull(centers[train])
        keep = support & _inside_polygon_rc((height, width), centers[train][hull.vertices])
        if keep.sum() != fold["pixels"]:
            raise ValueError("The first-ring support changed")
        gradient = fit[:2].T @ rotation(-angle)
        grid, full, _ = solve_from_geometry(scale*gradient, BASIS, ("MoS2", "WSe2", "AP"))
        scores = {}
        for name, rigid in (("affine", nominal), ("warped", warped)):
            scores[name], power = detector_scores(grid, full, ORIGIN + scale*rigid, ORIGIN, scale, measured,
                                                  reciprocal, coefficients, keep, rings, probe, lambda field: field)
            check_model_score(scores[name]["full"], fold["diffraction_rmse"][f"{name}_relaxed"])
            check_model_score(scores[name]["rigid"], fold["diffraction_rmse"][f"{name}_rigid"])
        folds.append({"group": fold["withheld_group"], "pixels": fold["pixels"], "scores": scores,
                      "compressional_displacement_power_fraction": power})
    return {"folds": folds, "mean_over_folds": mean_rows([row["scores"] for row in folds])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    settings = json.loads(SETTINGS.read_text())
    benchmark = json.loads(BENCHMARK.read_text())
    frozen = json.loads(RING_FREEZE.read_text())
    for name, expected in {**benchmark["input_sha256"], **frozen["input_sha256"]}.items():
        if digest(PROJECT / name) != expected:
            raise ValueError(f"An original benchmark input changed: {name}")
    if digest(RING_RESULTS[0]) != frozen["qualification_sha256"]:
        raise ValueError("The first-ring qualification changed")
    rings = {scan: row for path in RING_RESULTS for scan, row in json.loads(path.read_text())["scans"].items()}
    author = load_author_release(args.release, args.dependencies)
    inputs = (Path(__file__), SETTINGS, BENCHMARK, RING_FREEZE, *RING_RESULTS)
    output = {
        "scope": "Retrospective parameter-free ablation of the periodic full displacement. Project each Fourier coefficient onto or perpendicular to its physical wavevector before sampling on the measured geometry. The components are controls, not independently relaxed states. The center-based geometry, detector parameters and mask match the corresponding benchmark.",
        "input_sha256": {p.name: digest(p) for p in inputs}, "same_scan": {}, "cross_ring": {},
    }
    for scan in SCAN_IDS:
        checkpoint_guard()
        data = load_dataset(scan, args.archive, allowed_ids=SCAN_IDS, allow_author_crop=True)
        reference = benchmark["scans"][scan]
        if data.archive_md5 != reference["archive_md5"]:
            raise ValueError("The public ZIP changed")
        detector = detector_data(args.archive, scan, author)
        output["same_scan"][scan] = same_scan_scores(data, detector, reference, settings)
        if scan in rings:
            native = read_native_detector(args.archive, scan, author)
            output["cross_ring"][scan] = cross_ring_scores(data, native, rings[scan], settings)
        print(scan, output["same_scan"][scan]["compressional_displacement_power_fraction"], flush=True)
        output["wall_seconds"] = perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, output)


if __name__ == "__main__":
    main()
