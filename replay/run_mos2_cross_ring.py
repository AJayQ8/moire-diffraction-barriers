"""Use first-ring MoS2 fringes as inputs and score the second-ring signal."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

from author_detector_replay import detector_intensity, rms
from author_release import load_author_release
from bilayer_relaxation import ConvergenceError, sample_disregistry
from observe_bilayers import _inside_polygon_rc, load_dataset
from run_detector_prediction import AUTHOR_BASIS
from run_detector_wall_profiles import to_scan_frame
from run_heterobilayer_cross_ring import refine_minimum
from run_heterobilayer_sparse_geometry import linear_response, solve_from_geometry
from run_heterobilayer_template_transfer import design_matrix
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_elastic_components import common_rings
from run_moire_regime_benchmark import SETTINGS, detector_data, relaxed_ray
from run_mos2_template_transfer import BENCHMARK
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


TEMPLATES = PROJECT / "mos2_template_replay_v01.json"
SEARCH_FRACTION = 0.20


def smooth_cost(values, coefficients, reciprocal, center):
    expected = detector_intensity(center, reciprocal, coefficients)
    contrast = np.hypot(coefficients[:, 0], coefficients[:, 1])
    if np.min(contrast) < 1e-4:
        raise ValueError("A first-ring channel has insufficient modulation")
    cost = np.mean(((values - expected) / contrast)**2, axis=-1)
    return gaussian_filter(cost, 1.0, mode="nearest")


def locate_centers(data, fit, basis, values, reciprocal, coefficients, center_offset):
    """Re-locate each archived integer center from first-ring intensities."""
    # Retain the lattice indices of the reference geometry. Intensity
    # coefficients and labels remain the shared calibration; positions are
    # recovered from the first-ring images before predicting the second.
    iy, ix = np.rint(data.centers_rc_px).astype(int).T
    observed = data.u_nm[iy, ix]
    centers = data.centers_rc_px[np.isfinite(observed).all(axis=-1)]
    observed = observed[np.isfinite(observed).all(axis=-1)]
    xy = centers[:, ::-1] * data.pixel_nm
    labels = np.rint(observed @ np.linalg.inv(basis).T).astype(int)
    translations = labels @ basis.T
    deviation = translations - (xy @ fit[:2] + fit[-1])
    retain = np.linalg.norm(deviation, axis=-1) < 0.065
    if np.count_nonzero(retain) < 5:
        raise ValueError("Fewer than five labeled centers in the common gauge")
    centers, labels, translations = centers[retain], labels[retain], translations[retain]
    if len(np.unique(labels, axis=0)) != len(labels):
        raise ValueError("The center labels repeat the same lattice translation")
    lattice = np.linalg.solve(fit[:2].T, basis) / data.pixel_nm
    radius = SEARCH_FRACTION * min(np.linalg.norm(lattice[:, 0]), np.linalg.norm(lattice[:, 1]),
                                   np.linalg.norm(lattice[:, 0]-lattice[:, 1]))
    cost = smooth_cost(values, coefficients, reciprocal, center_offset)
    yy, xx = np.indices(cost.shape)
    internal = (yy >= 2) & (yy < cost.shape[0]-2) & (xx >= 2) & (xx < cost.shape[1]-2)
    nominal = (translations - fit[-1]) @ np.linalg.inv(fit[:2]) / data.pixel_nm
    positions, indices = [], []
    for i, (x, y) in enumerate(nominal):
        inside = internal & ((yy-y)**2 + (xx-x)**2 < radius**2)
        if not inside.any():
            continue
        j, k = np.unravel_index(np.argmin(np.where(inside, cost, np.inf)), cost.shape)
        center = refine_minimum(cost, j, k)
        if np.linalg.norm(center - (y, x)) < radius - 0.5:
            positions.append(center)
            indices.append(i)
    if len(indices) < 5:
        raise ValueError("The first ring could not locate five centers")
    recovered = np.asarray(positions)
    difference = recovered - centers[indices]
    return recovered, translations[indices], {
        "labeled_centers": len(centers), "first_ring_centers": len(recovered),
        "mean_center_disagreement_nm": float(data.pixel_nm * np.mean(np.linalg.norm(difference, axis=1))),
        "max_center_disagreement_nm": float(data.pixel_nm * np.max(np.linalg.norm(difference, axis=1))),
        "search_radius_nm": float(radius * data.pixel_nm), "calibrated_target_registry_nm": center_offset.tolist(),
    }


def calibrated_origin(observed, first, reciprocal, coefficients, fit, points, basis):
    """Estimate a uniform phase offset from the first ring on the fixed affine seed."""
    a = np.linalg.norm(basis[:, 0])
    nominal = points @ fit[:2] + fit[-1]
    sample = np.s_[4:-4:3, 4:-4:3]
    x = nominal[sample]
    y = observed[sample][..., first]
    q, coeff = reciprocal[first], coefficients[first]
    limit = a / 6

    def residual(delta):
        return (detector_intensity(x + delta, q, coeff) - y).ravel()

    result = least_squares(residual, np.zeros(2), bounds=(-limit, limit), max_nfev=35)
    if not result.success:
        raise ValueError("The first-ring phase calibration did not converge")
    return result.x


def run_scan(data, detector, reference, learned, settings, recalibrate=False):
    measured, _, modes, coefficients, _ = detector
    align = reference["detector_alignment"]
    turns, reflection = align["spatial_turns"], align["spatial_reflection"]
    vector_reflect = np.diag((-1, 1)) if align["vector_reflection"] else np.eye(2)
    transform = np.linalg.inv(AUTHOR_BASIS) @ rotation(align["vector_rotation_60deg_steps"] * np.pi/3) @ vector_reflect
    reciprocal = modes @ transform / data.lattice_nm
    start_row, start_col = data.metadata.get("author_unwrap_crop_origin_row_col", (0, 0))
    height, width = data.u_nm.shape[:2]
    measured = to_scan_frame(measured[start_row:start_row+height, start_col:start_col+width], turns, reflection)
    rings = common_rings(modes)
    first = rings["first"]
    lattice = data.lattice_nm
    basis = lattice * np.array(((1.0, 0.5), (0.0, np.sqrt(3) / 2)))
    interface = tuple(reference["interface"])
    origin = basis @ np.full(2, 2/3 if data.orientation == "ap" else 0)
    yy, xx = np.indices((height, width), dtype=float)
    points = np.stack((xx, yy), axis=-1) * data.pixel_nm
    initial = np.asarray(reference["affine_geometry"]["fit_scan"])
    offset = calibrated_origin(measured, first, reciprocal, coefficients, initial, points, basis) if recalibrate else np.zeros(2)
    centers, labels, local = locate_centers(data, initial, basis, measured[..., first],
                                           reciprocal[first], coefficients[first], offset)
    xy = centers[:, ::-1] * data.pixel_nm
    fit = np.linalg.lstsq(np.column_stack((xy, np.ones(len(xy)))), labels, rcond=None)[0]
    residuals = np.linalg.norm(xy @ fit[:2] + fit[-1] - labels, axis=1)
    if np.max(residuals) > 0.10:
        raise ValueError("The first-ring centers do not support the affine lattice")
    gradient = fit[:2].T @ rotation(-np.deg2rad(data.sample_rotation_degrees))
    if np.linalg.det(gradient) * np.linalg.det(initial[:2]) <= 0:
        raise ValueError("The first-ring fit changed lattice handedness")
    rigid = points @ fit[:2] + fit[-1]
    support = _inside_polygon_rc((height, width), centers[ConvexHull(centers).vertices])
    distance2 = np.full((height, width), np.inf)
    for y, x in centers:
        distance2 = np.minimum(distance2, (yy-y)**2 + (xx-x)**2)
    support &= distance2 > (2*settings["smoothing_sigma_pixels"])**2
    support[:4] = support[-4:] = support[:, :4] = support[:, -4:] = False
    if support.sum() < 300:
        raise ValueError("The first-ring geometry covers too few interior pixels")
    grid, full, solve = solve_from_geometry(gradient, basis, interface)
    linear = linear_response(grid, interface)
    ray = relaxed_ray(grid, linear, interface)
    probe = settings["probe_fwhm_nm"] / (data.pixel_nm * np.sqrt(8 * np.log(2)))
    fields = {"rigid": rigid}
    for name, shift in (("first_order", linear), ("energy_scaled_first_order", ray*linear), ("full", full)):
        fields[name] = sample_disregistry(grid, shift, origin + rigid, grid.cell) - origin
    for name, template in learned.items():
        fields[name] = rigid + design_matrix(rigid, basis, np.asarray(template["modes"])) @ np.asarray(template["coefficients_nm"])
    scores = {}
    for name, field in fields.items():
        output = detector_intensity(field, reciprocal, coefficients)
        output = gaussian_filter(output, (probe, probe, 0), mode="nearest")
        scores[name] = {key: rms(measured[support][:, ring], output[support][:, ring]) for key, ring in rings.items()}
    return {"interface": interface, "score_pixels": int(support.sum()), "scores": scores, "solve": solve,
            "center_fit": local, "first_ring_affine": fit.tolist(), "archived_affine_seed": initial.tolist(),
            "maximum_affine_center_residual_nm": float(np.max(residuals))}


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
    templates = json.loads(TEMPLATES.read_text())
    for name, expected in {**benchmark["input_sha256"], **templates["input_sha256"]}.items():
        if digest(PROJECT / name) != expected:
            raise ValueError(f"Benchmark input {name} changed")
    author = load_author_release(args.release, args.dependencies)
    result = {
        "scope": "Retrospective split-ring check on the Van Winkle MoS2 scans. The saved affine registry and integer labels seed search windows. Re-locate centers and refit the lattice with first-ring intensity alone, then score the second ring away from those positions. Both rings inform the archived calibration and the seed. The comparison uses the same first-ring geometry for all physical and empirical predictions.",
        "choices": "Use the zero registry expected from the detector coefficients. As a sensitivity check, infer a uniform offset from first-ring intensities before center detection. Keep both checks.",
        "input_sha256": {p.name: digest(p) for p in (Path(__file__), SETTINGS, BENCHMARK, TEMPLATES)},
        "scans": {}, "failed_scans": {},
    }
    for orientation, batches in templates["split"].items():
        for scan in batches["development"] + batches["replay"]:
            checkpoint_guard()
            data = load_dataset(scan, args.archive, allowed_ids=(scan,), allow_author_crop=True)
            reference = benchmark["scans"][scan]
            if data.archive_md5 != reference["archive_md5"]:
                raise ValueError("The archive changed")
            detector = detector_data(args.archive, scan, author)
            row = {}
            for name, offset in (("published_zero", False), ("first_ring_phase_offset", True)):
                try:
                    row[name] = run_scan(data, detector, reference, templates["templates"][orientation], settings, offset)
                except (ConvergenceError, ValueError, np.linalg.LinAlgError) as exc:
                    result["failed_scans"].setdefault(scan, {})[name] = str(exc)
            if row:
                result["scans"][scan] = row
            print(scan, list(row), result["failed_scans"].get(scan), flush=True)
            result["wall_seconds"] = perf_counter() - start
            checkpoint_guard()
            atomic_json(args.output, result)


if __name__ == "__main__":
    main()
