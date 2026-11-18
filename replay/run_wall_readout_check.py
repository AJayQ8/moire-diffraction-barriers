"""Check microscope inversion and wall selection on already scored maps."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import RBFInterpolator

from bilayer_relaxation import relax, sample_disregistry
from observe_bilayers import (
    _all_triangles,
    _author_strain_gamma_percent,
    _good_interior_triangle_mask,
    _smooth_like_paper,
    _stacking_sector_labels,
    load_dataset,
)
from run_detector_prediction import apply_spatial, load_detector, match_archive_orientation
from shared_compute import atomic_json, checkpoint_guard


SCANS = ("DS1", "DS2", "DS7", "DS10")


def intensity(field, reciprocal, scale, offset):
    return scale * np.cos(np.pi * (field @ reciprocal.T))**2 + offset


def invert_intensity(target, initial, reciprocal, scale, offset):
    """Invert each synthetic pixel on the branch nearest the starting field."""
    field = initial.copy()
    phase_gradient = np.pi * reciprocal
    for _ in range(16):
        phase = field @ phase_gradient.T
        residual = scale * np.cos(phase)**2 + offset - target
        jacobian = (-scale * np.sin(2 * phase))[..., None] * phase_gradient
        transposed = jacobian.swapaxes(-2, -1)
        normal = transposed @ jacobian
        damping = 1e-5 + 1e-3 * np.trace(normal, axis1=-2, axis2=-1)
        step = np.linalg.solve(normal + damping[..., None, None] * np.eye(2),
                               transposed @ residual[..., None])[..., 0]
        length = np.linalg.norm(step, axis=-1)
        step *= np.minimum(1, 0.025 / np.maximum(length, 1e-12))[..., None]
        best_field = field
        best_cost = np.sum(residual**2, axis=-1)
        for fraction in (1, 0.5, 0.25):
            trial = field - fraction * step
            cost = np.sum((intensity(trial, reciprocal, scale, offset) - target)**2, axis=-1)
            improve = cost < best_cost
            best_field = np.where(improve[..., None], trial, best_field)
            best_cost = np.minimum(cost, best_cost)
        field = best_field
    return field


def wall_maps(dataset, field, triangles):
    supported = np.where(dataset.source_valid_mask[..., None], field, np.nan)
    smoothed = _smooth_like_paper(supported, float(dataset.metadata.get("SmoothingSigma", 2)))
    shear = _author_strain_gamma_percent(smoothed, dataset.pixel_nm, dataset.sample_rotation_degrees)
    labels, _, _ = _stacking_sector_labels(smoothed, dataset.lattice_nm, triangles & np.isfinite(shear))
    return shear, labels


def sector_means(shear, labels):
    return [float(np.mean(shear[labels == sector])) for sector in (1, 2, 3)]


def profile_shape(distance, profile):
    """Read a wall peak and its half-height width above the local background."""
    n_tail = max(2, len(distance) // 8)
    background = 0.5 * (profile[:n_tail].mean() + profile[-n_tail:].mean())
    peak = int(np.argmax(profile))
    height = profile[peak]
    level = 0.5 * (height + background)
    left = np.flatnonzero(profile[:peak] < level)
    right = np.flatnonzero(profile[peak + 1:] < level) + peak + 1
    width = None
    if left.size and right.size:
        lo, hi = left[-1], right[0]
        start = np.interp(level, profile[lo:lo + 2], distance[lo:lo + 2])
        stop = np.interp(level, profile[hi - 1:hi + 1][::-1], distance[hi - 1:hi + 1][::-1])
        width = float(stop - start)
    return {"peak_offset_nm": float(distance[peak]), "peak_percent": float(height),
            "background_percent": float(background), "half_height_width_nm": width}


def wall_profiles(dataset, shear_maps):
    """Read the same cross-sections between measured stacking centers."""
    edges = Counter()
    for i, j, k in _all_triangles(dataset.adjacency_type):
        edges.update((tuple(sorted(pair)) for pair in ((i, j), (j, k), (i, k))))
    sections = []
    for (i, j), count in edges.items():
        if count != 2:
            continue
        start, stop = dataset.centers_rc_px[[i, j]]
        edge = stop - start
        length = np.linalg.norm(edge)
        if length == 0:
            continue
        normal = np.array((-edge[1], edge[0])) / length
        reach = min(10.0, 0.28 * length * dataset.pixel_nm)
        offsets = np.linspace(-reach, reach, 2 * int(np.ceil(reach / 0.25)) + 1)
        along = start + np.linspace(0.3, 0.7, 7)[:, None] * edge
        points = along[:, None, :] + offsets[None, :, None] * normal / dataset.pixel_nm
        coordinates = points.transpose(2, 0, 1)
        valid = ndi.map_coordinates(dataset.source_valid_mask.astype(float), coordinates, order=1) > 0.999
        complete = np.all(valid, axis=1)
        if np.count_nonzero(complete) < 4:
            continue
        shapes = {}
        for name, shear in shear_maps.items():
            slices = ndi.map_coordinates(shear, coordinates[:, complete], order=1, cval=np.nan)
            if not np.isfinite(slices).all():
                break
            shapes[name] = profile_shape(offsets, slices.mean(axis=0))
        if len(shapes) == len(shear_maps):
            sections.append({"centers": [i, j], "wall_sector": int(dataset.adjacency_type[i, j]),
                             "half_window_nm": reach, "profiles": shapes})
    return sections


def local_geometry(dataset, points, fit, basis):
    """Warp the nominal lattice phase using only archived stacking centers."""
    centers = dataset.centers_rc_px[:, ::-1] * dataset.pixel_nm
    nominal = centers @ fit[:2] + fit[-1]
    lattice = np.rint(nominal @ np.linalg.inv(basis).T) @ basis.T
    residual = lattice - nominal
    inliers = np.linalg.norm(residual, axis=-1) < 0.065
    if np.count_nonzero(inliers) < 3:
        raise ValueError(f"{dataset.dataset_id}: too few consistent centers for the local phase warp")
    interpolation = RBFInterpolator(centers[inliers], residual[inliers], kernel="thin_plate_spline")
    return interpolation(points.reshape(-1, 2)).reshape(points.shape), int(np.count_nonzero(inliers))


def audit_scan(archive, scan, row, probe_nm):
    dataset = load_dataset(scan, archive, allowed_ids=(scan,))
    raw, a, pixel, _, _ = dataset
    _, fitted, modes, scale, offset, _, calibration_rmse, fit_hash = load_detector(archive, scan)
    if fitted.shape[:2] != raw.shape[:2]:
        raise ValueError(f"{scan}: unequal detector and displacement raster shapes")
    alignment, turns, reflection, reciprocal, _, _ = match_archive_orientation(raw, fitted, a, modes, scale, offset)
    if alignment > 0.20:
        raise ValueError(f"{scan}: detector registration error {alignment:.3f}")
    basis = a * np.array(((1, 0.5), (0, np.sqrt(3) / 2)))
    fit = np.asarray(row["fit_scan"])
    yy, xx = np.indices(raw.shape[:2], dtype=float) * pixel
    points = np.stack((xx, yy), axis=-1)
    rigid = points @ fit[:2] + fit[-1]
    grid, shift, solve = relax(np.asarray(row["gradient_crystal"]), basis, ("MoS2", "MoS2", "P"),
                               row["grid_size"], checkpoint_guard)
    if not solve["success"] or solve["force_rms_eV_per_nm"] > 2e-4:
        raise RuntimeError(f"{scan}: the fixed model did not converge")
    sigma_nm = probe_nm / np.sqrt(8 * np.log(2))
    nominal = sample_disregistry(grid, shift, rigid, grid.cell, sigma_nm)
    ideal = apply_spatial(sample_disregistry(grid, shift, rigid, grid.cell), turns, reflection)
    initial = apply_spatial(nominal, turns, reflection)
    exact = intensity(ideal, reciprocal, scale, offset)
    sigma_px = sigma_nm / pixel
    blurred = ndi.gaussian_filter(exact, (sigma_px, sigma_px, 0), mode="nearest")
    inverse = invert_intensity(blurred, initial, reciprocal, scale, offset)
    replay = invert_intensity(exact, initial, reciprocal, scale, offset)
    interior = np.s_[5:-5, 5:-5]

    def residual(field, target):
        return float(np.sqrt(np.mean((intensity(field, reciprocal, scale, offset)[interior] - target[interior])**2)))

    round_trip = np.rot90(inverse, -turns, axes=(0, 1))
    if reflection:
        round_trip = round_trip[:, ::-1]
    triangles, _ = _good_interior_triangle_mask(dataset)
    observed_shear, observed_labels = wall_maps(dataset, raw, triangles)
    observed = np.asarray(sector_means(observed_shear, observed_labels))
    measured_walls = np.isin(observed_labels, (1, 2, 3))
    phase_warp, warp_centers = local_geometry(dataset, points, fit, basis)
    local_rigid = rigid + phase_warp
    locally_warped = sample_disregistry(grid, shift, local_rigid, grid.cell, sigma_nm)
    dy, dx = np.gradient(local_rigid, pixel, axis=(0, 1))
    determinant = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
    fold = (determinant * np.linalg.det(fit[:2])) <= 0
    result = {
        "archive_fit_sha256": fit_hash, "detector_alignment_rmse": alignment,
        "detector_calibration_rmse": calibration_rmse,
        "observed_sector_means_percent": observed.tolist(), "cases": {},
        "zero_blur_replay_intensity_rmse": residual(replay, exact),
        "zero_blur_replay_displacement_rmse_nm": float(np.sqrt(np.mean(np.sum((replay[interior] - ideal[interior])**2, axis=-1)))),
        "blurred_intensity_rmse_before_inversion": residual(initial, blurred),
        "blurred_intensity_rmse_after_inversion": residual(inverse, blurred),
        "phase_warp_centers": warp_centers,
        "phase_warp_fold_fraction_in_triangles": float(np.mean(fold[triangles])),
    }
    shear_maps = {"measured": observed_shear}
    fields = {"blur_displacement": nominal, "blur_intensity_then_invert": round_trip,
              "local_geometry_warp": locally_warped}
    for name, field in fields.items():
        shear, labels = wall_maps(dataset, field, triangles)
        shear_maps[name] = shear
        own_means = sector_means(shear, labels)
        observed_pixel_means = sector_means(shear, observed_labels)
        result["cases"][name] = {
            "sector_means_percent": own_means,
            "absolute_mean_bias_percentage_points": float(np.mean(own_means - observed)),
            "means_at_measured_wall_pixels_percent": observed_pixel_means,
            "bias_at_measured_wall_pixels_percentage_points": float(np.mean(observed_pixel_means - observed)),
            "same_label_fraction_at_measured_wall_pixels": float(np.mean(
                labels[measured_walls] == observed_labels[measured_walls])),
        }
    result["wall_cross_sections"] = wall_profiles(dataset, shear_maps)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    inputs = [project / name for name in ("registry_development_v01.json", "registry_held_out_v01.json")]
    geometry = {scan: row for path in inputs for scan, row in json.loads(path.read_text())["scans"].items()}
    probe_nm = json.loads((project / "registry_protocol_v01.json").read_text())["probe_fwhm_nm"]
    start = perf_counter()
    result = {
        "probe_fwhm_nm": probe_nm, "scans": {},
        "description": "The synthetic detector intensities are blurred and inverted without changing the physical model. The experimental wall labels and cross-sections diagnose spatial registration and wall shape. The local geometry control warps the fixed relaxation using only stacking centers and is not a new equilibrium calculation.",
        "sha256": {path.name: sha256(path.read_bytes()).hexdigest()
                   for path in (*inputs, Path(__file__), project / "bilayer_relaxation.py",
                                project / "observe_bilayers.py", project / "run_detector_prediction.py")},
    }
    for scan in SCANS:
        checkpoint_guard()
        row = audit_scan(args.archive, scan, geometry[scan], probe_nm)
        result["scans"][scan] = row
        result["wall_seconds"] = perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, result)
        print(scan, row["cases"], flush=True)


if __name__ == "__main__":
    main()
