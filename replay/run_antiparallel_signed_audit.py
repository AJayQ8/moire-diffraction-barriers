"""Replay the antiparallel MoS2 validation with the released signed readout."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter

from author_detector_replay import detector_intensity, load_calibration, rms
from author_release import RELEASE_COMMIT, load_author_release
from bilayer_relaxation import ConvergenceError, sample_disregistry
from observe_bilayers import _good_interior_triangle_mask, load_dataset
from run_detector_prediction import apply_spatial
from run_heterobilayer_diffraction_check import register_frame
from run_heterobilayer_residual_transfer import rms_vector
from run_heterobilayer_sparse_geometry import linear_response, solve_from_geometry
from run_heterobilayer_transfer import PROJECT, blur, digest, geometry_from_centers
from shared_compute import atomic_json, checkpoint_guard


PROTOCOL = PROJECT / "antiparallel_signed_protocol_v01.json"
SETTINGS = PROJECT / "registry_protocol_v01.json"
INTERFACE = ("MoS2", "MoS2", "AP")


def local_geometry(data, points, basis, fit, cutoff):
    iy, ix = np.rint(data.centers_rc_px).astype(int).T
    observed = data.u_nm[iy, ix]
    centers = data.centers_rc_px[:, ::-1] * data.pixel_nm
    finite = np.isfinite(observed).all(axis=-1)
    centers = centers[finite]
    observed = observed[finite]
    nominal = centers @ fit[:2] + fit[-1]
    translations = np.rint(observed @ np.linalg.inv(basis).T) @ basis.T
    residual = translations - nominal
    inliers = np.linalg.norm(residual, axis=-1) < cutoff
    if inliers.sum() < 5:
        raise ValueError("The affine geometry left fewer than five local centers")
    points_inside = centers[inliers]
    offset = residual[inliers]
    replay = np.empty_like(offset)
    for i in range(len(offset)):
        train = np.arange(len(offset)) != i
        interpolant = RBFInterpolator(points_inside[train], offset[train], kernel="thin_plate_spline")
        replay[i] = interpolant(points_inside[i:i+1])[0] - offset[i]
    field = RBFInterpolator(points_inside, offset, kernel="thin_plate_spline")(points.reshape(-1, 2)).reshape(points.shape)
    return field, {
        "used_centers": int(inliers.sum()), "affine_rmse_nm": rms_vector(offset),
        "leave_one_center_out_rmse_nm": rms_vector(replay), "warp_rms_nm": rms_vector(field.reshape(-1, 2)),
    }


def score_scan(data, calibration, settings):
    crop, turns, reflect, transform, alignment = register_frame(
        data, calibration.archived_fitted_intensity, calibration.modes, calibration.coefficients)
    if alignment["archived_field_registration_rmse"] > 0.20:
        raise ValueError("The author-calibrated diffraction and processed fields do not align")
    measured = calibration.observed[crop]
    fitted = calibration.archived_fitted_intensity[crop]
    reciprocal = calibration.modes @ transform / data.lattice_nm
    coefficients = calibration.coefficients
    basis = data.lattice_nm * np.array(((1.0, 0.5), (0.0, np.sqrt(3) / 2)))
    origin = basis @ np.full(2, 2 / 3)
    shape = data.u_nm.shape[:2]
    yy, xx = np.indices(shape, dtype=float)
    points = np.stack((xx, yy), axis=-1) * data.pixel_nm
    nominal, gradient, cell, geometry = geometry_from_centers(data, data.centers_rc_px, basis, points,
                                                              settings["center_inlier_nm"])
    correction, local = local_geometry(data, points, basis, np.asarray(geometry["fit_scan"]),
                                       settings["center_inlier_nm"])
    warped = nominal + correction
    valid = data.source_valid_mask
    sigma = settings["smoothing_sigma_pixels"]
    probe = settings["probe_fwhm_nm"] / np.sqrt(8 * np.log(2))
    probe_pixels = probe / data.pixel_nm
    weights = gaussian_filter(valid.astype(float), sigma, mode="constant")
    inside, _ = _good_interior_triangle_mask(data)
    distance2 = np.full(shape, np.inf)
    for y, x in data.centers_rc_px:
        distance2 = np.minimum(distance2, (yy - y)**2 + (xx - x)**2)
    support = inside & valid & (weights > 0.97) & (distance2 > (2 * sigma)**2)
    support[:4] = support[-4:] = support[:, :4] = support[:, -4:] = False
    if support.sum() < 300:
        raise ValueError("The common interior mask has fewer than 300 pixels")
    detector_support = apply_spatial(support, turns, reflect)
    h, k = calibration.modes.T
    ring_index = np.rint(h*h + h*k + k*k).astype(int)
    rings = {"both": np.ones(len(h), dtype=bool), "first": ring_index == 1, "second": ring_index == 3}
    if not all(channels.sum() >= 5 for channels in rings.values()):
        raise ValueError("A diffraction ring has fewer than five measured disks")
    detector_reference = {name: rms(measured[detector_support][:, channels], fitted[detector_support][:, channels])
                          for name, channels in rings.items()}
    models = {}

    def compare(vector, unblurred):
        moved = apply_spatial(unblurred, turns, reflect)
        prediction = detector_intensity(moved, reciprocal, coefficients)
        prediction = gaussian_filter(prediction, (probe_pixels, probe_pixels, 0), mode="nearest")
        delta = blur(vector - data.u_nm, valid, weights, sigma)[support]
        return {"layer_shift_rmse_nm": rms_vector(delta),
                "offset_free_layer_shift_rmse_nm": rms_vector(delta - delta.mean(axis=0)),
                "diffraction_rmse": {name: rms(measured[detector_support][:, channels],
                                               prediction[detector_support][:, channels])
                                     for name, channels in rings.items()}}

    models["affine_rigid"] = compare(nominal, nominal)
    models["warped_rigid"] = compare(warped, warped)
    solvers = {}
    for name, sign in (("original", -1), ("reversed", 1)):
        checkpoint_guard()
        grid, full, receipt = solve_from_geometry(sign * gradient, basis, INTERFACE)
        if not np.allclose(grid.cell, sign * cell):
            raise ValueError("AP registry sign changed the cell magnitude")
        solvers[name] = receipt
        linear = linear_response(grid, INTERFACE)
        for geometry_name, rigid in (("affine", nominal), ("warped", warped)):
            phase = origin + sign * rigid
            for physical_name, field in (("relaxed", full), ("linear", linear)):
                ideal = sign * (sample_disregistry(grid, field, phase, grid.cell) - origin)
                vector = sign * (sample_disregistry(grid, field, phase, grid.cell, probe) - origin)
                models[f"{geometry_name}_{physical_name}_{name}"] = compare(vector, ideal)
    dy, dx = np.gradient(warped, data.pixel_nm, axis=(0, 1))
    determinant = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
    return {
        "archive_md5": data.archive_md5, "pixels": int(support.sum()), "detector_alignment": alignment,
        "detector_calibration": calibration.diagnostics, "detector_archive_fit_rmse": detector_reference,
        "affine_geometry": geometry, "local_geometry": local, "solvers": solvers, "models": models,
        "warped_minimum_determinant_ratio": float(np.min(determinant[support] / np.linalg.det(np.asarray(geometry["fit_scan"])[:2]))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    author = load_author_release(args.release, args.dependencies)
    protocol = json.loads(PROTOCOL.read_text())
    settings = json.loads(SETTINGS.read_text())
    result = {
        "protocol": protocol, "author_commit": RELEASE_COMMIT,
        "input_sha256": {p.name: digest(p) for p in (Path(__file__), PROTOCOL, SETTINGS,
                                                   PROJECT / "author_detector_replay.py")},
        "scans": {}, "failed_scans": {},
    }
    for scan in protocol["development"] + protocol["replay"]:
        checkpoint_guard()
        try:
            data = load_dataset(scan, args.archive, allowed_ids=(scan,))
            if data.orientation != "ap" or data.metadata["heterobilayer_correction_applied"]:
                raise ValueError("The scan is outside the antiparallel homobilayer study")
            calibration = load_calibration(args.archive, scan, author)
            if calibration.diagnostics["native_replay_rmse"] >= calibration.diagnostics["native_even_misread_rmse"]:
                raise ValueError("The archived signed response did not improve the fit replay")
            result["scans"][scan] = score_scan(data, calibration, settings)
        except (ConvergenceError, ValueError, np.linalg.LinAlgError) as exc:
            result["failed_scans"][scan] = str(exc)
            print(scan, str(exc), flush=True)
        result["wall_seconds"] = perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
