"""Predict on experimental maps using a sparse subset of stacking centers."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull

from author_detector_replay import detector_intensity, rms
from author_release import RELEASE_COMMIT, load_author_release
from bilayer_relaxation import ConvergenceError, STACKING, apply_symbol, sample_disregistry, stack_energy
from observe_bilayers import _good_interior_triangle_mask, _inside_polygon_rc, load_dataset
from run_detector_prediction import apply_spatial
from run_heterobilayer_corrugation import BASIS, MODEL_A, ORIGIN
from run_heterobilayer_diffraction_check import read_native_detector, register_frame
from run_heterobilayer_residual_transfer import rms_vector
from run_heterobilayer_ring_geometry import mean_rows
from run_heterobilayer_transfer import PROJECT, PROTOCOL, blur, digest, fingerprints, geometry_from_centers, solve, verify_freeze
from shared_compute import atomic_json, checkpoint_guard


FREEZE = PROJECT / "heterobilayer_transfer_freeze_v01.json"
METHOD = PROJECT / "heterobilayer_sparse_geometry_protocol_v01.json"
SETTINGS = PROJECT / "registry_protocol_v01.json"
MODELS = ("affine_rigid", "affine_relaxed", "warped_rigid", "warped_linear", "warped_relaxed")


def solve_from_geometry(gradient, basis, interface):
    cell = np.linalg.solve(gradient, basis)
    n = max(64, 2 ** int(np.ceil(np.log2(np.linalg.norm(cell, axis=0).max() / 0.3))))
    if n > 512:
        raise ValueError(f"Local geometry would require a {n} by {n} grid")
    return solve(gradient, basis, interface, n)


def linear_response(grid, interface):
    """Freeze the adhesion force at the rigid registry, as in the earlier control."""
    _, force = stack_energy(grid.fractional @ grid.basis.T, grid.basis, STACKING[interface])
    stiffness = grid.stiffness.copy()
    stiffness[0, 0] = np.eye(2)
    compliance = np.linalg.inv(stiffness)
    compliance[0, 0] = 0
    return -apply_symbol(compliance, force)


def run_scan(data, settings, interface, detector):
    measured, fitted, modes, coefficients, calibration = detector
    crop, turns, reflect, transform, alignment = register_frame(data, fitted, modes, coefficients)
    measured, fitted = measured[crop], fitted[crop]
    if alignment["archived_field_registration_rmse"] > 0.20:
        raise ValueError("The native and unwrapped diffraction fields do not align")
    reciprocal = modes @ transform / data.lattice_nm
    h, k = modes.T
    ring_index = np.rint(h**2 + h*k + k**2).astype(int)
    rings = {"first": ring_index == 1, "second": ring_index == 3}
    if any(select.sum() != 6 for select in rings.values()):
        raise ValueError("The model comparison needs six disks in each ring")
    rings["both"] = np.ones(len(modes), dtype=bool)
    scale = MODEL_A / data.lattice_nm
    archive_basis = BASIS / scale
    sigma = settings["smoothing_sigma_pixels"]
    probe = settings["probe_fwhm_nm"] / np.sqrt(8 * np.log(2))
    shape = data.u_nm.shape[:2]
    yy, xx = np.indices(shape, dtype=float)
    points = np.stack((xx, yy), axis=-1) * data.pixel_nm
    valid = data.source_valid_mask
    weights = gaussian_filter(valid.astype(float), sigma, mode="constant")
    interior, _ = _good_interior_triangle_mask(data)
    support = interior & valid & (weights > 0.97)
    support[:4] = support[-4:] = support[:, :4] = support[:, -4:] = False
    distance2 = np.full(shape, np.inf)
    for y, x in data.centers_rc_px:
        distance2 = np.minimum(distance2, (yy - y)**2 + (xx - x)**2)
    support &= distance2 > (2 * sigma)**2

    yx = np.rint(data.centers_rc_px).astype(int).T
    center_value = data.u_nm[tuple(yx)]
    center_ok = np.isfinite(center_value).all(axis=-1)
    centers_rc = data.centers_rc_px[center_ok]
    centers_xy = centers_rc[:, ::-1] * data.pixel_nm
    indices = np.rint(center_value[center_ok] @ np.linalg.inv(archive_basis).T).astype(int)
    if len(np.unique(indices, axis=0)) != len(indices):
        raise ValueError("Two reported stacking centers share a lattice translation")
    translations = indices @ archive_basis.T
    groups = (indices[:, 0] - indices[:, 1]) % 3
    saved_signal = detector_intensity(apply_spatial(data.u_nm, turns, reflect), reciprocal, coefficients)

    def forward(field):
        values = detector_intensity(apply_spatial(field, turns, reflect), reciprocal, coefficients)
        width = probe / data.pixel_nm
        return gaussian_filter(values, (width, width, 0), mode="nearest")

    folds = []
    for group in range(3):
        checkpoint_guard()
        training = groups != group
        rigid, gradient, cell, geometry = geometry_from_centers(
            data, centers_rc[training], archive_basis, points, settings["center_inlier_nm"])
        fit = np.asarray(geometry["fit_scan"])
        residuals = translations - (centers_xy @ fit[:2] + fit[-1])
        inliers = training & (np.linalg.norm(residuals, axis=-1) < settings["center_inlier_nm"])
        if inliers.sum() < 8:
            raise ValueError("Too few training centers for a sparse local warp")
        local = RBFInterpolator(centers_xy[inliers], residuals[inliers], kernel="thin_plate_spline")
        warp = local(points.reshape(-1, 2)).reshape(points.shape)
        hull = ConvexHull(centers_rc[inliers])
        polygon = centers_rc[inliers][hull.vertices]
        select = support & _inside_polygon_rc(shape, polygon)
        detector_select = apply_spatial(select, turns, reflect)
        if select.sum() < 300:
            raise ValueError("Too little interior support inside the training hull")
        within_hull = np.max(centers_rc @ hull.equations[:, :2].T + hull.equations[:, -1], axis=-1) <= 1e-9
        test = ~training & within_hull
        if not test.any():
            raise ValueError("No withheld stacking centers lie inside the training hull")
        center_control = residuals[test]
        center_error = center_control - local(centers_xy[test])
        grid, shift, numerical = solve_from_geometry(gradient * scale, BASIS, interface)
        linear = linear_response(grid, interface)

        def predictions(probe_sigma):
            result = {}
            for name, phase in (("affine", rigid), ("warped", rigid + warp)):
                registry = ORIGIN + scale * phase
                result[f"{name}_rigid"] = phase
                result[f"{name}_relaxed"] = (sample_disregistry(grid, shift, registry, cell, probe_sigma) - ORIGIN) / scale
                if name == "warped":
                    result["warped_linear"] = (sample_disregistry(grid, linear, registry, cell, probe_sigma) - ORIGIN) / scale
            return result

        vector = predictions(probe)
        model_signals = predictions(0)
        diffraction = {name: forward(model_signals[name]) for name in MODELS}
        diffraction.update(archived_fit=fitted, archived_unwrapped=saved_signal)
        folds.append({
            "withheld_group": group, "training_centers": int(inliers.sum()), "test_centers": int(test.sum()),
            "test_center_errors_nm": {"affine": rms_vector(center_control), "warped": rms_vector(center_error)},
            "score_pixels": int(select.sum()), "solver": numerical,
            "layer_shift_rmse_nm": {name: rms_vector(blur(vector[name] - data.u_nm, valid, weights, sigma)[select])
                                    for name in MODELS},
            "diffraction_rmse": {ring: {name: rms(measured[detector_select][:, channels],
                                                  field[detector_select][:, channels])
                                        for name, field in diffraction.items()}
                                 for ring, channels in rings.items()},
        })
    return {
        "archive_md5": data.archive_md5, "calibration": calibration, "detector_frame": alignment,
        "folds": folds,
        "mean_over_folds": {key: mean_rows([row[key] for row in folds]) for key in (
            "test_center_errors_nm", "layer_shift_rmse_nm", "diffraction_rmse")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch", choices=("previously_scored", "additional_maps"), required=True)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    author = load_author_release(args.release, args.dependencies)
    protocol = json.loads(PROTOCOL.read_text())
    method = json.loads(METHOD.read_text())
    settings = json.loads(SETTINGS.read_text())
    verify_freeze(FREEZE, fingerprints(), protocol["devices"]["development"])
    hashes = {p.name: digest(p) for p in (Path(__file__), METHOD, SETTINGS, FREEZE,
                                         PROJECT / "run_heterobilayer_ring_geometry.py",
                                         PROJECT / "run_heterobilayer_diffraction_check.py",
                                         PROJECT / "run_heterobilayer_corrugation.py")}
    if args.batch == "additional_maps":
        if args.freeze is None:
            parser.error("The additional maps need the recorded sparse geometry freeze")
        receipt = json.loads(args.freeze.read_text())
        comparison = args.freeze.parent / receipt["previously_scored_file"]
        if receipt["input_sha256"] != hashes or receipt["previously_scored_sha256"] != digest(comparison):
            raise ValueError("The analysis changed after it was frozen for additional maps")
        qualified = json.loads(comparison.read_text())
        if set(qualified["scans"]) != set(method["previously_scored"]) or qualified["failed_scans"]:
            raise ValueError("The replay must complete on the six prior scans")
    cases = method[args.batch]
    result = {"batch": args.batch, "author_commit": RELEASE_COMMIT, "input_sha256": hashes,
              "method": method, "freeze_sha256": digest(args.freeze) if args.freeze else None,
              "scans": {}, "failed_scans": {}}
    for scan in cases:
        checkpoint_guard()
        try:
            data = load_dataset(scan, args.archive, allowed_ids=cases, allow_author_crop=True)
            if not data.metadata["heterobilayer_correction_applied"] or data.orientation != "ap":
                raise ValueError("The input is outside the antiparallel heterobilayer comparison")
            detector = read_native_detector(args.archive, scan, author)
            row = run_scan(data, settings, tuple(protocol["interface"]), detector)
        except (ConvergenceError, ValueError, np.linalg.LinAlgError) as exc:
            result["failed_scans"][scan] = str(exc)
            print(scan, str(exc), flush=True)
        else:
            result["scans"][scan] = row
            print(scan, row["mean_over_folds"]["layer_shift_rmse_nm"], flush=True)
        result["wall_seconds"] = perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
