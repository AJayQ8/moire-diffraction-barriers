"""Use first-ring stacking centers to predict diffraction in the second ring."""

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
from bilayer_relaxation import sample_disregistry
from observe_bilayers import _inside_polygon_rc, load_dataset
from run_detector_prediction import AUTHOR_BASIS
from run_heterobilayer_corrugation import BASIS, MODEL_A, ORIGIN
from run_heterobilayer_diffraction_check import read_native_detector
from run_heterobilayer_residual_transfer import rms_vector
from run_heterobilayer_ring_geometry import mean_rows
from run_heterobilayer_sparse_geometry import linear_response, solve_from_geometry
from run_heterobilayer_transfer import PROJECT, PROTOCOL, digest
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


METHOD = PROJECT / "heterobilayer_cross_ring_protocol_v01.json"
SETTINGS = PROJECT / "registry_protocol_v01.json"
FRAME = PROJECT / "heterobilayer_diffraction_v02.json"
SPARSE = PROJECT / "heterobilayer_sparse_geometry_additional_v01.json"
SEARCH_RADIUS_CELL_FRACTION = 0.41


def predict_zero_distance(observed, coefficients):
    """Score the local first-ring intensities against the XMMX stacking point."""
    zero = coefficients[:, 0] + coefficients[:, 2]
    scale = np.hypot(coefficients[:, 0], coefficients[:, 1])
    normalized = (observed - zero) / scale
    cost = np.mean(normalized**2, axis=-1)
    return gaussian_filter(cost, 1.0, mode="nearest")


def refine_minimum(cost, row, col):
    """Estimate a subpixel center from the nearest 3 by 3 cost samples."""
    neighborhood = cost[row-1:row+2, col-1:col+2]
    gy, gx = (neighborhood[2, 1] - neighborhood[0, 1]) / 2, (neighborhood[1, 2] - neighborhood[1, 0]) / 2
    hyy = neighborhood[2, 1] + neighborhood[0, 1] - 2 * neighborhood[1, 1]
    hxx = neighborhood[1, 2] + neighborhood[1, 0] - 2 * neighborhood[1, 1]
    hxy = (neighborhood[2, 2] + neighborhood[0, 0] - neighborhood[2, 0] - neighborhood[0, 2]) / 4
    hessian = np.array(((hyy, hxy), (hxy, hxx)))
    offset = np.zeros(2)
    if np.linalg.eigvalsh(hessian).min() > 0:
        step = np.linalg.solve(hessian, -np.array((gy, gx)))
        if np.max(np.abs(step)) < 0.8:
            offset = step
    return np.array((row, col)) + offset


def find_centers(measured, coefficients, basis, fit, pixel_nm):
    """Use affine lattice indices as search windows for a first-ring-only readout."""
    cost = predict_zero_distance(measured, coefficients)
    corners = pixel_nm * np.array(((0, 0), (0, cost.shape[0] - 1),
                                   (cost.shape[1] - 1, 0), (cost.shape[1] - 1, cost.shape[0] - 1)))
    corners_fractional = (corners @ fit[:2] + fit[-1]) @ np.linalg.inv(basis).T
    lower = np.floor(corners_fractional.min(axis=0)).astype(int)
    upper = np.ceil(corners_fractional.max(axis=0)).astype(int)
    indices = np.stack(np.meshgrid(*(np.arange(a, b+1) for a, b in zip(lower, upper)), indexing="ij"), axis=-1).reshape(-1, 2)
    points = (indices @ basis.T - fit[-1]) @ np.linalg.inv(fit[:2]) / pixel_nm
    lattice_xy = np.linalg.solve(fit[:2].T, basis) / pixel_nm
    shortest = min(np.linalg.norm(lattice_xy[:, 0]), np.linalg.norm(lattice_xy[:, 1]),
                   np.linalg.norm(lattice_xy[:, 0] - lattice_xy[:, 1]))
    radius = shortest * SEARCH_RADIUS_CELL_FRACTION
    rr, cc = np.indices(cost.shape)
    centers, accepted = [], []
    for index, (x, y) in zip(indices, points):
        if not (2 <= y < cost.shape[0]-2 and 2 <= x < cost.shape[1]-2):
            continue
        search = ((rr-y)**2 + (cc-x)**2 < radius**2) & (rr >= 2) & (cc >= 2)
        search &= (rr < cost.shape[0]-2) & (cc < cost.shape[1]-2)
        row, col = np.unravel_index(np.argmin(np.where(search, cost, np.inf)), cost.shape)
        position = refine_minimum(cost, row, col)
        if np.linalg.norm(position - (y, x)) < radius - 0.5:
            centers.append(position)
            accepted.append(index)
    if len(centers) < 18:
        raise ValueError("First-ring detector found too few centers")
    return np.asarray(centers), np.asarray(accepted)


def score(data, measured, modes, coefficients, frame, fit, settings, interface):
    if frame["spatial_turns"] != 0 or frame["spatial_reflection"]:
        raise ValueError("The cross-ring protocol uses an unrotated scan raster")
    mirror = np.diag((-1, 1)) if frame["vector_reflection"] else np.eye(2)
    transform = np.linalg.inv(AUTHOR_BASIS) @ rotation(frame["vector_rotation_60deg_steps"] * np.pi/3) @ mirror
    reciprocal = modes @ transform / data.lattice_nm
    row, col = data.metadata.get("author_unwrap_crop_origin_row_col", (0, 0))
    height, width = data.u_nm.shape[:2]
    measured = measured[row:row+height, col:col+width]
    scale = MODEL_A / data.lattice_nm
    basis = BASIS / scale
    h, k = modes.T
    index = np.rint(h*h + h*k + k*k).astype(int)
    first, second = index == 1, index == 3
    if first.sum() != 6 or second.sum() != 6:
        raise ValueError("The comparison requires six measured intensities in each ring")
    centers, labels = find_centers(measured[..., first], coefficients[first], basis, fit, data.pixel_nm)
    translations = labels @ basis.T
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
    for group in range(3):
        checkpoint_guard()
        train = groups != group
        inplane = np.column_stack((center_points[train], np.ones(train.sum())))
        affine = np.linalg.lstsq(inplane, translations[train], rcond=None)[0]
        nominal = points @ affine[:2] + affine[-1]
        delta = translations - (center_points @ affine[:2] + affine[-1])
        local = RBFInterpolator(center_points[train], delta[train], kernel="thin_plate_spline")
        warped = nominal + local(points.reshape(-1, 2)).reshape(points.shape)
        hull = ConvexHull(centers[train])
        keep = support & _inside_polygon_rc((height, width), centers[train][hull.vertices])
        if keep.sum() < 300:
            raise ValueError("Insufficient training-hull support after excluding centers")
        testing = ~train & (np.max(centers @ hull.equations[:, :2].T + hull.equations[:, -1], axis=1) <= 0)
        test_delta = delta[testing]
        if not len(test_delta):
            raise ValueError("No unused first-ring centers lie inside the training support")
        gradient = affine[:2].T @ rotation(-angle)
        grid, shift, info = solve_from_geometry(scale * gradient, BASIS, interface)
        linear = linear_response(grid, interface)
        outputs = {"affine_rigid": nominal, "warped_rigid": warped}
        for name, rigid in (("affine", nominal), ("warped", warped)):
            phase = ORIGIN + scale * rigid
            outputs[f"{name}_relaxed"] = (sample_disregistry(grid, shift, phase, grid.cell) - ORIGIN) / scale
        phase = ORIGIN + scale * warped
        outputs["warped_linear"] = (sample_disregistry(grid, linear, phase, grid.cell) - ORIGIN) / scale
        scores = {}
        for name, vector in outputs.items():
            predicted = detector_intensity(vector, reciprocal, coefficients)
            predicted = gaussian_filter(predicted, (probe, probe, 0), mode="nearest")
            scores[name] = {key: rms(measured[keep][:, channels], predicted[keep][:, channels])
                            for key, channels in (("first", first), ("second", second))}
        dy, dx = np.gradient(warped, data.pixel_nm, axis=(0, 1))
        jacobian = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
        folds.append({
            "withheld_group": group, "train_centers": int(train.sum()), "test_centers": int(testing.sum()),
            "withheld_center_rmse_nm": {"affine": rms_vector(test_delta),
                                        "warped": rms_vector(test_delta - local(center_points[testing]))},
            "pixels": int(keep.sum()), "diffraction_rmse": scores, "solver": info,
            "minimum_local_jacobian_ratio": float(np.min(jacobian[keep] / np.linalg.det(affine[:2]))),
        })
    return {"first_ring_centers_rc_px": centers.tolist(), "lattice_indices": labels.tolist(), "folds": folds,
            "mean_over_folds": {key: mean_rows([fold[key] for fold in folds]) for key in (
                "withheld_center_rmse_nm", "diffraction_rmse")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch", choices=("qualification", "replay"), required=True)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    author = load_author_release(args.release, args.dependencies)
    old = json.loads(PROTOCOL.read_text())
    method = json.loads(METHOD.read_text())
    settings = json.loads(SETTINGS.read_text())
    paths = [PROJECT / f"heterobilayer_transfer_{batch}_v01.json" for batch in old["devices"]]
    seed_geometry = {d: v for p in paths for d, v in json.loads(p.read_text())["scans"].items()}
    frames = json.loads(FRAME.read_text())["scans"]
    additional = json.loads(SPARSE.read_text())["scans"]
    hashes = {p.name: digest(p) for p in (Path(__file__), METHOD, SETTINGS, FRAME, SPARSE, *paths)}
    if args.batch == "replay":
        if args.freeze is None:
            parser.error("The second-device replay needs the qualification freeze")
        receipt = json.loads(args.freeze.read_text())
        qualified = args.freeze.parent / receipt["qualification_file"]
        if receipt["input_sha256"] != hashes or digest(qualified) != receipt["qualification_sha256"]:
            raise ValueError("The cross-ring analysis changed after the freeze")
    cases = method[args.batch]
    output = {"batch": args.batch, "scope": method, "author_commit": RELEASE_COMMIT, "input_sha256": hashes,
              "freeze_sha256": digest(args.freeze) if args.freeze else None, "scans": {}}
    for scan in cases:
        checkpoint_guard()
        data = load_dataset(scan, args.archive, allowed_ids=cases, allow_author_crop=True)
        if scan in seed_geometry:
            prior = seed_geometry[scan]
            frame = frames[scan]["alignment"]
            fit = np.asarray(prior["whole_geometry"]["fit_scan"])
            cap = prior["encapsulation"]
        else:
            prior = additional[scan]
            frame = prior["detector_frame"]
            # The additional scans have no preexisting affine fit. Infer a
            # single global lattice from their previously scored registry grid.
            centers = data.centers_rc_px[:, ::-1] * data.pixel_nm
            sampled = data.u_nm[tuple(np.rint(data.centers_rc_px).astype(int).T)]
            basis = BASIS * data.lattice_nm / MODEL_A
            finite = np.isfinite(sampled).all(axis=-1)
            labels = np.rint(sampled[finite] @ np.linalg.inv(basis).T) @ basis.T
            fit = np.linalg.lstsq(np.column_stack((centers[finite], np.ones(finite.sum()))), labels, rcond=None)[0]
            cap = "unverified"
        if data.archive_md5 != prior["archive_md5"]:
            raise ValueError("The supplied scan differs from the prior geometry archive")
        observed, _, modes, coefficients, calibration = read_native_detector(args.archive, scan, author)
        result = score(data, observed, modes, coefficients, frame, fit, settings, tuple(old["interface"]))
        output["scans"][scan] = {"encapsulation": cap, "archive_md5": data.archive_md5,
                                 "calibration": calibration, **result}
        print(scan, result["mean_over_folds"]["diffraction_rmse"], flush=True)
        output["wall_seconds"] = perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, output)


if __name__ == "__main__":
    main()
