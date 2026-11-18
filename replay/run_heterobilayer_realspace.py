"""Compare transported and re-equilibrated relaxation on a nonuniform moiré."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.fft import dstn, idstn
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

from author_detector_replay import detector_intensity, rms
from author_release import load_author_release
from bilayer_relaxation import STACKING, elastic_parameters_per_reference_cell, sample_disregistry, stack_energy
from observe_bilayers import load_dataset
from run_detector_prediction import AUTHOR_BASIS
from run_heterobilayer_corrugation import BASIS, MODEL_A, ORIGIN
from run_heterobilayer_diffraction_check import read_native_detector
from run_heterobilayer_ring_geometry import mean_rows
from run_heterobilayer_sparse_geometry import linear_response, solve_from_geometry
from run_heterobilayer_transfer import PROJECT, PROTOCOL, digest
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


METHOD = PROJECT / "heterobilayer_realspace_protocol_v01.json"
SETTINGS = PROJECT / "registry_protocol_v01.json"
MARGIN = 10


def forward_gradient(u, step, angle):
    """Difference on each cell, in the crystal frame used by the elastic model."""
    dx = (u[:-1, 1:] - u[:-1, :-1]) / step
    dy = (u[1:, :-1] - u[:-1, :-1]) / step
    return np.stack((dx, dy), axis=-1) @ rotation(-angle)


def gradient_adjoint(stress, step, angle, shape):
    raster = stress @ rotation(angle)
    out = np.zeros(shape)
    out[:-1, :-1] = -raster.sum(axis=-1) / step
    out[:-1, 1:] += raster[..., 0] / step
    out[1:, :-1] += raster[..., 1] / step
    return out


def elastic_energy(u, step, angle, bulk, shear):
    gradient = forward_gradient(u, step, angle)
    divergence = np.trace(gradient, axis1=-2, axis2=-1)
    energy = 0.5 * (bulk * np.sum(divergence**2) + shear * np.sum(gradient**2))
    stress = bulk * divergence[..., None, None] * np.eye(2) + shear * gradient
    return energy, gradient_adjoint(stress, step, angle, u.shape)


def realspace_equilibrium(reference, step, angle, interface, initial=None, linear=False, check_derivative=False):
    """Solve with fixed boundary disregistry and no material parameter fit."""
    coeff = STACKING[interface]
    bulk, shear = elastic_parameters_per_reference_cell(*interface[:2])
    stack, force_at_rigid = stack_energy(reference, BASIS, coeff)
    rigid_energy = float(stack.mean())
    shape = reference.shape
    interior = np.s_[1:-1, 1:-1, :]
    ny, nx = shape[0] - 2, shape[1] - 2
    row = 2 * np.sin(np.pi * np.arange(1, ny + 1) / (2 * (ny + 1))) / step
    col = 2 * np.sin(np.pi * np.arange(1, nx + 1) / (2 * (nx + 1))) / step
    stiffness = (shear + bulk / 2) * (row[:, None]**2 + col[None, :]**2)
    inverse = 1 / np.sqrt(stiffness + 10)

    def transform(y, factor):
        y = np.asarray(y).reshape((ny, nx, 2))
        return idstn(factor[..., None] * y, axes=(0, 1), type=1, norm="ortho")

    def physical(u):
        elastic, grad = elastic_energy(u, step, angle, bulk, shear)
        if linear:
            stack = rigid_energy + np.mean(np.sum(force_at_rigid * u, axis=-1))
            force = force_at_rigid
        else:
            stack, force = stack_energy(reference + u, BASIS, coeff)
        return elastic / np.prod(shape[:2]) + np.mean(stack), grad + force

    def objective(y):
        u = np.zeros(shape)
        u[interior] = transform(y, inverse)
        energy, force = physical(u)
        derivative = inverse[..., None] * dstn(force[interior], axes=(0, 1), type=1, norm="ortho")
        return energy, derivative.ravel() / np.prod(shape[:2])

    start = np.zeros((ny, nx, 2))
    if initial is not None:
        taper = distance_transform_edt(np.pad(np.ones((ny, nx), dtype=bool), 1))[1:-1, 1:-1]
        weighted = initial[interior] * np.minimum(1, taper[..., None] * step / 3)
        start = dstn(weighted, axes=(0, 1), type=1, norm="ortho") / inverse[..., None]
    start = start.ravel()
    derivative_error = None
    if check_derivative:
        direction = np.random.default_rng(5100).normal(size=start.shape)
        direction /= np.linalg.norm(direction)
        analytic = objective(start)[1] @ direction
        finite_difference = (objective(start + 1e-4 * direction)[0] - objective(start - 1e-4 * direction)[0]) / 2e-4
        derivative_error = float(abs(analytic - finite_difference))
        if derivative_error > 1e-8:
            raise ValueError("The real-space force failed the directional derivative check")
    checkpoint_guard()
    iteration = 0

    def callback(_):
        nonlocal iteration
        iteration += 1
        if iteration % 20 == 0:
            checkpoint_guard()

    fit = minimize(objective, start, jac=True, method="L-BFGS-B", callback=callback,
                   options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 350, "maxcor": 8})
    solution = np.zeros(shape)
    solution[interior] = transform(fit.x, inverse)
    energy, force = physical(solution)
    rms_force = float(np.sqrt(np.mean(force[interior]**2)))
    if not fit.success or rms_force > 2e-4:
        raise ValueError(f"The real-space solve failed the force check ({rms_force:.3g} eV/nm)")
    return solution, {"energy_eV_per_cell": float(energy), "rigid_energy_eV_per_cell": rigid_energy,
                      "iterations": int(fit.nit), "rms_force_eV_per_nm": rms_force,
                      "directional_derivative_error": derivative_error}


def block_predictions(data, row, settings, interface, measured, modes, coefficients, frame, fine):
    if frame["spatial_turns"] != 0 or frame["spatial_reflection"]:
        raise ValueError("Expected the unrotated diffraction raster")
    reflection = np.diag((-1, 1)) if frame["vector_reflection"] else np.eye(2)
    transform = np.linalg.inv(AUTHOR_BASIS) @ rotation(frame["vector_rotation_60deg_steps"] * np.pi / 3) @ reflection
    reciprocal = modes @ transform / data.lattice_nm
    y0, x0 = data.metadata.get("author_unwrap_crop_origin_row_col", (0, 0))
    height, width = data.u_nm.shape[:2]
    measured = measured[y0:y0+height, x0:x0+width]
    h, k = modes.T
    second = np.rint(h*h + h*k + k*k).astype(int) == 3
    measured = measured[..., second]
    reciprocal, coefficients = reciprocal[second], coefficients[second]
    scale = MODEL_A / data.lattice_nm
    basis = BASIS / scale
    angle = np.deg2rad(data.sample_rotation_degrees)
    step = data.pixel_nm
    yy, xx = np.indices((height, width), dtype=float)
    points = np.stack((xx, yy), axis=-1) * step
    centers = np.asarray(row["first_ring_centers_rc_px"])
    labels = np.asarray(row["lattice_indices"])
    translations = labels @ basis.T
    groups = (labels[:, 0] - labels[:, 1]) % 3
    center_points = centers[:, ::-1] * step
    distance = np.min((yy[..., None] - centers[:, 0])**2 + (xx[..., None] - centers[:, 1])**2, axis=-1)
    support = distance > (2 * settings["smoothing_sigma_pixels"])**2
    support[:MARGIN] = support[-MARGIN:] = support[:, :MARGIN] = support[:, -MARGIN:] = False
    probe_nm = settings["probe_fwhm_nm"] / np.sqrt(8 * np.log(2))
    pixel_probe = probe_nm / step
    folds = []
    for group in range(3):
        checkpoint_guard()
        train = groups != group
        t = center_points[train]
        design = np.column_stack((t, np.ones(len(t))))
        fit = np.linalg.lstsq(design, translations[train], rcond=None)[0]
        residuals = translations[train] - design @ fit
        warp = RBFInterpolator(t, residuals, kernel="thin_plate_spline")
        affine = points @ fit[:2] + fit[-1]
        local = affine + warp(points.reshape(-1, 2)).reshape(points.shape)
        local_model = ORIGIN + scale * local
        affine_model = ORIGIN + scale * affine
        hull = ConvexHull(centers[train])
        keep = support & np.all(np.stack((yy, xx), axis=-1) @ hull.equations[:, :2].T + hull.equations[:, -1] <= 0, axis=-1)
        if keep.sum() < 300:
            raise ValueError("Real-space score has too little common support")
        gradient = scale * fit[:2].T @ rotation(-angle)
        grid, shift, _ = solve_from_geometry(gradient, BASIS, interface)
        transported = sample_disregistry(grid, shift, local_model, grid.cell) - local_model
        lin = linear_response(grid, interface)
        real, info = realspace_equilibrium(local_model, step, angle, interface, transported, check_derivative=(group == 0))
        lin_real, lin_info = realspace_equilibrium(local_model, step, angle, interface, linear=True)
        predictions = {
            "affine_relaxed": (sample_disregistry(grid, shift, affine_model, grid.cell) - ORIGIN) / scale,
            "local_rigid": local,
            "transported_periodic": local + transported / scale,
            "transported_linear": (sample_disregistry(grid, lin, local_model, grid.cell) - ORIGIN) / scale,
            "realspace_equilibrium": local + real / scale,
            "realspace_linear": local + lin_real / scale,
        }
        resolution = None
        if fine and group == 0:
            fy, fx = np.mgrid[0:height-0.5:0.5, 0:width-0.5:0.5]
            fpoints = np.stack((fx, fy), axis=-1) * step
            flocal = fpoints @ fit[:2] + fit[-1] + warp(fpoints.reshape(-1, 2)).reshape(fpoints.shape)
            fref = ORIGIN + scale * flocal
            guess = sample_disregistry(grid, shift, fref, grid.cell) - fref
            refined, _ = realspace_equilibrium(fref, step / 2, angle, interface, guess)
            compare = refined[::2, ::2] - real
            resolution = {"layer_shift_change_rms_nm": float(np.sqrt(np.mean(np.sum(compare[keep]**2, axis=-1))) / scale)}
            predictions["realspace_equilibrium_refined"] = local + refined[::2, ::2] / scale
        scores = {}
        for name, field in predictions.items():
            signal = detector_intensity(field, reciprocal, coefficients)
            image = gaussian_filter(signal, (pixel_probe, pixel_probe, 0), mode="nearest")
            scores[name] = rms(measured[keep], image[keep])
        if resolution is not None:
            resolution["refined_second_ring_rmse"] = scores.pop("realspace_equilibrium_refined")
            resolution["absolute_change_in_second_ring_rmse"] = abs(
                resolution["refined_second_ring_rmse"] - scores["realspace_equilibrium"])
        folds.append({"withheld_group": group, "pixel_count": int(keep.sum()), "second_ring_rmse": scores,
                      "equilibrium": info, "linear": lin_info, "half_step_check": resolution})
    return {"folds": folds, "mean_second_ring_rmse": mean_rows([f["second_ring_rmse"] for f in folds])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    start = perf_counter()
    author = load_author_release(args.release, args.dependencies)
    files = [PROJECT / f"heterobilayer_cross_ring_{batch}_v01.json" for batch in ("qualification", "replay")]
    scans = {scan: row for path in files for scan, row in json.loads(path.read_text())["scans"].items()}
    frames = json.loads((PROJECT / "heterobilayer_diffraction_v02.json").read_text())["scans"]
    additional = json.loads((PROJECT / "heterobilayer_sparse_geometry_additional_v01.json").read_text())["scans"]
    settings = json.loads(SETTINGS.read_text())
    interface = tuple(json.loads(PROTOCOL.read_text())["interface"])
    receipt = {"scope": json.loads(METHOD.read_text()), "input_sha256": {
        p.name: digest(p) for p in (Path(__file__), METHOD, SETTINGS, *files)}, "scans": {}}
    cases = ("DS31",) if args.pilot else tuple(scans)
    for scan in cases:
        checkpoint_guard()
        row = scans[scan]
        frame = frames[scan]["alignment"] if scan in frames else additional[scan]["detector_frame"]
        data = load_dataset(scan, args.archive, allowed_ids=cases, allow_author_crop=True)
        if data.archive_md5 != row["archive_md5"]:
            raise ValueError(f"{scan}: the source archive changed")
        measured, _, modes, coefficients, calibration = read_native_detector(args.archive, scan, author)
        if calibration != row["calibration"]:
            raise ValueError(f"{scan}: the detector calibration changed")
        receipt["scans"][scan] = {"encapsulation": row["encapsulation"], **block_predictions(
            data, row, settings, interface, measured, modes, coefficients, frame, scan == "DS31")}
        receipt["wall_seconds"] = perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, receipt)


if __name__ == "__main__":
    main()
