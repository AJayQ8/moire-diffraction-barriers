"""Compare transported relaxation with constrained real-space elasticity in MoS2."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.fft import dstn, idstn
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

from author_detector_replay import rms
from author_release import load_author_release
from bilayer_relaxation import STACKING, elastic_parameters_per_reference_cell, sample_disregistry, stack_energy
from observe_bilayers import read_input_metadata
from run_heterobilayer_realspace import elastic_energy
from run_heterobilayer_sparse_geometry import solve_from_geometry
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_regime_benchmark import SETTINGS
from run_mos2_camera_spatial_replay import phase_camera
from run_mos2_first_ring_geometry import INTERFACE
from run_mos2_intensity_only_replay import camera_fit, disk_images, geometry
from run_mos2_local_intensity_geometry import SOURCE, geometric_folds
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


LOCAL = PROJECT / "mos2_local_intensity_geometry_v01.json"
PILOT = ("DS2", "DS6", "DS12", "DS13")


def bilinear_constraints(shape, positions):
    """Give the four interpolating nodes and weights for each center."""
    y, x = np.floor(positions).astype(int).T
    dy, dx = (positions - np.column_stack((y, x))).T
    rows = y[:, None] + np.array((0, 0, 1, 1))
    columns = x[:, None] + np.array((0, 1, 0, 1))
    weights = np.column_stack(((1-dy)*(1-dx), (1-dy)*dx, dy*(1-dx), dy*dx))
    if np.any(rows <= 0) or np.any(rows >= shape[0]-1) or np.any(columns <= 0) or np.any(columns >= shape[1]-1):
        raise ValueError("A measured center touches the prescribed outer boundary")
    return rows, columns, weights


def displacement_on_centers(field, constraints):
    rows, columns, weights = constraints
    return np.einsum("ij,ijk->ik", weights, field[rows, columns])


def constrained_equilibrium(affine, warp, initial, centers, targets, basis, step, angle, include_background_stress, guess=None):
    """Solve with the transported boundary and exact measured center positions."""
    checkpoint_guard()
    shape = affine.shape
    ny, nx = shape[0]-2, shape[1]-2
    inside = np.s_[1:-1, 1:-1]
    bulk, shear = elastic_parameters_per_reference_cell(*INTERFACE[:2])
    row = 2*np.sin(np.pi*np.arange(1, ny+1)/(2*(ny+1)))/step
    col = 2*np.sin(np.pi*np.arange(1, nx+1)/(2*(nx+1)))/step
    inverse = 1/np.sqrt((shear+bulk/2)*(row[:, None]**2+col[None, :]**2)+10)
    transform_factor = inverse[..., None]
    constraints = bilinear_constraints(shape, centers)
    iy, ix, weights = constraints
    # In the preconditioned variables, the constraint is one linear
    # functional per center per vector component. Store just its scalar part.
    rows = np.zeros((len(iy), ny, nx))
    rows[np.arange(len(iy))[:, None], iy-1, ix-1] = weights
    functional = (dstn(rows, type=1, axes=(1, 2), norm="ortho")*inverse).reshape(len(rows), -1)
    projection = np.linalg.solve(functional@functional.T, functional)
    mismatch = targets-displacement_on_centers(initial, constraints)
    particular = projection.T@mismatch
    del rows

    def project(values):
        return values-projection.T@(functional@values)

    def decode(values):
        constrained = particular+project(values.reshape(-1, 2))
        field = initial.copy()
        field[inside] += idstn(transform_factor*constrained.reshape(ny, nx, 2), type=1, axes=(0, 1), norm="ortho")
        return field

    def objective(values):
        field = decode(values)
        elastic_reference = affine if include_background_stress else warp
        elastic, derivative = elastic_energy(field-elastic_reference, step, angle, bulk, shear)
        adhesion, force = stack_energy(field, basis, STACKING[INTERFACE])
        constrained_force = transform_factor*dstn((derivative+force)[inside], type=1, axes=(0, 1), norm="ortho")
        return float(elastic/(shape[0]*shape[1])+adhesion.mean()), project(constrained_force.reshape(-1, 2)).ravel()/(shape[0]*shape[1])

    start = np.zeros(2*ny*nx)
    if guess is not None:
        start = (dstn((guess-initial)[inside], type=1, axes=(0, 1), norm="ortho")/transform_factor).ravel()
    direction = np.random.default_rng(7865970).normal(size=start.size)
    direction /= np.linalg.norm(direction)
    initial_energy, derivative = objective(start)
    difference = (objective(start+1e-4*direction)[0]-objective(start-1e-4*direction)[0])/2e-4
    if abs(difference-derivative@direction) > 1e-8:
        raise ValueError("Constrained equilibrium failed the analytic gradient check")
    steps = 0

    def progress(_):
        nonlocal steps
        steps += 1
        if steps % 25 == 0:
            checkpoint_guard()

    opt = minimize(objective, start, jac=True, method="L-BFGS-B", callback=progress,
                   options={"maxiter": 450, "ftol": 1e-14, "gtol": 1e-10, "maxcor": 9})
    final = decode(opt.x)
    center_error = float(np.max(np.linalg.norm(displacement_on_centers(final, constraints)-targets, axis=-1)))
    projected_force = float(np.sqrt(np.mean(opt.jac**2))*shape[0]*shape[1])
    if center_error > 1e-10 or not np.array_equal(final[[0,-1]], initial[[0,-1]]) or not np.array_equal(final[:,[0,-1]], initial[:,[0,-1]]):
        raise ValueError("Constrained solution moved a measured center or prescribed boundary")
    if not opt.success or projected_force > 3e-5 or opt.fun > initial_energy + 1e-12:
        raise ValueError(f"Constrained solver did not converge ({opt.message}, force {projected_force:.2g})")
    return final, {"steps": int(opt.nit), "energy_start_eV_per_cell": initial_energy,
                   "energy_final_eV_per_cell": float(opt.fun), "center_error_nm": center_error,
                   "projected_force_rms_eV_per_nm": projected_force}


def folds_for_scan(images, modes, metadata, settings, recorded):
    context = geometry(images, modes, metadata, settings)
    receipt = context["receipt"]
    if receipt != recorded["geometry"]:
        raise ValueError("The first-ring geometry changed")
    step = receipt["pixel_nm"]
    angle = np.deg2rad(float(metadata["SampleRotation"]))
    yy, xx = np.indices(images.shape[:2])
    xy = np.stack((xx, yy), axis=-1)*step
    centers = np.array(receipt["centers_rc_px"])
    affine_fit = np.array(receipt["first_ring_affine_fit"])
    centers_xy = centers[:, ::-1]*step
    center_design = np.column_stack((centers_xy, np.ones(len(centers))))
    basis = context["basis"]
    labels = np.rint((center_design@affine_fit)@np.linalg.inv(basis).T).astype(int)
    groups = (labels[:, 0]-labels[:, 1])%3
    for group, affine, warp, fit, mask, _ in geometric_folds(centers, xy, step, basis, affine_fit, context["support"]):
        train = groups != group
        targets = labels[train]@basis.T
        gradient = fit[:2].T@rotation(-angle)
        grid, shift, _ = solve_from_geometry(gradient, basis, INTERFACE)
        transported = sample_disregistry(grid, shift, warp, grid.cell)
        yield group, affine, warp, transported, centers[train], targets, mask, context


def score_fields(fields, context, measured, mask):
    second = context["rings"]["second"]
    intensity = measured[..., second]
    bands = np.indices(mask.shape)
    scores = []
    camera = {}
    for model, field in fields.items():
        camera[model] = phase_camera(field, context["reciprocal"][second], context["probe"])
    for name, axis in zip(("rows", "columns"), bands, strict=True):
        cut = np.median(axis[mask])
        buffer = np.ceil(2*context["probe"])
        halves = (mask & (axis < cut-buffer), mask & (axis > cut+buffer))
        if min(v.sum() for v in halves) < 300:
            continue
        for i, (train, test) in enumerate((halves, halves[::-1])):
            row = {}
            for model, (cos, sin) in camera.items():
                predicted, actual, _ = camera_fit(cos, sin, intensity, train, test, allow_odd=True)
                row[model] = rms(predicted, actual)
            scores.append({"axis": name, "calibration_half": i, "pixels": int(test.sum()), "rmse": row})
    return scores


def evaluate(images, modes, metadata, settings, reference, local, pilot, check_grid):
    angle = np.deg2rad(float(metadata["SampleRotation"]))
    results = []
    for group, affine, warp, transported, centers, targets, mask, context in folds_for_scan(
        images, modes, metadata, settings, reference,
    ):
        checkpoint_guard()
        step = context["receipt"]["pixel_nm"]
        outputs = {"transported": transported}
        diagnostics = {}
        resolution = {}
        # Use the first two scoreable folds in the qualification set for the
        # refined mesh. The half-step solve sees the same outer boundary,
        # interpolated from the coarse transported field.
        fine = check_grid and len(results) == 0
        if fine:
            yy, xx = np.mgrid[0:affine.shape[0]-0.5:0.5, 0:affine.shape[1]-0.5:0.5]
            coords = np.stack((yy, xx))

            def refine(field):
                return np.stack([map_coordinates(field[..., k], coords, order=3, mode="nearest")
                                 for k in range(2)], axis=-1)
        for name, stress in (("local_reference", False), ("include_background_stress", True)):
            outputs[name], diagnostics[name] = constrained_equilibrium(
                affine, warp, transported, centers, targets, context["basis"], step, angle, stress)
            diagnostics[name]["rms_change_from_transport_nm"] = float(np.sqrt(np.mean(np.sum(
                (outputs[name][mask]-transported[mask])**2, axis=-1))))
            if fine:
                checkpoint_guard()
                solution, receipt = constrained_equilibrium(
                    refine(affine), refine(warp), refine(transported), 2*centers, targets, context["basis"],
                    step/2, angle, stress, guess=refine(outputs[name]))
                refined = solution[::2, ::2]
                resolution[name] = receipt | {"coarse_to_fine_rms_nm": float(np.sqrt(np.mean(np.sum(
                    (refined[mask]-outputs[name][mask])**2, axis=-1))))}
                outputs[f"{name}_refined"] = refined
        scores = score_fields(outputs, context, images, mask)
        original = next(r for r in local["folds"] if r["withheld_group"] == group)["diffraction"]
        for a, b in zip(scores, original, strict=True):
            if abs(a["rmse"]["transported"]-b["rmse"]["local_full"]) > 1e-12 or a["pixels"] != b["score_pixels"]:
                raise ValueError("The common transported score changed")
        results.append({"withheld_group": group, "scores": scores, "solver": diagnostics, "half_step_check": resolution})
        if pilot:
            break
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--check-grid", action="store_true")
    args = parser.parse_args()
    start = perf_counter()
    source, local, settings = (json.loads(p.read_text()) for p in (SOURCE, LOCAL, SETTINGS))
    for name, fingerprint in {**source["input_sha256"], **local["input_sha256"]}.items():
        if digest(PROJECT/name) != fingerprint:
            raise ValueError(f"The fixed geometry input changed in {name}")
    author = load_author_release(args.release, args.dependencies)
    scans = PILOT if args.pilot else tuple(k for k, v in local["scans"].items() if v["folds"])
    result = {
        "scope": "A controlled real-space equilibrium calculation on the irregular first-ring geometry. Set the raster boundary to the transported periodic solution. Preserve the observed training centers by linear constraints. Compare treating the local geometry as the elastic reference with taking the total deformation relative to the affine lattice. Both calculations use the same published material properties and scoring pixels as the transported solution.",
        "limits": "The measured boundary positions are unavailable. The transported solution sets the outer boundary and the sparse center constraints. The calculation tests these boundary and deformation assumptions, and cannot identify the origin of the observed warp.",
        "input_sha256": {p.name: digest(p) for p in (Path(__file__), SOURCE, LOCAL, SETTINGS)},
        "scans": {}, "failed_scans": {},
    }
    for scan in scans:
        checkpoint_guard()
        metadata = read_input_metadata(scan, args.archive)
        measured, modes, fingerprint = disk_images(args.archive, scan, author)
        row = source["scans"][scan]
        if row["archive_md5"] != metadata["archive_md5"] or fingerprint != row["diskset_sha256"]:
            raise ValueError("The observed diffraction intensities changed")
        try:
            results = evaluate(measured, modes, metadata["metadata_fields"], settings, row, local["scans"][scan],
                               args.pilot, args.check_grid and scan in ("DS6", "DS12"))
        except (ValueError, np.linalg.LinAlgError) as exc:
            result["failed_scans"][scan] = str(exc)
            print(scan, str(exc), flush=True)
        else:
            result["scans"][scan] = results
            print(scan, [[q["rmse"] for q in f["scores"]] for f in results], flush=True)
        result["wall_seconds"] = perf_counter()-start
        checkpoint_guard()
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
