"""Test regional disregistry on a separate WSe2/MoS2 device."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import polar
from scipy.ndimage import gaussian_filter

from audit_registry_repeats import repeated_cell_reference
from bilayer_relaxation import ConvergenceError, MONOLAYERS, STACKING, relax, sample_disregistry, stack_energy
from observe_bilayers import _good_interior_triangle_mask, _inside_polygon_rc, load_dataset
from run_registry_prediction import fit_center_geometry, rotation
from shared_compute import atomic_json, checkpoint_guard


PROJECT = Path(__file__).resolve().parent
PROTOCOL = PROJECT / "heterobilayer_transfer_protocol_v01.json"
MODELS = ("whole_rigid", "whole_relaxed", "regional_rigid", "regional_relaxed")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def fingerprints():
    names = ("bilayer_relaxation.py", "observe_bilayers.py", "run_registry_prediction.py",
             "registry_protocol_v01.json", "audit_registry_repeats.py")
    return {p.name: digest(p) for p in (Path(__file__), PROTOCOL, *(PROJECT / name for name in names))}


def geometry_from_centers(dataset, centers, basis, points, tolerance):
    fit, row = fit_center_geometry(dataset.u_nm, centers, basis, dataset.pixel_nm, tolerance)
    # The archived heterobilayer displacement diverges. The SI orders the
    # larger WSe2 layer relative to MoS2, giving the same sign of dilation.
    gradient = fit[:2].T @ rotation(-np.deg2rad(dataset.sample_rotation_degrees))
    cell = np.linalg.solve(gradient, basis)
    orientation, stretch = polar(np.eye(2) + gradient)
    if np.linalg.det(orientation) <= 0:
        raise ValueError("The inferred relative lattice is not orientation preserving")
    row.update(fit_scan=fit.tolist(), gradient_crystal=gradient.tolist(), cell_nm=cell.tolist(),
               inferred_twist_deg=float(np.rad2deg(np.arctan2(orientation[1, 0], orientation[0, 0]))),
               mean_mismatch_percent=float(100 * (np.trace(stretch) / 2 - 1)))
    return points @ fit[:2] + fit[-1], gradient, cell, row


def solve(gradient, basis, interface, n):
    grid, shift, info = relax(gradient, basis, interface, n, checkpoint_guard)
    if (not info["success"] or info["force_rms_eV_per_nm"] > 2e-4
            or info["energy_eV_per_cell"] >= info["rigid_energy_eV_per_cell"]):
        raise ConvergenceError(f"The equilibrium failed the force and energy checks: {info}")
    return grid, shift, info


def blur(field, valid, weights, sigma):
    numerator = gaussian_filter(np.where(valid[..., None], field, 0.0), (sigma, sigma, 0), mode="constant")
    return numerator / np.maximum(weights[..., None], 1e-10)


def scores(deltas, mask):
    if not np.any(mask):
        raise ValueError("No measured layer shifts on the shared score mask")
    result = {}
    for model, delta in deltas.items():
        error = delta[mask]
        offset = error.mean(axis=0)
        result[model] = {"rmse_nm": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
                         "offset_free_rmse_nm": float(np.sqrt(np.mean(np.sum((error - offset)**2, axis=1))))}
    return {"pixel_count": int(mask.sum()), "models": result}


def average_scores(partitions, readout):
    halves = [half[readout] for halves in partitions.values() for half in halves if half[readout] is not None]
    if not halves:
        return None
    return {key: {metric: float(np.mean([half["models"][key][metric] for half in halves]))
                  for metric in ("rmse_nm", "offset_free_rmse_nm")}
            for key in halves[0]["models"]}


def run_scan(dataset, case, settings, interface, check_grid):
    shape = dataset.u_nm.shape[:2]
    if dataset.orientation != "ap" or not dataset.metadata["heterobilayer_correction_applied"]:
        raise ValueError("The score requires a source-identified antiparallel heterobilayer")
    if not np.isclose(dataset.lattice_nm, 0.328, rtol=0, atol=1e-6):
        raise ValueError("The saved registry period is inconsistent with WSe2")
    basis = dataset.lattice_nm * np.array(((1, 0.5), (0, np.sqrt(3) / 2)))
    # The saved vectors are measured in the WSe2 lattice convention. Eq. 10
    # uses the MoS2 bottom-layer basis. Map the fractional registry between
    # bases while preserving the physical period on the scan raster.
    model_lattice = MONOLAYERS[interface[0]][0]
    scale = model_lattice / dataset.lattice_nm
    model_basis = basis * scale
    yy, xx = np.indices(shape, dtype=float)
    points = np.stack((xx, yy), axis=-1) * dataset.pixel_nm
    mask, triangles = _good_interior_triangle_mask(dataset)
    vertices = np.asarray(triangles)
    polygon_centers = dataset.centers_rc_px[vertices].mean(axis=1)
    polygons = np.stack([_inside_polygon_rc(shape, dataset.centers_rc_px[tri]) for tri in vertices])
    rigid, gradient, cell, geometry = geometry_from_centers(
        dataset, dataset.centers_rc_px, basis, points, settings["center_inlier_nm"])
    n = max(64, 2 ** int(np.ceil(np.log2(np.linalg.norm(cell, axis=0).max() / 0.3))))
    if n > 512:
        raise ValueError(f"The fitted lattice would require a {n} by {n} grid")
    grid, shift, numerical = solve(scale * gradient, model_basis, interface, n)
    if not np.allclose(grid.cell, cell, rtol=0, atol=1e-10):
        raise ValueError("The registry unit conversion changed the physical moiré cell")
    probe = settings["probe_fwhm_nm"] / np.sqrt(8 * np.log(2))
    sigma = settings["smoothing_sigma_pixels"]
    origin = model_basis @ np.full(2, 2 / 3)

    def predicted_disregistry(local_rigid, local_cell, source_grid=grid, source_shift=shift):
        field = sample_disregistry(source_grid, source_shift, origin + scale * local_rigid, local_cell, probe)
        return (field - origin) / scale

    measured = dataset.u_nm
    valid = dataset.source_valid_mask & np.isfinite(measured).all(axis=-1)
    weights = gaussian_filter(valid.astype(float), sigma, mode="constant")
    support = mask & valid & (weights > 0.97)
    center_distance2 = np.full(shape, np.inf)
    for row, col in dataset.centers_rc_px:
        center_distance2 = np.minimum(center_distance2, (yy - row)**2 + (xx - col)**2)
    far_from_centers = center_distance2 > (2 * sigma)**2
    relaxed = predicted_disregistry(rigid, cell)
    baseline = {"whole_rigid": blur(rigid - measured, valid, weights, sigma),
                "whole_relaxed": blur(relaxed - measured, valid, weights, sigma)}
    smooth_reconstruction = -baseline["whole_rigid"]
    physical_cell_pixels = np.linalg.solve(np.asarray(geometry["fit_scan"])[:2].T, basis) / dataset.pixel_nm
    empirical, repeats = repeated_cell_reference(smooth_reconstruction, support, physical_cell_pixels)
    prior_cell_delta = empirical - smooth_reconstruction
    partitions = {}
    for axis, direction in enumerate(("row", "column")):
        ordered = np.argsort(polygon_centers[:, axis], kind="stable")
        groups = np.array_split(ordered, 2)
        regions = [polygons[group].any(axis=0) for group in groups]
        halves = []
        for i, group in enumerate(groups):
            center_ids = np.unique(vertices[group])
            local_rigid, _, local_cell, local_geo = geometry_from_centers(
                dataset, dataset.centers_rc_px[center_ids], basis, points, settings["center_inlier_nm"])
            local_relaxed = predicted_disregistry(local_rigid, local_cell)
            deltas = {**baseline, "regional_rigid": blur(local_rigid - measured, valid, weights, sigma),
                      "regional_relaxed": blur(local_relaxed - measured, valid, weights, sigma)}
            valid_half = support & regions[i] & ~regions[1 - i]
            primary = valid_half & far_from_centers
            repeat_support = primary & (repeats >= 3)
            repeat_scores = scores({**deltas, "other_cells": prior_cell_delta}, repeat_support) if np.any(repeat_support) else None
            halves.append({"centers": center_ids.tolist(), "triangles": group.tolist(), "geometry": local_geo,
                           "primary": scores(deltas, primary), "all_valid_pixels": scores(deltas, valid_half),
                           "repeated_cell_support": repeat_scores})
        partitions[direction] = halves
    qualification = {}
    if check_grid:
        fine_grid, fine_shift, _ = solve(scale * gradient, model_basis, interface, 2 * n)
        difference = predicted_disregistry(rigid, cell, fine_grid, fine_shift) - relaxed
        magnitude = np.linalg.norm(difference[support], axis=-1)
        qualification = {"comparison_grid": 2 * n, "rms_field_change_nm": float(np.sqrt(np.mean(magnitude**2))),
                         "max_field_change_nm": float(np.max(magnitude))}
        if qualification["rms_field_change_nm"] > 5e-4:
            raise ValueError(f"Grid convergence did not qualify: {qualification}")
    return {**case, "archive_md5": dataset.archive_md5, "saved_shape_row_col": list(shape),
            "reported_shape_row_col": dataset.metadata.get("reported_acquisition_shape_row_col", list(shape)),
            "whole_geometry": geometry, "grid": n, "solve": numerical, "qualification": qualification,
            "archive_registry_lattice_nm": dataset.lattice_nm, "model_registry_lattice_nm": model_lattice,
            "partitions": partitions, "primary_means": average_scores(partitions, "primary"),
            "all_valid_means": average_scores(partitions, "all_valid_pixels"),
            "repeated_cell_support_means": average_scores(partitions, "repeated_cell_support")}


def verify_freeze(path, current, development_ids):
    receipt = json.loads(path.read_text())
    development = path.parent / receipt["development_file"]
    if receipt["input_sha256"] != current or digest(development) != receipt["development_sha256"]:
        raise ValueError("The implementation or development result changed after the device test was frozen")
    earlier = json.loads(development.read_text())
    if (earlier["batch"] != "development" or earlier["input_sha256"] != current
            or set(earlier["scans"]) != set(development_ids) or earlier["failed_scans"]):
        raise ValueError("The frozen development result is incomplete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch", choices=("development", "held_out"), required=True)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    settings = json.loads((PROJECT / "registry_protocol_v01.json").read_text())
    cases = protocol["devices"][args.batch]
    hashes = fingerprints()
    if args.batch == "held_out":
        if args.freeze is None:
            parser.error("The held-out batch requires its recorded pre-test freeze")
        verify_freeze(args.freeze, hashes, protocol["devices"]["development"])
    interface = tuple(protocol["interface"])
    basis = MONOLAYERS[interface[0]][0] * np.array(((1, 0.5), (0, np.sqrt(3) / 2)))
    sites = np.array(((0, 0), (1 / 3, 1 / 3), (2 / 3, 2 / 3))) @ basis.T
    energies = stack_energy(sites, basis, STACKING[interface])[0]
    if not energies[0] > energies[1] > energies[2]:
        raise ValueError("Published AP stacking energies must order XX, MM, and XMMX in this basis")
    start = perf_counter()
    result = {"batch": args.batch, "input_sha256": hashes, "freeze_sha256": digest(args.freeze) if args.freeze else None,
              "stacking_energies_meV_XX_MM_XMMX": (1000 * energies).tolist(), "scans": {}, "failed_scans": {}}
    for scan in cases:
        checkpoint_guard()
        try:
            data = load_dataset(scan, args.archive, allowed_ids=cases, allow_author_crop=True)
            row = run_scan(data, protocol["cases"][scan], settings, interface, args.batch == "development")
        except (ConvergenceError, ValueError, np.linalg.LinAlgError) as exc:
            result["failed_scans"][scan] = str(exc)
            print(scan, "failed", str(exc), flush=True)
        else:
            result["scans"][scan] = row
            print(scan, {key: round(row["primary_means"][key]["rmse_nm"], 6) for key in MODELS}, flush=True)
        result["wall_seconds"] = perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
