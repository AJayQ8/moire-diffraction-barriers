"""Test whether regional moiré geometry improves fixed shear predictions."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import polar
from scipy.ndimage import distance_transform_edt

from bilayer_relaxation import ConvergenceError, relax, sample_disregistry
from observe_bilayers import (
    _author_strain_gamma_percent,
    _good_interior_triangle_mask,
    _inside_polygon_rc,
    _smooth_like_paper,
    _stacking_sector_labels,
    load_dataset,
)
from run_bilayer_prediction import error
from run_registry_prediction import fit_center_geometry, rotation
from shared_compute import atomic_json, checkpoint_guard


PROJECT = Path(__file__).resolve().parent
INTERFACE = ("MoS2", "MoS2", "P")


def read_json(path):
    return json.loads(path.read_text())


def affine_from_fit(fit, dataset):
    return fit[:2].T @ rotation(-np.deg2rad(dataset.sample_rotation_degrees))


def geometry_properties(gradient):
    rot, stretch = polar(np.eye(2) + gradient)
    if np.linalg.det(rot) <= 0 or np.linalg.det(stretch) <= 0:
        raise ValueError("The inferred local deformation is not orientation preserving")
    stretches = np.linalg.eigvalsh(stretch)
    return {
        "rotation_degrees": float(np.rad2deg(np.arctan2(rot[1, 0], rot[0, 0]))),
        "area_ratio": float(np.linalg.det(stretch)),
        "directional_stretch_percent": float(100 * np.ptp(stretches)),
    }


def with_scan_stretch(local, whole):
    """Keep local twist and area while substituting the scan-wide stretch shape."""
    local_rot, local_stretch = polar(np.eye(2) + local)
    _, scan_stretch = polar(np.eye(2) + whole)
    scale = np.sqrt(np.linalg.det(local_stretch) / np.linalg.det(scan_stretch))
    return local_rot @ (scale * scan_stretch) - np.eye(2)


def solve(gradient, basis, minimum_grid):
    cell = np.linalg.solve(gradient, basis)
    longest = np.linalg.norm(cell, axis=0).max()
    n = max(minimum_grid, 2 ** int(np.ceil(np.log2(longest / 0.3))))
    if n > 512:
        raise ValueError(f"Regional moiré cell would require a {n} by {n} grid")
    grid, field, receipt = relax(gradient, basis, INTERFACE, n, checkpoint_guard)
    if (not receipt["success"] or receipt["force_rms_eV_per_nm"] > 2e-4
            or receipt["energy_eV_per_cell"] >= receipt["rigid_energy_eV_per_cell"]):
        raise ConvergenceError(f"Unqualified regional equilibrium on grid {n}: {receipt}")
    return grid, field, {"grid": n, **receipt}


def shear_field(dataset, field, mask, sigma):
    masked = np.where(dataset.source_valid_mask[..., None], field, np.nan)
    smoothed = _smooth_like_paper(masked, sigma)
    gamma = _author_strain_gamma_percent(smoothed, dataset.pixel_nm, dataset.sample_rotation_degrees)
    labels, _, _ = _stacking_sector_labels(smoothed, dataset.lattice_nm, mask & np.isfinite(gamma))
    return gamma, labels


def sector_means(gamma, labels, mask):
    samples = [gamma[mask & (labels == label) & np.isfinite(gamma)] for label in (1, 2, 3)]
    if any(len(values) == 0 for values in samples):
        raise ValueError("The region does not cover all three wall sectors")
    return [float(values.mean()) for values in samples], [int(values.size) for values in samples]


def score_region(fields, mask, use_observed_labels):
    target_gamma, target_labels = fields["observed"]
    observed, counts = sector_means(target_gamma, target_labels, mask)
    scores = {"observed": {"wall_shear_percent": observed, "pixel_counts": counts}}
    for name, (gamma, labels) in fields.items():
        if name == "observed":
            continue
        shear, counts = sector_means(gamma, target_labels if use_observed_labels else labels, mask)
        scores[name] = {"wall_shear_percent": shear, "pixel_counts": counts, **error(observed, shear)}
    equal_score = error(observed, [0, 0, 0])["directional_rmse_percentage_points"]
    scores["equal_direction"] = {"directional_rmse_percentage_points": equal_score}
    return scores


def score_change(first, second):
    measured = np.subtract(first["observed"]["wall_shear_percent"], second["observed"]["wall_shear_percent"])
    result = {"observed": {"shear_change_percentage_points": measured.tolist()}}
    for name in first.keys() - {"observed", "equal_direction"}:
        predicted = np.subtract(first[name]["wall_shear_percent"], second[name]["wall_shear_percent"])
        result[name] = {"shear_change_percentage_points": predicted.tolist(), **error(measured, predicted)}
    result["equal_direction"] = {"directional_rmse_percentage_points":
                                 error(measured, [0, 0, 0])["directional_rmse_percentage_points"]}
    return result


def regional_fields(dataset, centers, basis, points, whole, settings):
    fit, info = fit_center_geometry(dataset.u_nm, centers, basis, dataset.pixel_nm, settings["center_inlier_nm"])
    gradient = affine_from_fit(fit, dataset)
    geometry = geometry_properties(gradient)
    mixed = with_scan_stretch(gradient, whole["gradient"])
    mixed_geometry = geometry_properties(mixed)
    for name in ("rotation_degrees", "area_ratio"):
        if not np.isclose(geometry[name], mixed_geometry[name], rtol=0, atol=1e-10):
            raise ValueError(f"The scan-stretch control changed the local {name}")
    cell = np.linalg.solve(gradient, basis)
    local = points @ fit[:2] + fit[-1]
    n = whole["n"]
    local_grid, local_shift, local_receipt = solve(gradient, basis, n)
    mixed_grid, mixed_shift, mixed_receipt = solve(mixed, basis, local_receipt["grid"])
    grid, shift = whole["relaxation"]
    probe = settings["probe_fwhm_nm"] / np.sqrt(8 * np.log(2))
    fields = {
        "whole_relaxation_on_local_geometry": sample_disregistry(grid, shift, local, cell, probe),
        "scan_stretch_at_local_twist": sample_disregistry(mixed_grid, mixed_shift, local, cell, probe),
        "local_relaxation": sample_disregistry(local_grid, local_shift, local, cell, probe),
    }
    info.update(fit_scan=fit.tolist(), affine_crystal=gradient.tolist(), cell_nm=cell.tolist(),
                geometry=geometry, scan_stretch_control_geometry=mixed_geometry,
                local_solve=local_receipt, scan_stretch_solve=mixed_receipt)
    return fields, info


def run_scan(dataset, original, fit, settings):
    mask, triangles = _good_interior_triangle_mask(dataset)
    vertices = np.asarray(triangles)
    centroids = dataset.centers_rc_px[vertices].mean(axis=1)
    supports = np.stack([_inside_polygon_rc(mask.shape, dataset.centers_rc_px[triangle]) for triangle in vertices])
    a = dataset.lattice_nm
    basis = a * np.array(((1, 0.5), (0, np.sqrt(3) / 2)))
    yy, xx = np.indices(mask.shape, dtype=float) * dataset.pixel_nm
    points = np.stack((xx, yy), axis=-1)
    gradient = affine_from_fit(fit, dataset)
    if not np.allclose(gradient, original["affine_crystal"], atol=1e-12, rtol=0):
        raise ValueError("The recorded whole-scan affine geometry changed")
    grid, shift, receipt = solve(gradient, basis, original["solver_grid"])
    whole = {"gradient": gradient, "relaxation": (grid, shift), "n": receipt["grid"]}
    rigid = points @ fit[:2] + fit[-1]
    probe = settings["probe_fwhm_nm"] / np.sqrt(8 * np.log(2))
    field = sample_disregistry(grid, shift, rigid, grid.cell, probe)
    sigma = settings["smoothing_sigma_pixels"]
    initial = {name: shear_field(dataset, value, mask, sigma)
               for name, value in (("observed", dataset.u_nm), ("whole_scan", field))}
    for name, ref in (("observed", original["observed"]), ("whole_scan", original["comparisons"]["full"])):
        actual, _ = sector_means(*initial[name], mask)
        if not np.allclose(actual, ref["wall_shear_percent"], atol=1e-8, rtol=0):
            raise ValueError(f"The original {name} shear readout did not reproduce")
    result = {"whole_scan_geometry": geometry_properties(gradient), "whole_scan_solve": receipt, "partitions": {}}
    for axis, name in enumerate(("row", "column")):
        groups = np.array_split(np.argsort(centroids[:, axis], kind="stable"), 2)
        region_masks = [supports[group].any(axis=0) for group in groups]
        exclusive = region_masks[0] ^ region_masks[1]
        partition = {"regions": []}
        for i, group in enumerate(groups):
            checkpoint_guard()
            center_ids = np.unique(vertices[group])
            region = {"triangles": group.tolist(), "center_indices": center_ids.tolist()}
            partition["regions"].append(region)
            try:
                local, info = regional_fields(dataset, dataset.centers_rc_px[center_ids], basis, points, whole, settings)
                fields = {**initial, **{key: shear_field(dataset, field, mask, sigma) for key, field in local.items()}}
            except (ConvergenceError, ValueError, np.linalg.LinAlgError) as exc:
                region["failure"] = str(exc)
                continue
            region["center_fit"] = info
            support = region_masks[i] & exclusive
            away = distance_transform_edt(~region_masks[1 - i]) > 2 * sigma
            scores = {}
            for tag, valid in (("original_support", support), ("trimmed_support", support & away)):
                for labels, observed_labels in (("field_labels", False), ("observed_labels", True)):
                    try:
                        scores[f"{tag}_{labels}"] = score_region(fields, valid, observed_labels)
                    except ValueError as exc:
                        scores[f"{tag}_{labels}"] = {"failure": str(exc)}
            region["scores"] = scores
        first, second = (region.get("scores", {}) for region in partition["regions"])
        partition["shear_change"] = {mode: score_change(first[mode], second[mode]) for mode in first.keys() & second.keys()
                                     if "failure" not in first[mode] and "failure" not in second[mode]}
        result["partitions"][name] = partition
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch", choices=("development", "held_out"), required=True)
    parser.add_argument("--scan", nargs="+")
    args = parser.parse_args()
    shear_protocol = PROJECT / "shear_protocol_v01.json"
    geometry_protocol = PROJECT / "registry_protocol_v01.json"
    settings = read_json(shear_protocol)
    settings["center_inlier_nm"] = read_json(geometry_protocol)["center_inlier_nm"]
    original = PROJECT / f"shear_{args.batch}_v02.json"
    registry = PROJECT / f"registry_{args.batch}_v01.json"
    old_scores, old_fits = (read_json(path)["scans"] for path in (original, registry))
    allowed = settings["batches"][args.batch]
    cases = args.scan or allowed
    if not set(cases) <= set(allowed):
        parser.error("Scan selection must belong to the chosen scored batch")
    inputs = (Path(__file__), PROJECT / "regional_shear_protocol_v01.json", shear_protocol, geometry_protocol,
              original, registry, PROJECT / "bilayer_relaxation.py", PROJECT / "observe_bilayers.py",
              PROJECT / "run_registry_prediction.py", PROJECT / "run_bilayer_prediction.py")
    result = {"batch": args.batch, "scans": {}, "failed_scans": {}, "input_sha256": {
        path.name: sha256(path.read_bytes()).hexdigest() for path in inputs}}
    start = perf_counter()
    for scan in cases:
        checkpoint_guard()
        data = load_dataset(scan, args.archive, allowed_ids=allowed)
        if data.archive_md5 != old_scores[scan]["archive_md5"]:
            raise ValueError(f"{scan}: the experimental archive has changed")
        try:
            result["scans"][scan] = run_scan(data, old_scores[scan], np.asarray(old_fits[scan]["fit_scan"]), settings)
        except (ConvergenceError, ValueError, np.linalg.LinAlgError) as exc:
            result["failed_scans"][scan] = str(exc)
        print(scan, "failed" if scan in result["failed_scans"] else "complete", round(perf_counter() - start, 2), flush=True)
        result["wall_seconds"] = perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
