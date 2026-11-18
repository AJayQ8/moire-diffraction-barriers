"""Replay the domain-based position fit and test its effect on MoS2 wall widths."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from author_detector_replay import detector_intensity
from author_release import load_author_release
from bilayer_relaxation import make_grid, sample_disregistry
from observe_bilayers import read_input_metadata
from run_heterobilayer_sparse_geometry import solve_from_geometry
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_directional_diffraction import wall_sectors
from run_mos2_frame_sensitivity import fold_masks
from run_mos2_intensity_only_replay import disk_images
from run_mos2_local_intensity_geometry import geometric_folds
from run_mos2_local_wall_mechanics import SCANS, local_gradient
from run_mos2_nonlinear_controls import minimize_all
from run_mos2_observation_controls import (
    HALF_WINDOW, INPUT_NAMES, NOMINAL, gaussian_gain, raster_coordinates, render_traces, sample_cuts,
)
from run_mos2_opposite_widths import DISJOINT, EVEN, check_frozen, paired_spots
from run_mos2_phase_warp_transfer import warp_field
from run_mos2_slow_readout_transfer import render, rms, smooth_basis
from run_mos2_wall_landscape import CROP, capture_footprint, get_observer, width_pair
from run_mos2_wall_prediction_from_domains import fit_coordinate_motion, wall_readout
from run_mos2_wall_profile_closure import transfer
from run_mos2_wall_width_transfer import DISTANCE, geometry_sections
from run_registry_prediction import rotation
from run_wall_heterostrain import SOURCE
from shared_compute import atomic_json, checkpoint_guard


CASES = ("nominal", "archived_residual", "guarded_residual", "guarded_physical")


def read_xy(field, xy):
    """Read x- and y-displacements at nominal raster coordinates."""
    rc = np.moveaxis(xy[..., ::-1], -1, 0)
    return np.stack([map_coordinates(field[..., j], rc, order=1, mode="nearest") for j in (0, 1)], axis=-1)


def anchored_affine(points, values):
    """Match the edge and project the two side centers as in geometry_sections."""
    edge = points[1] - points[0]
    length = np.linalg.norm(edge)
    tangent = edge / length
    normal = np.array([-tangent[1], tangent[0]])
    offsets = points[2:] - points[0]
    height = offsets @ normal
    along = (values[1] - values[0]) / length
    away = values[2:] - values[0] - (offsets @ tangent)[:, None] * along
    return np.outer(tangent, along) + np.outer(normal, (height @ away) / (height @ height))


def position_residual(field, points_xy, corners_xy):
    values = read_xy(field, points_xy)
    correction = anchored_affine(points_xy, values)
    affine = values[0] + (corners_xy - points_xy[0]) @ correction
    return read_xy(field, corners_xy) - affine, np.eye(2) + correction


def verify_position_convention():
    y, x = np.indices((21, 21), dtype=float)
    xy = np.stack((x, y), axis=-1)
    centers = np.array([[4., 10.], [16., 10.], [10., 4.], [10., 16.]])
    jacobian = np.array([[.1, -.04], [.067, .028]])
    residual, affine = position_residual(xy @ jacobian + [1., -2.], centers, xy)
    np.testing.assert_allclose(residual, 0., atol=1e-13)
    np.testing.assert_allclose(affine, np.eye(2) + jacobian, atol=1e-13)
    height = y - 10
    warp = np.stack((np.zeros_like(height), .2 * height * (1 - (height / 6)**2)), axis=-1)
    residual, affine = position_residual(warp, centers, xy)
    np.testing.assert_allclose(affine, np.eye(2), atol=1e-13)
    np.testing.assert_allclose(residual, warp, atol=1e-13)


def fit_positions(images, context, angle, interface, sections, group, archive):
    pixel = context["receipt"]["pixel_nm"]
    basis, q, probe = (context[k] for k in ("basis", "reciprocal", "probe"))
    first, second = (context["rings"][name] for name in ("first", "second"))
    yy, xx = np.indices(images.shape[:2], dtype=float)
    xy = np.stack((xx, yy), axis=-1)
    centers = np.asarray(context["receipt"]["centers_rc_px"])
    fit = np.asarray(context["receipt"]["first_ring_affine_fit"])
    folds = list(geometric_folds(centers, xy * pixel, pixel, basis, fit, context["support"]))
    selected, _, background, fit, support, _ = max(folds, key=lambda row: row[4].sum())
    if selected != group:
        raise ValueError("The archived center fold changed")
    gradient = fit[:2].T @ rotation(-angle)
    grid, motion, _ = solve_from_geometry(gradient, basis, interface)
    field = sample_disregistry(grid, motion, background, grid.cell)
    wall = support & np.logical_or.reduce(tuple(wall_sectors(background, basis).values()))
    footprint = capture_footprint(sections, support.shape)
    guard_px = int(np.ceil(2 * probe))
    period_px = np.sqrt(abs(np.linalg.det(grid.cell))) / pixel
    blur = render(field, np.ones(support.shape), q, probe)
    positions, receipts = {}, {}
    for name, excluded in (("archived", wall), ("guarded", wall | footprint)):
        checkpoint_guard()
        source = support & (distance_transform_edt(~excluded) > guard_px)
        if source.sum() < 300:
            raise ValueError("The source region outside the wall cuts is too small")
        controls, _, _ = smooth_basis(support.shape, source, period_px)
        warp, bound = fit_coordinate_motion(field, fit, controls, probe, images[..., first], q[first],
                                            source, blur[..., first, :], pixel)
        corrected = render(warp_field(field, warp), np.ones(support.shape), q[second], probe)
        signals = {"nominal": blur[..., second, :], "warped": corrected}
        runs = []
        for train, test in fold_masks(support, probe):
            calibration, target = train & source, test & wall
            if calibration.sum() < 200 or target.sum() < 60:
                continue
            scored = wall_readout(images[..., second], signals, calibration, (target,))
            runs.append({"calibration_pixels": int(calibration.sum()), "score_pixels": int(target.sum()),
                         **scored["rmse"]["0"]})
        displacement = rms(np.linalg.norm(warp[support], axis=-1))
        du_dy, du_dx = np.gradient(warp, axis=(0, 1))
        jacobian_det = (1 + du_dx[..., 0]) * (1 + du_dy[..., 1]) - du_dy[..., 0] * du_dx[..., 1]
        positions[name] = warp
        receipts[name] = {"source_pixels": int(source.sum()), "cut_pixels_in_source": int((source & footprint).sum()),
                          "control_nodes": controls.shape[-1], "coefficient_bound_fraction": bound,
                          "displacement_rms_px": displacement, "guard_px": guard_px, "wall_intensity": runs,
                          "coordinate_jacobian_min": float(jacobian_det[support].min())}
        if name == "archived":
            if source.sum() != archive["source_pixels"]:
                raise ValueError("The original domain source mask changed")
            np.testing.assert_allclose(displacement, archive["fits"]["full"]["position_rms_px"], atol=1e-8)
            for new, old in zip(runs, archive["folds"], strict=True):
                if new["calibration_pixels"] != old["calibration_pixels"] or new["score_pixels"] != old["wall_pixels"]["0"]:
                    raise ValueError("The original domain intensity scoring mask changed")
                np.testing.assert_allclose([new["nominal"], new["warped"]],
                                           [old["rmse"]["0"]["full"], old["rmse"]["0"]["full_warped"]],
                                           atol=1e-9, rtol=0)
        if receipts[name]["coordinate_jacobian_min"] <= 0:
            raise ValueError("The fitted positional map folds on its support")
    return positions, receipts, support


def model_rasters(section, gradient, angle, pixel, basis, n, points_xy, warps):
    """Keep nominal bilinear weights and move the physical evaluation points."""
    _, cuts, corners, weights = raster_coordinates(section, basis, pixel, n)
    matrix = gradient @ rotation(angle)
    pixel_to_grid = pixel * n * (matrix.T @ np.linalg.inv(basis).T)
    index = np.moveaxis(corners, 0, -1)
    origin_xy = section["coordinates"][1, 0, ::-1]
    origin_index = n * (section["phase"][1, 0] @ np.linalg.inv(basis).T)
    corners_xy = origin_xy + (index - origin_index) @ np.linalg.inv(pixel_to_grid)
    np.testing.assert_allclose(corners_xy, np.rint(corners_xy), atol=1e-8, rtol=0)
    corners_xy = np.rint(corners_xy)
    residuals, affines = {}, {}
    for name, warp in warps.items():
        residuals[name], affines[name] = position_residual(warp, points_xy, corners_xy)
    nonaffine_indices = {
        name: corners + np.moveaxis(residuals[name] @ np.linalg.solve(affines[name], pixel_to_grid), -1, 0)
        for name in warps
    }
    physical = np.linalg.solve(affines["guarded"], matrix.T).T @ rotation(-angle)
    physical_cell = np.linalg.solve(physical, basis)
    nodes = np.max(np.linalg.norm(physical_cell, axis=0)) / .30
    physical_n = max(n, 2**int(np.ceil(np.log2(nodes))))
    if physical_n > 512:
        raise ValueError("Recalibrated wall exceeds the archived mesh limit")
    physical_corners = nonaffine_indices["guarded"] * (physical_n / n)
    corners_by_case = (corners, nonaffine_indices["archived"], nonaffine_indices["guarded"], physical_corners)
    cut_mask = CROP.copy()
    cut_mask[CROP] = abs(DISTANCE) <= HALF_WINDOW
    return (cuts, weights, corners_by_case, physical, physical_n,
            {name: {"affine": affines[name].tolist(),
                    "residual_rms_px": rms(np.linalg.norm(residuals[name][..., cut_mask, :], axis=-1))}
             for name in warps})


def neighbors(context):
    centers = np.asarray(context["receipt"]["centers_rc_px"])[:, ::-1]
    pixel = context["receipt"]["pixel_nm"]
    fit = np.asarray(context["receipt"]["first_ring_affine_fit"])
    basis = context["basis"]
    registry = (centers * pixel) @ fit[:2] + fit[2]
    labels = np.rint(registry @ np.linalg.inv(basis).T)
    lattice = labels @ basis.T
    adjacent = np.isclose(np.linalg.norm(lattice[:, None] - lattice, axis=-1), np.linalg.norm(basis[:, 0]), atol=1e-9)
    return centers, adjacent


def wall_scores(context, images, modes, angle, original, cameras, opposite, local, warp_reference, old_center_group):
    pixel = original["pixel_nm"]
    basis, q, rings = (context[k] for k in ("basis", "reciprocal", "rings"))
    sections = {tuple(s["centers"]): s for s in geometry_sections(context, rotation(angle))}
    sections = [{**sections[tuple(row["centers"])], "keep": abs(DISTANCE) <= HALF_WINDOW} for row in opposite]
    physics = original["physics"]
    stiffness = physics["stiffness_eV_per_cell"]
    adhesion = physics["stacking_coefficients_meV_per_cell"]["anchor"]
    frozen = {tuple(row["centers"]): row for row in original["walls"]}
    by_mode = {tuple(mode): i for i, mode in enumerate(modes)}
    warps, fit_receipts, support = fit_positions(images, context, angle, ("MoS2", "MoS2", "P"), sections,
                                                  old_center_group, warp_reference)
    centers, adjacent = neighbors(context)
    walls = []
    for section, observed, saved in zip(sections, opposite, local, strict=True):
        checkpoint_guard()
        i, j = section["centers"]
        if saved["centers"] != [i, j]:
            raise ValueError("A local case does not match its measured wall")
        ref = frozen[(i, j)]
        selected = [np.asarray([by_mode[tuple(mode)] for mode in group]) for group in ref["miller_indices"]]
        halves = paired_spots(modes, rings, selected)
        if [[modes[order].tolist() for order in side] for side in halves] != observed["half_modes"]:
            raise ValueError("The archived spot selection changed")
        gradient = local_gradient(section, pixel, angle)
        np.testing.assert_allclose(gradient, saved["local_gradient"], atol=1e-12, rtol=0)
        n = saved["grid"]
        points = centers[[i, j, *np.flatnonzero(adjacent[i] & adjacent[j])]]
        if len(points) != 4:
            raise ValueError("A wall lacks its two opposing centers")
        cuts, weights, indices, physical, physical_n, positions = model_rasters(
            section, gradient, angle, pixel, basis, n, points, warps)
        grids = (make_grid(gradient, basis, n, *stiffness), make_grid(physical, basis, physical_n, *stiffness))
        registries = [grid.fractional @ basis.T + minimize_all(grid, adhesion)[0] for grid in grids]
        gains = [gaussian_gain(np.sum(grid.k**2, axis=-1), NOMINAL) for grid in grids]
        pair_readers = [get_observer(section, q, cameras[EVEN], rings, NOMINAL, group) for group in halves]
        reader = get_observer(section, q, cameras[EVEN], rings, NOMINAL, selected)
        observation = sample_cuts(images, section["coordinates"]).mean(axis=0)
        measured = observed["windows"][f"{HALF_WINDOW:g}"]["measured"]
        for half, prior in zip(pair_readers, measured, strict=True):
            check_frozen(width_pair(half, observation), prior)
        target = reader.shape(observation)
        model_scores = {}
        for camera, coeff in cameras.items():
            spectra = [np.fft.fft2(detector_intensity(registry, q, coeff), axes=(0, 1)) for registry in registries]
            model_scores[camera] = {}
            for name, corner in zip(CASES, indices, strict=True):
                idx = int(name == "guarded_physical")
                # The continuous trace is unused. Its indices scale with the mesh.
                _, trace = render_traces(spectra[idx], gains[idx], cuts * (physical_n / n if idx else 1), corner, weights)
                width = [width_pair(half, trace[CROP]) for half in pair_readers]
                if name == "nominal":
                    for fit, prior in zip(width, saved["local"][camera]["widths"], strict=True):
                        check_frozen(fit, prior)
                model_scores[camera][name] = {"widths": width, "registration": transfer(reader, target, trace)[0]}
        occupancy = map_coordinates(support.astype(float),
                                    np.moveaxis(section["coordinates"][:, section["keep"]], -1, 0),
                                    order=1, mode="constant")
        walls.append({**{k: observed[k] for k in ("centers", "midpoint_xy_nm", "spacing_nm")},
                      "measured": measured, "models": model_scores, "positions": positions,
                      "within_fit_hull": bool(np.all(occupancy >= 1 - 1e-9)), "physical_mesh": physical_n})
    return {"warp_fits": fit_receipts, "walls": walls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "release", "dependencies", "evidence", "opposite_result", "local_result",
                 "domain_result", "warp_result", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    verify_position_convention()
    start = perf_counter()
    paths = (SOURCE, BENCHMARK, SETTINGS, *(args.evidence / name for name in INPUT_NAMES),
             args.opposite_result, args.local_result, args.domain_result, args.warp_result)
    source, bench, settings, primary, control, optical, opposite, local, domain, reference = [
        json.loads(p.read_bytes()) for p in paths]
    libraries = ("run_mos2_wall_prediction_from_domains.py", "run_mos2_phase_warp_transfer.py",
                 "run_mos2_warp_model_controls.py", "run_mos2_reduced_equilibrium.py", "run_mos2_slow_readout_transfer.py",
                 "run_mos2_local_intensity_geometry.py", "run_mos2_local_wall_mechanics.py", "run_mos2_opposite_widths.py",
                 "run_mos2_observation_controls.py", "run_mos2_wall_landscape.py", "run_mos2_wall_width_transfer.py",
                 "run_mos2_wall_profile_closure.py", "bilayer_relaxation.py", "author_detector_replay.py")
    hashes = {p.name: digest(p) for p in (Path(__file__), *paths, *(PROJECT / name for name in libraries))}
    out = {"input_sha256": hashes, "requested_scans": list(SCANS), "cases": list(CASES), "scans": {},
           "scope": "Replay the published-data position fit from the first ring outside nominal wall sectors. "
                    "Repeat with the actual scored cut pixels also excluded, with the same probe guard. "
                    "Apply each map after probe convolution and before nominal raster interpolation. Subtract its "
                    "center-anchored affine part. In the guarded_physical case also use that affine part as a "
                    "coordinate calibration of the local elastic geometry. All experimental widths remain fixed."}
    elapsed = 0.
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        out = json.loads(args.output.read_bytes())
        if out["input_sha256"] != hashes or out["requested_scans"] != list(SCANS) or out["cases"] != list(CASES):
            raise ValueError("Resume requires identical geometry, code and fit inputs")
        elapsed = out["seconds"]
    author = load_author_release(args.release, args.dependencies)
    even = optical["frozen_common_coefficients"]
    for scan in SCANS:
        if scan in out["scans"]:
            continue
        checkpoint_guard()
        metadata = read_input_metadata(scan, args.archive)
        images, modes, checksum = disk_images(args.archive, scan, author)
        ref = source["scans"][scan]
        if checksum != ref["diskset_sha256"] or metadata["archive_md5"] != bench["scans"][scan]["archive_md5"]:
            raise ValueError(f"The public archive changed for {scan}")
        fields = metadata["metadata_fields"]
        context = centers_and_context(images, modes, fields, settings, tuple(ref["interface"]))
        if context["receipt"] != ref["geometry"]:
            raise ValueError(f"The frozen scan geometry changed for {scan}")
        angle = np.deg2rad(float(fields["SampleRotation"]))
        disjoint = control["scans"][scan]["camera_diagnostic"]["by_model"]["anchor"]["coefficients"]
        cameras = {EVEN: np.broadcast_to([even[0], 0., even[1]], (len(modes), 3)), DISJOINT: np.asarray(disjoint)}
        out["scans"][scan] = wall_scores(
            context, images, modes, angle, primary["scans"][scan], cameras, opposite["scans"][scan]["walls"],
            local["scans"][scan], domain["scans"][scan]["geometry"]["local"],
            reference["scans"][scan]["geometry"]["local"]["unused_center_group"])
        out["seconds"] = elapsed + perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, out)
        print(scan, len(out["scans"][scan]["walls"]), round(out["seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
