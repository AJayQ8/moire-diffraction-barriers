"""Compare measured, removed and rotated directional stretch on the second ring."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import polar
from scipy.spatial import ConvexHull

from author_detector_replay import rms
from author_release import load_author_release
from bilayer_relaxation import sample_disregistry
from observe_bilayers import _inside_polygon_rc, load_dataset
from run_detector_prediction import AUTHOR_BASIS
from run_detector_wall_profiles import to_scan_frame
from run_heterobilayer_sparse_geometry import solve_from_geometry
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_elastic_components import common_rings
from run_moire_regime_benchmark import SETTINGS, detector_data
from run_mos2_cross_ring import locate_centers
from run_mos2_directional_diffraction import camera, wall_sectors
from run_mos2_template_transfer import BENCHMARK
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


SPLIT_RING = PROJECT / "mos2_cross_ring_v01.json"
DIRECTIONAL = PROJECT / "mos2_directional_diffraction_v01.json"


def readout_context(data, detector, benchmark, reference, settings):
    measured, _, modes, coefficients, _ = detector
    frame = benchmark["detector_alignment"]
    row, col = data.metadata.get("author_unwrap_crop_origin_row_col", (0, 0))
    height, width = data.u_nm.shape[:2]
    measured = to_scan_frame(measured[row:row+height, col:col+width], frame["spatial_turns"], frame["spatial_reflection"])
    mirror = np.diag((-1, 1)) if frame["vector_reflection"] else np.eye(2)
    transform = np.linalg.inv(AUTHOR_BASIS) @ rotation(frame["vector_rotation_60deg_steps"]*np.pi/3) @ mirror
    q = modes @ transform / data.lattice_nm
    rings = common_rings(modes)
    basis = data.lattice_nm*np.array(((1, 0.5), (0, np.sqrt(3)/2)))
    first = rings["first"]
    initial = np.asarray(reference["archived_affine_seed"])
    centers, _, _ = locate_centers(data, initial, basis, measured[..., first], q[first], coefficients[first], np.zeros(2))
    yy, xx = np.indices((height, width), dtype=float)
    points = np.stack((xx, yy), axis=-1)*data.pixel_nm
    fit = np.asarray(reference["first_ring_affine"])
    background = points @ fit[:2] + fit[-1]
    support = _inside_polygon_rc((height, width), centers[ConvexHull(centers).vertices])
    distance2 = np.full((height, width), np.inf)
    for y, x in centers:
        distance2 = np.minimum(distance2, (yy-y)**2 + (xx-x)**2)
    support &= distance2 > (2*settings["smoothing_sigma_pixels"])**2
    support[:4] = support[-4:] = support[:, :4] = support[:, -4:] = False
    if support.sum() != reference["score_pixels"]:
        raise ValueError("The established split-ring mask changed")
    gradient = fit[:2].T @ rotation(-np.deg2rad(data.sample_rotation_degrees))
    context = {"measured": measured, "reciprocal": q, "coefficients": coefficients,
               "probe_pixels": settings["probe_fwhm_nm"]/(data.pixel_nm*np.sqrt(8*np.log(2)))}
    return context, basis, background, gradient, support, rings


def stretch_controls(gradient):
    angle, stretch = polar(np.eye(2) + gradient)
    area_ratio = np.linalg.det(stretch)
    yield "measured_direction", gradient
    yield "no_directional_stretch", np.sqrt(area_ratio)*angle - np.eye(2)
    for degrees in (-60, 60):
        turn = rotation(np.deg2rad(degrees))
        rotated = turn @ stretch @ turn.T
        yield f"stretch_rotated_{degrees:+d}_degrees", angle @ rotated - np.eye(2)


def score_scan(data, detector, benchmark, reference, settings):
    context, basis, background, gradient, support, rings = readout_context(data, detector, benchmark, reference, settings)
    interface = tuple(benchmark["interface"])
    origin = basis @ np.full(2, 2/3 if interface[-1] == "AP" else 0.0)
    observed_cell = np.linalg.solve(gradient, basis)
    masks = {"interior": support}
    if interface[-1] == "P":
        all_walls = np.logical_or.reduce(tuple(wall_sectors(background, basis).values()))
        if (support & all_walls).sum() >= 75:
            masks["wall_regions"] = support & all_walls
    predictions = {"rigid": camera(background, context)}
    for name, test_gradient in stretch_controls(gradient):
        checkpoint_guard()
        grid, motion, _ = solve_from_geometry(test_gradient, basis, interface)
        field = sample_disregistry(grid, motion, origin + background, observed_cell) - origin
        predictions[name] = camera(field, context)
    actual = context["measured"]
    scores = {
        region: {name: {k: rms(actual[mask][:, ring], field[mask][:, ring]) for k, ring in rings.items()}
                 for name, field in predictions.items()} for region, mask in masks.items()
    }
    for current, earlier in (("rigid", "rigid"), ("measured_direction", "full")):
        for k, value in scores["interior"][current].items():
            if abs(value-reference["scores"][earlier][k]) > 1e-10:
                raise ValueError(f"The earlier {current} comparison changed")
    projections = {}
    for k, ring in rings.items():
        isotropic = predictions["no_directional_stretch"][support][:, ring]
        residual = (actual[support][:, ring] - isotropic).ravel()
        step = (predictions["measured_direction"][support][:, ring] - isotropic).ravel()
        projections[k] = float(step @ residual / (step @ step))
    _, stretch = polar(np.eye(2) + gradient)
    return {"interface": interface, "mask_pixels": {k: int(v.sum()) for k, v in masks.items()},
            "directional_stretch_percent": float(100*np.ptp(np.linalg.eigvalsh(stretch))),
            "rmse": scores, "measured_projection_onto_directional_change": projections}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    cross, benchmark, settings, directional = (json.loads(p.read_text()) for p in (SPLIT_RING, BENCHMARK, SETTINGS, DIRECTIONAL))
    for name, expected in {**cross["input_sha256"], **directional["input_sha256"]}.items():
        if digest(PROJECT / name) != expected:
            raise ValueError(f"The split-ring or directional input changed in {name}")
    author = load_author_release(args.release, args.dependencies)
    scans = [name for name, row in cross["scans"].items() if "published_zero" in row]
    result = {
        "scope": "Estimate global geometry from first-ring stacking centers as in the earlier split-ring check. Score the two diffraction rings separately under four relaxations with identical geometry: measured directional stretch, isotropic stretch at the same area and twist, and rotations of the measured stretch by plus and minus 60 degrees. The archive calibration and search-window seed use both rings. This is a retrospective comparison within each acquisition.",
        "input_sha256": {p.name: digest(p) for p in (Path(__file__), SPLIT_RING, BENCHMARK, SETTINGS, DIRECTIONAL)},
        "scans": {},
    }
    for scan in scans:
        checkpoint_guard()
        data = load_dataset(scan, args.archive, allowed_ids=scans, allow_author_crop=True)
        reference = benchmark["scans"][scan]
        if data.archive_md5 != reference["archive_md5"]:
            raise ValueError("The map archive changed")
        row = score_scan(data, detector_data(args.archive, scan, author), reference, cross["scans"][scan]["published_zero"], settings)
        result["scans"][scan] = row
        result["wall_seconds"] = perf_counter()-start
        checkpoint_guard()
        atomic_json(args.output, result)
        print(scan, {k: v["second"] for k, v in row["rmse"]["interior"].items()}, flush=True)


if __name__ == "__main__":
    main()
