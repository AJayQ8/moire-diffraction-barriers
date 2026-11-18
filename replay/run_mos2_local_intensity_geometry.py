"""Test whether first-ring local geometry improves second-ring model prediction."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull

from author_detector_replay import rms
from author_release import load_author_release
from bilayer_relaxation import sample_disregistry
from observe_bilayers import _inside_polygon_rc, read_input_metadata
from run_heterobilayer_sparse_geometry import linear_response, solve_from_geometry
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_regime_benchmark import SETTINGS, relaxed_ray
from run_mos2_camera_spatial_replay import phase_camera
from run_mos2_directional_split_ring import stretch_controls
from run_mos2_first_ring_geometry import INTERFACE
from run_mos2_intensity_only_replay import camera_fit, disk_images, geometry
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


SOURCE = PROJECT / "mos2_intensity_only_replay_v01.json"


def squared_vector_error(displacement):
    return float(np.sqrt(np.mean(np.sum(displacement**2, axis=-1))))


def geometric_folds(centers, xy, pixel, basis, reference, support):
    centers_xy = centers[:, ::-1]*pixel
    design = np.column_stack((centers_xy, np.ones(len(centers))))
    translations = np.rint((design@reference)@np.linalg.inv(basis).T)@basis.T
    labels = np.rint(translations@np.linalg.inv(basis).T).astype(int)
    groups = (labels[:, 0]-labels[:, 1]) % 3
    for i in range(3):
        train = groups != i
        if train.sum() < 5:
            continue
        hull = ConvexHull(centers[train])
        held_inside = ~train & (np.max(centers @ hull.equations[:, :2].T+hull.equations[:, -1], axis=1) <= 1e-8)
        if not np.any(held_inside):
            continue
        fitted = np.linalg.lstsq(design[train], translations[train], rcond=None)[0]
        if np.linalg.det(fitted[:2]) <= 0:
            continue
        nominal = xy @ fitted[:2]+fitted[-1]
        residual = translations-design@fitted
        local = RBFInterpolator(centers_xy[train], residual[train], kernel="thin_plate_spline")
        warped = nominal+local(xy.reshape(-1, 2)).reshape(xy.shape)
        mask = support & _inside_polygon_rc(support.shape, centers[train][hull.vertices])
        dy, dx = np.gradient(warped, pixel, axis=(0, 1))
        relative_jacobian = (dx[..., 0]*dy[..., 1] - dy[..., 0]*dx[..., 1])/np.linalg.det(fitted[:2])
        if not mask.any() or np.min(relative_jacobian[mask]) <= 0:
            continue
        diagnostic = {"training_centers": int(train.sum()), "unused_centers_inside_hull": int(held_inside.sum()),
                      "affine_unused_center_error_nm": squared_vector_error(residual[held_inside]),
                      "warped_unused_center_error_nm": squared_vector_error(
                          residual[held_inside]-local(centers_xy[held_inside])),
                      "minimum_jacobian_ratio": float(relative_jacobian[mask].min())}
        yield i, nominal, warped, fitted, mask, diagnostic


def predict_signals(nominal, warped, gradient, context):
    basis = context["basis"]
    observed_cell = np.linalg.solve(gradient, basis)
    models = {}
    grid, shift, _ = solve_from_geometry(gradient, basis, INTERFACE)
    linear = linear_response(grid, INTERFACE)
    amplitude = relaxed_ray(grid, linear, INTERFACE)
    _, iso_gradient = next(row for row in stretch_controls(gradient) if row[0] == "no_directional_stretch")
    isotropic, iso_shift, _ = solve_from_geometry(iso_gradient, basis, INTERFACE)
    for label, rigid in (("affine", nominal), ("local", warped)):
        fields = {"rigid": rigid, "full": sample_disregistry(grid, shift, rigid, observed_cell),
                  "scaled_first_order": sample_disregistry(grid, amplitude*linear, rigid, observed_cell),
                  "isotropic_relaxation": sample_disregistry(isotropic, iso_shift, rigid, observed_cell)}
        for name, field in fields.items():
            cos, sin = phase_camera(field, context["reciprocal"], context["probe"])
            second = context["rings"]["second"]
            models[f"{label}_{name}"] = (cos[..., second], sin[..., second])
    return models


def evaluate(images, modes, metadata, settings, recorded):
    context = geometry(images, modes, metadata, settings)
    receipt = context["receipt"]
    if receipt != recorded["geometry"]:
        raise ValueError("The reference first-ring geometry changed")
    pixel = receipt["pixel_nm"]
    angle = np.deg2rad(float(metadata["SampleRotation"]))
    yy, xx = np.indices(images.shape[:2])
    xy = np.stack((xx, yy), axis=-1)*pixel
    centers = np.array(receipt["centers_rc_px"])
    reference = np.array(receipt["first_ring_affine_fit"])
    target = images[..., context["rings"]["second"]]
    results = []
    for fold, nominal, warped, fitted, mask, diagnostic in geometric_folds(
        centers, xy, pixel, context["basis"], reference, context["support"],
    ):
        checkpoint_guard()
        gradient = fitted[:2].T @ rotation(-angle)
        fields = predict_signals(nominal, warped, gradient, context)
        scores = []
        for name, axis in (("rows", yy), ("columns", xx)):
            cut = np.median(axis[mask])
            buffer = np.ceil(2*context["probe"])
            halves = (mask & (axis < cut-buffer), mask & (axis > cut+buffer))
            if min(half.sum() for half in halves) < 300:
                continue
            for half, (train, test) in enumerate((halves, halves[::-1])):
                readout = {}
                for label, (cos, sin) in fields.items():
                    estimate, actual, _ = camera_fit(cos, sin, target, train, test, allow_odd=True)
                    readout[label] = rms(estimate, actual)
                scores.append({"axis": name, "calibration_half": half, "score_pixels": int(test.sum()), "rmse": readout})
        results.append({"withheld_group": fold, **diagnostic, "diffraction": scores})
    sample = [row["rmse"] for fold in results for row in fold["diffraction"]]
    return {"center_count": len(centers), "folds": results,
            "mean_second_ring_rmse": {key: float(np.mean([row[key] for row in sample])) for key in sample[0]} if sample else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    previous, settings = (json.loads(p.read_text()) for p in (SOURCE, SETTINGS))
    for file, expected in previous["input_sha256"].items():
        if digest(PROJECT/file) != expected:
            raise ValueError(f"The frozen intensity method changed in {file}")
    author = load_author_release(args.release, args.dependencies)
    result = {
        "scope": "First-ring geometry only. Withhold one of three deterministic center groups. Fit global and locally warped reference geometries on identical centers, score unused centers inside the hull, and predict second-ring intensities away from the centers. Estimate detector terms on one half and score the other with the original normalized intensity method. Compare fixed full, energy-scaled first order, isotropic relaxation and rigid fields on both geometries. Small maps may lack a supported interior center fold.",
        "input_sha256": {p.name: digest(p) for p in (Path(__file__), SOURCE, SETTINGS)},
        "scans": {},
    }
    for scan, reference in previous["scans"].items():
        checkpoint_guard()
        metadata = read_input_metadata(scan, args.archive)
        images, modes, fingerprint = disk_images(args.archive, scan, author)
        if fingerprint != reference["diskset_sha256"] or metadata["archive_md5"] != reference["archive_md5"]:
            raise ValueError("The intensity-only source archive changed")
        score = evaluate(images, modes, metadata["metadata_fields"], settings, reference)
        result["scans"][scan] = score
        result["wall_seconds"] = perf_counter()-start
        checkpoint_guard()
        atomic_json(args.output, result)
        print(scan, len(score["folds"]), score["mean_second_ring_rmse"], flush=True)


if __name__ == "__main__":
    main()
