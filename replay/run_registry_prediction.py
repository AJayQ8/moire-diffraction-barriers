"""Test the fixed bilayer model against experimental relative disregistry."""

import argparse
from hashlib import sha256
from io import BytesIO
from itertools import combinations
import json
from pathlib import Path
import pickle
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from numpy.core import multiarray
from scipy import ndimage as ndi

from bilayer_relaxation import ConvergenceError, disregistry_predictions


PROJECT = Path(__file__).resolve().parent


class ArrayUnpickler(pickle.Unpickler):
    """Read archived numerical fields without loading the authors' classes."""

    def find_class(self, module, name):
        if (module, name) in (("numpy", "ndarray"), ("numpy", "dtype")):
            return getattr(np, name)
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("_reconstruct", "scalar"):
            return getattr(multiarray, name)
        raise pickle.UnpicklingError(f"Unexpected type {module}.{name}")


def rotation(angle):
    return np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))


def load_registry(archive_dir, scan):
    with ZipFile(archive_dir / f"{scan}.zip") as archive:
        field_payload = archive.read(f"{scan}/unwrap.pkl")
        metadata_payload = archive.read(f"{scan}/info.txt")
    raw, centers, _, _ = ArrayUnpickler(BytesIO(field_payload)).load()
    metadata = {}
    for line in metadata_payload.decode().splitlines():
        if ":" in line and not line.startswith("#"):
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.split("#", 1)[0].strip()
    raw = np.asarray(raw, dtype=float) * float(metadata["LatticeConstant"])
    raw = raw @ rotation(float(metadata["UnwrapSPCorrectionRotation"])).T
    return raw, np.asarray(centers, dtype=float), metadata, {
        "unwrap_sha256": sha256(field_payload).hexdigest(),
        "metadata_sha256": sha256(metadata_payload).hexdigest(),
    }


def fit_center_geometry(raw, centers, basis, pixel_nm, inlier_nm):
    """Fit only AA center positions and their integer lattice translations."""
    positions = centers[:, ::-1] * pixel_nm
    iy, ix = np.rint(centers).astype(int).T
    measured = raw[iy, ix]
    finite = np.isfinite(measured).all(axis=-1)
    design = np.column_stack((positions[finite], np.ones(finite.sum())))
    target = np.rint(measured[finite] @ np.linalg.inv(basis).T) @ basis.T
    best = (0, -np.inf)
    for indices in combinations(range(len(design)), 3):
        choice = np.array(indices)
        triangle = design[choice]
        if abs(np.linalg.det(triangle)) < 10:  # twice the triangle area in nm²
            continue
        residual = np.linalg.norm(design @ np.linalg.solve(triangle, target[choice]) - target, axis=1)
        inliers = residual < inlier_nm
        score = (int(inliers.sum()), -float(np.median(residual)))
        if score > best:
            best = score
            retained = inliers
    if best[0] < 3 or best[0] * 2 <= len(design):
        raise ValueError("No non-collinear majority of archived AA centers supports an affine cell")
    fit = np.linalg.lstsq(design[retained], target[retained], rcond=None)[0]
    error = design @ fit - target
    return fit, {
        "finite_centers": len(design),
        "retained_centers": int(retained.sum()),
        "center_rms_nm": float(np.sqrt(np.mean(np.sum(error[retained] ** 2, axis=1)))),
    }


def score_fields(observed, predictions, affine, smoothing):
    mask = np.isfinite(observed).all(axis=-1)
    weights = ndi.gaussian_filter(mask.astype(float), smoothing, mode="constant")
    interior = mask & (weights > 0.97)
    if not np.any(interior):
        raise ValueError("No supported interior pixels for the disregistry comparison")

    def process(values):
        filtered = ndi.gaussian_filter(np.where(mask[..., None], values, 0), (smoothing, smoothing, 0),
                                       mode="constant")
        return filtered[interior] / weights[interior, None]

    target = process(observed - affine)
    result = {}
    for name, prediction in predictions.items():
        field = process(prediction - affine)
        diff = field - target
        offset = diff.mean(axis=0)
        centered = field - field.mean(axis=0)
        norm = np.sum(centered**2)
        result[name] = {
            "rmse_nm": float(np.sqrt(np.mean(np.sum(diff**2, axis=-1)))),
            "offset_free_rmse_nm": float(np.sqrt(np.mean(np.sum((diff - offset)**2, axis=-1)))),
            "residual_offset_nm": offset.tolist(),
            "observed_over_predicted_projection": float(np.sum(centered * target) / norm) if norm else None,
        }
    return result, int(interior.sum())


def run_case(scan, archive_dir, protocol, guard):
    observed, centers, metadata, sources = load_registry(archive_dir, scan)
    pixel = float(metadata["PixelSize"])
    lattice = float(metadata["LatticeConstant"])
    basis = lattice * np.array(((1, 0.5), (0, np.sqrt(3) / 2)))
    fit, geometry = fit_center_geometry(observed, centers, basis, pixel, protocol["center_inlier_nm"])
    affine = fit[:2].T @ rotation(-np.deg2rad(float(metadata["SampleRotation"])))
    cell = np.linalg.solve(affine, basis)
    n = max(64, 2**int(np.ceil(np.log2(np.max(np.linalg.norm(cell, axis=0)) / 0.3))))
    if n > 512:
        raise ValueError(f"The fitted cell would need a grid beyond this job's reservation: {n}")
    yy, xx = np.indices(observed.shape[:2], dtype=float) * pixel
    rigid = np.stack((xx, yy), axis=-1) @ fit[:2] + fit[-1]
    predictions, solves = disregistry_predictions(affine, basis, rigid, ("MoS2", "MoS2", "P"), n,
                                                  protocol["probe_fwhm_nm"], guard, include_linear_response=True)
    errors, interior = score_fields(observed, predictions, rigid, protocol["smoothing_sigma_pixels"])
    return {"geometry": geometry, "fit_scan": fit.tolist(), "gradient_crystal": affine.tolist(),
            "cell_nm": cell.tolist(), "grid_size": n, "interior_pixels": interior,
            "implied_twist_deg": abs(float(np.rad2deg((affine[1, 0] - affine[0, 1]) / 2))),
            "reported_twist_deg": float(metadata["AvgMoireTwist"]),
            "errors": errors, "solves": solves, "source_sha256": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_dir", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch", choices=("development", "held_out"), required=True)
    args = parser.parse_args()
    from shared_compute import atomic_json, checkpoint_guard

    settings = json.loads(args.protocol.read_text())
    split = PROJECT / "shear_protocol_v01.json"
    cases = json.loads(split.read_text())["batches"][args.batch]
    start = perf_counter()
    result = {"batch": args.batch, "scans": {}, "failed_scans": {},
              "sha256": {path.name: sha256(path.read_bytes()).hexdigest()
                         for path in (args.protocol, split, Path(__file__), PROJECT / "bilayer_relaxation.py")}}
    for scan in cases:
        checkpoint_guard()
        try:
            row = run_case(scan, args.archive_dir, settings, checkpoint_guard)
        except (ConvergenceError, ValueError, np.linalg.LinAlgError) as error:
            result["failed_scans"][scan] = str(error)
            print(scan, str(error), flush=True)
        else:
            result["scans"][scan] = row
            print(scan, {name: round(score["rmse_nm"], 4) for name, score in row["errors"].items()}, flush=True)
        result["wall_seconds"] = perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
