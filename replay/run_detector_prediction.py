"""Compare fixed model predictions with measured, integrated diffraction disks."""

import argparse
from hashlib import sha256
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from bilayer_relaxation import ConvergenceError, disregistry_predictions
from run_registry_prediction import ArrayUnpickler, fit_center_geometry, load_registry, rotation
from shared_compute import atomic_json, checkpoint_guard


PROJECT = Path(__file__).resolve().parent
AUTHOR_BASIS = np.array(((-1, 0.5), (0, np.sqrt(3) / 2)))
CANDIDATES = np.array(((1, 0), (0, 1), (1, -1), (1, 1), (2, -1), (1, -2)))
MODES = np.repeat(CANDIDATES, 2, axis=0)  # Two opposite disks per axis.


class DiskInfo:
    """Inert holder for the archive's numerical disk attributes."""


class DiskReadback(ArrayUnpickler):
    def find_class(self, module, name):
        if (module, name) == ('diskset', 'DiskSet'):
            return DiskInfo
        return super().find_class(module, name)


def load_detector(archive, scan):
    with ZipFile(archive / f"{scan}.zip") as z:
        member = z.getinfo(f"{scan}/fit.pkl")
        if member.file_size > 128 * 1024**2:
            raise ValueError("Unexpected integrated disk archive size")
        payload = z.read(member)
    fit = DiskReadback(BytesIO(payload)).load()
    if len(fit) != 4:
        raise ValueError("Expected the published two-coefficient homobilayer detector fit")
    disk, scale, offset, stored = fit
    measured = np.asarray(disk._df[disk._in_use.astype(bool)]).transpose(1, 2, 0)
    if len(scale) != measured.shape[-1] or measured.shape[:2] != stored.shape[:2]:
        raise ValueError("Inconsistent archived detector geometry")
    if not 1 <= len(scale) <= len(MODES):
        raise ValueError("Unexpected number of integrated diffraction disks")
    minimum = np.min(measured, axis=(0, 1), keepdims=True)
    span = np.ptp(measured, axis=(0, 1), keepdims=True)
    if not np.all(np.isfinite(span) & (span > 0)):
        raise ValueError("An integrated disk lacks a finite intensity range")
    measured = (measured - minimum) / span
    best = None
    for basis in ("lattice", "cartesian"):
        coords = stored if basis == "lattice" else stored @ np.linalg.inv(AUTHOR_BASIS).T
        fringes = np.cos(np.pi * (coords[2:-2:2, 2:-2:2] @ MODES.T))**2
        raw = measured[2:-2:2, 2:-2:2]
        prediction = fringes[..., None, :] * scale[:, None] + offset[:, None]
        cost = np.mean((prediction - raw[..., None])**2, axis=(0, 1))
        disks, modes = linear_sum_assignment(cost)
        loss = float(cost[disks, modes].mean())
        if best is None or loss < best[0]:
            best = (loss, basis, MODES[modes], coords)
    loss, basis, modes, coords = best
    fitted = np.cos(np.pi * (coords @ modes.T))**2 * scale + offset
    return measured, fitted, modes, scale, offset, basis, np.sqrt(loss), sha256(payload).hexdigest()


def apply_spatial(field, turns, reflect):
    if reflect:
        field = field[:, ::-1]
    return np.rot90(field, turns, axes=(0, 1))


def match_archive_orientation(raw, fitted, a, modes, scale, offset):
    best = None
    for turns in range(4):
        for spatial_reflection in (False, True):
            sampled = apply_spatial(raw, turns, spatial_reflection)[2:-2:2, 2:-2:2]
            truth = fitted[2:-2:2, 2:-2:2]
            for crystal_reflection in (False, True):
                reflect = np.diag((-1, 1)) if crystal_reflection else np.eye(2)
                for index in range(3):  # A global sign is invisible to cosine squared.
                    transform = rotation(index * np.pi / 3) @ reflect
                    reciprocal = modes @ np.linalg.inv(AUTHOR_BASIS) @ transform / a
                    observed = scale * np.cos(np.pi * (sampled @ reciprocal.T))**2 + offset
                    loss = float(np.sqrt(np.nanmean((observed - truth)**2)))
                    if best is None or loss < best[0]:
                        best = (loss, turns, spatial_reflection, reciprocal, index, crystal_reflection)
    return best


def run_case(archive, scan, settings):
    raw, centers, metadata, sources = load_registry(archive, scan)
    measured, fitted, modes, scale, offset, fit_basis, fit_rmse, fit_hash = load_detector(archive, scan)
    if raw.shape[:2] != fitted.shape[:2]:
        raise ValueError("The unwrapped and detector maps use different raster shapes")
    a = float(metadata["LatticeConstant"])
    pixel = float(metadata["PixelSize"])
    alignment, turns, spatial_reflection, reciprocal, crystal_index, crystal_reflection = match_archive_orientation(
        raw, fitted, a, modes, scale, offset)
    result = {
        "alignment_rmse": alignment, "archived_fit_calibration_rmse": fit_rmse, "fit_basis": fit_basis,
        "spatial_quarter_turns": turns, "spatial_reflection": spatial_reflection,
        "crystal_rotation_60degree_steps": crystal_index, "crystal_reflection": crystal_reflection,
        "disk_modes": modes.tolist(), "detector_amplitudes": scale.tolist(),
        "source_sha256": {**sources, "fit_sha256": fit_hash},
    }
    if alignment > settings["max_alignment_rmse"]:
        return {**result, "scoreable": False}
    basis = a * np.array(((1, 0.5), (0, np.sqrt(3) / 2)))
    fit, geometry = fit_center_geometry(raw, centers, basis, pixel, settings["center_inlier_nm"])
    gradient = fit[:2].T @ rotation(-np.deg2rad(float(metadata["SampleRotation"])))
    yy, xx = np.indices(raw.shape[:2], dtype=float) * pixel
    rigid = np.stack((xx, yy), axis=-1) @ fit[:2] + fit[-1]
    cell = np.linalg.solve(gradient, basis)
    n = max(64, 2**int(np.ceil(np.log2(np.max(np.linalg.norm(cell, axis=0)) / 0.3))))
    if n > 512:
        raise ValueError(f"The fitted cell would need a grid beyond this job's reservation: {n}")
    interface = ("MoS2", "MoS2", metadata["Orientation"].upper())
    if interface[-1] == "P":
        fields, _ = disregistry_predictions(gradient, basis, rigid, interface, n, 0, checkpoint_guard,
                                           include_linear_response=True)
    else:
        phase = np.array((0, a / np.sqrt(3)))
        simulation, _ = disregistry_predictions(-gradient, basis, phase - rigid, interface, n, 0, checkpoint_guard,
                                                include_linear_response=True)
        mirror, _ = disregistry_predictions(gradient, basis, phase + rigid, interface, n, 0, checkpoint_guard)
        fields = {name: phase - field for name, field in simulation.items()}
        fields["inverted_gauge"] = mirror["full"] - phase
    valid = apply_spatial(np.isfinite(raw).all(axis=-1), turns, spatial_reflection)
    valid[:4] = valid[-4:] = valid[:, :4] = valid[:, -4:] = False
    if not np.any(valid):
        raise ValueError("No finite registered detector interior")
    sigma = settings["probe_fwhm_nm"] / (np.sqrt(8 * np.log(2)) * pixel)

    def predict(field, blur_sigma):
        aligned = apply_spatial(field, turns, spatial_reflection)
        values = scale * np.cos(np.pi * (aligned @ reciprocal.T))**2 + offset
        return ndi.gaussian_filter(values, (blur_sigma, blur_sigma, 0), mode="nearest") if blur_sigma else values

    def score(field):
        return float(np.sqrt(np.mean((field[valid] - measured[valid])**2)))

    result.update({
        "scoreable": True, "pixels": int(valid.sum()), "geometry": geometry, "grid_size": n,
        "archived_fit_rmse": score(fitted), "unwrapped_experiment_rmse": score(predict(raw, 0)),
        "frozen_model_rmse": {name: score(predict(field, sigma)) for name, field in fields.items()},
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch", choices=("development", "held_out"), required=True)
    args = parser.parse_args()
    protocol = PROJECT / "detector_protocol_v01.json"
    settings = json.loads(protocol.read_text())
    cases = settings["batches"][args.batch]
    result = {"batch": args.batch, "scans": {}, "failed_scans": {},
              "sha256": {path.name: sha256(path.read_bytes()).hexdigest() for path in
                         (protocol, Path(__file__), PROJECT / "bilayer_relaxation.py",
                          PROJECT / "run_registry_prediction.py")}}
    for scan in cases:
        checkpoint_guard()
        try:
            row = run_case(args.archive, scan, settings)
        except (ConvergenceError, ValueError, np.linalg.LinAlgError) as error:
            result["failed_scans"][scan] = str(error)
            print(scan, str(error), flush=True)
        else:
            result["scans"][scan] = row
            print(scan, row.get("frozen_model_rmse", "detector alignment failed"), flush=True)
        checkpoint_guard()
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
