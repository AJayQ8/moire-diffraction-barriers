"""Rebuild WSe2/MoS2 geometry from first-ring spatial frequencies and phases."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from author_detector_replay import detector_intensity
from author_release import load_author_release
from bilayer_relaxation import ConvergenceError
from observe_bilayers import load_dataset
from run_detector_prediction import AUTHOR_BASIS
from run_heterobilayer_corrugation import BASIS, MODEL_A
from run_heterobilayer_cross_ring import score
from run_heterobilayer_diffraction_check import read_native_detector
from run_heterobilayer_transfer import PROJECT, PROTOCOL, digest
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


METHOD = PROJECT / "heterobilayer_cross_ring_protocol_v01.json"
SETTINGS = PROJECT / "registry_protocol_v01.json"


def circular_error(cycles):
    return (cycles + 0.5) % 1 - 0.5


def spectral_peaks(values, coefficients, pixel_nm):
    """Find the fundamental spatial frequency of each first-ring fringe."""
    shape = values.shape[:2]
    weights = np.outer(np.hanning(shape[0]), np.hanning(shape[1]))
    rr, cc = np.indices(shape, dtype=float)
    xy = np.stack((cc, rr), axis=-1) * pixel_nm
    qy = np.fft.fftfreq(4*shape[0], d=pixel_nm)
    qx = np.fft.fftfreq(4*shape[1], d=pixel_nm)
    ky, kx = np.meshgrid(qy, qx, indexing="ij")
    length = np.hypot(kx, ky)
    select = (length >= 0.06) & (length <= 0.23) & ((ky > 0) | ((ky == 0) & (kx > 0)))
    peaks = []
    for image, coeff in zip(np.moveaxis(values, -1, 0), coefficients):
        contrast = weights * (image - np.average(image, weights=weights))
        spectrum = np.fft.fft2(contrast, s=(len(qy), len(qx)))
        i, j = np.unravel_index(np.argmax(np.where(select, abs(spectrum), -1)), spectrum.shape)
        q = np.array((qx[j], qy[i]))
        amplitude = np.sum(contrast * np.exp(-2j * np.pi * (xy @ q)))
        phase = (np.angle(amplitude) - np.angle(coeff[0] - 1j*coeff[1])) / (2*np.pi)
        peaks.append((q, phase))
    return xy, weights, np.asarray([q for q, _ in peaks]), np.asarray([phase for _, phase in peaks])


def fit_affine_seed(values, modes, coefficients, pixel_nm, lattice_nm, sample_angle):
    """Infer the global geometry without the archive's unwrapped displacement."""
    xy, weights, peaks, phases = spectral_peaks(values, coefficients, pixel_nm)
    reciprocal_basis = modes @ np.linalg.inv(AUTHOR_BASIS) / lattice_nm
    nominal_mismatch = (lattice_nm - 0.315) / lattice_nm
    candidates = []
    x = xy.reshape(-1, 2)
    for mirror in (False, True):
        reflect = np.diag((-1, 1)) if mirror else np.eye(2)
        for turn in range(6):
            transform = rotation(turn*np.pi/3) @ reflect
            q = reciprocal_basis @ transform
            signs = np.where(np.sum(peaks * (q @ rotation(sample_angle)), axis=1) >= 0, 1.0, -1.0)
            gradient = np.linalg.lstsq(q, peaks * signs[:, None], rcond=None)[0]
            peak_disagreement = np.sqrt(np.mean(np.sum((q @ gradient - peaks * signs[:, None])**2, axis=-1)))
            crystal = gradient @ rotation(-sample_angle)
            distance = np.linalg.norm(crystal - nominal_mismatch * np.eye(2)) / np.sqrt(2)
            if np.linalg.det(crystal) <= 0 or distance > 0.8 * nominal_mismatch or peak_disagreement > 0.025:
                continue
            candidate = {"spatial_turns": 0, "spatial_reflection": False,
                         "vector_rotation_60deg_steps": turn, "vector_reflection": mirror}
            candidates.append((candidate, q, gradient, signs, peak_disagreement))
    if not candidates:
        raise ValueError("No first-ring lattice is consistent with positive WSe2/MoS2 mismatch")
    xy_center = np.average(x, axis=0, weights=weights.ravel())
    base = BASIS * lattice_nm / MODEL_A
    phase_grid = np.stack(np.meshgrid(np.arange(21) / 21, np.arange(21) / 21, indexing="ij"), axis=-1).reshape(-1, 2)
    offsets = phase_grid @ base.T
    coarse = []
    for frame, q, gradient, signs, peak_disagreement in candidates:
        mismatch = circular_error(offsets @ q.T - phases[None] * signs)
        error = np.mean(mismatch**2, axis=1)
        index = int(np.argmin(error))
        # Exact lattice symmetries have equal first-ring phase loss. Resolve
        # that ambiguity by the metadata crystal orientation and expected
        # small twist, without comparing the second-ring outcomes.
        orientation_error = float(np.linalg.norm(gradient @ rotation(-sample_angle) - nominal_mismatch*np.eye(2)))
        coarse.append((float(error[index]), orientation_error, frame, q, gradient, offsets[index], peak_disagreement))
    best_phase = min(row[0] for row in coarse)
    tied = [row for row in coarse if row[0] <= best_phase + 1e-8]
    sample = np.s_[::2, ::2]
    points = xy[sample].reshape(-1, 2) - xy_center
    target = values[sample].reshape(-1, values.shape[-1])
    weight = np.sqrt(weights[sample].reshape(-1, 1))
    coeff, odd, _ = coefficients.T

    def refine(q, gradient, offset):
        start = np.concatenate((gradient.ravel(), offset + gradient @ xy_center))

        def prediction(parameters):
            g = parameters[:4].reshape(2, 2)
            b = points @ g.T + parameters[4:]
            phase = np.pi * (b @ q.T)
            return detector_intensity(b, q, coefficients), phase

        def residual(parameters):
            values, _ = prediction(parameters)
            return ((values - target) * weight).ravel()

        def jacobian(parameters):
            _, phase = prediction(parameters)
            derivative = np.pi * (-coeff * np.sin(2*phase) + odd * np.cos(2*phase)) * weight
            matrix = np.einsum("ij,ja,ib->ijab", derivative, q, points).reshape(-1, 4)
            translation = (derivative[..., None] * q).reshape(-1, 2)
            return np.hstack((matrix, translation))

        return least_squares(residual, start, jac=jacobian, x_scale="jac", max_nfev=80, ftol=1e-6, gtol=1e-7)

    solved = []
    for _, orientation_error, frame, q, gradient, offset, disagreement in tied:
        result = refine(q, gradient, offset)
        if result.success:
            error = float(np.sqrt(np.mean(result.fun**2) / np.mean(weight**2)))
            solved.append((error, orientation_error, frame, gradient, disagreement, result))
    if not solved:
        raise ValueError("None of the first-ring phase-equivalent frames converged")
    best_error = min(row[0] for row in solved)
    equivalent = [row for row in solved if row[0] <= best_error + 1e-5]
    first_rmse, _, frame, gradient, peak_disagreement, fit = min(equivalent, key=lambda row: row[1])
    g = fit.x[:4].reshape(2, 2)
    offset = fit.x[4:] - g @ xy_center
    refined = np.vstack((g.T, offset))
    return frame, refined, {
        "frequencies_cycles_per_nm": peaks.tolist(), "initial_gradients": gradient.tolist(),
        "first_ring_fit_rmse": first_rmse,
        "fitted_gradient": g.tolist(), "fitted_offset_nm": offset.tolist(), "candidate_frames": len(candidates),
        "phase_equivalent_frames": len(tied),
        "intensity_equivalent_frames": len(equivalent),
        "peak_frequency_disagreement_cycles_per_nm": float(peak_disagreement),
        "evaluations": int(fit.nfev), "optimality": float(fit.optimality),
    }


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
    method = json.loads(METHOD.read_text())
    settings = json.loads(SETTINGS.read_text())
    interface = tuple(json.loads(PROTOCOL.read_text())["interface"])
    author = load_author_release(args.release, args.dependencies)
    hashes = {p.name: digest(p) for p in (Path(__file__), METHOD, SETTINGS, PROJECT / "run_heterobilayer_cross_ring.py")}
    if args.batch == "replay":
        if args.freeze is None:
            parser.error("The replay requires the first-ring method freeze")
        receipt = json.loads(args.freeze.read_text())
        qualification = args.freeze.parent / receipt["qualification_file"]
        if hashes != receipt["input_sha256"] or digest(qualification) != receipt["qualification_sha256"]:
            raise ValueError("First-ring seed method changed after qualification")
    cases = method[args.batch]
    output = {
        "scope": "Re-estimate the affine seed, lattice orientation and individual stacking-center positions with the first diffraction ring. Archived geometry is excluded. The source instrument calibration, per-channel response coefficients, and crop convention remain inputs. Score the second-ring intensities with fixed coefficients on the shared interior.",
        "batch": args.batch, "input_sha256": hashes, "scans": {}, "failed_scans": {},
    }
    for scan in cases:
        checkpoint_guard()
        data = load_dataset(scan, args.archive, allowed_ids=cases, allow_author_crop=True)
        observed, _, modes, coefficients, calibration = read_native_detector(args.archive, scan, author)
        i, j = data.metadata.get("author_unwrap_crop_origin_row_col", (0, 0))
        y, x = data.u_nm.shape[:2]
        h, k = modes.T
        first = np.rint(h*h + h*k + k*k).astype(int) == 1
        try:
            frame, fit, diagnostics = fit_affine_seed(observed[i:i+y, j:j+x, :][..., first], modes[first],
                                                       coefficients[first], data.pixel_nm, data.lattice_nm,
                                                       np.deg2rad(data.sample_rotation_degrees))
            prediction = score(data, observed, modes, coefficients, frame, fit, settings, interface)
        except (ConvergenceError, ValueError, np.linalg.LinAlgError) as exc:
            output["failed_scans"][scan] = str(exc)
            print(scan, str(exc), flush=True)
        else:
            output["scans"][scan] = {"archive_md5": data.archive_md5, "calibration": calibration,
                                     "first_ring_frame": frame, "first_ring_affine": diagnostics, **prediction}
            print(scan, frame, prediction["mean_over_folds"]["diffraction_rmse"], flush=True)
        output["wall_seconds"] = perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, output)


if __name__ == "__main__":
    main()
