"""Measure recovery of a known differential strain in the DS31 raw readout."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from author_detector_replay import read_detector_fit
from author_release import load_author_release
from run_detector_prediction import AUTHOR_BASIS
from run_heterobilayer_raw_ds31 import (
    COORDINATES, FIRST_COLUMN, ROWS_PER_CHUNK, SCAN, WINDOW, fit_patch,
    masks_from_pair, reciprocal_matrix,
)
from run_heterobilayer_raw_ring_controls import higher_orders
from run_heterobilayer_transfer import PROJECT, digest
from shared_compute import atomic_json, checkpoint_guard


RAW = PROJECT / "heterobilayer_raw_ds31_v01.json"
HIGH = PROJECT / "heterobilayer_raw_ring_controls_v02.json"
FACTORS = (0.5, 1.0, -1.0)


def paired_matrices(indices, nominal):
    modes = indices @ np.linalg.inv(AUTHOR_BASIS)
    modes /= np.median(np.linalg.norm(modes, axis=1))
    _, reference, _ = reciprocal_matrix(nominal, modes)
    return modes, reference


def real_to_detector(theta, reference, local_trace):
    """Change each layer's isotropic strain by half the stated difference."""
    reciprocal = reference + theta[2:10].reshape(2, 2, 2)
    real = np.linalg.inv(reciprocal @ np.linalg.inv(reference)).swapaxes(-1, -2)
    sign = np.array((1.0, -1.0))
    strained = real + (local_trace / 200.0 * sign)[:, None, None] * np.eye(2)
    shifted = np.linalg.inv(strained).swapaxes(-1, -2) @ reference
    new = theta.copy()
    new[2:10] = (shifted - reference).ravel()
    return new


def per_frame_response(observed, design):
    """Estimate each frame's four intensity coefficients at fixed geometry."""
    weight = 1.0 / np.maximum(observed, 0.6)
    transpose = design.swapaxes(-2, -1) * weight[..., None, :]
    return np.linalg.solve(transpose @ design + 1e-7 * np.eye(4),
                           transpose @ observed[..., None])[..., 0]


def score_injection(counts, nominal, indices, radius, theta, bound, block_signal):
    centers = np.rint(nominal).astype(int)
    coordinates = COORDINATES[None] + centers[:, None, None] - nominal[:, None, None]
    modes, reference = paired_matrices(indices, nominal)
    original, no_injection = fit_patch(counts.mean(axis=0), indices, nominal, centers, radius, theta,
                                       matrix_bound=bound)
    design = masks_from_pair(original, modes, coordinates.reshape(len(modes), -1, 2), radius)
    frames = counts.reshape(len(counts), len(modes), -1).astype(float)
    roi = (np.linalg.norm(COORDINATES, axis=-1) < WINDOW - 1).ravel()
    strengths = per_frame_response(frames[..., roi], design[:, roi])
    before = np.maximum(np.einsum("dpk,ndk->ndp", design, strengths), 1e-4)
    before_model = before.mean(axis=0).reshape(counts.shape[1:])
    _, clean_base = fit_patch(before_model, indices, nominal, centers, radius, original, matrix_bound=bound)
    background = np.maximum(strengths[..., 0], 1e-4)[..., None]
    before_signal = np.maximum(before - background, 0.0)
    rows = {}
    for factor in FACTORS:
        shifts = factor * block_signal
        adjusted = []
        for intensity, shift in zip(strengths, shifts):
            parameters = real_to_detector(original, reference, shift)
            shifted = masks_from_pair(parameters, modes, coordinates.reshape(len(modes), -1, 2), radius)
            adjusted.append(np.einsum("dpk,dk->dp", shifted, intensity))
        after = np.maximum(np.asarray(adjusted), 1e-4)
        after_signal = np.maximum(after - background, 0.0)
        loss = np.maximum(before_signal - after_signal, 0.0) / np.maximum(before_signal, 1e-4)
        gain = np.maximum(after_signal - before_signal, 0.0)
        # Where the edge retreats, remove at most the measured intensity
        # above the estimated background. Where it advances, add the
        # predicted signal. Compare this with the noise-free matched model.
        modified = frames - loss * np.maximum(frames - background, 0.0) + gain
        rows[str(factor)] = {}
        for name, values, baseline in (("raw_edge", modified, no_injection),
                                       ("matched_model", after, clean_base)):
            injected = values.mean(axis=0).reshape(counts.shape[1:])
            _, score = fit_patch(injected, indices, nominal, centers, radius, original, matrix_bound=bound)
            rows[str(factor)][name] = {
                "truth_pp": float(shifts.mean()),
                "recovered_pp": score["relative_trace_percent"] - baseline["relative_trace_percent"],
                "converged": bool(score["converged"]),
                "matrix_parameters_at_bound": score["reciprocal_matrix_parameters_at_bound"],
                "negative_average_roi_fraction": float(np.mean(injected.reshape(len(modes), -1)[:, roi] < 0)),
            }
    return no_injection["relative_trace_percent"], rows


def summarize(rows):
    truth = np.asarray([row["truth_pp"] for row in rows])
    recovered = np.asarray([row["recovered_pp"] for row in rows])
    difference = recovered - truth
    gain = float(np.sum(truth * recovered) / np.sum(truth**2))
    return {"gain": gain, "truth_rms_pp": float(np.sqrt(np.mean(truth**2))),
            "error_rms_pp": float(np.sqrt(np.mean(difference**2))),
            "correlation": float(np.corrcoef(truth, recovered)[0, 1]),
            "mean_error_pp": float(np.mean(difference)),
            "converged_blocks": sum(row["converged"] for row in rows),
            "at_bound_blocks": sum(row["matrix_parameters_at_bound"] > 0 for row in rows),
            "mean_negative_average_roi_fraction": float(np.mean([row["negative_average_roi_fraction"] for row in rows])),
            "resolved_at_prespecified_scale": bool(
                0.9 <= gain <= 1.1 and np.sqrt(np.mean(difference**2)) <= 0.05),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("ranges", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    raw = json.loads(RAW.read_text())
    high = json.loads(HIGH.read_text())
    for name, expected in raw["range_sha256"].items():
        checkpoint_guard()
        if digest(args.ranges / name) != expected:
            raise ValueError(f"The raw DS31 data changed in {name}")
    author = load_author_release(args.release, args.dependencies)
    disk, _, _, fit_hash = read_detector_fit(args.archive, SCAN, "fit", author)
    if high["input_sha256"]["DS31_fit.pkl"] != fit_hash:
        raise ValueError("The native reciprocal geometry changed")
    mask = disk._in_use.astype(bool)
    positions = np.column_stack((disk._xvec[mask], disk._yvec[mask]))
    modes = disk.clean_normgset(sanity_plot=False)
    indices, nominal = higher_orders(modes, positions)
    centers = np.rint(nominal).astype(int)
    if not np.allclose(nominal, high["higher_order_nominal_centers_row_col"]):
        raise ValueError("The higher-order diffraction centers changed")
    square = indices[:, 0]**2 + indices[:, 0]*indices[:, 1] + indices[:, 1]**2
    groups = {"higher_both": np.ones(len(indices), dtype=bool), "higher_order_4": square == 4,
              "higher_order_9": square == 9}
    baseline = {(b["row"], b["column_start"]): b for b in high["blocks"]}
    radius = 1.71 / raw["source"]["detector_spacing_mrad"]
    trace = np.array([b["matched_model_dilations_percent"]["warped_relaxed"] for b in raw["blocks"]])
    if [(b["row"], b["column_start"]) for b in raw["blocks"]] != list(baseline):
        raise ValueError("The two raw readouts use different model blocks")
    block_trace = trace - trace.mean()
    row_ids = raw["source"]["sample_rows"]
    trace_by_row = block_trace.reshape(len(row_ids), -1)
    result = {
        "scope": "Synthetic injection using public DS31 raw diffraction patches. Move paired disk edges isotropically at the fixed-model centered trace before the same block averaging and fit. Read a half-amplitude, full-amplitude and sign-inverted perturbation. Vary per-frame diffraction intensities and known strain within each block. Compare a nonnegative edge change applied to measured intensities with the same change applied to fitted synthetic patterns.",
        "decision": "A slope between 0.9 and 1.1 and RMS error under 0.05 percentage points qualifies recovery at the injected amplitude. This readout sensitivity does not establish a quantitative relationship between disk appearance and true local strain in the microscope.",
        "input_sha256": {p.name: digest(p) for p in (Path(__file__), RAW, HIGH, PROJECT / "run_heterobilayer_raw_ds31.py")},
        "archive_fit_sha256": fit_hash, "rows": [],
    }
    for i, row in enumerate(row_ids):
        checkpoint_guard()
        path = args.ranges / f"row-{row}.bin"
        frames = np.memmap(path, mode="r", dtype=np.uint8, shape=(ROWS_PER_CHUNK, 864, 864))
        counts = np.stack([frames[:, y-WINDOW:y+WINDOW+1, x-WINDOW:x+WINDOW+1]
                           for y, x in centers], axis=1)
        # Interpolate the block means to the ten sampled scan positions. This
        # adds a known sub-block strain gradient before averaging. Recenter
        # each group so its true mean equals the archived model's trace.
        x = np.arange(ROWS_PER_CHUNK)
        interpolated = np.interp(x, np.arange(4)*10 + 4.5, trace_by_row[i]).reshape(4, 10)
        interpolated += trace_by_row[i, :, None] - interpolated.mean(axis=1, keepdims=True)
        for j, block_signal in enumerate(interpolated):
            checkpoint_guard()
            column = FIRST_COLUMN + j*10
            b = baseline[(row, column)]
            row_result = {"row": row, "column_start": column, "sub_block_trace_span_pp": float(np.ptp(block_signal))}
            for label, selected in groups.items():
                previous = b[label]["mean"]
                trace_now, injections = score_injection(
                    counts[j*10:j*10+10, selected], nominal[selected], indices[selected], radius,
                    np.asarray(previous["geometry_fit_parameters"]), previous["reciprocal_matrix_bound_pixels"],
                    block_signal)
                row_result[label] = {"baseline_replay_difference_pp": trace_now - previous["relative_trace_percent"],
                                     "injections": injections}
            result["rows"].append(row_result)
        print(row, len(result["rows"]), flush=True)
    result["summary"] = {
        label: {str(factor): {mode: summarize([r[label]["injections"][str(factor)][mode] for r in result["rows"]])
                             for mode in ("raw_edge", "matched_model")} for factor in FACTORS}
        for label in groups}
    result["wall_seconds"] = perf_counter() - start
    checkpoint_guard()
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
