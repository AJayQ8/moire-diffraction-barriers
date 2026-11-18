"""Check whether DS31 interference and spot positions favor the same motion."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter

from author_detector_replay import read_detector_fit, rms
from author_release import load_author_release
from bilayer_relaxation import sample_disregistry
from observe_bilayers import load_dataset
from run_detector_prediction import AUTHOR_BASIS
from run_heterobilayer_corrugation import BASIS, MODEL_A, ORIGIN, dilation
from run_heterobilayer_raw_ds31 import COORDINATES, FIRST_COLUMN, SCAN, WINDOW, fit_patch, load_counts, masks_from_pair
from run_heterobilayer_raw_injection import HIGH, RAW, paired_matrices, per_frame_response
from run_heterobilayer_ring_geometry import smooth_geometry
from run_heterobilayer_strain_observer import ORIGINAL_ARCS, SETTINGS, SPATIAL, TRANSFER, use_blocks
from run_heterobilayer_transfer import PROJECT, PROTOCOL, digest, solve
from run_moire_elastic_components import common_rings, split_displacement
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


BENCHMARK = PROJECT / "moire_regime_benchmark_v02.json"
MIN_LAYER_COUNTS = 0.10
BLOCK = 10


def interference_samples(raw_rows, centers, modes, nominal, radii, source):
    """Separate overlap contrast from fitted single-layer count contributions."""
    radius = 1.71 / source["source"]["detector_spacing_mrad"]
    vectors, _ = paired_matrices(modes, nominal)
    theta = np.asarray(source["aggregate"]["both"]["geometry_fit_parameters"])
    coordinates = COORDINATES[None] + centers[:, None, None] - nominal[:, None, None]
    flat = coordinates.reshape(len(modes), -1, 2)
    roi = (np.linalg.norm(COORDINATES, axis=-1) < WINDOW-1).ravel()
    integration = np.linalg.norm(coordinates, axis=-1) < radii[:, None, None]
    fixed_mask = masks_from_pair(theta, vectors, flat[:, roi], radius)
    blocks = {}
    for raw in raw_rows:
        for start in range(0, len(raw["counts"]), BLOCK):
            checkpoint_guard()
            frames = raw["counts"][start:start+BLOCK]
            fitted, geometry = fit_patch(frames.mean(axis=0), modes, nominal, centers, radius, theta)
            moving_mask = masks_from_pair(fitted, vectors, flat[:, roi], radius)
            counts = frames.reshape(len(frames), len(modes), -1)[..., roi].astype(float)
            result = {"integrated": np.sum(frames*integration, axis=(-1, -2)), "geometry": {
                "trace_percent": geometry["relative_trace_percent"], "fit_success": geometry["converged"],
                "matrix_parameters_at_bound": geometry["reciprocal_matrix_parameters_at_bound"]}}
            for name, design in (("fixed_disk_geometry", fixed_mask), ("local_disk_geometry", moving_mask)):
                coefficients = per_frame_response(counts, design)
                _, first, second, overlap = np.moveaxis(coefficients, -1, 0)
                valid = (first > MIN_LAYER_COUNTS) & (second > MIN_LAYER_COUNTS)
                contrast = (overlap-first-second)/(2*np.sqrt(np.maximum(first, MIN_LAYER_COUNTS) *
                                                            np.maximum(second, MIN_LAYER_COUNTS)))
                result[name] = np.where(valid, contrast, np.nan)
            blocks[(raw["row"], FIRST_COLUMN+start)] = result
    return blocks


def sampled_blocks(rows, field, origin):
    points = []
    first_row, first_col = origin
    for row in rows:
        r = row["row"] - first_row
        c = row["column_start"] - first_col
        points.append(field[r, c:c+row["column_count"]])
    values = np.concatenate(points, axis=0)
    if len(values) != BLOCK*len(rows) or not np.isfinite(values).all():
        raise ValueError("A sampled field differs from the common DS31 support")
    return values


def calibrate(rows, response, field, stored_modes):
    targets = np.concatenate([r[response] for r in rows])
    phase = 2*np.pi*(field @ stored_modes.T)
    basis = np.stack((np.cos(phase), np.sin(phase), np.ones_like(phase)), axis=-1)
    coeff, reports = [], []
    for i in range(len(stored_modes)):
        use = np.isfinite(targets[:, i])
        if use.sum() < 0.8*len(targets):
            raise ValueError(f"Insufficient contrast support for channel {i}")
        x, y = basis[use, i], targets[use, i]
        estimate, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
        span = np.ptp(y)
        if rank != 3 or singular[0] / singular[-1] > 30 or span <= 0:
            raise ValueError(f"The training rows do not calibrate channel {i}")
        coeff.append(estimate)
        reports.append({"pixels": int(use.sum()), "mean": float(y.mean()), "source_range": float(span),
                        "training_normalized_rmse": rms(y/span, (x@estimate)/span)})
    return np.asarray(coeff), reports


def predict_basis(field, reciprocal, probe):
    phase = 2*np.pi*(field @ reciprocal.T)
    return tuple(gaussian_filter(values, (probe, probe, 0), mode="nearest")
                 for values in (np.cos(phase), np.sin(phase)))


def metrics(rows, response, prediction, source, rings):
    observed = np.concatenate([r[response] for r in rows])
    span = np.array([v["source_range"] for v in source])
    residual = (prediction-observed)/span
    result = {}
    for name, mask in rings.items():
        selected = residual[:, mask]
        valid = np.isfinite(selected)
        if not np.any(valid):
            raise ValueError(f"No raw interference support remains on {name}")
        result[name] = {"normalized_rmse": float(np.sqrt(np.mean(selected[valid]**2))),
                        "supported_fraction": float(valid.mean())}
    return result


def strain_scale(rows, background, compression, data, crop):
    """Freeze one compression multiplier using only the original raw rows."""
    base = sampled_blocks(rows, dilation(background, data), crop).reshape(-1, BLOCK).mean(axis=1)
    signal = sampled_blocks(rows, dilation(compression, data), crop).reshape(-1, BLOCK).mean(axis=1)
    measured = np.array([(r["paired"]+r["arc"])/2 for r in rows])
    center = signal-signal.mean()
    if np.linalg.norm(center) < 1e-8:
        raise ValueError("The original rows do not constrain a compression multiplier")
    scale = float(center @ (measured-base)/(center @ center))
    return scale, {"target_trace_rms_pp": float(measured.std()), "background_trace_rms_pp": float(base.std()),
                   "periodic_compression_trace_rms_pp": float(signal.std())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("original_ranges", type=Path)
    parser.add_argument("replay_ranges", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    raw, high, arcs, spatial, settings, transfer, benchmark, protocol = (
        json.loads(p.read_text()) for p in (RAW, HIGH, ORIGINAL_ARCS, SPATIAL, SETTINGS, TRANSFER, BENCHMARK, PROTOCOL))
    support = use_blocks(raw, high, arcs, spatial)
    data = load_dataset(SCAN, args.archive, allowed_ids=(SCAN,), allow_author_crop=True)
    reference, prior = benchmark["scans"][SCAN], transfer["scans"][SCAN]
    if data.archive_md5 != reference["archive_md5"] or data.archive_md5 != prior["archive_md5"]:
        raise ValueError("The DS31 archive changed")
    author = load_author_release(args.release, args.dependencies)
    disk, _, native, fit_hash = read_detector_fit(args.archive, SCAN, "fit", author)
    if fit_hash != raw["input_sha256"]["DS31_fit.pkl"] or fit_hash != reference["detector_calibration"]["fit_sha256"]:
        raise ValueError("The source detector frame changed")
    select = disk._in_use.astype(bool)
    nominal = np.column_stack((disk._xvec[select], disk._yvec[select]))
    radii = disk._rvec[select]
    modes = disk.clean_normgset(sanity_plot=False)
    rings = common_rings(modes)
    fields = []
    for path, receipt in ((args.original_ranges, raw), (args.replay_ranges, spatial)):
        _, centers, raw_rows, checks = load_counts(path, nominal, radii, disk.df_set())
        if checks != receipt["range_sha256"]:
            raise ValueError("One sampled data range changed")
        fields.append(interference_samples(raw_rows, centers, modes, nominal, radii, raw))
    source, replay = ([{**row, **blocks[(row["row"], row["column_start"])]} for row in rows]
                      for rows, blocks in zip(support, fields, strict=True))
    rr, cc = data.metadata["author_unwrap_crop_origin_row_col"]
    height, width = data.u_nm.shape[:2]
    crop = np.s_[rr:rr+height, cc:cc+width]
    native = native[crop] @ np.linalg.inv(AUTHOR_BASIS).T
    if reference["detector_calibration"]["saved_field_basis"] != "cartesian":
        raise ValueError("Expected the calibrated Cartesian source phases")
    frame = reference["detector_alignment"]
    if frame["spatial_turns"] or frame["spatial_reflection"]:
        raise ValueError("This DS31 strain comparison needs an unrotated detector raster")
    mirror = np.diag((-1, 1)) if frame["vector_reflection"] else np.eye(2)
    transform = np.linalg.inv(AUTHOR_BASIS) @ rotation(frame["vector_rotation_60deg_steps"]*np.pi/3) @ mirror
    reciprocal = modes @ transform / data.lattice_nm
    fit = np.asarray(prior["whole_geometry"]["fit_scan"])
    y, x = np.indices(data.u_nm.shape[:2], dtype=float)
    points = np.stack((x, y), axis=-1)*data.pixel_nm
    correction, _ = smooth_geometry(data, fit, points)
    background = points @ fit[:2] + fit[-1] + correction
    scale = MODEL_A/data.lattice_nm
    gradient = np.asarray(prior["whole_geometry"]["gradient_crystal"])*scale
    grid, motion, solver = solve(gradient, BASIS, tuple(protocol["interface"]), prior["grid"])
    components, _ = split_displacement(grid, motion)
    probe = settings["probe_fwhm_nm"]/np.sqrt(8*np.log(2))
    phase = ORIGIN + scale*background

    def transport(displacement, sigma=0):
        return (sample_disregistry(grid, displacement, phase, grid.cell, sigma)-phase)/scale

    rotation_only = transport(components["rotation_only"])
    compression = transport(components["compression_only"])
    scalar, calibration = strain_scale(source, background+transport(components["rotation_only"], probe),
                                       transport(components["compression_only"], probe), data, (rr, cc))
    if abs(scalar) > 2:
        raise ValueError("The source raw readouts give an implausibly large compression scale")
    physical = {"rigid": background, "rotation_only": background+rotation_only,
                "full": background+transport(motion), "raw_scaled_compression": background+rotation_only+scalar*compression}
    predictions = {label: predict_basis(field, reciprocal, probe/data.pixel_nm) for label, field in physical.items()}
    native_phase = 2*np.pi*(native @ modes.T)
    predictions["archived_native_phase"] = (np.cos(native_phase), np.sin(native_phase))
    measurements = ("integrated", "fixed_disk_geometry", "local_disk_geometry")
    source_phase = sampled_blocks(source, native, (rr, cc))
    result = {
        "scope": "DS31 raw intensity and lattice shifts from two sets of rows in the same acquisition. Separate fitted overlap intensity from the two single-layer disk contributions, then calibrate cosine, sine and offset terms for each channel on original rows. Score the raw contrast from the replay rows with fixed coefficients and compare to integrated intensity. The compression multiplier uses only the source rows' outer-disk strain readouts.",
        "limits": "The experimental camera calibration assumes the original fitted phase on training rows. The sample, detector calibration, and microscope acquisition are shared. This comparison probes overlap contrast and projected displacement and leaves absolute 3D strain open.",
        "input_sha256": {p.name: digest(p) for p in (
            Path(__file__), RAW, HIGH, ORIGINAL_ARCS, SPATIAL, TRANSFER, SETTINGS, PROTOCOL, BENCHMARK)},
        "raw_compression_multiplier": scalar, "multiplier_fit": calibration, "solver": solver, "signals": {},
        "disk_geometry_diagnostics": {
            label: {"converged_blocks": sum(r["geometry"]["fit_success"] for r in rows),
                    "blocks_at_bound": sum(r["geometry"]["matrix_parameters_at_bound"] > 0 for r in rows)}
            for label, rows in (("original", source), ("replay", replay))
        },
        "strain_scores": {},
    }
    for label, rows in (("original", source), ("replay", replay)):
        observed = {k: np.array([r[k] for r in rows]) for k in ("paired", "arc")}
        result["strain_scores"][label] = {}
        for name, displacement in (
            ("rigid", np.zeros_like(motion)), ("rotation_only", components["rotation_only"]),
            ("full", motion), ("raw_scaled_compression", components["rotation_only"] + scalar*components["compression_only"]),
        ):
            trace = dilation(background + transport(displacement, probe), data)
            predicted = sampled_blocks(rows, trace, (rr, cc)).reshape(-1, BLOCK).mean(axis=1)
            result["strain_scores"][label][name] = {
                "variability_pp": float(predicted.std()),
                "centered_error_pp": {k: float((predicted-v).std()) for k, v in observed.items()},
            }
    for response in measurements:
        coeff, reports = calibrate(source, response, source_phase, modes)
        result["signals"][response] = {"coefficients": coeff.tolist(), "source_camera_calibration": reports, "cohorts": {}}
        for label, rows in (("original", source), ("replay", replay)):
            scores = {}
            for name, (cos, sin) in predictions.items():
                first = sampled_blocks(rows, cos, (rr, cc))
                second = sampled_blocks(rows, sin, (rr, cc))
                camera = first*coeff[:, 0] + second*coeff[:, 1] + coeff[:, 2]
                scores[name] = metrics(rows, response, camera, reports, rings)
            result["signals"][response]["cohorts"][label] = {
                "blocks": len(rows), "scan_positions": len(rows)*BLOCK, "scores": scores,
            }
        print(response, result["signals"][response]["cohorts"]["replay"]["scores"], flush=True)
    result["wall_seconds"] = perf_counter()-start
    checkpoint_guard()
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
