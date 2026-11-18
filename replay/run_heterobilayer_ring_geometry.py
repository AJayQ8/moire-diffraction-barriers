"""Compare diffraction readouts and center-based local geometry on the capping devices."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter

from author_detector_replay import detector_intensity, rms
from author_release import RELEASE_COMMIT, load_author_release
from bilayer_relaxation import sample_disregistry
from observe_bilayers import _good_interior_triangle_mask, _inside_polygon_rc, load_dataset
from run_detector_prediction import apply_spatial
from run_detector_wall_profiles import to_scan_frame
from run_heterobilayer_corrugation import BASIS, MODEL_A, ORIGIN, dilation
from run_heterobilayer_diffraction_check import read_native_detector, register_frame
from run_heterobilayer_residual_transfer import rms_vector
from run_heterobilayer_transfer import PROJECT, PROTOCOL, blur, digest, fingerprints, solve, verify_freeze
from shared_compute import atomic_json, checkpoint_guard


FREEZE = PROJECT / "heterobilayer_transfer_freeze_v01.json"
SETTINGS = PROJECT / "registry_protocol_v01.json"
DETECTOR = PROJECT / "heterobilayer_diffraction_v02.json"


def invert_local(target, seed_nm, reciprocal, coefficients):
    """Re-fit on the local periodic branch around the archived field."""
    field = seed_nm.copy()
    scale, odd, _ = coefficients.T
    for _ in range(20):
        phase = np.pi * (field @ reciprocal.T)
        residual = detector_intensity(field, reciprocal, coefficients) - target
        derivatives = -scale * np.sin(2 * phase) + odd * np.cos(2 * phase)
        jacobian = np.pi * derivatives[..., None] * reciprocal
        transposed = jacobian.swapaxes(-2, -1)
        normal = transposed @ jacobian
        damping = 1e-4 + 1e-3 * np.trace(normal, axis1=-2, axis2=-1)
        step = np.linalg.solve(normal + damping[..., None, None] * np.eye(2),
                               (transposed @ residual[..., None]))[..., 0]
        step *= np.minimum(1, 0.015 / np.maximum(np.linalg.norm(step, axis=-1), 1e-12))[..., None]
        best = np.sum(residual**2, axis=-1)
        next_field = field
        for fraction in (1, 0.5, 0.25):
            candidate = field - fraction * step
            difference = candidate - seed_nm
            # Preserve the branch selected before splitting the diffraction disks.
            candidate = seed_nm + difference * np.minimum(
                1, 0.085 / np.maximum(np.linalg.norm(difference, axis=-1), 1e-12))[..., None]
            error = np.sum((detector_intensity(candidate, reciprocal, coefficients) - target)**2, axis=-1)
            improve = error < best
            next_field = np.where(improve[..., None], candidate, next_field)
            best = np.minimum(best, error)
        if np.max(np.abs(next_field - field)) < 1e-7:
            break
        field = next_field
    return field


def smooth_geometry(data, fit, points):
    """Use only the centers and lattice translations used by the transfer test."""
    centers = data.centers_rc_px[:, ::-1] * data.pixel_nm
    a = data.lattice_nm
    basis = BASIS * (a / MODEL_A)
    iy, ix = np.rint(data.centers_rc_px).astype(int).T
    measured = data.u_nm[iy, ix]
    finite = np.isfinite(measured).all(axis=-1)
    nominal = centers @ fit[:2] + fit[-1]
    translation = np.rint(measured[finite] @ np.linalg.inv(basis).T) @ basis.T
    residual = translation - nominal[finite]
    same_branch = np.linalg.norm(residual, axis=-1) < 0.065
    centers = centers[finite][same_branch]
    residual = residual[same_branch]
    if len(centers) < 10:
        raise ValueError("The local geometry diagnostic needs ten consistent centers")

    def interpolate(source, offsets):
        return RBFInterpolator(source, offsets, kernel="thin_plate_spline")

    loo = np.empty_like(residual)
    for i in range(len(centers)):
        training = np.arange(len(centers)) != i
        loo[i] = interpolate(centers[training], residual[training])(centers[i:i+1])[0] - residual[i]
    correction = interpolate(centers, residual)(points.reshape(-1, 2)).reshape(points.shape)
    return correction, {
        "centers": len(centers), "affine_center_rmse_nm": rms_vector(residual),
        "leave_one_center_out_rmse_nm": rms_vector(loo),
        "max_leave_one_center_out_error_nm": float(np.linalg.norm(loo, axis=-1).max()),
        "whole_map_warp_rms_nm": rms_vector(correction.reshape(-1, 2)),
    }


def mean_rows(rows):
    if isinstance(rows[0], dict):
        return {key: mean_rows([row[key] for row in rows]) for key in rows[0]}
    return float(np.mean(rows))


def run_scan(data, prior, amplitude, settings, interface, measured, fitted, modes, coefficients):
    crop, turns, reflect, transform, alignment = register_frame(data, fitted, modes, coefficients)
    measured, fitted = measured[crop], fitted[crop]
    reciprocal = modes @ transform / data.lattice_nm
    sigma = settings["smoothing_sigma_pixels"]
    probe_nm = settings["probe_fwhm_nm"] / np.sqrt(8 * np.log(2))
    probe_pixels = probe_nm / data.pixel_nm
    shape = data.u_nm.shape[:2]
    yy, xx = np.indices(shape, dtype=float)
    points = np.stack((xx, yy), axis=-1) * data.pixel_nm
    mask, triangles = _good_interior_triangle_mask(data)
    polygons = np.stack([_inside_polygon_rc(shape, data.centers_rc_px[list(tri)]) for tri in triangles])
    valid = data.source_valid_mask
    weights = gaussian_filter(valid.astype(float), sigma, mode="constant")
    support = mask & valid & (weights > 0.97)
    support[:4] = support[-4:] = support[:, :4] = support[:, -4:] = False
    distance2 = np.full(shape, np.inf)
    for y, x in data.centers_rc_px:
        distance2 = np.minimum(distance2, (yy - y)**2 + (xx - x)**2)
    support &= distance2 > (2 * sigma)**2
    whole_fit = np.asarray(prior["whole_geometry"]["fit_scan"])
    whole_rigid = points @ whole_fit[:2] + whole_fit[-1]
    warp, geometry = smooth_geometry(data, whole_fit, points)
    warped_rigid = whole_rigid + warp
    scale = MODEL_A / data.lattice_nm
    gradient = np.asarray(prior["whole_geometry"]["gradient_crystal"])
    grid, shift, _ = solve(scale * gradient, BASIS, interface, prior["grid"])

    def predicted_shift(rigid, cell, probe):
        phase = ORIGIN + scale * rigid
        return (sample_disregistry(grid, shift, phase, cell, probe) - phase) / scale

    def forward(field, probe):
        oriented = apply_spatial(field, turns, reflect)
        signal = detector_intensity(oriented, reciprocal, coefficients)
        return gaussian_filter(signal, (probe_pixels, probe_pixels, 0), mode="nearest") if probe else signal

    warped_shift = predicted_shift(warped_rigid, grid.cell, 0)
    warped_shift_blurred = predicted_shift(warped_rigid, grid.cell, probe_nm)
    signals = {
        "warped_rigid": forward(warped_rigid, True),
        "warped_relaxed": forward(warped_rigid + warped_shift, True),
        "warped_scaled": forward(warped_rigid + amplitude * warped_shift, True),
    }
    h, k = modes.T
    ring_index = np.rint(h**2 + h*k + k**2).astype(int)
    rings = {"first": ring_index == 1, "second": ring_index == 3}
    if any(channels.sum() != 6 for channels in rings.values()):
        raise ValueError("Expected two complete rings of diffraction disks")
    seed = apply_spatial(data.u_nm, turns, reflect)
    finite_seed = np.isfinite(seed).all(axis=-1)
    seed = np.where(finite_seed[..., None], seed, 0)
    oracle = forward(warped_rigid + amplitude * warped_shift, False)
    synthetic_seed = apply_spatial(warped_rigid + amplitude * warped_shift, turns, reflect) + (0.006, -0.006)
    support_detector = apply_spatial(support, turns, reflect)
    ring_fields = {}
    ring_metrics = {}
    for ring, channels in {**rings, "both": np.ones(len(modes), dtype=bool)}.items():
        checkpoint_guard()
        q, c = reciprocal[channels], coefficients[channels]
        recovered = invert_local(measured[..., channels], seed, q, c)
        recovered_synthetic = invert_local(oracle[..., channels], synthetic_seed, q, c)
        recovered_error = recovered_synthetic - (synthetic_seed - (0.006, -0.006))
        ring_metrics[ring] = {
            "model_oracle_branch_recovery_rms_nm": rms_vector(recovered_error[support_detector]),
            "fraction_at_branch_limit": float(np.mean(np.linalg.norm(
                recovered[support_detector] - seed[support_detector], axis=-1) >= 0.085 - 1e-6)),
            "detector_rmse": {key: rms(measured[support_detector][:, select],
                                       detector_intensity(recovered[support_detector], reciprocal[select], coefficients[select]))
                              for key, select in rings.items()},
            "archived_native_fit_detector_rmse": {key: rms(measured[support_detector][:, select],
                                                            fitted[support_detector][:, select])
                                                   for key, select in rings.items()},
        }
        ring_fields[ring] = to_scan_frame(recovered, turns, reflect)
    smooth_ring_difference = blur(ring_fields["first"] - ring_fields["second"], valid, weights, sigma)
    smooth_readout_change = blur(ring_fields["both"] - data.u_nm, valid, weights, sigma)
    halves = []
    for axis, partition in prior["partitions"].items():
        regions = [polygons[row["triangles"]].any(axis=0) for row in partition]
        for i, half in enumerate(partition):
            fit = np.asarray(half["geometry"]["fit_scan"])
            rigid = points @ fit[:2] + fit[-1]
            cell = np.asarray(half["geometry"]["cell_nm"])
            relaxation = predicted_shift(rigid, cell, 0)
            outputs = {**signals, "regional_rigid": forward(rigid, True),
                       "regional_relaxed": forward(rigid + relaxation, True),
                       "regional_scaled": forward(rigid + amplitude * relaxation, True)}
            select = support & regions[i] & ~regions[1 - i]
            detector_select = apply_spatial(select, turns, reflect)
            detector_rmse = {key: {name: rms(measured[detector_select][:, channels],
                                           prediction[detector_select][:, channels])
                                   for name, prediction in outputs.items()}
                             for key, channels in {**rings, "both": np.ones(len(modes), dtype=bool)}.items()}
            regional = rigid + amplitude * predicted_shift(rigid, cell, probe_nm)
            disregistry = {
                "regional_scaled": regional, "regional_rigid": rigid, "warped_rigid": warped_rigid,
                "warped_relaxed": warped_rigid + warped_shift_blurred,
                "warped_scaled": warped_rigid + amplitude * warped_shift_blurred,
            }
            residuals = {name: blur(field - regional, valid, weights, sigma)
                         for name, field in {**ring_fields, "archive": data.u_nm}.items()}
            map_rmse = {name: {key: rms_vector(blur(field - measured_field, valid, weights, sigma)[select])
                               for key, field in disregistry.items()}
                        for name, measured_field in {**ring_fields, "archive": data.u_nm}.items()}
            centered = [residuals[key][select] - residuals[key][select].mean(axis=0) for key in rings]
            correlation = float(np.sum(centered[0] * centered[1]) /
                                (np.linalg.norm(centered[0]) * np.linalg.norm(centered[1])))
            halves.append({
                "axis": axis, "half": i, "pixels": int(detector_select.sum()),
                "detector_rmse": detector_rmse, "map_rmse_nm": map_rmse,
                "regional_model_residual_correlation_between_rings": correlation,
                "between_rings_shift_rms_nm": rms_vector(smooth_ring_difference[select]),
                "between_rings_dilation_rms_pp": float(np.sqrt(np.mean(dilation(smooth_ring_difference, data)[select]**2))),
                "native_refit_vs_archive_shift_rms_nm": rms_vector(smooth_readout_change[select]),
            })
    fields = ("detector_rmse", "map_rmse_nm", "regional_model_residual_correlation_between_rings",
              "between_rings_shift_rms_nm", "between_rings_dilation_rms_pp", "native_refit_vs_archive_shift_rms_nm")
    return {"encapsulation": prior["encapsulation"], "alignment": alignment, "geometry": geometry,
            "ring_metrics": ring_metrics, "halves": halves,
            "mean_over_halves": {key: mean_rows([row[key] for row in halves]) for key in fields}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    author = load_author_release(args.release, args.dependencies)
    protocol = json.loads(PROTOCOL.read_text())
    settings = json.loads(SETTINGS.read_text())
    verify_freeze(FREEZE, fingerprints(), protocol["devices"]["development"])
    amplitude_path = PROJECT / "heterobilayer_capping_development_v01.json"
    amplitudes = json.loads(amplitude_path.read_text())["fitted_amplitudes"]["by_encapsulation"]
    detector = json.loads(DETECTOR.read_text())
    result = {
        "scope": "Retrospective on the two capping devices. Each ring is fit separately from the same archived branch and with the original coefficients. Both rings can share acquisition and fit biases. Center warp uses the measured center geometry and no fitted material parameters. Evaluate on the original regional detector support.",
        "author_commit": RELEASE_COMMIT,
        "input_sha256": {p.name: digest(p) for p in (Path(__file__), DETECTOR, SETTINGS, FREEZE, amplitude_path)},
        "scans": {},
    }
    for batch, cases in protocol["devices"].items():
        previous = PROJECT / f"heterobilayer_transfer_{batch}_v01.json"
        if digest(previous) != detector["input_sha256"][previous.name]:
            raise ValueError("The existing diffraction test used a different transfer run")
        result["input_sha256"][previous.name] = digest(previous)
        rows = json.loads(previous.read_text())["scans"]
        for scan in cases:
            checkpoint_guard()
            data = load_dataset(scan, args.archive, allowed_ids=cases, allow_author_crop=True)
            prior = rows[scan]
            if data.archive_md5 != prior["archive_md5"]:
                raise ValueError(f"{scan}: changed source archive")
            measured, fitted, modes, coefficients, calibration = read_native_detector(args.archive, scan, author)
            reference = detector["scans"][scan]
            if calibration != reference["calibration"]:
                raise ValueError(f"{scan}: changed detector calibration")
            row = run_scan(data, prior, amplitudes[prior["encapsulation"]], settings, tuple(protocol["interface"]),
                           measured, fitted, modes, coefficients)
            for half, old in zip(row["halves"], reference["halves"]):
                if half["pixels"] != old["pixel_count"]:
                    raise ValueError(f"{scan}: changed primary detector mask")
                if not np.isclose(half["detector_rmse"]["both"]["regional_scaled"], old["rmse"]["regional_scaled"],
                                  rtol=0, atol=1e-10):
                    raise ValueError(f"{scan}: changed prior detector score")
            result["scans"][scan] = {"batch": batch, **row}
            result["wall_seconds"] = perf_counter() - start
            checkpoint_guard()
            atomic_json(args.output, result)
            print(scan, row["mean_over_halves"]["detector_rmse"]["both"], flush=True)


if __name__ == "__main__":
    main()
