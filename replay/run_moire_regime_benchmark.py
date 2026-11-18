"""Compare the same fixed elasticity approximations across the archive."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize_scalar

from author_detector_replay import detector_intensity, normalize_channels, read_detector_fit, rms
from author_release import RELEASE_COMMIT, load_author_release
from bilayer_relaxation import ConvergenceError, MODES, MONOLAYERS, STACKING, energy_gradient, sample_disregistry
from observe_bilayers import _good_interior_triangle_mask, load_dataset
from run_antiparallel_signed_audit import local_geometry
from run_detector_prediction import AUTHOR_BASIS, apply_spatial
from run_heterobilayer_diffraction_check import register_frame
from run_heterobilayer_residual_transfer import rms_vector
from run_heterobilayer_sparse_geometry import linear_response, solve_from_geometry
from run_heterobilayer_transfer import PROJECT, blur, digest, geometry_from_centers
from shared_compute import atomic_json, checkpoint_guard


SETTINGS = PROJECT / "registry_protocol_v01.json"
SCAN_IDS = tuple(f"DS{i}" for i in range(1, 38) if i not in (17, 18, 27))


def detector_data(archive, scan, author):
    """Read the saved detector response in the format of its release."""
    disk, saved, field, fingerprint = read_detector_fit(archive, scan, "fit", author)
    measured = normalize_channels(disk.df_set(), author)
    modes = disk.clean_normgset(sanity_plot=False)
    if len(saved) != len(modes) or measured.shape[:2] != field.shape[:2] or saved.shape[1] not in (2, 3):
        raise ValueError("The native detector fit has an unsupported format")
    candidates = {}
    for label, lattice in (("lattice", field), ("cartesian", field @ np.linalg.inv(AUTHOR_BASIS).T)):
        coefficients = saved
        if saved.shape[1] == 2:
            # DS13 and DS14 have no binned fit. The native constant can still
            # be recovered from the original fitted coordinates and A, B.
            native = np.column_stack((saved, np.zeros(len(modes))))
            offset = np.mean(measured - detector_intensity(lattice, modes, native), axis=(0, 1))
            coefficients = np.column_stack((saved, offset))
        prediction = detector_intensity(lattice, modes, coefficients)
        candidates[label] = (rms(measured, prediction), prediction, coefficients)
    frame = min(candidates, key=lambda label: candidates[label][0])
    _, prediction, coefficients = candidates[frame]
    diagnostics = {"fit_sha256": fingerprint, "saved_columns": int(saved.shape[1]), "saved_field_basis": frame,
                   "coefficients": coefficients.tolist(),
                   "native_replay_rmse": {label: row[0] for label, row in candidates.items()}}
    return measured, prediction, modes, coefficients, diagnostics


def relaxed_ray(grid, linear, interface):
    """Optimize the amplitude along the fixed first-order displacement field."""
    coefficients = STACKING[interface]

    def energy(fraction):
        return energy_gradient(grid, coefficients, fraction * linear)[0]

    # The energy can have several minima along a large displacement. Scan
    # the interval before refining the best bracket and retain the endpoints.
    maximum_phase = np.max(np.abs(linear @ (2 * np.pi * MODES @ np.linalg.inv(grid.basis)).T))
    divisions = max(24, int(np.ceil(4 * maximum_phase / np.pi)))
    factors = np.linspace(0, 1, divisions + 1)
    energies = np.array([energy(factor) for factor in factors])
    best = int(np.argmin(energies))
    fraction = factors[best]
    bracket = factors[max(0, best-1):min(divisions, best+1)+1]
    fit = minimize_scalar(energy, method="bounded", bounds=(bracket[0], bracket[-1]),
                          options={"xatol": 1e-6})
    if fit.success and fit.fun < energies[best]:
        fraction = fit.x
    return float(fraction)


def score_scan(data, detector, settings):
    measured, fitted, modes, coefficients, calibration = detector
    crop, turns, reflect, transform, alignment = register_frame(data, fitted, modes, coefficients)
    if alignment["archived_field_registration_rmse"] > 0.20:
        raise ValueError("The saved and native fields do not register within the established tolerance")
    measured, fitted = measured[crop], fitted[crop]
    h, k = modes.T
    ring_index = np.rint(h*h + h*k + k*k).astype(int)
    rings = {"first": ring_index == 1, "second": ring_index == 3}
    if any(channels.sum() < 5 for channels in rings.values()):
        raise ValueError("A diffraction ring has fewer than five recorded disks")
    rings["both"] = np.ones(len(h), dtype=bool)
    reciprocal = modes @ transform / data.lattice_nm

    hetero = data.metadata["heterobilayer_correction_applied"]
    interface = ("MoS2", "WSe2" if hetero else "MoS2", data.orientation.upper())
    scale = MONOLAYERS["MoS2"][0] / data.lattice_nm if hetero else 1.0
    a = data.lattice_nm
    basis = a * np.array(((1.0, 0.5), (0.0, np.sqrt(3) / 2)))
    origin = scale * basis @ np.full(2, 2 / 3 if data.orientation == "ap" else 0.0)
    shape = data.u_nm.shape[:2]
    yy, xx = np.indices(shape, dtype=float)
    points = np.stack((xx, yy), axis=-1) * data.pixel_nm
    cutoff = settings["center_inlier_nm"]
    nominal, gradient, _, geometry = geometry_from_centers(data, data.centers_rc_px, basis, points, cutoff)
    probe = settings["probe_fwhm_nm"] / np.sqrt(8 * np.log(2))
    probe_pixels = probe / data.pixel_nm
    sigma = settings["smoothing_sigma_pixels"]
    valid = data.source_valid_mask
    weights = gaussian_filter(valid.astype(float), sigma, mode="constant")
    interior, _ = _good_interior_triangle_mask(data)
    distance2 = np.full(shape, np.inf)
    for y, x in data.centers_rc_px:
        distance2 = np.minimum(distance2, (yy-y)**2 + (xx-x)**2)
    support = interior & valid & (weights > 0.97) & (distance2 > (2*sigma)**2)
    support[:4] = support[-4:] = support[:, :4] = support[:, -4:] = False
    if support.sum() < 300:
        raise ValueError("The common interior has fewer than 300 supported pixels")
    detector_support = apply_spatial(support, turns, reflect)
    reference = {name: rms(measured[detector_support][:, channels], fitted[detector_support][:, channels])
                 for name, channels in rings.items()}

    def compare(smoothed, unblurred):
        oriented = apply_spatial(unblurred, turns, reflect)
        prediction = detector_intensity(oriented, reciprocal, coefficients)
        prediction = gaussian_filter(prediction, (probe_pixels, probe_pixels, 0), mode="nearest")
        error = blur(smoothed - data.u_nm, valid, weights, sigma)[support]
        return {
            "layer_shift_rmse_nm": rms_vector(error),
            "offset_free_layer_shift_rmse_nm": rms_vector(error - error.mean(axis=0)),
            "diffraction_rmse": {name: rms(measured[detector_support][:, channels],
                                           prediction[detector_support][:, channels])
                                 for name, channels in rings.items()},
        }

    grid, full, receipt = solve_from_geometry(scale * gradient, scale * basis, interface)
    linear = linear_response(grid, interface)
    fraction = relaxed_ray(grid, linear, interface)
    first_modes = 2 * np.pi * MODES[:3] @ np.linalg.inv(grid.basis)
    phase_excursion = linear @ first_modes.T
    eta = float(np.sqrt(np.mean(phase_excursion**2)))
    # Fourier power outside the harmonics in the specified adhesion energy.
    # These displacement modes are identically zero at first order.
    active = np.zeros(full.shape[:2], dtype=bool)
    for m, n in np.vstack((MODES, -MODES)):
        active[m, n] = True
    active[0, 0] = True
    power = np.sum(abs(np.fft.fft2(full, axes=(0, 1)))**2, axis=-1)
    controls = {
        "first_harmonic_rms_phase_radians": eta,
        "first_harmonic_maximum_phase_radians": float(np.max(np.abs(phase_excursion))),
        "first_order_rms_displacement_nm": rms_vector(linear.reshape(-1, 2)),
        "full_rms_displacement_nm": rms_vector(full.reshape(-1, 2)),
        "full_power_outside_adhesion_harmonics": float(np.sum(power[~active]) / np.sum(power)),
        "ray_amplitude": fraction, "moire_cell_lengths_nm": np.linalg.norm(grid.cell, axis=0).tolist(),
    }
    backgrounds = {"affine": nominal}
    try:
        correction, local = local_geometry(data, points, basis, np.asarray(geometry["fit_scan"]), cutoff)
        warped = nominal + correction
        dy, dx = np.gradient(warped, data.pixel_nm, axis=(0, 1))
        jacobian = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
        determinant = np.linalg.det(np.asarray(geometry["fit_scan"])[:2])
        local["minimum_jacobian_ratio"] = float(np.min(jacobian[support] / determinant))
        if local["minimum_jacobian_ratio"] > 0:
            backgrounds["warped"] = warped
        else:
            local["failure"] = "Warp folds in the shared interior"
    except (ValueError, np.linalg.LinAlgError) as exc:
        local = {"failure": str(exc)}
    models = {}
    for name, rigid in backgrounds.items():
        phase = origin + scale * rigid
        models[name] = {"rigid": compare(rigid, rigid)}
        for method, displacement in (("first_order", linear), ("energy_scaled_first_order", fraction * linear),
                                     ("full", full)):
            ideal = (sample_disregistry(grid, displacement, phase, grid.cell) - origin) / scale
            filtered = (sample_disregistry(grid, displacement, phase, grid.cell, probe) - origin) / scale
            models[name][method] = compare(filtered, ideal)
    return {
        "archive_md5": data.archive_md5, "interface": interface,
        "reported_twist_degrees": data.metadata.get("AvgMoireTwist"),
        "reported_heterostrain_percent": data.metadata.get("AvgHeteroStrain"),
        "score_pixels": int(support.sum()), "detector_alignment": alignment,
        "detector_calibration": calibration, "detector_archive_fit_rmse": reference,
        "affine_geometry": geometry, "local_geometry": local, "regime": controls, "solvers": receipt,
        "models": models,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    settings = json.loads(SETTINGS.read_text())
    author = load_author_release(args.release, args.dependencies)
    inputs = (Path(__file__), SETTINGS, *(PROJECT / name for name in (
        "bilayer_relaxation.py", "observe_bilayers.py", "author_detector_replay.py",
        "run_antiparallel_signed_audit.py", "run_heterobilayer_sparse_geometry.py",
        "run_heterobilayer_transfer.py", "run_heterobilayer_diffraction_check.py")))
    result = {
        "scope": "Retrospective comparison on all unique processed maps. Geometry and instrument calibration derive from the same map. Use the previously established AP gauge and high-energy origin in DS28. Every comparison uses the same geometry and score pixels. The energy-scaled first-order control uses no measured shifts beyond the geometry. Aliases DS17 and DS18 are excluded.",
        "author_commit": RELEASE_COMMIT, "input_sha256": {p.name: digest(p) for p in inputs},
        "scans": {}, "failed_scans": {},
    }
    for scan in SCAN_IDS:
        checkpoint_guard()
        try:
            data = load_dataset(scan, args.archive, allowed_ids=SCAN_IDS, allow_author_crop=True)
            row = score_scan(data, detector_data(args.archive, scan, author), settings)
            result["scans"][scan] = row
            print(scan, row["regime"]["first_harmonic_rms_phase_radians"],
                  row["regime"]["ray_amplitude"], flush=True)
        except (ConvergenceError, ValueError, np.linalg.LinAlgError, FileNotFoundError, KeyError) as exc:
            result["failed_scans"][scan] = str(exc)
            print(scan, str(exc), flush=True)
        result["wall_seconds"] = perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
