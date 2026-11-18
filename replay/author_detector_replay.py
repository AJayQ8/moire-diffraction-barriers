"""Reconstruct archived detector calibration and run the authors' fitting code."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from scipy import ndimage as ndi

from run_detector_prediction import AUTHOR_BASIS, DiskReadback, apply_spatial
from run_registry_prediction import rotation


@dataclass(frozen=True)
class DetectorCalibration:
    observed: np.ndarray
    modes: np.ndarray
    native_fit_lattice: np.ndarray
    coefficients: np.ndarray
    binned_coefficients: np.ndarray
    binned_minimum: np.ndarray
    binned_span: np.ndarray
    archived_fitted_intensity: np.ndarray
    diagnostics: dict


def detector_intensity(disregistry_lattice, modes, coefficients):
    phase = np.pi * (disregistry_lattice @ modes.T)
    cosine = np.cos(phase)
    scale, odd, offset = coefficients.T
    return scale * cosine**2 + odd * cosine * np.sin(phase) + offset


def read_detector_fit(archive, scan, filename, author):
    with ZipFile(archive / f"{scan}.zip") as zf:
        member = zf.getinfo(f"{scan}/{filename}.pkl")
        if member.file_size > 128 * 1024**2:
            raise ValueError(f"Unexpected detector payload size for {scan} {filename}")
        payload = zf.read(member)
    disk, *arrays = DiskReadback(BytesIO(payload)).load()
    holder = object.__new__(author.diskset.DiskSet)
    holder.__dict__.update(vars(disk))
    return holder, np.stack(arrays[:-1], axis=1), np.asarray(arrays[-1]), sha256(payload).hexdigest()


def normalize_channels(images, author):
    return np.stack([author.utils.normalize(image) for image in images], axis=-1)


def rms(observed, predicted):
    return float(np.sqrt(np.mean((observed - predicted)**2)))


def load_calibration(archive, scan, author):
    disk, native, lattice, native_hash = read_detector_fit(archive, scan, "fit", author)
    coarse_disk, coarse, coarse_lattice, coarse_hash = read_detector_fit(archive, scan, "fitbin2", author)
    images = disk.df_set()
    modes = disk.clean_normgset(sanity_plot=False)
    if not np.array_equal(modes, coarse_disk.clean_normgset(sanity_plot=False)):
        raise ValueError(f"{scan}: native and binned reciprocal axes differ")
    if native.shape != (len(modes), 2) or coarse.shape != (len(modes), 3):
        raise ValueError(f"{scan}: unexpected detector coefficient format")
    observed = normalize_channels(images, author)
    native_columns = np.column_stack((native, np.zeros(len(native))))
    unshifted = detector_intensity(lattice, modes, native_columns)
    offset = np.mean(observed - unshifted, axis=(0, 1))
    recovered = np.column_stack((native, offset))
    archived_prediction = detector_intensity(lattice, modes, recovered)
    wrong_interpretation = detector_intensity(lattice, modes, np.column_stack((native[:, 0], np.zeros(len(native)), native[:, 1])))
    coarse_images = np.stack([author.utils.bin(image, bin_w=2, method=np.sum) for image in coarse_disk.df_set()])
    coarse_observed = normalize_channels(coarse_images, author)
    binned_native = np.stack([author.utils.bin(image, bin_w=2, method=np.sum)
                              for image in np.moveaxis(observed, -1, 0)])
    binned_minimum = binned_native.min(axis=(1, 2))
    binned_span = np.ptp(binned_native, axis=(1, 2))
    if not np.allclose((binned_native - binned_minimum[:, None, None]) / binned_span[:, None, None],
                       np.moveaxis(coarse_observed, -1, 0), rtol=0, atol=1e-8):
        raise ValueError(f"{scan}: binned images differ between the saved detector fits")
    coarse_prediction = detector_intensity(coarse_lattice, modes, coarse)
    # Verify the saved columns against an independent linear regression on
    # the recorded, already fitted lattice coordinates.
    phase = np.pi * (lattice @ modes.T)
    cosine = np.cos(phase)
    regressors = np.stack((cosine**2, cosine * np.sin(phase), np.ones_like(phase)), axis=-1)
    fitted_coefficients = np.stack([np.linalg.lstsq(regressors[..., i, :].reshape(-1, 3),
                                                   observed[..., i].ravel(), rcond=None)[0]
                                    for i in range(len(modes))])
    diagnostics = {"native_fit_sha256": native_hash, "binned_fit_sha256": coarse_hash,
                   "binned_fit_raster": list(coarse_lattice.shape[:2]), "modes": modes.tolist(),
                   "saved_native_A_B": native.tolist(), "reconstructed_native_C": offset.tolist(),
                   "saved_binned_A_B_C": coarse.tolist(),
                   "native_replay_rmse": rms(observed, archived_prediction),
                   "native_even_misread_rmse": rms(observed, wrong_interpretation),
                   "binned_replay_rmse": rms(coarse_observed, coarse_prediction),
                   "regressed_A_B_RMSE_vs_saved": rms(fitted_coefficients[:, :2], native),
                   "independently_regressed_A_B_C": fitted_coefficients.tolist()}
    return DetectorCalibration(observed, modes, lattice, recovered, coarse, binned_minimum, binned_span,
                               archived_prediction, diagnostics)


def calibrate_frame(unwrapped_nm, lattice_nm, calibration):
    """Match the authors' detector frame using measured archive intermediates."""
    truth = calibration.archived_fitted_intensity[2:-2:2, 2:-2:2]
    inv_basis = np.linalg.inv(AUTHOR_BASIS)
    best = None
    for turns in range(4):
        for spatial_reflection in (False, True):
            sampled = apply_spatial(unwrapped_nm, turns, spatial_reflection)[2:-2:2, 2:-2:2] / lattice_nm
            if sampled.shape[:2] != truth.shape[:2]:
                continue
            for vector_reflection in (False, True):
                reflect = np.diag((-1, 1)) if vector_reflection else np.eye(2)
                for index in range(6):
                    transform = inv_basis @ rotation(index * np.pi / 3) @ reflect
                    displacement = sampled @ transform.T
                    predicted = detector_intensity(displacement, calibration.modes, calibration.coefficients)
                    loss = float(np.sqrt(np.nanmean((predicted - truth)**2)))
                    if best is None or loss < best[0]:
                        best = (loss, turns, spatial_reflection, transform, index, vector_reflection)
    if best is None:
        raise ValueError("Detector and archive rasters cannot be registered")
    return best


def synthetic_detector(field_nm, lattice_nm, pixel_nm, calibration, frame, probe_fwhm_nm):
    _, turns, spatial_reflection, transform, _, _ = frame
    detector_nm = apply_spatial(field_nm, turns, spatial_reflection)
    lattice = detector_nm @ transform.T / lattice_nm
    data = detector_intensity(lattice, calibration.modes, calibration.coefficients)
    sigma = probe_fwhm_nm / (np.sqrt(8 * np.log(2)) * pixel_nm)
    return ndi.gaussian_filter(data, (sigma, sigma, 0), mode="nearest")


def author_fit_disk_images(normalized, calibration, author, workers, guard):
    """Use the released multistart and binned-to-native fit routines."""
    images = np.moveaxis(normalized, -1, 0)
    modes = calibration.modes
    stages = {}

    def fit(stage, values, coefficients, **settings):
        guard()
        start = perf_counter()
        nx, ny = values.shape[1:]
        displacement, residual = author.fit.fit_u(values, coefficients, nx, ny, modes, nproc=workers,
                                                  norm_bool=False, parallel=workers > 1, **settings)
        stages[stage] = {"wall_seconds": perf_counter() - start, "pixels": int(np.prod(residual.shape)),
                         "median_sum_absolute_intensity_residual": float(np.median(residual))}
        guard()
        return displacement

    binned = np.stack([author.utils.bin(image, bin_w=2, method=np.sum) for image in images])
    binned = (binned - calibration.binned_minimum[:, None, None]) / calibration.binned_span[:, None, None]
    coarse = fit("binned_2_multistart", binned, calibration.binned_coefficients, multistart_bool=True)
    seed = np.stack([author.utils.unbin(coarse[..., j], 2) for j in range(2)], axis=-1)
    native = fit("native_from_bin", images, calibration.coefficients, guess=seed)
    refined = fit("native_neighbor_refit", images, calibration.coefficients, guess=native, multistart_neighbor_bool=True)
    return refined, stages
