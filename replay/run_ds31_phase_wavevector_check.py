"""Compare relative reciprocal vectors with spatial gradients of interference phase."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.constants import c, e, h, m_e
from scipy.linalg import orthogonal_procrustes
from scipy.ndimage import gaussian_filter

from author_detector_replay import read_detector_fit
from author_release import load_author_release
from bilayer_relaxation import sample_disregistry
from observe_bilayers import load_dataset
from run_detector_prediction import AUTHOR_BASIS
from run_heterobilayer_corrugation import BASIS, MODEL_A, ORIGIN
from run_heterobilayer_raw_arc import select_arcs, solve_profiles, take_profiles
from run_heterobilayer_raw_ds31 import SCAN, fit_patch
from run_heterobilayer_raw_injection import HIGH, RAW, paired_matrices
from run_heterobilayer_raw_interference import BENCHMARK, BLOCK, sampled_blocks
from run_heterobilayer_raw_ring_controls import block_counts, higher_orders
from run_heterobilayer_ring_geometry import invert_local, smooth_geometry
from run_heterobilayer_strain_observer import ORIGINAL_ARCS, SETTINGS, SPATIAL, TRANSFER, use_blocks
from run_heterobilayer_transfer import PROJECT, PROTOCOL, blur, digest, solve
from run_moire_regime_benchmark import detector_data
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


def electron_wavelength_nm(energy_eV):
    energy = e*energy_eV
    return float(1e9*h*c / np.sqrt(energy*(energy+2*m_e*c**2)))


def raw_gradients(blocks, support, arcs, theta, outer_modes, outer_nominal, centers, radius, bound, common_modes):
    """Recover relative reciprocal vectors in detector pixels, before any strain conversion."""
    reference = arcs[6]
    _, checked = paired_matrices(outer_modes, outer_nominal)
    if not np.allclose(checked, reference):
        raise ValueError("The exclusive-arc calibration changed")
    # The reciprocal matrix fits use unit vectors for the median outer spot.
    # Recover the same fractional reciprocal scale for both inner rings.
    outer = outer_modes @ np.linalg.inv(AUTHOR_BASIS)
    common_vectors = (common_modes @ np.linalg.inv(AUTHOR_BASIS))/np.median(np.linalg.norm(outer, axis=1))
    lookup = {(b["row"], b["column_start"]): b for b in blocks}
    result = []
    for row in support:
        checkpoint_guard()
        block = lookup[(row["row"], row["column_start"])]
        output = {k: row[k] for k in ("row", "column_start", "column_count")}
        for phase in ("mean", "even", "odd"):
            values = block[phase]
            fitted, geometry = fit_patch(values, outer_modes, outer_nominal, centers, radius, theta, matrix_bound=bound)
            edge, diagnostics = solve_profiles(take_profiles(values, arcs), arcs)
            paired, exclusive = (reference + p[2:10].reshape(2, 2, 2) for p in (fitted, theta+edge))
            output[phase] = {
                "paired": (common_vectors @ (paired[0]-paired[1])).tolist(),
                "exclusive": (common_vectors @ (exclusive[0]-exclusive[1])).tolist(),
                "fit_qualified": bool(geometry["converged"] and not geometry["reciprocal_matrix_parameters_at_bound"]
                                      and diagnostics["converged"] and not diagnostics["arcs_at_bound"]),
            }
        result.append(output)
    return result


def phase_gradients(data, reciprocal, background, full, fitted_phase, rows, pixel_nm, settings):
    """Differentiate each Cartesian registry on the real scan grid."""
    valid = data.source_valid_mask
    sigma = settings["smoothing_sigma_pixels"]
    weight = gaussian_filter(valid.astype(float), sigma, mode="constant")
    fields = {"archived_unsmoothed": data.u_nm, "archived_smoothed": blur(data.u_nm, valid, weight, sigma),
              "detector_refit_smoothed": blur(fitted_phase, valid, weight, sigma),
              "warped_rigid": background, "model": full, "model_smoothed": blur(full, valid, weight, sigma)}
    offset = data.metadata["author_unwrap_crop_origin_row_col"]
    output = {}
    for name, field in fields.items():
        dy, dx = np.gradient(field, pixel_nm, axis=(0, 1))
        gradient = np.stack((dx, dy), axis=-1)
        phase = np.einsum("ja,...ab->...jb", reciprocal, gradient)
        output[name] = sampled_blocks(rows, phase, offset).reshape(len(rows), BLOCK, len(reciprocal), 2).mean(axis=1)
    return output


def score(values, reference):
    delta = values-reference
    centered = delta-delta.mean(axis=0)
    rms = np.sqrt(np.mean(centered**2))
    amplitude = np.sqrt(np.mean((values-values.mean(axis=0))**2))
    covariance = np.sum((values-values.mean(axis=0))*(reference-reference.mean(axis=0)))
    scale = np.linalg.norm(values-values.mean(axis=0))*np.linalg.norm(reference-reference.mean(axis=0))
    return {"centered_rms": float(rms), "amplitude_rms": float(amplitude),
            "mean_difference_rms": float(np.sqrt(np.mean(delta.mean(axis=0)**2))),
            "correlation": float(covariance/scale) if scale > 1e-14 else None}


def tensor_components(wavevectors, reciprocal, angle):
    """Convert the phase gradients to the common relative displacement gradient."""
    gradient = np.einsum("ij,...jk->...ik", np.linalg.pinv(reciprocal), wavevectors)
    crystal = gradient @ rotation(-angle)
    return {
        "dilation_pp": 50*np.trace(crystal, axis1=-2, axis2=-1),
        "shear_components_pp": 50*np.stack((crystal[:, 0, 0]-crystal[:, 1, 1],
                                           crystal[:, 0, 1]+crystal[:, 1, 0]), axis=-1),
        "relative_rotation_degrees": np.rad2deg((crystal[:, 1, 0]-crystal[:, 0, 1])/2),
    }


def report(raw, phases, reciprocal, angle, scale, transform):
    raw_methods = ("paired", "exclusive")
    measured = {name: scale*np.array([row["mean"][name] for row in raw]) @ transform for name in raw_methods}
    result = {"blocks": len(raw), "all_fits_qualified": sum(all(r[p]["fit_qualified"] for p in ("mean", "even", "odd"))
                                                         for r in raw),
              "wavevector_comparisons_cycles_per_nm": {}, "component_comparisons": {}, "repeatability": {},
              "transformed_raw_wavevectors_cycles_per_nm": {name: value.tolist() for name, value in measured.items()}}
    for key, values in measured.items():
        repeat = scale*np.array([np.asarray(r["even"][key])-r["odd"][key] for r in raw]) @ transform
        result["repeatability"][key] = {
            "even_odd_wavevector_rms_cycles_per_nm": float(np.sqrt(np.mean(repeat**2))),
            "even_odd_component_rms": {name: float(np.sqrt(np.mean(v**2))) for name, v in
                                       tensor_components(repeat, reciprocal, angle).items()},
        }
        result["wavevector_comparisons_cycles_per_nm"][key] = {name: score(p, values) for name, p in phases.items()}
        target = tensor_components(values, reciprocal, angle)
        result["component_comparisons"][key] = {
            "observed_rms": {name: float(np.sqrt(np.mean((v-v.mean(axis=0))**2))) for name, v in target.items()},
            "models": {name: {k: score(v, target[k]) for k, v in tensor_components(p, reciprocal, angle).items()}
                       for name, p in phases.items()},
        }
    result["paired_vs_exclusive_wavevector"] = score(measured["paired"], measured["exclusive"])
    return result


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
    paths = (RAW, HIGH, ORIGINAL_ARCS, SPATIAL, SETTINGS, TRANSFER, PROTOCOL, BENCHMARK)
    raw, high, arc_reference, replay, settings, transfer, protocol, benchmark = (json.loads(p.read_text()) for p in paths)
    supports = use_blocks(raw, high, arc_reference, replay)
    for folder, receipt in ((args.original_ranges, raw), (args.replay_ranges, replay)):
        for file, expected in receipt["range_sha256"].items():
            checkpoint_guard()
            if digest(folder/file) != expected:
                raise ValueError(f"The sampled source changed in {file}")
    data = load_dataset(SCAN, args.archive, allowed_ids=(SCAN,), allow_author_crop=True)
    reference, prior = benchmark["scans"][SCAN], transfer["scans"][SCAN]
    if data.archive_md5 != reference["archive_md5"] or data.archive_md5 != prior["archive_md5"]:
        raise ValueError("The DS31 archive changed")
    author = load_author_release(args.release, args.dependencies)
    disk, _, _, fit_hash = read_detector_fit(args.archive, SCAN, "fit", author)
    if fit_hash != raw["input_sha256"]["DS31_fit.pkl"]:
        raise ValueError("The original detector geometry changed")
    valid = disk._in_use.astype(bool)
    positions = np.column_stack((disk._xvec[valid], disk._yvec[valid]))
    common = disk.clean_normgset(sanity_plot=False)
    indices, nominal = higher_orders(common, positions)
    h_idx, k_idx = indices.T
    outer = h_idx**2 + h_idx*k_idx + k_idx**2 == 9
    indices, nominal = indices[outer], nominal[outer]
    centers, original = block_counts(args.original_ranges, nominal, raw["source"])
    _, unused = block_counts(args.replay_ranges, nominal, replay["source"])
    config = high["aggregate"]["higher_order_9"]
    theta = np.asarray(config["geometry_fit_parameters"])
    bound = config["reciprocal_matrix_bound_pixels"]
    pixel_mrad = raw["source"]["detector_spacing_mrad"]
    radius = 1.71/pixel_mrad
    arcs = select_arcs(np.mean([b["mean"] for b in original], axis=0), nominal, centers, indices, radius, theta)
    if arcs[-1].tolist() != arc_reference["groups"]["higher_order_9"]["arcs_per_layer"]:
        raise ValueError("The outer diffraction arc geometry changed")
    source, test = (raw_gradients(b, s, arcs, theta, indices, nominal, centers, radius, bound, common)
                    for b, s in zip((original, unused), supports, strict=True))
    pixel_nm = raw["source"]["dm4_scan_spacing_nm"]
    angle = np.deg2rad(data.sample_rotation_degrees)
    fit = np.asarray(prior["whole_geometry"]["fit_scan"])
    y, x = np.indices(data.u_nm.shape[:2], dtype=float)
    points = np.stack((x, y), axis=-1)*data.pixel_nm
    warp, _ = smooth_geometry(data, fit, points)
    background = points @ fit[:2] + fit[-1] + warp
    lattice_scale = MODEL_A/data.lattice_nm
    grid, motion, _ = solve(lattice_scale*np.asarray(prior["whole_geometry"]["gradient_crystal"]),
                            BASIS, tuple(protocol["interface"]), prior["grid"])
    phase = ORIGIN + lattice_scale*background
    probe = settings["probe_fwhm_nm"]/np.sqrt(8*np.log(2))
    full = (sample_disregistry(grid, motion, phase, grid.cell, probe)-ORIGIN)/lattice_scale
    detector = detector_data(args.archive, SCAN, author)
    frame = reference["detector_alignment"]
    if frame["spatial_turns"] or frame["spatial_reflection"]:
        raise ValueError("DS31 no longer uses the unrotated detector raster")
    mirror = np.diag((-1, 1)) if frame["vector_reflection"] else np.eye(2)
    transform = np.linalg.inv(AUTHOR_BASIS) @ rotation(frame["vector_rotation_60deg_steps"]*np.pi/3) @ mirror
    reciprocal = common @ transform / data.lattice_nm
    rr, cc = data.metadata["author_unwrap_crop_origin_row_col"]
    height, width = data.u_nm.shape[:2]
    measured = detector[0][rr:rr+height, cc:cc+width]
    seed = np.where(data.source_valid_mask[..., None], data.u_nm, 0.0)
    phase_refit = invert_local(measured, seed, reciprocal, detector[3])
    source_phase, test_phase = (phase_gradients(data, reciprocal, background, full, phase_refit, s, pixel_nm, settings)
                                for s in supports)
    wavelength = electron_wavelength_nm(80000)
    scale = pixel_mrad*1e-3/wavelength
    source_mean = scale*np.mean([r["mean"]["paired"] for r in source], axis=0)
    phase_mean = source_phase["archived_smoothed"].mean(axis=0)
    orientation, singular_sum = orthogonal_procrustes(source_mean, phase_mean)
    result = {
        "scope": "Test the equality between projected local reciprocal-vector differences and real-space gradients of the interference phase. Estimate a single orthogonal detector-to-scan transformation using the average of the original paired spot and archived phase data. The detector pixel and electron wavelength set the scale. Apply the same frame and fixed scale to the original and unused raw rows, to both edge readouts, and to all local wavevector components. No local displacement amplitudes are fitted.",
        "limits": "Both readouts share DS31's microscope acquisition. The relation uses planar local reciprocal vectors and the stored interferometry calibration. A failed equality can identify a measurement inconsistency and cannot determine the unprojected strain.",
        "input_sha256": {p.name: digest(p) for p in (Path(__file__), *paths)},
        "electron_wavelength_nm": wavelength, "reciprocal_pixel_cycles_per_nm": scale,
        "raw_to_scan_orthogonal": orientation.tolist(),
        "best_isotropic_scale_from_source_means_diagnostic": float(singular_sum/np.sum(source_mean**2)),
        "source_mean_wavevector_rmse_cycles_per_nm": float(np.sqrt(np.mean((source_mean@orientation-phase_mean)**2))),
        "cohorts": {
            label: {"metrics": report(r, p, reciprocal, angle, scale, orientation),
                    "raw_gradients_detector_pixels": r,
                    "phase_gradients_cycles_per_nm": {k: v.tolist() for k,v in p.items()}}
            for label, r, p in (("original", source, source_phase), ("replay", test, test_phase))
        },
        "wall_seconds": perf_counter()-start,
    }
    checkpoint_guard()
    atomic_json(args.output, result)
    print(result["cohorts"]["replay"]["metrics"]["wavevector_comparisons_cycles_per_nm"], flush=True)


if __name__ == "__main__":
    main()
