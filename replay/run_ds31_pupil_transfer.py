"""Compute coherent DS31 diffraction from periodic sidebands without cropping the probe."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import expit

from bilayer_relaxation import sample_disregistry
from observe_bilayers import load_dataset
from run_ds31_coherent_probe import INJECTED_DILATION, STRAIN_COMPARISON
from run_ds31_phase_wavevector_check import electron_wavelength_nm
from run_ds31_probe_edge_response import WAVEVECTOR, edge_geometry, elastic_partition, native_gradients
from run_heterobilayer_corrugation import BASIS, MODEL_A, ORIGIN
from run_heterobilayer_raw_arc import RADIAL_PIXELS, TEMPLATE_PIXELS, read_trace, solve_profiles
from run_heterobilayer_raw_ds31 import COORDINATES, SCAN, fit_patch
from run_heterobilayer_raw_injection import HIGH, RAW, paired_matrices
from run_heterobilayer_ring_geometry import smooth_geometry
from run_heterobilayer_strain_observer import TRANSFER
from run_heterobilayer_transfer import PROTOCOL, digest, solve
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


PUPIL_WIDTHS_PIXELS = (0.25, 0.50)
SIDE_BAND_TOLERANCE = 0.001
RAW_INTERFERENCE = STRAIN_COMPARISON.with_name("heterobilayer_raw_interference_v01.json")


def bands(shift, wavevector, tolerance):
    """Expand a periodic phase factor, with an absolute bound on omitted amplitude."""
    size = len(shift)
    modes = np.fft.fftfreq(size, d=1/size)
    indices = np.stack(np.meshgrid(modes, modes, indexing="ij"), axis=-1).reshape(-1, 2)
    wave = np.exp(-2j*np.pi*(shift@wavevector))
    coefficients = np.fft.fft2(wave).ravel()/size**2
    order = np.argsort(np.abs(coefficients))[::-1]
    tail = np.cumsum(np.abs(coefficients[order])[::-1])[::-1]
    retained = int(np.sum(tail > tolerance))
    return indices[order[:retained]], coefficients[order[:retained]], float(tail[retained]) if retained < len(tail) else 0.0


def pupil(points, radius, width):
    return np.sqrt(expit((radius-np.linalg.norm(points, axis=-1))/width))


def prepare(geometry, scale, frame, shift, width):
    layer, disk, normal, g, radius, reference, design = geometry
    track = (radius[layer,None,None]+TEMPLATE_PIXELS[None,:,None])*normal[:,None]
    template = pupil(track, radius[layer,None], width)**2
    contrast = template[:, TEMPLATE_PIXELS < -1].mean(axis=1)-template[:, TEMPLATE_PIXELS > 1].mean(axis=1)
    arcs = (disk, None, normal, template, contrast, design, reference, np.bincount(layer))
    tracks = (radius[layer,None,None]+RADIAL_PIXELS[None,:,None])*normal[:,None]
    shares = np.array((elastic_partition("MoS2", "WSe2"), 0.0))
    shares[1] = shares[0]-1
    channels = []
    for l in range(2):
        for d in range(g.shape[1]):
            subset = np.flatnonzero((layer == l) & (disk == d))
            if not len(subset):
                continue
            q = shares[l]*scale*(g[l,d]@frame)
            mode, coefficient, tail = bands(shift, q, SIDE_BAND_TOLERANCE)
            channels.append((subset, tracks[subset], radius[l], q, mode, coefficient, tail))
    return arcs, channels


def profiles(channels, width, pixel_q, scan_to_raw, basis_inverse, phase, phase_gradient, background_gradient, bare=False):
    # Coordinates in the pupil, including affine change of the local lattice.
    all_arcs = sum(len(s[0]) for s in channels)
    output = np.empty((all_arcs, len(RADIAL_PIXELS)))
    for indices, tracks, radius, q, modes, coefficients, _ in channels:
        background_shift = -(q@background_gradient)@scan_to_raw/pixel_q
        if bare:
            amplitude = pupil(tracks[None]-background_shift[:,None,None,:], radius, width)
        else:
            sideband = ((modes@basis_inverse)@phase_gradient)@scan_to_raw/pixel_q
            weights = np.exp(2j*np.pi*(phase@modes.T))*coefficients
            amplitude = []
            for offset, displacement, values in zip(background_shift, sideband, weights, strict=True):
                relative = tracks[...,None,:]-offset-displacement
                amplitude.append(pupil(relative, radius, width)@values)
        output[indices] = np.mean(np.abs(amplitude)**2, axis=0)
    return output


def disk_channels(geometry, nominal, modes, theta, frame, shifts, pixel_q):
    _, _, _, reciprocal, radius, reference, _ = geometry
    offset = reciprocal-modes@reference+theta[:2]
    partition = elastic_partition("MoS2", "WSe2")
    relative = COORDINATES[None] + (np.rint(nominal)-nominal)[:,None,None]
    channels = []
    for d, pixels in enumerate(relative):
        layers = []
        for l, share in enumerate((partition, partition-1)):
            q = share*pixel_q*(reciprocal[l,d]@frame)
            mode, coefficient, _ = bands(shifts, q, SIDE_BAND_TOLERANCE)
            layers.append((pixels-offset[l,d], radius[l], q, mode, coefficient))
        channels.append(layers)
    return channels


def simulated_disks(channels, width, pixel_q, scan_to_raw, basis_inverse, phase,
                    registry_gradient, background_gradient):
    """Render the two disks incoherently for the estimator with a free overlap term."""
    arrays = []
    for layers in channels:
        checkpoint_guard()
        counts = np.zeros(COORDINATES.shape[:2])
        for track, radius, q, mode, coefficient in layers:
            offsets = ((mode@basis_inverse)@registry_gradient)@scan_to_raw/pixel_q
            baseline = -(q@background_gradient)@scan_to_raw/pixel_q
            amplitudes = np.exp(2j*np.pi*(phase@mode.T))*coefficient
            for center, sidebands, coeff in zip(baseline, offsets, amplitudes, strict=True):
                pupil_amplitude = pupil(track[...,None,:]-center-sidebands, radius, width) @ coeff
                counts += np.abs(pupil_amplitude)**2/len(phase)
        arrays.append(counts+0.05)
    return np.array(arrays)


def readout(prediction, arcs, theta):
    fit, quality = solve_profiles(prediction, arcs)
    return {"trace_pp": read_trace(theta+fit, arcs[6]), "fit_qualified": bool(
        quality["converged"] and not quality["arcs_at_bound"]), "profile_rmse": quality["weighted_profile_rmse"]}


def compare_rows(rows, key="readouts"):
    reference = {k: np.array([r[k] for r in rows]) for k in ("arc", "paired", "ideal_full", "ideal_warp")}
    result = {"block_count": len(rows), "reference_spread_pp": {k: float(v.std()) for k,v in reference.items()}}
    for name in ("rigid", "warp_only", "full", "reduced_compression"):
        values = np.array([r[key][name]["trace_pp"] for r in rows])
        centered = values-values.mean()
        result[name] = {"mean_pp": float(values.mean()), "spread_pp": float(values.std()),
                        "qualified_blocks": sum(r[key][name]["fit_qualified"] for r in rows),
                        "centered_errors_pp": {k: float((values-v).std()) for k,v in reference.items()},
                        "recovery_slope": float(np.mean(centered*(reference["ideal_full"]-reference["ideal_full"].mean()))/
                                                reference["ideal_full"].var())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--paired", action="store_true")
    args = parser.parse_args()
    start = perf_counter()
    paths = (WAVEVECTOR, HIGH, RAW, TRANSFER, PROTOCOL, STRAIN_COMPARISON, RAW_INTERFERENCE)
    wave, high, raw, transfer, protocol, reference, interference = (json.loads(p.read_text()) for p in paths)
    data = load_dataset(SCAN, args.archive, allowed_ids=(SCAN,), allow_author_crop=True)
    prior = transfer["scans"][SCAN]
    if data.archive_md5 != prior["archive_md5"]:
        raise ValueError("The DS31 source scan changed")
    pixel = raw["source"]["dm4_scan_spacing_nm"]
    pixel_q = raw["source"]["detector_spacing_mrad"]*1e-3/electron_wavelength_nm(80000)
    angle = np.deg2rad(data.sample_rotation_degrees)
    raw_to_scan = np.asarray(wave["raw_to_scan_orthogonal"])
    raw_to_crystal = raw_to_scan @ rotation(-angle)
    fit = np.asarray(prior["whole_geometry"]["fit_scan"])
    yy, xx = np.indices(data.u_nm.shape[:2])
    points = np.stack((xx, yy), axis=-1)*data.pixel_nm
    affine = points@fit[:2]+fit[-1]
    warp, _ = smooth_geometry(data, fit, points)
    scale = MODEL_A/data.lattice_nm
    registry = ORIGIN+scale*(affine+warp)
    inverse = np.linalg.inv(BASIS)
    grid, shift, _ = solve(scale*np.asarray(prior["whole_geometry"]["gradient_crystal"]), BASIS,
                           tuple(protocol["interface"]), prior["grid"])
    full = (sample_disregistry(grid, shift, registry, grid.cell)-ORIGIN)/scale-affine
    gradients = native_gradients(data, warp, full, pixel)
    scan_background = gradients["background"] @ rotation(angle)
    registry_gradient = scale*(fit[:2].T*data.pixel_nm/pixel + scan_background)
    # Keep the preceding source-row estimate fixed. The projector removes
    # only a portion of the periodic longitudinal motion, not the center warp.
    reduced_fraction = interference["raw_compression_multiplier"]
    k = grid.k
    k2 = np.sum(k**2, axis=-1)
    longitudinal = k*(np.sum(np.fft.fft2(shift, axes=(0, 1))*k, axis=-1)/np.maximum(k2, 1e-12))[...,None]
    reduced = shift-(1-reduced_fraction)*np.fft.ifft2(longitudinal, axes=(0, 1)).real
    config = high["aggregate"]["higher_order_9"]
    theta = np.asarray(config["geometry_fit_parameters"])
    geometry = edge_geometry(high, raw, theta)
    indices = np.asarray(high["higher_order_indices"])
    h, k = indices.T
    outer = h*h+h*k+k*k == 9
    indices = indices[outer]
    nominal = np.asarray(high["higher_order_nominal_centers_row_col"])[outer]
    modes, _ = paired_matrices(indices, nominal)
    centers = np.rint(nominal).astype(int)
    radius = 1.71/raw["source"]["detector_spacing_mrad"]

    def paired_readout(disks, arguments):
        counts = simulated_disks(disks, *arguments)
        _, score = fit_patch(counts, indices, nominal, centers, radius, theta,
                             matrix_bound=config["reciprocal_matrix_bound_pixels"])
        return {"trace_pp": score["relative_trace_percent"],
                "fit_qualified": score["converged"] and not score["reciprocal_matrix_parameters_at_bound"],
                "deviance": score["poisson_deviance_per_dof"]}

    crop = data.metadata["author_unwrap_crop_origin_row_col"]
    result = {"scope": "DS31 weak-phase projected waves. Expand the periodic continuum phase in reciprocal sidebands and sum each shifted pupil coherently. The local center warp is linearized over each probe. Evaluate the same synthetic exclusive arcs as the finite-window simulation. Two fixed widths bound sensitivity to a softened pupil edge.",
              "limits": "Independent layer amplitudes in exclusive arcs. The optional two-disk estimator uses the incoherent sum of both projected layers with equal scattering amplitudes. No dynamical scattering, atomic form factors or aberration model. Spot orientations and radius are based on the original raw average. The reduced periodic longitudinal amplitude is a source-fit diagnostic. Centering each prediction permits an offset.",
              "input_sha256": {p.name: digest(p) for p in (Path(__file__),*paths)},
              "reduced_periodic_compression_factor": reduced_fraction,
              "sideband_omitted_amplitude_limit": SIDE_BAND_TOLERANCE, "cases": {}}
    case_rows = {key: [r for r in reference["samples"][key] if 15 <= r["row"] <= 82]
                 for key in ("original", "replay")}
    if args.pilot:
        case_rows = {k: v[::max(1,len(v)//6)][:6] for k,v in case_rows.items()}
    paired_models = {}
    if args.paired:
        paired_models = {name: disk_channels(geometry, nominal, modes, theta, raw_to_crystal, values, pixel_q)
                         for name, values in (("warp_only", np.zeros_like(shift)), ("full", shift/scale),
                                              ("reduced_compression", reduced/scale))}
    for width in PUPIL_WIDTHS_PIXELS:
        arcs, channels = prepare(geometry, pixel_q, raw_to_crystal, shift/scale, width)
        _, weaker = prepare(geometry, pixel_q, raw_to_crystal, reduced/scale, width)
        case = {"width_pixels": width, "cohorts": {},
                "maximum_sidebands": max(len(ch[4]) for ch in channels),
                "maximum_omitted_amplitude": max(ch[-1] for ch in channels)}
        zero, center = np.zeros((1, 2, 2)), np.zeros((1, 2))
        arguments = (width, pixel_q, raw_to_scan.T, inverse, center, zero)
        rigid = readout(profiles(channels, *arguments, zero, bare=True), arcs, theta)
        imposed = (INJECTED_DILATION*rotation(angle))[None]
        injected = readout(profiles(channels, *arguments, imposed, bare=True), arcs, theta)
        case["affine_calibration"] = {"input_pp": 100*INJECTED_DILATION,
                                      "recovered_pp": injected["trace_pp"]-rigid["trace_pp"],
                                      "qualified": rigid["fit_qualified"] and injected["fit_qualified"]}
        if paired_models:
            paired_rigid = paired_readout(paired_models["warp_only"], (*arguments, zero))
            paired_injected = paired_readout(paired_models["warp_only"], (*arguments, imposed))
            case["paired_affine_calibration"] = {"input_pp": 100*INJECTED_DILATION,
                                                 "recovered_pp": paired_injected["trace_pp"]-paired_rigid["trace_pp"],
                                                 "qualified": paired_rigid["fit_qualified"] and paired_injected["fit_qualified"]}
        for cohort, rows in case_rows.items():
            outputs = []
            for row in rows:
                checkpoint_guard()
                y = row["row"]-crop[0]
                x = row["column_start"]-crop[1]
                region = np.s_[y,x:x+row["column_count"]]
                args_local = (width, pixel_q, raw_to_scan.T, inverse, registry[region]@inverse.T,
                              registry_gradient[region], scan_background[region])
                measured = {k: row[k] for k in ("row","column_start","column_count","arc","paired")}
                output = {**measured, "readouts": {"rigid": rigid}}
                for name, q in (("warp_only", "background"), ("full", "full")):
                    output[f"ideal_{'warp' if name == 'warp_only' else 'full'}"] = float(
                        50*np.trace(gradients[q][region], axis1=-2, axis2=-1).mean())
                for name, prepared in (("warp_only", channels), ("full", channels), ("reduced_compression", weaker)):
                    predicted = profiles(prepared, *args_local, bare=name == "warp_only")
                    output["readouts"][name] = readout(predicted, arcs, theta)
                if paired_models:
                    output["paired_readouts"] = {"rigid": paired_rigid}
                    for name, disks in paired_models.items():
                        output["paired_readouts"][name] = paired_readout(disks, args_local)
                outputs.append(output)
            case["cohorts"][cohort] = {"summary": compare_rows(outputs), "rows": outputs}
            if paired_models:
                case["cohorts"][cohort]["paired_summary"] = compare_rows(outputs, "paired_readouts")
        result["cases"][str(width)] = case
        result["wall_seconds"] = perf_counter()-start
        atomic_json(args.output, result)
        print(width, case["affine_calibration"], {k: v["summary"] for k,v in case["cohorts"].items()}, flush=True)


if __name__ == "__main__":
    main()
