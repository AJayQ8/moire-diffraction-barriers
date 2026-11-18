"""Propagate projected continuum displacement through a coherent weak-phase probe."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.special import j1

from bilayer_relaxation import sample_disregistry
from observe_bilayers import load_dataset
from run_ds31_phase_wavevector_check import electron_wavelength_nm
from run_ds31_probe_edge_response import WAVEVECTOR, edge_geometry, elastic_partition, native_gradients
from run_heterobilayer_corrugation import BASIS, MODEL_A, ORIGIN
from run_heterobilayer_raw_arc import RADIAL_PIXELS, TEMPLATE_PIXELS, read_trace, solve_profiles
from run_heterobilayer_raw_ds31 import SCAN
from run_heterobilayer_raw_injection import HIGH, RAW
from run_heterobilayer_ring_geometry import smooth_geometry
from run_heterobilayer_strain_observer import TRANSFER
from run_heterobilayer_transfer import PROJECT, PROTOCOL, digest, solve
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


PHYSICAL_PROBE_FWHM_NM = 1.25
COHERENCE_PHASES = (0, np.pi/2, -np.pi/2, np.pi)
INJECTED_DILATION = 0.003
STRAIN_COMPARISON = PROJECT / "heterobilayer_strain_observer_v01.json"


def aperture_probe(size, step, convergence, wavelength, fwhm, defocus_nm):
    """Disk aperture with the stated convergence and a Gaussian source blur."""
    local = (np.arange(size)-size//2)*step
    yy, xx = np.meshgrid(local, local, indexing="ij")
    points = np.stack((yy, xx), axis=-1)
    r = np.hypot(xx, yy)
    k = 2*np.pi*(convergence/wavelength)*r
    profile = np.ones(k.shape)
    nonzero = k > 0
    profile[nonzero] = 2*j1(k[nonzero])/k[nonzero]
    frequencies = np.fft.fftfreq(size, step)
    qy, qx = np.meshgrid(frequencies, frequencies, indexing="ij")
    pupil = np.fft.fft2(np.fft.ifftshift(profile))
    # Apply defocus as a coherent phase, then represent finite source size
    # as an incoherent Gaussian average of the final simulated intensities.
    pupil *= np.exp(-1j*np.pi*wavelength*defocus_nm*(qx**2+qy**2))
    probe = np.fft.fftshift(np.fft.ifft2(pupil))
    line = np.abs(probe[size//2, size//2:])**2
    top = line[0]/2
    half = np.flatnonzero(line < top)[0]
    coherent_fwhm = 2*np.interp(top, line[half-1:half+1][::-1], local[size//2+half-1:size//2+half+1][::-1])
    source_sigma = np.sqrt(max(fwhm**2-coherent_fwhm**2, 0))/np.sqrt(8*np.log(2))
    return points, probe, float(coherent_fwhm), float(source_sigma)


def displacement_patches(field, locations, points, pixel_nm, source_sigma, source_samples):
    """Sample the same continuum field under a set of displaced probes."""
    centers = locations[:, None, None, None, :] + source_samples[None, :, None, None, :]*source_sigma/pixel_nm
    sample = centers+points[None, None]/pixel_nm
    coordinates = np.moveaxis(sample, -1, 0)
    dimensions = field.shape[:2]
    # The aperture side lobes are weak at the crop boundary. Continue the
    # nearest finite deformation there rather than interpolate a vacuum.
    patches = np.stack([map_coordinates(field[..., i], coordinates, order=3, mode="nearest")
                        for i in range(2)], axis=-1)
    coverage = np.all((sample >= 0)&(sample < np.array(dimensions)-1), axis=-1)
    return patches, coverage


def edge_readout(geometry, q_per_pixel, step, pad, probe, patches, registry, weights,
                 raw_to_scan, raw_to_crystal, partition):
    """Use coherent amplitudes where the two disks overlap in detector space."""
    layer, disk, normals, g, radius, _, _ = geometry
    scan_count, source_count = patches.shape[:2]
    spatial_center = patches.shape[2]//2
    common = patches[:, :, spatial_center, spatial_center]
    relative = patches-common[:, :, None, None]
    # Smaller reciprocal radius belongs to the larger lattice. The
    # common translation cancels in the exclusive-edge response. Restore
    # relative phase at the probe centers only for the overlap calculation.
    fractions = np.array((partition, -(1-partition)))
    fft_scale = pad*step*q_per_pixel
    fft_center = pad//2
    start = pad//2-probe.shape[0]//2
    profiles = np.empty((len(COHERENCE_PHASES)+1, len(layer), len(RADIAL_PIXELS)))
    for d in range(len(g[0])):
        checkpoint_guard()
        arcs = np.flatnonzero(disk == d)
        coordinates = g[layer[arcs], d, None, :] + (radius[layer[arcs], None, None]+RADIAL_PIXELS[:, None])*normals[arcs, None]
        sampled = []
        for l in range(2):
            reciprocal = q_per_pixel*(g[l,d]@raw_to_crystal)
            phase = relative@reciprocal
            wave = probe[None, None]*np.exp(-2j*np.pi*fractions[l]*phase)
            volume = np.zeros((scan_count, source_count, pad, pad), dtype=np.complex64)
            volume[..., start:start+probe.shape[0], start:start+probe.shape[1]] = wave
            field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(volume, axes=(-2,-1)), axes=(-2,-1)), axes=(-2,-1))
            displacement = ((coordinates-g[l,d])@raw_to_scan)[..., ::-1]
            indices = np.moveaxis(fft_center+fft_scale*displacement, -1, 0)
            values = np.array([map_coordinates(v, indices, order=1, mode="constant")
                               for v in field.reshape(scan_count*source_count, pad, pad)]).reshape(
                                   scan_count, source_count, len(arcs), -1)
            sampled.append(values)
        first, second = sampled
        phase = 2*np.pi*registry@(q_per_pixel*(g[:,d].mean(axis=0)@raw_to_crystal))
        mixed = np.stack([np.abs(first+np.exp(1j*(phase+p))[..., None, None]*second)**2
                          for p in COHERENCE_PHASES])
        exclusive = np.where(layer[arcs, None] == 0, np.abs(first)**2, np.abs(second)**2)
        profiles[:, arcs] = np.einsum("s,pnsar->par", weights, np.concatenate((exclusive[None], mixed), axis=0))/scan_count
    return profiles


def make_template(geometry, q_per_pixel, step, pad, probe, raw_to_scan):
    layer, disk, normal, _, radius, reference, design = geometry
    shifted = np.zeros((pad,pad), dtype=np.complex64)
    start = pad//2-probe.shape[0]//2
    shifted[start:start+probe.shape[0], start:start+probe.shape[1]] = probe
    plane = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(shifted))))**2
    track = (radius[layer,None,None]+TEMPLATE_PIXELS[None,:,None])*normal[:,None]
    centers = pad//2+pad*step*q_per_pixel*(track@raw_to_scan)[..., ::-1]
    raw = np.stack([map_coordinates(plane, p.T, order=1, mode="constant") for p in centers])
    scaled = raw/np.median(raw[:, TEMPLATE_PIXELS < -2])
    # The same aperture template is used in the affine injection test.
    template = gaussian_filter(scaled, (0, 0.55), mode="nearest")
    contrast = template[:, TEMPLATE_PIXELS < -1].mean(axis=1)-template[:, TEMPLATE_PIXELS > 1].mean(axis=1)
    if np.min(contrast) < 0.2:
        raise ValueError("The synthetic aperture does not resolve all chosen edges")
    return disk, None, normal, template, contrast, design, reference, np.bincount(layer)


def evaluate(geometry, theta, q_per_pixel, input_rows, fields, affine, pixel, crop, raw_to_scan, raw_to_crystal,
             step=0.25, size=48, pad=144, defocus_nm=0):
    wavelength = electron_wavelength_nm(80000)
    convergence = 1.71e-3
    points, probe, fwhm, source_sigma = aperture_probe(size, step, convergence, wavelength, PHYSICAL_PROBE_FWHM_NM, defocus_nm)
    if source_sigma > 0:
        weights = np.array((1/3, 1/6, 1/6, 1/6, 1/6))
        source_samples = np.sqrt(3)*np.array(((0.,0.),(1,0),(-1,0),(0,1),(0,-1)))
    else:
        weights = np.ones(1)
        source_samples = np.zeros((1, 2))
    arcs = make_template(geometry, q_per_pixel, step, pad, probe, raw_to_scan)
    output = {"coherent_probe_fwhm_nm": fwhm, "source_sigma_nm": source_sigma,
              "real_space_step_nm": step, "window_nm": size*step,
              "detector_samples_per_native_pixel": pad*step*q_per_pixel, "samples": []}
    partition = elastic_partition("MoS2","WSe2")
    for row in input_rows:
        checkpoint_guard()
        locations = np.column_stack((np.arange(row["column_count"])*0+row["row"]-crop[0],
                                     row["column_start"]-crop[1]+np.arange(row["column_count"])))
        receipt = {k: row[k] for k in ("row","column_start","column_count")}
        receipt["readouts"] = {}
        for name, displacement in fields.items():
            patches, inside = displacement_patches(displacement, locations, points, pixel, source_sigma, source_samples)
            source_center = locations[:, None]+source_samples[None]*source_sigma/pixel
            rc = np.moveaxis(source_center, -1, 0)
            rigid = np.stack([map_coordinates(affine[..., k], rc, order=1) for k in range(2)], axis=-1)
            registry = rigid+patches[:, :, size//2, size//2]
            profiles = edge_readout(geometry, q_per_pixel, step, pad, probe, patches, registry, weights,
                                    raw_to_scan, raw_to_crystal, partition)
            readings = []
            for profile in profiles:
                scale = np.median(profile[:, RADIAL_PIXELS < -1])
                fit, quality = solve_profiles(profile/max(scale, 1e-12), arcs)
                readings.append({"trace_percent": read_trace(theta+fit, arcs[6]),
                                 "qualified": bool(quality["converged"] and not quality["arcs_at_bound"]),
                                 "fit": quality})
            receipt["readouts"][name] = {"exclusive": readings[0], "mixed": readings[1:],
                                          "probe_patch_inside_fraction": float(inside.mean())}
        output["samples"].append(receipt)
    injection = np.array([r["readouts"]["linear_calibration"]["exclusive"]["trace_percent"]
                          - r["readouts"]["rigid"]["exclusive"]["trace_percent"] for r in output["samples"]])
    output["affine_injection"] = {"input_relative_dilation_pp": 100*INJECTED_DILATION,
                                  "recovered_mean_pp": float(injection.mean()),
                                  "recovered_std_pp": float(injection.std()),
                                  "all_fits_qualified": all(r["readouts"][case]["exclusive"]["qualified"]
                                                            for r in output["samples"] for case in ("rigid", "linear_calibration"))}
    return output


def compare_strain(case, inputs, gradient, crop):
    measured = {(r["row"], r["column_start"]): r for r in inputs}
    output = {"n_blocks": len(case["samples"]), "recovered": {}}
    for sample in case["samples"]:
        row, col = sample["row"], sample["column_start"]
        block = gradient[row-crop[0], col-crop[1]:col-crop[1]+sample["column_count"]]
        sample["reference"] = {
            "model_at_probe_centers_pp": float(50*np.trace(block, axis1=-2, axis2=-1).mean()),
            **{key: measured[row, col][key] for key in ("arc", "paired")},
        }
    reference = {key: np.array([s["reference"][key] for s in case["samples"]])
                 for key in ("model_at_probe_centers_pp", "arc", "paired")}
    output["reference_spread_pp"] = {key: float(value.std()) for key, value in reference.items()}
    output["minimum_patch_coverage"] = min(s["readouts"]["full"]["probe_patch_inside_fraction"] for s in case["samples"])
    for key in ("exclusive", *(str(i) for i in range(len(COHERENCE_PHASES)))):
        reading = [s["readouts"]["full"]["exclusive"] if key == "exclusive" else s["readouts"]["full"]["mixed"][int(key)]
                   for s in case["samples"]]
        scores = np.array([r["trace_percent"] for r in reading])
        truth = reference["model_at_probe_centers_pp"]
        covariance = np.mean((scores-scores.mean())*(truth-truth.mean()))
        output["recovered"][key] = {"spread_pp": float(scores.std()),
                                     "qualified_blocks": sum(r["qualified"] for r in reading),
                                     "recovery_slope": float(covariance/truth.var()),
                                     "centered_errors_pp": {name: float((scores-v).std()) for name,v in reference.items()}}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resolution-check", action="store_true")
    args = parser.parse_args()
    start = perf_counter()
    inputs = (WAVEVECTOR, HIGH, RAW, TRANSFER, PROTOCOL, STRAIN_COMPARISON)
    wave, high, raw, transfer, protocol, strain = (json.loads(p.read_text()) for p in inputs)
    data = load_dataset(SCAN, args.archive, allowed_ids=(SCAN,), allow_author_crop=True)
    prior = transfer["scans"][SCAN]
    if data.archive_md5 != prior["archive_md5"]:
        raise ValueError("The source DS31 scan changed")
    nominal_fit = np.asarray(prior["whole_geometry"]["fit_scan"])
    yy, xx = np.indices(data.u_nm.shape[:2], dtype=float)
    positions = np.stack((xx, yy), axis=-1)*data.pixel_nm
    affine = positions@nominal_fit[:2]+nominal_fit[-1]
    warp, _ = smooth_geometry(data, nominal_fit, positions)
    registry = affine+warp
    scale = MODEL_A/data.lattice_nm
    grid, shift, _ = solve(scale*np.asarray(prior["whole_geometry"]["gradient_crystal"]), BASIS,
                           tuple(protocol["interface"]), prior["grid"])
    phase = ORIGIN+scale*registry
    # The original affine deformation is already in the measured reciprocal
    # vectors. Project only deviations from that affine background.
    model = (sample_disregistry(grid, shift, phase, grid.cell)-ORIGIN)/scale-affine
    edge = edge_geometry(high, raw, np.asarray(high["aggregate"]["higher_order_9"]["geometry_fit_parameters"]))
    theta = np.asarray(high["aggregate"]["higher_order_9"]["geometry_fit_parameters"])
    q_per_pixel = raw["source"]["detector_spacing_mrad"]*1e-3/electron_wavelength_nm(80000)
    sample_rotation = np.deg2rad(data.sample_rotation_degrees)
    orientation = np.asarray(wave["raw_to_scan_orthogonal"])
    frame = orientation@rotation(-sample_rotation)
    # Model vectors are in the nominal crystal frame. In the following
    # inner products, g @ frame maps the reciprocal vector into that frame.
    rows = [r for r in wave["cohorts"]["replay"]["raw_gradients_detector_pixels"]
            if r["mean"]["fit_qualified"] and 15 <= r["row"] <= 82]
    if args.pilot:
        rows = rows[::max(1,len(rows)//6)][:6]
    result = {
        "scope": "One-pass projected weak-phase coherent simulation of a continuum displacement field and the DS31 outer diffraction spots. Test phase variation across the probe and interference spillover onto the same exclusive arc geometry. Use the measured average reciprocal lattice, published convergence and effective probe, and a simple aperture. This is a microscope sensitivity check, with no atomic scattering or thickness model.",
        "limits": "Unknown aberrations, interlayer propagation, atomic form factors, tilt and detector transfer are excluded. The four relative phases test coherent contamination at identical layer amplitudes. Diffraction amplitudes of the high-order layers are not calibrated by the experiment.",
        "input_sha256": {p.name: digest(p) for p in (Path(__file__),*inputs)},
        "relative_phases_radians": COHERENCE_PHASES,
        "cases": {},
    }
    pixel = raw["source"]["dm4_scan_spacing_nm"]
    crop = data.metadata["author_unwrap_crop_origin_row_col"]
    # A positive isotropic calibration has gradient 0.003 I in crystal
    # coordinates. The row-vector position transform is the inverse of the
    # gradient convention used by native_gradients.
    crystal = (positions*(pixel/data.pixel_nm))@rotation(-sample_rotation)
    fields = {"rigid": np.zeros_like(model), "linear_calibration": INJECTED_DILATION*crystal, "full": model}
    gradients = native_gradients(data, fields["linear_calibration"], model, pixel)
    if not np.allclose(gradients["background"], INJECTED_DILATION*np.eye(2), rtol=0, atol=1e-12):
        raise ValueError("The calibration field does not have the intended crystal-frame strain")
    if args.resolution_check:
        cases = (("coarse", 48, 144, 0), ("oversampled_detector", 48, 384, 0),
                 ("larger_aperture_window", 96, 384, 0))
    elif args.pilot:
        cases = (("focused", 48, 144, 0), ("defocused", 48, 144, 12))
    else:
        cases = (("focused", 48, 192, 0), ("larger_aperture_window", 96, 384, 0))
    for name, size, pad, defocus in cases:
        case_rows = rows[:3] if args.resolution_check else rows
        case = evaluate(edge, theta, q_per_pixel, case_rows, fields, affine, pixel, crop,
                        orientation, frame, size=size, pad=pad, defocus_nm=defocus)
        case["comparison"] = compare_strain(case, strain["samples"]["replay"], gradients["full"], crop)
        result["cases"][name] = case
        result["wall_seconds"] = perf_counter()-start
        checkpoint_guard()
        atomic_json(args.output,result)
        print(name, case["affine_injection"], case["comparison"], flush=True)


if __name__ == "__main__":
    main()
