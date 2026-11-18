"""Predict high-order diffraction edges of smooth projected lattice distortions."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.ndimage import map_coordinates

from bilayer_relaxation import MONOLAYERS, sample_disregistry
from observe_bilayers import load_dataset
from run_detector_prediction import AUTHOR_BASIS
from run_ds31_phase_wavevector_check import electron_wavelength_nm
from run_heterobilayer_corrugation import BASIS, MODEL_A, ORIGIN
from run_heterobilayer_raw_arc import RADIAL_PIXELS, TEMPLATE_PIXELS, read_trace, solve_profiles
from run_heterobilayer_raw_ds31 import SCAN
from run_heterobilayer_raw_injection import HIGH, RAW, paired_matrices
from run_heterobilayer_ring_geometry import smooth_geometry
from run_heterobilayer_strain_observer import SETTINGS, TRANSFER
from run_heterobilayer_transfer import PROJECT, PROTOCOL, digest, solve
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


WAVEVECTOR = PROJECT / "ds31_phase_wavevector_check_v01.json"
GAUSSIAN_SAMPLES = 7


def edge_geometry(order, settings, theta):
    """Use the same exclusive arc support and count-independent geometry."""
    all_modes = np.asarray(order["higher_order_indices"])
    h, k = all_modes.T
    outer = h*h+h*k+k*k == 9
    modes, reference = paired_matrices(all_modes[outer], np.asarray(order["higher_order_nominal_centers_row_col"])[outer])
    g = modes @ (reference + theta[2:10].reshape(2, 2, 2))
    radius = 1.71/settings["source"]["detector_spacing_mrad"] + theta[10:12]
    angle = np.arange(32)*(2*np.pi/32)
    directions = np.column_stack((np.sin(angle), np.cos(angle)))
    origin = g[:, :, None, :] + radius[:, None, None, None]*directions
    separation = np.linalg.norm(origin[..., None, :] + RADIAL_PIXELS[:, None]*directions[:, None, :]
                                - g[::-1, :, None, None, :], axis=-1)
    layer, disk, azimuth = np.where(separation.min(axis=-1) > radius[::-1, None, None]+2.25)
    n = directions[azimuth]
    design = np.zeros((len(n), 12))
    design[:, :2] = n
    for i in range(2):
        use = layer == i
        design[use, 2+4*i:6+4*i] = np.einsum("ij,ik->ijk", modes[disk[use]], n[use]).reshape(-1, 4)
        design[use, 10+i] = 1
    return layer, disk, n, g, radius, reference, design


def native_gradients(data, reference, model, pixel_nm):
    """Build Cartesian gradients in the archive's crystal frame."""
    sample = rotation(-np.deg2rad(data.sample_rotation_degrees))
    output = {}
    for name, field in (("background", reference), ("full", model)):
        dy, dx = np.gradient(field, pixel_nm, axis=(0, 1))
        output[name] = np.stack((dx, dy), axis=-1) @ sample
    return output


def elastic_partition(first, second):
    """Estimate the larger-lattice sheet's share from the two elastic moduli."""
    a1, bulk1, shear1 = MONOLAYERS[first]
    a2, bulk2, shear2 = MONOLAYERS[second]
    area_ratio = (a1/a2)**2
    bulk2, shear2 = bulk2*area_ratio, shear2*area_ratio
    weights = ((bulk1+shear1)/(bulk1+shear1+bulk2+shear2), shear1/(shear1+shear2))
    return float(np.mean(weights))


def probe_nodes(sigma_nm, pixel_nm):
    coordinates, weights = hermgauss(GAUSSIAN_SAMPLES)
    shift = np.sqrt(2)*sigma_nm*coordinates/pixel_nm
    return np.stack(np.meshgrid(shift, shift, indexing="ij"), axis=-1).reshape(-1, 2), np.outer(weights, weights).ravel()/np.pi


def physical_sample(field, row, crop, sigma_nm, pixel_nm):
    count = row["column_count"]
    offset, weight = probe_nodes(sigma_nm, pixel_nm)
    x = np.arange(row["column_start"], row["column_start"]+count)-crop[1]
    y = np.full(count, row["row"]-crop[0])
    centers = np.column_stack((y, x))
    points = centers[:, None]+offset
    corners = points.reshape(-1, 2)
    h, w = field.shape[:2]
    if not np.all((corners >= 0) & (corners < np.array((h-1, w-1)))):
        raise ValueError("A synthetic probe extends outside the archive crop")
    values = np.stack([map_coordinates(field[..., i, j], corners.T, order=1, mode="nearest")
                       for i in range(2) for j in range(2)], axis=-1)
    values = values.reshape(count, len(weight), 2, 2)
    if not np.isfinite(values).all():
        raise ValueError("A synthetic probe has unsupported deformation values")
    return values, weight


def make_arc_profiles(geometry, matrices, weights, radial, width):
    """Average locally shifted disc edges within the probe and raw blocks."""
    layer, disk, n, g, _, _, _ = geometry
    local = matrices[..., layer, :, :]
    shift = np.einsum("...aij,aj->...ai", local, g[layer, disk])
    projected = np.einsum("...ai,ai->...a", shift, n)
    edge = 0.1 + 1/(1+np.exp((radial[None, None, None, :]-projected[..., None])/width))
    return np.mean(np.einsum("s,nsar->nar", weights, edge), axis=0)


def relative_matrices(deformation, partition, frame):
    # The first set of fitted detector spots is closer to the direct beam.
    # It corresponds to the larger lattice in the real-space convention.
    signs = np.array((partition, -(1-partition)))
    local = signs[:, None, None]*deformation[..., None, :, :]
    shift = np.linalg.inv(np.eye(2)+local).swapaxes(-1, -2)-np.eye(2)
    return frame @ shift @ frame.T


def simulate(geometry, theta, rows, field, sigma_nm, pixel_nm, crop, partition, frame, width):
    layer, disk, n, _, _, reference, design = geometry
    template = np.broadcast_to(0.1+1/(1+np.exp(TEMPLATE_PIXELS/width)), (len(n), len(TEMPLATE_PIXELS)))
    contrast = template[:, TEMPLATE_PIXELS < -1].mean(axis=1)-template[:, TEMPLATE_PIXELS > 1].mean(axis=1)
    template_arcs = (disk, None, n, template, contrast, design, reference, np.bincount(layer))
    values = []
    for row in rows:
        checkpoint_guard()
        sample, weight = physical_sample(field, row, crop, sigma_nm, pixel_nm)
        # Use the same published relative modulus split in both controls.
        matrices = relative_matrices(sample, partition, frame)
        profiles = make_arc_profiles(geometry, matrices, weight, RADIAL_PIXELS, width)
        fitted, quality = solve_profiles(profiles, template_arcs)
        mean = np.einsum("s,nsij->ij", weight, sample)/len(sample)
        values.append({"in_plane_trace_pp": float(50*np.trace(mean)),
                       "simulated_trace_pp": read_trace(theta+fitted, reference),
                       "profile_rmse": quality["weighted_profile_rmse"],
                       "row": row["row"], "column_start": row["column_start"],
                       "fit_converged": quality["converged"], "arcs_at_bound": quality["arcs_at_bound"]})
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    inputs = (WAVEVECTOR, HIGH, RAW, TRANSFER, SETTINGS, PROTOCOL)
    wave, high, raw, transfer, settings, protocol = (json.loads(p.read_text()) for p in inputs)
    data = load_dataset(SCAN, args.archive, allowed_ids=(SCAN,), allow_author_crop=True)
    reference = transfer["scans"][SCAN]
    if data.archive_md5 != reference["archive_md5"]:
        raise ValueError("The source displacement data changed")
    fit = np.asarray(reference["whole_geometry"]["fit_scan"])
    y, x = np.indices(data.u_nm.shape[:2], dtype=float)
    points = np.stack((x, y), axis=-1)*data.pixel_nm
    warp, _ = smooth_geometry(data, fit, points)
    nominal = points @ fit[:2]+fit[-1]
    background = nominal+warp
    scale = MODEL_A/data.lattice_nm
    interface = tuple(protocol["interface"])
    grid, motion, _ = solve(scale*np.asarray(reference["whole_geometry"]["gradient_crystal"]),
                            BASIS, interface, reference["grid"])
    phase = ORIGIN+scale*background
    field = (sample_disregistry(grid, motion, phase, grid.cell)-ORIGIN)/scale
    pixel_nm = raw["source"]["dm4_scan_spacing_nm"]
    sigma = settings["probe_fwhm_nm"]/np.sqrt(8*np.log(2))
    gradients = native_gradients(data, background, field, pixel_nm)
    affine = fit[:2].T @ rotation(-np.deg2rad(data.sample_rotation_degrees))/ (pixel_nm/data.pixel_nm)
    gradients = {k: g-affine for k, g in gradients.items()}
    crop = data.metadata["author_unwrap_crop_origin_row_col"]
    partition = elastic_partition(*interface[:2])
    config = high["aggregate"]["higher_order_9"]
    theta = np.asarray(config["geometry_fit_parameters"])
    geometry = edge_geometry(high, raw, theta)
    frame = np.asarray(wave["raw_to_scan_orthogonal"]) @ rotation(-np.deg2rad(data.sample_rotation_degrees))
    if not np.allclose(frame @ frame.T, np.eye(2)):
        raise ValueError("The detector frame must preserve distances")
    result = {
        "scope": "A kinematic incoherent Gaussian-probe calculation of the higher-order exclusive disk-edge readout. Predict local reciprocal shifts from projected strain about the mean affine lattice. Propagate the spatial variation across the finite probe and average the ten raw positions in each block. Invert the resulting profiles with the same joint arc readout and compare its centered strain with the known input. Model the pure instrument averaging effect; the specimen-dependent intensity and projection are not simulated.",
        "input_sha256": {p.name: digest(p) for p in (Path(__file__), *inputs)},
        "probe_sigma_nm": sigma, "gaussian_nodes_per_axis": GAUSSIAN_SAMPLES,
        "relative_displacement_fraction_of_first_layer": partition,
        "reciprocal_pixels_per_cycle_nm": 1/(raw["source"]["detector_spacing_mrad"]*1e-3/electron_wavelength_nm(80000)),
        "cases": {},
    }
    for label, cohort in wave["cohorts"].items():
        rows = [r for r in cohort["raw_gradients_detector_pixels"]
                if r["mean"]["fit_qualified"] and 12 <= r["row"] <= 85]
        cases = {}
        for key in ("full", "background"):
            for width in (0.55, 0.85, 1.15):
                checkpoint_guard()
                samples = simulate(geometry, theta, rows, gradients[key], sigma, pixel_nm, crop, partition, frame, width)
                valid = np.array([s["fit_converged"] and s["arcs_at_bound"] == 0 for s in samples])
                expected = np.array([r["in_plane_trace_pp"] for r in samples])
                actual = np.array([r["simulated_trace_pp"] for r in samples])
                metrics = {}
                for mode, used in (("all_finite", np.ones(len(samples), dtype=bool)), ("qualified", valid)):
                    if used.sum() < 20:
                        raise ValueError("Too few qualified synthetic blocks")
                    ideal, observed = expected[used], actual[used]
                    signal = ideal-ideal.mean()
                    metrics[mode] = {"block_count": int(used.sum()), "ideal_strain_rms_pp": float(ideal.std()),
                                     "recovered_strain_rms_pp": float(observed.std()),
                                     "centered_rmse_pp": float((observed-ideal).std()),
                                     "recovery_slope": float(signal@(observed-observed.mean())/(signal@signal))}
                cases[f"{key}_edge_width_{width}"] = {
                    **metrics, "maximum_profile_rmse": max(row["profile_rmse"] for row in samples),
                    "unqualified_blocks": [r for r, keep in zip(samples, valid) if not keep],
                }
        result["cases"][label] = cases
        result["wall_seconds"] = perf_counter()-start
        checkpoint_guard()
        atomic_json(args.output, result)
        print(label, cases, flush=True)


if __name__ == "__main__":
    main()
