"""Measure wall contrast widths and positions on integrated Bragg disks."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi

from bilayer_relaxation import STACKING, relax, sample_disregistry
from observe_bilayers import _all_triangles, load_dataset
from run_detector_prediction import load_detector, match_archive_orientation
from run_wall_readout_check import intensity, local_geometry
from shared_compute import atomic_json, checkpoint_guard
from wall_registry_fit import detrended, ring_widths, wall_detector


SCANS = ("DS1", "DS2", "DS7", "DS10")
INTERFACE = ("MoS2", "MoS2", "P")
PROBE_FWHM_NM = 1.25
PROFILE_SMOOTHING_NM = 0.5


def to_scan_frame(field, turns, reflection):
    result = np.rot90(field, -turns, axes=(0, 1))
    return result[:, ::-1] if reflection else result


def camera_field(grid, relative, rigid, reciprocal, scale, offset, pixel_nm):
    displacement = sample_disregistry(grid, relative, rigid, grid.cell)
    field = intensity(displacement, reciprocal, scale, offset)
    sigma_px = PROBE_FWHM_NM / (np.sqrt(8 * np.log(2)) * pixel_nm)
    return ndi.gaussian_filter(field, (sigma_px, sigma_px, 0), mode="nearest")


def peak_width(distance, profile):
    curve, noise = detrended(distance, profile, PROFILE_SMOOTHING_NM)
    peak = int(np.argmax(np.where(np.abs(distance) <= 0.70 * distance[-1], curve, -np.inf)))
    height = float(curve[peak])
    width = None
    if height > 0:
        left = np.flatnonzero(curve[:peak] < 0.5 * height)
        right = np.flatnonzero(curve[peak + 1:] < 0.5 * height) + peak + 1
        if left.size and right.size:
            lo, hi = left[-1], right[0]
            start = np.interp(0.5 * height, curve[lo:lo + 2], distance[lo:lo + 2])
            stop = np.interp(0.5 * height, curve[hi - 1:hi + 1][::-1], distance[hi - 1:hi + 1][::-1])
            width = float(stop - start)
    return {"peak_position_nm": float(distance[peak]), "contrast_width_nm": width,
            "peak_contrast": height, "tail_noise_mad": noise,
            "contrast_to_tail_noise": height / noise if noise > 1e-10 else None}


def split_channels(modes):
    h, k = modes.T
    ring = h**2 + k**2 + h * k
    first, second = [], []
    for mode in np.unique(modes, axis=0):
        members = np.flatnonzero(np.all(modes == mode, axis=1))
        if len(members) == 2:
            first.append(members[0])
            second.append(members[1])
    return {"combined": np.arange(len(modes)), "first_ring": np.flatnonzero(ring == 1),
            "second_ring": np.flatnonzero(ring == 3),
            "friedel_a": np.asarray(first, dtype=int), "friedel_b": np.asarray(second, dtype=int)}


def measure_sections(dataset, fit, basis, reciprocal, scale, offset, fields, channels):
    edges = Counter()
    for i, j, k in _all_triangles(dataset.adjacency_type):
        edges.update(tuple(sorted(pair)) for pair in ((i, j), (j, k), (i, k)))
    domain = basis @ np.full(2, 1 / 3)
    domain_intensity = intensity(domain, reciprocal, scale, offset)
    inverse_basis = np.linalg.inv(basis)
    sections = []
    for (i, j), count in edges.items():
        if count != 2:
            continue
        start, stop = dataset.centers_rc_px[[i, j]]
        edge = stop - start
        length = np.linalg.norm(edge)
        if length == 0:
            continue
        # A wall's halfway registry is half the lattice translation between
        # its two junctions. This fixes the contrast sign and disk weights.
        phase_edge = (edge[::-1] * dataset.pixel_nm) @ fit[:2]
        lattice_edge = np.rint(phase_edge @ inverse_basis.T) @ basis.T
        if not np.isclose(np.linalg.norm(lattice_edge), dataset.lattice_nm, atol=1e-5):
            continue
        contrast = intensity(0.5 * lattice_edge, reciprocal, scale, offset) - domain_intensity
        normal = np.array((-edge[1], edge[0])) / length
        reach = min(11.0, 0.32 * length * dataset.pixel_nm)
        distance = np.linspace(-reach, reach, 2 * int(np.ceil(reach / 0.25)) + 1)
        along = start + np.linspace(0.30, 0.70, 7)[:, None] * edge
        points = along[:, None, :] + distance[None, :, None] * normal / dataset.pixel_nm
        coordinates = points.transpose(2, 0, 1)
        valid = ndi.map_coordinates(dataset.source_valid_mask.astype(float), coordinates, order=1) > 0.999
        inside = ((points[..., 0] >= 4) & (points[..., 0] <= dataset.u_nm.shape[0] - 5)
                  & (points[..., 1] >= 4) & (points[..., 1] <= dataset.u_nm.shape[1] - 5))
        complete = np.all(valid & inside, axis=1)
        if np.count_nonzero(complete) < 4:
            continue
        coordinates = coordinates[:, complete]
        samples = {name: np.stack([ndi.map_coordinates(field[..., i], coordinates, order=1)
                                    for i in range(field.shape[-1])], axis=-1)
                   for name, field in fields.items()}
        sections.append({"centers": [i, j], "sector": int(dataset.adjacency_type[i, j]),
                         "half_window_nm": reach, "complete_strips": int(complete.sum()), "groups": {}})
        for group, indices in channels.items():
            weights = contrast[indices]
            norm = weights @ weights
            if norm < 1e-8:
                continue
            measurements = {}
            for name, profile in samples.items():
                strips = (profile[..., indices] @ weights) / norm
                measurements[name] = {
                    "average_profile": peak_width(distance, strips.mean(axis=0)),
                    "individual_strips": [peak_width(distance, strip) for strip in strips],
                }
            sections[-1]["groups"][group] = measurements
        rings = (channels["first_ring"], channels["second_ring"])
        sections[-1]["registry_profile_fits"] = ring_widths(
            distance, samples["measured"].mean(axis=0), samples["fixed_model"].mean(axis=0), lattice_edge,
            reciprocal, scale, offset, contrast, rings, PROBE_FWHM_NM, PROFILE_SMOOTHING_NM)
    return sections


def audit_scan(archive, scan, row):
    dataset = load_dataset(scan, archive, allowed_ids=(scan,))
    raw, a, pixel, _, _ = dataset
    measured, fitted, modes, scale, offset, _, fit_rmse, fit_hash = load_detector(archive, scan)
    if measured.shape[:2] != raw.shape[:2]:
        raise ValueError(f"{scan}: displacement and diffraction shapes differ")
    alignment, turns, reflection, reciprocal, _, _ = match_archive_orientation(raw, fitted, a, modes, scale, offset)
    if alignment > 0.20:
        raise ValueError(f"{scan}: detector registration error {alignment:.3f}")
    basis = a * np.array(((1, 0.5), (0, np.sqrt(3) / 2)))
    fit = np.asarray(row["fit_scan"])
    yy, xx = np.indices(raw.shape[:2], dtype=float) * pixel
    points = np.stack((xx, yy), axis=-1)
    rigid = points @ fit[:2] + fit[-1]
    warp, _ = local_geometry(dataset, points, fit, basis)
    fields = {"measured": to_scan_frame(measured, turns, reflection),
              "archive_fit": to_scan_frame(fitted, turns, reflection)}
    checks = {}
    for strength, name in ((1.0, "fixed_model"), (0.75, "adhesion_075"), (0.5, "adhesion_050")):
        coefficients = strength * np.array(STACKING[INTERFACE])
        grid, relative, solve = relax(np.asarray(row["gradient_crystal"]), basis, INTERFACE, row["grid_size"],
                                     checkpoint_guard, coefficients)
        if not solve["success"] or solve["force_rms_eV_per_nm"] > 2e-4:
            raise RuntimeError(f"{scan}: {name} did not converge")
        fields[name] = camera_field(grid, relative, rigid, reciprocal, scale, offset, pixel)
        checks[name] = solve
        if name == "fixed_model":
            fields["local_geometry_warp"] = camera_field(grid, relative, rigid + warp, reciprocal, scale, offset, pixel)
    channels = split_channels(modes)
    return {"fit_sha256": fit_hash, "alignment_rmse": alignment, "archived_fit_calibration_rmse": fit_rmse,
            "channel_counts": {name: len(ids) for name, ids in channels.items()}, "solves": checks,
            "wall_sections": measure_sections(dataset, fit, basis, reciprocal, scale, offset, fields, channels)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    inputs = [project / name for name in ("registry_development_v01.json", "registry_held_out_v01.json")]
    geometry = {scan: row for path in inputs for scan, row in json.loads(path.read_text())["scans"].items()}
    distance = np.linspace(-10, 10, 81)
    synthetic = np.exp(-0.5 * ((distance - 1.25) / (4 / np.sqrt(8 * np.log(2))))**2)
    smoke = peak_width(distance, synthetic + 0.25 + 0.03 * distance)
    if abs(smoke["peak_position_nm"] - 1.25) > 0.25 or abs(smoke["contrast_width_nm"] - 4.17) > 0.1:
        raise RuntimeError(f"Wall width reader failed its synthetic check: {smoke}")
    a = 0.315
    basis = a * np.array(((1, 0.5), (0, np.sqrt(3) / 2)))
    lattice_edge = basis[:, 0]
    reciprocal = np.array(((1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2))) @ np.linalg.inv(basis)
    scale, offset = np.ones(6), np.zeros(6)
    contrast = intensity(lattice_edge / 2, reciprocal, scale, offset) - intensity(
        basis @ np.full(2, 1 / 3), reciprocal, scale, offset)
    generated = wall_detector(distance, 1.25, 4.0, lattice_edge, "sine_gordon",
                              reciprocal, scale, offset, PROBE_FWHM_NM)
    replay = ring_widths(distance, generated, generated, lattice_edge, reciprocal, scale, offset,
                         contrast, (np.arange(3), np.arange(3, 6)), PROBE_FWHM_NM, PROFILE_SMOOTHING_NM)
    fitted = replay["sine_gordon"]["measured"]
    if abs(fitted["center_nm"] - 1.25) > 0.01 or abs(fitted["intrinsic_gradient_fwhm_nm"] - 4.0) > 0.01:
        raise RuntimeError(f"The two-ring fit failed its synthetic check: {fitted}")
    start = perf_counter()
    result = {"scans": {}, "probe_fwhm_nm": PROBE_FWHM_NM, "profile_smoothing_nm": PROFILE_SMOOTHING_NM,
              "synthetic_profile_check": smoke, "synthetic_registry_check": fitted,
              "description": "Direct diffraction contrast across internal wall chords. The measured and simulated camera profiles use the same chord, background subtraction and smoothing. Contrast width and inferred disregistry-gradient width are separate observables. The latter assumes a single straight monotone wall and includes prediction across diffraction rings.",
              "sha256": {path.name: sha256(path.read_bytes()).hexdigest()
                         for path in (*inputs, Path(__file__), project / "bilayer_relaxation.py",
                                      project / "run_detector_prediction.py", project / "run_wall_readout_check.py",
                                      project / "wall_registry_fit.py")}}
    for scan in SCANS:
        checkpoint_guard()
        result["scans"][scan] = audit_scan(args.archive, scan, geometry[scan])
        result["wall_seconds"] = perf_counter() - start
        atomic_json(args.output, result)
        print(scan, len(result["scans"][scan]["wall_sections"]), "wall sections", flush=True)


if __name__ == "__main__":
    main()
