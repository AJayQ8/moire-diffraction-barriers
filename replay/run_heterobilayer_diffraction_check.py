"""Score the fixed mixed-material model against integrated diffraction images."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter

from author_detector_replay import detector_intensity, normalize_channels, read_detector_fit, rms
from author_release import RELEASE_COMMIT, load_author_release
from bilayer_relaxation import MONOLAYERS, sample_disregistry
from observe_bilayers import _good_interior_triangle_mask, _inside_polygon_rc, load_dataset
from run_detector_prediction import AUTHOR_BASIS, apply_spatial
from run_heterobilayer_transfer import PROJECT, PROTOCOL, digest, fingerprints, solve, verify_freeze
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


FREEZE = PROJECT / "heterobilayer_transfer_freeze_v01.json"


def read_native_detector(archive, scan, author):
    disk, coefficients, field, payload_hash = read_detector_fit(archive, scan, "fit", author)
    measured = normalize_channels(disk.df_set(), author)
    modes = disk.clean_normgset(sanity_plot=False)
    if coefficients.shape != (len(modes), 3) or measured.shape[:2] != field.shape[:2]:
        raise ValueError(f"{scan}: expected a three-coefficient native fit")
    # The archive marks these native files as Cartesian. Verify that directly
    # against the alternative lattice representation of the saved fit.
    lattice = field @ np.linalg.inv(AUTHOR_BASIS).T
    fitted = detector_intensity(lattice, modes, coefficients)
    cartesian_error = rms(measured, fitted)
    wrong_error = rms(measured, detector_intensity(field, modes, coefficients))
    if not cartesian_error < wrong_error:
        raise ValueError(f"{scan}: saved native fit fails the Cartesian-basis check")
    return measured, fitted, modes, coefficients, {
        "fit_sha256": payload_hash, "native_fit_rmse": cartesian_error,
        "native_fit_rmse_if_read_as_lattice": wrong_error,
    }


def register_frame(dataset, fitted, modes, coefficients):
    """Determine the detector coordinate convention from archived fields alone."""
    y, x = dataset.metadata.get("author_unwrap_crop_origin_row_col", (0, 0))
    height, width = dataset.u_nm.shape[:2]
    crop = np.s_[y:y + height, x:x + width]
    target = fitted[crop][2:-2:2, 2:-2:2]
    inverse = np.linalg.inv(AUTHOR_BASIS)
    best = None
    for turns in range(4):
        for spatial_reflection in (False, True):
            field = apply_spatial(dataset.u_nm, turns, spatial_reflection)[2:-2:2, 2:-2:2] / dataset.lattice_nm
            finite = np.isfinite(field).all(axis=-1)
            if field.shape[:2] != target.shape[:2]:
                continue
            for vector_reflection in (False, True):
                mirror = np.diag((-1, 1)) if vector_reflection else np.eye(2)
                for k in range(6):
                    transform = inverse @ rotation(k * np.pi / 3) @ mirror
                    prediction = detector_intensity(field[finite] @ transform.T, modes, coefficients)
                    error = rms(target[finite], prediction)
                    if best is None or error < best[0]:
                        best = error, turns, spatial_reflection, k, vector_reflection, transform
    if best is None:
        raise ValueError("The native and unwrapped diffraction rasters cannot be registered")
    loss, turns, reflect, k, vector_reflect, transform = best
    return crop, turns, reflect, transform, {"archived_field_registration_rmse": loss, "spatial_turns": turns,
                                           "spatial_reflection": reflect, "vector_rotation_60deg_steps": k,
                                           "vector_reflection": vector_reflect}


def run_scan(data, prior, amplitude, settings, interface, measured, fitted, modes, coefficients):
    crop, turns, reflect, transform, alignment = register_frame(data, fitted, modes, coefficients)
    measured, fitted = measured[crop], fitted[crop]
    shape = data.u_nm.shape[:2]
    yy, xx = np.indices(shape, dtype=float)
    points = np.stack((xx, yy), axis=-1) * data.pixel_nm
    mask, triangles = _good_interior_triangle_mask(data)
    polygons = np.stack([_inside_polygon_rc(shape, data.centers_rc_px[list(tri)]) for tri in triangles])
    model_a = MONOLAYERS[interface[0]][0]
    basis = model_a * np.array(((1, 0.5), (0, np.sqrt(3) / 2)))
    scale = model_a / data.lattice_nm
    origin = basis @ np.full(2, 2 / 3)
    gradient = np.asarray(prior["whole_geometry"]["gradient_crystal"])
    grid, shift, _ = solve(scale * gradient, basis, interface, prior["grid"])
    sigma = settings["probe_fwhm_nm"] / (np.sqrt(8 * np.log(2)) * data.pixel_nm)

    def forward(field, probe_sigma):
        field = apply_spatial(field, turns, reflect) / data.lattice_nm
        signal = detector_intensity(field @ transform.T, modes, coefficients)
        return gaussian_filter(signal, (probe_sigma, probe_sigma, 0), mode="nearest") if probe_sigma else signal

    valid = data.source_valid_mask & np.isfinite(data.u_nm).all(axis=-1)
    valid &= gaussian_filter(valid.astype(float), settings["smoothing_sigma_pixels"], mode="constant") > 0.97
    valid[:4] = valid[-4:] = valid[:, :4] = valid[:, -4:] = False
    center_distance2 = np.full(shape, np.inf)
    for row, col in data.centers_rc_px:
        center_distance2 = np.minimum(center_distance2, (yy - row)**2 + (xx - col)**2)
    far = center_distance2 > (2 * settings["smoothing_sigma_pixels"])**2
    experimental = forward(data.u_nm, 0)
    controls = {"archive_unwrapped": experimental, "archived_native_fit": fitted}
    whole_fit = np.asarray(prior["whole_geometry"]["fit_scan"])
    rigid_whole = points @ whole_fit[:2] + whole_fit[-1]
    phase_whole = origin + scale * rigid_whole
    relaxed_whole = (sample_disregistry(grid, shift, phase_whole, grid.cell) - origin) / scale
    controls.update(whole_rigid=forward(rigid_whole, sigma), whole_relaxed=forward(relaxed_whole, sigma))
    h, k = modes.T
    ring_index = np.rint(h**2 + h * k + k**2).astype(int)
    rings = {"first": ring_index == 1, "second": ring_index == 3}
    if any(channels.sum() != 6 for channels in rings.values()):
        raise ValueError("The comparison requires six integrated diffraction disks in each ring")
    halves = []
    for axis, partition in prior["partitions"].items():
        regions = [polygons[row["triangles"]].any(axis=0) for row in partition]
        for i, row in enumerate(partition):
            fit = np.asarray(row["geometry"]["fit_scan"])
            rigid = points @ fit[:2] + fit[-1]
            phase = origin + scale * rigid
            relaxation = (sample_disregistry(grid, shift, phase, np.asarray(row["geometry"]["cell_nm"])) - phase) / scale
            signals = {**controls, "regional_rigid": forward(rigid, sigma),
                       "regional_relaxed": forward(rigid + relaxation, sigma),
                       "regional_scaled": forward(rigid + amplitude * relaxation, sigma)}
            region = apply_spatial(mask & valid & far & regions[i] & ~regions[1 - i], turns, reflect)
            region &= np.isfinite(experimental).all(axis=-1)
            if not np.any(region):
                raise ValueError("No common diffraction score pixels")
            errors = {name: rms(measured[region], signal[region]) for name, signal in signals.items()}
            ring_errors = {ring: {name: rms(measured[region][:, channels], signal[region][:, channels])
                                  for name, signal in signals.items()} for ring, channels in rings.items()}
            halves.append({"axis": axis, "half": i, "pixel_count": int(region.sum()), "rmse": errors,
                           "ring_rmse": ring_errors})
    return {"encapsulation": prior["encapsulation"], "amplitude_from_setup_device": amplitude,
            "alignment": alignment, "halves": halves,
            "mean_half_ring_rmse": {ring: {name: float(np.mean([row["ring_rmse"][ring][name] for row in halves]))
                                          for name in controls.keys() | {"regional_rigid", "regional_relaxed", "regional_scaled"}}
                                     for ring in rings},
            "mean_half_rmse": {name: float(np.mean([row["rmse"][name] for row in halves])) for name in halves[0]["rmse"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    author = load_author_release(args.release, args.dependencies)
    protocol = json.loads(PROTOCOL.read_text())
    settings = json.loads((PROJECT / "registry_protocol_v01.json").read_text())
    verify_freeze(FREEZE, fingerprints(), protocol["devices"]["development"])
    amplitude_path = PROJECT / "heterobilayer_capping_development_v01.json"
    setup = json.loads(amplitude_path.read_text())
    if setup["transfer_freeze_sha256"] != digest(FREEZE):
        raise ValueError("The amplitude diagnostic used a different transfer protocol")
    amplitudes = setup["fitted_amplitudes"]["by_encapsulation"]
    result = {"author_commit": RELEASE_COMMIT, "input_sha256": {p.name: digest(p) for p in (
        Path(__file__), FREEZE, amplitude_path, PROJECT / "author_detector_replay.py", PROJECT / "author_release.py")},
        "method": "Retrospective forward comparison to the measured normalized integrated disks from the two Figure 4 devices. Use saved native detector coefficients and identify a lattice-symmetry alignment from archived measured fields before evaluating any model prediction. The detector calibration and orientation use the same scan. Compare the same regional pixels for every model. The published Gaussian probe is applied to simulated intensities. Report all disks together and separate first- and second-ring scores.",
        "scans": {}}
    start = perf_counter()
    for batch, cases in protocol["devices"].items():
        path = PROJECT / f"heterobilayer_transfer_{batch}_v01.json"
        previous = json.loads(path.read_text())
        if (previous["input_sha256"] != fingerprints() or set(previous["scans"]) != set(cases)
                or previous["failed_scans"]):
            raise ValueError("The earlier device prediction is incomplete or has changed")
        result["input_sha256"][path.name] = digest(path)
        for scan in cases:
            checkpoint_guard()
            data = load_dataset(scan, args.archive, allowed_ids=cases, allow_author_crop=True)
            prior = previous["scans"][scan]
            if data.archive_md5 != prior["archive_md5"]:
                raise ValueError(f"{scan}: archive changed")
            measured, fitted, modes, coefficients, calibration = read_native_detector(args.archive, scan, author)
            row = run_scan(data, prior, amplitudes[prior["encapsulation"]], settings, tuple(protocol["interface"]),
                           measured, fitted, modes, coefficients)
            result["scans"][scan] = {"batch": batch, "calibration": calibration, **row}
            result["wall_seconds"] = perf_counter() - start
            checkpoint_guard()
            atomic_json(args.output, result)
            print(scan, calibration, row["alignment"], row["mean_half_rmse"], flush=True)


if __name__ == "__main__":
    main()
