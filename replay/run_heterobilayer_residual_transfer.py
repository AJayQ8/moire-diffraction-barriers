"""Test whether a periodic displacement residual repeats across the two devices."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from bilayer_relaxation import MONOLAYERS, sample_disregistry
from observe_bilayers import _good_interior_triangle_mask, _inside_polygon_rc, load_dataset
from run_heterobilayer_transfer import PROJECT, PROTOCOL, blur, digest, fingerprints, solve, verify_freeze
from shared_compute import atomic_json, checkpoint_guard


FREEZE = PROJECT / "heterobilayer_transfer_freeze_v01.json"
SETTINGS = PROJECT / "registry_protocol_v01.json"


def rms_vector(field):
    return float(np.sqrt(np.mean(np.sum(field**2, axis=-1))))


def regional_residuals(data, row, settings, interface, amplitude, expected):
    shape = data.u_nm.shape[:2]
    yy, xx = np.indices(shape, dtype=float)
    points = np.stack((xx, yy), axis=-1) * data.pixel_nm
    mask, triangles = _good_interior_triangle_mask(data)
    polygons = np.stack([_inside_polygon_rc(shape, data.centers_rc_px[list(tri)]) for tri in triangles])
    basis = data.lattice_nm * np.array(((1, 0.5), (0, np.sqrt(3) / 2)))
    inverse = np.linalg.inv(basis)
    scale = MONOLAYERS[interface[0]][0] / data.lattice_nm
    model_basis = scale * basis
    origin = model_basis @ np.full(2, 2 / 3)
    gradient = np.asarray(row["whole_geometry"]["gradient_crystal"])
    grid, shift, _ = solve(scale * gradient, model_basis, interface, row["grid"])
    probe = settings["probe_fwhm_nm"] / np.sqrt(8 * np.log(2))
    sigma = settings["smoothing_sigma_pixels"]
    valid = data.source_valid_mask & np.isfinite(data.u_nm).all(axis=-1)
    weights = gaussian_filter(valid.astype(float), sigma, mode="constant")
    distance2 = np.full(shape, np.inf)
    for y, x in data.centers_rc_px:
        distance2 = np.minimum(distance2, (yy - y)**2 + (xx - x)**2)
    primary = mask & valid & (weights > 0.97) & (distance2 > (2 * sigma)**2)
    halves = []
    for axis, rows in row["partitions"].items():
        regions = [polygons[half["triangles"]].any(axis=0) for half in rows]
        for i, half in enumerate(rows):
            fit = np.asarray(half["geometry"]["fit_scan"])
            rigid = points @ fit[:2] + fit[-1]
            phase = origin + scale * rigid
            predicted = (sample_disregistry(grid, shift, phase, np.asarray(half["geometry"]["cell_nm"]), probe) - phase) / scale
            support = primary & regions[i] & ~regions[1 - i]
            displacement = blur(predicted, valid, weights, sigma)[support]
            observed = blur(data.u_nm - rigid, valid, weights, sigma)[support]
            halves.append({"axis": axis, "half": i, "fractional": (rigid[support] @ inverse.T) % 1.0,
                           "model": displacement, "residual": observed - amplitude * displacement})
            if support.sum() != half["primary"]["pixel_count"]:
                raise ValueError(f"{data.dataset_id}: previous score support changed")
    base = np.mean([rms_vector(half["residual"]) for half in halves])
    if not np.isclose(base, expected, atol=1e-10, rtol=0):
        raise ValueError(f"{data.dataset_id}: amplitude-only baseline changed")
    return halves


def residual_template(halves, size):
    weight = np.zeros((size, size))
    total = np.zeros((size, size, 2))
    for half in halves:
        values = half["residual"]
        indices = tuple((half["fractional"] * size).astype(int).T)
        np.add.at(weight, indices, 1 / len(values))
        np.add.at(total, indices, (values - values.mean(axis=0)) / len(values))
    # The fractional bins are periodic. Use one bin of smoothing on both
    # the accumulated values and counts, without filling unsupported bins.
    weight = gaussian_filter(weight, 1.0, mode="wrap")
    total = gaussian_filter(total, (1.0, 1.0, 0), mode="wrap")
    return np.divide(total, weight[..., None], out=np.zeros_like(total), where=weight[..., None] > 0)


def sample_correction(template, half):
    coords = template.shape[0] * half["fractional"].T - 0.5
    correction = np.stack([map_coordinates(template[..., i], coords, order=1, mode="grid-wrap") for i in range(2)], axis=-1)
    correction -= correction.mean(axis=0)
    model = half["model"] - half["model"].mean(axis=0)
    # The transfer comparison is for residual shape. Remove any correction
    # that simply changes the offset or amplitude of the existing prediction.
    return correction - model * np.sum(model * correction) / np.sum(model**2)


def evaluate(template, half):
    residual = half["residual"]
    correction = sample_correction(template, half)
    return {"pixel_count": len(residual), "amplitude_only_rmse_nm": rms_vector(residual),
            "with_periodic_correction_rmse_nm": rms_vector(residual - correction),
            "correction_rms_nm": rms_vector(correction),
            "residual_dot_correction_nm2": float(np.mean(np.sum(residual * correction, axis=-1)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    settings = json.loads(SETTINGS.read_text())
    verify_freeze(FREEZE, fingerprints(), protocol["devices"]["development"])
    paths = {batch: PROJECT / f"heterobilayer_capping_{batch}_v01.json" for batch in protocol["devices"]}
    previous = {batch: json.loads(path.read_text()) for batch, path in paths.items()}
    amplitudes = previous["development"]["fitted_amplitudes"]["by_encapsulation"]
    start = perf_counter()
    result = {"method": "Retrospective diagnostic. Phase-average the remaining vector error in the higher-twist device after the capping-specific amplitude prediction. Apply that correction at the same fractional registry on the lower-twist device. Remove its mean and projection onto the model using target geometry and model predictions. Use a shared template as a control. Repeat on 8, 12 and 16 bins per period. Score exactly the same pixels as the amplitude-only comparison.",
              "input_sha256": {p.name: digest(p) for p in (Path(__file__), FREEZE, *paths.values())}, "scans": {}}
    samples = {}
    for batch, cases in protocol["devices"].items():
        transfer = PROJECT / f"heterobilayer_transfer_{batch}_v01.json"
        rows = json.loads(transfer.read_text())["scans"]
        if digest(transfer) != previous[batch]["frozen_score_sha256"] or set(rows) != set(cases):
            raise ValueError("The transfer result used by the amplitude diagnostic changed")
        result["input_sha256"][transfer.name] = digest(transfer)
        samples[batch] = {}
        for scan in cases:
            checkpoint_guard()
            dataset = load_dataset(scan, args.archive, allowed_ids=cases, allow_author_crop=True)
            row = rows[scan]
            if dataset.archive_md5 != row["archive_md5"]:
                raise ValueError(f"{scan}: archive changed")
            cap = row["encapsulation"]
            reference = previous[batch]["scans"][scan]["primary"]["average_half_rmse_nm"]["by_encapsulation"]
            samples[batch][cap] = (scan, regional_residuals(dataset, row, settings, tuple(protocol["interface"]),
                                                            amplitudes[cap], reference))
    for size in (8, 12, 16):
        setup = samples["development"]
        templates = {cap: residual_template(halves, size) for cap, (_, halves) in setup.items()}
        shared = residual_template([half for _, halves in setup.values() for half in halves], size)
        for cap, (scan, halves) in samples["held_out"].items():
            readings = {name: [evaluate(template, half) for half in halves] for name, template in (
                ("matching_capping", templates[cap]), ("shared_across_capping", shared))}
            summary = {name: {key: float(np.mean([half[key] for half in values])) for key in values[0]}
                       for name, values in readings.items()}
            result["scans"].setdefault(scan, {"encapsulation": cap, "grids": {}})["grids"][size] = {
                "halves": readings, "mean_over_halves": summary}
    result["wall_seconds"] = perf_counter() - start
    checkpoint_guard()
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
