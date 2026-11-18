"""Check whether the public raw scan can resolve local reciprocal lattices."""

import argparse
import json
import re
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

from author_detector_replay import read_detector_fit
from author_release import load_author_release
from bilayer_relaxation import sample_disregistry
from observe_bilayers import _good_interior_triangle_mask, load_dataset
from run_detector_prediction import AUTHOR_BASIS
from run_heterobilayer_corrugation import BASIS, MODEL_A, ORIGIN, dilation
from run_heterobilayer_ring_geometry import smooth_geometry
from run_heterobilayer_transfer import PROJECT, PROTOCOL, digest, fingerprints, solve, verify_freeze
from shared_compute import atomic_json, checkpoint_guard


SCAN = "DS31"
WINDOW = 25
ROWS_PER_CHUNK = 40
FIRST_COLUMN = 20
COORDINATES = np.stack(np.meshgrid(np.arange(-WINDOW, WINDOW + 1),
                                    np.arange(-WINDOW, WINDOW + 1), indexing="ij"), axis=-1)


def reciprocal_matrix(positions, indices):
    """Map primitive reciprocal components to the detector raster."""
    design = np.column_stack((np.ones(len(indices)), indices))
    fit = np.linalg.lstsq(design, positions, rcond=None)[0]
    return fit[0], fit[1:], float(np.sqrt(np.mean(np.sum((design @ fit - positions)**2, axis=-1))))


def load_counts(ranges, centers, radii, archived):
    source = json.loads((ranges / "source.json").read_text())
    if (source["shape"] != [100, 100, 864, 864] or source["raw_dtype"] != "uint8"
            or source["starting_column"] != FIRST_COLUMN or source["column_count"] != ROWS_PER_CHUNK):
        raise ValueError("Unexpected DM4 layout or range sampling")
    rounded = np.rint(centers).astype(int)
    masks = np.linalg.norm(COORDINATES[None] + rounded[:, None, None] - centers[:, None, None], axis=-1) < radii[:, None, None]
    rows = []
    hashes = {}
    for row in source["sample_rows"]:
        path = ranges / f"row-{row}.bin"
        checkpoint_guard()
        frame_bytes = int(np.prod(source["shape"][-2:]))
        start = source["raw_offset_bytes"] + (100 * row + FIRST_COLUMN) * frame_bytes
        length = ROWS_PER_CHUNK * frame_bytes
        receipt = (ranges / f"row-{row}.headers").read_text().lower()
        expected = f"bytes {start}-{start + length - 1}/{source['archive_bytes']}"
        if path.stat().st_size != length or not re.search(rf"^content-range:\s*{expected}$", receipt, re.M):
            raise ValueError(f"Row {row} does not have the expected recorded byte range")
        hashes[path.name] = digest(path)
        frames = np.memmap(path, mode="r", dtype=np.uint8, shape=(ROWS_PER_CHUNK, 864, 864))
        cropped = np.stack([frames[:, y-WINDOW:y+WINDOW+1, x-WINDOW:x+WINDOW+1]
                            for y, x in rounded], axis=1)
        check = np.sum(cropped * masks, axis=(-1, -2))
        expected_counts = archived[:, row, FIRST_COLUMN:FIRST_COLUMN + ROWS_PER_CHUNK].T
        corr = float(np.corrcoef(check.ravel(), expected_counts.ravel())[0, 1])
        difference = np.sqrt(np.mean((check - expected_counts)**2)) / np.mean(expected_counts)
        if corr < 0.98 or difference > 0.12:
            raise ValueError(f"Row {row} does not reproduce the archived integrated counts")
        rows.append({"row": row, "counts": cropped, "archive_correlation": corr,
                     "archive_normalized_rms": float(difference)})
    return source, rounded, rows, hashes


def masks_from_pair(theta, positions, coordinates, radius):
    """Model a small common detector shift plus independent reciprocal matrices."""
    drift = theta[:2]
    lattices = theta[2:10].reshape(2, 2, 2)
    center = positions @ lattices + drift
    d1, d2 = (np.linalg.norm(coordinates - c[:, None, :], axis=-1) for c in center)
    radius_correction = theta[10:12]
    first = expit((radius + radius_correction[0] - d1) / 0.85)
    second = expit((radius + radius_correction[1] - d2) / 0.85)
    overlap = first * second
    return np.stack((np.ones_like(first), first - overlap, second - overlap, overlap), axis=-1)


def poisson_residual(observed, fitted):
    fitted = np.maximum(fitted, 1e-5)
    reference = np.maximum(observed, 1e-5)
    return np.sign(fitted - observed) * np.sqrt(np.maximum(0, 2 * (
        fitted - observed + observed * np.log(reference / fitted))))


def fit_patch(counts, indices, nominal_positions, centers, radius, initial=None, matrix_bound=9):
    """Recover disk geometry while giving the overlap its own intensity."""
    coordinates = COORDINATES[None] + centers[:, None, None] - nominal_positions[:, None, None]
    g = indices @ np.linalg.inv(AUTHOR_BASIS)
    g /= np.median(np.linalg.norm(g, axis=1))
    pair = np.zeros((2, len(g), 2))
    radial = nominal_positions - nominal_positions.mean(axis=0)
    pair[0] = -0.02 * radial
    pair[1] = 0.02 * radial
    matrices = np.stack([np.linalg.lstsq(g, row, rcond=None)[0] for row in pair])
    start = np.concatenate((np.zeros(2), matrices.ravel(), np.zeros(2))) if initial is None else initial
    low = np.array([-3, -3, *([-matrix_bound] * 8), -3, -3])
    high = -low
    start = np.clip(start, low + 1e-6, high - 1e-6)
    roi = np.linalg.norm(COORDINATES, axis=-1) < WINDOW - 1
    coords = coordinates[:, roi]
    values = counts[:, roi].astype(float)
    n = counts.shape[0]
    weight = 1 / np.maximum(values, 0.6)

    def residual(theta):
        design = masks_from_pair(theta, g, coords, radius)
        weighted = design.swapaxes(-2, -1) * weight[:, None, :]
        coefficients = np.linalg.solve(weighted @ design + 1e-7 * np.eye(4), weighted @ values[..., None])
        fitted = (design @ coefficients)[..., 0]
        return ((fitted - values) * np.sqrt(weight)).ravel()

    opt = least_squares(residual, start, bounds=(low, high), max_nfev=65, diff_step=1e-3,
                        ftol=3e-5, xtol=1e-5, gtol=1e-3)
    _, nominal, _ = reciprocal_matrix(nominal_positions, g)
    relative = opt.x[2:10].reshape(2, 2, 2)
    inferred = nominal + relative
    real_strain = np.stack([np.linalg.inv(mat @ np.linalg.inv(nominal)).T - np.eye(2) for mat in inferred])
    lattice_ratio = np.sqrt(np.linalg.det(inferred[0]) / np.linalg.det(inferred[1]))
    design = masks_from_pair(opt.x, g, coords, radius)
    weighted = design.swapaxes(-2, -1) * weight[:, None, :]
    coefficients = np.linalg.solve(weighted @ design + 1e-7 * np.eye(4), weighted @ values[..., None])
    fitted = (design @ coefficients)[..., 0]
    errors = poisson_residual(values, fitted)
    n_dof = errors.size - n * 4 - len(opt.x)
    condition = np.linalg.cond(opt.jac)
    return opt.x, {
        "converged": bool(opt.success), "evaluations": int(opt.nfev),
        "relative_trace_percent": float(50 * np.trace(real_strain[0] - real_strain[1])),
        "mismatch_from_reciprocal_areas_percent": float(100 * (1 - lattice_ratio)),
        "radius_correction_pixels": opt.x[10:12].tolist(),
        "common_detector_shift_pixels": opt.x[:2].tolist(),
        "poisson_deviance_per_dof": float(np.sum(errors**2) / n_dof),
        "geometry_fit_condition": float(condition),
        "geometry_fit_parameters": opt.x.tolist(),
        "reciprocal_matrix_bound_pixels": matrix_bound,
        "reciprocal_matrix_parameters_at_bound": int(np.sum(opt.active_mask[2:10] != 0)),
        "minimum_fitted_intensity": float(fitted.min()),
        "reciprocal_matrices_detector_pixels": inferred.tolist(),
    }


def ring_readout(rows, modes, positions, centers, radius, aggregate_only):
    """Compare ring and scan-pixel repeats before interpreting local values."""
    counts = np.stack([row["counts"] for row in rows])
    h, k = modes.T
    number = np.rint(h*h + h*k + k*k).astype(int)
    groups = {"first": number == 1, "second": number == 3, "both": np.ones(len(modes), dtype=bool)}
    average = counts.mean(axis=(0, 1))
    aggregate = {}
    blocks = []
    initial = {}
    for name, selection in groups.items():
        checkpoint_guard()
        initial[name], score = fit_patch(average[selection], modes[selection], positions[selection],
                                         centers[selection], radius)
        aggregate[name] = score
    if aggregate_only:
        return aggregate, blocks
    for i, row in enumerate(rows):
        checkpoint_guard()
        for start in (0, 10, 20, 30):
            block = counts[i, start:start+10]
            block_means = {"all": block.mean(axis=0), "even": block[::2].mean(axis=0), "odd": block[1::2].mean(axis=0)}
            report = {"row": row["row"], "column_start": FIRST_COLUMN + start, "column_count": 10}
            for name, phase in (("first", "all"), ("second", "all"), ("both", "all"), ("both", "even"), ("both", "odd")):
                selection = groups[name]
                _, score = fit_patch(block_means[phase][selection], modes[selection], positions[selection],
                                     centers[selection], radius, initial[name])
                report[f"{name}_{phase}"] = score
            blocks.append(report)
    return aggregate, blocks


def model_trace_comparison(archive, blocks):
    """Compare diffraction-edge and predicted dilations on the same spatial blocks."""
    protocol = json.loads(PROTOCOL.read_text())
    verify_freeze(PROJECT / "heterobilayer_transfer_freeze_v01.json", fingerprints(),
                  protocol["devices"]["development"])
    data = load_dataset(SCAN, archive, allowed_ids=(SCAN,), allow_author_crop=True)
    prior = json.loads((PROJECT / "heterobilayer_transfer_development_v01.json").read_text())["scans"][SCAN]
    if data.archive_md5 != prior["archive_md5"]:
        raise ValueError("The raw data comparison must use the matched DS31 archive")
    scale = MODEL_A / data.lattice_nm
    fit = np.asarray(prior["whole_geometry"]["fit_scan"])
    yy, xx = np.indices(data.u_nm.shape[:2], dtype=float)
    points = np.stack((xx, yy), axis=-1) * data.pixel_nm
    nominal = points @ fit[:2] + fit[-1]
    warp, _ = smooth_geometry(data, fit, points)
    gradient = np.asarray(prior["whole_geometry"]["gradient_crystal"]) * scale
    grid, shift, _ = solve(gradient, BASIS, tuple(protocol["interface"]), prior["grid"])
    probe = 1.25 / np.sqrt(8 * np.log(2))
    traces = {}
    for name, rigid in (("affine", nominal), ("warped", nominal + warp)):
        phase = ORIGIN + scale * rigid
        relaxed = (sample_disregistry(grid, shift, phase, grid.cell, probe) - ORIGIN) / scale
        traces[f"{name}_rigid"] = dilation(rigid, data)
        traces[f"{name}_relaxed"] = dilation(relaxed, data)
    interior, _ = _good_interior_triangle_mask(data)
    used = []
    for block in blocks:
        row, column = block["row"] - 1, block["column_start"] - 1
        indices = np.s_[row, column:column + block["column_count"]]
        if np.all(interior[indices]):
            block["matched_model_dilations_percent"] = {name: float(field[indices].mean()) for name, field in traces.items()}
            used.append(block)
    if len(used) < 10:
        return {"matching_blocks": len(used)}
    measured = np.array([block["both_all"]["relative_trace_percent"] for block in used])
    scores = {}
    for name in traces:
        predicted = np.array([block["matched_model_dilations_percent"][name] for block in used])
        difference = predicted - measured
        scores[name] = {"bias_pp": float(difference.mean()), "offset_free_rmse_pp": float(difference.std()),
                        "predicted_variability_pp": float(predicted.std())}
    return {"matching_blocks": len(used), "measured_variability_pp": float(measured.std()), "scores": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("ranges", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    start = perf_counter()
    author = load_author_release(args.release, args.dependencies)
    disk, _, _, fit_hash = read_detector_fit(args.archive, SCAN, "fit", author)
    active = disk._in_use.astype(bool)
    positions = np.column_stack((disk._xvec[active], disk._yvec[active]))
    radii = disk._rvec[active]
    modes = disk.clean_normgset(sanity_plot=False)
    source, centers, rows, hashes = load_counts(args.ranges, positions, radii, disk.df_set())
    convergence_radius = 1.71 / source["detector_spacing_mrad"]
    aggregate, blocks = ring_readout(rows, modes, positions, centers, convergence_radius, args.aggregate_only)
    repeat = []
    ring_difference = []
    for block in blocks:
        even, odd, first, second = (block[key]["relative_trace_percent"]
                                    for key in ("both_even", "both_odd", "first_all", "second_all"))
        repeat.append(even - odd)
        ring_difference.append(first - second)
    model_comparison = model_trace_comparison(args.archive, blocks) if blocks else None
    checkpoint_guard()
    atomic_json(args.output, {
        "scope": "Public raw DS31 frames. This is a feasibility test of a two-disk edge fit. The overlap receives an independent intensity. Reciprocal peak shifts share acquisition biases. The selected lines come from one partially capped region.",
        "source": source,
        "range_sha256": hashes,
        "input_sha256": {Path(__file__).name: digest(Path(__file__)), "DS31_fit.pkl": fit_hash},
        "alignment": [{k: v for k, v in row.items() if k != "counts"} for row in rows],
        "aggregate": aggregate,
        "blocks": blocks,
        "model_trace_comparison": model_comparison,
        "block_even_odd_trace_rms_pp": float(np.sqrt(np.mean(np.square(repeat)))) if repeat else None,
        "block_ring_trace_rms_difference_pp": float(np.sqrt(np.mean(np.square(ring_difference)))) if ring_difference else None,
        "wall_seconds": perf_counter() - start,
    })


if __name__ == "__main__":
    main()
