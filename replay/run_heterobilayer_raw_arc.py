"""Track raw DS31 reciprocal shifts using the outer arcs of paired disks."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.optimize import least_squares

from author_detector_replay import read_detector_fit
from author_release import load_author_release
from run_heterobilayer_raw_ds31 import SCAN, reciprocal_matrix
from run_heterobilayer_raw_ring_controls import block_counts, higher_orders
from run_heterobilayer_raw_injection import HIGH, RAW, paired_matrices, real_to_detector, summarize
from run_heterobilayer_transfer import digest
from shared_compute import atomic_json, checkpoint_guard


RADIAL_PIXELS = np.linspace(-3.5, 3.5, 29)
TEMPLATE_PIXELS = np.linspace(-6.5, 6.5, 53)
MAX_EDGE_CHANGE = 2.3


def select_arcs(grand, nominal, center, indices, radius, theta):
    """Build edge templates using the calibration average and unshared pixels."""
    modes, reference = paired_matrices(indices, nominal)
    matrices = theta[2:10].reshape(2, 2, 2)
    disk_centers = nominal - center + theta[:2] + modes @ matrices
    disk_radius = radius + theta[10:12]
    angles = np.arange(32) * (2 * np.pi / 32)
    normals = np.column_stack((np.sin(angles), np.cos(angles)))
    directions = np.broadcast_to(normals[None, None], (*disk_centers.shape[:2], len(angles), 2))
    origin = disk_centers[:, :, None, :] + disk_radius[:, None, None, None] * directions
    # The other disk must stay at least 2.25 pixels away from the edge band
    # over the fit window. This gate is evaluated on the aggregate only.
    separation = np.linalg.norm(
        origin[..., None, :] + RADIAL_PIXELS[None, None, None, :, None] * directions[..., None, :]
        - disk_centers[::-1, :, None, None, :], axis=-1)
    exclusive = separation.min(axis=-1) > disk_radius[::-1, None, None] + 2.25
    pair, disk, angular = np.where(exclusive)
    n = directions[pair, disk, angular]
    positions = origin[pair, disk, angular]
    # The nominal raw patches are centered at WINDOW. Extract every template
    # at its own angular position, with no symmetric disk-shape assumption.
    base = positions + (grand.shape[-1] - 1) / 2
    tracks = base[:, None] + TEMPLATE_PIXELS[None, :, None] * n[:, None, :]
    template = np.stack([map_coordinates(grand[d], xy.T, order=1, mode="nearest") for d, xy in zip(disk, tracks)])
    template = gaussian_filter1d(template, 0.65, axis=1, mode="nearest")
    contrast = template[:, TEMPLATE_PIXELS < -1].mean(axis=1) - template[:, TEMPLATE_PIXELS > 1].mean(axis=1)
    usable = contrast > 0.15
    pair, disk, n, base, template, contrast = (x[usable] for x in (pair, disk, n, base, template, contrast))
    if min(np.bincount(pair, minlength=2)) < 18:
        raise ValueError("Too few nonoverlapping, contrasted arcs for the two-layer lattice fit")
    # A radial edge change is the projection of the reciprocal-lattice shift
    # onto the outward normal, plus a common motion and each disk radius.
    design = np.zeros((len(pair), 12))
    design[:, :2] = n
    for layer in range(2):
        choose = pair == layer
        design[choose, 2+4*layer:6+4*layer] = np.einsum("ij,ik->ijk", modes[disk[choose]], n[choose]).reshape(-1, 4)
        design[choose, 10+layer] = 1
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Nonoverlapping arcs do not constrain all reciprocal components")
    return disk, base, n, template, contrast, design, reference, np.bincount(pair, minlength=2)


def take_profiles(counts, arcs):
    disk, base, n, _, _, _, _, _ = arcs
    points = base[:, None] + RADIAL_PIXELS[None, :, None] * n[:, None, :]
    return np.stack([map_coordinates(counts[d], xy.T, order=1, mode="nearest") for d, xy in zip(disk, points)])


def solve_profiles(profiles, arcs):
    """Fit projected edge shifts while eliminating per-arc scale and offset."""
    _, _, _, template, contrast, design, _, _ = arcs
    target = profiles - profiles.mean(axis=1, keepdims=True)
    variance = np.maximum(profiles.mean(axis=1), 0.1)
    weight = 1 / np.sqrt(variance)
    origin = RADIAL_PIXELS[None, :]

    def residual(parameters):
        offsets = np.clip(design @ parameters, -MAX_EDGE_CHANGE, MAX_EDGE_CHANGE)
        shifted = np.stack([np.interp(x, TEMPLATE_PIXELS, v) for x, v in zip(origin - offsets[:, None], template)])
        shifted -= shifted.mean(axis=1, keepdims=True)
        scale = np.einsum("ij,ij->i", shifted, target) / np.maximum(np.sum(shifted**2, axis=1), 1e-12)
        return (weight[:, None] * (np.maximum(scale, 0)[:, None] * shifted - target)).ravel()

    fit = least_squares(residual, np.zeros(design.shape[1]), max_nfev=65, diff_step=1e-4,
                        ftol=1e-6, xtol=1e-6, gtol=1e-6)
    fitted_shifts = design @ fit.x
    return fit.x, {"converged": bool(fit.success), "fit_condition": float(np.linalg.cond(fit.jac)),
                   "weighted_profile_rmse": float(np.sqrt(np.mean(fit.fun**2))),
                   "weighted_edge_contrast": float(np.mean(contrast * weight)),
                   "arcs_at_bound": int(np.sum(np.abs(fitted_shifts) >= MAX_EDGE_CHANGE)),
                   "maximum_arc_shift_pixels": float(np.max(np.abs(fitted_shifts)))}


def read_trace(theta, reference):
    reciprocal = reference + theta[2:10].reshape(2, 2, 2)
    real = np.linalg.inv(reciprocal @ np.linalg.inv(reference)).swapaxes(-1, -2)
    return float(50 * np.trace(real[0] - real[1]))


def injection_readout(profiles, arcs, theta, local_truth):
    """Translate the measured arc profile by a known projected lattice change."""
    design, reference = arcs[5:7]
    base, _ = solve_profiles(profiles, arcs)
    before = read_trace(theta + base, reference)
    target = real_to_detector(theta + base, reference, local_truth)
    shift = design @ (target - theta - base)
    moved = np.stack([np.interp(RADIAL_PIXELS - x, RADIAL_PIXELS, v) for x, v in zip(shift, profiles)])
    answer, diagnostics = solve_profiles(moved, arcs)
    return {"truth_pp": float(local_truth), "recovered_pp": read_trace(theta + answer, reference) - before,
            "converged": diagnostics["converged"], "matrix_parameters_at_bound": diagnostics["arcs_at_bound"],
            "negative_average_roi_fraction": float(np.mean(moved < 0))}


def comparison(rows, old):
    values = np.array([b["relative_trace_percent"] for b in rows])
    reference = np.array([b["mean"]["relative_trace_percent"] for b in old])
    model = np.array([b["model_trace_percent"] for b in rows])
    even = np.array([b["even_trace_percent"] for b in rows])
    odd = np.array([b["odd_trace_percent"] for b in rows])
    return {
        "arc_variability_pp": float(values.std()), "paired_fit_variability_pp": float(reference.std()),
        "model_variability_pp": float(model.std()), "model_offset_free_rmse_pp": float((model - values).std()),
        "arc_vs_paired_fit_offset_free_rmse_pp": float((values - reference).std()),
        "arc_vs_paired_fit_correlation": float(np.corrcoef(values, reference)[0, 1]),
        "even_odd_offset_free_disagreement_pp": float((even - odd).std()),
        "injected_strain": summarize([b["injection"] for b in rows]),
        "nonconverged_blocks": sum(not b["fit"]["converged"] for b in rows),
        "blocks_with_arcs_at_bound": sum(b["fit"]["arcs_at_bound"] > 0 for b in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("ranges", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    raw = json.loads(RAW.read_text())
    high = json.loads(HIGH.read_text())
    for name, expected in raw["range_sha256"].items():
        checkpoint_guard()
        if digest(args.ranges / name) != expected:
            raise ValueError(f"The raw DS31 data changed in {name}")
    author = load_author_release(args.release, args.dependencies)
    disk, _, _, fit_hash = read_detector_fit(args.archive, SCAN, "fit", author)
    if high["input_sha256"]["DS31_fit.pkl"] != fit_hash:
        raise ValueError("The native reciprocal geometry changed")
    mask = disk._in_use.astype(bool)
    positions = np.column_stack((disk._xvec[mask], disk._yvec[mask]))
    indices, nominal = higher_orders(disk.clean_normgset(sanity_plot=False), positions)
    radius = 1.71 / raw["source"]["detector_spacing_mrad"]
    results = {
        "scope": "Track one layer's outer edge where the other layer's disk is absent. Use measured edge templates from the scan-wide average, remove per-edge brightness offset and scale, and fit both reciprocal matrices to their edge shifts. Track the same 44 DS31 blocks. Global paired fits define the initial arc locations and reference matrices. This alternative readout uses the same microscope acquisition.",
        "input_sha256": {p.name: digest(p) for p in (
            Path(__file__), HIGH, RAW, Path(__file__).with_name("run_heterobilayer_raw_injection.py"))},
        "archive_fit_sha256": fit_hash, "groups": {}, "failed_groups": {},
    }
    square = indices[:, 0]**2 + indices[:, 0]*indices[:, 1] + indices[:, 1]**2
    centered_model = np.array([b["matched_model_dilations_percent"]["warped_relaxed"] for b in raw["blocks"]])
    centered_model -= centered_model.mean()
    for label, select in (("higher_both", np.ones(len(indices), dtype=bool)),
                          ("higher_order_4", square == 4), ("higher_order_9", square == 9)):
        checkpoint_guard()
        centers, blocks = block_counts(args.ranges, nominal[select], raw["source"])
        grand = np.mean([b["mean"] for b in blocks], axis=0)
        theta = np.asarray(high["aggregate"][label]["geometry_fit_parameters"])
        try:
            arcs = select_arcs(grand, nominal[select], centers, indices[select], radius, theta)
            fitted_grand, diagnostics = solve_profiles(take_profiles(grand, arcs), arcs)
            rows = []
            for i, block in enumerate(blocks):
                checkpoint_guard()
                reference = raw["blocks"][i]
                if (reference["row"], reference["column_start"]) != (block["row"], block["column_start"]):
                    raise ValueError("The raw block support changed")
                profile = {phase: take_profiles(block[phase], arcs) for phase in ("mean", "even", "odd")}
                results_fit = {phase: solve_profiles(values, arcs) for phase, values in profile.items()}
                rows.append({
                    "row": block["row"], "column_start": block["column_start"], "fit": results_fit["mean"][1],
                    "relative_trace_percent": read_trace(theta + results_fit["mean"][0], arcs[6]),
                    "even_trace_percent": read_trace(theta + results_fit["even"][0], arcs[6]),
                    "odd_trace_percent": read_trace(theta + results_fit["odd"][0], arcs[6]),
                    "model_trace_percent": reference["matched_model_dilations_percent"]["warped_relaxed"],
                    "injection": injection_readout(profile["mean"], arcs, theta, centered_model[i]),
                })
            results["groups"][label] = {
                "arcs_per_layer": arcs[-1].tolist(), "projection_matrix_condition": float(np.linalg.cond(arcs[5])),
                "grand_fit": diagnostics, "grand_trace_percent": read_trace(theta + fitted_grand, arcs[6]),
                "summary": comparison(rows, [b[label] for b in high["blocks"]]), "blocks": rows,
            }
            print(label, results["groups"][label]["summary"], flush=True)
        except (ValueError, np.linalg.LinAlgError) as exc:
            results["failed_groups"][label] = str(exc)
            print(label, str(exc), flush=True)
        results["wall_seconds"] = perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, results)


if __name__ == "__main__":
    main()
