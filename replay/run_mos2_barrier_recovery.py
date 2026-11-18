"""Recover concealed MoS2 sliding barriers from synthetic two-ring microscope images."""

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.linalg import solve_triangular
from scipy.ndimage import gaussian_filter

from author_detector_replay import detector_intensity
from bilayer_relaxation import (MODES, MONOLAYERS, STACKING, elastic_parameters_per_reference_cell,
                                make_grid, rigid_gradient, sample_disregistry)
from run_mos2_nonlinear_controls import alternative_inputs, minimize_all
from run_mos2_wall_profile_closure import isolated_field
from run_mos2_wall_width_transfer import isolated_width
from shared_compute import atomic_json, checkpoint_guard


PROJECT = Path(__file__).resolve().parent
INTERFACE = ("MoS2", "MoS2", "P")
BASIS = MONOLAYERS["MoS2"][0] * np.array(((1., .5), (0., np.sqrt(3) / 2)))
ELASTIC = elastic_parameters_per_reference_cell(*INTERFACE[:2])
RECIPROCAL = np.r_[MODES[:6], -MODES[:6]] @ np.linalg.inv(BASIS)
RING = np.tile((1, 1, 1, 2, 2, 2), 2)
ODD = np.array((-.18, .14, -.08, -.16, .11, .19, -.12, .09, .10, -.08, .14, -.17))
CAMERA = np.column_stack((np.ones(len(ODD)), ODD, np.full(len(ODD), .20)))
MODES_USED = {"first": RING == 1, "second": RING == 2, "both": RING > 0}
GAUSSIAN_FWHM = np.sqrt(8 * np.log(2))
# This Fourier perturbation changes the curvature at AB and keeps the
# SP–AB and AA–AB energy differences fixed.
CURVATURE = np.array((3., 1., -1., -1., 0., 0.))


@dataclass(frozen=True)
class Wall:
    gradient: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    along: np.ndarray
    across: np.ndarray
    spacing_nm: float


def fingerprint(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def sources(protocol):
    paths = (protocol, Path(__file__), PROJECT / "bilayer_relaxation.py", PROJECT / "author_detector_replay.py",
             PROJECT / "run_mos2_nonlinear_controls.py", PROJECT / "run_mos2_wall_profile_closure.py",
             PROJECT / "run_mos2_wall_width_transfer.py", PROJECT / "moire_regime_benchmark_v02.json",
             PROJECT / "mos2_wall_signal_repeatability_v01.json")
    return {path.name: fingerprint(path) for path in paths}


def wall_geometry(angle, pixel):
    gradient = rigid_gradient(angle)
    edge = BASIS[:, 0]
    direction = np.linalg.solve(gradient, edge)
    length = np.linalg.norm(direction)
    direction /= length
    normal = np.array((-direction[1], direction[0]))
    slip = np.array((-edge[1], edge[0])) / np.linalg.norm(edge)
    if (gradient @ normal) @ slip < 0:
        normal *= -1
    spacing = np.linalg.norm(edge) * np.sqrt(3) / (2 * abs((gradient @ normal) @ slip))
    rows = int(np.floor(.08 * length / pixel))
    columns = int(np.floor(min(8., .68 * spacing) / pixel))
    return Wall(gradient, direction, normal, pixel * np.arange(-rows, rows + 1),
                pixel * np.arange(-columns, columns + 1), float(spacing))


def landscape_family():
    anchor = np.asarray(STACKING[INTERFACE])
    alternate = alternative_inputs(BASIS, "P")[0]
    reference = isolated_width(BASIS, BASIS[:, 0], anchor)[1]
    alternate *= reference / isolated_width(BASIS, BASIS[:, 0], alternate)[1]
    return 1000 * reference, anchor, alternate


def adhesion_at(scales, mixture, scale):
    _, anchor, alternate = scales
    return scale * ((1 - mixture) * anchor + mixture * alternate)


def grid_for(wall, refinement=1):
    length = np.linalg.norm(np.linalg.solve(wall.gradient, BASIS), axis=0).max()
    points = max(64, 2 ** int(np.ceil(np.log2(length / .30))))
    return make_grid(wall.gradient, BASIS, points * refinement, *ELASTIC)


def padded_images(wall, step):
    # Extend the physical profile beyond both the probe and allowed registration shifts.
    pad = int(np.ceil(4.0 / step))
    nr = int(np.rint(wall.along[-1] / step)) + pad
    nc = int(np.rint(wall.across[-1] / step)) + pad
    x, y = np.meshgrid(step * np.arange(-nc, nc + 1), step * np.arange(-nr, nr + 1))
    world = y[..., None] * wall.tangent + x[..., None] * wall.normal
    phase = BASIS[:, 0] / 2 + world @ wall.gradient.T
    return x[0], y[:, 0], phase


def detector_image(field, camera, probe_yx, pixel):
    signal = detector_intensity(field, RECIPROCAL, camera)
    sigma_y, sigma_x = np.asarray(probe_yx) / (GAUSSIAN_FWHM * pixel)
    blurred = gaussian_filter(signal, (sigma_y, sigma_x, 0), mode="nearest")
    return blurred


def read_pixels(image, xy, step, wall, shifts):
    x, y = xy
    rows = np.rint((wall.along - y[0]) / step).astype(int)
    distance = wall.across - np.asarray(shifts)[:, None]
    index = (distance - x[0]) / step
    left = np.floor(index).astype(int)
    alpha = index - left
    raster = image[rows]
    points = raster[:, left] * (1 - alpha[..., None]) + raster[:, left + 1] * alpha[..., None]
    return np.moveaxis(points, 0, 1)


def noise_covariance(wall, condition, settings):
    distance = np.abs(wall.across[:, None] - wall.across)
    correlation = np.exp(-distance / settings["noise_correlation_length_nm"])
    column = condition["column_noise_sd"]**2 * correlation
    # The known exposure gives the shot-noise scale. Use the typical
    # midfringe intensity for the pre-fit Gaussian approximation.
    shot = .70 / (condition["counts_per_unit_signal"] * len(wall.along))
    return column + shot * np.eye(len(distance))


def synthetic_capture(rng, field, xy, wall, step, condition, settings, nominal_probe, nominal):
    if nominal:
        probe = (nominal_probe, nominal_probe)
        camera = CAMERA
    else:
        fwhm = rng.uniform(*settings["variable_probe_fwhm_nm"])
        tilt = settings["maximum_absolute_log_probe_aspect"]
        probe = fwhm * np.exp(np.array((1., -1.)) * rng.uniform(-tilt, tilt))
        camera = CAMERA.copy()
        error = settings["maximum_detector_odd_coefficient_error"]
        camera[:, 1] += rng.uniform(-error, error, len(CAMERA))
    maximum_shift = settings["maximum_normal_shift_nm"]
    shift = rng.uniform(-maximum_shift, maximum_shift)
    image = detector_image(field, camera, probe, step)
    pixels = read_pixels(image, xy, step, wall, (shift,))[0]
    gains = rng.uniform(.88, 1.12, len(CAMERA))
    background = rng.uniform(0, .08, len(CAMERA))
    slope = rng.uniform(-.02, .02, len(CAMERA)) / wall.across[-1]
    pixels = pixels * gains + background + wall.across[None, :, None] * slope
    if np.min(pixels) <= 0:
        raise ValueError("Negative electron-count expectation")
    counts = condition["counts_per_unit_signal"]
    shot = rng.poisson(pixels * counts) / counts
    distance = np.abs(wall.across[:, None] - wall.across)
    correlation = np.exp(-distance / settings["noise_correlation_length_nm"])
    correlated = np.linalg.cholesky(correlation) @ rng.normal(size=pixels.shape[1:])
    noisy = shot + condition["column_noise_sd"] * correlated
    return noisy.astype(np.float32), {"true_probe_yx_nm": list(probe), "true_offset_nm": float(shift)}


def generate(args, protocol, source_hashes):
    rng = np.random.default_rng(args.seed)
    settings = protocol["generator"]
    barrier_family = landscape_family()
    reference = barrier_family[0]
    scales = [rng.uniform(*interval) for interval in settings["relative_barrier_intervals"]]
    shape_variants = [(rng.uniform(*settings["alternate_shape_mixture_interval"]),
                       rng.uniform(-1., 1.) * settings["maximum_unmodelled_curvature_perturbation_meV_per_cell"])
                      for _ in scales]
    cases = [(float(angle), name, scale, mixture, curvature) for name in settings["conditions"]
             for angle in protocol["twist_degrees"]
             for scale, (mixture, curvature) in zip(scales, shape_variants, strict=True)]
    rng.shuffle(cases)
    images, metadata, key = {}, {}, {}
    step = protocol["raster_pixel_nm"] / 8
    for index, (angle, name, scale, mixture, curvature) in enumerate(cases):
        label = f"sample_{index:02d}"
        wall = wall_geometry(angle, protocol["raster_pixel_nm"])
        if not name.endswith("landscape"):
            mixture, curvature = 0., 0.
        adhesion = adhesion_at(barrier_family, mixture, scale)
        adhesion += curvature * CURVATURE
        true_barrier = 1000 * isolated_width(BASIS, BASIS[:, 0], adhesion)[1]
        if not np.isclose(true_barrier, scale * reference):
            raise ValueError("The independently determined barrier disagrees with the requested truth")
        grid = grid_for(wall, refinement=2)
        motion, solver = minimize_all(grid, adhesion)
        x, y, phase = padded_images(wall, step)
        field = sample_disregistry(grid, motion, phase, grid.cell)
        captures, readouts = zip(*[synthetic_capture(rng, field, (x, y), wall, step, settings["conditions"][name],
                                                    settings, protocol["nominal_probe_fwhm_nm"], name == "nominal")
                                    for _ in range(settings["replicates_per_case"])], strict=True)
        images[label] = np.stack(captures)
        metadata[label] = {"twist_degrees": angle, "condition": name, "spacing_nm": wall.spacing_nm}
        key[label] = {"barrier_meV_per_cell": true_barrier, "landscape_mixture": float(mixture),
                      "curvature_perturbation_meV_per_cell": float(curvature),
                      "grid": grid.fractional.shape[0], "solver": solver, "readouts": readouts}
        checkpoint_guard()
    atomic_json(args.truth, {"seed": args.seed, "source_sha256": source_hashes, "truth": key})
    receipt = {"source_sha256": source_hashes, "truth_sha256": fingerprint(args.truth), "samples": metadata}
    with NamedTemporaryFile(dir=args.images.parent, suffix=".npz", delete=False) as partial:
        np.savez_compressed(partial, receipt=json.dumps(receipt), **images)
    os.replace(partial.name, args.images)
    print(f"Saved {len(images)} synthetic cases and sealed the separate truth key", flush=True)


def profile_templates(wall, protocol, family, material, coarse, fine, step):
    settings = protocol["fitting"]
    grid = grid_for(wall)
    x, y, phase = padded_images(wall, step)
    probes = settings["fitted_probe_fwhm_nm"]
    shifts = settings["fitted_normal_shifts_nm"]
    mixtures = settings["landscape_mixtures"]
    templates = np.empty((len(mixtures), len(coarse), len(probes), len(shifts), len(wall.across), len(CAMERA)))
    for q, mixture in enumerate(mixtures):
        for k, scale in enumerate(coarse):
            adhesion = adhesion_at(material, mixture, scale)
            if family == "periodic":
                motion, _ = minimize_all(grid, adhesion)
                field = sample_disregistry(grid, motion, phase, grid.cell)
            else:
                slip = np.array((0., 1.))
                line = isolated_field(x, BASIS[:, 0], (adhesion, ELASTIC), BASIS, 1., (wall.normal @ slip)**2)
                field = np.broadcast_to(line, phase.shape)
            for p, fwhm in enumerate(probes):
                blurred = detector_image(field, CAMERA, (fwhm, fwhm), step)
                templates[q, k, p] = read_pixels(blurred, (x, y), step, wall, shifts).mean(axis=1)
        checkpoint_guard()
    return PchipInterpolator(np.log(coarse), templates, axis=1)(np.log(fine))


def whitened_shape_operator(wall, covariance):
    identity = np.eye(len(wall.across))
    whiten = solve_triangular(np.linalg.cholesky(covariance), identity, lower=True)
    background = whiten @ np.column_stack((np.ones(len(wall.across)), wall.across))
    basis = np.linalg.qr(background)[0]
    return (identity - basis @ basis.T) @ whiten


def compare_profiles(templates, measured, shape_operator, gain_bounds):
    # Eliminate per-channel constant and linear background, then fit the
    # bounded optical gains analytically for every library point.
    n, channels = templates.shape[-2:]
    library = (shape_operator @ templates).reshape(-1, n, channels).transpose(2, 0, 1)
    data = (shape_operator @ measured).transpose(2, 1, 0)
    cross = library @ data
    norms = np.sum(library**2, axis=2)[..., None]
    target_power = np.sum(data**2, axis=1)[:, None, :]
    gains = np.clip(cross / np.maximum(norms, 1e-20), *gain_bounds)
    costs = np.maximum(0., target_power - 2 * gains * cross + gains**2 * norms)
    return costs.reshape(channels, *templates.shape[:-2], len(measured))


def interval(log_barrier, curve, delta):
    accepted = np.flatnonzero(curve <= np.min(curve) + delta)
    low, high = accepted[0], accepted[-1]
    left, right = log_barrier[[low, high]]
    target = np.min(curve) + delta
    if low > 0:
        left = np.interp(target, curve[low-1:low+1][::-1], log_barrier[low-1:low+1][::-1])
    if high < len(curve)-1:
        right = np.interp(target, curve[high:high+2], log_barrier[high:high+2])
    return np.exp((left, right)).tolist()


def estimates(costs, barriers, mixtures, probes, shifts, n_pixels, delta):
    log_barrier = np.log(barriers)
    output = {}
    for region, used in MODES_USED.items():
        group = costs[used].sum(axis=0)
        output[region] = {}
        for policy, selection in (("anchor_shape", group[:1]), ("profile_shapes", group)):
            points = []
            for i in range(selection.shape[-1]):
                loss = selection[..., i]
                t, k, p, s = np.unravel_index(np.argmin(loss), loss.shape)
                profile = loss.min(axis=(0, 2, 3))
                dof = n_pixels * used.sum() - 3 * used.sum() - 2 - int(len(probes) > 1) - int(len(selection) > 1)
                points.append({"barrier_meV_per_cell": float(barriers[k]), "barrier_interval": interval(log_barrier, profile, delta),
                               "hit_barrier_bound": bool(k in (0, len(barriers)-1)),
                               "probe_fwhm_nm": probes[p], "shift_nm": shifts[s], "landscape_mixture": mixtures[t],
                               "reduced_noise_loss": float(loss[t, k, p, s] / dof)})
            output[region][policy] = points
    return output


def check_isolated_fit(templates, wall, protocol, material, barriers):
    step = protocol["raster_pixel_nm"] / 8
    x, y, phase = padded_images(wall, step)
    scale, position = 1.137, .15
    adhesion = adhesion_at(material, 0., scale)
    line = isolated_field(x, BASIS[:, 0], (adhesion, ELASTIC), BASIS, 1., wall.normal[1]**2)
    probe = protocol["nominal_probe_fwhm_nm"]
    blurred = detector_image(np.broadcast_to(line, phase.shape), CAMERA, (probe, probe), step)
    measured = read_pixels(blurred, (x, y), step, wall, (position,)).mean(axis=1)
    settings = protocol["fitting"]
    nominal = settings["fitted_probe_fwhm_nm"].index(probe)
    operator = whitened_shape_operator(wall, np.eye(len(wall.across)))
    loss = compare_profiles(templates[:1, :, [nominal]], measured, operator, settings["relative_gain_bounds"])
    output = {}
    expected = scale * material[0]
    for region, used in MODES_USED.items():
        summed = loss[used].sum(axis=0)
        _, energy_index, _, center_index, _ = np.unravel_index(np.argmin(summed), summed.shape)
        actual = float(barriers[energy_index])
        shift = settings["fitted_normal_shifts_nm"][center_index]
        if abs(actual / expected - 1) > .035 or abs(shift - position) > .01:
            raise ValueError(f"The matched isolated barrier/shift recovery failed in {region}: {actual}, {shift}")
        output[region] = {"expected_meV_per_cell": expected, "recovered_meV_per_cell": actual, "shift_nm": shift}
    return output


def fit(args, protocol, source_hashes):
    start = perf_counter()
    options = protocol["fitting"]
    settings = protocol["generator"]
    material = landscape_family()
    coarse = np.geomspace(*options["relative_barrier_bounds"], options["simulation_grid_barrier_points"])
    fine = np.geomspace(*options["relative_barrier_bounds"], options["search_grid_barrier_points"])
    barriers = fine * material[0]
    nominal_probe = protocol["nominal_probe_fwhm_nm"]
    all_probes = options["fitted_probe_fwhm_nm"]
    result = {"source_sha256": source_hashes, "images_sha256": fingerprint(args.images), "samples": {},
              "noiseless_isolated_recovery": {}}
    with np.load(args.images) as archive:
        receipt = json.loads(archive["receipt"].item())
        if receipt["source_sha256"] != source_hashes:
            raise ValueError("The fitting code or inputs differ from the freeze for these target images")
        result["truth_sha256"] = receipt["truth_sha256"]
        for angle in protocol["twist_degrees"]:
            wall = wall_geometry(angle, protocol["raster_pixel_nm"])
            selected = {key: row for key, row in receipt["samples"].items() if row["twist_degrees"] == angle}
            for family in ("isolated", "periodic"):
                templates = profile_templates(wall, protocol, family, material, coarse, fine, protocol["raster_pixel_nm"] / 4)
                if family == "isolated":
                    result["noiseless_isolated_recovery"][str(angle)] = check_isolated_fit(templates, wall, protocol,
                                                                                          material, barriers)
                for label, tags in selected.items():
                    name = tags["condition"]
                    probes = [nominal_probe] if name == "nominal" else all_probes
                    probe_indices = [all_probes.index(value) for value in probes]
                    target = archive[label].mean(axis=1, dtype=float)
                    covariance = noise_covariance(wall, settings["conditions"][name], settings)
                    operator = whitened_shape_operator(wall, covariance)
                    losses = compare_profiles(templates[:, :, probe_indices], target, operator, options["relative_gain_bounds"])
                    fits = estimates(losses, barriers, options["landscape_mixtures"], probes,
                                     options["fitted_normal_shifts_nm"], len(wall.across), options["noise_likelihood_delta"])
                    result["samples"].setdefault(label, {**tags, "fits": {}})["fits"][family] = fits
                print(f"Fitted {family} model at {angle:g} degrees in {perf_counter()-start:.1f}s", flush=True)
                result["wall_seconds"] = perf_counter() - start
                checkpoint_guard()
                atomic_json(args.estimates, result)


def summarize(args, protocol, source_hashes):
    scored = json.loads(args.estimates.read_text())
    key = json.loads(args.truth.read_text())
    if fingerprint(args.truth) != scored["truth_sha256"] or scored["source_sha256"] != source_hashes:
        raise ValueError("The sealed key or fitting sources changed")
    if set(scored["samples"]) != set(key["truth"]):
        raise ValueError("The fitted sample set is incomplete")
    groups = {}
    for label, case in scored["samples"].items():
        true = key["truth"][label]["barrier_meV_per_cell"]
        case["truth"] = key["truth"][label]
        group = groups.setdefault(f"{case['twist_degrees']:g}/{case['condition']}", {})
        for family, regions in case["fits"].items():
            for region, shapes in regions.items():
                for shape, runs in shapes.items():
                    values = group.setdefault(f"{family}/{region}/{shape}", [])
                    values.extend((item["barrier_meV_per_cell"] / true, item["hit_barrier_bound"],
                                   item["barrier_interval"][0] <= true <= item["barrier_interval"][1],
                                   (item["barrier_interval"][1] - item["barrier_interval"][0]) / true,
                                   item["reduced_noise_loss"]) for item in runs)
    summary = {}
    for group, methods in groups.items():
        summary[group] = {}
        for method, values in methods.items():
            ratios, bounds, coverage, interval_width, noise_loss = np.array(values).T
            summary[group][method] = {"realizations": len(ratios), "median_ratio": float(np.median(ratios)),
                                      "ratio_p05_p95": np.quantile(ratios, (.05, .95)).tolist(),
                                      "median_absolute_relative_error": float(np.median(abs(ratios-1))),
                                      "fraction_within_20_percent": float(np.mean(abs(ratios-1) <= .20)),
                                      "bound_hits": int(bounds.sum()), "noise_interval_coverage": float(coverage.mean()),
                                      "median_noise_interval_width_over_truth": float(np.median(interval_width)),
                                      "median_reduced_noise_loss": float(np.median(noise_loss))}
    scored["summary"] = summary
    scored["reference_barrier_meV_per_cell"] = landscape_family()[0]
    scored["fit_before_reveal_sha256"] = fingerprint(args.estimates)
    scored["simulation_seed"] = key["seed"]
    checkpoint_guard()
    atomic_json(args.output, scored)
    for name, group in sorted(summary.items()):
        stats = group["periodic/both/profile_shapes"]
        print(name, "joint periodic median relative error", round(stats["median_absolute_relative_error"], 4),
              "within 20%", round(stats["fraction_within_20_percent"], 3), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    stages = parser.add_subparsers(dest="stage", required=True)
    simulate = stages.add_parser("generate")
    simulate.add_argument("images", type=Path)
    simulate.add_argument("truth", type=Path)
    simulate.add_argument("--seed", type=int, required=True)
    inverse = stages.add_parser("fit")
    inverse.add_argument("images", type=Path)
    inverse.add_argument("estimates", type=Path)
    report = stages.add_parser("score")
    report.add_argument("estimates", type=Path)
    report.add_argument("truth", type=Path)
    report.add_argument("output", type=Path)
    args = parser.parse_args()
    operations = {"generate": generate, "fit": fit, "score": summarize}
    operations[args.stage](args, json.loads(args.protocol.read_text()), sources(args.protocol))


if __name__ == "__main__":
    main()
