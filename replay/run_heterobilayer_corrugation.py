"""Audit the published ripple projection on the two WSe2/MoS2 devices."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from bilayer_relaxation import MONOLAYERS, gradient_field, sample_disregistry, stack_energy, wavevectors
from observe_bilayers import _cartesian_to_rz_wz, _good_interior_triangle_mask, _inside_polygon_rc
from observe_bilayers import _smooth_like_paper, load_dataset
from run_heterobilayer_residual_transfer import rms_vector
from run_heterobilayer_transfer import PROJECT, PROTOCOL, blur, digest, fingerprints, solve, verify_freeze
from shared_compute import atomic_json, checkpoint_guard


FREEZE = PROJECT / "heterobilayer_transfer_freeze_v01.json"
SETTINGS = PROJECT / "registry_protocol_v01.json"
MODEL_A = MONOLAYERS["MoS2"][0]
BASIS = MODEL_A * np.array(((1.0, 0.5), (0.0, np.sqrt(3) / 2)))
ORIGIN = BASIS @ np.full(2, 2 / 3)
MISMATCH_DFT_PERCENT = 100 * (1 - MODEL_A / MONOLAYERS["WSe2"][0])
HEIGHT_COEFFICIENTS_ANGSTROM = np.array((6.558, 0.115, -0.002, -0.019, -0.011, -0.013))
FIGURE_X = (1496, 1728, 1960)
FIGURE_Y_ZERO = 772
FIGURE_PIXELS_PER_POINT = 313
HALF_RANGE = 1 / np.sqrt(3)
LINE_BINS = np.arange(-9, 10) / 15
AP_STACKING_COORDINATES = np.array(((0, -HALF_RANGE), (0, 0), (0, HALF_RANGE)))


def table3_height(registry_nm, sign=1):
    """Return separation in nm and its derivative with respect to model registry."""
    # The shared harmonic evaluator divides its coefficients by 1000.
    return stack_energy(ORIGIN + sign * registry_nm, BASIS, 100 * HEIGHT_COEFFICIENTS_ANGSTROM)


def note9_dilation(registry_nm, mismatch_percent, sign):
    # Equal splitting of height and relative in-plane shift cancels in the
    # derivative of top-layer height with respect to top-layer shift.
    _, slope = table3_height(registry_nm, sign)
    compression = (1 - 1 / np.sqrt(1 + slope**2)).sum(axis=-1)
    return -mismatch_percent * compression


def digitize_figure(path):
    """Read curve center lines from the publisher's 2049 x 1132 figure."""
    rgb = np.asarray(Image.open(path).convert("RGB"))
    if rgb.shape != (1132, 2049, 3):
        raise ValueError("Unexpected dimensions for the published Fig. 4 image")
    curves = {}
    for name, color in (
        ("suspended_minus_full", (62, 75, 138)),
        ("suspended_minus_one_sided", (143, 151, 187)),
        ("gray_theory", (158, 158, 162)),
    ):
        r, c = np.nonzero((rgb == color).all(axis=-1))
        support = ((c >= 1501) & (c <= 1949) & (r >= 625) & (r <= 997)
                   & ~((c < 1738) & (r >= 896)))
        if name == "gray_theory":
            support &= (r >= FIGURE_Y_ZERO) & (r <= 969)
        r, c = r[support], c[support]
        column = np.unique(c)
        row = np.array([np.median(r[c == x]) for x in column])
        x = (column - FIGURE_X[1]) / (FIGURE_X[2] - FIGURE_X[1]) * HALF_RANGE
        y = (FIGURE_Y_ZERO - row) / FIGURE_PIXELS_PER_POINT
        curves[name] = {"stacking_coordinate_a0": x.tolist(), "dilation_percent": y.tolist()}
    gray = curves["gray_theory"]
    x, y = np.asarray(gray["stacking_coordinate_a0"]), np.asarray(gray["dilation_percent"])
    phase = MODEL_A * np.column_stack((np.zeros_like(x), x))
    literal = note9_dilation(phase, MISMATCH_DFT_PERCENT, -1)
    reversed_registry = note9_dilation(phase, MISMATCH_DFT_PERCENT, 1)
    factor = float(literal @ y / (literal @ literal))
    gray.update(note9_rmse_pp=float(np.sqrt(np.mean((y - literal)**2))),
                note9_flipped_rmse_pp=float(np.sqrt(np.mean((y - reversed_registry)**2))),
                note9_multiplier=factor, scaled_note9_rmse_pp=float(np.sqrt(np.mean((y - factor * literal)**2))),
                digitized_minimum_pp=float(y.min()), literal_minimum_pp=float(literal.min()))
    return curves


def dilation(field, data):
    """Use the source finite differences and single-layer trace convention."""
    dx = np.gradient(field, axis=1) / data.pixel_nm
    dy = np.gradient(field, axis=0) / data.pixel_nm
    angle = np.deg2rad(data.sample_rotation_degrees)
    c, s = np.cos(angle), np.sin(angle)
    return 50 * (c * (dx[..., 0] + dy[..., 1]) + s * (dx[..., 1] - dy[..., 0]))


def source_rigid_dilation(data, triangles, polygons, twist_degrees):
    """Fit the source lattice mismatch from the mean length of each triangle."""
    rigid = np.full(data.u_nm.shape[:2], np.nan)
    angle = np.deg2rad(twist_degrees)
    for triangle, support in zip(triangles, polygons):
        vertices = data.centers_rc_px[list(triangle)] * data.pixel_nm
        edges = np.linalg.norm(vertices - np.roll(vertices, 1, axis=0), axis=1)
        a, b, c = np.sort(edges)
        if (a*a + b*b - c*c) / (2*a*b) <= np.cos(np.deg2rad(160)):
            continue
        length = edges.mean()
        offset = data.lattice_nm**2 + length**2 * (np.cos(angle) - 1)
        if offset <= 0 or length <= data.lattice_nm:
            continue
        mismatch = (-offset + length * np.sqrt(offset)) / (length**2 - data.lattice_nm**2)
        rigid[support] = 100 * mismatch
    return rigid


def linecut(registry, support, fields):
    """Bin the vertical stacking cut as in the released linecut_avghex."""
    index_x, index_y = np.rint(registry * 15).astype(int).T
    result = {"bin_center_a0": [], "mean_registry_a0": [], "pixels": []}
    result.update({name: [] for name in fields})
    for j, coordinate in enumerate(LINE_BINS, -9):
        mask = support & (index_x == 0) & (index_y == j)
        if np.any(mask):
            result["bin_center_a0"].append(float(coordinate))
            result["mean_registry_a0"].append(float(registry[mask, 1].mean()))
            result["pixels"].append(int(mask.sum()))
            for name, field in fields.items():
                result[name].append(float(field[mask].mean()))
    return result


def periodic_correction(registry_nm, cell, factor, sign):
    """One curl-free periodic shift with the mean-removed Note 9 trace."""
    delta = note9_dilation(registry_nm, MISMATCH_DFT_PERCENT, sign) * factor / 100
    n = registry_nm.shape[0]
    k = wavevectors(cell, n)
    k2 = np.sum(k**2, axis=-1)
    trace = 2 * np.fft.fft2(delta - delta.mean())
    trace[n // 2] = 0
    trace[:, n // 2] = 0
    potential = np.divide(-1j * trace, k2, out=np.zeros_like(trace), where=k2 > 0)
    vector = np.fft.ifft2(k * potential[..., None], axes=(0, 1)).real
    return vector


def scores(observed, correction, support):
    error = observed[support]
    model = correction[support]
    error = error - error.mean(axis=0)
    model = model - model.mean(axis=0)
    centered = error - model
    mean_square = float(np.mean(error**2))
    model_square = float(np.mean(model**2))
    product = float(np.mean(error * model))
    is_vector = error.ndim == 2
    factor = np.sqrt(error.shape[1]) if is_vector else 1
    return {
        "baseline_rmse": factor * np.sqrt(mean_square),
        "corrected_rmse": factor * np.sqrt(np.mean(centered**2)),
        "correction_rms": factor * np.sqrt(model_square),
        "correlation": product / np.sqrt(mean_square * model_square),
        "optimal_multiplier_diagnostic": product / model_square,
    }


def run_scan(data, row, settings, interface, amplitudes, figure_factor):
    shape = data.u_nm.shape[:2]
    scale = MODEL_A / data.lattice_nm
    probe = settings["probe_fwhm_nm"] / np.sqrt(8 * np.log(2))
    sigma = settings["smoothing_sigma_pixels"]
    cap = row["encapsulation"]
    amplitude = amplitudes[cap]
    valid = data.source_valid_mask
    weights = gaussian_filter(valid.astype(float), sigma, mode="constant")
    yy, xx = np.indices(shape, dtype=float)
    points = np.stack((xx, yy), axis=-1) * data.pixel_nm
    mask, triangles = _good_interior_triangle_mask(data)
    polygons = np.stack([_inside_polygon_rc(shape, data.centers_rc_px[list(tri)]) for tri in triangles])
    gradient = np.asarray(row["whole_geometry"]["gradient_crystal"])
    grid, shift, _ = solve(scale * gradient, BASIS, interface, row["grid"])
    model_registry = grid.fractional @ BASIS.T + amplitude * shift
    _, height_slope = table3_height(model_registry)
    in_plane_gradient = scale * gradient + gradient_field(grid, amplitude * shift)
    spatial_slope = 0.5 * np.einsum("...i,...ij->...j", height_slope, in_plane_gradient)
    real_space_projection = -50 * np.sum(spatial_slope**2, axis=-1)
    source_smoothed = _smooth_like_paper(data.u_nm, sigma)
    measured_strain = dilation(source_smoothed, data)
    rigid_strain = source_rigid_dilation(data, triangles, polygons, row["diffraction_twist_deg"])
    if np.nanmean(measured_strain) <= 0:
        raise ValueError(f"{data.dataset_id}: unexpected sign of measured dilation")
    measured_registry = _cartesian_to_rz_wz(source_smoothed / data.lattice_nm)
    finite_registry = np.isfinite(measured_registry).all(axis=-1)
    registry = np.where(finite_registry[..., None], measured_registry, 0)
    chart_mask = mask & valid & np.isfinite(rigid_strain) & finite_registry
    fields = {"dilation_percent": measured_strain, "source_subtracted_percent": measured_strain - rigid_strain,
              "note9_at_measured_registry_percent": note9_dilation(registry * MODEL_A, MISMATCH_DFT_PERCENT, -1)}
    chart = linecut(registry.reshape(-1, 2), chart_mask.ravel(), {name: f.ravel() for name, f in fields.items()})
    distance2 = np.full(shape, np.inf)
    for y, x in data.centers_rc_px:
        distance2 = np.minimum(distance2, (yy - y)**2 + (xx - x)**2)
    shared = mask & valid & (weights > 0.97) & (distance2 > (2 * sigma)**2)
    predictions = {"note9_reflected_gauge": (-1, 1.0), "note9_frozen_gauge": (1, 1.0),
                   "figure_scale_reflected_gauge": (-1, figure_factor),
                   "figure_scale_frozen_gauge": (1, figure_factor)}
    halves = []
    for axis, partition in row["partitions"].items():
        regions = [polygons[h["triangles"]].any(axis=0) for h in partition]
        for i, half in enumerate(partition):
            fit = np.asarray(half["geometry"]["fit_scan"])
            phase = ORIGIN + scale * (points @ fit[:2] + fit[-1])
            cell = np.asarray(half["geometry"]["cell_nm"])
            rigid = (phase - ORIGIN) / scale
            relaxed = (sample_disregistry(grid, shift, phase, cell, probe) - ORIGIN) / scale
            residual = blur(data.u_nm - (rigid + amplitude * (relaxed - rigid)), valid, weights, sigma)
            support = shared & regions[i] & ~regions[1 - i]
            if support.sum() != half["primary"]["pixel_count"]:
                raise ValueError(f"{data.dataset_id}: primary support changed")
            measured_residual_strain = dilation(residual, data)
            values = {}
            for name, (sign, strength) in predictions.items():
                vector = periodic_correction(model_registry, cell, strength, sign)
                vector = (sample_disregistry(grid, vector, phase, cell, probe) - phase) / scale
                correction = blur(vector, valid, weights, sigma)
                values[name] = {
                    "layer_shift_nm": scores(residual, correction, support),
                    "dilation_pp": scores(measured_residual_strain, dilation(correction, data), support),
                }
            halves.append({"axis": axis, "half": i, "pixels": int(support.sum()),
                           "amplitude_only_rmse_nm": rms_vector(residual[support]), "scores": values})
    means = {
        name: {quantity: {key: float(np.mean([h["scores"][name][quantity][key] for h in halves]))
                          for key in halves[0]["scores"][name][quantity]}
               for quantity in ("layer_shift_nm", "dilation_pp")}
        for name in predictions
    }
    heights = table3_height(model_registry)[0]
    return {
        "encapsulation": cap, "archive_md5": data.archive_md5, "amplitude_from_setup": amplitude,
        "height_variation_one_layer_nm": float(np.ptp(heights) / 2),
        "real_space_single_layer_foreshortening_max_abs_pp": float(-real_space_projection.min()),
        "chart_style_linecut": chart, "halves": halves, "mean_over_halves": means,
    }


def compare_profiles(samples, figure):
    suspended = samples["suspended"]["chart_style_linecut"]
    result = {}
    for cap, figure_name in (("full", "suspended_minus_full"), ("one-sided", "suspended_minus_one_sided")):
        covered = samples[cap]["chart_style_linecut"]
        x = -np.asarray(suspended["mean_registry_a0"])[::-1]
        covered_x = -np.asarray(covered["mean_registry_a0"])[::-1]
        figure_x = np.asarray(figure[figure_name]["stacking_coordinate_a0"])
        # The outer bins land at slightly different registry coordinates.
        # Interpolate only where the two scans and the chart have support.
        common = (x >= max(covered_x.min(), figure_x.min())) & (x <= min(covered_x.max(), figure_x.max()))
        x = x[common]
        phase = MODEL_A * np.column_stack((np.zeros_like(x), x))
        theory = note9_dilation(phase, MISMATCH_DFT_PERCENT, -1)
        gray = figure["gray_theory"]
        gray_value = np.interp(x, gray["stacking_coordinate_a0"], gray["dilation_percent"])
        figure_y = figure[figure_name]["dilation_percent"]
        published = np.interp(x, figure_x, figure_y)
        values = {"coordinate_a0": x.tolist(), "note9_percent": theory.tolist(), "digitized_gray_percent": gray_value.tolist(),
                  "published_difference_percent": published.tolist()}
        for key in ("dilation_percent", "source_subtracted_percent"):
            contrast = np.asarray(suspended[key])[::-1][common] - np.interp(x, covered_x, covered[key][::-1])
            values[key] = {
                "contrast_percent": contrast.tolist(),
                "rmse_to_published_lower_twist_pp": float(np.sqrt(np.mean((contrast - published)**2))),
                "rmse_without_registry_reversal_pp": float(np.sqrt(np.mean((contrast - np.interp(-x, figure_x, figure_y))**2))),
                "rmse_to_published_lower_twist_after_offset_pp": float(np.std(contrast - published)),
                "vs_note9": scores(contrast, theory, np.ones_like(x, dtype=bool)),
                "vs_digitized_gray": scores(contrast, gray_value, np.ones_like(x, dtype=bool)),
            }
        result[figure_name] = values
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("figure", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    protocol = json.loads(PROTOCOL.read_text())
    settings = json.loads(SETTINGS.read_text())
    verify_freeze(FREEZE, fingerprints(), protocol["devices"]["development"])
    amplitude_file = PROJECT / "heterobilayer_capping_development_v01.json"
    amplitudes = json.loads(amplitude_file.read_text())["fitted_amplitudes"]["by_encapsulation"]
    figure = digitize_figure(args.figure)
    figure_factor = figure["gray_theory"]["note9_multiplier"]
    heights = (table3_height(AP_STACKING_COORDINATES * MODEL_A, -1)[0] * 10).tolist()
    result = {
        "scope": "Retrospective two-device diagnostic using published height parameters. Figure 4k uses the negative of the numerical archive y coordinate for MM to XX. This matches the frozen in-plane model's registry order and improves agreement with the digitized measured profiles. The public line-cut helper uses the opposite endpoint labels. Compare the Note 9 trace and figure-scale sensitivity under both displacement gauges. The curl-free vector is an extra assumption for the diagnostic. Neither device supplies independent height data.",
        "input_sha256": {p.name: digest(p) for p in (Path(__file__), FREEZE, SETTINGS, amplitude_file, args.figure)},
        "stacking_height_angstrom_MM_XMMX_XX": heights,
        "figure4k": figure,
        "scans": {},
        "contrasts": {},
    }
    for batch, cases in protocol["devices"].items():
        frozen_score = PROJECT / f"heterobilayer_transfer_{batch}_v01.json"
        capping_path = PROJECT / f"heterobilayer_capping_{batch}_v01.json"
        capping = json.loads(capping_path.read_text())
        if digest(frozen_score) != capping["frozen_score_sha256"]:
            raise ValueError("The amplitude comparison used a different transfer result")
        result["input_sha256"].update({p.name: digest(p) for p in (frozen_score, capping_path)})
        prior = json.loads(frozen_score.read_text())
        device = {}
        for scan in cases:
            checkpoint_guard()
            data = load_dataset(scan, args.archive, allowed_ids=cases, allow_author_crop=True)
            row = prior["scans"][scan]
            if data.archive_md5 != row["archive_md5"]:
                raise ValueError(f"{scan}: archive changed")
            diagnostic = run_scan(data, row, settings, tuple(protocol["interface"]), amplitudes, figure_factor)
            reference = capping["scans"][scan]["primary"]["average_half_rmse_nm"]["by_encapsulation"]
            replay = np.mean([h["amplitude_only_rmse_nm"] for h in diagnostic["halves"]])
            if not np.isclose(reference, replay, rtol=0, atol=1e-10):
                raise ValueError(f"{scan}: amplitude-only reference changed")
            result["scans"][scan] = diagnostic
            device[diagnostic["encapsulation"]] = diagnostic
            print(scan, diagnostic["encapsulation"], f"{replay:.5f} nm", flush=True)
        result["contrasts"][batch] = compare_profiles(device, figure)
        result["wall_seconds"] = perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
