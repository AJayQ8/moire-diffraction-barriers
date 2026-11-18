"""Rebuild parallel MoS2 geometry using only the first-ring fringes."""

import argparse
import json
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

from author_detector_replay import detector_intensity, rms
from author_release import load_author_release
from bilayer_relaxation import MONOLAYERS, STACKING, sample_disregistry
from observe_bilayers import _inside_polygon_rc, load_dataset
from run_detector_prediction import AUTHOR_BASIS
from run_heterobilayer_cross_ring import refine_minimum
from run_heterobilayer_first_ring_seed import circular_error
from run_heterobilayer_sparse_geometry import linear_response, solve_from_geometry
from run_heterobilayer_template_transfer import design_matrix
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_elastic_components import common_rings
from run_moire_regime_benchmark import SETTINGS, detector_data
from run_mos2_cross_ring import TEMPLATES, smooth_cost
from run_mos2_directional_diffraction import camera
from run_mos2_directional_split_ring import stretch_controls
from run_mos2_soliton_approximation import soliton_motion
from run_mos2_template_transfer import BENCHMARK, PARALLEL
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


INTERFACE = ("MoS2", "MoS2", "P")
DEVELOPMENT = "development"
REPLAY = "held_out"
SIGNS = np.asarray([(1, s, t) for s, t in product((-1, 1), repeat=2)])


def paired_axes(modes):
    """Use opposite reflections as repeats of each of three spatial frequencies."""
    keys = []
    signs = []
    for mode in modes:
        h, k = mode
        sign = 1 if h > 0 or h == 0 and k > 0 else -1
        keys.append(tuple(sign*mode))
        signs.append(sign)
    unique, index = np.unique(keys, axis=0, return_inverse=True)
    if len(unique) != 3:
        raise ValueError("The first ring must contain three nonparallel reciprocal axes")
    return unique, index, np.array(signs)


def fringes(values, coefficients, modes, pixel_nm):
    """Return separated Fourier candidates without archived center positions."""
    height, width = values.shape[:2]
    yy, xx = np.indices((height, width), dtype=float)
    xy = np.stack((xx, yy), axis=-1)*pixel_nm
    window = np.outer(np.hanning(height), np.hanning(width))
    shape = (4*height, 4*width)
    fy, fx = np.meshgrid(np.fft.fftfreq(shape[0], pixel_nm),
                         np.fft.fftfreq(shape[1], pixel_nm), indexing="ij")
    native_bin = 1/(min(height, width)*pixel_nm)
    length = np.hypot(fx, fy)
    half = (fy > 0) | ((fy == 0) & (fx > 0))
    allowed = half & (length >= 0.75*native_bin) & (length <= 0.25)
    center = np.average(values.reshape(-1, values.shape[-1]), axis=0, weights=window.ravel())
    contrast = np.hypot(coefficients[:, 0], coefficients[:, 1])
    if np.min(contrast) < 0.02:
        raise ValueError("A first-ring channel has weak spatial contrast")
    weighted = window[..., None]*(values-center)/contrast
    spectrum = np.fft.fft2(weighted, s=shape, axes=(0, 1))
    unique, axis, direction = paired_axes(modes)
    peaks, powers = [], []
    for i in range(3):
        power = np.mean(np.abs(spectrum[..., axis == i])**2, axis=-1)
        local = allowed & (power == maximum_filter(power, size=7, mode="wrap"))
        rows, columns = np.nonzero(local)
        order = np.argsort(power[rows, columns])[-5:][::-1]
        rows, columns = rows[order], columns[order]
        if len(rows) < 2:
            raise ValueError("A first-ring axis has too few spectral maxima")
        peaks.append(np.column_stack((fx[rows, columns], fy[rows, columns])))
        powers.append(power[rows, columns]/power[rows[0], columns[0]])
    return xy, window, weighted, unique, axis, direction, peaks, powers, native_bin


def calibrate_frame(reciprocal_basis, frequencies, sample_angle):
    """Resolve the lattice-frame ambiguity with the published sample rotation."""
    signed = np.linalg.lstsq(reciprocal_basis, frequencies, rcond=None)[0]
    if np.linalg.det(signed) == 0:
        return None
    # The sign of the displacement basis and 60-degree indexing are
    # diffraction gauges. In this project negative twist is the convention.
    reflection = np.diag((-1, 1)) if np.linalg.det(signed) < 0 else np.eye(2)
    trials = []
    for turns in range(6):
        transform = rotation(turns*np.pi/3) @ reflection
        gradient = transform.T @ signed
        crystal = gradient @ rotation(-sample_angle)
        theta = (crystal[1, 0]-crystal[0, 1])/2
        mismatch = np.trace(crystal)/2
        symmetric = (crystal+crystal.T)/2
        if -0.070 < theta < -0.003 and abs(mismatch) < abs(theta):
            relative = float(np.linalg.norm(symmetric)/np.linalg.norm(crystal))
            trials.append((relative, transform, gradient))
    return min(trials, key=lambda x: x[0]) if trials else None


def affine_seed(values, coefficients, modes, pixel_nm, lattice_nm, sample_angle, phase_basis):
    xy, weights, weighted, unique, axis, direction, peaks, powers, frequency_bin = fringes(
        values, coefficients, modes, pixel_nm)
    unit = np.linalg.inv(AUTHOR_BASIS)/lattice_nm
    basis = unique @ unit
    raw_reciprocal = modes @ unit
    choices = []
    for indices in product(*(range(len(p)) for p in peaks)):
        coordinates = np.array([p[k] for p, k in zip(peaks, indices)])
        strength = -sum(np.log(max(p[k], 1e-10)) for p, k in zip(powers, indices))
        for sign in SIGNS:
            q = coordinates*sign[:, None]
            spectral_gradient = np.linalg.lstsq(basis, q, rcond=None)[0]
            disagreement = np.sqrt(np.mean(np.sum((basis@spectral_gradient-q)**2, axis=-1)))
            if disagreement > 0.45*frequency_bin:
                continue
            frame = calibrate_frame(basis, q, sample_angle)
            if frame is None:
                continue
            relative, transform, gradient = frame
            loss = strength + 8*(disagreement/frequency_bin)**2 + 3*relative**2
            choices.append((loss, transform, gradient, q, disagreement))
    if not choices:
        raise ValueError("The first-ring frequencies do not form a small-twist lattice")
    flat = xy.reshape(-1, 2)
    center = np.average(flat, weights=weights.ravel(), axis=0)
    reduced = np.s_[::2, ::2]
    points = xy[reduced].reshape(-1, 2)-center
    expected = values[reduced].reshape(-1, len(modes))
    weight = np.sqrt(weights[reduced].reshape(-1, 1))
    frac = np.arange(20)/20
    offsets = np.column_stack([v.ravel() for v in np.meshgrid(frac, frac)]) @ phase_basis.T
    best = []
    for loss, transform, gradient, peaks_now, error in sorted(choices, key=lambda x: x[0])[:4]:
        reciprocal = raw_reciprocal @ transform
        frequencies = peaks_now[axis]*direction[:, None]
        phase = []
        for i, (frequency, (a, b, _)) in enumerate(zip(frequencies, coefficients)):
            signal = np.sum(weighted[..., i]*np.exp(-2j*np.pi*(xy@frequency)))
            phase.append(np.angle(signal/(a-1j*b))/(2*np.pi))
        phase -= frequencies@center
        offset_index = np.argmin(np.mean(circular_error(offsets@reciprocal.T-phase)**2, axis=1))
        initial = np.r_[gradient.ravel(), offsets[offset_index]+gradient@center]
        a, b, _ = coefficients.T

        def residual(parameters):
            field = points@parameters[:4].reshape(2, 2).T+parameters[4:]
            return ((detector_intensity(field, reciprocal, coefficients)-expected)*weight).ravel()

        def jacobian(parameters):
            field = points@parameters[:4].reshape(2, 2).T+parameters[4:]
            wave = 2*np.pi*(field@reciprocal.T)
            derivative = np.pi*(-a*np.sin(wave)+b*np.cos(wave))*weight
            slope = np.einsum("ij,ja,ib->ijab", derivative, reciprocal, points).reshape(-1, 4)
            origin = (derivative[..., None]*reciprocal).reshape(-1, 2)
            return np.hstack((slope, origin))

        fit = least_squares(residual, initial, jac=jacobian, x_scale="jac", max_nfev=85,
                            ftol=1e-6, gtol=1e-7)
        if fit.success:
            g = fit.x[:4].reshape(2, 2)
            if np.linalg.det(g) > 0 and np.linalg.norm(g-gradient) < 0.4*np.linalg.norm(gradient):
                seed = np.vstack((g.T, fit.x[4:]-g@center))
                best.append((float(np.sqrt(np.sum(fit.fun**2)/(np.sum(weight**2)*len(modes)))),
                             transform, seed, {"seed_disagreement_cycles_per_nm": float(error),
                                               "spectral_cost": float(loss), "frequencies_cycles_per_nm": frequencies.tolist()}))
    if not best:
        raise ValueError("No first-ring spectral candidate fit the observed fringes")
    fit_error, transform, seed, diagnostics = min(best, key=lambda x: x[0])
    diagnostics["affine_first_ring_rmse"] = fit_error
    return seed, transform, diagnostics


def centers_from_seed(values, coefficients, reciprocal, basis, seed, pixel_nm):
    cost = smooth_cost(values, coefficients, reciprocal, np.zeros(2))
    height, width = cost.shape
    yy, xx = np.indices(cost.shape)
    corner = pixel_nm*np.array(((0, 0), (0, height-1), (width-1, 0), (width-1, height-1)))
    frac = (corner@seed[:2]+seed[-1])@np.linalg.inv(basis).T
    low, high = np.floor(frac.min(axis=0)).astype(int), np.ceil(frac.max(axis=0)).astype(int)
    labels = np.array(list(product(*(range(a, b+1) for a, b in zip(low, high)))))
    translations = labels@basis.T
    expected = (translations-seed[-1])@np.linalg.inv(seed[:2])/pixel_nm
    moire = np.linalg.solve(seed[:2].T, basis)/pixel_nm
    shortest = min(np.linalg.norm(moire[:, 0]), np.linalg.norm(moire[:, 1]), np.linalg.norm(moire[:, 0]-moire[:, 1]))
    radius = 0.26*shortest
    inside = (yy >= 2) & (yy < height-2) & (xx >= 2) & (xx < width-2)
    centers, keep = [], []
    for i, (x, y) in enumerate(expected):
        select = inside & ((xx-x)**2+(yy-y)**2 <= radius**2)
        if not (2 <= x < width-2 and 2 <= y < height-2) or not np.any(select):
            continue
        r, c = np.unravel_index(np.argmin(np.where(select, cost, np.inf)), cost.shape)
        center = refine_minimum(cost, r, c)
        if np.linalg.norm(center-(y, x)) < radius-0.5:
            centers.append(center)
            keep.append(i)
    if len(centers) < 5:
        raise ValueError("The first ring supports fewer than five stacking centers")
    centers = np.array(centers)
    points = centers[:, ::-1]*pixel_nm
    design = np.column_stack((points, np.ones(len(points))))
    fit = np.linalg.lstsq(design, translations[keep], rcond=None)[0]
    if np.linalg.det(fit[:2]) <= 0:
        raise ValueError("Center positions changed the lattice handedness")
    error = np.linalg.norm(design@fit-translations[keep], axis=1)
    if error.max() > 0.075 or np.linalg.norm(fit[:2]-seed[:2]) > 0.35*np.linalg.norm(seed[:2]):
        raise ValueError("The first-ring center residual exceeds the fixed geometry check")
    return fit, centers, {"first_ring_centers": len(centers), "center_maximum_residual_nm": float(error.max()),
                          "center_rms_residual_nm": float(np.sqrt(np.mean(error**2)))}


def predict(data, detector, settings, template):
    measured, _, modes, coefficients, _ = detector
    row, col = data.metadata.get("author_unwrap_crop_origin_row_col", (0, 0))
    height, width = data.u_nm.shape[:2]
    observed = measured[row:row+height, col:col+width]
    rings = common_rings(modes)
    first = rings["first"]
    basis = data.lattice_nm*np.array(((1, 0.5), (0, np.sqrt(3)/2)))
    sample_angle = np.deg2rad(data.sample_rotation_degrees)
    seed, orient, spectral = affine_seed(observed[..., first], coefficients[first], modes[first], data.pixel_nm,
                                          data.lattice_nm, sample_angle, basis)
    reciprocal = (modes @ np.linalg.inv(AUTHOR_BASIS) @ orient)/data.lattice_nm
    fit, centers, local = centers_from_seed(observed[..., first], coefficients[first], reciprocal[first],
                                             basis, seed, data.pixel_nm)
    yy, xx = np.indices((height, width), dtype=float)
    points = np.stack((xx, yy), axis=-1)*data.pixel_nm
    background = points@fit[:2]+fit[-1]
    gradient = fit[:2].T@rotation(-sample_angle)
    support = _inside_polygon_rc((height, width), centers[ConvexHull(centers).vertices])
    distance = np.full((height, width), np.inf)
    for y, x in centers:
        distance = np.minimum(distance, (yy-y)**2+(xx-x)**2)
    support &= distance > (2*settings["smoothing_sigma_pixels"])**2
    support[:4] = support[-4:] = support[:, :4] = support[:, -4:] = False
    if support.sum() < 300:
        raise ValueError("The first-ring geometry supports fewer than 300 scored positions")
    context = {"reciprocal": reciprocal, "coefficients": coefficients,
               "probe_pixels": settings["probe_fwhm_nm"]/(data.pixel_nm*np.sqrt(8*np.log(2)))}
    observed_cell = np.linalg.solve(gradient, basis)
    fields = {"rigid": background}
    for name, candidate in stretch_controls(gradient):
        checkpoint_guard()
        grid, shift, _ = solve_from_geometry(candidate, basis, INTERFACE)
        fields[name] = sample_disregistry(grid, shift, background, observed_cell)
        if name == "measured_direction":
            fields["first_order"] = sample_disregistry(grid, linear_response(grid, INTERFACE), background, observed_cell)
    phase = design_matrix(background, basis, np.asarray(template["modes"]))
    fields["empirical_three_harmonics"] = background + phase@np.asarray(template["coefficients_nm"])
    _, c1, c2, c3, _, _ = STACKING[INTERFACE]
    modulus = MONOLAYERS["MoS2"][2]
    twist = (gradient[1, 0]-gradient[0, 1])/2
    for name, coupling in (("soliton_first_star", c1), ("soliton_effective_adhesion", c1-8*c2+9*c3)):
        motion, _ = soliton_motion(background, basis, twist, coupling/(2000*modulus))
        fields[name] = background+motion
    scores = {}
    for name, field in fields.items():
        image = camera(field, context)
        scores[name] = {k: rms(observed[support][:, channels], image[support][:, channels])
                        for k, channels in rings.items()}
    return {
        "score_pixels": int(support.sum()), "geometry_only_from_first_ring": {**spectral, **local},
        "archive_md5": data.archive_md5, "fit_affine_scan": fit.tolist(),
        "directional_gradient_crystal": gradient.tolist(), "detector_to_crystal_transform": orient.tolist(),
        "centers_rc_px": centers.tolist(), "score_rmse": scores,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch", choices=(DEVELOPMENT, REPLAY), required=True)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    settings, protocol, template = (json.loads(p.read_text()) for p in (SETTINGS, PARALLEL, TEMPLATES))
    template = template["templates"]["P"]["empirical_three_harmonics"]
    scans = protocol["batches"][args.batch]
    inputs = (Path(__file__), SETTINGS, PARALLEL, TEMPLATES, BENCHMARK,
              *(PROJECT / f for f in ("run_heterobilayer_cross_ring.py", "run_heterobilayer_first_ring_seed.py",
                                       "run_mos2_cross_ring.py", "run_mos2_directional_diffraction.py",
                                       "run_mos2_directional_split_ring.py", "run_mos2_soliton_approximation.py",
                                       "run_moire_regime_benchmark.py", "bilayer_relaxation.py")))
    fingerprints = {p.name: digest(p) for p in inputs}
    if args.batch == REPLAY:
        if args.freeze is None:
            parser.error("Use the development freeze for the replay")
        frozen = json.loads(args.freeze.read_text())
        source = args.freeze.parent / frozen["development_file"]
        if frozen["input_sha256"] != fingerprints or frozen["development_sha256"] != digest(source):
            raise ValueError("The first-ring geometry method changed after development")
        if frozen["replay_scans"] != scans:
            raise ValueError("The first-ring replay changed its map list")
    author = load_author_release(args.release, args.dependencies)
    result = {
        "scope": "Infer all positions and the lattice geometry from first-ring spatial frequencies, phases and intensities. Keep the original metadata pixel size and sample rotation, and the scan-wide detector calibration. Use the same geometry and scoring positions for each model. Compare the second ring with no archived displacement field or stacking-center positions as geometry inputs.",
        "limits": "This retrospective analysis reuses the released scan-wide detector coefficients and coordinate format. The empirical template is trained on the original development maps and is a replay baseline on the other maps.",
        "batch": args.batch, "input_sha256": fingerprints,
        "scans": {}, "failed_scans": {},
    }
    if args.freeze:
        result["freeze_sha256"] = digest(args.freeze)
    for scan in scans:
        checkpoint_guard()
        data = load_dataset(scan, args.archive, allowed_ids=scans, allow_author_crop=True)
        detector = detector_data(args.archive, scan, author)
        try:
            row = predict(data, detector, settings, template)
        except (ValueError, np.linalg.LinAlgError) as exc:
            result["failed_scans"][scan] = str(exc)
            print(scan, str(exc), flush=True)
        else:
            result["scans"][scan] = row
            print(scan, {k: round(v["second"], 5) for k, v in row["score_rmse"].items()}, flush=True)
        result["wall_seconds"] = perf_counter()-start
        checkpoint_guard()
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
