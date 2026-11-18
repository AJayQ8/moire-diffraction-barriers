"""Compare the frozen physical model with a transferred periodic template."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

from author_detector_replay import detector_intensity, rms
from author_release import load_author_release
from bilayer_relaxation import MODES, sample_disregistry
from observe_bilayers import _inside_polygon_rc, load_dataset
from run_detector_prediction import AUTHOR_BASIS
from run_heterobilayer_corrugation import BASIS, MODEL_A, ORIGIN
from run_heterobilayer_diffraction_check import read_native_detector
from run_heterobilayer_ring_geometry import invert_local, mean_rows
from run_heterobilayer_sparse_geometry import linear_response, solve_from_geometry
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_elastic_components import RING_FREEZE, RING_RESULTS, check_model_score, common_rings
from run_moire_regime_benchmark import SETTINGS
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


DEV = ("DS30", "DS31", "DS33")
TEST = ("DS34", "DS35", "DS37")
INTERFACE = ("MoS2", "WSe2", "AP")
EXPOSURES = ("archived_reference", "fixed_full", "fixed_first_order", "rigid",
             "template_first_harmonic", "template_three_harmonics")


def design_matrix(rigid, lattice, modes):
    cycles = (rigid @ np.linalg.inv(lattice).T) @ modes.T
    phase = 2 * np.pi * cycles
    return np.concatenate((np.cos(phase), np.sin(phase)), axis=-1)


def get_context(data, detector, reference, settings):
    measured, _, modes, coefficients, calibration = detector
    frame = reference["first_ring_frame"]
    mirror = np.diag((-1, 1)) if frame["vector_reflection"] else np.eye(2)
    transform = np.linalg.inv(AUTHOR_BASIS) @ rotation(frame["vector_rotation_60deg_steps"] * np.pi/3) @ mirror
    reciprocal = modes @ transform / data.lattice_nm
    row, col = data.metadata.get("author_unwrap_crop_origin_row_col", (0, 0))
    height, width = data.u_nm.shape[:2]
    measured = measured[row:row+height, col:col+width]
    scale = MODEL_A / data.lattice_nm
    basis = BASIS / scale
    centers = np.asarray(reference["first_ring_centers_rc_px"])
    labels = np.asarray(reference["lattice_indices"])
    translations = labels @ basis.T
    groups = (labels[:, 0] - labels[:, 1]) % 3
    rows, columns = np.indices((height, width), dtype=float)
    points = np.stack((columns, rows), axis=-1) * data.pixel_nm
    center_points = centers[:, ::-1] * data.pixel_nm
    distance2 = np.full((height, width), np.inf)
    for y, x in centers:
        distance2 = np.minimum(distance2, (rows-y)**2 + (columns-x)**2)
    support = distance2 > (2 * settings["smoothing_sigma_pixels"])**2
    support[:4] = support[-4:] = support[:, :4] = support[:, -4:] = False
    probe = settings["probe_fwhm_nm"] / (data.pixel_nm * np.sqrt(8 * np.log(2)))
    folds = []
    for row in reference["folds"]:
        checkpoint_guard()
        train = groups != row["withheld_group"]
        inplane = np.column_stack((center_points[train], np.ones(train.sum())))
        fit = np.linalg.lstsq(inplane, translations[train], rcond=None)[0]
        nominal = points @ fit[:2] + fit[-1]
        delta = translations - (center_points @ fit[:2] + fit[-1])
        local = RBFInterpolator(center_points[train], delta[train], kernel="thin_plate_spline")
        warp = nominal + local(points.reshape(-1, 2)).reshape(points.shape)
        hull = ConvexHull(centers[train])
        keep = support & _inside_polygon_rc((height, width), centers[train][hull.vertices])
        if keep.sum() != row["pixels"]:
            raise ValueError("The first-ring training hull changed")
        gradient = fit[:2].T @ rotation(-np.deg2rad(data.sample_rotation_degrees))
        grid, shift, _ = solve_from_geometry(scale*gradient, BASIS, INTERFACE)
        phase = ORIGIN + scale*warp
        full = (sample_disregistry(grid, shift, phase, grid.cell) - ORIGIN) / scale
        first = linear_response(grid, INTERFACE)
        linear = (sample_disregistry(grid, first, phase, grid.cell) - ORIGIN) / scale
        folds.append({"rigid": warp, "full": full, "linear": linear, "support": keep,
                      "id": row["withheld_group"], "ref": row["diffraction_rmse"]})
    return {
        "measured": measured, "reciprocal": reciprocal, "coefficients": coefficients, "basis": basis,
        "rings": {k: v for k, v in common_rings(modes).items() if k != "both"},
        "probe_pixels": probe, "folds": folds, "calibration": calibration,
    }


def fit_templates(source, modes):
    """Fit a periodic relative displacement to development maps only."""
    inputs, values, examples = [], [], []
    for context in source:
        target = context["measured"]
        q, coeff = context["reciprocal"], context["coefficients"]
        for fold in context["folds"]:
            rigid = fold["rigid"]
            branch = invert_local(target, rigid, q, coeff)
            keep = fold["support"]
            design = design_matrix(rigid, context["basis"], modes)
            x = design[keep]
            y = (branch - rigid)[keep]
            weight = 1 / np.sqrt(len(y))
            inputs.append(weight * x)
            values.append(weight * y)
            examples.append((rigid, design, keep, target, q, coeff, context["probe_pixels"]))
    design = np.concatenate(inputs)
    displacement = np.concatenate(values)
    coefficients, _, rank, singular = np.linalg.lstsq(design, displacement, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("The development data do not constrain every empirical harmonic")

    def objective(flat):
        energy = 0.0
        derivative = np.zeros(coefficients.shape)
        template = flat.reshape(coefficients.shape)
        for rigid, design, keep, target, q, coeff, probe in examples:
            field = rigid + design @ template
            detector = detector_intensity(field, q, coeff)
            predicted = gaussian_filter(detector, (probe, probe, 0), mode="nearest")
            residual = np.where(keep[..., None], predicted - target, 0)
            normalization = keep.sum() * len(q) * len(examples)
            energy += 0.5 * np.sum(residual**2) / normalization
            adjoint = gaussian_filter(residual, (probe, probe, 0), mode="constant")
            phase = 2*np.pi * (field @ q.T)
            slope = np.pi * (-coeff[:, 0]*np.sin(phase) + coeff[:, 1]*np.cos(phase))
            gradient = (adjoint * slope) @ q
            derivative += design.reshape(-1, design.shape[-1]).T @ gradient.reshape(-1, 2) / normalization
        return float(energy), derivative.ravel()

    initial = coefficients.ravel()
    direction = np.random.default_rng(7865970).normal(size=initial.size)
    direction /= np.linalg.norm(direction)
    before, gradient = objective(initial)
    finite_difference = (objective(initial+1e-6*direction)[0] - objective(initial-1e-6*direction)[0]) / 2e-6
    derivative_error = float(abs(gradient @ direction - finite_difference))
    if derivative_error > 1e-7:
        raise ValueError("The detector template gradient failed its finite-difference check")
    fitted = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      callback=lambda _: checkpoint_guard(),
                      options={"gtol": 1e-7, "ftol": 1e-11, "maxiter": 130})
    if not fitted.success or fitted.fun > before:
        raise ValueError(f"The empirical detector template did not converge: {fitted.message}")
    return fitted.x.reshape(coefficients.shape), {
        "coefficient_count": int(coefficients.size), "design_condition": float(singular[0] / singular[-1]),
        "starting_train_rmse": float(np.sqrt(2*before)), "trained_rmse": float(np.sqrt(2*fitted.fun)),
        "iterations": int(fitted.nit), "directional_derivative_error": derivative_error,
    }


def evaluate(data, context, templates):
    measured = context["measured"]
    coefficients = context["coefficients"]
    reciprocal = context["reciprocal"]
    probe = context["probe_pixels"]
    folds = []
    for fold in context["folds"]:
        checkpoint_guard()
        rigid = fold["rigid"]
        support = fold["support"]
        models = {"rigid": rigid, "fixed_full": fold["full"], "fixed_first_order": fold["linear"]}
        for label, (modes, learned) in templates.items():
            models[label] = rigid + design_matrix(rigid, context["basis"], modes) @ learned
        scores = {}
        for label, field in models.items():
            signal = detector_intensity(field, reciprocal, coefficients)
            signal = gaussian_filter(signal, (probe, probe, 0), mode="nearest")
            scores[label] = {name: rms(measured[support][:, ring], signal[support][:, ring])
                             for name, ring in context["rings"].items()}
        check_model_score(scores["fixed_full"], fold["ref"]["warped_relaxed"])
        check_model_score(scores["fixed_first_order"], fold["ref"]["warped_linear"])
        check_model_score(scores["rigid"], fold["ref"]["warped_rigid"])
        fitted = detector_intensity(data.u_nm, reciprocal, coefficients)
        scores["archived_reference"] = {name: rms(measured[support][:, ring], fitted[support][:, ring])
                                        for name, ring in context["rings"].items()}
        if set(scores) != set(EXPOSURES):
            raise ValueError("The transfer comparison did not score every prespecified model")
        folds.append({"fold": fold["id"], "pixels": int(support.sum()), "scores": scores})
    return {"archive_md5": data.archive_md5, "calibration": context["calibration"], "folds": folds,
            "mean_over_folds": mean_rows([fold["scores"] for fold in folds])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch", choices=("fit", "replay"), required=True)
    parser.add_argument("--fitted", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    if args.batch == "replay" and not args.fitted:
        parser.error("The replay requires the saved development templates")
    frozen = json.loads(RING_FREEZE.read_text())
    if digest(RING_RESULTS[0]) != frozen["qualification_sha256"]:
        raise ValueError("The first-ring geometry qualification changed")
    for name, expected in frozen["input_sha256"].items():
        if digest(PROJECT / name) != expected:
            raise ValueError(f"Frozen geometry source {name} changed")
    settings = json.loads(SETTINGS.read_text())
    author = load_author_release(args.release, args.dependencies)
    inputs = (Path(__file__), SETTINGS, RING_FREEZE, RING_RESULTS[0], PROJECT / "run_moire_elastic_components.py")
    result = {
        "scope": "Transfer empirical Fourier displacement templates between the two source-identified capping devices. The template fit uses only source-device diffraction and phase coordinates inferred from first-ring centers. Optimize the source diffraction intensity through the same detector response and probe as the physics model. The two template orders are fixed. Compare to periodic elasticity on the same first-ring-only input geometry and score the second ring. Retrospective, with both devices previously examined.",
        "batch": args.batch, "development": DEV, "replay": TEST,
        "input_sha256": {p.name: digest(p) for p in inputs}, "scans": {},
    }
    batch = 0 if args.batch == "fit" else 1
    geometry = json.loads(RING_RESULTS[batch].read_text())["scans"]
    if args.batch == "fit":
        contexts = []
        for scan in DEV:
            checkpoint_guard()
            data = load_dataset(scan, args.archive, allowed_ids=DEV, allow_author_crop=True)
            detector = read_native_detector(args.archive, scan, author)
            context = get_context(data, detector, geometry[scan], settings)
            contexts.append((data, context))
        models = {
            "template_first_harmonic": MODES[:3],
            "template_three_harmonics": MODES,
        }
        templates = {}
        result["templates"] = {}
        for label, modes in models.items():
            coefficients, diagnostics = fit_templates([c for _, c in contexts], modes)
            templates[label] = (modes, coefficients)
            result["templates"][label] = {"modes": modes.tolist(), "coefficients_nm": coefficients.tolist(),
                                          "diagnostics": diagnostics}
        for scan, (data, context) in zip(DEV, contexts):
            result["scans"][scan] = evaluate(data, context, templates)
    else:
        learned = json.loads(args.fitted.read_text())
        if learned["input_sha256"] != result["input_sha256"] or learned["batch"] != "fit":
            raise ValueError("The development template protocol changed")
        templates = {label: (np.asarray(row["modes"]), np.asarray(row["coefficients_nm"]))
                     for label, row in learned["templates"].items()}
        result["fitted_sha256"] = digest(args.fitted)
        result["geometry_replay_sha256"] = digest(RING_RESULTS[1])
        for scan in TEST:
            checkpoint_guard()
            data = load_dataset(scan, args.archive, allowed_ids=TEST, allow_author_crop=True)
            detector = read_native_detector(args.archive, scan, author)
            context = get_context(data, detector, geometry[scan], settings)
            result["scans"][scan] = evaluate(data, context, templates)
            print(scan, {k: v["second"] for k, v in result["scans"][scan]["mean_over_folds"].items()}, flush=True)
    result["wall_seconds"] = perf_counter() - start
    checkpoint_guard()
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
