"""Transfer MoS2 empirical templates across the established map split."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter

from author_detector_replay import detector_intensity, rms
from author_release import load_author_release
from bilayer_relaxation import MODES, sample_disregistry
from observe_bilayers import _good_interior_triangle_mask, load_dataset
from run_antiparallel_signed_audit import local_geometry
from run_detector_prediction import apply_spatial
from run_detector_wall_profiles import to_scan_frame
from run_heterobilayer_diffraction_check import register_frame
from run_heterobilayer_ring_geometry import mean_rows
from run_heterobilayer_sparse_geometry import linear_response, solve_from_geometry
from run_heterobilayer_template_transfer import design_matrix, fit_templates
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_elastic_components import check_model_score, common_rings
from run_moire_regime_benchmark import SETTINGS, detector_data
from shared_compute import atomic_json, checkpoint_guard


BENCHMARK = PROJECT / "moire_regime_benchmark_v02.json"
PARALLEL = PROJECT / "shear_protocol_v01.json"
ANTIPARALLEL = PROJECT / "antiparallel_signed_protocol_v01.json"


def make_context(data, detector, reference, settings):
    measured, fitted, modes, coefficients, _ = detector
    crop, turns, reflect, transform, alignment = register_frame(data, fitted, modes, coefficients)
    if alignment != reference["detector_alignment"]:
        raise ValueError("The detector registration differs from the native replay")
    measured = to_scan_frame(measured[crop], turns, reflect)
    reciprocal = modes @ transform / data.lattice_nm
    rings = common_rings(modes)
    interface = tuple(reference["interface"])
    a = data.lattice_nm
    basis = a * np.array(((1.0, 0.5), (0.0, np.sqrt(3) / 2)))
    origin = basis @ np.full(2, 2/3 if data.orientation == "ap" else 0)
    fit = np.asarray(reference["affine_geometry"]["fit_scan"])
    gradient = np.asarray(reference["affine_geometry"]["gradient_crystal"])
    shape = data.u_nm.shape[:2]
    yy, xx = np.indices(shape, dtype=float)
    points = np.stack((xx, yy), axis=-1) * data.pixel_nm
    nominal = points @ fit[:2] + fit[-1]
    correction, _ = local_geometry(data, points, basis, fit, settings["center_inlier_nm"])
    rigid = nominal + correction
    valid = data.source_valid_mask
    sigma = settings["smoothing_sigma_pixels"]
    weights = gaussian_filter(valid.astype(float), sigma, mode="constant")
    interior, _ = _good_interior_triangle_mask(data)
    distance2 = np.full(shape, np.inf)
    for y, x in data.centers_rc_px:
        distance2 = np.minimum(distance2, (yy-y)**2 + (xx-x)**2)
    support = interior & valid & (weights > 0.97) & (distance2 > (2*sigma)**2)
    support[:4] = support[-4:] = support[:, :4] = support[:, -4:] = False
    if support.sum() != reference["score_pixels"]:
        raise ValueError("The native replay support changed")
    grid, full, _ = solve_from_geometry(gradient, basis, interface)
    linear = linear_response(grid, interface)
    ray_scale = reference["regime"]["ray_amplitude"]
    probe = settings["probe_fwhm_nm"] / (data.pixel_nm * np.sqrt(8 * np.log(2)))
    fields = {"rigid": rigid}
    for label, u in (("first_order", linear), ("energy_scaled_first_order", ray_scale*linear), ("full", full)):
        fields[label] = sample_disregistry(grid, u, origin + rigid, grid.cell) - origin
    return {
        "measured": measured, "reciprocal": reciprocal, "coefficients": coefficients,
        "basis": basis, "probe_pixels": probe, "rings": rings, "fields": fields, "support": support,
        "folds": [{"rigid": rigid, "support": support}],
    }


def score(context, templates, reference):
    support = context["support"]
    probe = context["probe_pixels"]
    rigid = context["fields"]["rigid"]
    fields = dict(context["fields"])
    for label, (modes, learned) in templates.items():
        fields[label] = rigid + design_matrix(rigid, context["basis"], modes) @ learned
    result = {}
    for name, field in fields.items():
        detector = detector_intensity(field, context["reciprocal"], context["coefficients"])
        signal = gaussian_filter(detector, (probe, probe, 0), mode="nearest")
        result[name] = {label: rms(context["measured"][support][:, ring], signal[support][:, ring])
                        for label, ring in context["rings"].items()}
    for key, values in reference["models"]["warped"].items():
        check_model_score(result[key], values["diffraction_rmse"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    parallel = json.loads(PARALLEL.read_text())["batches"]
    antiparallel = json.loads(ANTIPARALLEL.read_text())
    split = {"P": {"development": parallel["development"], "replay": parallel["held_out"]},
             "AP": {key: antiparallel[key] for key in ("development", "replay")}}
    settings = json.loads(SETTINGS.read_text())
    benchmark = json.loads(BENCHMARK.read_text())
    for name, expected in benchmark["input_sha256"].items():
        if digest(PROJECT / name) != expected:
            raise ValueError(f"The native replay input {name} changed")
    author = load_author_release(args.release, args.dependencies)
    inputs = (Path(__file__), BENCHMARK, PARALLEL, ANTIPARALLEL, SETTINGS, PROJECT / "run_heterobilayer_template_transfer.py")
    output = {
        "scope": "Learn one orientation-specific empirical Fourier displacement template on the existing MoS2 development scans, then predict the other previously examined scans. Both models get the same locally warped geometry and probe response. Device identity is incomplete. No coefficients fitted on the replay scans enter the empirical templates beyond their archived detector calibration.",
        "input_sha256": {p.name: digest(p) for p in inputs},
        "split": split, "templates": {}, "scans": {},
    }

    def load(scan, allowed):
        checkpoint_guard()
        data = load_dataset(scan, args.archive, allowed_ids=allowed, allow_author_crop=True)
        ref = benchmark["scans"][scan]
        if data.archive_md5 != ref["archive_md5"]:
            raise ValueError("The MoS2 archive changed")
        return make_context(data, detector_data(args.archive, scan, author), ref, settings)

    for orientation, cases in split.items():
        training = {scan: load(scan, cases["development"]) for scan in cases["development"]}
        templates = {}
        output["templates"][orientation] = {}
        for name, modes in (("empirical_first_harmonic", MODES[:3]), ("empirical_three_harmonics", MODES)):
            coefficients, diagnostics = fit_templates(list(training.values()), modes)
            templates[name] = (modes, coefficients)
            output["templates"][orientation][name] = {"modes": modes.tolist(), "coefficients_nm": coefficients.tolist(),
                                                       "diagnostics": diagnostics}
        for scan, context in training.items():
            output["scans"][scan] = score(context, templates, benchmark["scans"][scan])
        checkpoint_guard()
        atomic_json(args.output, output)
        for scan in cases["replay"]:
            context = load(scan, cases["replay"])
            result = score(context, templates, benchmark["scans"][scan])
            output["scans"][scan] = result
            print(scan, {name: row["both"] for name, row in result.items()}, flush=True)
        output["wall_seconds"] = perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, output)
    output["replay_mean"] = {k: mean_rows([output["scans"][scan] for scan in v["replay"]]) for k, v in split.items()}
    checkpoint_guard()
    atomic_json(args.output, output)


if __name__ == "__main__":
    main()
