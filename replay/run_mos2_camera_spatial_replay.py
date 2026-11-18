"""Test the diffraction predictions after spatially disjoint detector calibration."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter

from author_detector_replay import read_detector_fit, rms
from author_release import load_author_release
from bilayer_relaxation import MONOLAYERS, STACKING, sample_disregistry
from observe_bilayers import load_dataset
from run_detector_prediction import AUTHOR_BASIS
from run_detector_wall_profiles import to_scan_frame
from run_heterobilayer_sparse_geometry import solve_from_geometry
from run_heterobilayer_template_transfer import design_matrix
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_regime_benchmark import SETTINGS, detector_data
from run_mos2_directional_diffraction import camera
from run_mos2_directional_split_ring import SPLIT_RING, readout_context, stretch_controls
from run_mos2_soliton_approximation import soliton_motion
from run_mos2_template_transfer import BENCHMARK
from run_mos2_cross_ring import TEMPLATES
from shared_compute import atomic_json, checkpoint_guard


def fit_camera(data, detector, native_phase, benchmark, context, training):
    """Estimate camera terms on training pixels using the archived local phase."""
    _, _, modes, _, _ = detector
    row, col = data.metadata.get("author_unwrap_crop_origin_row_col", (0, 0))
    height, width = data.u_nm.shape[:2]
    actual = context["measured"][training]
    # The fit's local phases were extracted pixelwise with the released
    # scan-wide coefficients. Select the detector coordinate convention
    # verified in the native replay and apply the same spatial alignment.
    native = benchmark["detector_calibration"]["saved_field_basis"]
    stored = native_phase[row:row+height, col:col+width]
    field = stored if native == "lattice" else stored @ np.linalg.inv(AUTHOR_BASIS).T
    frame = benchmark["detector_alignment"]
    field = to_scan_frame(field, frame["spatial_turns"], frame["spatial_reflection"])
    phase = np.pi * (field[training] @ modes.T)
    cos, sin = np.cos(phase), np.sin(phase)
    matrix = np.stack((cos**2, sin*cos, np.ones_like(phase)), axis=-1)
    coeff, condition, rms_values = [], [], []
    for channel in range(phase.shape[1]):
        x, y = matrix[:, channel], actual[:, channel]
        estimate, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
        if rank != 3 or singular[0] / singular[-1] > 30:
            raise ValueError("A training half does not identify its detector coefficients")
        coeff.append(estimate)
        condition.append(float(singular[0] / singular[-1]))
        rms_values.append(rms(x @ estimate, y))
    return np.array(coeff), {"worst_condition": max(condition), "training_channel_rmse": rms_values}


def phase_camera(field, reciprocal, probe):
    phase = np.pi * (field @ reciprocal.T)
    cos, sin = np.cos(phase), np.sin(phase)
    return tuple(gaussian_filter(f, (probe, probe, 0), mode="nearest") for f in (cos**2, sin*cos))


def score_scan(data, detector, native_phase, benchmark, reference, settings, learned):
    context, basis, background, gradient, support, rings = readout_context(data, detector, benchmark, reference, settings)
    interface = tuple(benchmark["interface"])
    origin = basis @ np.full(2, 2/3 if interface[-1] == "AP" else 0.0)
    observed_cell = np.linalg.solve(gradient, basis)
    fields = {"rigid": background}
    for label, geometry in stretch_controls(gradient):
        grid, motion, _ = solve_from_geometry(geometry, basis, interface)
        fields[label] = sample_disregistry(grid, motion, origin + background, observed_cell) - origin
    for label, template in learned.items():
        field = design_matrix(background, basis, np.asarray(template["modes"]))
        fields[label] = background + field @ np.asarray(template["coefficients_nm"])
    if interface[-1] == "P":
        _, first, second, third, _, _ = STACKING[interface]
        modulus = MONOLAYERS["MoS2"][2]
        twist = (gradient[1, 0]-gradient[0, 1])/2
        displacement, _ = soliton_motion(background, basis, twist, (first-8*second+9*third)/(2000*modulus))
        fields["marginal_twist_formula"] = background + displacement
    predictions = {name: phase_camera(field, context["reciprocal"], context["probe_pixels"])
                   for name, field in fields.items()}
    coeff = context["coefficients"]
    actual = context["measured"]
    for key in ("measured_direction", "rigid"):
        a, b = predictions[key]
        check = a*coeff[:, 0] + b*coeff[:, 1] + coeff[:, 2]
        if not np.allclose(check, camera(fields[key], context), atol=1e-12, rtol=0):
            raise ValueError("The separate camera basis changed the predicted diffraction")
    row, col = np.indices(support.shape)
    folds = []
    for name, axis in (("rows", row), ("columns", col)):
        cut = np.median(axis[support])
        buffer = np.ceil(2*context["probe_pixels"])
        partitions = (support & (axis < cut-buffer), support & (axis > cut+buffer))
        for half, (train, test) in enumerate((partitions, partitions[::-1])):
            if min(train.sum(), test.sum()) < 300:
                raise ValueError("A half-scan has fewer than 300 supported pixels")
            fitted, quality = fit_camera(data, detector, native_phase, benchmark, context, train)
            scores = {}
            for mode, calibration in (("spatial_calibration", fitted), ("published_calibration", coeff)):
                scores[mode] = {}
                for model, (cos, sin) in predictions.items():
                    camera_signal = cos[test]*calibration[:, 0] + sin[test]*calibration[:, 1] + calibration[:, 2]
                    scores[mode][model] = {key: rms(actual[test][:, ring], camera_signal[:, ring])
                                           for key, ring in rings.items()}
            folds.append({"axis": name, "scored_half": 1-half, "calibration_pixels": int(train.sum()),
                          "score_pixels": int(test.sum()), "calibration_quality": quality,
                          "coefficient_change_rms": rms(fitted, coeff), "scores": scores})
    return {"interface": interface, "folds": folds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    cross, benchmark, settings, templates = (json.loads(p.read_text()) for p in (SPLIT_RING, BENCHMARK, SETTINGS, TEMPLATES))
    for name, expected in {**cross["input_sha256"], **templates["input_sha256"]}.items():
        if digest(PROJECT / name) != expected:
            raise ValueError(f"The detector control input changed in {name}")
    author = load_author_release(args.release, args.dependencies)
    scans = [name for name, row in cross["scans"].items() if "published_zero" in row]
    result = {
        "scope": "Same split-ring first-ring geometry and independently parameterized elastic predictions. Estimate each diffraction channel's affine intensity coefficients from archived phase on one half-scan. Score the other half with an excluded band. Fit the same calibration for every candidate and compare to published calibration on identical score masks.",
        "limits": "The archived phase and first-ring seed still depend on the original scan-wide extraction and detector calibration. The spatial replay measures coefficient stability and sensitivity to the scored half. It is not an independent calibration of the full microscope model.",
        "input_sha256": {p.name: digest(p) for p in (Path(__file__), SPLIT_RING, BENCHMARK, SETTINGS, TEMPLATES)},
        "scans": {}, "failed_scans": {},
    }
    for scan in scans:
        checkpoint_guard()
        data = load_dataset(scan, args.archive, allowed_ids=scans, allow_author_crop=True)
        reference = benchmark["scans"][scan]
        detector = detector_data(args.archive, scan, author)
        _, _, native_phase, fingerprint = read_detector_fit(args.archive, scan, "fit", author)
        if data.archive_md5 != reference["archive_md5"] or fingerprint != detector[4]["fit_sha256"]:
            raise ValueError("The measurement archive changed")
        orientation = reference["interface"][-1]
        try:
            row = score_scan(data, detector, native_phase, reference, cross["scans"][scan]["published_zero"],
                             settings, templates["templates"][orientation])
            result["scans"][scan] = row
            print(scan, [f["coefficient_change_rms"] for f in row["folds"]], flush=True)
        except (ValueError, np.linalg.LinAlgError) as exc:
            result["failed_scans"][scan] = str(exc)
            print(scan, str(exc), flush=True)
        result["wall_seconds"] = perf_counter()-start
        checkpoint_guard()
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
