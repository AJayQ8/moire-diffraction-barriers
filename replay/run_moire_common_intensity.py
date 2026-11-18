"""Score rotational and compressional relaxation with the same intensity test in each material."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import ConvexHull

from author_detector_replay import rms
from author_release import load_author_release
from bilayer_relaxation import ConvergenceError, MONOLAYERS, sample_disregistry
from observe_bilayers import _inside_polygon_rc, read_input_metadata
from run_detector_prediction import AUTHOR_BASIS
from run_heterobilayer_first_ring_seed import fit_affine_seed
from run_heterobilayer_sparse_geometry import linear_response, solve_from_geometry
from run_heterobilayer_template_transfer import design_matrix
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_elastic_components import common_rings, split_displacement
from run_moire_regime_benchmark import SETTINGS, relaxed_ray
from run_mos2_constrained_equilibrium import score_fields
from run_mos2_cross_ring import TEMPLATES
from run_mos2_first_ring_geometry import affine_seed, centers_from_seed
from run_mos2_frame_sensitivity import fold_masks
from run_mos2_intensity_only_replay import camera_fit, disk_images, geometry, normalized_first_ring
from run_mos2_local_intensity_geometry import geometric_folds
from run_mos2_camera_spatial_replay import phase_camera
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


BENCHMARK = PROJECT / "moire_regime_benchmark_v02.json"
PARALLEL = PROJECT / "mos2_intensity_only_replay_v01.json"
LOCAL_PARALLEL = PROJECT / "mos2_local_intensity_geometry_v01.json"
MIXED_TEMPLATES = PROJECT / "heterobilayer_template_fit_v01.json"


def centers_and_context(values, modes, meta, settings, interface):
    if interface == ("MoS2", "MoS2", "P"):
        return geometry(values, modes, meta, settings)
    pixel, a = (float(meta[k]) for k in ("PixelSize", "LatticeConstant"))
    angle = np.deg2rad(float(meta["SampleRotation"]))
    basis = a*np.array(((1.0, 0.5), (0, np.sqrt(3)/2)))
    rings = common_rings(modes)
    first = rings["first"]
    normalized, coefficient = normalized_first_ring(values[..., first])
    if interface[0] == interface[1]:
        seed, frame, _ = affine_seed(normalized, coefficient, modes[first], pixel, a, angle, basis)
    else:
        chosen, seed, _ = fit_affine_seed(normalized, modes[first], coefficient, pixel, a, angle)
        mirror = np.diag((-1, 1)) if chosen["vector_reflection"] else np.eye(2)
        frame = rotation(chosen["vector_rotation_60deg_steps"]*np.pi/3) @ mirror
    q = modes @ np.linalg.inv(AUTHOR_BASIS) @ frame / a
    fit, centers, _ = centers_from_seed(normalized, coefficient, q[first], basis, seed, pixel)
    yy, xx = np.indices(values.shape[:2])
    points = np.stack((xx, yy), axis=-1)*pixel
    support = _inside_polygon_rc(yy.shape, centers[ConvexHull(centers).vertices])
    distance2 = np.full(yy.shape, np.inf)
    for r, c in centers:
        distance2 = np.minimum(distance2, (yy-r)**2+(xx-c)**2)
    support &= distance2 > (2*settings["smoothing_sigma_pixels"])**2
    support[:4] = support[-4:] = support[:, :4] = support[:, -4:] = False
    return {"basis": basis, "rigid": points@fit[:2]+fit[-1],
            "gradient": fit[:2].T@rotation(-angle), "reciprocal": q, "support": support,
            "probe": settings["probe_fwhm_nm"]/(pixel*np.sqrt(8*np.log(2))), "rings": rings,
            "receipt": {"first_ring_affine_fit": fit.tolist(), "centers_rc_px": centers.tolist(), "pixel_nm": pixel}}


def model_fields(warp, gradient, basis, interface, templates):
    scale = MONOLAYERS["MoS2"][0]/np.linalg.norm(basis[:,0]) if interface[:2] == ("MoS2", "WSe2") else 1.0
    origin = scale*basis@np.full(2, 2/3 if interface[-1] == "AP" else 0)
    grid, shift, _ = solve_from_geometry(scale*gradient, scale*basis, interface)
    first = linear_response(grid, interface)
    amplitude = relaxed_ray(grid, first, interface)
    components, power = split_displacement(grid, shift)
    phase = origin+scale*warp
    fields = {key: (sample_disregistry(grid, motion, phase, grid.cell)-origin)/scale
              for key,motion in (components | {"energy_scaled_first_order": amplitude*first}).items()}
    for key, template in templates.items():
        design = design_matrix(warp, basis, np.asarray(template["modes"]))
        fields[key] = warp+design@np.asarray(template["coefficients_nm"])
    return fields, power


def calibrate_and_score(fields, context, values, mask):
    second = context["rings"]["second"]
    observations = values[..., second]
    result = []
    signals = {key: phase_camera(v, context["reciprocal"][second], context["probe"]) for key,v in fields.items()}
    for train, test in fold_masks(mask, context["probe"]):
        if min(train.sum(), test.sum()) < 300:
            continue
        scores = {}
        for key,(cos,sin) in signals.items():
            predicted, actual, _ = camera_fit(cos, sin, observations, train, test, allow_odd=True)
            scores[key] = rms(predicted, actual)
        result.append(scores)
    return result


def averages(rows):
    return {key: float(np.mean([r[key] for r in rows])) for key in rows[0]} if rows else {}


def evaluate(images, modes, meta, settings, interface, template):
    context = centers_and_context(images, modes, meta, settings, interface)
    receipt = context["receipt"]
    pixel = receipt["pixel_nm"]
    fit = np.asarray(receipt["first_ring_affine_fit"])
    centers = np.asarray(receipt["centers_rc_px"])
    angle = np.deg2rad(float(meta["SampleRotation"]))
    yy, xx = np.indices(images.shape[:2])
    points = np.stack((xx, yy), axis=-1)*pixel
    basis = context["basis"]
    affine, power = model_fields(context["rigid"], context["gradient"], basis, interface, template)
    affine_scores = calibrate_and_score(affine, context, images, context["support"])
    local = []
    for group, _, warp, fit_local, mask, _ in geometric_folds(
        centers, points, pixel, basis, fit, context["support"],
    ):
        checkpoint_guard()
        gradient = fit_local[:2].T@rotation(-angle)
        models, fraction = model_fields(warp, gradient, basis, interface, template)
        scores = score_fields(models, context, images, mask)
        if scores:
            local.append({"center_group": group, "compression_displacement_power": fraction, "scores": scores})
    return {
        "geometry": receipt, "affine_compression_displacement_power": power,
        "affine_scores": affine_scores, "mean_affine": averages(affine_scores),
        "local_scores": local, "mean_local": averages([r["rmse"] for fold in local for r in fold["scores"]]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    start = perf_counter()
    paths = (BENCHMARK, PARALLEL, LOCAL_PARALLEL, SETTINGS, TEMPLATES, MIXED_TEMPLATES)
    benchmark, parallel, local, settings, templates, mixed = (json.loads(p.read_text()) for p in paths)
    author = load_author_release(args.release, args.dependencies)
    cases = ("DS2", "DS25", "DS31", "DS34") if args.pilot else tuple(benchmark["scans"])
    result = {"scope": "Use only the integrated diffraction images and instrument metadata to find geometry in the first ring. Score the second ring with equal per-model camera calibration on buffered raster halves. Compare fixed full, its longitudinal and transverse components, energy-scaled first order, rigid and frozen templates on both affine and local geometry. All maps have been seen previously.",
              "limits": "A pure first-ring cosine gives a phase/index convention using metadata and small twist or known positive mismatch. The archive has incomplete specimen IDs. The physical model comparison is conditional on within-scan geometry and camera calibration.",
              "input_sha256": {p.name: digest(p) for p in (Path(__file__), *paths)}, "scans": {}, "failed_scans": {}}
    for scan in cases:
        checkpoint_guard()
        record = benchmark["scans"][scan]
        interface = tuple(record["interface"])
        meta = read_input_metadata(scan, args.archive)
        values, modes, fingerprint = disk_images(args.archive, scan, author)
        if meta["archive_md5"] != record["archive_md5"]:
            raise ValueError("One source archive changed")
        template = templates["templates"][interface[-1]] if interface[0] == interface[1] else mixed["templates"]
        try:
            row = evaluate(values, modes, meta["metadata_fields"], settings, interface, template)
        except (ValueError, ConvergenceError, np.linalg.LinAlgError) as exc:
            result["failed_scans"][scan] = str(exc)
            print(scan, exc, flush=True)
        else:
            if scan in parallel["scans"]:
                expected = parallel["scans"][scan]["average_fold_rmse"]["signed"]["measured_direction"]
                if abs(row["mean_affine"]["full"]-expected) > 1e-12:
                    raise ValueError("The parallel intensity-only result did not reproduce")
                earlier = local["scans"][scan]["mean_second_ring_rmse"]
                if earlier and abs(row["mean_local"]["full"]-earlier["local_full"]) > 1e-12:
                    raise ValueError("The parallel local result did not reproduce")
            result["scans"][scan] = {"interface": list(interface), "diskset_sha256": fingerprint, **row}
            print(scan, row["mean_affine"], row["mean_local"], flush=True)
        result["wall_seconds"] = perf_counter()-start
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
