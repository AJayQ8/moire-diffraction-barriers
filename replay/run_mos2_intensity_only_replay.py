"""Predict parallel MoS2 diffraction using the measured disk intensities alone."""

import argparse
import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from scipy.spatial import ConvexHull

from author_detector_replay import rms
from author_release import load_author_release
from bilayer_relaxation import MONOLAYERS, STACKING, sample_disregistry
from observe_bilayers import _inside_polygon_rc, read_input_metadata
from run_detector_prediction import AUTHOR_BASIS, DiskReadback
from run_heterobilayer_sparse_geometry import linear_response, solve_from_geometry
from run_heterobilayer_template_transfer import design_matrix
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_elastic_components import common_rings
from run_moire_regime_benchmark import SETTINGS, relaxed_ray
from run_mos2_camera_spatial_replay import phase_camera
from run_mos2_directional_split_ring import stretch_controls
from run_mos2_first_ring_geometry import INTERFACE, affine_seed, centers_from_seed
from run_mos2_soliton_approximation import soliton_motion
from run_mos2_template_transfer import PARALLEL
from run_mos2_cross_ring import TEMPLATES
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


def disk_images(archive, scan, author):
    """Open the integrated intensity member without the author phase fit."""
    with ZipFile(archive/f"{scan}.zip") as source:
        member = source.getinfo(f"{scan}/diskset.pkl")
        if member.file_size > 128*1024**2:
            raise ValueError("Unexpected integrated disk archive size")
        payload = source.read(member)
    saved = DiskReadback(BytesIO(payload)).load()
    disk = object.__new__(author.diskset.DiskSet)
    disk.__dict__.update(vars(saved))
    modes = disk.clean_normgset(sanity_plot=False)
    data = np.moveaxis(disk.df_set(), 0, -1)
    if data.shape[-1] != len(modes) or not np.isfinite(data).all():
        raise ValueError("The archive lacks a consistent set of integrated disks")
    return data, modes, sha256(payload).hexdigest()


def normalized_first_ring(values):
    """Use the leading cosine contrast to locate first-ring stacking centers."""
    dims = values.shape[-1]
    q05, q95 = np.quantile(values.reshape(-1, dims), (0.05, 0.95), axis=0)
    if np.any(q95 <= q05):
        raise ValueError("One first-ring channel has no resolved intensity variation")
    normalized = (values-q05)/(q95-q05)
    return normalized, np.column_stack((np.ones(dims), np.zeros((dims, 2))))


def geometry(images, modes, metadata, settings):
    pixel = float(metadata["PixelSize"])
    a = float(metadata["LatticeConstant"])
    basis = a*np.array(((1.0, 0.5), (0, np.sqrt(3)/2)))
    angle = np.deg2rad(float(metadata["SampleRotation"]))
    rings = common_rings(modes)
    first = rings["first"]
    values, coefficients = normalized_first_ring(images[..., first])
    seed, frame, diagnostic = affine_seed(values, coefficients, modes[first], pixel, a, angle, basis)
    q = modes @ np.linalg.inv(AUTHOR_BASIS) @ frame / a
    fit, centers, local = centers_from_seed(values, coefficients, q[first], basis, seed, pixel)
    height, width = images.shape[:2]
    yy, xx = np.indices((height, width), dtype=float)
    xy = np.stack((xx, yy), axis=-1)*pixel
    rigid = xy@fit[:2]+fit[-1]
    support = _inside_polygon_rc((height, width), centers[ConvexHull(centers).vertices])
    distance2 = np.full((height, width), np.inf)
    for y, x in centers:
        distance2 = np.minimum(distance2, (yy-y)**2+(xx-x)**2)
    support &= distance2 > (2*settings["smoothing_sigma_pixels"])**2
    support[:4] = support[-4:] = support[:, :4] = support[:, -4:] = False
    if support.sum() < 600:
        raise ValueError("The first-ring geometry leaves too few calibration and validation pixels")
    return {"basis": basis, "rigid": rigid, "gradient": fit[:2].T@rotation(-angle), "reciprocal": q,
            "support": support, "probe": settings["probe_fwhm_nm"]/(pixel*np.sqrt(8*np.log(2))),
            "rings": rings, "receipt": {**diagnostic, **local, "centers_rc_px": centers.tolist(),
                                        "first_ring_affine_fit": fit.tolist(), "pixel_nm": pixel}}


def predictions(context, templates):
    gradient, basis, rigid = (context[k] for k in ("gradient", "basis", "rigid"))
    cell = np.linalg.solve(gradient, basis)
    fields = {"rigid": rigid}
    for name, test in stretch_controls(gradient):
        checkpoint_guard()
        grid, shift, _ = solve_from_geometry(test, basis, INTERFACE)
        fields[name] = sample_disregistry(grid, shift, rigid, cell)
        if name == "measured_direction":
            linear = linear_response(grid, INTERFACE)
            amplitude = relaxed_ray(grid, linear, INTERFACE)
            fields["energy_scaled_first_order"] = sample_disregistry(grid, amplitude*linear, rigid, cell)
    _, c1, c2, c3, _, _ = STACKING[INTERFACE]
    modulus = MONOLAYERS["MoS2"][2]
    twist = (gradient[1, 0]-gradient[0, 1])/2
    for name, coupling in (("soliton_first_star", c1), ("soliton_effective_adhesion", c1-8*c2+9*c3)):
        field, _ = soliton_motion(rigid, basis, twist, coupling/(2000*modulus))
        fields[name] = rigid+field
    for name, template in templates.items():
        design = design_matrix(rigid, basis, np.asarray(template["modes"]))
        fields[name] = rigid+design@np.asarray(template["coefficients_nm"])
    return {name: phase_camera(field, context["reciprocal"], context["probe"]) for name, field in fields.items()}


def camera_fit(cos, sin, observed, train, test, allow_odd):
    """Calibrate each candidate on the same training mask and score held pixels."""
    span = np.ptp(observed[train], axis=0)
    if np.min(span) <= 0:
        raise ValueError("A calibration region has no diffraction contrast")
    normalized = observed/span
    values = []
    quality = []
    for channel in range(observed.shape[-1]):
        predictors = [cos[..., channel], np.ones_like(cos[..., channel])]
        if allow_odd:
            predictors.append(sin[..., channel])
        matrix = np.stack(predictors, axis=-1)
        actual = normalized[train, channel]
        estimate, _, rank, singular = np.linalg.lstsq(matrix[train], actual, rcond=None)
        if rank != matrix.shape[-1]:
            raise ValueError("A detector coefficient is unidentified on the calibration pixels")
        values.append(matrix[test]@estimate)
        quality.append({"condition": float(singular[0]/singular[-1]),
                        "coefficients_in_training_range_units": estimate.tolist(),
                        "training_rmse": rms(actual, matrix[train]@estimate)})
    return np.column_stack(values), normalized[test], quality


def evaluate(images, modes, metadata, settings, templates):
    context = geometry(images, modes, metadata, settings)
    fields = predictions(context, templates)
    second = context["rings"]["second"]
    observed = images[..., second]
    support = context["support"]
    rows, columns = np.indices(support.shape)
    folds = []
    for label, position in (("rows", rows), ("columns", columns)):
        cut = np.median(position[support])
        buffer = np.ceil(2*context["probe"])
        halves = (support & (position < cut-buffer), support & (position > cut+buffer))
        if min(half.sum() for half in halves) < 300:
            raise ValueError("A scored half has too little diffraction support")
        for training, (train, test) in enumerate((halves, halves[::-1])):
            checkpoint_guard()
            scores, diagnostics = {}, {}
            for mode in ("even", "signed"):
                scores[mode], diagnostics[mode] = {}, {}
                for name, (cos, sin) in fields.items():
                    predicted, actual, quality = camera_fit(cos[..., second], sin[..., second], observed, train, test,
                                                            allow_odd=mode == "signed")
                    scores[mode][name] = rms(actual, predicted)
                    diagnostics[mode][name] = quality
            folds.append({"axis": label, "calibration_half": training, "calibration_pixels": int(train.sum()),
                          "score_pixels": int(test.sum()), "normalized_second_ring_rmse": scores,
                          "camera_fits": diagnostics})
    means = {mode: {name: float(np.mean([row["normalized_second_ring_rmse"][mode][name] for row in folds]))
                    for name in fields} for mode in ("even", "signed")}
    return {"geometry": context["receipt"], "diffraction_ring_counts": {k: int(v.sum()) for k,v in context["rings"].items()},
            "folds": folds, "average_fold_rmse": means}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch", choices=("development", "held_out"), required=True)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    paths = (SETTINGS, PARALLEL, TEMPLATES, Path(__file__),
             *(PROJECT/n for n in ("run_mos2_first_ring_geometry.py", "run_mos2_camera_spatial_replay.py",
                                     "run_mos2_directional_split_ring.py", "bilayer_relaxation.py")))
    checks = {p.name: digest(p) for p in paths}
    settings, protocol, empirical = (json.loads(p.read_text()) for p in (SETTINGS, PARALLEL, TEMPLATES))
    scans = protocol["batches"][args.batch]
    if args.batch == "held_out":
        if args.freeze is None:
            parser.error("Supply the development freeze for the replay")
        frozen = json.loads(args.freeze.read_text())
        source = args.freeze.parent/frozen["development_file"]
        if frozen["input_sha256"] != checks or frozen["replay_scans"] != scans or frozen["development_sha256"] != digest(source):
            raise ValueError("The calibration method changed after development")
    author = load_author_release(args.release, args.dependencies)
    result = {
        "scope": "Read the integrated diskset and instrument metadata. Get the geometry from first-ring fringes using the ideal parallel-homobilayer cosine contrast. Estimate camera gain, offset and optional sine term separately for every candidate and channel on half of the second-ring pixels. Carry these coefficients to the other half. All candidates share the same geometry, mask, probe and number of calibration parameters. The frozen empirical template uses its original development data.",
        "limits": "The integrated intensities and the sample rotation use the archive's selected detector disks. The geometry and scored pixels share an acquisition. These maps have been compared previously.",
        "batch": args.batch, "input_sha256": checks, "scans": {}, "failed_scans": {},
    }
    if args.freeze:
        result["freeze_sha256"] = digest(args.freeze)
    for scan in scans:
        checkpoint_guard()
        metadata = read_input_metadata(scan, args.archive)
        values, modes, fingerprint = disk_images(args.archive, scan, author)
        try:
            row = evaluate(values, modes, metadata["metadata_fields"], settings, empirical["templates"]["P"])
        except (ValueError, np.linalg.LinAlgError) as exc:
            result["failed_scans"][scan] = str(exc)
            print(scan, str(exc), flush=True)
        else:
            result["scans"][scan] = {"diskset_sha256": fingerprint, "archive_md5": metadata["archive_md5"], **row}
            print(scan, row["average_fold_rmse"], flush=True)
        result["wall_seconds"] = perf_counter()-start
        checkpoint_guard()
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
