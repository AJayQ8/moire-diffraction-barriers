"""Score the published marginal-twist triangular network on parallel MoS2."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull

from author_detector_replay import detector_intensity, rms
from author_release import load_author_release
from bilayer_relaxation import MONOLAYERS, STACKING
from observe_bilayers import _inside_polygon_rc, load_dataset
from run_detector_prediction import AUTHOR_BASIS
from run_detector_wall_profiles import to_scan_frame
from run_moire_elastic_components import common_rings, check_model_score
from run_moire_regime_benchmark import SETTINGS, detector_data
from run_mos2_cross_ring import locate_centers
from run_mos2_template_transfer import BENCHMARK, PARALLEL, make_context
from run_heterobilayer_transfer import PROJECT, digest
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


SPLIT_RING = PROJECT / "mos2_cross_ring_v01.json"
SOURCE = "https://doi.org/10.1103/vm93-prv6"


def soliton_motion(background, basis, twist, effective):
    """Evaluate Eqs. 5 and 8 of Pallewela et al. with the archive's twist sign."""
    lattice = np.linalg.norm(basis[:, 0])
    # Each reciprocal triplet sums to zero. In the paper's coordinates
    # g = M^T b with M = theta times a ninety-degree rotation. It follows
    # that z cross ghat is sign(theta) times bhat for an isotropic cell.
    modes = np.array(((1, 0), (0, 1), (-1, -1)))
    reciprocal = modes @ np.linalg.inv(basis)
    polarization = reciprocal / np.linalg.norm(reciprocal, axis=1)[:, None]
    theta = abs(twist)
    if theta < 1e-5 or effective <= 0:
        raise ValueError("The triangular approximation needs positive adhesion and nonzero twist")
    damping = 2*theta/(3*np.sqrt(effective))
    nmax = max(1, int(np.ceil(15/damping)))
    n = np.arange(1, nmax+1)
    weights = (2/3) / (n * np.cosh(damping*n)**2)
    phase = 2*np.pi * (background @ reciprocal.T)
    # The imposed registry expands in low-energy domains for the positive
    # V1 convention. For small relaxation the correction is +sin(b dot u)
    # along b, independently of how the signed twist indexes physical k.
    result = np.zeros_like(background)
    for index, weight in zip(n, weights):
        result += weight * (np.sin(index*phase) @ polarization)
    return (np.sqrt(3)*lattice/(2*np.pi))*result, nmax


def evaluate(background, support, target, q, coeff, rings, probe, basis, gradient):
    _, first, second, third, _, _ = STACKING[("MoS2", "MoS2", "P")]
    mu = MONOLAYERS["MoS2"][2]
    # Their Fourier coefficients V_i are half of the cosine coefficients.
    coupling = {"first_star": first/(2000*mu), "effective_three_stars": (first-8*second+9*third)/(2000*mu)}
    twist = (gradient[1, 0] - gradient[0, 1]) / 2
    result = {}
    for label, c in coupling.items():
        shift, modes = soliton_motion(background, basis, twist, c)
        field = background + shift
        camera = gaussian_filter(detector_intensity(field, q, coeff), (probe, probe, 0), mode="nearest")
        result[label] = {
            "harmonics_per_axis": modes, "c_V_over_mu": c, "eta": c/twist**2,
            "relative_displacement_rms_nm": float(np.sqrt(np.mean(np.sum(shift[support]**2, axis=1)))),
            "diffraction_rmse": {key: rms(target[support][:, mask], camera[support][:, mask])
                                 for key, mask in rings.items()},
        }
    return result


def split_ring_context(data, detector, benchmark, reference):
    measured, _, modes, coeff, _ = detector
    frame = benchmark["detector_alignment"]
    mirror = np.diag((-1, 1)) if frame["vector_reflection"] else np.eye(2)
    transform = np.linalg.inv(AUTHOR_BASIS) @ rotation(frame["vector_rotation_60deg_steps"]*np.pi/3) @ mirror
    q = modes @ transform / data.lattice_nm
    row, col = data.metadata.get("author_unwrap_crop_origin_row_col", (0, 0))
    shape = data.u_nm.shape[:2]
    measured = to_scan_frame(measured[row:row+shape[0], col:col+shape[1]], frame["spatial_turns"], frame["spatial_reflection"])
    rings = common_rings(modes)
    basis = data.lattice_nm*np.array(((1.0, 0.5), (0.0, np.sqrt(3)/2)))
    initial = np.asarray(reference["archived_affine_seed"])
    first = rings["first"]
    centers, _, _ = locate_centers(data, initial, basis, measured[..., first], q[first], coeff[first], np.zeros(2))
    yy, xx = np.indices(shape, dtype=float)
    points = np.stack((xx, yy), axis=-1)*data.pixel_nm
    fit = np.asarray(reference["first_ring_affine"])
    background = points @ fit[:2] + fit[-1]
    support = _inside_polygon_rc(shape, centers[ConvexHull(centers).vertices])
    distance2 = np.full(shape, np.inf)
    for y, x in centers:
        distance2 = np.minimum(distance2, (yy-y)**2 + (xx-x)**2)
    support &= distance2 > 16
    support[:4] = support[-4:] = support[:, :4] = support[:, -4:] = False
    if support.sum() != reference["score_pixels"]:
        raise ValueError("The split-ring support changed")
    gradient = fit[:2].T @ rotation(-np.deg2rad(data.sample_rotation_degrees))
    probe = 1.25/(np.sqrt(8*np.log(2))*data.pixel_nm)
    rigid = gaussian_filter(detector_intensity(background, q, coeff), (probe, probe, 0), mode="nearest")
    check_model_score({k: rms(measured[support][:, ring], rigid[support][:, ring]) for k, ring in rings.items()},
                      reference["scores"]["rigid"])
    return evaluate(background, support, measured, q, coeff, rings, probe, basis, gradient)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    settings = json.loads(SETTINGS.read_text())
    benchmark = json.loads(BENCHMARK.read_text())
    cross = json.loads(SPLIT_RING.read_text())
    for name, expected in {**benchmark["input_sha256"], **cross["input_sha256"]}.items():
        if digest(PROJECT / name) != expected:
            raise ValueError(f"A comparison input changed: {name}")
    author = load_author_release(args.release, args.dependencies)
    batches = json.loads(PARALLEL.read_text())["batches"]
    cases = batches["development"] + batches["held_out"]
    result = {
        "scope": "Transport a published marginal-twist triangular soliton field into the measured geometry of the 14 previously examined parallel MoS2 maps. The isotropic solution uses the affine twist and the published MoS2 material constants. Report both the first-star and the source's three-star effective adhesion. No material coefficients or amplitudes fit these measurements. This tests the approximation outside its asymptotic range as well as within it.",
        "source": SOURCE,
        "input_sha256": {p.name: digest(p) for p in (Path(__file__), SETTINGS, PARALLEL, BENCHMARK, SPLIT_RING)},
        "scans": {},
    }
    for scan in cases:
        checkpoint_guard()
        reference = benchmark["scans"][scan]
        data = load_dataset(scan, args.archive, allowed_ids=cases, allow_author_crop=True)
        if data.archive_md5 != reference["archive_md5"]:
            raise ValueError("The archive changed")
        detector = detector_data(args.archive, scan, author)
        context = make_context(data, detector, reference, settings)
        gradient = np.asarray(reference["affine_geometry"]["gradient_crystal"])
        same = evaluate(context["fields"]["rigid"], context["support"], context["measured"],
                        context["reciprocal"], context["coefficients"], context["rings"],
                        context["probe_pixels"], context["basis"], gradient)
        score = {"same_scan": same, "same_scan_references": {k: v["diffraction_rmse"] for k,v in reference["models"]["warped"].items()}}
        if scan in cross["scans"] and "published_zero" in cross["scans"][scan]:
            other = cross["scans"][scan]["published_zero"]
            score["split_ring"] = split_ring_context(data, detector, reference, other)
            score["split_ring_references"] = other["scores"]
        result["scans"][scan] = score
        print(scan, {k: v["diffraction_rmse"]["both"] for k, v in same.items()}, flush=True)
        result["wall_seconds"] = perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
