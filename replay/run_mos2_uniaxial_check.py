"""Test the anchor's uniaxial geometry on the two ambiguous parallel MoS2 maps."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import polar

from author_release import load_author_release
from bilayer_relaxation import sample_disregistry
from observe_bilayers import read_input_metadata
from run_heterobilayer_sparse_geometry import solve_from_geometry
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_regime_benchmark import SETTINGS
from run_mos2_constrained_equilibrium import score_fields
from run_mos2_first_ring_geometry import INTERFACE
from run_mos2_frame_sensitivity import SOURCE
from run_mos2_intensity_only_replay import disk_images, geometry
from run_mos2_local_baselines import predict_fields
from run_mos2_local_intensity_geometry import geometric_folds
from run_mos2_cross_ring import TEMPLATES
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


SCANS = ("DS13", "DS14", "DS2", "DS10")
FRAME = PROJECT / "mos2_frame_sensitivity_v01.json"
LOCAL = PROJECT / "mos2_local_baselines_v01.json"
POISSON = 0.234


def decompose(gradient):
    """Project onto the stated Poisson relation while preserving stretch axes and twist."""
    angle, stretch = polar(np.eye(2)+gradient)
    eigenvalues, axes = np.linalg.eigh(stretch-np.eye(2))
    candidates = []
    for order in ((0,1), (1,0)):
        template = np.array((1., -POISSON))
        magnitude = float(eigenvalues[list(order)] @ template/(template@template))
        principal = np.empty(2)
        principal[list(order)] = template*magnitude
        modeled = angle @ (np.eye(2)+(axes*principal)@axes.T) - np.eye(2)
        residual = float(np.linalg.norm(gradient-modeled))
        candidates.append((residual, modeled, magnitude))
    residual, uniaxial, magnitude = min(candidates, key=lambda x: x[0])
    isotropic = angle*np.sqrt(np.linalg.det(stretch))-np.eye(2)
    no_mean = angle @ (stretch-(np.trace(stretch)/2-1)*np.eye(2))-np.eye(2)
    return {
        "twist_degrees": float(np.rad2deg(np.arctan2(angle[1,0], angle[0,0]))),
        "principal_strains_percent": (100*eigenvalues).tolist(),
        "uniaxial_magnitude_percent": 100*magnitude,
        "uniaxial_fit_residual_percent": 100*residual,
    }, {"uniaxial": uniaxial, "isotropic": isotropic, "no_mean_dilation": no_mean}


def score_branches(images, context, gradient, warp, mask, templates, branches):
    results = {}
    for turn in branches:
        checkpoint_guard()
        turn_matrix = rotation(turn*np.pi/3)
        branch_gradient = turn_matrix@gradient
        branch_warp = warp@turn_matrix.T
        branch_context = context | {"reciprocal": context["reciprocal"]@turn_matrix.T}
        basis = context["basis"]
        observed_cell = np.linalg.solve(branch_gradient, basis)
        diagnostic, controls = decompose(branch_gradient)
        fields, excursion = predict_fields(branch_warp, branch_gradient, basis, templates)
        for key, g in controls.items():
            grid, shift, _ = solve_from_geometry(g, basis, INTERFACE)
            fields[key] = sample_disregistry(grid, shift, branch_warp, observed_cell)
        scores = score_fields(fields, branch_context, images, mask)
        results[str(turn)] = {**diagnostic, "first_order_phase_excursion_rad": excursion, "fold_scores": scores}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = (SOURCE, FRAME, LOCAL, SETTINGS, TEMPLATES)
    source, frame, local, settings, templates = (json.loads(p.read_text()) for p in paths)
    author = load_author_release(args.release, args.dependencies)
    result = {"scope": "Retrospective mechanism check. On each previously scored local geometry fold, hold registry and camera pixels fixed while changing the imposed strain in the equilibrium calculation. Project the measured stretch onto the uniaxial relation in the anchor, remove mean dilation, or remove directional stretch. Compare both admissible lattice indexings on DS13 and DS14 and retain two controls with a single indexing.",
              "limits": "The uniaxial relation is a modeling assumption for the reference gradient. It is not an independently imposed loading condition. No branch or model is selected using these scores.",
              "poisson_ratio": POISSON, "input_sha256": {p.name: digest(p) for p in (Path(__file__), *paths)}, "scans": {}}
    for scan in SCANS:
        checkpoint_guard()
        metadata = read_input_metadata(scan, args.archive)
        images, modes, fingerprint = disk_images(args.archive, scan, author)
        recorded = source["scans"][scan]
        if fingerprint != recorded["diskset_sha256"] or metadata["archive_md5"] != recorded["archive_md5"]:
            raise ValueError("The input intensities changed")
        meta = metadata["metadata_fields"]
        context = geometry(images, modes, meta, settings)
        receipt = context["receipt"]
        if receipt != recorded["geometry"]:
            raise ValueError("The reference lattice changed")
        branches = [int(k) for k, v in frame["scans"][scan]["candidate_60_degree_steps"].items() if "scores" in v]
        pixel = receipt["pixel_nm"]
        yy, xx = np.indices(images.shape[:2])
        points = np.stack((xx, yy), axis=-1)*pixel
        angle = np.deg2rad(float(meta["SampleRotation"]))
        reference = np.asarray(receipt["first_ring_affine_fit"])
        centers = np.asarray(receipt["centers_rc_px"])
        comparisons = []
        for group, _, warp, fit, mask, _ in geometric_folds(
            centers, points, pixel, context["basis"], reference, context["support"],
        ):
            expected = next((v for v in local["scans"][scan]["folds"] if v["withheld_group"] == group), None)
            if expected is None:
                continue
            values = score_branches(images, context, fit[:2].T@rotation(-angle), warp, mask,
                                    templates["templates"]["P"], branches)
            for measured, old in zip(values["0"]["fold_scores"], expected["scores"], strict=True):
                if abs(measured["rmse"]["full"]-old["rmse"]["full"]) > 1e-12 or measured["pixels"] != old["pixels"]:
                    raise ValueError("The local reference score changed")
            comparisons.append({"withheld_group": group, "branches": values})
        result["scans"][scan] = comparisons
        atomic_json(args.output, result)
        print(scan, len(comparisons), flush=True)


if __name__ == "__main__":
    main()
