"""Compare the full and simpler MoS2 models on identical local geometry."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from author_release import load_author_release
from bilayer_relaxation import MODES, MONOLAYERS, STACKING, make_grid, sample_disregistry
from observe_bilayers import read_input_metadata
from run_heterobilayer_sparse_geometry import linear_response, solve_from_geometry
from run_heterobilayer_template_transfer import design_matrix
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_regime_benchmark import SETTINGS, relaxed_ray
from run_mos2_constrained_equilibrium import score_fields
from run_mos2_cross_ring import TEMPLATES
from run_mos2_first_ring_geometry import INTERFACE
from run_mos2_intensity_only_replay import disk_images, geometry
from run_mos2_local_intensity_geometry import geometric_folds
from run_mos2_nonlinear_controls import alternative_inputs, minimize_all
from run_mos2_soliton_approximation import soliton_motion
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


INTENSITY = PROJECT / "mos2_intensity_only_replay_v01.json"
LOCAL = PROJECT / "mos2_local_intensity_geometry_v01.json"
CONSTRAINED = PROJECT / "mos2_constrained_equilibrium_v01.json"


def predict_fields(warp, gradient, basis, templates):
    grid, shift, _ = solve_from_geometry(gradient, basis, INTERFACE)
    linear = linear_response(grid, INTERFACE)
    amplitude = relaxed_ray(grid, linear, INTERFACE)
    modes = 2*np.pi*MODES[:3] @ np.linalg.inv(basis)
    phase_excursion = float(np.sqrt(np.mean((linear@modes.T)**2)))
    coefficients, moduli, _ = alternative_inputs(basis, INTERFACE[-1])
    alternate_grid = make_grid(gradient, basis, len(shift), *moduli)
    alternate, _ = minimize_all(alternate_grid, coefficients)
    fields = {
        "rigid": warp,
        "full": sample_disregistry(grid, shift, warp, grid.cell),
        "alternate_inputs": sample_disregistry(alternate_grid, alternate, warp, grid.cell),
        "energy_scaled_first_order": sample_disregistry(grid, amplitude*linear, warp, grid.cell),
    }
    _, c1, c2, c3, _, _ = STACKING[INTERFACE]
    shear = MONOLAYERS["MoS2"][2]
    twist = (gradient[1, 0]-gradient[0, 1])/2
    for name, coupling in (("soliton_first_star", c1), ("soliton_effective_adhesion", c1-8*c2+9*c3)):
        motion, _ = soliton_motion(warp, basis, twist, coupling/(2000*shear))
        fields[name] = warp+motion
    for name, template in templates.items():
        design = design_matrix(warp, basis, np.asarray(template["modes"]))
        fields[name] = warp+design@np.asarray(template["coefficients_nm"])
    return fields, phase_excursion


def evaluate(images, modes, metadata, settings, recorded, prior, direct, templates):
    context = geometry(images, modes, metadata, settings)
    receipt = context["receipt"]
    if receipt != recorded["geometry"]:
        raise ValueError("The first-ring geometry changed")
    pixel = receipt["pixel_nm"]
    angle = np.deg2rad(float(metadata["SampleRotation"]))
    yy, xx = np.indices(images.shape[:2])
    xy = np.stack((xx, yy), axis=-1)*pixel
    centers = np.asarray(receipt["centers_rc_px"])
    affine = np.asarray(receipt["first_ring_affine_fit"])
    folds = []
    for group, _, warp, fit, mask, _ in geometric_folds(
        centers, xy, pixel, context["basis"], affine, context["support"],
    ):
        checkpoint_guard()
        target = next(r for r in prior["folds"] if r["withheld_group"] == group)
        if not target["diffraction"]:
            continue
        gradient = fit[:2].T @ rotation(-angle)
        fields, excursion = predict_fields(warp, gradient, context["basis"], templates)
        scores = score_fields(fields, context, images, mask)
        comparison = next(r for r in direct if r["withheld_group"] == group)["scores"]
        for score, expected, relaxed in zip(scores, target["diffraction"], comparison, strict=True):
            for key, name in (("full", "local_full"), ("rigid", "local_rigid"),
                              ("energy_scaled_first_order", "local_scaled_first_order")):
                if abs(score["rmse"][key]-expected["rmse"][name]) > 1e-12 or score["pixels"] != expected["score_pixels"]:
                    raise ValueError("A local baseline or score mask changed")
            score["rmse"]["direct_equilibrium"] = relaxed["rmse"]["local_reference"]
        folds.append({"withheld_group": group, "first_order_phase_excursion_rad": excursion, "scores": scores})
    values = [r["rmse"] for fold in folds for r in fold["scores"]]
    return {"folds": folds, "mean_second_ring_rmse": {
        key: float(np.mean([r[key] for r in values])) for key in values[0]
    } if values else {}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    paths = (INTENSITY, LOCAL, CONSTRAINED, SETTINGS, TEMPLATES)
    intensity, local, direct, settings, templates = (json.loads(p.read_text()) for p in paths)
    for name, expected in (intensity["input_sha256"] | local["input_sha256"]).items():
        if digest(PROJECT/name) != expected:
            raise ValueError(f"The original validation input changed: {name}")
    author = load_author_release(args.release, args.dependencies)
    result = {
        "scope": "Compare the fixed continuum model, independent published MoS2 parameters, the published isotropic soliton formula, and the previously frozen empirical pattern on the same local center warp. Refit identical camera parameters on the same training pixels and predict the same test pixels. Include the stored direct equilibrium scores on those folds. These scans have already been examined.",
        "limits": "Moiré geometry and calibration come from the same acquisition as the score. The block-based camera folds are repeated within each map. The original nine supported maps have incomplete specimen identities.",
        "input_sha256": {p.name: digest(p) for p in (Path(__file__),*paths)},
        "scans": {},
    }
    for scan, reference in intensity["scans"].items():
        checkpoint_guard()
        meta = read_input_metadata(scan, args.archive)
        images, modes, fingerprint = disk_images(args.archive, scan, author)
        if meta["archive_md5"] != reference["archive_md5"] or fingerprint != reference["diskset_sha256"]:
            raise ValueError(f"The source intensity archive changed in {scan}")
        score = evaluate(images, modes, meta["metadata_fields"], settings, reference,
                         local["scans"][scan], direct["scans"].get(scan, ()), templates["templates"]["P"])
        result["scans"][scan] = score
        result["wall_seconds"] = perf_counter()-start
        atomic_json(args.output, result)
        print(scan, score["mean_second_ring_rmse"], flush=True)


if __name__ == "__main__":
    main()
