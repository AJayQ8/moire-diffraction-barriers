"""Test lattice indexing ambiguity with fixed first-ring geometry and score pixels."""

import argparse
import json
from pathlib import Path

import numpy as np

from author_detector_replay import rms
from author_release import load_author_release
from bilayer_relaxation import sample_disregistry
from observe_bilayers import read_input_metadata
from run_heterobilayer_sparse_geometry import solve_from_geometry
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_regime_benchmark import SETTINGS
from run_mos2_camera_spatial_replay import phase_camera
from run_mos2_directional_split_ring import stretch_controls
from run_mos2_first_ring_geometry import INTERFACE
from run_mos2_intensity_only_replay import camera_fit, disk_images, geometry
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


SOURCE = PROJECT / "mos2_intensity_only_replay_v01.json"
DEVELOPMENT = PROJECT / "mos2_intensity_only_development_v01.json"
ARCHIVED = PROJECT / "moire_regime_benchmark_v02.json"


def frame_options(gradient):
    """The leading diffraction cosine leaves 60° indexing choices."""
    for turns in range(6):
        rot = rotation(turns*np.pi/3)
        g = rot @ gradient
        twist = (g[1, 0]-g[0, 1])/2
        dilation = np.trace(g)/2
        admissible = -0.070 < twist < -0.003 and abs(dilation) < abs(twist)
        yield turns, rot, g, {"twist_deg": float(np.rad2deg(twist)),
                              "dilation_percent": float(100*dilation),
                              "passes_original_twist_dilation_bounds": bool(admissible)}


def fold_masks(support, probe):
    gap = np.ceil(2*probe)
    for position in np.indices(support.shape):
        cut = np.median(position[support])
        before = support & (position < cut-gap)
        after = support & (position > cut+gap)
        yield before, after
        yield after, before


def compare(context, observations, rotation_matrix, gradient, folds):
    basis, rigid, reciprocal = (context[name] for name in ("basis", "rigid", "reciprocal"))
    rigid = rigid @ rotation_matrix.T
    reciprocal = reciprocal @ rotation_matrix.T
    first, second = (context["rings"][name] for name in ("first", "second"))
    original = phase_camera(context["rigid"], context["reciprocal"], context["probe"])
    reindexed = phase_camera(rigid, reciprocal, context["probe"])
    if not all(np.allclose(a, b, atol=1e-12, rtol=0) for a, b in zip(original, reindexed)):
        raise ValueError("Reindexing changed the rigid diffraction prediction")
    scores = {}
    for name, g in stretch_controls(gradient):
        checkpoint_guard()
        grid, shift, _ = solve_from_geometry(g, basis, INTERFACE)
        field = sample_disregistry(grid, shift, rigid, np.linalg.solve(gradient, basis))
        cos, sin = phase_camera(field, reciprocal, context["probe"])
        scores[name] = {"first": [], "second": []}
        for train, test in folds:
            for key, ring in (("first", first), ("second", second)):
                prediction, actual, _ = camera_fit(cos[..., ring], sin[..., ring],
                                                   observations[..., ring], train, test, allow_odd=True)
                scores[name][key].append(rms(actual, prediction))
    cos, sin = reindexed
    scores["rigid"] = {}
    for key, ring in (("first", first), ("second", second)):
        values = [camera_fit(cos[..., ring], sin[..., ring], observations[..., ring],
                             train, test, allow_odd=True)[:2] for train, test in folds]
        scores["rigid"][key] = [rms(actual, prediction) for prediction, actual in values]
    return {name: {ring: {"fold_rmse": values, "mean_rmse": float(np.mean(values))}
                   for ring, values in rings.items()} for name, rings in scores.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    files = (SOURCE, DEVELOPMENT, ARCHIVED, SETTINGS)
    replay, development, archived, settings = (json.loads(f.read_text()) for f in files)
    author = load_author_release(args.release, args.dependencies)
    result = {
        "scope": "Reindex the same first-ring registry and reciprocal rows by 60 degrees while holding the calibrated scan coordinates fixed. Rigid images are exactly invariant. Changing the relative orientation of displacement and spatial coordinates changes the mechanics. Both choices use the same pixels, camera calibration and physical constants. This is a sensitivity check on previously examined maps.",
        "limits": "The small-twist convention and metadata define the admissible alternatives. The results do not label individual layers or determine absolute dilation. Camera parameters are learned on one half and scored on the other, and the first ring supplied the moire geometry.",
        "input_sha256": {p.name: digest(p) for p in (Path(__file__), *files)},
        "scans": {},
    }
    for scan, record in (replay["scans"] | development["scans"]).items():
        checkpoint_guard()
        meta = read_input_metadata(scan, args.archive)
        images, modes, fingerprint = disk_images(args.archive, scan, author)
        if meta["archive_md5"] != record["archive_md5"] or fingerprint != record["diskset_sha256"]:
            raise ValueError(f"{scan}: source data changed")
        context = geometry(images, modes, meta["metadata_fields"], settings)
        fit = context["receipt"]["first_ring_affine_fit"]
        if not np.allclose(fit, record["geometry"]["first_ring_affine_fit"], atol=1e-10, rtol=0):
            raise ValueError(f"{scan}: the first-ring geometry changed")
        g_archive = np.asarray(archived["scans"][scan]["affine_geometry"]["gradient_crystal"])
        archive_rotation = g_archive @ np.linalg.inv(context["gradient"])
        folds = list(fold_masks(context["support"], context["probe"]))
        options = {}
        for turns, rot, gradient, candidate in frame_options(context["gradient"]):
            candidate["relative_difference_from_archived_gradient"] = float(
                np.linalg.norm(gradient-g_archive)/np.linalg.norm(g_archive))
            if turns == 0 or candidate["passes_original_twist_dilation_bounds"]:
                candidate["scores"] = compare(context, images, rot, gradient, folds)
            options[str(turns)] = candidate
        original = options["0"]["scores"]["measured_direction"]["second"]["mean_rmse"]
        if abs(original-record["average_fold_rmse"]["signed"]["measured_direction"]) > 1e-10:
            raise ValueError(f"{scan}: the original second-ring result did not reproduce")
        result["scans"][scan] = {
            "archive_md5": meta["archive_md5"], "sample_rotation_degrees": float(meta["metadata_fields"]["SampleRotation"]),
            "archive_gradient_crystal": g_archive.tolist(),
            "archived_over_first_ring_linear_map": archive_rotation.tolist(),
            "candidate_60_degree_steps": options,
        }
        atomic_json(args.output, result)
        print(scan, [(k, v["dilation_percent"], v.get("scores", {}).get("measured_direction", {}).get("second", {}).get("mean_rmse"))
                     for k, v in options.items() if "scores" in v], flush=True)


if __name__ == "__main__":
    main()
