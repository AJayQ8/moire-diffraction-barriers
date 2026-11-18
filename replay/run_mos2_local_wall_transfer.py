"""Compare directional shear under the same cross-validated local registry maps."""

import argparse
import json
from pathlib import Path

import numpy as np

from author_release import load_author_release
from bilayer_relaxation import sample_disregistry
from observe_bilayers import load_dataset, read_input_metadata
from run_heterobilayer_sparse_geometry import linear_response, solve_from_geometry
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, averages, calibrate_and_score, centers_and_context
from run_moire_regime_benchmark import SETTINGS, relaxed_ray
from run_mos2_directional_split_ring import stretch_controls
from run_mos2_intensity_only_replay import disk_images
from run_mos2_joint_parameter_check import measure
from run_mos2_local_intensity_geometry import geometric_folds
from run_registry_prediction import rotation
from run_wall_heterostrain import PROTOCOL, SOURCE
from shared_compute import atomic_json, checkpoint_guard


def predictions(backgrounds, gradient, basis, interface):
    cell = np.linalg.solve(gradient, basis)
    origin = basis @ np.full(2, 2/3 if interface[-1] == "AP" else 0.)
    full_grid, full, _ = solve_from_geometry(gradient, basis, interface)
    isotropic = dict(stretch_controls(gradient))["no_directional_stretch"]
    iso_grid, iso, _ = solve_from_geometry(isotropic, basis, interface)
    linear = linear_response(full_grid, interface)
    methods = {"full": (full_grid, full), "isotropic_warp": (iso_grid, iso),
               "energy_scaled_first_order": (full_grid, linear*relaxed_ray(full_grid, linear, interface))}
    return {kind: {"rigid": background, **{name: sample_disregistry(
        model, field, origin+background, cell) - origin for name, (model, field) in methods.items()}}
        for kind, background in backgrounds.items()}


def compare(scan, reference, images, context, data, metadata, benchmark, interface, settings):
    receipt = context["receipt"]
    pixel = receipt["pixel_nm"]
    angle = np.deg2rad(float(metadata["SampleRotation"]))
    y, x = np.indices(images.shape[:2], dtype=float)
    grid = pixel * np.stack((x, y), axis=-1)
    fit = np.asarray(receipt["first_ring_affine_fit"])
    centers = np.asarray(receipt["centers_rc_px"])
    basis = context["basis"]
    report = []
    for group, affine, warp, local, mask, diagnostic in geometric_folds(
        centers, grid, pixel, basis, fit, context["support"],
    ):
        checkpoint_guard()
        earlier = next((v for v in reference["local_scores"] if v["center_group"] == group), None)
        if earlier is None:
            continue
        g = local[:2].T @ rotation(-angle)
        backgrounds = {"affine": affine, "local": warp}
        fields = predictions(backgrounds, g, basis, interface)
        result = {"withheld_group": group, "first_ring_center_errors": diagnostic, "readouts": {}}
        for kind, selected in fields.items():
            scores = calibrate_and_score(selected, context, images, mask)
            if kind == "local":
                for a, b in zip(scores, earlier["scores"], strict=True):
                    if abs(a["full"] - b["rmse"]["full"]) > 1e-12:
                        raise ValueError(f"{scan}: local full diffraction did not reproduce")
            row = measure(selected, data, benchmark, context, settings, mask, backgrounds[kind])
            if row:
                y = np.asarray(row["target_percent"])
                row["models"]["equal_shear"] = {"directional_rmse_pp": float(y.std())}
            result["readouts"][kind] = {"mean_diffraction_rmse": averages(scores), "shear": row}
        if all(row["shear"] for row in result["readouts"].values()):
            report.append(result)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    files = (PROTOCOL, SOURCE, BENCHMARK, SETTINGS)
    protocol, source, benchmark, settings = (json.loads(path.read_text()) for path in files)
    author = load_author_release(args.release, args.dependencies)
    output = {"scope": "Test the wall-direction prediction on the pre-existing disjoint first-ring center folds. Each fold holds out one lattice color while the affine or interpolated center map defines wall locations. Full equilibrium, isotropic transport and scaled first order share the geometry and mask within each background choice. Compare with second-ring holdouts on the same fold and the established phase-gradient observer.",
              "limits": "This is a retrospective geometry sensitivity test. Measuring the same direction in the affine and warped analyses uses different wall pixels defined by their first-ring inputs. The two rings share an acquisition and the shift or distortion inferred from center placement is not a separate physical strain measurement.",
              "input_sha256": {f.name: digest(f) for f in (Path(__file__), *files)}, "scans": {}}
    for batch, names in protocol["cases"].items():
        for scan in names:
            checkpoint_guard()
            prior = source["scans"][scan]
            meta = read_input_metadata(scan, args.archive)
            archive = benchmark["scans"][scan]
            intensity, modes, fingerprint = disk_images(args.archive, scan, author)
            if fingerprint != prior["diskset_sha256"] or meta["archive_md5"] != archive["archive_md5"]:
                raise ValueError(f"{scan}: changed archive")
            sample = load_dataset(scan, args.archive, allowed_ids=(scan,), allow_author_crop=True)
            interface = tuple(prior["interface"])
            context = centers_and_context(intensity, modes, meta["metadata_fields"], settings, interface)
            if context["receipt"] != prior["geometry"]:
                raise ValueError(f"{scan}: changed geometry")
            rows = compare(scan, prior, intensity, context, sample, meta["metadata_fields"], archive, interface, settings)
            output["scans"][scan] = {"group": batch, "folds": rows}
            atomic_json(args.output, output)
            print(scan, len(rows), flush=True)


if __name__ == "__main__":
    main()
