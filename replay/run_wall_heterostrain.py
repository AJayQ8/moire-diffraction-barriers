"""Compare the direction and magnitude of wall shear at fixed measured geometry."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from author_release import load_author_release
from bilayer_relaxation import sample_disregistry
from observe_bilayers import load_dataset, read_input_metadata
from run_heterobilayer_sparse_geometry import linear_response, solve_from_geometry
from run_heterobilayer_transfer import PROJECT, blur, digest
from run_moire_common_intensity import BENCHMARK, calibrate_and_score, centers_and_context
from run_moire_regime_benchmark import SETTINGS, relaxed_ray
from run_mos2_directional_diffraction import wall_sectors
from run_mos2_directional_split_ring import stretch_controls
from run_mos2_intensity_only_replay import disk_images
from run_mos2_shear_readout_comparison import shear
from run_regional_shear import geometry_properties
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


PROTOCOL = PROJECT / "wall_heterostrain_protocol_v01.json"
SOURCE = PROJECT / "moire_common_intensity_replay_v01.json"
PARALLEL_REFERENCE = PROJECT / "mos2_shear_readout_comparison_v01.json"


def wall_fields(context, interface):
    gradient, basis, rigid = (context[key] for key in ("gradient", "basis", "rigid"))
    cell = np.linalg.solve(gradient, basis)
    origin = basis @ np.full(2, 2 / 3 if interface[-1] == "AP" else 0)
    phase = rigid + origin
    names = {"measured_direction": "full", "no_directional_stretch": "isotropic_warp"}
    fields = {"rigid": rigid}
    for name, test_gradient in stretch_controls(gradient):
        checkpoint_guard()
        grid, shift, _ = solve_from_geometry(test_gradient, basis, interface)
        measured = name == "measured_direction"
        observation_cell = grid.cell if measured else cell
        fields[names.get(name, name)] = sample_disregistry(grid, shift, phase, observation_cell) - origin
        if measured:
            linear = linear_response(grid, interface)
            amplitude = relaxed_ray(grid, linear, interface)
            fields["energy_scaled_first_order"] = (
                sample_disregistry(grid, amplitude * linear, phase, grid.cell) - origin
            )
    return fields


def contrast_score(values, target):
    centered = values - values.mean()
    measured = target - target.mean()
    rmse = np.sqrt(np.mean((centered - measured)**2))
    # A constant prediction has no weakest wall and no fractional contrast.
    lowest = int(values.argmin()) if np.ptp(values) > 1e-6 else None
    relative = centered / values.mean() if np.ptp(values) > 1e-6 else centered
    return {
        "wall_shear_percent": values.tolist(),
        "directional_rmse_pp": float(rmse),
        "relative_contrast_rmse": float(np.sqrt(np.mean((relative - measured / target.mean())**2))),
        "lowest_wall_matches": (lowest == int(target.argmin())) if lowest is not None else None,
    }


def wall_scores(data, archive, context, fields, settings):
    row, col = data.metadata.get("author_unwrap_crop_origin_row_col", (0, 0))
    height, width = data.u_nm.shape[:2]
    crop = np.s_[row:row + height, col:col + width]
    rigid = context["rigid"]
    if rigid[crop].shape != data.u_nm.shape:
        raise ValueError("The saved phase and intensity shapes do not align")
    valid = data.source_valid_mask
    sigma = settings["smoothing_sigma_pixels"]
    weights = gaussian_filter(valid.astype(float), sigma, mode="constant")
    support = context["support"][crop] & valid & (weights > 0.97)
    observed = shear(blur(data.u_nm, valid, weights, sigma), data.pixel_nm,
                     np.deg2rad(data.sample_rotation_degrees))
    masks = [support & sector for sector in wall_sectors(rigid[crop], context["basis"]).values()]
    counts = [int(mask.sum()) for mask in masks]
    if min(counts) < 20:
        return {"failure": "Fewer than 20 pixels in a wall sector", "sector_pixels": counts}
    target = np.array([observed[mask].mean() for mask in masks])
    if not np.all(np.isfinite(target)) or target.mean() <= 0:
        raise ValueError("Archived wall-sector shear is invalid")
    nominal = np.asarray(archive["affine_geometry"]["fit_scan"])
    angle = np.deg2rad(data.sample_rotation_degrees)
    align = (nominal[:2].T @ rotation(-angle)) @ np.linalg.inv(context["gradient"])
    rotations = [rotation(k * np.pi / 3) for k in range(6)]
    branch = min(rotations, key=lambda candidate: np.linalg.norm(candidate - align))
    scores = {"equal_shear": contrast_score(np.zeros(3), target)}
    probe = context["probe"]
    for name, field in fields.items():
        filtered = gaussian_filter((field - rigid)[crop], (probe, probe, 0), mode="nearest")
        smoothed = blur((rigid[crop] + filtered) @ branch.T, valid, weights, sigma)
        values = shear(smoothed, data.pixel_nm, angle)
        scores[name] = contrast_score(np.array([values[mask].mean() for mask in masks]), target)
    return {"observed_wall_shear_percent": target.tolist(), "sector_pixels": counts,
            "frame_fit_residual": float(np.linalg.norm(align - branch)), "models": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    inputs = (PROTOCOL, SOURCE, BENCHMARK, SETTINGS, PARALLEL_REFERENCE)
    protocol, source, benchmark, settings, parallel = (json.loads(path.read_text()) for path in inputs)
    author = load_author_release(args.release, args.dependencies)
    result = {"input_sha256": {p.name: digest(p) for p in (Path(__file__), *inputs)},
              "scans": {}}
    for group, scans in protocol["cases"].items():
        for scan in scans:
            checkpoint_guard()
            record = source["scans"][scan]
            archive = benchmark["scans"][scan]
            interface = tuple(record["interface"])
            metadata = read_input_metadata(scan, args.archive)
            images, modes, digest_disk = disk_images(args.archive, scan, author)
            if digest_disk != record["diskset_sha256"] or metadata["archive_md5"] != archive["archive_md5"]:
                raise ValueError(f"{scan}: source changed")
            context = centers_and_context(images, modes, metadata["metadata_fields"], settings, interface)
            if context["receipt"] != record["geometry"]:
                raise ValueError(f"{scan}: first-ring fit changed")
            fields = wall_fields(context, interface)
            diffraction = calibrate_and_score(fields, context, images, context["support"])
            diffraction = {key: float(np.mean([fold[key] for fold in diffraction])) for key in fields}
            if abs(diffraction["full"] - record["mean_affine"]["full"]) > 1e-12:
                raise ValueError(f"{scan}: full-model diffraction replay changed")
            data = load_dataset(scan, args.archive, allowed_ids=(scan,), allow_author_crop=True)
            wall = wall_scores(data, archive, context, fields, settings)
            prior = parallel["scans"].get(scan)
            if prior:
                recent = wall.get("models", {}).get("full", {}).get("directional_rmse_pp")
                expected = prior["models"]["full"]["directional_rmse_pp"]
                if recent is None or abs(recent - expected) > 1e-12:
                    raise ValueError(f"{scan}: directional wall comparison changed from {expected} to {recent}")
            result["scans"][scan] = {
                "group": group, "interface": list(interface), "diskset_sha256": digest_disk,
                "geometry": geometry_properties(context["gradient"]),
                "published_heterostrain_percent": archive["reported_heterostrain_percent"],
                "second_ring_mean_rmse": diffraction, "wall_scores": wall,
            }
            atomic_json(args.output, result)
            print(scan, {name: round(row["directional_rmse_pp"], 5)
                         for name, row in wall.get("models", {}).items()} or wall, flush=True)


if __name__ == "__main__":
    main()
