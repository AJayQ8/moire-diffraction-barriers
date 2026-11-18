"""Replay the blinded MoS2 inversion with a supplied normal probe width."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from run_mos2_barrier_recovery import (compare_profiles, estimates, fingerprint, landscape_family,
                                       noise_covariance, profile_templates, sources, wall_geometry,
                                       whitened_shape_operator)
from shared_compute import atomic_json, checkpoint_guard


def profile_at_probe(templates, probe_grid, probe_fwhm):
    high = int(np.searchsorted(probe_grid, probe_fwhm, side="right"))
    if not 0 < high < len(probe_grid):
        raise ValueError("The supplied normal probe lies outside the interpolation grid")
    low = high - 1
    weight = (probe_fwhm - probe_grid[low]) / (probe_grid[high] - probe_grid[low])
    return (1 - weight) * templates[:, :, low:high] + weight * templates[:, :, high:high+1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("known_probe_widths", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    protocol = json.loads(args.protocol.read_text())
    calibration = json.loads(args.known_probe_widths.read_text())
    options = protocol["fitting"]
    settings = protocol["generator"]
    material = landscape_family()
    coarse = np.geomspace(*options["relative_barrier_bounds"], options["simulation_grid_barrier_points"])
    fine = np.geomspace(*options["relative_barrier_bounds"], options["search_grid_barrier_points"])
    barriers = material[0] * fine
    probe_grid = np.linspace(.65, 1.95, 11)
    options["fitted_probe_fwhm_nm"] = probe_grid.tolist()
    report = {"scope": "Sensitivity control after the original blinded score. Replay the periodic model with the normal "
                        "probe width supplied for each simulated image. Blur is still represented by an isotropic Gaussian. "
                        "Use the same noisy pixels and permit the original nuisance freedoms for stacking shape, camera and center.",
              "source_sha256": {**sources(args.protocol), Path(__file__).name: fingerprint(Path(__file__))},
              "images_sha256": fingerprint(args.images), "known_probe_sha256": fingerprint(args.known_probe_widths),
              "probe_library_nm": probe_grid.tolist(), "samples": {}}
    with np.load(args.images) as archive:
        receipt = json.loads(archive["receipt"].item())
        if receipt["source_sha256"] != sources(args.protocol) or calibration["sealed_truth_sha256"] != receipt["truth_sha256"]:
            raise ValueError("The images or supplied probe calibration do not match the original frozen run")
        expected = {sample for sample, tags in receipt["samples"].items() if tags["condition"] != "nominal"}
        if set(calibration["normal_fwhm_nm"]) != expected:
            raise ValueError("The supplied probe list is incomplete")
        report["sealed_truth_sha256"] = receipt["truth_sha256"]
        for angle in protocol["twist_degrees"]:
            wall = wall_geometry(angle, protocol["raster_pixel_nm"])
            templates = profile_templates(wall, protocol, "periodic", material, coarse, fine, protocol["raster_pixel_nm"] / 4)
            for sample, widths in calibration["normal_fwhm_nm"].items():
                tags = receipt["samples"][sample]
                if tags["twist_degrees"] != angle:
                    continue
                measured = archive[sample].mean(axis=1, dtype=float)
                covariance = noise_covariance(wall, settings["conditions"][tags["condition"]], settings)
                operator = whitened_shape_operator(wall, covariance)
                fits = []
                for probe, target in zip(widths, measured, strict=True):
                    prediction = profile_at_probe(templates, probe_grid, probe)
                    loss = compare_profiles(prediction, target[None], operator, options["relative_gain_bounds"])
                    outputs = estimates(loss, barriers, options["landscape_mixtures"], [probe],
                                        options["fitted_normal_shifts_nm"], len(wall.across), options["noise_likelihood_delta"])
                    fits.append({policy: values[0] for policy, values in outputs["both"].items()})
                report["samples"][sample] = {**tags, "fits": fits}
            report["wall_seconds"] = perf_counter() - start
            checkpoint_guard()
            atomic_json(args.output, report)
            print(f"Replayed {angle:g} degree noisy images in {report['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
