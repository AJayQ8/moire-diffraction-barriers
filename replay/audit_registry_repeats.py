"""Measure how closely different cells of one experimental moiré map agree."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from bilayer_relaxation import disregistry_predictions
from run_registry_prediction import load_registry


def repeated_cell_reference(field, valid, cell_pixels):
    """At each pixel average the same stacking coordinate in other cells."""
    yy, xx = np.indices(valid.shape, dtype=float)
    coordinates = np.stack((yy, xx))
    corners = np.array(((0, 0), (xx.max(), 0), (0, yy.max()), (xx.max(), yy.max())))
    reach = np.ceil(np.ptp(corners @ np.linalg.inv(cell_pixels).T, axis=0)).astype(int)
    counts = np.zeros(valid.shape, dtype=int)
    total = np.zeros_like(field)
    for i in range(-reach[0], reach[0] + 1):
        for j in range(-reach[1], reach[1] + 1):
            if (i, j) == (0, 0):
                continue
            dx, dy = cell_pixels @ np.array((i, j))
            if abs(dx) >= xx.max() or abs(dy) >= yy.max():
                continue
            shifted = coordinates + np.array((dy, dx))[:, None, None]
            common = valid & (ndi.map_coordinates(valid.astype(float), shifted, order=1) > 0.999)
            if not np.any(common):
                continue
            for d in range(2):
                total[..., d][common] += ndi.map_coordinates(field[..., d], shifted, order=1)[common]
            counts += common
    return total / np.maximum(counts[..., None], 1), counts


def audit_case(scan, row, archive, protocol, guard):
    observed, _, metadata, _ = load_registry(archive, scan)
    pixel = float(metadata["PixelSize"])
    lattice = float(metadata["LatticeConstant"])
    basis = lattice * np.array(((1, 0.5), (0, np.sqrt(3) / 2)))
    fit = np.asarray(row["fit_scan"])
    yy, xx = np.indices(observed.shape[:2], dtype=float) * pixel
    rigid = np.stack((xx, yy), axis=-1) @ fit[:2] + fit[-1]
    observed_mask = np.isfinite(observed).all(axis=-1)
    sigma = protocol["smoothing_sigma_pixels"]
    weights = ndi.gaussian_filter(observed_mask.astype(float), sigma, mode="constant")
    valid = observed_mask & (weights > 0.97)

    def process(field):
        data = np.where(observed_mask[..., None], field - rigid, 0)
        filtered = ndi.gaussian_filter(data, (sigma, sigma, 0), mode="constant")
        return filtered / np.maximum(weights[..., None], 1e-10)

    target = process(observed)
    cell_pixels = np.linalg.solve(fit[:2].T, basis) / pixel
    reference, counts = repeated_cell_reference(target, valid, cell_pixels)
    support = valid & (counts >= 3)
    fraction = np.count_nonzero(support) / np.count_nonzero(valid)
    result = {"fraction_with_three_other_cells": fraction, "scoreable": bool(fraction >= 0.2)}
    if not result["scoreable"]:
        return result
    predictions, _ = disregistry_predictions(np.asarray(row["gradient_crystal"]), basis, rigid, ("MoS2", "MoS2", "P"),
                                              row["grid_size"], protocol["probe_fwhm_nm"], guard)

    def scores(field):
        difference = field[support] - target[support]
        return {
            "rmse_nm": float(np.sqrt(np.mean(np.sum(difference**2, axis=-1)))),
            "offset_free_rmse_nm": float(np.sqrt(np.mean(np.sum((difference - difference.mean(axis=0))**2, axis=-1)))),
        }

    result["errors"] = {name: scores(process(field)) for name, field in predictions.items()}
    result["errors"]["other_cells_template"] = scores(reference)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    from shared_compute import atomic_json, checkpoint_guard

    project = Path(__file__).resolve().parent
    protocol = json.loads((project / "registry_protocol_v01.json").read_text())
    original = json.loads((project / "registry_held_out_v01.json").read_text())
    result = {"interpretation": "Leave-one-cell-out reference within each already tested scan. It measures spatial repeatability at the fitted geometry and includes both experimental error and genuine cell-to-cell variation. It is not an independent prediction from materials properties.",
              "scans": {}}
    for scan, row in original["scans"].items():
        checkpoint_guard()
        audited = audit_case(scan, row, args.archive, protocol, checkpoint_guard)
        result["scans"][scan] = audited
        print(scan, audited, flush=True)
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
