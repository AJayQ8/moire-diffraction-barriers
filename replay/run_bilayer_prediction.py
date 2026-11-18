"""Compare fixed continuum predictions with archived MoS2 homobilayer wall shear."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from bilayer_relaxation import ConvergenceError, disregistry_predictions
from run_registry_prediction import fit_center_geometry, score_fields


PROJECT = Path(__file__).resolve().parent


def error(observed, predicted):
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)
    if observed.shape != (3,) or predicted.shape != (3,) or not np.all(np.isfinite([observed, predicted])):
        raise ValueError("The comparison requires a finite mean shear in all three wall directions")
    residual = predicted - observed
    largest_matches = bool(predicted.argmax() == observed.argmax()) if np.ptp(predicted) > 1e-6 else None
    return {
        "absolute_rmse_percentage_points": float(np.sqrt(np.mean(residual**2))),
        "directional_rmse_percentage_points": float(np.sqrt(np.mean((residual - residual.mean())**2))),
        "strongest_wall_matches": largest_matches,
    }


def mean_errors(scans):
    scores = {model: [row["comparisons"][model]["errors"] for row in scans]
              for model in scans[0]["comparisons"]}
    result = {}
    for model, values in scores.items():
        result[model] = {}
        for key in values[0]:
            entries = [row[key] for row in values if row[key] is not None]
            result[model][key] = float(np.mean(entries)) if entries else None
    return result


def run_case(dataset, measure, settings, guard):
    # The archive's x axis is the column direction and its y axis is the
    # row direction. The vectors are expressed in the crystal frame.
    raw, lattice_nm, pixel_nm, angle, orientation = dataset
    basis = lattice_nm * np.array(((1, 0.5), (0, np.sqrt(3) / 2)))
    fit, geometry = fit_center_geometry(raw, dataset.centers_rc_px, basis, pixel_nm, settings["center_inlier_nm"])
    yy, xx = np.indices(raw.shape[:2], dtype=float) * pixel_nm
    points = np.stack((xx, yy), axis=-1)
    rigid_field = points @ fit[:2] + fit[-1]
    alpha = np.deg2rad(angle)
    rotation = np.array(((np.cos(alpha), np.sin(alpha)), (-np.sin(alpha), np.cos(alpha))))
    affine = fit[:2].T @ rotation
    cell = np.linalg.solve(affine, basis)
    n = max(64, 2 ** int(np.ceil(np.log2(np.max(np.linalg.norm(cell, axis=0)) / 0.3))))
    if n > 512:
        raise ValueError(f"The fitted cell would need a grid beyond this job's reservation: {n}")
    interface = ("MoS2", "MoS2", orientation.upper())
    fields, solves = disregistry_predictions(affine, basis, rigid_field, interface, n,
                                             settings["probe_fwhm_nm"], guard)
    observed = measure(dataset)
    shear_means = np.array([observed["sp_details"][f"SP{i}"]["mean_percent"] for i in (1, 2, 3)])
    field_scores, _ = score_fields(raw, fields, rigid_field, settings["smoothing_sigma_pixels"])
    observed["wall_shear_percent"] = shear_means.tolist()
    comparisons = {}
    for name, field in fields.items():
        measured = measure(dataset, u_nm=field)
        predicted_means = [measured["sp_details"][f"SP{i}"]["mean_percent"] for i in (1, 2, 3)]
        errors = error(shear_means, predicted_means)
        if name == "rigid":
            # Uniform affine strain cannot distinguish the wall directions.
            # Tiny edge and mask effects must not create a rank prediction.
            errors["strongest_wall_matches"] = None
        errors["relative_disregistry_rmse_nm"] = field_scores[name]["rmse_nm"]
        comparisons[name] = {"wall_shear_percent": predicted_means, "errors": errors,
                             "stacking_sector_counts": measured["stacking_sector_counts"]}
    return {
        "orientation": orientation, "archive_md5": dataset.archive_md5,
        "geometry": geometry, "affine_crystal": affine.tolist(), "cell_nm": cell.tolist(),
        "solver_grid": int(n), "solver_checks": solves,
        "observed": observed, "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_dir", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch", choices=("development", "held_out"), required=True)
    args = parser.parse_args()
    from observe_bilayers import load_dataset, observe_wall_shear
    from shared_compute import atomic_json, checkpoint_guard

    protocol = json.loads(args.protocol.read_text())
    geometry_protocol = PROJECT / "registry_protocol_v01.json"
    protocol["center_inlier_nm"] = json.loads(geometry_protocol.read_text())["center_inlier_nm"]
    cases = protocol["batches"][args.batch]
    start = perf_counter()
    result = {"batch": args.batch, "protocol_sha256": sha256(args.protocol.read_bytes()).hexdigest(),
              "geometry_protocol_sha256": sha256(geometry_protocol.read_bytes()).hexdigest(),
              "implementation_sha256": {name: sha256((PROJECT / name).read_bytes()).hexdigest()
                                        for name in ("bilayer_relaxation.py", "observe_bilayers.py",
                                                     "run_bilayer_prediction.py", "run_registry_prediction.py")},
              "scans": {}, "failed_scans": {}}
    for scan in cases:
        checkpoint_guard()
        try:
            dataset = load_dataset(scan, args.archive_dir, allowed_ids=cases)
            row = run_case(dataset, observe_wall_shear, protocol, checkpoint_guard)
        except (ConvergenceError, ValueError, np.linalg.LinAlgError) as failure:
            result["failed_scans"][scan] = str(failure)
            print(scan, str(failure), flush=True)
        else:
            result["scans"][scan] = row
            result["means"] = mean_errors(list(result["scans"].values()))
            print(scan, "observed", row["observed"]["wall_shear_percent"],
                  "directional error",
                  {name: round(item["errors"]["directional_rmse_percentage_points"], 3)
                   for name, item in row["comparisons"].items()}, flush=True)
        result["wall_seconds"] = perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
