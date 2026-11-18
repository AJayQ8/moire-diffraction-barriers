"""Report width and registered profile repeatability across opposite diffraction spots."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from summarize_mos2_geometry_probe import PHASES, block_draws, interval, qualified


CAMERAS = ("transferred_even", "separate_disjoint")
PRIMARY = "4.25"
CROSSED = ((0, 0, 1, 1), (1, 0, 0, 1))


def widths(row):
    measured = np.asarray([side["widths_nm"] for side in row["measured"]])
    predicted = np.asarray([[side["widths_nm"] for side in row["models"][camera]["widths"]]
                            for camera in CAMERAS])
    return measured, predicted


def valid(row):
    return qualified(*row["measured"], *(side for camera in CAMERAS for side in row["models"][camera]["widths"]))


def rms(a):
    return np.sqrt(np.mean(a**2, axis=0))


def correlation(a, b):
    if len(a) < 3 or min(np.std(a), np.std(b)) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def width_scores(walls, window, use, rng):
    if not use:
        return {"walls": 0, "cameras": {}}
    draw, support = block_draws(walls, use, rng)
    samples = [widths(walls[i]["windows"][window]) for i in use]
    measured, predicted = [np.asarray(values) for values in zip(*samples, strict=True)]
    reports = {}
    for i, camera in enumerate(CAMERAS):
        residual = measured - predicted[:, i]
        common = residual.mean(axis=1)
        difference = (residual[:, 0] - residual[:, 1]) / 2
        contrasts = (draw @ residual.reshape(len(use), -1)).reshape(-1, 2, 2)
        reports[camera] = {
            "mean_predicted_nm": predicted[:, i].mean(axis=0).tolist(),
            "predicted_wall_sd_nm": predicted[:, i].std(axis=0).tolist(),
            "mean_residual_nm": residual.mean(axis=0).tolist(),
            "mean_residual_p05_p95_nm": interval(contrasts),
            "common_rms_nm": rms(common).tolist(),
            "opposite_half_difference_rms_nm": rms(difference).tolist(),
            "common_centered_rms_nm": rms(common - common.mean(axis=0)).tolist(),
            "difference_centered_rms_nm": rms(difference - difference.mean(axis=0)).tolist(),
            "opposite_residual_correlation_by_order": [
                correlation(residual[:, 0, j], residual[:, 1, j]) for j in (0, 1)
            ],
            "cross_order_cross_spot_residual_correlation": [
                correlation(residual[:, a, b], residual[:, c, d]) for a, b, c, d in CROSSED
            ],
        }
    return {"walls": len(use), "block_support": support, "center_pairs": [walls[i]["centers"] for i in use],
            "mean_measured_nm": measured.mean(axis=0).tolist(), "measured_wall_sd_nm": measured.std(axis=0).tolist(),
            "cross_order_cross_spot_measured_correlation": [
                correlation(measured[:, a, b], measured[:, c, d]) for a, b, c, d in CROSSED
            ],
            "cameras": reports}


def shape_scores(walls, window, camera):
    rows = [wall["windows"][window]["models"][camera]["profile"] for wall in walls]
    common = np.mean([[item["common_squared"] for item in row] for row in rows], axis=0)
    contrast = np.mean([[item["difference_squared"] for item in row] for row in rows], axis=0)
    return {
        "wall_sections": len(walls),
        "source_registration_at_bound_by_order": np.sum([
            [item["source_center_at_bound"] for item in row] for row in rows
        ], axis=0).tolist(),
        "common_rms": np.sqrt(common).tolist(),
        "opposite_half_difference_rms": np.sqrt(contrast).tolist(),
        "replicated_residual_fraction_of_mean_squared_error": ((common-contrast)/(common+contrast)).tolist(),
    }


def defocus_scores(reference):
    results = {}
    for name, record in reference["scans"].items():
        walls = [wall for wall in record["walls"] if qualified(
            wall["nominal"][CAMERAS[0]]["measured"], wall["nominal"][CAMERAS[0]]["models"]["anchor"]["native"],
            *(pupil["anchor"] for pupil in wall["pupil"].values()),
        )]
        measured = np.mean([wall["nominal"][CAMERAS[0]]["measured"]["widths_nm"] for wall in walls], axis=0)
        nominal = np.mean([w["nominal"][CAMERAS[0]]["models"]["anchor"]["native"]["widths_nm"]
                           for w in walls], axis=0)
        result = {"wall_sections": len(walls), "nominal_gaussian": (measured - nominal).tolist()}
        for defocus in reference["ideal_pupil_defocus_nm"]:
            predicted = np.mean([w["pupil"][str(defocus)]["anchor"]["widths_nm"] for w in walls], axis=0)
            result[str(defocus)] = (measured - predicted).tolist()
        results[name] = result
    return results


def summarize(raw, reference):
    if set(raw["scans"]) != set(raw["requested_scans"]):
        raise ValueError("Missing maps from the matched wall cohort")
    windows = [f"{value:g}" for value in raw["half_windows_nm"]]
    rng = np.random.default_rng(20261116)
    scans = {}
    for name in raw["requested_scans"]:
        walls = raw["scans"][name]["walls"]
        joint = [i for i, wall in enumerate(walls) if all(valid(wall["windows"][window]) for window in windows)]
        scores = {}
        for window in windows:
            qualified_walls = [i for i, wall in enumerate(walls) if valid(wall["windows"][window])]
            scores[window] = {
                "widths": width_scores(walls, window, qualified_walls, rng),
                "widths_on_joint_window_cohort": width_scores(walls, window, joint, rng),
                "profiles": {cam: shape_scores(walls, window, cam) for cam in CAMERAS},
            }
        scans[name] = scores
    return {"scans": scans, "defocus_width_residuals_nm": defocus_scores(reference)}


def profile_rms(entry):
    return [np.sqrt(item["common_squared"] + item["difference_squared"]) for item in entry["profile"]]


def scan_directions(walls):
    angle = 2 * np.deg2rad([wall["scan_angle_deg"] for wall in walls])
    return np.column_stack((np.cos(angle), np.sin(angle)))


def spatial_screen(walls, residual, support, source=None):
    """Predict held second-ring residuals from the opposite first-ring readout."""
    xy = np.asarray([wall["midpoint_xy_nm"] for wall in walls])
    directions = scan_directions(walls)
    kinds = {
        "constant": np.empty((len(walls), 0)),
        "scan_position": xy / support["side_nm"],
        "scan_angle": directions,
        "local_geometry": np.column_stack((
            [wall["character"] for wall in walls], np.log([wall["spacing_nm"] for wall in walls]),
        )),
    }
    design = {name: np.column_stack((np.ones(len(walls)), v)) for name, v in kinds.items()}
    transferred = None
    if source is not None:
        source_walls, source_residual = source
        X = np.column_stack((np.ones(len(source_walls)), scan_directions(source_walls)))
        transferred = np.linalg.lstsq(X, source_residual[:, :, 0], rcond=None)[0][1:]
    separation = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    per_origin = []
    side = support["side_nm"]
    # The halo limits overlap between training and test profile footprints.
    for phase in PHASES:
        bins = np.floor((xy - xy.min(axis=0)) / side + phase).astype(int)
        _, groups = np.unique(bins, axis=0, return_inverse=True)
        errors = {name: [] for name in design}
        if transferred is not None:
            errors["scan_angle_from_other_scan"] = []
        folds = 0
        for group in np.unique(groups):
            test = groups == group
            train = ~test & (np.min(separation[:, test], axis=1) > max(10., .5 * side))
            if train.sum() < 6 or any(np.linalg.matrix_rank(a[train]) < a.shape[1] for a in design.values()):
                continue
            folds += 1
            for first_half, second_half in ((0, 1), (1, 0)):
                a, b = residual[:, first_half, 0], residual[:, second_half, 1]
                for name, X in design.items():
                    regression = np.linalg.lstsq(X[train], a[train], rcond=None)[0]
                    errors[name].extend((b[test] - X[test] @ regression)**2)
                if transferred is not None:
                    offset = directions[test] - directions[train].mean(axis=0)
                    prediction = a[train].mean() + offset @ transferred[:, first_half]
                    errors["scan_angle_from_other_scan"].extend((b[test] - prediction)**2)
        point = {name: float(np.sqrt(np.mean(v))) for name, v in errors.items() if v}
        per_origin.append({"folds": folds, "predictions": len(errors["constant"]), "error_nm": point})
    return per_origin


def compare_local(global_result, local_result):
    if set(local_result["requested_scans"]) != set(local_result["scans"]):
        raise ValueError("The local equilibrium comparison is incomplete")
    rng = np.random.default_rng(20261117)
    result = {}
    screen_inputs = {}
    for name in local_result["requested_scans"]:
        baseline = global_result["scans"][name]["walls"]
        local = local_result["scans"][name]
        if [r["centers"] for r in local] != [r["centers"] for r in baseline]:
            raise ValueError(f"Baseline and local wall sections differ in {name}")
        old = [r["windows"][PRIMARY] for r in baseline]
        new = [r["local"] for r in local]
        use = [i for i, (a, b) in enumerate(zip(old, new, strict=True)) if valid(a)
               and qualified(*(fit for cam in CAMERAS for fit in b[cam]["widths"]))]
        if not use:
            raise ValueError(f"There are no common absolute-width fits for {name}")
        width_draw, support = block_draws(baseline, use, rng)
        shape_draw, _ = block_draws(baseline, range(len(baseline)), rng)
        measured = np.asarray([[w["widths_nm"] for w in old[i]["measured"]] for i in use])
        area = [r["local_to_global_cell_area"] for r in local]
        report = {
            "shape_sections": len(baseline), "common_width_sections": len(use), "width_block_support": support,
            "grids": sorted({r["grid"] for r in local}),
            "local_to_global_cell_area_range": [min(area), max(area)],
            "measurements_centered_rms_nm": rms(measured-measured.mean(axis=0)).mean(axis=0).tolist(),
            "cameras": {},
        }
        screen_inputs[name] = {}
        for camera in CAMERAS:
            base_shape = np.asarray([profile_rms(r["models"][camera]) for r in old])
            local_shape = np.asarray([profile_rms(r[camera]) for r in new])
            predictions = [np.asarray([[fit["widths_nm"] for fit in row["widths"]] for row in rows])
                           for rows in ([old[i]["models"][camera] for i in use], [new[i][camera] for i in use])]
            residual = [measured - v for v in predictions]
            mae_by_wall = [np.mean(abs(v), axis=1) for v in residual]
            report["cameras"][camera] = {
                "shape_global": base_shape.mean(axis=0).tolist(),
                "shape_local": local_shape.mean(axis=0).tolist(),
                "shape_local_minus_global_p05_p95": interval(shape_draw @ (local_shape-base_shape)),
                "width_mae_global_nm": mae_by_wall[0].mean(axis=0).tolist(),
                "width_mae_local_nm": mae_by_wall[1].mean(axis=0).tolist(),
                "width_mae_local_minus_global_p05_p95_nm": interval(width_draw @ (mae_by_wall[1]-mae_by_wall[0])),
                "width_centered_rms_global_nm": rms(residual[0]-residual[0].mean(axis=0)).mean(axis=0).tolist(),
                "width_centered_rms_local_nm": rms(residual[1]-residual[1].mean(axis=0)).mean(axis=0).tolist(),
                "mean_residual_global_nm": residual[0].mean(axis=0).tolist(),
                "mean_residual_local_nm": residual[1].mean(axis=0).tolist(),
            }
            screen_inputs[name][camera] = [baseline[i] for i in use], residual[1]
        result[name] = report
    counterparts = {"DS13": "DS14", "DS14": "DS13"}
    for name, report in result.items():
        for camera, row in report["cameras"].items():
            walls, residual = screen_inputs[name][camera]
            source = screen_inputs.get(counterparts.get(name), {}).get(camera)
            row["held_width_residual_screen"] = spatial_screen(walls, residual, report["width_block_support"], source)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case_scores", "native_reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--local-result", type=Path)
    args = parser.parse_args()
    paths = (args.case_scores, args.native_reference)
    raw, native = (json.loads(path.read_bytes()) for path in paths)
    result = summarize(raw, native)
    if args.local_result:
        result["local_mechanics"] = compare_local(raw, json.loads(args.local_result.read_bytes()))
        paths += (args.local_result,)
    result["input_sha256"] = {path.name: sha256(path.read_bytes()).hexdigest() for path in paths}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
