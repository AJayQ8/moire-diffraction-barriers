"""Summarize native-raster width and profile comparisons on paired sections."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


LOW = ("DS1", "DS2", "DS14", "DS13")
CAMERAS = ("transferred_even", "separate_disjoint")
FIXED = ("anchor", "raman_shape", "raman_scale")
PHASES = ((0., 0.), (.5, 0.), (0., .5), (.5, .5))
DRAW_COUNT = 1000
GEOMETRY_SEED = 20261116


def qualified(*fits):
    return all(all(fit["qualified"]) for fit in fits)


def interval(values):
    return np.percentile(values, [5, 95], axis=0).tolist()


def block_draws(walls, retained, rng):
    xy = np.asarray([wall["midpoint_xy_nm"] for wall in walls])
    side = np.sqrt(2 / np.sqrt(3)) * np.median([wall["spacing_nm"] for wall in walls])
    retained_xy = (xy[np.asarray(retained)] - xy.min(axis=0)) / side
    draws, support = [], []
    for phase in PHASES:
        blocks = np.floor(retained_xy + phase).astype(int)
        _, ids = np.unique(blocks, axis=0, return_inverse=True)
        count = int(ids.max()) + 1
        indices = rng.integers(0, count, (DRAW_COUNT, count))
        counts = np.eye(count)[indices].sum(axis=1)[:, ids]
        draws.append(counts / counts.sum(axis=1, keepdims=True))
        support.append(count)
    return np.concatenate(draws), {"side_nm": float(side), "occupied_blocks": support}


def error_stats(measured, predicted):
    return np.mean(abs(predicted - measured), axis=-1)


def geometry_and_headline(scans, camera):
    stats, scores, errors = {}, {}, {}
    rng = np.random.default_rng(GEOMETRY_SEED)
    for name, scan in scans.items():
        rows = [wall["nominal"][camera] for wall in scan["walls"]]
        models = [model for model in rows[0]["models"] if model not in (*FIXED[1:], "rigid")]
        kinds = [(model, "native") for model in models] + [("anchor", "continuous")]
        take = [i for i, row in enumerate(rows) if qualified(
            row["measured"], *(row["models"][model][kind] for model, kind in kinds),
        )]
        if not take:
            raise ValueError(f"No matched nominal widths in {name} with {camera}")
        draw, blocks = block_draws(scan["walls"], take, rng)
        measured = np.array([rows[i]["measured"]["widths_nm"] for i in take])
        center = measured.mean(axis=0)
        m_draw = draw @ measured
        stats[name] = {
            "walls": len(take), "eligible_walls": len(rows), "width_cohort": take, **blocks,
            "measured_nm": center.tolist(), "measured_p05_p95_nm": interval(m_draw), "models": {},
        }
        for model, kind in kinds:
            tag = f"{model}_continuous" if kind == "continuous" else model
            width = np.array([rows[i]["models"][model][kind]["widths_nm"] for i in take])
            avg = width.mean(axis=0)
            pred_draw = draw @ width
            stats[name]["models"][tag] = {
                "width_nm": avg.tolist(), "width_p05_p95_nm": interval(pred_draw),
                "predicted_minus_measured_nm": (avg - center).tolist(),
                "predicted_minus_measured_p05_p95_nm": interval(pred_draw - m_draw),
                "mean_absolute_width_error_nm": float(error_stats(center, avg)),
                "mean_absolute_width_error_p05_p95_nm": interval(error_stats(m_draw, pred_draw)),
            }
            if tag == "anchor":
                errors[name] = error_stats(m_draw, pred_draw)
        scores[name] = {
            model: {kind: float(np.mean([row["models"][model][kind]["transfer_error"] for row in rows]))
                    for kind in ("native", "continuous")}
            for model in ("anchor", "rigid")
        }
        scores[name]["width_control_profiles"] = {
            model: float(np.mean([row["models"][model]["native"]["transfer_error"] for row in rows]))
            for model in models
        }
        stats[name]["per_order_unqualified"] = {
            model: np.sum([np.logical_not(row["models"][model]["native"]["qualified"]) for row in rows],
                          axis=0).tolist()
            for model in rows[0]["models"]
        } | {"measured": np.sum([np.logical_not(row["measured"]["qualified"]) for row in rows], axis=0).tolist()}
    low = [m for m in scans if m in LOW]
    high = [m for m in scans if m not in LOW]
    grouped = {}
    for group, maps in (("low", low), ("high", high), ("all", list(scans))):
        if not maps:
            continue
        points = [stats[n]["models"]["anchor"]["mean_absolute_width_error_nm"] for n in maps]
        grouped[group] = {
            "maps": maps, "walls": sum(stats[n]["walls"] for n in maps), "mean_error_nm": float(np.mean(points)),
            "mean_error_p05_p95_nm": interval(np.mean([errors[n] for n in maps], axis=0)),
            "leave_one_map_out_nm": {n: float(np.mean([p for m, p in zip(maps, points, strict=True) if m != n]))
                                    for n in maps} if len(maps) > 1 else {},
        }
    if low and high:
        grouped["low_minus_high_nm"] = grouped["low"]["mean_error_nm"] - grouped["high"]["mean_error_nm"]
        grouped["low_minus_high_p05_p95_nm"] = interval(
            np.mean([errors[n] for n in low], axis=0) - np.mean([errors[n] for n in high], axis=0)
        )
    profile_mean = {model: {kind: float(np.mean([scores[n][model][kind] for n in scans]))
                            for kind in ("native", "continuous")}
                    for model in ("anchor", "rigid")}
    profile_mean["anchor_native_map_wins"] = sum(
        row["anchor"]["native"] < row["rigid"]["native"] for row in scores.values()
    )
    return {"maps": stats, "width_error": grouped, "profiles": scores, "pooled_profiles": profile_mean}


def error_curves(scans, keys, names, get_fits):
    """Score curves with one section set and paired spatial draws across candidates."""
    rng = np.random.default_rng(GEOMETRY_SEED)
    totals, resamples, by_map = {}, {}, {}
    for scan, record in scans.items():
        all_rows = [[get_fits(wall, key, model) for model in names] for wall in record["walls"] for key in keys]
        keep = np.array([qualified(*row) for row in all_rows]).reshape(-1, len(keys))
        take = np.flatnonzero(keep.all(axis=1))
        by_map[scan] = {"common": int(len(take)), "per_setting": dict(zip(keys, keep.sum(axis=0).tolist(), strict=True))}
        if not len(take):
            continue
        draw, blocks = block_draws(record["walls"], take, rng)
        widths = np.asarray([[fit["widths_nm"] for fit in row] for row in all_rows]).reshape(
            -1, len(keys), len(names), 2
        )[take]
        point = widths.mean(axis=0)
        averages = (draw @ widths.reshape(len(take), -1)).reshape(-1, len(keys), len(names), 2)
        totals[scan] = error_stats(point[:, :1], point[:, 1:])
        resamples[scan] = error_stats(averages[:, :, :1], averages[:, :, 1:])
        by_map[scan].update(blocks)
        by_map[scan]["width_error_nm"] = {k: curve.tolist() for k, curve in zip(names[1:], totals[scan].T, strict=True)}
    output = {"coverage": by_map, "settings": keys, "groups": {}}
    for group, maps in (("low", [m for m in scans if m in LOW]), ("high", [m for m in scans if m not in LOW]),
                        ("all", list(scans))):
        if not maps or any(m not in totals for m in maps):
            continue
        point = np.mean([totals[m] for m in maps], axis=0)
        dist = np.mean([resamples[m] for m in maps], axis=0)
        table = {"maps": maps, "sections": sum(by_map[m]["common"] for m in maps), "curves": {}}
        for i, name in enumerate(names[1:]):
            best = int(np.argmin(point[:, i]))
            table["curves"][name] = {
                "mean_curve_nm": point[:, i].tolist(), "pointwise_p05_p95_nm": interval(dist[:, :, i]),
                "sampled_minimum_nm": float(point[best, i]), "best_setting": keys[best],
            }
        table["contrasts"] = {}
        for prefix in ("", *(f"{c}/" for c in CAMERAS)):
            if prefix + "raman_shape" in names:
                a, b = (names.index(prefix + k) - 1 for k in FIXED[1:])
                table["contrasts"][prefix.rstrip("/") or "transferred_even"] = contrast(point, dist, keys, a, b)
        output["groups"][group] = table
    return output


def contrast(point, dist, keys, a, b):
    gap = point[:, a] - point[:, b]
    flips = []
    if all(isinstance(key, float) for key in keys):
        for i, (left, right) in enumerate(zip(gap[:-1], gap[1:], strict=True)):
            if left * right < 0:
                flips.append(float(keys[i] + (keys[i+1] - keys[i]) * -left / (right - left)))
    minima = np.argmin(point, axis=0)
    return {
        "raman_shape_minus_scale_curve_nm": gap.tolist(),
        "raman_shape_minus_scale_pointwise_p05_p95_nm": interval(dist[:, :, a] - dist[:, :, b]),
        "raman_best_setting_error_difference_nm": float(point[minima[a], a] - point[minima[b], b]),
        "raman_best_setting_error_difference_p05_p95_nm": interval(dist[:, minima[a], a] - dist[:, minima[b], b]),
        "interpolated_crossings": flips,
    }


def summarize(data):
    scans = dict(sorted(data["scans"].items(), key=lambda item: item[1]["twist_deg"]))
    report = {"native": {camera: geometry_and_headline(scans, camera) for camera in CAMERAS}}
    report["scans"] = {name: {key: row[key] for key in (
        "twist_deg", "polar_principal_strain_percent", "polar_mean_strain_percent", "barriers_meV_per_cell",
    )} for name, row in scans.items()}
    if data["geometry_only"]:
        return report

    def gaussian(wall, probe, arm):
        row = wall["gaussian"][f"{probe:g}"]
        if arm == "measured":
            return row["measured"]
        camera, model = arm.split("/")
        return row["models"][camera][model]

    def pupil(wall, defocus, model):
        if model == "measured":
            return wall["nominal"]["transferred_even"]["measured"]
        return wall["pupil"][str(defocus)][model]

    probes = data["gaussian_fwhm_nm"]
    report["gaussian"] = {}
    names = ("measured", *(f"{camera}/{k}" for camera in CAMERAS for k in FIXED))
    for endpoint in (1.75, 1.9, 2.2):
        report["gaussian"][f"{endpoint:g}"] = error_curves(scans, [p for p in probes if p <= endpoint], names, gaussian)
    family = ["raman_shape", *(f"raman_{fraction:g}" for fraction in data["raman_interpolation_fractions"]), "raman_scale"]
    report["raman_family"] = error_curves(scans, [p for p in probes if p <= 1.9],
                                          ("measured", *(f"transferred_even/{k}" for k in family)), gaussian)
    report["ideal_pupil"] = error_curves(scans, data["ideal_pupil_defocus_nm"], ("measured", *FIXED), pupil)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = args.input.read_bytes()
    data = json.loads(payload)
    if set(data["scans"]) != set(data["requested_scans"]):
        raise ValueError("The run has not finished all requested maps")
    result = summarize(data)
    result["wall_score_sha256"] = sha256(payload).hexdigest()
    result["block_settings"] = {"seed": GEOMETRY_SEED, "draws_per_origin": DRAW_COUNT, "origins": PHASES}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
