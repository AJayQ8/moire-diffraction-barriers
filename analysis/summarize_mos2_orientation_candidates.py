"""Rebuild the paired ten-map orientation/barrier scores and Bragg-radius audit."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
import pickle
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from summarize_mos2_geometry_probe import CAMERAS, DRAW_COUNT, PHASES, block_draws, interval, qualified


PAPER_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PAPER_DIR / "evidence" / "mos2_saddle_control_v01.json"
DEFAULT_OUTPUT = PAPER_DIR / "evidence" / "mos2_orientation_candidates_v01.json"
SCAN_IDS = ("DS1", "DS2", "DS3", "DS4", "DS6", "DS9", "DS10", "DS12", "DS13", "DS14")
LOW_TWIST_MAPS = ("DS13", "DS14")
PRIMARY = "archive_13.2"
ALTERNATE = "two_rotated_10"
SEED = 20261116
SOURCE_DOI = "10.5281/zenodo.7865970"
READER_COMMIT = "85f21dd9134b600000e517ad1b22acc725571376"
CANDIDATES = (
    {"name": "archive_13.2", "frame": "nominal", "barrier_meV_per_cell": 13.2},
    {"name": "archive_10", "frame": "nominal", "barrier_meV_per_cell": 10.0},
    {"name": "two_rotated_13.2", "frame": "selective_DS13_DS14_rotation", "barrier_meV_per_cell": 13.2},
    {"name": "two_rotated_10", "frame": "selective_DS13_DS14_rotation", "barrier_meV_per_cell": 10.0},
)


def file_sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def payload_sha256(payload):
    return sha256(payload).hexdigest()


def candidate_fit(wall, camera, candidate):
    """Use rotated rows where explicitly saved, otherwise reuse the nominal row."""
    geometry = "nominal"
    if candidate["frame"] == "selective_DS13_DS14_rotation" and "rotated" in wall["arms"][camera]:
        geometry = "rotated"
    barrier_key = f"{candidate['barrier_meV_per_cell']:g}"
    return wall["arms"][camera][geometry][barrier_key]


def summarize_widths(data, input_path):
    if set(data["scans"]) != set(data["requested_scans"]):
        raise ValueError("The saved saddle-control input does not contain all requested maps")
    if set(data["scans"]) != set(SCAN_IDS):
        raise ValueError(f"Expected the ten scored maps {SCAN_IDS}, got {tuple(data['scans'])}")

    rng = np.random.default_rng(SEED)
    map_rows = {}
    map_draws = {camera: {candidate["name"]: {} for candidate in CANDIDATES} for camera in CAMERAS}
    map_mae_draws = {camera: {candidate["name"]: {} for candidate in CANDIDATES} for camera in CAMERAS}
    selected_total = 0
    total_source_walls = 0

    for scan in data["requested_scans"]:
        record = data["scans"][scan]
        walls = record["walls"]
        total_source_walls += len(walls)
        retained = []
        for index, wall in enumerate(walls):
            fits = [wall["measured"]]
            fits.extend(candidate_fit(wall, camera, candidate)
                        for camera in CAMERAS for candidate in CANDIDATES)
            if qualified(*fits):
                retained.append(index)
        if not retained:
            raise ValueError(f"No common qualified widths in {scan}")

        weights, block_info = block_draws(walls, retained, rng)
        measured = np.asarray([walls[i]["measured"]["widths_nm"] for i in retained], dtype=float)
        measured_mean = measured.mean(axis=0)
        measured_draws = weights @ measured
        selected_total += len(retained)

        current = {
            "source_wall_count": len(walls),
            "common_qualified_section_count": len(retained),
            "common_qualified_source_indices_zero_based": retained,
            "measured_mean_width_nm": measured_mean.tolist(),
            "block": block_info,
            "cameras": {},
        }

        for camera in CAMERAS:
            current["cameras"][camera] = {"candidates": {}}
            candidate_errors = {}
            candidate_error_draws = {}
            candidate_maes = {}
            candidate_mae_draws = {}
            for candidate in CANDIDATES:
                name = candidate["name"]
                predictions = np.asarray([
                    candidate_fit(walls[i], camera, candidate)["widths_nm"] for i in retained
                ], dtype=float)
                predicted_mean = predictions.mean(axis=0)
                predicted_draws = weights @ predictions
                residual = predicted_mean - measured_mean
                point_score = float(np.mean(np.abs(residual)))
                draw_scores = np.mean(np.abs(predicted_draws - measured_draws), axis=1)
                wall_mae_values = np.mean(np.abs(predictions - measured), axis=1)
                wall_mae_draws = weights @ wall_mae_values
                row = {
                    "frame": candidate["frame"],
                    "barrier_meV_per_cell": candidate["barrier_meV_per_cell"],
                    "measured_mean_width_nm": measured_mean.tolist(),
                    "predicted_mean_width_nm": predicted_mean.tolist(),
                    "predicted_minus_measured_mean_nm": residual.tolist(),
                    "absolute_order_residual_nm": np.abs(residual).tolist(),
                    "equal_order_map_score_nm": point_score,
                    "equal_order_map_score_p05_p95_nm": interval(draw_scores),
                    "wall_order_mae_nm": float(wall_mae_values.mean()),
                    "wall_order_mae_p05_p95_nm": interval(wall_mae_draws),
                }
                current["cameras"][camera]["candidates"][name] = row
                candidate_errors[name] = point_score
                candidate_error_draws[name] = draw_scores
                candidate_maes[name] = float(wall_mae_values.mean())
                candidate_mae_draws[name] = wall_mae_draws
                map_draws[camera][name][scan] = draw_scores
                map_mae_draws[camera][name][scan] = wall_mae_draws

            score_delta = candidate_errors[ALTERNATE] - candidate_errors[PRIMARY]
            draw_delta = candidate_error_draws[ALTERNATE] - candidate_error_draws[PRIMARY]
            mae_delta = candidate_maes[ALTERNATE] - candidate_maes[PRIMARY]
            mae_draw_delta = candidate_mae_draws[ALTERNATE] - candidate_mae_draws[PRIMARY]
            current["cameras"][camera]["primary_vs_alternate"] = {
                "primary": PRIMARY,
                "alternate": ALTERNATE,
                "alternate_minus_primary_equal_order_score_nm": float(score_delta),
                "alternate_minus_primary_equal_order_score_p05_p95_nm": interval(draw_delta),
                "alternate_minus_primary_wall_order_mae_nm": float(mae_delta),
                "alternate_minus_primary_wall_order_mae_p05_p95_nm": interval(mae_draw_delta),
            }
        map_rows[scan] = current

    if selected_total != 242:
        raise ValueError(f"Expected 242 common qualified sections, selected {selected_total}")

    overall = {}
    for camera in CAMERAS:
        scores = {}
        for candidate in CANDIDATES:
            name = candidate["name"]
            map_points = {
                scan: map_rows[scan]["cameras"][camera]["candidates"][name]["equal_order_map_score_nm"]
                for scan in data["requested_scans"]
            }
            point = float(np.mean(list(map_points.values())))
            draws = np.mean([map_draws[camera][name][scan] for scan in data["requested_scans"]], axis=0)
            scores[name] = {
                "equal_map_score_nm": point,
                "equal_map_score_p05_p95_nm": interval(draws),
                "map_scores_nm": map_points,
                "leave_one_map_out_equal_map_score_nm": {
                    scan: float(np.mean([value for other, value in map_points.items() if other != scan]))
                    for scan in data["requested_scans"]
                },
            }

        point_diff = scores[ALTERNATE]["equal_map_score_nm"] - scores[PRIMARY]["equal_map_score_nm"]
        diff_draws = np.mean([
            map_draws[camera][ALTERNATE][scan] - map_draws[camera][PRIMARY][scan]
            for scan in data["requested_scans"]
        ], axis=0)
        mae_points = {
            candidate["name"]: {
                scan: map_rows[scan]["cameras"][camera]["candidates"][candidate["name"]]["wall_order_mae_nm"]
                for scan in data["requested_scans"]
            }
            for candidate in CANDIDATES
        }
        mae_diffs = {
            scan: mae_points[ALTERNATE][scan] - mae_points[PRIMARY][scan]
            for scan in data["requested_scans"]
        }
        mae_diff_draws = np.mean([
            map_mae_draws[camera][ALTERNATE][scan] - map_mae_draws[camera][PRIMARY][scan]
            for scan in data["requested_scans"]
        ], axis=0)
        loo_diff = {
            scan: float(np.mean([value for other, value in {
                name: scores[ALTERNATE]["map_scores_nm"][name] - scores[PRIMARY]["map_scores_nm"][name]
                for name in data["requested_scans"]
            }.items() if other != scan]))
            for scan in data["requested_scans"]
        }
        loo_diff_interval = {}
        for omitted in data["requested_scans"]:
            loo = np.mean([
                map_draws[camera][ALTERNATE][scan] - map_draws[camera][PRIMARY][scan]
                for scan in data["requested_scans"] if scan != omitted
            ], axis=0)
            loo_diff_interval[omitted] = interval(loo)

        overall[camera] = {
            "equal_map_scores": scores,
            "primary_vs_alternate": {
                "primary": PRIMARY,
                "alternate": ALTERNATE,
                "alternate_minus_primary_equal_map_score_nm": float(point_diff),
                "alternate_minus_primary_equal_map_score_p05_p95_nm": interval(diff_draws),
                "leave_one_map_out_alternate_minus_primary_nm": loo_diff,
                "leave_one_map_out_alternate_minus_primary_p05_p95_nm": loo_diff_interval,
                "equal_map_wall_order_mae_nm": {
                    name: float(np.mean(list(values.values()))) for name, values in mae_points.items()
                },
                "per_map_wall_order_mae_nm": mae_points,
                "per_map_alternate_minus_primary_wall_order_mae_nm": mae_diffs,
                "equal_map_alternate_minus_primary_wall_order_mae_nm": float(np.mean(list(mae_diffs.values()))),
                "equal_map_alternate_minus_primary_wall_order_mae_p05_p95_nm": interval(mae_diff_draws),
            },
        }

    return {
        "source": "mos2_saddle_control_v01.json",
        "input_sha256": file_sha256(input_path),
        "maps": list(data["requested_scans"]),
        "source_wall_count": total_source_walls,
        "common_qualified_sections": selected_total,
        "candidate_definitions": list(CANDIDATES),
        "score_definition": "For each map and candidate, average over the two orders of abs(mean predicted width - mean measured width); all-map scores average the ten map scores equally.",
        "qualification": "Common section must qualify on measured widths and all four candidate/frame fits under both camera calibrations, using the shared geometry-probe qualified() helper.",
        "primary_vs_alternate_definition": "two_rotated_10 minus archive_13.2; rotated geometry is used for DS13/DS14 where present, while other maps use the archived nominal geometry.",
        "spatial_resampling": {
            "helper": "summarize_mos2_geometry_probe.block_draws and interval",
            "seed": SEED,
            "total_draws_per_map": int(DRAW_COUNT * len(PHASES)),
            "origins": [list(phase) for phase in PHASES],
            "draws_per_origin": int(DRAW_COUNT),
            "draw_policy": "Each map uses a common spatial block draw matrix for every candidate and both measured orders; one seeded RNG is consumed in requested_scans order.",
        },
        "overall": overall,
        "per_map": map_rows,
    }


class InertDiskset:
    """Inert target for the numeric state of the archived DiskSet object."""


class RestrictedUnpickler(pickle.Unpickler):
    """Accept only NumPy reconstruction globals and the exact DiskSet class."""

    def find_class(self, module, name):
        if (module, name) == ("diskset", "DiskSet"):
            return InertDiskset
        if (module, name) in (("numpy", "ndarray"), ("numpy", "dtype")):
            return getattr(np, name)
        if module == "numpy.core.multiarray" and name in ("_reconstruct", "scalar"):
            from numpy.core import multiarray
            return getattr(multiarray, name)
        raise pickle.UnpicklingError(f"Rejected unexpected pickle global {module}.{name}")


def parse_info(payload):
    fields = {}
    for line in payload.decode("utf-8", "replace").splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.split("#", 1)[0].strip()
    return fields


def read_bragg_scan(scan, archive_dir):
    archive_path = Path(archive_dir) / f"{scan}.zip"
    with ZipFile(archive_path) as archive:
        fit_bytes = archive.read(f"{scan}/fit.pkl")
        info_bytes = archive.read(f"{scan}/info.txt")
    record = RestrictedUnpickler(BytesIO(fit_bytes)).load()
    if not isinstance(record, (list, tuple)) or len(record) < 4:
        raise ValueError(f"Unexpected {scan}/fit.pkl structure")
    disk = record[0]
    required = ("_xvec", "_yvec", "_in_use")
    if not all(hasattr(disk, key) for key in required):
        raise ValueError(f"{scan}/fit.pkl lacks expected DiskSet coordinates")

    active = np.asarray(disk._in_use, dtype=bool)
    xy = np.column_stack((np.asarray(disk._xvec, dtype=float)[active],
                          np.asarray(disk._yvec, dtype=float)[active]))
    if xy.shape != (12, 2) or not np.isfinite(xy).all():
        raise ValueError(f"Expected twelve finite active disk centers in {scan}, got {xy.shape}")
    center = xy.mean(axis=0)
    radius = np.linalg.norm(xy - center, axis=1)
    normalized = radius / radius.min()
    ring_masks = {
        "first": np.abs(normalized - 1.0) < 0.5,
        "second": np.abs(normalized - np.sqrt(3.0)) < 0.5,
    }
    if any(int(mask.sum()) != 6 for mask in ring_masks.values()):
        raise ValueError(f"Pinned ring convention did not identify six disks per order in {scan}")

    info = parse_info(info_bytes)
    rings = {}
    for key, mask in ring_masks.items():
        values = np.sort(radius[mask])
        rings[key] = {
            "count": int(len(values)),
            "radii_detector_px_sorted": values.tolist(),
            "median_detector_px": float(np.median(values)),
            "mean_detector_px": float(values.mean()),
            "min_max_detector_px": [float(values.min()), float(values.max())],
        }
    return {
        "fit_pkl_record_length": len(record),
        "active_disk_count": int(active.sum()),
        "center_detector_px_xy": center.tolist(),
        "first_to_second_median_radius_ratio": rings["second"]["median_detector_px"] / rings["first"]["median_detector_px"],
        "rings": rings,
        "metadata": {
            "K3toHAADFRotation_Used": info.get("K3toHAADFRotation_Used"),
            "SampleRotation": info.get("SampleRotation"),
            "PixelSize": info.get("PixelSize"),
            "ProbeUsed": info.get("ProbeUsed"),
            "OriginalName": info.get("OriginalName"),
            "camera_length_fields_present": [key for key in info if "camera" in key.lower() or "detector" in key.lower()],
        },
        "sha256": {
            "zip": file_sha256(archive_path),
            "fit_pkl_member": payload_sha256(fit_bytes),
            "info_txt_member": payload_sha256(info_bytes),
        },
    }


def describe_group(rows, scans):
    result = {}
    for ring in ("first", "second"):
        values = np.asarray([rows[scan]["rings"][ring]["median_detector_px"] for scan in scans])
        result[ring] = {
            "map_median_radii_detector_px": {scan: float(value) for scan, value in zip(scans, values)},
            "mean_of_map_medians_detector_px": float(values.mean()),
            "sample_sd_across_maps_detector_px": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min_max_map_medians_detector_px": [float(values.min()), float(values.max())],
        }
    return result


def summarize_bragg(archive_dir, reader_path):
    rows = {scan: read_bragg_scan(scan, archive_dir) for scan in SCAN_IDS}
    focal = list(LOW_TWIST_MAPS)
    comparison = [scan for scan in SCAN_IDS if scan not in LOW_TWIST_MAPS]
    focal_summary = describe_group(rows, focal)
    comparison_summary = describe_group(rows, comparison)
    differences = {}
    for ring in ("first", "second"):
        focal_mean = focal_summary[ring]["mean_of_map_medians_detector_px"]
        comparison_mean = comparison_summary[ring]["mean_of_map_medians_detector_px"]
        differences[ring] = {
            "DS13_DS14_minus_other_eight_detector_px": focal_mean - comparison_mean,
            "percent_of_other_eight_mean": 100.0 * (focal_mean - comparison_mean) / comparison_mean,
            "groups_nonoverlap": (
                focal_summary[ring]["min_max_map_medians_detector_px"][1]
                < comparison_summary[ring]["min_max_map_medians_detector_px"][0]
            ),
        }
    shared = {}
    for key in ("K3toHAADFRotation_Used", "ProbeUsed", "OriginalName", "PixelSize"):
        shared[key] = sorted({rows[scan]["metadata"][key] for scan in SCAN_IDS}, key=str)
    return {
        "source_doi": SOURCE_DOI,
        "archive_contents": "DS1.zip, DS2.zip, DS3.zip, DS4.zip, DS6.zip, DS9.zip, DS10.zip, DS12.zip, DS13.zip, DS14.zip",
        "reader": {
            "repository": "bediakolab/pyInterferometry",
            "commit": READER_COMMIT,
            "diskset_source_path": "src/diskset.py",
            "diskset_source_sha256": file_sha256(reader_path),
            "radius_convention": "Pinned DiskSet.set_com_central uses the mean of active x/y centers. DiskSet.d_set normalizes all distances by the smallest radius; determine_rings(tol=0.5) assigns first ring within 0.5 of ratio 1 and second ring within 0.5 of sqrt(3). The x-coordinate sign flip in d_set does not affect radii.",
            "unpickler": "Only diskset.DiskSet is mapped to an inert holder; NumPy ndarray/dtype and NumPy multiarray _reconstruct/scalar globals are allowed. All other pickle globals are rejected.",
        },
        "scans": rows,
        "group_comparison": {
            "DS13_DS14": focal_summary,
            "other_eight": comparison_summary,
            "DS13_DS14_minus_other_eight": differences,
            "DS1_DS2_camera_metadata": {
                "radii": {scan: rows[scan]["rings"] for scan in ("DS1", "DS2")},
                "metadata": {scan: rows[scan]["metadata"] for scan in ("DS1", "DS2")},
                "distinct_camera_length_or_scan_matched_calibration_recorded": False,
            },
        },
        "shared_metadata_values_across_all_ten": shared,
        "metadata_fields_absent_in_all_ten": ["CameraLength", "DetectorPixelSize", "AcceleratingVoltage"],
        "interpretation_limit": "The repeated −11.3° archive correction is not a per-scan measurement; the 191° paper calibration has no scan ID in the public Methods. Bragg radii suggest a modest scale difference for DS13/DS14, but are not an independent camera-length calibration and must not be used to infer a true orientation/frame rotation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Saved saddle-control JSON")
    parser.add_argument("--archive-dir", type=Path, required=True,
                        help="Directory containing the public DS*.zip files")
    parser.add_argument("--reader-source", type=Path, required=True,
                        help="Pinned pyInterferometry src/diskset.py file, used for provenance hashing")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Combined JSON output path")
    args = parser.parse_args()

    input_bytes = args.input.read_bytes()
    data = json.loads(input_bytes)
    width_summary = summarize_widths(data, args.input)
    bragg_summary = summarize_bragg(args.archive_dir, args.reader_source)
    helper_path = Path(__file__).with_name("summarize_mos2_geometry_probe.py")
    result = {
        "schema": "mos2_orientation_candidates_v01",
        "inputs": {
            "saddle_control": {"path": args.input.name, "sha256": payload_sha256(input_bytes)},
            "geometry_probe_helper": {"path": "analysis/summarize_mos2_geometry_probe.py", "sha256": file_sha256(helper_path)},
            "summarizer": {"path": "analysis/summarize_mos2_orientation_candidates.py", "sha256": file_sha256(__file__)},
            "bragg_archive_directory_argument": "--archive-dir",
            "pinned_reader_source_argument": "--reader-source",
        },
        "width_comparison": width_summary,
        "bragg_radius_audit": bragg_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
