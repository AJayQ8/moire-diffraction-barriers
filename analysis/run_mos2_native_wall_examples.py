"""Replay the fixed MoS2 example walls and fit windows on the native raster."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from author_release import load_author_release
from observe_bilayers import read_input_metadata
from run_heterobilayer_transfer import digest
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_geometry_probe import make_spectra
from run_mos2_intensity_only_replay import disk_images
from run_mos2_observation_controls import (
    HALF_WINDOW, INPUT_NAMES, NOMINAL, gaussian_gain, raster_coordinates, render_traces, sample_cuts,
)
from run_mos2_opposite_widths import check_frozen
from run_mos2_wall_landscape import CROP, get_observer, width_pair
from run_mos2_wall_profile_closure import PROFILE_DISTANCE, transfer
from run_mos2_wall_width_transfer import DISTANCE, geometry_sections
from run_registry_prediction import rotation
from run_wall_heterostrain import SOURCE
from shared_compute import atomic_json, checkpoint_guard


TRANSFERRED = "transferred_even"
WINDOWS = ("adaptive", "4.25", "2.50")
ORDERS = ("first", "second")
LIBRARIES = (
    "bilayer_relaxation", "author_release", "run_moire_common_intensity",
    "run_mos2_geometry_probe", "run_mos2_observation_controls", "run_mos2_nonlinear_controls",
    "run_mos2_wall_landscape", "run_mos2_wall_profile_closure", "run_mos2_wall_width_transfer",
    "run_mos2_opposite_widths",
)


def plotted_example(section, legacy, observer, observed, traces, modes, widths):
    """Keep the original example walls, intensity crop and displayed spots."""
    if abs(section["spacing_nm"] - legacy["row_spacing_nm"]) > 1e-9:
        raise ValueError("The original example wall geometry changed")
    image = dict(legacy["first_ring_image"])
    distance = DISTANCE[observer.keep]
    np.testing.assert_allclose(image["midpoint_cut_distance_nm_full"], DISTANCE, rtol=0, atol=1e-10)
    image["midpoint_cut_displayed_indices"] = np.flatnonzero(observer.keep).tolist()
    image["midpoint_cut_profile_window_range_nm"] = [float(distance[0]), float(distance[-1])]
    image["midpoint_cut_display_note"] = "Central cut, shown within the fixed 4.25 nm half-window."
    target = observer.shape(observed)
    columns = []
    profiles = {}
    for index, order in enumerate(ORDERS):
        mode = legacy["profile_channels"][order]["representative_mode_hk"]
        channels = observer.channels[index]
        column, = np.flatnonzero(np.all(modes[channels] == mode, axis=1))
        columns.append(column)
        profiles[order] = {
            "representative_mode_hk": mode,
            "all_mode_hk": modes[channels].tolist(),
            "plotted_measured_unit_l2_shape": (target[index][:, column] * np.sqrt(len(channels))).tolist(),
        }

    curves = {}
    for name, trace in traces.items():
        first, second = transfer(observer, target, trace)
        distance_from_center = DISTANCE - first["center_nm"]
        shifted = np.column_stack([np.interp(distance_from_center, PROFILE_DISTANCE, spot) for spot in trace.T])
        shapes = observer.shape(shifted)
        curves[name] = {
            "position_transfer": [first, second],
            "unit_l2_shape_by_ring": {
                order: (shapes[index][:, columns[index]] * np.sqrt(len(observer.channels[index]))).tolist()
                for index, order in enumerate(ORDERS)
            },
        }
    return {
        "twist_deg": legacy["twist_deg"],
        "pixel_size_nm": legacy["pixel_size_nm"],
        "row_spacing_nm": section["spacing_nm"],
        "selected_center_indices": section["centers"],
        "first_ring_image": image,
        "profile_distance_nm": distance.tolist(),
        "profile_channels": profiles,
        "model_curves": curves,
        "measured_widths_nm": dict(zip(ORDERS, widths["measured"]["widths_nm"], strict=True)),
        "periodic_widths_nm": dict(zip(ORDERS, widths["models"][TRANSFERRED]["widths_nm"], strict=True)),
    }


def examine(context, images, modes, fields, original, control, detector, frozen, example):
    basis, q, rings = (context[key] for key in ("basis", "reciprocal", "rings"))
    geometry = geometry_sections(context, rotation(np.deg2rad(float(fields["SampleRotation"]))))
    if [section["centers"] for section in geometry] != [row["centers"] for row in original["walls"]]:
        raise ValueError("The frozen wall identities changed")
    spectra, _, solvers, _ = make_spectra(
        context, original, control["raman_constraint"]["models"], control["camera_diagnostic"]["by_model"],
        detector, geometry=False, geometry_only=True,
    )
    section_index = {tuple(row["centers"]): (row, ref) for row, ref in zip(geometry, original["walls"], strict=True)}
    mode_index = {tuple(mode): i for i, mode in enumerate(modes)}
    n = original["physics"]["grid"]
    pixel = context["receipt"]["pixel_nm"]
    results = []
    display = None
    for saved in frozen["walls"]:
        checkpoint_guard()
        section, ref = section_index[tuple(saved["centers"])]
        channels = [np.array([mode_index[tuple(mode)] for mode in ring]) for ring in ref["miller_indices"]]
        observed = sample_cuts(images, section["coordinates"]).mean(axis=0)
        freq2, cut, corners, weights = raster_coordinates(section, basis, pixel, n)
        gain = gaussian_gain(freq2, NOMINAL)
        traces = {
            camera: render_traces(cases["anchor"], gain, cut, corners, weights)[1]
            for camera, cases in spectra.items()
        }
        is_display = example is not None and section["centers"] == example["selected_center_indices"]
        windows = {}
        for name in WINDOWS:
            view = section if name == "adaptive" else {**section, "keep": abs(DISTANCE) <= float(name)}
            reader = get_observer(view, q, detector, rings, NOMINAL, channels)
            widths = {
                "measured": width_pair(reader, observed),
                "models": {camera: width_pair(reader, trace[CROP]) for camera, trace in traces.items()},
            }
            windows[name] = widths
            if name == f"{HALF_WINDOW:g}":
                for camera in spectra:
                    check_frozen(widths["measured"], saved["nominal"][camera]["measured"])
                    check_frozen(widths["models"][camera], saved["nominal"][camera]["models"]["anchor"]["native"])
                if is_display:
                    rigid = render_traces(spectra[TRANSFERRED]["rigid"], gain, cut, corners, weights)[1]
                    display = plotted_example(
                        section, example, reader, observed,
                        {"periodic_anchor": traces[TRANSFERRED], "rigid": rigid}, modes, widths,
                    )
                    for model, key in (("periodic_anchor", "anchor"), ("rigid", "rigid")):
                        scores = display["model_curves"][model]["position_transfer"]
                        np.testing.assert_allclose(
                            [scores[0]["error_by_ring"][1], scores[1]["error_by_ring"][0]],
                            saved["nominal"][TRANSFERRED]["models"][key]["native"]["transfer_error"],
                            rtol=0, atol=1e-8,
                        )
        results.append({"centers": saved["centers"], "windows": windows})
    if example is not None and display is None:
        raise ValueError("The original example wall is missing from the native comparison")
    return {"solvers": solvers, "walls": results}, display


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "release", "dependencies", "evidence", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    start = perf_counter()
    inputs = (SOURCE, BENCHMARK, SETTINGS, *(args.evidence / name for name in INPUT_NAMES),
              *(args.evidence / name for name in ("mos2_geometry_probe_v005.json", "measured_wall_examples_v01.json")))
    source, benchmark, settings, primary, control, optical, frozen, examples = (
        json.loads(path.read_text()) for path in inputs
    )
    path = Path(__file__)
    hashes = {p.name: digest(p) for p in (path, *inputs, *(path.parent / (name + ".py") for name in LIBRARIES))}
    hashes["release.zip"] = digest(args.release.parents[1] / "release.zip")
    report = {
        "schema": "mos2-native-wall-examples-v01",
        "input_sha256": hashes,
        "windows": WINDOWS,
        "probe_fwhm_nm": NOMINAL,
        "scans": {},
        "window_fits": {},
    }
    elapsed = 0.
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        report = json.loads(args.output.read_text())
        if report["input_sha256"] != hashes:
            raise ValueError("The checkpoint was created from different code or inputs")
        elapsed = report["seconds"]
    reader = load_author_release(args.release, args.dependencies)
    for name in sorted(frozen["scans"], key=lambda scan: int(scan[2:])):
        if name in report["window_fits"]:
            continue
        checkpoint_guard()
        metadata = read_input_metadata(name, args.archive)
        images, modes, checksum = disk_images(args.archive, name, reader)
        ref = source["scans"][name]
        if checksum != ref["diskset_sha256"] or metadata["archive_md5"] != benchmark["scans"][name]["archive_md5"]:
            raise ValueError(f"The diffraction input changed for {name}")
        fields = metadata["metadata_fields"]
        context = centers_and_context(images, modes, fields, settings, tuple(ref["interface"]))
        if context["receipt"] != ref["geometry"]:
            raise ValueError(f"The inferred geometry changed for {name}")
        even = optical["frozen_common_coefficients"]
        detector = np.broadcast_to([even[0], 0., even[1]], (len(context["reciprocal"]), 3))
        result, display = examine(
            context, images, np.asarray(modes), fields, primary["scans"][name], control["scans"][name],
            detector, frozen["scans"][name], examples["scans"].get(name),
        )
        report["window_fits"][name] = {"archive_md5": metadata["archive_md5"], "diskset_sha256": checksum, **result}
        if display is not None:
            report["scans"][name] = display
        report["seconds"] = elapsed + perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, report)
        print(f"{name}: {len(result['walls'])} sections", flush=True)


if __name__ == "__main__":
    main()
