"""Vary reflection selection and probe transfer on the frozen MoS2 wall sections."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from author_detector_replay import detector_intensity
from author_release import load_author_release
from observe_bilayers import read_input_metadata
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_circular_probe_check import circular_transfer
from run_mos2_geometry_probe import FIXED, LOW, make_spectra
from run_mos2_intensity_only_replay import disk_images
from run_mos2_observation_controls import (
    HALF_WINDOW, INPUT_NAMES, NOMINAL, gaussian_gain, raster_coordinates, render_traces, sample_cuts, score,
)
from run_mos2_opposite_widths import check_frozen
from run_mos2_wall_landscape import CROP, get_observer, width_pair
from run_mos2_wall_width_transfer import DISTANCE, geometry_sections, wall_disregistry
from run_registry_prediction import rotation
from run_wall_heterostrain import SOURCE
from shared_compute import atomic_json, checkpoint_guard


PERCENTAGES = (20, 30, 40)
GAUSSIANS = (1.25, 1.75)
FOCUS_STEPS = (0, 300, 400, 500, 600, 700, 800, 900, 1200)
BANDS = ("below_0.10", "0.10_to_0.30", "above_0.30")
BAND_EDGES = (0.10, 0.30)
BAND_TRANSITION = 0.025
DEFOCUS_FOR_BANDS = 900
TRANSFERRED = "transferred_even"
TABLE_FREQUENCY = np.linspace(0., 2., 2501)


def reflection_sets(section, q, compact, rings):
    field = wall_disregistry(np.array([0., 9., -9.]), 0., 2., section["jump"], "sine_gordon")
    intensity = detector_intensity(field, q, compact)
    contrast = np.max(abs(intensity[1:] - intensity[0]), axis=0)
    sets = {}
    for percent in PERCENTAGES:
        sets[str(percent)] = []
        for order in ("first", "second"):
            indices = np.flatnonzero(rings[order])
            significant = contrast[indices] > (percent / 100) * contrast[indices].max()
            if not np.any(significant):
                raise ValueError("No contrast-qualified reflections in an order")
            sets[str(percent)].append(indices[significant])
    return sets


def band_weights(frequency):
    steps = []
    for edge in BAND_EDGES:
        fraction = np.clip((frequency - edge + BAND_TRANSITION) / (2 * BAND_TRANSITION), 0, 1)
        steps.append(.5 - .5 * np.cos(np.pi * fraction))
    return 1 - steps[0], steps[0] - steps[1], steps[1]


def pupil_table():
    by_focus = {str(z): circular_transfer(TABLE_FREQUENCY, z) for z in FOCUS_STEPS}
    delta = by_focus[str(DEFOCUS_FOR_BANDS)] - by_focus["0"]
    weights = band_weights(TABLE_FREQUENCY)
    np.testing.assert_allclose(np.sum(weights, axis=0), 1., atol=1e-14)
    return {**by_focus, **{name: by_focus["0"] + w * delta for name, w in zip(BANDS, weights, strict=True)}}


def gaussian_fits(section, q, rings, detector, observed, spectra, sets, geometry, frozen):
    freq2, cut, corners, weights = geometry
    output = {percent: {} for percent in sets}
    for probe in GAUSSIANS:
        key = f"{probe:g}"
        names = (*FIXED, "rigid") if probe == NOMINAL else FIXED
        gain = gaussian_gain(freq2, probe)
        traces = {camera: {name: render_traces(values[name], gain, cut, corners, weights)[1] for name in names}
                  for camera, values in spectra.items()}
        for percent, channels in sets.items():
            observer = get_observer(section, q, detector, rings, probe, channels)
            target = observer.shape(observed)
            row = {"measured": width_pair(observer, observed), "models": {}}
            for camera, cases in traces.items():
                fits = {}
                for name, trace in cases.items():
                    if probe == NOMINAL and name in ("anchor", "rigid"):
                        fits[name] = score(observer, target, trace)
                    else:
                        fits[name] = width_pair(observer, trace[CROP])
                    if percent == "30":
                        old = frozen["gaussian"][key]["models"][camera][name] if name != "rigid" else (
                            frozen["nominal"][camera]["models"]["rigid"]["native"])
                        check_frozen(fits[name], old)
                        if "transfer_error" in fits[name]:
                            np.testing.assert_allclose(fits[name]["transfer_error"], old["transfer_error"], atol=1e-8)
                row["models"][camera] = fits
            if percent == "30":
                check_frozen(row["measured"], frozen["gaussian"][key]["measured"])
            output[percent][key] = row
    return output


def pupil_fits(section, q, rings, detector, spectra, channels, geometry, table, frozen):
    freq2, cut, corners, weights = geometry
    frequency = np.sqrt(freq2) / (2 * np.pi)
    observer = get_observer(section, q, detector, rings, NOMINAL, channels)
    output = {}
    for setting, response in table.items():
        gain = np.interp(frequency, TABLE_FREQUENCY, response, right=0.)
        fits = {}
        for model in FIXED:
            _, trace = render_traces(spectra[model], gain, cut, corners, weights)
            fits[model] = width_pair(observer, trace[CROP])
            if setting in frozen["pupil"]:
                check_frozen(fits[model], frozen["pupil"][setting][model])
        output[setting] = fits
    return output


def examine(scan, context, images, modes, angle, primary, control, optical, old, table):
    basis, q, rings = (context[k] for k in ("basis", "reciprocal", "rings"))
    sections = geometry_sections(context, rotation(angle))
    references = {tuple(w["centers"]): w for w in primary["walls"]}
    sections = {tuple(s["centers"]): s for s in sections}
    n = primary["physics"]["grid"]
    pixel = primary["pixel_nm"]
    even = optical["frozen_common_coefficients"]
    detector = np.broadcast_to([even[0], 0., even[1]], (len(q), 3))
    compact = np.asarray(primary["cameras"]["compact"]["channels"])
    spectra, _, solvers, barriers = make_spectra(
        context, primary, control["raman_constraint"]["models"], control["camera_diagnostic"]["by_model"],
        detector, False, False)
    walls = []
    for frozen in old["walls"]:
        checkpoint_guard()
        index = tuple(frozen["centers"])
        section = sections[index]
        ref = references[index]
        sets = reflection_sets(section, q, compact, rings)
        if [modes[channels].tolist() for channels in sets["30"]] != ref["miller_indices"]:
            raise ValueError("The nominal reflection selection changed")
        geometry = raster_coordinates(section, basis, pixel, n)
        observed = sample_cuts(images, section["coordinates"]).mean(axis=0)
        section = {**section, "keep": abs(DISTANCE) <= HALF_WINDOW}
        item = {key: frozen[key] for key in ("centers", "midpoint_xy_nm", "spacing_nm")}
        item["miller_indices"] = {p: [modes[channels].tolist() for channels in selected]
                                  for p, selected in sets.items()}
        item["cutoff"] = gaussian_fits(section, q, rings, detector, observed, spectra, sets, geometry, frozen)
        if scan not in LOW:
            item["pupil"] = pupil_fits(section, q, rings, detector, spectra[TRANSFERRED], sets["30"],
                                       geometry, table, frozen)
        walls.append(item)
    return {"twist_deg": old["twist_deg"], "solvers": solvers, "barriers_meV_per_cell": barriers, "walls": walls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "release", "dependencies", "evidence", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    paths = (SOURCE, BENCHMARK, SETTINGS, *(args.evidence / n for n in INPUT_NAMES),
             args.evidence / "mos2_geometry_probe_v005.json")
    source, benchmark, settings, primary, control, optical, previous = [json.loads(p.read_bytes()) for p in paths]
    modules = ("run_mos2_geometry_probe.py", "run_mos2_wall_width_transfer.py", "run_mos2_circular_probe_check.py",
               "run_mos2_observation_controls.py", "run_mos2_opposite_widths.py", "run_mos2_wall_landscape.py",
               "run_mos2_wall_profile_closure.py", "run_mos2_nonlinear_controls.py", "bilayer_relaxation.py",
               "author_detector_replay.py")
    hashes = {p.name: digest(p) for p in (Path(__file__), *paths, *(PROJECT / module for module in modules))}
    table = pupil_table()
    output = {"input_sha256": hashes, "cutoffs_percent": list(PERCENTAGES), "gaussian_fwhm_nm": list(GAUSSIANS),
              "defocus_nm": list(FOCUS_STEPS), "frequency_bands": list(BANDS), "band_edges_cycles_nm": list(BAND_EDGES),
              "band_transition_half_width_cycles_nm": BAND_TRANSITION, "band_target_defocus_nm": DEFOCUS_FOR_BANDS,
              "pupil_transfer_at_0.2_cycles_per_nm": {
                  key: float(np.interp(.2, TABLE_FREQUENCY, transfer)) for key, transfer in table.items()},
              "lower_twist_scans": list(LOW), "requested_scans": list(previous["scans"]), "scans": {},
              "scope": "Retrospective controls on frozen sections, mechanical inputs, rasters and detector responses. "
                       "Fit cutoffs of 20, 30 and 40 percent of calculated contrast. In the six higher-twist scans "
                       "compare fixed optical responses with the same 1.25 nm Gaussian width reader. Replace "
                       "transfer in one smooth frequency band at a time as a sensitivity test."}
    elapsed = 0.
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        output = json.loads(args.output.read_bytes())
        if output["input_sha256"] != hashes:
            raise ValueError("Resume requires the original code and evidence")
        elapsed = output["seconds"]
    author = load_author_release(args.release, args.dependencies)
    for scan in output["requested_scans"]:
        if scan in output["scans"]:
            continue
        checkpoint_guard()
        metadata = read_input_metadata(scan, args.archive)
        images, modes, checksum = disk_images(args.archive, scan, author)
        ref = source["scans"][scan]
        if checksum != ref["diskset_sha256"] or metadata["archive_md5"] != benchmark["scans"][scan]["archive_md5"]:
            raise ValueError(f"The public input changed for {scan}")
        fields = metadata["metadata_fields"]
        context = centers_and_context(images, modes, fields, settings, tuple(ref["interface"]))
        if context["receipt"] != ref["geometry"]:
            raise ValueError(f"The measured geometry changed for {scan}")
        angle = np.deg2rad(float(fields["SampleRotation"]))
        output["scans"][scan] = examine(scan, context, images, modes, angle, primary["scans"][scan],
                                         control["scans"][scan], optical, previous["scans"][scan], table)
        output["seconds"] = elapsed + perf_counter() - started
        checkpoint_guard()
        atomic_json(args.output, output)
        print(scan, len(output["scans"][scan]["walls"]), round(output["seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
