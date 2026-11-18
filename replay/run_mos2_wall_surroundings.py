"""Identify which registry changes away from an isolated wall distort its readout."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from author_release import load_author_release
from bilayer_relaxation import MODES, STACKING, elastic_parameters_per_reference_cell, sample_disregistry
from observe_bilayers import read_input_metadata
from run_heterobilayer_sparse_geometry import solve_from_geometry
from run_heterobilayer_template_transfer import design_matrix
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_directional_diffraction import wall_sectors
from run_mos2_intensity_only_replay import disk_images
from run_mos2_slow_readout_transfer import calibrate, render
from run_mos2_wall_profile_closure import (PROFILE_DISTANCE, WIDTHS, isolated_field, render_profiles,
                                           transfer)
from run_mos2_wall_width_transfer import DISTANCE, REDUCED, Profiles, actual_profiles, geometry_sections
from run_registry_prediction import rotation
from run_wall_heterostrain import PROTOCOL, SOURCE
from shared_compute import atomic_json, checkpoint_guard


PREVIOUS = PROJECT / "mos2_wall_profile_closure_v01.json"
PROBE = 1.25
WINDOWS_NM = (1.5, 2.5, 3.5)


def fields(section, context, grid, motion, material):
    sigma = PROBE / np.sqrt(8*np.log(2))
    points, weights = np.polynomial.hermite.hermgauss(5)
    weights /= np.sqrt(np.pi)
    q0 = len(DISTANCE)//2
    origin = section["phase"][:, q0]
    slope = (section["phase"][0, q0+1] - origin[0])/(DISTANCE[1]-DISTANCE[0])
    tangent = section["jump"]/section["length_nm"]
    span = np.sqrt(2.)*sigma*points
    phase = (origin[:, None, None, :] + span[None, :, None, None]*tangent
             + PROFILE_DISTANCE[None, None, :, None]*slope)
    full = sample_disregistry(grid, motion, phase, grid.cell)
    isolated = isolated_field(PROFILE_DISTANCE, section["jump"], material, context["basis"],
                              section["slip_sign"], section["k_factor"])
    base = np.broadcast_to(isolated, phase.shape)
    difference = full - base
    unit = section["jump"]/np.linalg.norm(section["jump"])
    across_slip = (difference @ unit)[..., None]*unit
    candidates = {"full": full, "isolated": base, "registry_along_slip": full - across_slip,
                  "registry_perpendicular_to_slip": base + across_slip}
    for distance in WINDOWS_NM:
        # Keep the same unit-slip core and add the finite-cell difference
        # only in the wall surroundings, or keep it only in the core.
        outside = .5*(1 + np.tanh((abs(PROFILE_DISTANCE) - distance)/.45))
        correction = outside[None, None, :, None]*difference
        candidates[f"surroundings_{distance:g}nm"] = base + correction
        candidates[f"core_{distance:g}nm"] = full - correction
    return candidates, weights


def evaluate(context, images, meta, interface, compact, widths, reference):
    sections = geometry_sections(context, rotation(np.deg2rad(float(meta["SampleRotation"]))))
    basis, gradient, nominal = (context[k] for k in ("basis", "gradient", "rigid"))
    q, support = context["reciprocal"], context["support"]
    phase = nominal + design_matrix(nominal, basis, MODES) @ compact
    train = support & ~np.logical_or.reduce(tuple(wall_sectors(nominal, basis).values()))
    normalized, camera = calibrate(render(phase, np.ones(support.shape), q, context["probe"]), images, train)
    grid, motion, _ = solve_from_geometry(gradient, basis, interface)
    material = (STACKING[interface], elastic_parameters_per_reference_cell(*interface[:2]))
    records = []
    for section, measured, old in zip(sections, widths["walls"], reference["walls"], strict=True):
        if section["centers"] != old["centers"] or old["centers"] != measured["centers"]:
            raise ValueError("Mixed wall chords in the comparison")
        receipt = measured["fits"]["1.25"]
        fits = (receipt["joint"], *receipt["separate"], *receipt["synthetic"].values())
        if not all(f["qualified"] for f in fits):
            continue
        checkpoint_guard()
        candidates, weights = fields(section, context, grid, motion, material)
        predicted = render_profiles(candidates, q, camera, weights, PROBE)
        observer = Profiles(section, q, camera, context["rings"], PROBE, "sine_gordon")
        target = observer.shape(actual_profiles(normalized, section["coordinates"]))
        results = {}
        for name, signal in predicted.items():
            central = signal[np.abs(PROFILE_DISTANCE) <= DISTANCE[-1]]
            shape = observer.shape(central)
            widths_by_ring = [observer.fit(shape, [i]) for i in (0, 1)]
            fit = transfer(observer, target, signal)
            results[name] = {"width_difference_nm": widths_by_ring[0]["width_nm"] - widths_by_ring[1]["width_nm"],
                             "widths_qualified": all(a["qualified"] for a in widths_by_ring), "transfer": fit}
        for name, prior in (("full", "periodic_anchor"), ("isolated", "isolated_anchor")):
            old_fit = old["probe"]["1.25"][prior]
            if any(abs(a["error_by_ring"][1-i] - b["error_by_ring"][1-i])>1e-10
                   for i, (a, b) in enumerate(zip(results[name]["transfer"], old_fit["transfer"], strict=True))):
                raise ValueError("The unmodified wall profile did not reproduce")
        records.append({"centers": section["centers"], "k_factor": section["k_factor"],
                        "measured_width_difference_nm": receipt["separate"][0]["width_nm"] - receipt["separate"][1]["width_nm"],
                        "models": results})
    return records


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("release", type=Path)
    cli.add_argument("dependencies", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    started = perf_counter()
    inputs = PROTOCOL, SOURCE, BENCHMARK, SETTINGS, REDUCED, WIDTHS, PREVIOUS
    protocol, source, bench, settings, reduced, widths, previous = (json.loads(p.read_text()) for p in inputs)
    author = load_author_release(args.release, args.dependencies)
    result = {"scope": "Add the periodic-minus-isolated registry correction either along the slip or perpendicular to it, and either inside or outside nominal wall-core radii of 1.5, 2.5 and 3.5 nm. Use frozen material inputs and the same chord, detector, probe and per-model position fit. Compare with the original isolated wall and the full periodic network. Report the apparent width difference in the synthetic profiles and the held-out intensity profile error.",
              "limits": "The components are frozen diagnostic fields rather than additional equilibria. A normal or tangential component in registry space is not a spatial direction on the image. Blending wall cores and surroundings does not impose force balance at the blend. All comparisons are retrospective and the two rings share the same acquisition.",
              "input_sha256": {p.name: digest(p) for p in (Path(__file__), *inputs,
                                                           PROJECT/"run_mos2_wall_profile_closure.py")}, "scans": {}}
    for scan in protocol["cases"]["parallel_comparison"]:
        checkpoint_guard()
        saved = source["scans"][scan]
        metadata = read_input_metadata(scan, args.archive)
        images, modes, sha = disk_images(args.archive, scan, author)
        if sha != saved["diskset_sha256"] or metadata["archive_md5"] != bench["scans"][scan]["archive_md5"]:
            raise ValueError("The intensity archive changed")
        interface = tuple(saved["interface"])
        ctx = centers_and_context(images, modes, metadata["metadata_fields"], settings, interface)
        if ctx["receipt"] != saved["geometry"]:
            raise ValueError("The common geometry changed")
        coeff = np.asarray(reduced["scans"][scan]["coefficients_nm"]["three_stars"])
        walls = evaluate(ctx, images, metadata["metadata_fields"], interface, coeff,
                         widths["scans"][scan], previous["scans"][scan])
        result["scans"][scan] = {"walls": walls}
        result["seconds"] = perf_counter() - started
        atomic_json(args.output, result)
        print(scan, len(walls), round(result["seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
