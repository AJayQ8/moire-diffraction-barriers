"""Check opposite diffraction spots and a common wall-profile window in MoS2."""

import argparse
import json
from pathlib import Path

import numpy as np

from author_release import load_author_release
from bilayer_relaxation import MODES, MONOLAYERS, STACKING, make_grid
from observe_bilayers import read_input_metadata
from run_heterobilayer_sparse_geometry import solve_from_geometry
from run_heterobilayer_template_transfer import design_matrix
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_directional_diffraction import wall_sectors
from run_mos2_intensity_only_replay import disk_images
from run_mos2_nonlinear_controls import alternative_inputs, minimize_all
from run_mos2_slow_readout_transfer import calibrate, render
from run_mos2_wall_profile_closure import PROFILE_DISTANCE, fields_on_chord, render_profiles
from run_mos2_wall_width_transfer import DISTANCE, PRIMARY_PROBE, REDUCED, Profiles, actual_profiles, geometry_sections
from run_registry_prediction import rotation
from run_wall_heterostrain import PROTOCOL, SOURCE
from shared_compute import atomic_json, checkpoint_guard


def opposite_masks(miller, rings):
    ready = (miller[:, 0] > 0) | ((miller[:, 0] == 0) & (miller[:, 1] > 0))
    for ring in ("first", "second"):
        for row in miller[rings[ring]]:
            if np.count_nonzero(np.all(miller[rings[ring]] == -row, axis=1)) != 1:
                raise ValueError("An opposite spot is missing")
    return [{name: mask & selector for name, mask in rings.items()}
            for selector in (ready, ~ready)]


def width_gap(section, values, q, camera, rings):
    observer = Profiles(section, q, camera, rings, PRIMARY_PROBE, "sine_gordon")
    shape = observer.shape(values)
    fits = [observer.fit(shape, [i]) for i in (0, 1)]
    return {"gap_nm": fits[0]["width_nm"] - fits[1]["width_nm"],
            "qualified": all(v["qualified"] for v in fits)}


def scan_result(images, modes, geometry, metadata, interface, compact):
    angle = np.deg2rad(float(metadata["SampleRotation"]))
    pieces = geometry_sections(geometry, rotation(angle))
    basis, gradient, nominal, q = (geometry[k] for k in ("basis", "gradient", "rigid", "reciprocal"))
    fit_mask = geometry["support"] & ~np.logical_or.reduce(tuple(wall_sectors(nominal, basis).values()))
    field = nominal + design_matrix(nominal, basis, MODES) @ compact
    observed, camera = calibrate(render(field, np.ones(fit_mask.shape), q, geometry["probe"]), images, fit_mask)
    rings = geometry["rings"]
    opposite = opposite_masks(modes, rings)
    grid, motion, solver = solve_from_geometry(gradient, basis, interface)
    alternate_adhesion, alternate_moduli, _ = alternative_inputs(basis, "P")
    alternate_grid = make_grid(gradient, basis, motion.shape[0], *alternate_moduli)
    alternate_motion, _ = minimize_all(alternate_grid, alternate_adhesion)
    bulk, shear = MONOLAYERS["MoS2"][1:]
    material = {"anchor": (STACKING[interface], (bulk/2, shear/2))}
    records = []
    for piece in pieces:
        checkpoint_guard()
        observed_line = actual_profiles(observed, piece["coordinates"])
        full_fields, weights = fields_on_chord(piece, geometry, angle, grid, motion,
                                              alternate_grid, alternate_motion, compact, material, PRIMARY_PROBE)
        states = {name: full_fields[name] for name in ("periodic_anchor", "isolated_anchor")}
        predicted = render_profiles(states, q, camera, weights, PRIMARY_PROBE)
        central = {name: line[np.abs(PROFILE_DISTANCE) <= DISTANCE[-1]] for name, line in predicted.items()}
        fixed = {**piece, "keep": abs(DISTANCE) <= 4.25} if piece["reach_nm"] >= 4.25 else None
        results = {}
        for window, subsection in (("adaptive", piece), ("fixed_4.25_nm", fixed)):
            if subsection is None:
                continue
            results[window] = {"measured": width_gap(subsection, observed_line, q, camera, rings),
                               "opposite": [width_gap(subsection, observed_line, q, camera, group)
                                            for group in opposite],
                               "models": {name: width_gap(subsection, line, q, camera, rings)
                                          for name, line in central.items()}}
        records.append({"centers": piece["centers"], "spacing_nm": piece["spacing_nm"], "fits": results})
    return {"equilibrium": solver, "walls": records}


def summarize(scans):
    answer = {}
    for scan, item in scans.items():
        answer[scan] = {}
        for window in ("adaptive", "fixed_4.25_nm"):
            data = [piece["fits"][window] for piece in item["walls"] if window in piece["fits"]]
            pairs = np.array([[row["gap_nm"] for row in piece["opposite"]] for piece in data
                              if all(row["qualified"] for row in piece["opposite"])])
            fit_names = ("periodic_anchor", "isolated_anchor")
            common = [piece for piece in data if piece["measured"]["qualified"]
                      and all(piece["models"][name]["qualified"] for name in fit_names)]
            gaps = np.array([piece["measured"]["gap_nm"] for piece in common])
            varying = np.array([piece["models"]["periodic_anchor"]["gap_nm"] for piece in common])
            summary = {"eligible_walls": len(data), "common_fits": len(common),
                       "opposite_pairs": len(pairs), "observed_mean_gap_nm": float(gaps.mean()) if len(gaps) else None,
                       "predicted_mean_gap_nm": float(varying.mean()) if len(varying) else None}
            if len(pairs):
                summary["opposite_median_absolute_difference_nm"] = float(np.median(abs(pairs[:, 0] - pairs[:, 1])))
            if len(gaps):
                measured = gaps - gaps.mean()
                residual = varying - varying.mean() - measured
                summary["within_scan_rmse_nm"] = float(np.sqrt(np.mean(residual**2)))
                summary["constant_rmse_nm"] = float(np.sqrt(np.mean(measured**2)))
            answer[scan][window] = summary
    return answer


def main():
    command = argparse.ArgumentParser(description=__doc__)
    for label in ("archive", "release", "dependencies", "output"):
        command.add_argument(label, type=Path)
    options = command.parse_args()
    inputs = (PROTOCOL, SOURCE, BENCHMARK, SETTINGS, REDUCED)
    protocol, source, benchmark, settings, compact = (json.loads(p.read_text()) for p in inputs)
    author = load_author_release(options.release, options.dependencies)
    output = {"scope": "All ten established comparison maps. Score both positive and negative diffraction triples. Apply a common 4.25 nm half-chord as an observer sensitivity, retaining the established adaptive-window result.",
              "limits": "Opposite spots share the acquisition and can differ physically. Their disagreement is a repeatability proxy. The fixed window truncates the wall tails and tests whether a width difference is identifiable through that same fixed window. The absolute primary fit is still the established adaptive one.",
              "input_sha256": {p.name: digest(p) for p in (*inputs, Path(__file__))},
              "scans": {}}
    for scan in protocol["cases"]["parallel_comparison"]:
        checkpoint_guard()
        prior = source["scans"][scan]
        interface = tuple(prior["interface"])
        metadata = read_input_metadata(scan, options.archive)
        images, modes, sha = disk_images(options.archive, scan, author)
        if sha != prior["diskset_sha256"] or metadata["archive_md5"] != benchmark["scans"][scan]["archive_md5"]:
            raise ValueError("An archived intensity source changed")
        geometry = centers_and_context(images, modes, metadata["metadata_fields"], settings, interface)
        if geometry["receipt"] != prior["geometry"]:
            raise ValueError("The qualified first-ring geometry changed")
        response = np.asarray(compact["scans"][scan]["coefficients_nm"]["three_stars"])
        output["scans"][scan] = scan_result(images, modes, geometry, metadata["metadata_fields"], interface, response)
        output["summary"] = summarize(output["scans"])
        atomic_json(options.output, output)
        print(scan, output["summary"][scan], flush=True)


if __name__ == "__main__":
    main()
