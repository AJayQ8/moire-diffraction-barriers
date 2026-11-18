"""Explain the two diffraction rings' wall-width readouts with forward profiles."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar

from author_detector_replay import detector_intensity
from author_release import load_author_release
from bilayer_relaxation import (MODES, MONOLAYERS, STACKING, elastic_parameters_per_reference_cell,
                                make_grid, sample_disregistry, stack_energy)
from observe_bilayers import read_input_metadata
from run_heterobilayer_sparse_geometry import solve_from_geometry
from run_heterobilayer_template_transfer import design_matrix
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_directional_diffraction import wall_sectors
from run_mos2_directional_soliton import directional_solitons
from run_mos2_intensity_only_replay import disk_images
from run_mos2_nonlinear_controls import alternative_inputs, minimize_all
from run_mos2_slow_readout_transfer import calibrate, render
from run_mos2_wall_width_transfer import (DISTANCE, PROBES, REDUCED, Profiles,
                                          actual_profiles, geometry_sections, isolated_width)
from run_registry_prediction import rotation
from run_wall_heterostrain import PROTOCOL, SOURCE
from shared_compute import atomic_json, checkpoint_guard


PROFILE_DISTANCE = np.arange(-15., 15.01, .25)
WIDTHS = PROJECT / "mos2_wall_width_transfer_v01.json"


def isolated_field(distance, edge, material, basis, slope_sign, orientation):
    """Integrate the exact fixed-slip, straight-wall energy balance."""
    adhesion, (bulk, shear) = material
    slip = np.array((-edge[1], edge[0])) / np.sqrt(3)
    path = np.r_[np.linspace(0., .499, 1100), .5 - np.geomspace(.0009, 1e-6, 100)]
    phases = .5 * edge + path[:, None] * slip
    minimum = stack_energy(.5 * edge + .5 * slip, basis, adhesion)[0]
    barrier = stack_energy(phases, basis, adhesion)[0] - minimum
    if np.any(barrier <= 0):
        raise ValueError("The input adhesion does not support the specified straight-wall path")
    stiffness = shear + bulk * orientation
    x = cumulative_trapezoid(np.linalg.norm(slip)*np.sqrt(stiffness/(2*barrier)), path, initial=0.)
    fitted = np.interp(abs(distance), x, path, right=.5) * np.sign(distance) * slope_sign
    halfwidth = np.interp(-barrier[0]/4, -barrier, x)
    exact_width = isolated_width(basis, edge, adhesion)[0] * np.sqrt(stiffness)
    if abs(2*halfwidth - exact_width) > .005:
        raise ValueError("The isolated-wall first-integral check failed")
    return .5*edge + fitted[..., None]*slip


def fields_on_chord(section, context, scan_angle, grid, motion, second_grid, second_motion, compact, materials, probe):
    """Convolve first along the wall, and leave the normal for the intensity response."""
    sigma = probe / np.sqrt(8*np.log(2))
    points, weights = np.polynomial.hermite.hermgauss(5)
    weights /= np.sqrt(np.pi)
    q0 = len(DISTANCE)//2
    origin = section["phase"][:, q0]
    slope = (section["phase"][0, q0+1] - origin[0])/(DISTANCE[1]-DISTANCE[0])
    tangent = section["jump"]/section["length_nm"]
    span = np.sqrt(2.)*sigma*points
    phase = (origin[:, None, None, :] + span[None, :, None, None]*tangent
             + PROFILE_DISTANCE[None, None, :, None]*slope)
    xy = section["coordinates"][..., ::-1]*context["receipt"]["pixel_nm"]
    flat = xy.reshape(-1, 2)
    coordinates = np.column_stack((flat, np.ones(len(flat))))
    fit = np.linalg.lstsq(coordinates, section["phase"].reshape(-1, 2), rcond=None)[0]
    gradient = fit[:2].T @ rotation(-scan_angle)
    basis = context["basis"]
    if np.max(abs(coordinates @ fit - section["phase"].reshape(-1, 2))) > 1e-10:
        raise ValueError("The local wall geometry was not affine across a chord")
    _, c1, c2, c3, _, _ = materials["anchor"][0]
    _, bulk, shear = MONOLAYERS["MoS2"]
    walls, _ = directional_solitons(phase, basis, gradient, (c1 - 8*c2 + 9*c3)/(2000*shear), bulk/shear)
    states = {"periodic_anchor": sample_disregistry(grid, motion, phase, grid.cell),
              "periodic_alternate": sample_disregistry(second_grid, second_motion, phase, second_grid.cell),
              "compact": phase + design_matrix(phase, basis, MODES) @ compact,
              "analytic_network": walls["orientation_stiffness"], "rigid": phase}
    for name, material in materials.items():
        isolated = isolated_field(PROFILE_DISTANCE, section["jump"], material, basis, section["slip_sign"], section["k_factor"])
        states[f"isolated_{name}"] = np.broadcast_to(isolated, phase.shape)
    return states, weights


def render_profiles(states, q, coefficients, weights, probe):
    sigma = probe / ((PROFILE_DISTANCE[1]-PROFILE_DISTANCE[0])*np.sqrt(8*np.log(2)))
    result = {}
    for key, value in states.items():
        response = detector_intensity(value, q, coefficients)
        # Probe quadrature runs along each wall. The third dimension is normal.
        average = np.mean(np.einsum("abnc,b->anc", response, weights), axis=0)
        result[key] = gaussian_filter1d(average, sigma, axis=0, mode="nearest")
    return result


def transfer(profile, observed, signal):
    """Use the observed position from one ring to predict the shape in the other."""
    def at(location):
        sampled = np.column_stack([np.interp(DISTANCE-location, PROFILE_DISTANCE, c) for c in signal.T])
        return profile.shape(sampled)

    def residual(p, i):
        return float(np.linalg.norm(at(p)[i] - observed[i]))

    outputs = []
    for i in (0, 1):
        trials = np.linspace(-profile.center_bound, profile.center_bound, 21)
        k = int(np.argmin([residual(v, i) for v in trials]))
        lo, hi = trials[max(0, k-1)], trials[min(k+1, len(trials)-1)]
        if hi - lo < .01:
            raise ValueError("The translation fit has no interval")
        fit = minimize_scalar(lambda p: residual(p, i), bounds=(lo, hi), method="bounded")
        outputs.append({"center_nm": float(fit.x), "error_by_ring": [residual(fit.x, j) for j in (0, 1)],
                        "at_position_bound": bool(abs(fit.x) >= .99*profile.center_bound)})
    return outputs


def evaluate(context, images, meta, interface, compact, widths):
    scan_angle = np.deg2rad(float(meta["SampleRotation"]))
    sections = geometry_sections(context, rotation(scan_angle))
    basis, gradient, nominal = (context[key] for key in ("basis", "gradient", "rigid"))
    field = nominal + design_matrix(nominal, basis, MODES) @ compact
    q, support = context["reciprocal"], context["support"]
    source = support & ~np.logical_or.reduce(tuple(wall_sectors(nominal, basis).values()))
    normalized, camera = calibrate(render(field, np.ones(support.shape), q, context["probe"]), images, source)
    grid, motion, _ = solve_from_geometry(gradient, basis, interface)
    adhesion, moduli, _ = alternative_inputs(basis, "P")
    alternate_grid = make_grid(gradient, basis, motion.shape[0], *moduli)
    alternate, _ = minimize_all(alternate_grid, adhesion)
    mats = {"anchor": (STACKING[interface], elastic_parameters_per_reference_cell(*interface[:2])),
            "alternate": (adhesion, moduli)}
    walls = []
    for section, old in zip(sections, widths["walls"], strict=True):
        checkpoint_guard()
        if section["centers"] != old["centers"]:
            raise ValueError("A wall chord changed between the two measurements")
        measured = actual_profiles(normalized, section["coordinates"])
        row = {"centers": section["centers"], "spacing_nm": section["spacing_nm"], "probe": {}}
        for probe in PROBES:
            templates, weights = fields_on_chord(section, context, scan_angle, grid, motion,
                                                 alternate_grid, alternate, compact, mats, probe)
            values = render_profiles(templates, q, camera, weights, probe)
            obs = Profiles(section, q, camera, context["rings"], probe, "sine_gordon")
            observed = obs.shape(measured)
            fits = {}
            for name, signal in values.items():
                central = signal[np.abs(PROFILE_DISTANCE) <= DISTANCE[-1]]
                fits[name] = {"transfer": transfer(obs, observed, signal)}
                if name != "rigid":
                    fits[name]["apparent_widths"] = [obs.fit(obs.shape(central), [train]) for train in (0, 1)]
            row["probe"][f"{probe:g}"] = fits
        walls.append(row)
    return walls


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("release", type=Path)
    cli.add_argument("dependencies", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    started = perf_counter()
    sources = (PROTOCOL, SOURCE, BENCHMARK, SETTINGS, REDUCED, WIDTHS)
    protocol, source, reference, settings, reduced, measured = (json.loads(p.read_text()) for p in sources)
    author = load_author_release(args.release, args.dependencies)
    output = {
        "scope": "Use the same fixed wall chords and integrated intensities as the width transfer. Predict the full profile from fixed material parameters. Model the finite probe along and across the chords, and give each model and each ring one position shift on the source ring. Score the normalized profile on the other ring. Also reread both rings' model widths with the original isolated-shape observer to determine whether an apparent width difference arises without an instrument anomaly. The controls are rigid layers, the compact model, an orientation-aware analytic wall network, and both published material inputs with and without the neighboring domain walls.",
        "limits": "All acquisitions were previously examined. The two rings share the sample and geometry source. Positions and widths refer to projections and use the measured geometry. The profile shape is the scored variable, after removing any overall scale and linear background per channel. A shared camera law and nominal Gaussian probe are approximations. The exact isolated-width and full-network controls differ in both wall interaction and remaining deformation within the neighboring domains.",
        "primary": "Nominal 1.25 nm probe, original ten comparison maps, report both transfer directions separately with equal weight per map. Keep the previously qualified width profiles for the apparent-width comparison. Report measured and simulated ring differences before averaging them.",
        "input_sha256": {p.name: digest(p) for p in (Path(__file__), *sources, PROJECT/"run_mos2_wall_width_transfer.py",
                                                      PROJECT/"run_mos2_directional_soliton.py")}, "scans": {},
    }
    for group in ("parallel_development", "parallel_comparison"):
        for scan in protocol["cases"][group]:
            checkpoint_guard()
            prior = source["scans"][scan]
            metadata = read_input_metadata(scan, args.archive)
            images, modes, checked = disk_images(args.archive, scan, author)
            if checked != prior["diskset_sha256"] or metadata["archive_md5"] != reference["scans"][scan]["archive_md5"]:
                raise ValueError("The profile source changed")
            interface = tuple(prior["interface"])
            view = centers_and_context(images, modes, metadata["metadata_fields"], settings, interface)
            if view["receipt"] != prior["geometry"]:
                raise ValueError("The fixed geometry changed")
            compact = np.asarray(reduced["scans"][scan]["coefficients_nm"]["three_stars"])
            walls = evaluate(view, images, metadata["metadata_fields"], interface, compact, measured["scans"][scan])
            output["scans"][scan] = {"group": group, "walls": walls}
            output["seconds"] = perf_counter()-started
            atomic_json(args.output, output)
            print(scan, len(walls), round(output["seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
