"""Measure parallel MoS2 wall widths and compare how they follow the moiré size."""

import argparse
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.optimize import brentq, least_squares, minimize_scalar

from author_detector_replay import detector_intensity
from author_release import load_author_release
from bilayer_relaxation import (MODES, STACKING, elastic_parameters_per_reference_cell, make_grid,
                                sample_disregistry, stack_energy)
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
from run_registry_prediction import rotation
from run_wall_heterostrain import PROTOCOL, SOURCE
from shared_compute import atomic_json, checkpoint_guard
from wall_registry_fit import wall_disregistry


DISTANCE = np.arange(-10., 10.01, .25)
LONGITUDINAL = np.array((.42, .50, .58))
PRIMARY_PROBE = 1.25
PROBES = (.75, PRIMARY_PROBE, 1.75)
FAMILIES = ("sine_gordon", "tanh")
RINGS = ("first", "second")
REDUCED = PROJECT / "mos2_reduced_equilibrium_v01.json"
PILOT = {"DS1", "DS2", "DS7", "DS10"}


def rms(values):
    return float(np.sqrt(np.mean(values**2)))


def geometry_sections(context, scan_to_crystal):
    """Place wall chords using only neighboring stacking centers."""
    basis = context["basis"]
    lattice = np.linalg.norm(basis[:, 0])
    fit = np.asarray(context["receipt"]["first_ring_affine_fit"])
    pixel = context["receipt"]["pixel_nm"]
    points = pixel * np.asarray(context["receipt"]["centers_rc_px"])[:, ::-1]
    labels = np.rint((points @ fit[:2] + fit[-1]) @ np.linalg.inv(basis).T).astype(int)
    if len(np.unique(labels, axis=0)) != len(labels):
        raise ValueError("Repeated lattice labels in the wall geometry")
    translations = labels @ basis.T
    neighbors = np.isclose(np.linalg.norm(translations[:, None] - translations, axis=-1), lattice, atol=1e-9)
    height, width = context["support"].shape
    sections = []
    for i, j in combinations(range(len(points)), 2):
        if not neighbors[i, j]:
            continue
        adjacent = neighbors[i] & neighbors[j]
        if adjacent.sum() != 2:
            continue
        edge = points[j] - points[i]
        length = np.linalg.norm(edge)
        tangent = edge / length
        normal = np.array((-tangent[1], tangent[0]))
        corner = points[adjacent] - points[i]
        height_nm = corner @ normal
        if np.prod(height_nm) >= 0:
            continue
        jump = translations[j] - translations[i]
        jump_normal = np.array((-jump[1], jump[0])) / lattice
        diff = translations[adjacent] - translations[i] - (corner @ tangent)[:, None] * jump / length
        normal_gradient = (height_nm @ diff) / (height_nm @ height_nm)
        slope = normal_gradient @ jump_normal
        if abs(slope) < 1e-4:
            continue
        reach = min(8., .68 * np.min(abs(height_nm)))
        keep = abs(DISTANCE) <= reach
        if reach < 2.5:
            continue
        query = points[i] + LONGITUDINAL[:, None, None] * edge + DISTANCE[None, :, None] * normal
        coords = (query / pixel)[..., ::-1]
        usable = coords[:, keep]
        if np.any((usable < 4) | (usable > np.array((height, width)) - 5)):
            continue
        phase = LONGITUDINAL[:, None, None] * jump + DISTANCE[None, :, None] * normal_gradient
        sections.append({"centers": [i, j], "jump": jump, "slip_sign": np.sign(slope),
                         "k_factor": float((scan_to_crystal @ normal) @ jump_normal)**2,
                         "spacing_nm": float(lattice*np.sqrt(3)/(2*abs(slope))), "reach_nm": float(reach),
                         "length_nm": float(length), "phase": phase, "coordinates": coords, "keep": keep})
    return sections


def isolated_width(basis, edge, adhesion):
    """Return the FWHM per square root of elastic stiffness for one partial wall."""
    jump = np.array((-edge[1], edge[0])) / np.sqrt(3)

    def energy(x):
        return float(stack_energy(.5 * edge + x * jump, basis, adhesion)[0])

    minimum = energy(.5)

    def barrier(x):
        return energy(x) - minimum

    peak = barrier(0.)
    half = brentq(lambda x: barrier(x) - peak / 4, 0., .5)
    distance = quad(lambda x: np.linalg.norm(jump) / np.sqrt(2*barrier(x)), 0., half, epsabs=1e-10)[0]
    return 2 * distance, peak


def observe(field, q, coefficients, fwhm):
    raw = detector_intensity(field, q, coefficients)
    sigma = fwhm / (np.sqrt(8*np.log(2)) * (DISTANCE[1] - DISTANCE[0]))
    return gaussian_filter1d(raw, sigma, axis=-2, mode="nearest")


def sample_wall_states(context, interface, sections):
    gradient, basis = (context[k] for k in ("gradient", "basis"))
    grid, motion, _ = solve_from_geometry(gradient, basis, interface)
    adhesion, moduli, _ = alternative_inputs(basis, "P")
    alternate_grid = make_grid(gradient, basis, motion.shape[0], *moduli)
    alternate, receipt = minimize_all(alternate_grid, adhesion)
    phase = np.stack([s["phase"] for s in sections])
    return {"periodic_anchor": sample_disregistry(grid, motion, phase, grid.cell),
            "periodic_alternate": sample_disregistry(alternate_grid, alternate, phase, grid.cell)}, {
                "isolated_anchor": (STACKING[interface], elastic_parameters_per_reference_cell(*interface[:2])),
                "isolated_alternate": (adhesion, moduli),
            }, receipt


class Profiles:
    def __init__(self, section, q, coefficients, rings, probe, family):
        self.keep = section["keep"]
        self.distance = DISTANCE[self.keep]
        self.phase = section["jump"]
        self.sign = section["slip_sign"]
        self.q, self.coefficients = q, coefficients
        self.probe, self.family = probe, family
        self.center_bound = min(3., .50 * self.distance[-1])
        self.width_bounds = (.75, min(9., 2.25 * self.distance[-1]))
        linear = np.column_stack((self.distance, np.ones(len(self.distance))))
        self.trend = linear @ np.linalg.pinv(linear)
        halfway = wall_disregistry(np.array((0., 9., -9.)), 0., 2., self.phase, self.family)
        anchors = detector_intensity(halfway, self.q, self.coefficients)
        contrast = np.max(abs(anchors[1:] - anchors[0]), axis=0)
        self.channels = []
        for ring in RINGS:
            take = np.flatnonzero(rings[ring])
            self.channels.append(take[contrast[take] > .30 * contrast[take].max()])

    def shape(self, profiles):
        part = profiles[self.keep]
        detrended = part - self.trend @ part
        normalized = detrended / np.maximum(np.linalg.norm(detrended, axis=0), 1e-12)
        return tuple(normalized[:, j] / np.sqrt(len(j)) for j in self.channels)

    def model(self, center, width):
        field = wall_disregistry(self.sign * DISTANCE, self.sign * center, width, self.phase, self.family)
        return self.shape(observe(field, self.q, self.coefficients, self.probe))

    def fit(self, target, training, width=None):
        def objective(p):
            estimate = self.model(p[0], p[1] if width is None else width)
            return np.concatenate([(estimate[i] - target[i]).ravel() for i in training])

        if width is None:
            init = np.clip(3., self.width_bounds[0] + .1, self.width_bounds[1] - .1)
            bounds = ([-self.center_bound, self.width_bounds[0]], [self.center_bound, self.width_bounds[1]])
            candidates = [least_squares(objective, [x, init], bounds=bounds, max_nfev=50)
                          for x in (0., -.4*self.center_bound, .4*self.center_bound)]
            result = min(candidates, key=lambda value: value.cost)
            center, width = result.x
            qualified = result.success and not np.any(result.active_mask)
        else:
            trial = np.linspace(-self.center_bound, self.center_bound, 13)
            cost = [np.sum(objective([x])**2) for x in trial]
            found = int(np.argmin(cost))
            lo, hi = trial[max(0, found-1)], trial[min(len(trial)-1, found+1)]
            result = minimize_scalar(lambda location: np.sum(objective([location])**2),
                                     bounds=(lo, hi), method="bounded")
            center = result.x
            qualified = result.success and abs(center) < .99 * self.center_bound
        fitted = self.model(center, width)
        return {"center_nm": float(center), "width_nm": float(width), "qualified": bool(qualified),
                "error_by_ring": [float(np.linalg.norm(u-v)) for u,v in zip(target, fitted, strict=True)]}


def actual_profiles(images, coords):
    coordinates = np.moveaxis(coords, -1, 0)
    return np.stack([map_coordinates(frame, coordinates, order=1, mode="nearest").mean(axis=0)
                     for frame in np.moveaxis(images, -1, 0)], axis=-1)


def score_sections(images, context, section, states, material, camera):
    actual = actual_profiles(images, section["coordinates"])
    q, basis = (context[k] for k in ("reciprocal", "basis"))
    predicted_isolated = {}
    for name, (adhesion, (bulk, shear)) in material.items():
        coefficient, barrier = isolated_width(basis, section["jump"], adhesion)
        width = coefficient * np.sqrt(shear + bulk * section["k_factor"])
        predicted_isolated[name] = {"width_nm": float(width), "barrier_meV": 1000*barrier,
                                    "effective_stiffness_eV": float(shear + bulk*section["k_factor"])}
    result = {k: section[k] for k in ("centers", "reach_nm", "spacing_nm", "length_nm", "k_factor")}
    result["isolated"] = predicted_isolated
    result["fits"] = {}
    for probe, family in ((p, "sine_gordon") for p in PROBES):
        state = Profiles(section, q, camera, context["rings"], probe, family)
        target = state.shape(actual)
        synthetic = {name: state.shape(observe(phase, q, camera, probe).mean(axis=0))
                     for name, phase in states.items()}
        fits = {"joint": state.fit(target, [0, 1]), "separate": [state.fit(target, [i]) for i in (0, 1)],
                "synthetic": {name: state.fit(value, [0, 1]) for name, value in synthetic.items()}}
        if probe == PRIMARY_PROBE:
            candidates = {**{k: v["width_nm"] for k,v in fits["synthetic"].items()},
                          **{k: v["width_nm"] for k,v in predicted_isolated.items()}}
            fits["transferred_shape"] = {
                name: [state.fit(target, [i], width=width) for i in (0, 1)] for name,width in candidates.items()
            }
            sensitivity = Profiles(section, q, camera, context["rings"], probe, "tanh")
            replay = sensitivity.shape(actual)
            fits["local_shape_family"] = {"tanh_joint": sensitivity.fit(replay, [0, 1])}
            # Synthetic truth goes through the exact same finite chord and masks.
            injection = state.model(.20, 3.5)
            recovery = [state.fit(injection, [i]) for i in (0, 1)]
            if any(abs(p["width_nm"]-3.5)>.01 or abs(p["center_nm"]-.20)>.01 for p in recovery):
                raise ValueError("The width observer did not recover its injected truth")
        result["fits"][f"{probe:g}"] = fits
    return result


def run_scan(images, context, metadata, interface, compact):
    phi = np.deg2rad(float(metadata["SampleRotation"]))
    sections = geometry_sections(context, rotation(phi))
    if not sections:
        return {"walls": [], "reason": "No complete wall chord between two internal cells"}
    nominal, basis, q = (context[k] for k in ("rigid", "basis", "reciprocal"))
    field = nominal + design_matrix(nominal, basis, MODES) @ compact
    masks = tuple(wall_sectors(nominal, basis).values())
    train = context["support"] & ~np.logical_or.reduce(masks)
    signals = render(field, np.ones(nominal.shape[:2]), q, context["probe"])
    normalized, camera = calibrate(signals, images, train)
    states, materials, solve = sample_wall_states(context, interface, sections)
    walls = []
    for k, piece in enumerate(sections):
        checkpoint_guard()
        wall = score_sections(normalized, context, piece, {key: value[k] for key, value in states.items()},
                              materials, camera)
        walls.append(wall)
    return {"camera_training_pixels": int(train.sum()), "alternate_solve": solve, "walls": walls}


def main():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("archive", type=Path)
    command.add_argument("release", type=Path)
    command.add_argument("dependencies", type=Path)
    command.add_argument("output", type=Path)
    command.add_argument("--pilot", action="store_true")
    args = command.parse_args()
    started = perf_counter()
    paths = PROTOCOL, SOURCE, BENCHMARK, SETTINGS, REDUCED
    protocol, source, benchmark, settings, reduced = (json.loads(p.read_text()) for p in paths)
    author = load_author_release(args.release, args.dependencies)
    report = {
        "scope": "Read wall profiles halfway between directly detected first-ring AA centers. Require both adjacent triangles and a chord that remains inside the image. Read three strips per wall at fixed offsets. Estimate signed contrast from the rest of the image. Read widths by passing a partial-dislocation profile through the same probe and normalizing the detrended shape. Compare one published periodic solution and an alternative with the isolated partial from the same inputs. Their periodic profiles use the local geometry from the adjacent centers and the equilibrium at the mean geometry. Learn each width and position on one ring and compare with the separate ring. Repeat at three probe widths and with a tanh shape at the nominal width.",
        "primary": "Compare nominal-probe gradient FWHM for walls with both single-ring fits, joint fit, and the two synthetic periodic fits off their fitted bounds. Score the prediction level as an equal-map average, with DS1–DS2 low-twist and all fourteen maps reported. Compare the nominal sine-Gordon result to other probe widths and the tanh profile. The model comparisons measure widths under the same observer, and all wall- and map-level estimates stay visible.",
        "limits": "Map-level comparisons inside one previously examined study. Wall profiles and rings within a map are correlated. Nominal Gaussian probe, approximate shared global camera, local planar geometry, and straight profiles at the wall midpoint. Fitted widths are estimates under these observation assumptions. Each profile's linear background and scale are removed for all candidates. The finite-width periodic solution has the geometry and material parameters used in the original project.",
        "inputs_sha256": {p.name: digest(p) for p in (Path(__file__), *paths, PROJECT/"wall_registry_fit.py",
                                                       PROJECT/"run_mos2_nonlinear_controls.py")},
        "scans": {},
    }
    for group in ("parallel_development", "parallel_comparison"):
        for scan in protocol["cases"][group]:
            if args.pilot and scan not in PILOT:
                continue
            checkpoint_guard()
            saved = source["scans"][scan]
            interface = tuple(saved["interface"])
            meta = read_input_metadata(scan, args.archive)
            images, modes, sha = disk_images(args.archive, scan, author)
            if sha != saved["diskset_sha256"] or meta["archive_md5"] != benchmark["scans"][scan]["archive_md5"]:
                raise ValueError("The wall scan changed")
            context = centers_and_context(images, modes, meta["metadata_fields"], settings, interface)
            if context["receipt"] != saved["geometry"]:
                raise ValueError("The original wall-center geometry changed")
            compact = np.asarray(reduced["scans"][scan]["coefficients_nm"]["three_stars"])
            row = run_scan(images, context, meta["metadata_fields"], interface, compact)
            twist = abs(float(np.rad2deg((context["gradient"][1, 0] - context["gradient"][0, 1])/2)))
            report["scans"][scan] = {"group": group, "inferred_twist_deg": twist, **row}
            report["seconds"] = perf_counter() - started
            atomic_json(args.output, report)
            survivors = [wall["fits"]["1.25"]["joint"]["width_nm"] for wall in row["walls"]
                         if wall["fits"]["1.25"]["joint"]["qualified"]]
            print(scan, len(row["walls"]), len(survivors), np.median(survivors) if survivors else None, flush=True)


if __name__ == "__main__":
    main()
