"""Transfer the two-ring apparent width prediction to public graphene experiments."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import polar
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull

from author_detector_replay import detector_intensity
from bilayer_relaxation import make_grid, rigid_gradient, sample_disregistry
from extract_graphene_wall_signal import PROTOCOL, RECOVERY, REFINEMENT
from run_detector_prediction import AUTHOR_BASIS
from run_heterobilayer_transfer import digest
from run_mos2_first_ring_geometry import affine_seed, centers_from_seed
from run_mos2_intensity_only_replay import normalized_first_ring
from run_mos2_nonlinear_controls import minimize_all
from run_mos2_wall_period_scaling import half_maximum
from run_mos2_wall_profile_closure import PROFILE_DISTANCE, isolated_field
from run_mos2_wall_width_transfer import DISTANCE, Profiles, actual_profiles, geometry_sections
from run_registry_prediction import rotation
from shared_compute import atomic_json, checkpoint_guard


FIRST = np.array(((-1, 1), (0, 1), (1, 0), (1, -1), (0, -1), (-1, 0)))
SECOND = np.array(((-1, 2), (1, 1), (2, -1), (1, -2), (-1, -1), (-2, 1)))
MODES = np.vstack((FIRST, SECOND))
MAX_WALLS_PER_FAMILY = 10


def reconstruct_geometry(data, pixel, lattice, twist, available):
    """Get a registry geometry from the first-ring frames alone."""
    y, x = np.indices((2 * data.shape[0] - 1, data.shape[1]), dtype=float)
    coordinates = np.stack((y / 2, x))
    images = np.stack([map_coordinates(signal, coordinates, order=1)
                       for signal in np.moveaxis(data, -1, 0)], axis=-1)
    basis = lattice * np.array(((1.0, 0.5), (0., np.sqrt(3)/2)))
    first_mask = available[:6]
    first, camera = normalized_first_ring(images[..., :6][..., first_mask])
    seed, transform, diagnostics = affine_seed(first, camera, FIRST[first_mask], pixel, lattice, 0., basis)
    q = MODES @ np.linalg.inv(AUTHOR_BASIS) @ transform / lattice
    fit, centers, local = centers_from_seed(first, camera, q[:6][first_mask], basis, seed, pixel)
    aa_hull = ConvexHull(centers)
    affine = fit[:2].T
    turn, _ = polar(affine)
    # The twist-dominated affine phase gradient is nearly a quarter turn
    # from the crystal orientation. Zero mean lattice mismatch fixes this
    # remaining angular gauge without using the target diffraction ring.
    angle = np.arctan2(turn[1, 0], turn[0, 0]) + np.pi / 2 + np.deg2rad(twist) / 2
    gradient = affine @ rotation(-angle)
    inferred_twist = np.rad2deg(abs(gradient[1, 0] - gradient[0, 1]) / 2)
    diagnostics.update(local)
    diagnostics.update({"first_ring_affine_fit": fit.tolist(), "centers_rc_px": centers.tolist(),
                        "center_hull_fraction": float(aa_hull.volume / images[..., 0].size),
                        "pixel_nm": pixel, "inferred_twist_degrees": float(inferred_twist),
                        "inferred_scan_to_crystal_degrees": float(np.rad2deg(angle))})
    context = {"basis": basis, "receipt": diagnostics, "reciprocal": q, "gradient": gradient,
               "support": np.ones(images.shape[:2], dtype=bool)}
    sections = geometry_sections(context, rotation(angle))
    by_family = {}
    for section in sections:
        label = np.rint(np.linalg.solve(basis, section["jump"])).astype(int)
        key = min(tuple(label), tuple(-label))
        by_family.setdefault(key, []).append(section)
    selected = []
    for family, walls in sorted(by_family.items()):
        # The first ring fixes the candidate locations. Space the selection
        # over the scan, including the full range of spacing in each family.
        ordered = sorted(walls, key=lambda wall: (wall["spacing_nm"], wall["centers"]))
        indices = np.linspace(0, len(walls) - 1, min(len(walls), MAX_WALLS_PER_FAMILY)).round().astype(int)
        for i in indices:
            selected.append({**ordered[i], "family": str(family)})
    return context, selected, len(sections)


def sampled_templates(section, grid, motion, adhesion, moduli, q, camera, pixel, probe, stride):
    """Render and sample synthetic intensity on the acquired rectangular raster."""
    # Exact affine map of each local chord from raster positions to registry.
    xy = section["coordinates"][..., ::-1] * pixel
    core = xy[:, len(DISTANCE) // 2]
    normal = (xy[0, 1] - xy[0, 0]) / (DISTANCE[1] - DISTANCE[0])
    points = xy.reshape(-1, 2)
    transform = np.linalg.lstsq(np.column_stack((points, np.ones(len(points)))),
                                 section["phase"].reshape(-1, 2), rcond=None)[0]
    bounds = core[:, None, :] + PROFILE_DISTANCE[[0, -1], None] * normal
    step = pixel * np.array((1., stride))
    lo = np.floor(bounds.min(axis=(0, 1)) / step).astype(int) - 2
    hi = np.ceil(bounds.max(axis=(0, 1)) / step).astype(int) + 3
    x, y = (np.arange(a, b) * s for a, b, s in zip(lo, hi, step, strict=True))
    positions = np.stack(np.meshgrid(x, y), axis=-1)
    phase = positions @ transform[:2] + transform[-1]
    nodes, weights = np.polynomial.hermite.hermgauss(5)
    delta = np.stack(np.meshgrid(nodes, nodes), axis=-1).reshape(-1, 2)
    delta *= probe / (2 * np.sqrt(np.log(2)))
    weights = np.outer(weights, weights).ravel() / np.pi
    phase = phase[..., None, :] + delta @ transform[:2]
    periodic = sample_disregistry(grid, motion, phase, grid.cell)
    distance = np.einsum("...i,i->...", positions[..., None, :] - core[1] + delta, normal)
    isolated = isolated_field(distance, section["jump"], (adhesion, moduli), grid.basis,
                              section["slip_sign"], section["k_factor"])
    interpolators = {}
    for label, field in (("periodic", periodic), ("isolated", isolated)):
        response = detector_intensity(field, q, camera)
        rendered = np.einsum("yxpc,p->yxc", response, weights)
        interpolators[label] = RegularGridInterpolator((y, x), rendered, bounds_error=True)

    def profile(model, displacement=0.):
        xy = core[:, None, :] + (DISTANCE - displacement)[None, :, None] * normal
        return interpolators[model](xy[..., ::-1]).mean(axis=0)

    return profile


def sampled_transfer(observer, observed, templates, model):
    def error(position, ring):
        return float(np.linalg.norm(observer.shape(templates(model, position))[ring] - observed[ring]))

    outputs = []
    for ring in range(2):
        trials = np.linspace(-observer.center_bound, observer.center_bound, 21)
        k = int(np.argmin([error(value, ring) for value in trials]))
        lo, hi = trials[max(k-1, 0)], trials[min(k+1, len(trials)-1)]
        result = minimize_scalar(lambda value: error(value, ring), bounds=(lo, hi), method="bounded")
        outputs.append({"center_nm": float(result.x),
                        "error_by_ring": [error(result.x, i) for i in range(2)],
                        "at_position_bound": bool(abs(result.x) >= .99 * observer.center_bound)})
    return outputs


def assess_wall(section, raw, available, context, grid, motion, adhesion, moduli, probe, stride):
    pixel = context["receipt"]["pixel_nm"]
    q = context["reciprocal"]
    camera = np.tile((1., 0., 0.), (12, 1))
    index = np.arange(len(q))
    rings = {"first": (index < 6) & available, "second": (index >= 6) & available}
    if min(map(np.count_nonzero, rings.values())) < 3:
        raise ValueError("The instrument apertures do not resolve both diffraction rings")
    query = section["coordinates"].copy()
    query[..., 0] /= stride
    actual = actual_profiles(raw, query)
    observer = Profiles(section, q, camera, rings, probe, "sine_gordon")
    observed = observer.shape(actual)
    measured = [observer.fit(observed, [i]) for i in (0, 1)]
    modeled = sampled_templates(section, grid, motion, adhesion, moduli, q, camera, pixel, probe, stride)
    candidates = {}
    for label in ("periodic", "isolated"):
        shape = observer.shape(modeled(label))
        candidates[label] = {"widths": [observer.fit(shape, [i]) for i in (0, 1)],
                             "transfer": sampled_transfer(observer, observed, modeled, label)}
    split = []
    opposite_available = available & np.roll(available.reshape(2, 6), 3, axis=1).ravel()
    for half in (index % 6 < 3, index % 6 >= 3):
        paired = {label: mask & half & opposite_available for label, mask in rings.items()}
        if min(map(np.count_nonzero, paired.values())) < 2:
            split.append(None)
            continue
        # Every other first-ring orientation is constant for a given ideal
        # isolated wall. A surviving triplet must contain a sensitive channel.
        partner = (index // 6)*6 + (index + 3) % 6
        if any(not np.any(group[channels] | group[partner[channels]])
               for group, channels in zip(paired.values(), observer.channels, strict=True)):
            split.append(None)
            continue
        split_observer = Profiles(section, q, camera, paired, probe, "sine_gordon")
        if min(map(len, split_observer.channels)) == 0:
            split.append(None)
            continue
        shape = split_observer.shape(actual)
        split.append([split_observer.fit(shape, [i]) for i in (0, 1)])
    position = section["coordinates"][1, len(DISTANCE) // 2]
    return {"centers": section["centers"], "family": section["family"], "midpoint_rc": position.tolist(),
            "spacing_nm": section["spacing_nm"], "reach_nm": section["reach_nm"],
            "measured": measured, "opposite_triplets": split, "models": candidates}


def gap(pair, qualified=True):
    if pair is None or (qualified and not all(fit["qualified"] for fit in pair)):
        return None
    return pair[0]["width_nm"] - pair[1]["width_nm"]


def summarize(walls):
    paired = [row for row in walls if gap(row["measured"]) is not None]
    report = {"wall_candidates": len(walls), "paired_widths": len(paired)}
    if not paired:
        return report
    common = [row for row in paired if all(gap(candidate["widths"]) is not None
                                            for candidate in row["models"].values())]
    measured = np.array([gap(row["measured"]) for row in paired])
    repeats = [v for row in paired if None not in (v := [gap(s) for s in row["opposite_triplets"]])]
    report.update({"spacing_nm_median": float(np.median([row["spacing_nm"] for row in paired])),
                   "observed_widths_nm_median": np.median([[v["width_nm"] for v in row["measured"]]
                                                           for row in paired], axis=0).tolist(),
                   "observed_gap_nm_median": float(np.median(measured)),
                   "common_model_pairs": len(common),
                   "observed_gap_on_common_pairs_nm_median":
                       float(np.median([gap(row["measured"]) for row in common])) if common else None,
                   "positive_gaps": int(np.count_nonzero(measured > 0)),
                   "opposite_triplet_pairs": len(repeats),
                   "opposite_gap_med_abs_difference_nm":
                       float(np.median(np.abs(np.diff(repeats, axis=1)))) if repeats else None,
                   "models": {}})
    for label in ("periodic", "isolated"):
        predicted = [gap(row["models"][label]["widths"]) for row in common]
        width_errors = [abs(value - gap(row["measured"])) for value, row in zip(predicted, common, strict=True)]
        transfer_errors = [[row["models"][label]["transfer"][i]["error_by_ring"][1-i] for i in (0, 1)]
                           for row in paired]
        report["models"][label] = {"qualified_gap_pairs":
                                      sum(gap(row["models"][label]["widths"]) is not None for row in paired),
                                  "gap_nm_median": float(np.median(predicted)) if predicted else None,
                                  "gap_mae_nm": float(np.mean(width_errors)) if width_errors else None,
                                  "mean_cross_ring_errors": np.mean(transfer_errors, axis=0).tolist()}
    report["source_ring_fitted_isolated_mean_cross_ring_errors"] = np.mean(
        [[f["error_by_ring"][1-i] for i, f in enumerate(row["measured"])] for row in paired], axis=0
    ).tolist()
    return report


def nominal_prediction(protocol):
    """Predict the twist sweep before using any scan's measured wall profiles."""
    inputs = protocol["inputs"]
    a = inputs["lattice_nm"]
    pixel = inputs["raster_pixel_nm"]
    basis = a * np.array(((1., .5), (0., np.sqrt(3)/2)))
    adhesion = np.r_[inputs["adhesion_even_coefficients_meV_per_cell"], 0., 0.]
    moduli = np.asarray(inputs["single_layer_bulk_shear_eV_per_cell"]) / 2
    q = MODES @ np.linalg.inv(AUTHOR_BASIS) / a
    camera = np.tile((1., 0., 0.), (12, 1))
    rings = {"first": np.arange(12) < 6, "second": np.arange(12) >= 6}
    predictions = {}
    for label, scan in protocol["scans"].items():
        checkpoint_guard()
        gradient = rigid_gradient(scan["published_twist_degrees"])
        grid = make_grid(gradient, basis, 96, *moduli)
        motion, receipt = minimize_all(grid, adhesion)
        edge = basis[:, 0]
        direction = np.linalg.solve(gradient, edge)
        tangent = direction / np.linalg.norm(direction)
        normal = np.array((-tangent[1], tangent[0]))
        slip = np.array((-edge[1], edge[0])) / a
        slope = (gradient @ normal) @ slip
        normal *= np.sign(slope)
        spacing = a * np.sqrt(3) / (2 * abs(slope))
        along = .08 * (np.arange(3)[:, None, None] - 1)
        phase = edge/2 + along*edge + DISTANCE[None, :, None]*(gradient @ normal)
        chord = along*direction + DISTANCE[None, :, None]*normal + np.array((10., 10.))
        section = {"jump": edge, "slip_sign": 1., "k_factor": float(normal @ slip)**2, "phase": phase,
                   "coordinates": chord[..., ::-1]/pixel, "keep": abs(DISTANCE) <= min(8., .68*spacing)}
        probe = scan["probe_fwhm_nm"]
        templates = sampled_templates(section, grid, motion, adhesion, moduli, q, camera, pixel, probe, 2)
        observer = Profiles(section, q, camera, rings, probe, "sine_gordon")
        fits = {}
        for model in ("periodic", "isolated"):
            signal = observer.shape(templates(model))
            fits[model] = [observer.fit(signal, [i]) for i in (0, 1)]
        distance = np.arange(-6., 6.01, .025)
        layer_shift = sample_disregistry(grid, motion, edge/2 + distance[:, None]*(gradient @ normal), grid.cell)
        predictions[label] = {"row_spacing_nm": spacing, "predicted_apparent_widths": fits,
                              "actual_slip_gradient_width_nm":
                                  half_maximum(distance, np.gradient(layer_shift @ slip, distance)),
                              "gap_nm": {name: gap(pair, qualified=False) for name, pair in fits.items()},
                              "equilibrium": receipt}
    return predictions


def run_scan(protocol_path, protocol, label, directory):
    scan = protocol["scans"][label]
    inputs = protocol["inputs"]
    extraction = json.loads((directory / "receipt.json").read_text())
    validate_aperture_receipt(protocol_path, scan, extraction)
    with np.load(directory / "intensity.npz") as handle:
        raw = handle["intensity"]
    available = np.asarray(extraction["usable"], dtype=bool)
    first = raw[0, ..., :6].reshape(-1, 6)
    pairs = [(i, i + 3) for i in range(3) if available[i] and available[i + 3]]
    correlation = [float(np.corrcoef(first[:, i], first[:, j])[0, 1]) for i, j in pairs]
    second = raw[0, ..., 6:].reshape(-1, 6)
    second_correlation = [float(np.corrcoef(second[:, i], second[:, i+3])[0, 1])
                          for i in range(3) if available[i+6] and available[i+9]]
    qualification = {"first_ring_opposite_disk_correlations": correlation,
                     "second_ring_opposite_disk_correlations": second_correlation,
                     "receipt": extraction, "integrated_intensity_sha256": digest(directory / "intensity.npz")}
    gates = json.loads(RECOVERY.read_text())["geometry_gates"]
    use_gates = "aperture_recovery_sha256" in extraction or "author_aperture_sha256" in extraction
    if use_gates and (
        len(correlation) < 2 or np.median(correlation) < gates["median_opposite_first_ring_correlation_minimum"]
    ):
        return {**qualification, "qualification_failure": "Opposite first-ring disks do not show repeatable fringes"}
    try:
        context, sections, candidates = reconstruct_geometry(raw[0], inputs["raster_pixel_nm"], inputs["lattice_nm"],
                                                             scan["published_twist_degrees"], available)
    except ValueError as error:
        return {**qualification, "qualification_failure": str(error)}
    diagnostics = context["receipt"]
    relative_error = abs(diagnostics["inferred_twist_degrees"] / scan["published_twist_degrees"] - 1)
    if use_gates and relative_error > gates["first_ring_twist_relative_error_maximum"]:
        return {**qualification, "first_ring_geometry": diagnostics,
                "qualification_failure": "First-ring twist disagrees with published geometry"}
    if not sections:
        return {**qualification, "first_ring_geometry": diagnostics, "reason": "No quadrilateral interior chords"}
    adhesion = np.r_[inputs["adhesion_even_coefficients_meV_per_cell"], 0., 0.]
    moduli = np.asarray(inputs["single_layer_bulk_shear_eV_per_cell"]) / 2
    grid = make_grid(context["gradient"], context["basis"], 96, *moduli)
    motion, solver = minimize_all(grid, adhesion)
    summaries, walls = [], []
    for aperture in raw:
        current = []
        for section in sections:
            checkpoint_guard()
            current.append(assess_wall(section, aperture, available, context, grid, motion, adhesion,
                                       moduli, scan["probe_fwhm_nm"], extraction["row_stride"]))
        summaries.append(summarize(current))
        walls.append(current)
    return {**qualification, "first_ring_geometry": diagnostics, "equilibrium": solver,
            "eligible_geometry_chords": candidates,
            "apertures": summaries, "walls": walls}


def validate_aperture_receipt(protocol, scan, receipt):
    """Confirm the extraction and comparison refer to the advertised acquisition."""
    if receipt["protocol_sha256"] != digest(protocol):
        raise ValueError("The extraction was not produced with the frozen protocol")
    suffix = f'/records/{scan["record"]}/files/{scan["file"]}?download=1'
    if not receipt["input_url"].endswith(suffix) or receipt["region_rc"] != scan["region_rc"]:
        raise ValueError("The raw scan or its region differs from the protocol")
    for key, path in (("aperture_recovery_sha256", RECOVERY), ("aperture_refinement_sha256", REFINEMENT)):
        if key in receipt and receipt[key] != digest(path):
            raise ValueError("The detector calibration protocol changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("intensities", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scans", nargs="+")
    parser.add_argument("--nominal-only", action="store_true", help="Score the published twists in ideal periodic cells")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    started = perf_counter()
    protocol = json.loads(args.protocol.read_text())
    if args.nominal_only:
        atomic_json(args.output, nominal_prediction(protocol))
        return
    inputs = (Path(__file__), args.protocol, RECOVERY, REFINEMENT,
              *(Path(__file__).with_name(name) for name in ("extract_graphene_wall_signal.py",
                                                            "bilayer_relaxation.py",
                                                            "run_mos2_first_ring_geometry.py",
                                                            "run_mos2_wall_width_transfer.py",
                                                            "run_mos2_wall_profile_closure.py")))
    result = {"status": "running", "inputs_sha256": {p.name: digest(p) for p in inputs}, "scans": {}}
    for label in args.scans or protocol["scans"]:
        checkpoint_guard()
        try:
            result["scans"][label] = run_scan(args.protocol, protocol, label, args.intensities / label)
        except (ValueError, OSError) as error:
            result["scans"][label] = {"qualification_failure": str(error)}
        result["elapsed_s"] = perf_counter() - started
        atomic_json(args.output, result)
        row = result["scans"][label]
        print(label, row.get("qualification_failure") or row.get("reason") or row["apertures"], flush=True)
    result["status"] = "finished"
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
