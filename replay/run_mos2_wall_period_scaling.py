"""Change the moiré period at fixed material parameters and cell shape."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from bilayer_relaxation import MODES, MONOLAYERS, STACKING, rigid_gradient, sample_disregistry
from run_heterobilayer_sparse_geometry import solve_from_geometry
from run_heterobilayer_transfer import PROJECT, digest
from run_mos2_directional_soliton import directional_solitons
from run_mos2_wall_width_transfer import DISTANCE, Profiles, isolated_width, observe
from run_mos2_wall_profile_closure import isolated_field
from shared_compute import atomic_json, checkpoint_guard


SOURCE = PROJECT / "moire_common_intensity_replay_v01.json"
INTERFACE = ("MoS2", "MoS2", "P")
ANGLES = (0.40, 0.55, 0.75, 1.0, 1.3, 1.6, 1.9)


def half_maximum(distance, gradient):
    center = np.argmin(abs(distance))
    midpoint = abs(gradient[center]) / 2
    for i in range(center + 1, len(distance)):
        if abs(gradient[i]) < midpoint:
            r = np.interp(midpoint, abs(gradient[i-1:i+1])[::-1], distance[i-1:i+1][::-1])
            return float(2*r)
    return None


def inspect(gradient, basis, edge, probe, q, detector, rings):
    grid, motion, receipt = solve_from_geometry(gradient, basis, INTERFACE)
    a = np.linalg.norm(edge)
    tangent = np.linalg.solve(gradient, edge)
    tangent /= np.linalg.norm(tangent)
    normal = np.array((-tangent[1], tangent[0]))
    slip = np.array((-edge[1], edge[0])) / a
    if (gradient @ normal) @ slip < 0:
        normal *= -1
    phase_per_nm = gradient @ normal
    spacing = a*np.sqrt(3)/(2*abs(phase_per_nm @ slip))
    fine = np.r_[np.arange(-.49*spacing, 0., .025), 0., np.arange(.025, .49*spacing, .025)]
    physical = sample_disregistry(grid, motion, edge/2 + fine[:, None]*phase_per_nm, grid.cell)
    partial = STACKING[INTERFACE]
    _, c1, c2, c3, _, _ = partial
    _, bulk, shear = MONOLAYERS["MoS2"]
    ratio = (c1 - 8*c2 + 9*c3)/(2000*shear)
    phase = edge/2 + DISTANCE[:, None]*phase_per_nm
    network = directional_solitons(phase, basis, gradient, ratio, bulk/shear)[0]["orientation_stiffness"]
    full = sample_disregistry(grid, motion, phase, grid.cell)
    orientation = float(normal @ slip)**2
    isolated = isolated_field(DISTANCE, edge, (partial, (bulk/2, shear/2)), basis, 1., orientation)
    reach = min(8., .68*spacing)
    section = {"keep": abs(DISTANCE) <= reach, "jump": edge, "slip_sign": 1.}
    observer = Profiles(section, q, detector, rings, probe, "sine_gordon")
    responses = {}
    for name, v in (("periodic", full), ("analytic_network", network), ("isolated", isolated)):
        signal = observe(v, q, detector, probe)
        shape = observer.shape(signal)
        responses[name] = [observer.fit(shape, [ring]) for ring in (0, 1)]
    baseline, _ = isolated_width(basis, edge, partial)
    return {"true_periodic_width_nm": half_maximum(fine, np.gradient(physical @ slip, fine, edge_order=2)),
            "true_isolated_width_nm": float(baseline*np.sqrt((shear + bulk*orientation)/2)),
            "row_spacing_nm": float(spacing), "reconstruction_widths": responses,
            "effective_orientation_factor": orientation, "solver": receipt}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    started = perf_counter()
    a = MONOLAYERS["MoS2"][0]
    basis = a*np.array(((1., .5), (0., np.sqrt(3)/2)))
    # Symmetric spot positions with both diffraction orders. A positive even
    # response suffices because these profiles have no detector offsets.
    miller = np.r_[MODES[:6], -MODES[:6]]
    order = np.tile((0, 0, 0, 1, 1, 1), 2)
    q = miller @ np.linalg.inv(basis)
    camera = np.tile((1., 0., 0.), (len(q), 1))
    rings = {"first": order == 0, "second": order == 1}
    records = json.loads(SOURCE.read_text())["scans"]
    fit = np.asarray(records["DS2"]["geometry"]["first_ring_affine_fit"])
    # A scalar multiplier preserves the oblique DS2 wall geometry. The
    # crystal rotation is immaterial for isotropic elasticity if the slip
    # and receptor basis rotate together, so use the physical scan frame.
    gradient = fit[:2].T
    ds2_angle_scale = np.sqrt(abs(np.linalg.det(gradient)))
    if ds2_angle_scale <= 0:
        raise ValueError("The source cell has no finite period")
    paths = (Path(__file__), SOURCE, PROJECT/"bilayer_relaxation.py", PROJECT/"run_mos2_wall_width_transfer.py",
             PROJECT/"run_mos2_wall_profile_closure.py")
    result = {"scope": "Vary the global twist for the unstrained bilayer and scale the measured linear registry map of DS2 to vary only its period at fixed anisotropy. Hold adhesion, elasticity and pixel-to-probe physics fixed. Record true derivative FWHM along the wall midpoint and the fitted single-wall widths on the two ideal even diffraction orders. Compare with an exact fixed-slip isolated wall and the analytic finite network.",
              "limits": "Controlled model comparison using an ideal microscope. The second series uses the relative registry gradient in the scan frame and labels each point by its equivalent sinusoidal registry gradient, which is not a pure geometric twist angle for the anisotropic case. The calculation applies published parameters without fitting to new samples.",
              "input_sha256": {p.name: digest(p) for p in paths}, "cases": {}}
    for shape in ("pure_twist", "fixed_ds2_shape"):
        for angle in ANGLES:
            checkpoint_guard()
            g = (rigid_gradient(angle) if shape == "pure_twist"
                 else gradient * np.deg2rad(angle)/ds2_angle_scale)
            result["cases"][f"{shape}/{angle:g}"] = inspect(g, basis, basis[:, 0], 1.25, q, camera, rings)
            result["seconds"] = perf_counter() - started
            atomic_json(args.output, result)
            row = result["cases"][f"{shape}/{angle:g}"]
            print(shape, angle, round(row["row_spacing_nm"], 2), row["true_periodic_width_nm"], flush=True)


if __name__ == "__main__":
    main()
