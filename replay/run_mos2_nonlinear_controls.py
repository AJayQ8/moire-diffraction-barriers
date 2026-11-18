"""Test the MoS2 diffraction gain against simpler solves and published inputs."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from author_release import load_author_release
from bilayer_relaxation import MODES, STACKING, apply_symbol, energy_gradient, make_grid, sample_disregistry, stack_energy
from observe_bilayers import load_dataset
from run_heterobilayer_sparse_geometry import solve_from_geometry
from run_heterobilayer_template_transfer import design_matrix
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_regime_benchmark import SETTINGS, detector_data
from run_mos2_template_transfer import ANTIPARALLEL, BENCHMARK, PARALLEL, make_context, score
from shared_compute import atomic_json, checkpoint_guard


SOURCE = "https://arxiv.org/abs/1911.12804"
ENALDIEV = {
    "first_cosine_eV_nm2": 71928800, "first_second_term_eV_nm2": 56411,
    "decay_length_nm": 0.0496, "average_interlayer_gap_nm": 0.65, "height_curvature_eV_nm4": 214,
    "young_modulus_gpa": 277, "poisson_ratio": 0.27,
}


def alternative_inputs(basis, orientation):
    """Convert Eq. 2 and Tables S1/S3 into the shared cell-energy convention."""
    a = np.linalg.norm(basis[:, 0])
    g = 4*np.pi / (np.sqrt(3) * a)
    q = np.hypot(g, 1/ENALDIEV["decay_length_nm"])
    d = ENALDIEV["average_interlayer_gap_nm"]
    height_curvature = ENALDIEV["height_curvature_eV_nm4"]
    c1 = ENALDIEV["first_cosine_eV_nm2"] * np.exp(-q*d)
    c2 = ENALDIEV["first_second_term_eV_nm2"] * np.exp(-g*d)
    nu = ENALDIEV["poisson_ratio"]
    # A GPa nm equals 1/0.1602176634 eV/nm^2. The shared relative
    # displacement stiffness for identical layers is half the monolayer's.
    young = ENALDIEV["young_modulus_gpa"] * d / 0.1602176634
    shear = young / (2*(1+nu))
    lame = young * nu / ((1+nu)*(1-2*nu))
    area = np.linalg.det(basis)
    moduli = ((lame+shear)*area/2, shear*area/2)
    points = np.arange(16) / 16
    v, w = np.meshgrid(2*np.pi*points, 2*np.pi*points, indexing="ij")
    cosine = np.cos(v) + np.cos(w) + np.cos(v+w)
    sine = np.sin(v) + np.sin(w) - np.sin(v+w)
    second = cosine if orientation == "P" else sine
    density = c1*cosine + c2*second - (q*c1*cosine + g*c2*second)**2/(4*height_curvature)
    features = np.stack((
        np.ones_like(v), cosine, np.cos(v+2*w)+np.cos(v-w)+np.cos(2*v+w),
        np.cos(2*v)+np.cos(2*w)+np.cos(2*v+2*w), sine,
        -np.sin(2*v)-np.sin(2*w)+np.sin(2*v+2*w)), axis=-1)
    energy = (area * density).ravel()
    matrix = features.reshape(-1, features.shape[-1])
    coefficients = np.linalg.lstsq(matrix, energy, rcond=None)[0]
    if np.max(np.abs(matrix @ coefficients - energy)) > 1e-12:
        raise ValueError("The published alternative energy does not fit the shared Fourier basis")
    # The two tabulations use different real-space reference axes. Match
    # XX to the high-energy origin and 2H to the low-energy AP registry.
    stacks = np.array(((0, 0), (1/3, 1/3), (2/3, 2/3)))
    energies, _ = stack_energy(stacks @ basis.T, basis, coefficients * 1000)
    if np.argmax(energies) != 0 or (orientation == "AP" and np.argmin(energies) != 2):
        raise ValueError("The alternate stacking energies do not match the stated registry convention")
    return coefficients*1000, moduli, energies.tolist()


def minimize_modes(grid, coefficients, modes, seed):
    """Solve the nonlinear energy in a specified Fourier subspace."""
    coordinates = grid.fractional @ grid.basis.T
    design = design_matrix(coordinates, grid.basis, modes)
    rows = design.reshape(-1, design.shape[-1])
    size = rows.shape[1]
    initial = np.linalg.lstsq(rows, seed.reshape(-1, 2), rcond=None)[0]
    # Each sine/cosine pair has the same 2x2 elastic stiffness. Scale the
    # optimizer variables using the existing spectral preconditioner.
    h, k = modes.T
    change = np.tile(grid.change_of_variables[h, k], (2, 1, 1))

    def physical(y):
        return design @ np.einsum("nij,nj->ni", change, y.reshape(size, 2))

    def objective(y):
        field = physical(y)
        energy, force, _ = energy_gradient(grid, coefficients, field)
        gradient = rows.T @ force.reshape(-1, 2) / len(rows)
        return energy, np.einsum("nij,nj->ni", change, gradient).ravel()

    start = np.linalg.solve(change, initial[..., None])[..., 0].ravel()
    direction = np.random.default_rng(2989).normal(size=start.shape)
    direction /= np.linalg.norm(direction)
    before, jacobian = objective(start)
    central = (objective(start+1e-5*direction)[0] - objective(start-1e-5*direction)[0]) / 2e-5
    if abs(jacobian @ direction - central) > 1e-7:
        raise ValueError("The restricted Fourier force failed its derivative check")
    fit = minimize(objective, start, jac=True, method="L-BFGS-B",
                   callback=lambda _: checkpoint_guard(),
                   options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-10})
    if not fit.success or fit.fun > before + 1e-12:
        raise ValueError(f"The restricted Fourier equilibrium failed: {fit.message}")
    return physical(fit.x), {
        "degrees_of_freedom": int(start.size), "iterations": int(fit.nit),
        "starting_energy": float(before), "final_energy": float(fit.fun),
        "derivative_error": float(abs(jacobian @ direction - central)),
    }


def minimize_all(grid, coefficients):
    """Use the qualified periodic minimizer with the alternate published moduli."""
    shape = (*grid.fractional.shape[:2], 2)

    def objective(y):
        field = apply_symbol(grid.change_of_variables, y.reshape(shape))
        energy, force, _ = energy_gradient(grid, coefficients, field)
        return energy, (apply_symbol(grid.change_of_variables, force) / np.prod(shape[:2])).ravel()

    steps = 0

    def callback(_):
        nonlocal steps
        steps += 1
        if steps % 25 == 0:
            checkpoint_guard()

    start = np.zeros(np.prod(shape))
    fit = minimize(objective, start, jac=True, method="L-BFGS-B", callback=callback,
                   options={"maxiter": 750, "maxcor": 10, "ftol": 1e-15, "gtol": 1e-10})
    field = apply_symbol(grid.change_of_variables, fit.x.reshape(shape))
    energy, force, _ = energy_gradient(grid, coefficients, field)
    force -= force.mean(axis=(0, 1), keepdims=True)
    rms_force = np.sqrt(np.mean(force**2))
    if not fit.success or energy >= objective(start)[0] or rms_force > 2e-4:
        raise ValueError("The alternative equilibrium failed its force or energy check")
    return field, {"iterations": int(fit.nit), "final_energy": float(energy), "force_rms_eV_nm": float(rms_force)}


def controls(context, reference):
    interface = tuple(reference["interface"])
    gradient = np.asarray(reference["affine_geometry"]["gradient_crystal"])
    basis = context["basis"]
    rigid = context["fields"]["rigid"]
    origin = basis @ np.full(2, 2/3 if interface[-1] == "AP" else 0)
    grid, full, _ = solve_from_geometry(gradient, basis, interface)
    coefficients = STACKING[interface]
    diagnostics = {}
    for name, modes in (("nonlinear_first_harmonic", MODES[:3]), ("nonlinear_three_harmonics", MODES)):
        field, diagnostics[name] = minimize_modes(grid, coefficients, modes, full)
        context["fields"][name] = sample_disregistry(grid, field, origin+rigid, grid.cell) - origin
    alternative, moduli, energies = alternative_inputs(basis, interface[-1])
    diagnostic = {"adhesion_coefficients_meV_cell": alternative.tolist(), "effective_moduli_eV_cell": moduli,
                  "high_symmetry_energies_eV_cell": energies}
    diagnostics["alternative_input"] = diagnostic
    n = full.shape[0]
    for name, candidate_grid in (("alternative_adhesion", grid), ("alternative_adhesion_and_elasticity",
                                                                 make_grid(gradient, basis, n, *moduli))):
        field, diagnostics[name] = minimize_all(candidate_grid, alternative)
        context["fields"][name] = sample_disregistry(candidate_grid, field, origin+rigid, candidate_grid.cell) - origin
    return {"interface": interface, "solvers": diagnostics, "scores": score(context, {}, reference)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    settings = json.loads(SETTINGS.read_text())
    benchmark = json.loads(BENCHMARK.read_text())
    for name, expected in benchmark["input_sha256"].items():
        if digest(PROJECT / name) != expected:
            raise ValueError(f"The native replay input {name} changed")
    parallel = json.loads(PARALLEL.read_text())["batches"]
    antiparallel = json.loads(ANTIPARALLEL.read_text())
    cases = (*parallel["development"], *parallel["held_out"], *antiparallel["development"], *antiparallel["replay"])
    author = load_author_release(args.release, args.dependencies)
    inputs = (Path(__file__), SETTINGS, BENCHMARK, PARALLEL, ANTIPARALLEL, PROJECT / "run_mos2_template_transfer.py")
    output = {
        "scope": "Retrospective controls on the 24 unique MoS2 maps. Same measured local geometry and diffraction support. The restricted model minimizes the full energy with the stated harmonic basis, seeded by the projection of the full solution. The alternative uses independently published 2020 adhesion with the same elasticity or the source's elasticity. No experimental fit of any material parameters.",
        "alternative_source": SOURCE, "alternative_parameters": ENALDIEV,
        "input_sha256": {p.name: digest(p) for p in inputs}, "scans": {}, "failed_scans": {},
    }
    for scan in cases:
        checkpoint_guard()
        try:
            data = load_dataset(scan, args.archive, allowed_ids=cases, allow_author_crop=True)
            reference = benchmark["scans"][scan]
            if data.archive_md5 != reference["archive_md5"]:
                raise ValueError("The MoS2 archive changed")
            context = make_context(data, detector_data(args.archive, scan, author), reference, settings)
            output["scans"][scan] = controls(context, reference)
            print(scan, {k: v["both"] for k, v in output["scans"][scan]["scores"].items()}, flush=True)
        except (ValueError, np.linalg.LinAlgError) as exc:
            output["failed_scans"][scan] = str(exc)
            print(scan, str(exc), flush=True)
        output["wall_seconds"] = perf_counter() - start
        checkpoint_guard()
        atomic_json(args.output, output)


if __name__ == "__main__":
    main()
