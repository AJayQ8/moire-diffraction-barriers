"""Periodic continuum relaxation from Van Winkle et al., SI equations 9–14."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.linalg import polar
from scipy.optimize import minimize


# Tables 1 and 2. Elastic moduli are energies per primitive cell. The
# adhesion coefficients are meV per MoS2 primitive cell.
MONOLAYERS = {"MoS2": (0.317, 49.866, 31.548), "WSe2": (0.328, 43.113, 30.770)}
STACKING = {
    ("MoS2", "MoS2", "P"): (27.332, 14.020, -2.542, -0.884, 0.0, 0.0),
    ("MoS2", "MoS2", "AP"): (30.423, 12.322, -2.077, -0.783, 2.397, 0.259),
    ("MoS2", "WSe2", "P"): (32.967, 13.888, -3.281, -0.748, -1.139, -0.175),
    ("MoS2", "WSe2", "AP"): (37.233, 12.317, -2.691, -0.234, 3.535, 0.712),
}
MODES = np.array(((1, 0), (0, 1), (1, 1), (1, 2), (1, -1), (2, 1), (2, 0), (0, 2), (2, 2)))


class ConvergenceError(RuntimeError):
    """A periodic relaxation failed its numerical force check."""


def primitive_basis(lattice_nm):
    """Columns are the primitive lattice vectors in Cartesian coordinates."""
    return lattice_nm * np.array(((np.sqrt(3) / 2, np.sqrt(3) / 2), (-0.5, 0.5)))


def rigid_gradient(twist_degrees, isotropic_mismatch=0.0, shear=0.0, shear_axis_degrees=0.0):
    """Return SR-I in the frame and sign convention of SI equation 13."""
    theta, phi = np.deg2rad((twist_degrees, shear_axis_degrees))
    rotation = np.array(((np.cos(theta), np.sin(theta)), (-np.sin(theta), np.cos(theta))))
    angle = 2 * phi + np.pi / 6 - theta / 2
    diag = shear / 2 * np.cos(angle)
    off_diag = shear / 2 * np.sin(angle)
    stretch = np.array(((1 + isotropic_mismatch + diag, off_diag),
                        (off_diag, 1 + isotropic_mismatch - diag)))
    return stretch @ rotation - np.eye(2)


def stack_energy(disregistry_nm, basis, coefficients_mev):
    """Evaluate adhesion energy and its Cartesian gradient per MoS2 cell."""
    constant, first, second, third, odd_first, odd_third = np.asarray(coefficients_mev) / 1000
    cosine = np.repeat((first, second, third), 3)
    sine = np.array((odd_first, odd_first, -odd_first, 0.0, 0.0, 0.0, -odd_third, -odd_third, odd_third))
    reciprocal = 2 * np.pi * MODES @ np.linalg.inv(basis)
    phase = disregistry_nm @ reciprocal.T
    cos_phase, sin_phase = np.cos(phase), np.sin(phase)
    values = constant + cos_phase @ cosine + sin_phase @ sine
    gradient = (-sin_phase * cosine + cos_phase * sine) @ reciprocal
    return values, gradient


def elastic_parameters_per_reference_cell(first, second):
    """Eliminate the common displacement mode of the two isotropic layers."""
    ref_area = np.linalg.det(primitive_basis(MONOLAYERS["MoS2"][0]))

    def moduli(material):
        a, bulk, shear = MONOLAYERS[material]
        scale = ref_area / np.linalg.det(primitive_basis(a))
        return (bulk + shear) * scale, shear * scale

    longitudinal1, transverse1 = moduli(first)
    longitudinal2, transverse2 = moduli(second)
    longitudinal = 1 / (1 / longitudinal1 + 1 / longitudinal2)
    transverse = 1 / (1 / transverse1 + 1 / transverse2)
    return longitudinal - transverse, transverse


@dataclass(frozen=True)
class PeriodicGrid:
    basis: np.ndarray
    cell: np.ndarray
    fractional: np.ndarray
    k: np.ndarray
    stiffness: np.ndarray
    change_of_variables: np.ndarray


def wavevectors(cell, n):
    frequency = 2 * np.pi * np.fft.fftfreq(n, d=1 / n)
    return np.stack(np.meshgrid(frequency, frequency, indexing="ij"), axis=-1) @ np.linalg.inv(cell)


def make_grid(gradient, basis, n, bulk, shear):
    cell = np.linalg.solve(gradient, basis)
    t = np.arange(n) / n
    fractional = np.stack(np.meshgrid(t, t, indexing="ij"), axis=-1)
    k = wavevectors(cell, n)
    k_outer = k[..., :, None] * k[..., None, :]
    stiffness = shear * np.sum(k**2, axis=-1)[..., None, None] * np.eye(2) + bulk * k_outer
    # For an even grid, average the two representations of a Nyquist pair
    # to make the real-valued elastic Hessian self-adjoint and positive.
    opposite = (-np.arange(n)) % n
    stiffness = 0.5 * (stiffness + stiffness[np.ix_(opposite, opposite)])
    eigenvalues, eigenvectors = np.linalg.eigh(stiffness)
    # Only changes the optimizer's variables. The positive shift roughly
    # balances adhesion curvature at low frequency.
    weights = 1 / np.sqrt(np.maximum(eigenvalues, 0.0) + 12.0)
    change = (eigenvectors * weights[..., None, :]) @ eigenvectors.swapaxes(-1, -2)
    change[0, 0] = 0.0  # Choose the origin of the periodic moiré pattern.
    return PeriodicGrid(basis, cell, fractional, k, stiffness, change)


def apply_symbol(symbol, field):
    spectrum = np.fft.fft2(field, axes=(0, 1))
    transformed = np.einsum("...ij,...j->...i", symbol, spectrum)
    return np.fft.ifft2(transformed, axes=(0, 1)).real


def energy_gradient(grid, coefficients, displacement):
    elastic_gradient = apply_symbol(grid.stiffness, displacement)
    stack, stack_gradient = stack_energy(grid.fractional @ grid.basis.T + displacement, grid.basis, coefficients)
    elastic = 0.5 * np.mean(np.sum(displacement * elastic_gradient, axis=-1))
    adhesion = float(stack.mean())
    return elastic + adhesion, elastic_gradient + stack_gradient, (float(elastic), adhesion)


def relax(gradient, basis, interface, n, guard, coefficients=None):
    """Relax the interlayer displacement on one measured periodic moiré cell."""
    bulk, shear = elastic_parameters_per_reference_cell(*interface[:2])
    grid = make_grid(np.asarray(gradient), np.asarray(basis), n, bulk, shear)
    coefficients = STACKING[interface] if coefficients is None else coefficients
    shape = (n, n, 2)

    def objective(y):
        relative = apply_symbol(grid.change_of_variables, y.reshape(shape))
        energy, physical_gradient, _ = energy_gradient(grid, coefficients, relative)
        return energy, (apply_symbol(grid.change_of_variables, physical_gradient) / (n * n)).ravel()

    start = np.zeros(shape)
    initial_energy, _, _ = energy_gradient(grid, coefficients, start)
    iteration = 0

    def callback(_):
        nonlocal iteration
        iteration += 1
        if iteration % 25 == 0:
            guard()

    fit = minimize(objective, start.ravel(), jac=True, method="L-BFGS-B", callback=callback,
                   options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 750, "maxcor": 10})
    relative = apply_symbol(grid.change_of_variables, fit.x.reshape(shape))
    energy, residual, (elastic, adhesion) = energy_gradient(grid, coefficients, relative)
    residual -= residual.mean(axis=(0, 1), keepdims=True)
    guard()
    return grid, relative, {
        "success": bool(fit.success),
        "termination": str(fit.message),
        "iterations": int(fit.nit),
        "energy_eV_per_cell": float(energy),
        "rigid_energy_eV_per_cell": float(initial_energy),
        "elastic_eV_per_cell": elastic,
        "adhesion_eV_per_cell": adhesion,
        "force_rms_eV_per_nm": float(np.sqrt(np.mean(residual**2))),
        "max_relative_shift_nm": float(np.linalg.norm(relative, axis=-1).max()),
    }


def gradient_field(grid, displacement):
    """Return output-component × Cartesian-axis derivatives on the periodic cell."""
    spectrum = np.fft.fft2(displacement, axes=(0, 1))
    derivative = 1j * spectrum[..., :, None] * grid.k[..., None, :]
    return np.fft.ifft2(derivative, axes=(0, 1)).real


def sample_disregistry(grid, relative, rigid_field, observation_cell, probe_sigma_nm=0.0):
    """Evaluate affine-plus-relaxed disregistry at observed raster positions."""
    n = relative.shape[0]
    k = wavevectors(observation_cell, n)
    transfer = np.exp(-0.5 * probe_sigma_nm**2 * np.sum(k**2, axis=-1))
    blurred = np.fft.ifft2(transfer[..., None] * np.fft.fft2(relative, axes=(0, 1)), axes=(0, 1)).real
    frac = rigid_field @ np.linalg.inv(grid.basis).T
    coordinates = n * np.moveaxis(frac, -1, 0)
    perturbation = np.stack([ndi.map_coordinates(blurred[..., i], coordinates, order=3, mode="grid-wrap")
                             for i in range(2)], axis=-1)
    return rigid_field + perturbation


def disregistry_predictions(gradient, basis, rigid_field, interface, grid_size, probe_fwhm_nm, guard,
                            include_linear_response=False):
    """Compute matched relaxation predictions and two geometric baselines."""
    probe_sigma = probe_fwhm_nm / (2 * np.sqrt(2 * np.log(2)))
    full_grid, full_shift, full_receipt = relax(gradient, basis, interface, grid_size, guard)
    # Remove the directional stretch from the relaxation problem but keep
    # the measured cell and bulk affine field in the observation. This
    # tests whether changing the relaxation, rather than merely stretching
    # an isotropically relaxed pattern, improves the local prediction.
    rotation, stretch = polar(np.eye(2) + gradient)
    if np.linalg.det(rotation) <= 0 or np.linalg.det(stretch) <= 0:
        raise ValueError("The measured affine deformation is not orientation preserving")
    symmetric = rotation * np.sqrt(np.linalg.det(stretch)) - np.eye(2)
    iso_grid, iso_shift, iso_receipt = relax(symmetric, basis, interface, grid_size, guard)
    fields = {
        "full": sample_disregistry(full_grid, full_shift, rigid_field, full_grid.cell, probe_sigma),
        "isotropic_relaxation": sample_disregistry(iso_grid, iso_shift, rigid_field, full_grid.cell, probe_sigma),
        "rigid": rigid_field,
    }
    if include_linear_response:
        # Retain the adhesion force at the unrelaxed registry. This is the
        # linear elastic response without feedback from the ensuing shift.
        _, force = stack_energy(full_grid.fractional @ basis.T, basis, STACKING[interface])
        stiffness = full_grid.stiffness.copy()
        stiffness[0, 0] = np.eye(2)
        compliance = np.linalg.inv(stiffness)
        compliance[0, 0] = 0
        linear_shift = -apply_symbol(compliance, force)
        fields["linear_response"] = sample_disregistry(full_grid, linear_shift, rigid_field,
                                                       full_grid.cell, probe_sigma)
    receipts = {"full": full_receipt, "isotropic_relaxation": iso_receipt}
    if any(not row["success"] or row["force_rms_eV_per_nm"] > 2e-4 for row in receipts.values()):
        raise ConvergenceError(f"One of the periodic solutions failed its force check: {receipts}")
    return fields, receipts


def projected_observables(relative_gradient):
    """Author-convention curl, dilation and shear of half the relative field."""
    gradient = 0.5 * relative_gradient
    rotation = np.rad2deg(gradient[..., 1, 0] - gradient[..., 0, 1])
    dilation = 100 * (gradient[..., 0, 0] + gradient[..., 1, 1])
    shear = 100 * np.hypot(gradient[..., 0, 0] - gradient[..., 1, 1],
                          gradient[..., 1, 0] + gradient[..., 0, 1])
    return rotation, dilation, shear
