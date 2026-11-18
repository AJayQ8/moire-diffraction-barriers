"""Minimize a small Fourier model directly and test its wall predictions."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from author_release import load_author_release
from bilayer_relaxation import MODES, STACKING, elastic_parameters_per_reference_cell, stack_energy
from observe_bilayers import load_dataset, read_input_metadata
from run_heterobilayer_template_transfer import design_matrix
from run_heterobilayer_transfer import PROJECT, digest
from run_moire_common_intensity import BENCHMARK, averages, calibrate_and_score, centers_and_context
from run_moire_regime_benchmark import SETTINGS
from run_mos2_intensity_only_replay import disk_images
from run_wall_heterostrain import PROTOCOL, SOURCE, wall_scores
from shared_compute import atomic_json, checkpoint_guard


REFERENCE = PROJECT / "mos2_nonlinear_mechanics_v01.json"


@dataclass(frozen=True)
class FourierModel:
    modes: np.ndarray
    basis: np.ndarray
    stiffness: np.ndarray
    parameterization: np.ndarray
    adhesion: tuple

    @classmethod
    def from_geometry(cls, gradient, basis, interface, modes, transverse=False):
        """Build the physical elastic stiffness at the retained wavevectors."""
        bulk, shear = elastic_parameters_per_reference_cell(*interface[:2])
        cell = np.linalg.solve(gradient, basis)
        k = 2 * np.pi * modes @ np.linalg.inv(cell)
        norm = np.linalg.norm(k, axis=1)
        longitudinal = k / norm[:, None]
        rotation = longitudinal[:, [1, 0]] * (-1, 1)
        orientation = np.stack((rotation, longitudinal), axis=-1)
        eigenvalues = norm[:, None]**2 * np.array((shear, bulk + shear))
        stiffness = (orientation * eigenvalues[:, None, :]) @ orientation.swapaxes(-1, -2)
        param = orientation / np.sqrt(eigenvalues[:, None, :] + 12.0)
        if transverse:
            param = param[..., :1]
        return cls(modes, basis, np.tile(stiffness, (2, 1, 1)), np.tile(param, (2, 1, 1)),
                   STACKING[interface])

    def problem(self, points):
        """Integrate the stacking energy on a uniform quadrature grid."""
        t = np.arange(points) / points
        coordinate = np.stack(np.meshgrid(t, t, indexing="ij"), axis=-1).reshape(-1, 2) @ self.basis.T
        design = design_matrix(coordinate, self.basis, self.modes)

        def objective(parameters):
            c = self.coefficients(parameters)
            elastic_force = np.einsum("mij,mj->mi", self.stiffness, c) / 2
            energy, force = stack_energy(coordinate + design @ c, self.basis, self.adhesion)
            coefficient_force = elastic_force + design.T @ force / len(design)
            gradient = np.einsum("mij,mi->mj", self.parameterization, coefficient_force)
            return float(energy.mean() + (c * elastic_force).sum() / 2), gradient.ravel()

        return objective

    def coefficients(self, parameters):
        return np.einsum("mij,mj->mi", self.parameterization,
                         parameters.reshape(len(self.parameterization), -1))

    def displacement(self, coordinates, parameters):
        return design_matrix(coordinates, self.basis, self.modes) @ self.coefficients(parameters)

    def solve(self, quadrature=48, seed=None):
        start_time = perf_counter()
        objective = self.problem(quadrature)
        start = np.zeros(self.parameterization.shape[0] * self.parameterization.shape[-1]) if seed is None else seed
        initial, _ = objective(start)
        fit = minimize(objective, start, jac=True, method="L-BFGS-B", callback=lambda _: checkpoint_guard(),
                       options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-11})
        checked_energy, checked_force = self.problem(2 * quadrature)(fit.x)
        residual = float(np.linalg.norm(checked_force))
        if not fit.success or fit.fun > initial + 1e-12 or residual > 3e-7:
            raise ValueError(f"Restricted equilibrium did not qualify: {fit.message}, force {residual:.3g}")
        direction = np.random.default_rng(57133).normal(size=start.shape)
        direction /= np.linalg.norm(direction)
        _, analytical = objective(fit.x)
        step = 1e-5
        finite = (objective(fit.x + step * direction)[0] - objective(fit.x - step * direction)[0]) / (2 * step)
        derivative_error = float(abs(finite - analytical @ direction))
        if derivative_error > 1e-8:
            raise ValueError(f"Restricted equilibrium derivative discrepancy: {derivative_error:.3g}")
        return fit.x, {"dof": int(start.size), "iterations": int(fit.nit), "quadrature": quadrature,
                       "energy_eV_per_cell": checked_energy, "fine_quadrature_force_norm": residual,
                       "derivative_error": derivative_error, "wall_seconds": perf_counter() - start_time}


def evaluate(context, data, archive, images, settings, interface, reference):
    gradient, basis, rigid = (context[k] for k in ("gradient", "basis", "rigid"))
    phase = rigid + basis @ np.full(2, 2/3 if interface[-1] == "AP" else 0)
    models = {}
    fields = {}
    coefficients = {}
    for label, modes, transverse in (("first_star", MODES[:3], False), ("three_stars", MODES, False),
                                       ("three_stars_transverse", MODES, True)):
        model = FourierModel.from_geometry(gradient, basis, interface, modes, transverse)
        parameters, models[label] = model.solve()
        coefficients[label] = model.coefficients(parameters).tolist()
        fields[label] = rigid + model.displacement(phase, parameters)
        if label == "three_stars":
            longitudinal = model.parameterization[..., 1:2]
            longitudinal_only = longitudinal[..., 0] * parameters.reshape(-1, 2)[:, 1, None]
            fields["three_stars_transverse_projection"] = (
                fields[label] - design_matrix(phase, basis, MODES) @ longitudinal_only
            )
            models[label]["longitudinal_displacement_power_fraction"] = float(
                np.sum(longitudinal_only**2) / np.sum(model.coefficients(parameters)**2)
            )
            reversed_result, models["three_stars_reversed_start"] = model.solve(seed=-parameters)
            models["three_stars_reversed_start"]["coefficient_delta_nm"] = float(
                np.linalg.norm(model.coefficients(reversed_result) - model.coefficients(parameters))
            )

    scores = wall_scores(data, archive, context, fields, settings)
    diffraction = averages(calibrate_and_score(fields, context, images, context["support"]))
    # Compare with the frozen full-grid quadrature and full-solution starts.
    difference = {}
    for name, old in (("first_star", "first_star_equilibrium"), ("three_stars", "three_star_equilibrium")):
        key = "directional_rmse_pp"
        difference[name] = {"shear_pp": scores["models"][name][key] - reference["wall_scores"]["models"][old][key],
                            "diffraction_rmse": diffraction[name] - reference["second_ring_mean_rmse"][old]}
    return {"solver": models, "coefficients_nm": coefficients, "wall_scores": scores,
            "second_ring_mean_rmse": diffraction, "comparison_to_full_field_seed": difference}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("release", type=Path)
    cli.add_argument("dependencies", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--pilot", action="store_true")
    args = cli.parse_args()
    paths = (PROTOCOL, SOURCE, BENCHMARK, SETTINGS, REFERENCE)
    protocol, source, benchmark, settings, reference = (json.loads(p.read_text()) for p in paths)
    author = load_author_release(args.release, args.dependencies)
    result = {
        "scope": "Minimize the same published in-plane energy with 12 or 36 Fourier amplitudes, starting from rigid layers. Evaluate the resulting fields directly on the same observed geometry. Check the three-star branch from reversed displacement and the effect of removing longitudinal motion by projection or re-equilibration.",
        "limits": "This is a retrospective model comparison on the examined MoS2 maps. Common phase, detector and spatial observations match the original benchmark. Comparisons to the earlier modes include a change from bicubic interpolation to direct evaluation.",
        "input_sha256": {p.name: digest(p) for p in (Path(__file__), *paths)}, "scans": {},
    }
    for group, scans in protocol["cases"].items():
        for scan in scans:
            if args.pilot and scan not in ("DS7", "DS1", "DS2", "DS6", "DS13", "DS15", "DS20", "DS26"):
                continue
            checkpoint_guard()
            record = source["scans"][scan]
            interface = tuple(record["interface"])
            archive = benchmark["scans"][scan]
            data = load_dataset(scan, args.archive, allowed_ids=(scan,), allow_author_crop=True)
            metadata = read_input_metadata(scan, args.archive)["metadata_fields"]
            images, modes, fingerprint = disk_images(args.archive, scan, author)
            if fingerprint != record["diskset_sha256"] or data.archive_md5 != archive["archive_md5"]:
                raise ValueError(f"{scan}: source changed")
            context = centers_and_context(images, modes, metadata, settings, interface)
            if context["receipt"] != record["geometry"]:
                raise ValueError(f"{scan}: geometry changed")
            row = evaluate(context, data, archive, images, settings, interface, reference["scans"][scan])
            result["scans"][scan] = {"group": group, **row}
            atomic_json(args.output, result)
            print(scan, row["comparison_to_full_field_seed"], row["solver"]["three_stars_reversed_start"], flush=True)


if __name__ == "__main__":
    main()
