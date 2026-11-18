"""Infer a monotone wall profile and cross-check it between diffraction rings."""

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares

from run_wall_readout_check import intensity


def detrended(distance, profile, smooth_nm):
    """Remove a local linear background estimated on the two domain tails."""
    tail = np.abs(distance) >= 0.82 * distance[-1]
    design = np.column_stack((distance, np.ones_like(distance)))
    coefficients = np.linalg.lstsq(design[tail], profile[tail], rcond=None)[0]
    residual = profile - design @ coefficients
    tail_residual = residual[tail]
    mad = np.median(np.abs(tail_residual - np.median(tail_residual, axis=0)), axis=0)
    sigma = smooth_nm / (distance[1] - distance[0])
    return ndi.gaussian_filter1d(residual, sigma, axis=0, mode="nearest"), 1.4826 * mad


def wall_disregistry(distance, center, width, lattice_edge, family):
    """A straight partial dislocation with the prescribed derivative FWHM."""
    slip_half = np.array((-lattice_edge[1], lattice_edge[0])) / (2 * np.sqrt(3))
    if family == "sine_gordon":
        length = width / (2 * np.arccosh(2))
        fraction = 2 / np.pi * np.arcsin(np.tanh((distance - center) / length))
    else:
        length = width / (2 * np.arccosh(np.sqrt(2)))
        fraction = np.tanh((distance - center) / length)
    return 0.5 * lattice_edge + fraction[..., None] * slip_half


def wall_detector(distance, center, width, lattice_edge, family, reciprocal, scale, offset, probe_nm):
    field = wall_disregistry(distance, center, width, lattice_edge, family)
    sigma = probe_nm / (np.sqrt(8 * np.log(2)) * (distance[1] - distance[0]))
    return ndi.gaussian_filter1d(intensity(field, reciprocal, scale, offset), sigma, axis=0, mode="nearest")


def ring_widths(distance, measured, simulated, lattice_edge, reciprocal, scale, offset, contrast,
                rings, probe_nm, smooth_nm):
    """Cross-predict profile shape while allowing a local gain in each ring."""
    informative = [indices[np.abs(contrast[indices]) >= 0.25 * np.abs(contrast[indices]).max()]
                   for indices in rings if len(indices)]
    if len(informative) != 2 or any(np.max(np.abs(contrast[indices])) < 1e-6 for indices in informative):
        return {}

    def profiles(detector):
        values = detrended(distance, detector, smooth_nm)[0]
        norm = np.linalg.norm(values, axis=0)
        normalized = values * np.sign(contrast) / np.maximum(norm, 1e-12)
        return [normalized[:, indices] / np.sqrt(len(indices)) for indices in informative]

    targets = {"measured": profiles(measured), "fixed_model": profiles(simulated)}
    reach = distance[-1]
    center_bound = 0.70 * reach
    width_bounds = (0.8, 1.6 * reach)
    if width_bounds[1] <= width_bounds[0]:
        return {}
    start_width = min(4.5, 0.90 * width_bounds[1])

    def solve(family, target, select, fixed_width=None):
        peak = distance[int(np.argmax(sum(target[i].mean(axis=1) for i in select)))]
        center_guess = np.clip(peak, -0.90 * center_bound, 0.90 * center_bound)

        def predict(parameters):
            center = parameters[0]
            width = parameters[1] if fixed_width is None else fixed_width
            return profiles(wall_detector(distance, center, width, lattice_edge, family,
                                          reciprocal, scale, offset, probe_nm))

        def selected_residual(parameters):
            predictions = predict(parameters)
            return np.concatenate([(predictions[i] - target[i]).ravel() for i in select])

        if fixed_width is None:
            start = [center_guess, start_width]
            bounds = ([-center_bound, width_bounds[0]], [center_bound, width_bounds[1]])
        else:
            start = [center_guess]
            bounds = ([-center_bound], [center_bound])
        result = least_squares(selected_residual, start, bounds=bounds, max_nfev=50)
        residual = [float(np.linalg.norm(p - t)) for p, t in zip(predict(result.x), target)]
        width = result.x[1] if fixed_width is None else fixed_width
        return {"center_nm": float(result.x[0]), "intrinsic_gradient_fwhm_nm": float(width),
                "shape_residual_by_ring": residual, "success": bool(result.success),
                "active_bound": bool(np.any(result.active_mask != 0))}

    result = {"strong_channels_by_ring": [len(indices) for indices in informative]}
    for family in ("sine_gordon", "tanh"):
        fixed = solve(family, targets["fixed_model"], [0, 1])
        observed = solve(family, targets["measured"], [0, 1])
        wide_width = fixed["intrinsic_gradient_fwhm_nm"] + 1.8
        tests = []
        for train in (0, 1):
            alternative = solve(family, targets["measured"], [train])
            baseline = solve(family, targets["measured"], [train], fixed["intrinsic_gradient_fwhm_nm"])
            held_out = 1 - train
            test = {"training_ring": train + 1, "free_width": alternative, "fixed_width": baseline,
                    "held_out_shape_error_gain": baseline["shape_residual_by_ring"][held_out]
                    - alternative["shape_residual_by_ring"][held_out]}
            if wide_width < width_bounds[1]:
                wide = solve(family, targets["measured"], [train], wide_width)
                test["widened_by_1_8_nm"] = wide
                test["widening_held_out_error_change"] = (wide["shape_residual_by_ring"][held_out]
                                                          - baseline["shape_residual_by_ring"][held_out])
            tests.append(test)
        result[family] = {"measured": observed, "fixed_model": fixed, "ring_transfer": tests}
    return result
