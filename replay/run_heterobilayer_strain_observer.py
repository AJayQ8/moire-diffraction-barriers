"""Compare reciprocal, phase-gradient and model strain on the same DS31 blocks."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter

from bilayer_relaxation import sample_disregistry
from observe_bilayers import load_dataset
from run_heterobilayer_corrugation import BASIS, MODEL_A, ORIGIN, dilation
from run_heterobilayer_raw_ds31 import SCAN
from run_heterobilayer_raw_injection import HIGH, RAW
from run_heterobilayer_raw_spatial_replay import ORIGINAL_ARCS
from run_heterobilayer_ring_geometry import smooth_geometry
from run_heterobilayer_transfer import PROJECT, PROTOCOL, blur, digest, fingerprints, solve, verify_freeze
from shared_compute import atomic_json, checkpoint_guard


SPATIAL = PROJECT / "heterobilayer_raw_spatial_replay_v01.json"
TRANSFER = PROJECT / "heterobilayer_transfer_development_v01.json"
FREEZE = PROJECT / "heterobilayer_transfer_freeze_v01.json"
SETTINGS = PROJECT / "registry_protocol_v01.json"


def trace_field(field, data, sigma_pixels=0.0):
    if sigma_pixels:
        valid = data.source_valid_mask
        weight = gaussian_filter(valid.astype(float), sigma_pixels, mode="constant")
        field = blur(field, valid, weight, sigma_pixels)
    return dilation(field, data)


def use_blocks(original, high, arcs, spatial):
    current = []
    arc_data = arcs["groups"]["higher_order_9"]["blocks"]
    for raw, disk, arc in zip(original["blocks"], high["blocks"], arc_data, strict=True):
        indices = (raw["row"], raw["column_start"])
        if (disk["row"], disk["column_start"]) != indices or (arc["row"], arc["column_start"]) != indices:
            raise ValueError("Original block matching changed")
        current.append({
            **{k: raw[k] for k in ("row", "column_start", "column_count", "matched_model_dilations_percent")},
            "paired": disk["higher_order_9"]["mean"]["relative_trace_percent"],
            "arc": arc["relative_trace_percent"],
        })
    replay = [{
        **{k: b[k] for k in ("row", "column_start", "column_count", "matched_model_dilations_percent")},
        "paired": b["paired"]["mean"], "arc": b["arcs"]["mean"],
    } for b in spatial["blocks"] if "matched_model_dilations_percent" in b]
    return current, replay


def fit_response(signal, target):
    design = np.column_stack((signal, np.ones(len(signal))))
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    return {"slope": float(coefficients[0]), "intercept": float(coefficients[1])}


def assess(rows, estimates=None):
    names = rows[0]["traces"].keys()
    series = {k: np.array([r["traces"][k] for r in rows]) for k in names}
    measurements = {k: np.array([r[k] for r in rows]) for k in ("paired", "arc")}
    candidates = {}
    gains = {}
    for name, prediction in series.items():
        candidates[name] = {"variability_pp": float(prediction.std()), "score": {}}
        gains[name] = {}
        for readout, target in measurements.items():
            gains[name][readout] = fit_response(prediction, target) if prediction.std() > 1e-10 else None
            fitted = estimates[name][readout] if estimates is not None else None
            replay = None if fitted is None else fitted["slope"]*prediction + fitted["intercept"]
            candidates[name]["score"][readout] = {
                "centered_rmse_pp": float((prediction - target).std()),
                "correlation": float(np.corrcoef(prediction, target)[0, 1]) if prediction.std() > 1e-10 else None,
                "source_scale_centered_rmse_pp": float((replay - target).std()) if replay is not None else None,
                "source_scale_absolute_rmse_pp": float(np.sqrt(np.mean((replay - target)**2))) if replay is not None else None,
            }
    return {
        "rows": len(rows), "variability_pp": {k: float(v.std()) for k, v in measurements.items()},
        "phase_and_model": candidates, "local_projection_diagnostic": gains,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    protocol = json.loads(PROTOCOL.read_text())
    verify_freeze(FREEZE, fingerprints(), protocol["devices"]["development"])
    settings = json.loads(SETTINGS.read_text())
    data = load_dataset(SCAN, args.archive, allowed_ids=(SCAN,), allow_author_crop=True)
    prior = json.loads(TRANSFER.read_text())["scans"][SCAN]
    if data.archive_md5 != prior["archive_md5"]:
        raise ValueError("The DS31 map changed")
    raw, high, arcs, spatial = (json.loads(p.read_text()) for p in (RAW, HIGH, ORIGINAL_ARCS, SPATIAL))
    original, replay = use_blocks(raw, high, arcs, spatial)
    fit = np.asarray(prior["whole_geometry"]["fit_scan"])
    yy, xx = np.indices(data.u_nm.shape[:2], dtype=float)
    points = np.stack((xx, yy), axis=-1) * data.pixel_nm
    nominal = points @ fit[:2] + fit[-1]
    warp, _ = smooth_geometry(data, fit, points)
    scale = MODEL_A / data.lattice_nm
    interface = tuple(protocol["interface"])
    gradient = np.asarray(prior["whole_geometry"]["gradient_crystal"]) * scale
    grid, shift, _ = solve(gradient, BASIS, interface, prior["grid"])
    probe = settings["probe_fwhm_nm"] / np.sqrt(8 * np.log(2))
    warped = nominal + warp
    phase = ORIGIN + scale*warped
    motion = (sample_disregistry(grid, shift, phase, grid.cell, probe) - phase) / scale
    traces = {
        "model_full": trace_field(warped + motion, data),
        "affine_background": trace_field(nominal, data),
        "warped_background": trace_field(warped, data),
        "periodic_relaxation_on_affine_background": trace_field(nominal + motion, data),
        "observed_phase_unsmoothed": trace_field(data.u_nm, data),
        "observed_phase_smoothed": trace_field(data.u_nm, data, settings["smoothing_sigma_pixels"]),
        "model_full_smoothed": trace_field(warped + motion, data, settings["smoothing_sigma_pixels"]),
    }
    raw_pitch_ratio = raw["source"]["dm4_scan_spacing_nm"] / data.pixel_nm
    traces["model_full_actual_pitch"] = traces["model_full"] / raw_pitch_ratio
    start_row, start_col = data.metadata["author_unwrap_crop_origin_row_col"]
    for rows in (original, replay):
        for row in rows:
            rr, cc = row["row"]-start_row, row["column_start"]-start_col
            region = np.s_[rr, cc:cc+row["column_count"]]
            row["traces"] = {k: float(v[region].mean()) for k, v in traces.items()}
            expected = row["matched_model_dilations_percent"]["warped_relaxed"]
            if not np.isclose(row["traces"]["model_full"], expected, rtol=0, atol=1e-10):
                raise ValueError("The fixed physical prediction changed")
            if not all(np.isfinite(v) for v in row["traces"].values()):
                raise ValueError("One trace is missing within the matched raw blocks")
    dev = assess(original)
    test = assess(replay, dev["local_projection_diagnostic"])
    result = {
        "scope": "DS31 only. Score the direct reciprocal lattice readouts, the gradient of the archived interferometric phase field, and the model on identical blocks. The measurements share an acquisition and the processed phase field also supplies the model geometry. Source-row amplitude projections are diagnostic only and are carried unchanged to the disjoint raw replay rows.",
        "scale": "All traces are 50 times the trace of a relative in-plane distortion. The source phase gradient uses its calibrated raster orientation. The raw spot positions supply the relative projected deformation and need no in-plane scan-rotation correction.",
        "input_sha256": {p.name: digest(p) for p in (Path(__file__), RAW, HIGH, ORIGINAL_ARCS, SPATIAL, TRANSFER, SETTINGS, FREEZE)},
        "fit_rows": dev, "replay_rows": test, "samples": {"original": original, "replay": replay},
        "wall_seconds": perf_counter() - start,
    }
    checkpoint_guard()
    atomic_json(args.output, result)
    print(test["phase_and_model"], flush=True)


if __name__ == "__main__":
    main()
