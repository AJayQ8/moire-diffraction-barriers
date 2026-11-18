"""Check DS31 local strain against higher diffraction orders and intensity controls."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from author_detector_replay import read_detector_fit
from author_release import load_author_release
from run_heterobilayer_raw_ds31 import COORDINATES, FIRST_COLUMN, ROWS_PER_CHUNK, SCAN, WINDOW, fit_patch, model_trace_comparison
from run_heterobilayer_transfer import digest
from shared_compute import atomic_json, checkpoint_guard


def higher_orders(modes, positions):
    """Use the six third-order and six 3x first-order spots, defined by lattice symmetry."""
    g = np.array([(h, k) for h in range(-3, 4) for k in range(-3, 4) if h*h + h*k + k*k in (4, 9)])
    transform = np.linalg.lstsq(modes, positions - positions.mean(axis=0), rcond=None)[0]
    nominal = positions.mean(axis=0) + g @ transform
    if not np.all((nominal > WINDOW) & (nominal < 864 - WINDOW)):
        raise ValueError("A higher-order patch falls outside the detector")
    return g, nominal


def block_counts(ranges, positions, source):
    center = np.rint(positions).astype(int)
    blocks = []
    for row in source["sample_rows"]:
        checkpoint_guard()
        path = ranges / f"row-{row}.bin"
        frames = np.memmap(path, mode="r", dtype=np.uint8, shape=(ROWS_PER_CHUNK, 864, 864))
        cropped = np.stack([frames[:, y-WINDOW:y+WINDOW+1, x-WINDOW:x+WINDOW+1] for y, x in center], axis=1)
        for i in range(0, ROWS_PER_CHUNK, 10):
            block = cropped[i:i+10]
            blocks.append({"row": row, "column_start": FIRST_COLUMN + i, "column_count": 10,
                           "mean": block.mean(axis=0), "even": block[::2].mean(axis=0), "odd": block[1::2].mean(axis=0)})
    return center, blocks


def normalized_intensity_contrast(mean, nominal, center, radius):
    coordinates = COORDINATES[None] + center[:, None, None] - nominal[:, None, None]
    distance = np.linalg.norm(coordinates, axis=-1)
    signal = (mean * (distance <= radius)).sum(axis=(1, 2))
    background = (mean * (distance >= radius + 8)).sum(axis=(1, 2))
    return (signal / np.maximum(background, 1e-10)).tolist()


def read_set(blocks, indices, nominal, center, radius, repeat):
    checkpoint_guard()
    grand = np.mean([row["mean"] for row in blocks], axis=0)
    # The original ±9-pixel box clipped the ninth squared-order fit.
    # Scale the bound with the reciprocal-vector radius for these orders.
    matrix_bound = 0.065 * np.median(np.linalg.norm(nominal - nominal.mean(axis=0), axis=1))
    seed, summary = fit_patch(grand, indices, nominal, center, radius, matrix_bound=matrix_bound)
    summary["signal_to_outer_patch_count"] = normalized_intensity_contrast(grand, nominal, center, radius)
    phases = ("mean", "even", "odd") if repeat else ("mean",)
    result = []
    for block in blocks:
        checkpoint_guard()
        row = {}
        for phase in phases:
            _, fit = fit_patch(block[phase], indices, nominal, center, radius, seed, matrix_bound=matrix_bound)
            row[phase] = fit
        result.append(row)
    return summary, result


def measurement_summary(rows):
    measurements = {name: np.array([row[name]["mean"]["relative_trace_percent"] for row in rows])
                    for name in rows[0] if isinstance(rows[0][name], dict) and "mean" in rows[0][name]}
    names = list(measurements)
    spread = {name: float(x.std()) for name, x in measurements.items()}
    normalized = {name: (x - x.mean()) for name, x in measurements.items()}
    cross = {}
    for i, first in enumerate(names):
        for second in names[i+1:]:
            x, y = normalized[first], normalized[second]
            cross[f"{first}_vs_{second}"] = {"correlation": float(np.corrcoef(x, y)[0, 1]),
                                             "offset_free_disagreement_pp": float(np.sqrt(np.mean((x-y)**2)))}
    repeats = {}
    for name in names:
        if "even" in rows[0][name]:
            diff = [row[name]["even"]["relative_trace_percent"] - row[name]["odd"]["relative_trace_percent"] for row in rows]
            repeats[name] = float(np.sqrt(np.mean(np.square(diff))))
    return {"spread_pp": spread, "cross_readout": cross, "even_odd_disagreement_pp": repeats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("dependencies", type=Path)
    parser.add_argument("ranges", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    old = json.loads(args.baseline.read_text())
    for path, fingerprint in old["range_sha256"].items():
        checkpoint_guard()
        if digest(args.ranges / path) != fingerprint:
            raise ValueError(f"Raw sample {path} changed")
    author = load_author_release(args.release, args.dependencies)
    disk, _, _, fit_hash = read_detector_fit(args.archive, SCAN, "fit", author)
    active = disk._in_use.astype(bool)
    positions = np.column_stack((disk._xvec[active], disk._yvec[active]))
    modes = disk.clean_normgset(sanity_plot=False)
    high, nominal = higher_orders(modes, positions)
    source = old["source"]
    center, blocks = block_counts(args.ranges, nominal, source)
    r = 1.71 / source["detector_spacing_mrad"]
    high_ring_number = high[:, 0]**2 + high[:, 0] * high[:, 1] + high[:, 1]**2
    results = []
    for block, previous in zip(blocks, old["blocks"]):
        for key in ("row", "column_start", "column_count"):
            if block[key] != previous[key]:
                raise ValueError("Raw controls did not match the earlier block support")
        results.append({k: previous[k] for k in ("row", "column_start", "column_count")})
        results[-1].update({"original_first": {"mean": previous["first_all"]},
                            "original_second": {"mean": previous["second_all"]},
                            "original_both": {phase: previous[f"both_{code}"] for phase, code in (
                                ("mean", "all"), ("even", "even"), ("odd", "odd"))}})
    aggregates = {}
    for label, select, repeat in (("higher_both", np.ones(len(high), dtype=bool), True),
                                  ("higher_order_4", high_ring_number == 4, False),
                                  ("higher_order_9", high_ring_number == 9, False)):
        series = [{phase: block[phase][select] for phase in ("mean", "even", "odd")} for block in blocks]
        aggregates[label], score = read_set(series, high[select], nominal[select], center[select], r, repeat)
        for row, value in zip(results, score):
            row[label] = value
    direct_models = {}
    for name in ("higher_both", "higher_order_4", "higher_order_9"):
        comparison = [{**row, "both_all": row[name]["mean"]} for row in results]
        direct_models[name] = model_trace_comparison(args.archive, comparison)
    checkpoint_guard()
    atomic_json(args.output, {
        "scope": "Same raw DS31 sampled blocks and separate higher diffraction orders. Smaller reciprocal separations and other diffraction contrast differ between orders. All readouts still share the sample and acquisition.",
        "input_sha256": {Path(__file__).name: digest(Path(__file__)), args.baseline.name: digest(args.baseline),
                         "DS31_fit.pkl": fit_hash},
        "higher_order_indices": high.tolist(), "higher_order_nominal_centers_row_col": nominal.tolist(),
        "aggregate": aggregates, "summary": measurement_summary(results),
        "direct_model_comparisons": direct_models, "blocks": results, "wall_seconds": perf_counter() - start,
    })


if __name__ == "__main__":
    main()
