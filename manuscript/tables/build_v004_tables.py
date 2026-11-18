"""Build supplementary tables from the saved observation-control scores."""

import argparse
import hashlib
import json
from pathlib import Path


TABLES = Path(__file__).resolve().parent
INPUT = "mos2_observation_summary_v004.json"


def table(caption, name, columns, header, rows):
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        rf"\caption{{{caption}}}", rf"\label{{tab:{name}}}",
        rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}", r"\toprule",
        header + r" \\", r"\midrule",
    ]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def observation_table(data):
    by_map = data["scans"]
    rows = []
    for name, result in sorted(by_map.items(), key=lambda entry: entry[1]["twist_deg"]):
        before, individual, aligned = result["width_errors_unaligned_individual_aligned_nm"]
        shift = result["raster_continuous_and_native_width_shift_nm"]
        rows.append((name, result["cut_sections"], f"{before:.3f}", f"{individual:.3f}",
                     f"{aligned:.3f}", result["raster_sections"], f"{shift[0]:+.3f}", f"{shift[1]:+.3f}"))
    return table(
        r"\textbf{Cut averaging and sampling on the experimental scan grid.} "
        r"The three error columns compare the reference prediction to the same measured sections "
        r"using the original average, individual cuts or the average after aligning the cuts. "
        r"The last columns give the change in predicted width after evaluating the intensities on the "
        r"experimental scan grid and interpolating to the cuts. All widths and errors are in nm.",
        "cuts-and-raster", "lrrrrrrr",
        r"Map & $n_{\rm cut}$ & Original error & Single-cut error & Aligned error "
        r"& $n_{\rm grid}$ & $\Delta w_1$ & $\Delta w_2$", rows,
    )


def coverage_table(data):
    rows = []
    for ceiling, result in sorted(data["width_comparisons"].items(), key=lambda entry: float(entry[0])):
        fits = [result["detectors"][arm]["minima"][model]
                for arm in ("transferred_even", "separate_disjoint") for model in ("raman_shape", "raman_scale")]
        cells = [f"{fit['probe_nm']:.2f} / {fit['error']:.3f}" for fit in fits]
        rows.append((f"{float(ceiling):.2f}", result["width_walls"], *cells))
    return table(
        r"\textbf{Probe range, width-fit coverage, and sampled minima.} "
        r"Each row uses a single set of sections across the range starting at \(1.25\) nm, both models, "
        r"and both detector calibrations. All ten maps contribute to each row. Each cell lists the "
        r"probe FWHM at the sampled minimum and the error, both in nm. T uses coefficients fitted on four other maps. "
        r"D fits coefficients with the wall-cut pixels and buffer excluded. The two barriers are in meV per cell.",
        "extended-probe-range", "rrrrrr",
        r"Largest probe (nm) & Sections & T, 13.2 & T, 18.87 & D, 13.2 & D, 18.87", rows,
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--evidence", type=Path, default=TABLES.parents[1] / "evidence")
    cli.add_argument("--output", type=Path, default=TABLES)
    args = cli.parse_args()
    payload = (args.evidence / INPUT).read_bytes()
    data = json.loads(payload)
    body = observation_table(data["observation"]) + "\n" + coverage_table(data["extensions"])
    header = f"% Generated from {INPUT}\n% SHA-256 {hashlib.sha256(payload).hexdigest()}\n"
    (args.output / "v004_observation_controls.tex").write_text(header + body)


if __name__ == "__main__":
    main()
