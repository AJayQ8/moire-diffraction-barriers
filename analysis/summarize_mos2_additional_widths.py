"""Reproduce the four-map comparison from per-wall fits and the six-map spacing fit."""
import argparse
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
from statistics import fmean, linear_regression, median


EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"
PHYSICAL = ("network", "isolated")
METHODS = (*PHYSICAL, "spacing_only")


def read_maps(directory):
    extra = json.loads((directory / "mos2_additional_parallel_map_widths_v01.json").read_text())
    original = json.loads((directory / "mos2_spacing_and_weighting_v01.json").read_text())
    training = original["primary_close_group_spacing_and_ols"]["cohort"]["map_rows"]
    fit = linear_regression([m["median_wall_spacing_nm"] for m in training],
                            [m["mean_first_minus_second_width_nm"]["measured"] for m in training])
    assert {m["map"] for m in training} == set(extra["spacing_baseline"]["training_maps"])
    assert not {m["map"] for m in training}.intersection(extra["maps"])
    rows = []
    for name, case in extra["maps"].items():
        assert set(case["detector_training_maps"]) == set(extra["maps"]) - {name}
        walls = [wall for wall in case["walls"]
                 if all(all(wall["fits"][method]["qualified"]) for method in ("measured", *PHYSICAL))]
        d = median(wall["spacing_nm"] for wall in walls)
        gap = {}
        for method in ("measured", *PHYSICAL):
            widths = [wall["fits"][method]["widths_nm"] for wall in walls]
            gap[method] = fmean(first - second for first, second in widths)
        gap["spacing_only"] = fit.intercept + fit.slope * d
        assert len(walls) == case["primary"]["walls"]
        assert isclose(d, case["primary"]["median_spacing_nm"], abs_tol=1e-11)
        for method, value in gap.items():
            assert isclose(value, case["primary"]["first_minus_second_map_mean_nm"][method], abs_tol=1e-11)
        errors = {method: abs(gap[method] - gap["measured"]) for method in METHODS}
        rows.append((name, len(walls), d, gap, errors))
    rows.sort(key=lambda row: row[2])
    new_mae = {m: fmean(row[4][m] for row in rows) for m in METHODS}
    for m in METHODS:
        assert isclose(new_mae[m], extra["primary"]["equal_map_mae_nm"][m], abs_tol=1e-11)
    old_errors = [m["map_mean_gap_absolute_error_nm"] for m in training]
    all_errors = old_errors + [row[4] for row in rows]
    joint = {m: fmean(error[m] for error in all_errors) for m in PHYSICAL}
    return rows, new_mae, joint, sum(m["eligible_sections"] for m in training), all_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--tex", type=Path, help="Write the per-map data rows for the supplement")
    args = parser.parse_args()
    rows, new, joint, old_count, errors = read_maps(args.evidence)
    new_count = sum(row[1] for row in rows)
    wins = sum(row["network"] < row["isolated"] for row in errors)
    print(f"Additional sample: {len(rows)} maps and {new_count} wall segments. MAE (nm): {new}")
    print(f"Combined: {wins}/{len(errors)} network wins, {old_count + new_count} segments. MAE (nm): {joint}")
    if args.tex is not None:
        source = args.evidence / "mos2_additional_parallel_map_widths_v01.json"
        lines = [f"% {source.name} sha256: {sha256(source.read_bytes()).hexdigest()}",
                 r"\newcommand{\AdditionalMapRows}{%"]
        for name, count, d, gap, _ in rows:
            values = [name, str(count), f"{d:.2f}"] + [f"{gap[m]:.3f}" for m in ("measured", "network", "isolated", "spacing_only")]
            lines.append(" & ".join(values) + r" \\")
        lines.append("}")
        args.tex.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
