"""Build manuscript values from the paired, fixed-probe saddle scores."""

from hashlib import sha256
import json

from build_v007_tables import DISJOINT, EVEN, EVIDENCE, TABLES, macro, signed_range
from build_v009_tables import setting_range


def build(report):
    comparison = report["archived_width_comparison"]
    all_maps = comparison["all_maps"]
    groups = comparison["groups"]
    low = groups["low_four"][EVEN]["score_nm"]
    high = groups["high_six"][EVEN]["score_nm"]
    optima = report["optima"]["groups"]["all_maps"]["cameras"]

    def optimum_range(camera, frame):
        return setting_range(optima[camera]["frames"][frame]["sampled_best_setting_p05_p95_meV_per_cell"])

    maps = report["selected_map_points"]

    def best_error(name):
        return maps[name]["frames_by_camera"][EVEN]["archived"]["score_nm"]["15"]

    sparse_share = 100 * (best_error("DS1") + best_error("DS2")) / (4 * low["15"])
    return [
        macro("ReportPublishedError", f"{all_maps[EVEN]['score_nm']['13.2']:.3f}"),
        macro("ReportPublishedChangeRange", signed_range(all_maps[EVEN]["score_delta_p05_p95_nm"])),
        macro("ReportPublishedDisjointChangeRange", signed_range(all_maps[DISJOINT]["score_delta_p05_p95_nm"])),
        macro("ReportLowPublished", f"{low['13.2']:.3f}"),
        macro("ReportLowBest", f"{low['15']:.3f}"),
        macro("ReportHighPublished", f"{high['13.2']:.3f}"),
        macro("ReportHighBest", f"{high['15']:.3f}"),
        macro("ReportArchivedOptimumRange", optimum_range(EVEN, "archived")),
        macro("ReportRotatedOptimumRange", optimum_range(EVEN, "rotate_ds13_ds14")),
        macro("ReportArchivedDisjointRange", optimum_range(DISJOINT, "archived")),
        macro("ReportRotatedDisjointRange", optimum_range(DISJOINT, "rotate_ds13_ds14")),
        macro("ReportSparseErrorShare", f"{sparse_share:.0f}"),
        macro("ReportDSThirteenError", f"{best_error('DS13'):.3f}"),
    ]


def main():
    path = EVIDENCE / "mos2_saddle_reporting_v01.json"
    payload = path.read_bytes()
    values = [f"% {path.name} sha256: {sha256(payload).hexdigest()}", *build(json.loads(payload))]
    (TABLES / "v010_reporting_values.tex").write_text("\n".join(values) + "\n")


if __name__ == "__main__":
    main()
