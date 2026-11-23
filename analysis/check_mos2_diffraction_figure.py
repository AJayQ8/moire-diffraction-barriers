"""Replay the Fig. 1b spot-position comparison from saved source-image centroids."""

import argparse
import json
import pickletools
import zipfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


EVIDENCE = (
    Path(__file__).resolve().parents[1] / "evidence" / "mos2_fig1b_orientation_check_v01.json"
)
SCANS = tuple(f"DS{number}" for number in range(1, 15))
DISK_FIELDS = ("_xvec", "_yvec", "_in_use")
RECORDED_PRECISION_PX = 0.0005


def read_archive_centers(data_dir: Path, scan: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract named float64 arrays from the deposited DiskSet opcode stream."""
    with zipfile.ZipFile(data_dir / f"{scan}.zip") as archive:
        with archive.open(f"{scan}/diskset.pkl") as member:
            # The three arrays precede the much larger intensity data in the deposit.
            prefix = member.read(256 * 1024)

    # Inspect literals only. This never unpickles a class or executes a reduction.
    fields = {}
    field_name = None
    for _, value, _ in pickletools.genops(prefix):
        if isinstance(value, str) and value in DISK_FIELDS:
            field_name = value
        elif field_name is not None and isinstance(value, bytes) and len(value) > 32:
            fields[field_name] = np.frombuffer(value, dtype="<f8")
            field_name = None
            if len(fields) == len(DISK_FIELDS):
                break
    if len(fields) != len(DISK_FIELDS):
        raise ValueError(f"{scan} does not contain the three expected DiskSet fields")

    x, y, in_use = (fields[name] for name in DISK_FIELDS)
    if x.shape != y.shape or x.shape != in_use.shape or not np.all(np.isin(in_use, [0, 1])):
        raise ValueError(f"{scan} has mismatched coordinates or a nonbinary in-use mask")
    mask = in_use.astype(bool)
    if np.count_nonzero(mask) != 12 or np.count_nonzero(~mask) < 5:
        raise ValueError(f"{scan} must contain 12 in-use and at least five unused peaks")
    centers = np.column_stack((x, y))
    if not np.isfinite(centers).all():
        raise ValueError(f"{scan} contains a nonfinite disk center")
    return centers[mask], centers[~mask]


def rms(distances: np.ndarray) -> float:
    return float(np.sqrt(np.mean(distances**2)))


def match_diffraction_spots(
    mos2_source: np.ndarray,
    hbn_source: np.ndarray,
    mos2_archive: np.ndarray,
    unused_archive: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Pick the MoS2 registration with the smallest five-spot hBN separation."""
    source_center = mos2_source.mean(axis=0)
    source_first_ring = mos2_source[:6] - source_center
    archive_center = mos2_archive.mean(axis=0)
    centered_archive = mos2_archive - archive_center
    radii = np.linalg.norm(centered_archive, axis=1)
    first_ring = np.argsort(radii)[:6]
    scale = np.linalg.norm(source_first_ring, axis=1).mean() / radii[first_ring].mean()
    archive_design = np.column_stack((centered_archive, np.ones(len(mos2_archive))))
    unused_design = np.column_stack(
        (unused_archive - archive_center, np.ones(len(unused_archive)))
    )

    candidates = []
    for sign in (-1, 1):
        reflected = np.diag([sign, 1])
        first_x, first_y = centered_archive[first_ring[0]] @ reflected
        for source_x, source_y in source_first_ring:
            # Initialize the correspondence by placing one archived first-ring peak.
            angle = np.arctan2(source_y, source_x) - np.arctan2(first_y, first_x)
            cosine, sine = np.cos(angle), np.sin(angle)
            rotation = np.array([[cosine, sine], [-sine, cosine]])
            initial = centered_archive @ (reflected @ rotation * scale) + source_center
            _, source_indices = linear_sum_assignment(cdist(initial, mos2_source))
            target = mos2_source[source_indices]
            affine = np.linalg.lstsq(archive_design, target, rcond=None)[0]
            mos2_distances = np.linalg.norm(archive_design @ affine - target, axis=1)

            hbn_distances = cdist(hbn_source, unused_design @ affine)
            source_rows, archive_cols = linear_sum_assignment(hbn_distances)
            individual = hbn_distances[source_rows, archive_cols]
            candidates.append((rms(mos2_distances), rms(individual), individual))
    return min(candidates, key=lambda candidate: candidate[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="directory of DS1.zip to DS14.zip")
    parser.add_argument("--evidence", type=Path, default=EVIDENCE, help="saved centroids and residuals")
    parser.add_argument("--threshold", type=int, choices=(115, 135, 155), default=135)
    args = parser.parse_args()

    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    centroids = evidence[f"native_coordinates_at_level_{args.threshold}"]
    mos2_source = np.asarray(centroids["MoS2_xy"], dtype=float)
    hbn_source = np.asarray(centroids["hBN_xy"], dtype=float)
    if mos2_source.shape != (12, 2) or hbn_source.shape != (5, 2):
        raise ValueError("Expected twelve MoS2 and five hBN source-image centroids")

    recorded = evidence["source_image_pixel_residuals"][str(args.threshold)]
    recorded_individual = evidence["individual_hBN_residuals_at_level_135_px"]
    print(f"Green threshold {args.threshold}. RMS separations in source-image pixels.")
    print("Scan   MoS2      hBN")
    mismatches = []
    for scan in SCANS:
        mos2_archive, unused_archive = read_archive_centers(args.data_dir, scan)
        mos2_rms, hbn_rms, individual = match_diffraction_spots(
            mos2_source, hbn_source, mos2_archive, unused_archive
        )
        expected = [recorded[scan]["MoS2_rms_px"], recorded[scan]["hBN_rms_px"]]
        matches = np.allclose([mos2_rms, hbn_rms], expected, atol=RECORDED_PRECISION_PX, rtol=0)
        if args.threshold == 135 and scan in recorded_individual:
            matches &= np.allclose(
                individual, recorded_individual[scan], atol=RECORDED_PRECISION_PX, rtol=0
            )
        print(f"{scan:<5} {mos2_rms:6.3f}   {hbn_rms:7.3f}")
        if not matches:
            mismatches.append(scan)
    if mismatches:
        raise SystemExit(f"Recorded residuals differ for {', '.join(mismatches)}")
    print(f"Reproduced the recorded residuals for all {len(SCANS)} scans.")


if __name__ == "__main__":
    main()
