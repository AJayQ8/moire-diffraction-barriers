"""Integrate fixed Bragg apertures without downloading entire contiguous HDF5 cubes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from time import sleep

import numpy as np
import requests
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, map_coordinates, uniform_filter

from shared_compute import atomic_json, checkpoint_guard


ANGLES = np.deg2rad(np.arange(6) * 60)
PROTOCOL = Path(__file__).with_name("graphene_wall_signal_protocol_v01.json")
RECOVERY = Path(__file__).with_name("graphene_wall_signal_aperture_recovery_v01.json")
REFINEMENT = Path(__file__).with_name("graphene_wall_signal_aperture_refinement_v01.json")


def ring_positions(center, radius, angle):
    """Return the two sixfold rings in the detector's row-column frame."""
    orientations = np.concatenate((angle[..., None] + ANGLES,
                                   angle[..., None] + ANGLES + np.pi / 6), axis=-1)
    radii = np.repeat((1., np.sqrt(3)), 6)
    return center[..., None, :] + (radius[..., None] * radii)[..., None] * np.stack(
        (np.sin(orientations), np.cos(orientations)), axis=-1
    )


def find_interfering_lattice(frames, scan, center, radius, angle):
    """Find weak but jointly blinking opposite spots in the calibration frames."""
    y, x, r, t = np.meshgrid(
        center[0] + np.arange(-4., 4.1), center[1] + np.arange(-4., 4.1),
        radius + np.arange(-4., 4.1, .5), np.deg2rad(np.arange(0., 60., .5)), indexing="ij"
    )
    origins = np.stack((y, x), axis=-1)
    positions = ring_positions(origins, r, t)[..., :6, :]
    aperture = round(2 * scan["aperture_radii_px"][0] / np.sqrt(2)) // 2 * 2 + 1
    smooth = uniform_filter(frames.astype(float), size=(1, aperture, aperture))
    average = smooth.mean(axis=0)
    residual = (smooth - average) / np.sqrt(np.maximum(average, 1.))
    coords = np.moveaxis(positions, -1, 0)
    flicker = np.stack([map_coordinates(image, coords, order=1) for image in residual], axis=0)
    covariance = np.mean(flicker[..., :3] * flicker[..., 3:], axis=0)
    strength = np.sort(covariance, axis=-1)
    merit = strength[..., -2] + .25 * strength[..., 0]
    orientation_distance = abs((t - angle + np.pi/6) % (np.pi/3) - np.pi/6)
    brightness = map_coordinates(average, coords, order=1)
    bright_apertures = ring_positions(center, radius, angle)[:6]
    limit = .45 * np.median(map_coordinates(average, bright_apertures.T, order=1))
    eligible = (orientation_distance >= np.deg2rad(15)) & (np.median(brightness, axis=-1) < limit)
    if not np.any(eligible):
        raise ValueError("No calibration aperture outside the brightest lattice")
    best = np.unravel_index(np.argmax(np.where(eligible, merit, -np.inf)), merit.shape)
    return origins[best], r[best], t[best], covariance[best].tolist()


def refine_opposites(positions, frames, aperture):
    """Correct smooth detector distortion using opposite spot pairs."""
    box = 2 * int(np.round(aperture * .75)) + 1
    smoothed = uniform_filter(frames.astype(float), size=(1, box, box))
    average = smoothed.mean(axis=0)
    residual = (smoothed - average) / np.sqrt(np.maximum(average, 1.))
    base = positions[:6].mean(axis=0)
    yy, xx = np.indices(average.shape)
    output = positions.copy()
    covariance_strength = []
    for i in range(3):
        d1 = np.hypot(yy - positions[i, 0], xx - positions[i, 1])
        d2 = np.hypot(yy - positions[i + 3, 0], xx - positions[i + 3, 1])
        y, x = np.where(d1 <= 9.)
        v, u = np.where(d2 <= 9.)
        covariance = (residual[:, y, x].T @ residual[:, v, u]) / len(frames)
        opposite_error = np.hypot(y[:, None] + v - 2 * base[0], x[:, None] + u - 2 * base[1])
        valid = opposite_error < 9.
        a, b = np.unravel_index(np.argmax(np.where(valid, covariance, -np.inf)), covariance.shape)
        output[i] = (y[a], x[a])
        output[i + 3] = (v[b], u[b])
        covariance_strength.append(float(covariance[a, b]))
    # Only first-ring calibration data set the reciprocal-space corrections.
    # Apply the same affine detector correction to all six second-ring spots.
    design = np.column_stack((positions[:6], np.ones(6)))
    correction = np.linalg.lstsq(design, output[:6], rcond=None)[0]
    prediction = np.column_stack((positions[6:], np.ones(6))) @ correction
    output[6:] = prediction
    return output, covariance_strength


def aperture_geometry(frames, scan, recover=False, refine=False):
    average = frames.mean(axis=0)
    if "nominal_center_rc" in scan:
        center = np.array(scan["nominal_center_rc"])
        radius = np.asarray(scan["first_ring_radius_px"])
        angle = np.deg2rad(scan["first_ring_orientation_degrees"])
    else:
        # Search a measured instrument range around the direct beamstop. The
        # merit uses only the average detector frame, never a real-space map.
        y, x, radius, angle = np.meshgrid(
            np.arange(196., 218.), np.arange(178., 205.),
            np.arange(63., 73., .5), np.deg2rad(np.arange(0., 60., .75)), indexing="ij"
        )
        center = np.stack((y, x), axis=-1)
        pos = ring_positions(center, radius, angle)
        contrast = gaussian_filter(average, 2.8) - gaussian_filter(average, 8.)
        sample = map_coordinates(contrast, np.moveaxis(pos, -1, 0), order=1)
        merit = sum(np.sort(values, axis=-1)[..., 1:5].mean(axis=-1)
                    for values in np.split(sample, 2, axis=-1))
        best = np.unravel_index(merit.argmax(), merit.shape)
        center, radius, angle = center[best], radius[best], angle[best]
    evidence = None
    if recover:
        center, radius, angle, evidence = find_interfering_lattice(frames, scan, center, radius, angle)
    positions = ring_positions(center, radius, angle)
    refined_evidence = None
    if refine:
        positions, refined_evidence = refine_opposites(positions, frames, scan["aperture_radii_px"][0])
    yy, xx = np.indices(average.shape)
    distance = (yy - positions[:, 0, None, None])**2 + (xx - positions[:, 1, None, None])**2
    masks = [[np.flatnonzero(piece) for piece in distance <= aperture**2]
             for aperture in scan["aperture_radii_px"]]
    # Reject apertures crossed by the opaque beamstop. Their opposite spot
    # may still contribute to the width observer.
    contrast = np.array([average.ravel()[k].mean() for k in masks[0]])
    unmasked = np.array([(average.ravel()[k] > 0).mean() for k in masks[0]])
    available = (contrast > .18 * np.repeat([np.median(v) for v in np.split(contrast, 2)], 6)) & (unmasked > .8)
    return masks, {"center_rc": center.tolist(), "first_ring_radius_px": float(radius),
                   "first_ring_orientation_degrees": float(np.rad2deg(angle)),
                   "positions_rc": positions.tolist(), "usable": available.tolist(),
                   "average_aperture_counts_per_pixel": contrast.tolist(), "unmasked_fraction": unmasked.tolist(),
                   "opposite_calibration_covariance": evidence,
                   "refined_first_ring_covariance": refined_evidence}


def author_geometry(frames, scan, path):
    data = loadmat(path, squeeze_me=True)
    saved = np.asarray(data["disk_centers"], dtype=float) - 1
    if saved.shape != (12, 2):
        raise ValueError("Expected the authors' two sets of six circles")
    # Each six-spot set runs clockwise. Set both orderings to the same
    # relative reciprocal-angle convention as ring_positions.
    center = np.mean((saved[0] + saved[3], saved[1] + saved[4]), axis=0) / 2
    first = saved[[3, 2, 1, 0, 5, 4]]
    outer = saved[[9, 8, 7, 6, 11, 10]]
    positions = np.vstack((first, outer))
    available = np.isfinite(positions).all(axis=-1)
    yy, xx = np.indices(frames.shape[1:])
    dist = (yy - positions[:, 0, None, None])**2 + (xx - positions[:, 1, None, None])**2
    masks = [[np.flatnonzero(piece) for piece in dist <= aperture**2]
             for aperture in scan["aperture_radii_px"]]
    return masks, {"center_rc": center.tolist(), "usable": available.tolist(),
                   "positions_rc": [point.tolist() if good else None for point, good in zip(positions, available)],
                   "author_aperture_sha256": sha256(path.read_bytes()).hexdigest(),
                   "author_aperture_source": scan["author_aperture_source"]}


def read_row(url, offset, count, length):
    last = offset + count - 1
    expected = f"bytes {offset}-{last}/{length}"
    for attempt in range(3):
        try:
            with requests.get(url, headers={"Range": f"bytes={offset}-{last}", "Accept-Encoding": "identity"},
                              timeout=(10, 120), stream=True) as response:
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    raise OSError(f"Incorrect partial response {response.status_code}: {response.headers.get('Content-Range')}")
                payload = response.content
                if len(payload) != count:
                    raise OSError("Partial diffraction row")
                return payload
        except (requests.RequestException, OSError):
            if attempt == 2:
                raise
            sleep(1 + attempt)


def extract(calibration, scan, directory, threads, recover, refine, protocol, author_apertures):
    if author_apertures is not None:
        masks, geometry = author_geometry(calibration["frames"], scan, author_apertures)
    else:
        masks, geometry = aperture_geometry(calibration["frames"], scan, recover, refine)
    shape = tuple(int(n) for n in calibration["shape"])
    offset, length = int(calibration["offset"]), int(calibration["length"])
    dtype = np.dtype(str(calibration["dtype"]))
    url = str(calibration["url"])
    (r0, r1), (c0, c1) = scan["region_rc"]
    if not (0 <= r0 < r1 <= shape[0] and 0 <= c0 < c1 <= shape[1]):
        raise ValueError("Protocol region does not fit the published raw image")
    rows = list(range(r0, r1, 2))
    pixels = int(np.prod(shape[2:]))
    frame_bytes = pixels * dtype.itemsize
    row_bytes = (c1 - c0) * frame_bytes
    mask_digest = sha256(json.dumps([indices.tolist() for radius in masks for indices in radius]).encode()).hexdigest()

    def integrate(row):
        part = directory / f"row_{row:03d}.npz"
        if part.exists():
            return
        start = offset + (row * shape[1] + c0) * frame_bytes
        payload = read_row(url, start, row_bytes, length)
        frames = np.frombuffer(payload, dtype).reshape(c1 - c0, pixels)
        signals = np.array([[frames[:, k].sum(axis=-1) for k in radius] for radius in masks])
        with part.with_suffix(".tmp").open("wb") as handle:
            np.savez_compressed(handle, signals=signals, start=start, digest=sha256(payload).hexdigest(),
                                mask_sha256=mask_digest)
        part.with_suffix(".tmp").replace(part)

    checkpoint_guard()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for row in pool.map(integrate, rows):
            checkpoint_guard()
    values, chunks = [], []
    for row in rows:
        with np.load(directory / f"row_{row:03d}.npz") as result:
            expected = offset + (row * shape[1] + c0) * frame_bytes
            if int(result["start"]) != expected or result["signals"].shape != (len(masks), 12, c1-c0):
                raise ValueError(f"The checkpoint for row {row} has a different raster extent")
            if "mask_sha256" in result and str(result["mask_sha256"]) != mask_digest:
                raise ValueError(f"The checkpoint for row {row} has different detector apertures")
            values.append(result["signals"])
            chunks.append({"row": row, "byte_start": int(result["start"]), "count": row_bytes,
                           "sha256": str(result["digest"])})
    geometry.update({"input_url": url, "HDF5_shape": shape, "row_stride": 2,
                     "region_rc": scan["region_rc"], "downloaded_bytes": len(rows) * row_bytes,
                     "byte_ranges": chunks, "protocol_sha256": sha256(protocol.read_bytes()).hexdigest(),
                     "mask_sha256": mask_digest})
    if recover:
        geometry["aperture_recovery_sha256"] = sha256(RECOVERY.read_bytes()).hexdigest()
    if refine:
        geometry["aperture_refinement_sha256"] = sha256(REFINEMENT.read_bytes()).hexdigest()
    atomic_json(directory / "receipt.json", geometry)
    with (directory / "intensity.tmp").open("wb") as handle:
        np.savez_compressed(handle, intensity=np.stack(values, axis=1).transpose(0, 1, 3, 2))
    (directory / "intensity.tmp").replace(directory / "intensity.npz")
    print(f"Integrated {len(rows)} raster rows from {url}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan")
    parser.add_argument("instrument", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--refine", action="store_true", help="Refine pairs within the weaker lattice")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--author-apertures", type=Path)
    args = parser.parse_args()
    if args.refine and not args.recover:
        parser.error("--refine requires --recover")
    scan = json.loads(args.protocol.read_text())["scans"][args.scan]
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.instrument) as calibration:
        extract(calibration, scan, args.output, args.threads, args.recover, args.refine,
                args.protocol, args.author_apertures)


if __name__ == "__main__":
    main()
