"""Extend a direct-HDF5 image without redownloading the original rectangle."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from extract_graphene_wall_signal import author_geometry, read_row
from shared_compute import atomic_json, checkpoint_guard


def extend(instrument, scan, prior, apertures, directory, protocol, threads):
    masks, geometry = author_geometry(instrument["frames"], scan, apertures)
    shape = tuple(int(n) for n in instrument["shape"])
    dtype = np.dtype(str(instrument["dtype"]))
    offset, length = int(instrument["offset"]), int(instrument["length"])
    url = str(instrument["url"])
    (r0, r1), (c0, c1) = scan["region_rc"]
    (pr0, pr1), (pc0, pc1) = scan["prior_region_rc"]
    if not (0 <= r0 <= pr0 < pr1 <= r1 <= shape[0] and 0 <= c0 <= pc0 < pc1 <= c1 <= shape[1]
            and (pr0-r0) % 2 == 0):
        raise ValueError("The enlarged rectangle must contain the original raster samples")
    old = json.loads((prior/"receipt.json").read_text())
    mask_hash = sha256(json.dumps([x.tolist() for radii in masks for x in radii]).encode()).hexdigest()
    if (old["region_rc"] != scan["prior_region_rc"] or old["input_url"] != url
            or old["mask_sha256"] != mask_hash or old["row_stride"] != 2):
        raise ValueError("The original detector weights or acquisition changed")
    with np.load(prior/"intensity.npz") as original:
        old_data = original["intensity"]
    expected_old_shape = (len(masks), len(range(pr0, pr1, 2)), pc1-pc0, 12)
    if old_data.shape != expected_old_shape:
        raise ValueError("The original intensity file has an unexpected shape")
    pixels = shape[2]*shape[3]
    frame_bytes = pixels*dtype.itemsize
    rows = range(r0, r1, 2)

    def region(row, start, end):
        byte = offset + (row*shape[1]+start)*frame_bytes
        data = read_row(url, byte, (end-start)*frame_bytes, length)
        frames = np.frombuffer(data, dtype).reshape(end-start, pixels)
        counts = np.array([[frames[:, take].sum(axis=-1) for take in radii] for radii in masks])
        return counts.transpose(0, 2, 1), [byte, len(data), sha256(data).hexdigest()]

    def row_parts(row):
        part = directory/f"row_{row:03d}.npz"
        if part.exists():
            return part
        if pr0 <= row < pr1:
            left, le = region(row, c0, pc0)
            right, re = region(row, pc1, c1)
            counts = np.concatenate((left, old_data[:, (row-pr0)//2], right), axis=1)
            spans = [le, re]
        else:
            counts, line = region(row, c0, c1)
            spans = [line]
        with part.with_suffix(".tmp").open("wb") as stream:
            np.savez_compressed(stream, values=counts, spans=json.dumps(spans))
        part.with_suffix(".tmp").replace(part)
        return part

    checkpoint_guard()
    with ThreadPoolExecutor(max_workers=threads) as executor:
        outputs = []
        for path in executor.map(row_parts, rows):
            outputs.append(path)
            checkpoint_guard()
    blocks, receipts = [], []
    for row, stored in zip(rows, outputs, strict=True):
        with np.load(stored) as data:
            blocks.append(data["values"])
            receipts.append({"row":row, "ranges":json.loads(str(data["spans"]))})
    intensity = np.stack(blocks, axis=1)
    cut = intensity[:, (pr0-r0)//2:(pr1-r0+1)//2, pc0-c0:pc1-c0]
    if not np.array_equal(cut, old_data):
        raise ValueError("The extended output did not preserve the original intensities")
    checkpoint_guard()
    with (directory/"intensity.tmp").open("wb") as stream:
        np.savez_compressed(stream, intensity=intensity)
    (directory/"intensity.tmp").replace(directory/"intensity.npz")
    geometry.update({"input_url":url, "HDF5_shape":shape, "row_stride":2, "region_rc":scan["region_rc"],
                     "byte_ranges":receipts, "new_bytes":sum(x[1] for v in receipts for x in v["ranges"]),
                     "prior_intensity_sha256":sha256((prior/"intensity.npz").read_bytes()).hexdigest(),
                     "mask_sha256":mask_hash, "protocol_sha256":sha256(protocol.read_bytes()).hexdigest()})
    atomic_json(directory/"receipt.json", geometry)
    print(f"Extended to {len(rows)} raster rows with the original overlap verified.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("instrument", type=Path)
    parser.add_argument("prior", type=Path)
    parser.add_argument("apertures", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    scan = json.loads(args.protocol.read_text())["scans"]["s1-10"]
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.instrument) as instrument:
        extend(instrument, scan, args.prior, args.apertures, args.output, args.protocol, args.threads)


if __name__ == "__main__":
    main()
