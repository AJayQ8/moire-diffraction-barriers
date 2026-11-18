"""Read the metadata and nine detector-calibration frames of a public scan."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import re

import h5py
import numpy as np
import requests

from extract_graphene_wall_signal import PROTOCOL, read_row


DATASET = "/4DSTEM_experiment/data/datacubes/datacube_0/data"


class HttpRanges(io.RawIOBase):
    """Read just the HDF5 metadata blocks required to locate the raw array."""

    block_size = 65536

    def __init__(self, url):
        self.url = url
        self.offset = 0
        self.blocks = {}
        header = requests.get(url, headers={"Range": f"bytes=0-{self.block_size-1}",
                                            "Accept-Encoding": "identity"}, timeout=(10, 60))
        match = re.fullmatch(r"bytes 0-(\d+)/(\d+)", header.headers.get("Content-Range", ""))
        if header.status_code != 206 or not match or len(header.content) != int(match[1]) + 1:
            raise OSError("The raw file did not serve a verified HDF5 header")
        self.length = int(match[2])
        self.blocks[0] = header.content

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.offset

    def seek(self, offset, whence=io.SEEK_SET):
        target = (0, self.offset, self.length)[whence] + offset
        if target < 0:
            raise OSError("Negative HDF5 seek")
        self.offset = target
        return target

    def read(self, count=-1):
        stop = self.length if count < 0 else min(self.offset + count, self.length)
        payload = []
        while self.offset < stop:
            start = self.offset // self.block_size * self.block_size
            if start not in self.blocks:
                nbytes = min(self.block_size, self.length - start)
                self.blocks[start] = read_row(self.url, start, nbytes, self.length)
            block = self.blocks[start]
            end = min(start + len(block), stop)
            payload.append(block[self.offset-start:end-start])
            self.offset = end
        return b"".join(payload)

    def readinto(self, target):
        data = self.read(len(target))
        target[:len(data)] = data
        return len(data)


def read_instrument(scan):
    url = f'https://zenodo.org/records/{scan["record"]}/files/{scan["file"]}?download=1'
    with HttpRanges(url) as data, h5py.File(data, "r") as meta:
        array = meta[DATASET]
        shape, dtype, offset, length = array.shape, array.dtype, array.id.get_offset(), data.length
    if offset is None or len(shape) != 4:
        raise ValueError("A contiguous four-dimensional HDF5 array is required")
    frame_size = int(np.prod(shape[2:])) * dtype.itemsize
    points = [((shape[0]-1)*r//10, (shape[1]-1)*c//10) for r in (3, 5, 7) for c in (3, 5, 7)]

    def read_frame(point):
        row, column = point
        start = offset + (row * shape[1] + column) * frame_size
        payload = read_row(url, start, frame_size, length)
        return np.frombuffer(payload, dtype).reshape(shape[2:])

    with ThreadPoolExecutor(max_workers=3) as pool:
        frames = np.stack(list(pool.map(read_frame, points)))
    return {"url": url, "shape": shape, "dtype": dtype.str, "length": length,
            "offset": offset, "points": points, "frames": frames}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan")
    parser.add_argument("output", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    scan = json.loads(args.protocol.read_text())["scans"][args.scan]
    np.savez_compressed(args.output, **read_instrument(scan))


if __name__ == "__main__":
    main()
