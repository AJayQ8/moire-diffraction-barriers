"""Integrate the public Deflate64 diffraction archive using bounded streaming reads."""

import argparse
from contextlib import contextmanager
from hashlib import md5, sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import h5py
import numpy as np

from extract_graphene_wall_signal import author_geometry
from read_graphene_diffraction_metadata import DATASET
from shared_compute import atomic_json, checkpoint_guard


PROTOCOL = Path(__file__).with_name("graphene_wall_signal_repeat_protocol_v01.json")
METADATA_BYTES = 1024**2


@contextmanager
def member_stream(archive, member):
    """Use the system's Deflate64 decoder and verify its exit and CRC."""
    with subprocess.Popen(["/usr/bin/unzip", "-p", str(archive), member],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as decoder:
        try:
            yield decoder.stdout
            error = decoder.stderr.read().decode(errors="replace")
            if decoder.wait():
                raise OSError(f"Could not decode {member}. {error.strip()}")
        finally:
            if decoder.poll() is None:
                decoder.terminate()
                decoder.wait(timeout=5)


def archive_checksum(archive, scan):
    source = md5()
    with archive.open("rb") as stream:
        while data := stream.read(8 * METADATA_BYTES):
            source.update(data)
    if archive.stat().st_size != scan["archive_bytes"] or source.hexdigest() != scan["archive_md5"]:
        raise ValueError("The archive does not have the published size and MD5")
    with ZipFile(archive) as source_zip:
        metadata = source_zip.getinfo(scan["member"])
    return metadata.file_size, metadata.CRC


def read_hdf5_layout(archive, member, directory, expected_length):
    """Recover the HDF5 array layout from the verified member's metadata."""
    with member_stream(archive, member) as stream:
        head = stream.read(METADATA_BYTES)
        tail = head
        length = len(head)
        while block := stream.read(8 * METADATA_BYTES):
            length += len(block)
            tail = block[-METADATA_BYTES:] if len(block) >= METADATA_BYTES else (tail + block)[-METADATA_BYTES:]
            if length % (1024 * METADATA_BYTES) == METADATA_BYTES:
                checkpoint_guard()
    if length != expected_length:
        raise ValueError("The decoded data member has a different length")
    with TemporaryDirectory(dir=directory) as scratch:
        path = Path(scratch) / "layout.h5"
        with path.open("wb") as image:
            image.write(head)
            image.seek(length - len(tail))
            image.write(tail)
        with h5py.File(path) as handle:
            array = handle[DATASET]
            shape, dtype, offset = array.shape, array.dtype, array.id.get_offset()
    if offset is None or len(shape) != 4 or dtype.kind not in "ui":
        raise ValueError("A contiguous integer diffraction cube is required")
    return shape, dtype, offset


def integrate(archive, scan, apertures, directory):
    length, crc = archive_checksum(archive, scan)
    shape, dtype, offset = read_hdf5_layout(archive, scan["member"], directory, length)
    # author_geometry uses only the detector's shape with saved apertures.
    detector = np.empty((0, *shape[2:]), dtype=dtype)
    masks, geometry = author_geometry(detector, scan, apertures)
    row_stride = scan["row_stride"]
    pixels = shape[2] * shape[3]
    row_bytes = shape[1] * pixels * dtype.itemsize
    counts = []
    with member_stream(archive, scan["member"]) as stream:
        if len(stream.read(offset)) != offset:
            raise OSError("Incomplete HDF5 header")
        for row in range(shape[0]):
            payload = stream.read(row_bytes)
            if len(payload) != row_bytes:
                raise OSError(f"Incomplete diffraction raster row {row}")
            if row % row_stride == 0:
                frames = np.frombuffer(payload, dtype).reshape(shape[1], pixels)
                counts.append(np.array([[frames[:, take].sum(axis=-1) for take in radii] for radii in masks]))
            if row % 16 == 0:
                checkpoint_guard()
        while stream.read(METADATA_BYTES):
            pass
    values = np.stack(counts, axis=1).transpose(0, 1, 3, 2)
    temporary = directory / "intensity.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, intensity=values)
    temporary.replace(directory / "intensity.npz")
    url = f'https://zenodo.org/records/{scan["record"]}/files/{scan["file"]}?download=1'
    geometry.update({"input_url": url, "archive_md5": scan["archive_md5"], "hdf5_member": scan["member"],
                     "hdf5_member_bytes": length, "hdf5_member_crc32": f"{crc:08x}", "HDF5_shape": shape,
                     "dtype": dtype.str, "array_offset": offset, "row_stride": row_stride,
                     "protocol_sha256": sha256(PROTOCOL.read_bytes()).hexdigest(),
                     "mask_sha256": sha256(json.dumps([[x.tolist() for x in apertures]
                                                      for apertures in masks]).encode()).hexdigest(),
                     "integrated_sha256": sha256((directory / "intensity.npz").read_bytes()).hexdigest()})
    atomic_json(directory / "receipt.json", geometry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("author_apertures", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    scan = json.loads(PROTOCOL.read_text())["scans"]["s1-6"]
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_guard()
    integrate(args.archive, scan, args.author_apertures, args.output)


if __name__ == "__main__":
    main()
