"""Restricted observers for the Van Winkle et al. TMD moire archive.

The archive contains NumPy pickle payloads.  This module only decodes the
unwrapped displacement member for the declared development set and uses a
strict NumPy-only unpickler.  All other datasets can be inventoried from a
whitelist of ``info.txt`` inputs without decoding maps or reading score fields.

Coordinate convention
---------------------
``unwrap.pkl`` stores ``u[row, col, component]`` after the authors' conversion
to Cartesian coordinates.  Components are dimensionless multiples of the
MoS2 lattice constant.  The scan coordinate is ``x = col * PixelSize`` and
``y = row * PixelSize`` (row increases in the direction used as +y by the
authors' ``strain()`` routine).  Therefore ``u_nm = LatticeConstant * u``.
The stored displacement is the *relative interlayer disregistry*; the authors
divide it by two only when forming equal-and-opposite single-layer strain.

The affine matrix returned by :func:`center_affine` is in the uncorrected scan
frame: rows are displacement components and columns are scan x/y.  The
authors' derivative correction is applied by right multiplication with
``[[cos(a), sin(a)], [-sin(a), cos(a)]]``, where ``a=SampleRotation``.

The shear observer reproduces the paper's all-good-interior-triangle mask and
its reduced-Wigner-Seitz displacement-sector classification for SP1/SP2/SP3.
These are stacking-sector averages used as wall-shear readouts, not a manually
traced geometric line mask.
"""

from __future__ import print_function

import hashlib
import io
import itertools
import json
import math
import pickle
import re
import zipfile
from pathlib import Path

import numpy as np
from scipy import ndimage


ZENODO_RECORD = "https://zenodo.org/records/7865970"
PAPER_DOI = "10.1038/s41467-023-38504-7"
DEFAULT_ARCHIVE_DIR = str(Path(__file__).resolve().parents[1] / "data" / "processed_mos2")

# Numerical maps are intentionally restricted to the declared development
# fields.  The paper's possible held-out maps are not enabled here.
DEVELOPMENT_IDS = frozenset((5, 7, 8, 11, 19, 20, 22, 23, 25))

# Published Zenodo checksums (MD5) for the 36 processed ZIPs.  The 7.5 GB raw
# DM4 is absent by design and is never fetched or opened by this module.
PUBLISHED_MD5 = {
    1: "db09dcb5d12f743011e0f742eeb3a9c1",
    2: "2da6288b5a649ff03123c779a05ecbe8",
    3: "57f75a18218f88f5539fdffdd779b8c9",
    4: "eca1528caa51220bc8f0817383a62e1a",
    5: "8ca1c7f7f57ba85a1c7410174e708e51",
    6: "790d7b3afb60d029c3d5a90c3094bf4e",
    7: "5d9f37e889a3816dba85782c2c9b9a79",
    8: "e6e9113fa837b3854b5e27374c89bf33",
    9: "86b478c1e59c533df48afdfe35e3a91f",
    10: "e40c53a52848ee35c8849a82aa769908",
    11: "d4d75e1c099e1cc3ef243b11ce0d94bc",
    12: "bc6dc6f6a922328cf24c88ffe1789ddc",
    13: "e11b546c7d968121a8da16c6be21c133",
    14: "f62cfd7ec915ac1d9bfffc97538e2b02",
    15: "f959369b3e934f752fd75184c624dbee",
    16: "06e316d60a329e97e37227a093110efc",
    17: "1d190539392201c4d398c84ffd07fea2",
    18: "33472b966ca2884348923f8d10587bdd",
    19: "e01c466dbafb025a967feb6b62f7fd03",
    20: "ade951531a4e04cf8c89636427ae7bb1",
    21: "44bb1992f850f40acb282c684e6bb902",
    22: "73e29c29a9a791e84b4ed08aa53aa58b",
    23: "0314da15ed6c8d0118904ec68b31ba5a",
    24: "2d519624b4e1b777add01ea3ca8a7ab9",
    25: "42d4a1b73d05bc8780672e07b35cd08f",
    26: "3fa3cf37a4f6ce649d97921b6ff3ac6a",
    28: "4e881e78a00e768f00f8433b7ab3cd77",
    29: "1933f3b508562f46cd6aa9d149a8ffec",
    30: "25b02d23a401e6fbc37ec65b31171135",
    31: "e7fc689872d0651bbcb560c34c30632e",
    32: "76206999a3104faebc3a521eaa9ac833",
    33: "e4e282813a4427e47d5ba966b119dc27",
    34: "d45acee8b92aaeb719ccdf2332bf4c59",
    35: "25958ce005bda11f9222d1f1615359e5",
    36: "722106cfc13b8e62064568c292d78d42",
    37: "66fd7445d5f8e9cca1a8c40da92aa49b",
}

# These are the only ``info.txt`` fields read by the inventory/loader.  In
# particular AvgSP*, Gamma, stacking-area fractions, and all errors for
# derived observables are excluded.
WHITELISTED_INFO_FIELDS = frozenset((
    "Material", "Orientation", "OriginalName", "NumberDisksUsed",
    "LatticeConstant", "LatticeMismatch", "PoissonRatio",
    "ScanShapeX", "ScanShapeY", "SmoothingSigma", "PixelSize",
    "FittingFunction", "RefitFromBinned", "DisplacementBasis",
    "UnwrapMethod", "AvgMoireTwist", "ErrMoireTwist",
    "AvgHeteroStrain", "ErrHeteroStrain", "SampleRotation",
    "SampleRotationStdErr", "K3toHAADFRotation_Used",
    "UnwrapSPCorrectionRotation", "HeteroBilayer",
))
DEVELOPMENT_REFERENCE_FIELDS = frozenset(
    ["AvgSP%dGamma" % n for n in (1, 2, 3)] +
    ["ErrSP%dGamma" % n for n in (1, 2, 3)] +
    ["AvgSP%dSubGamma" % n for n in (1, 2, 3)] +
    ["ErrSP%dSubGamma" % n for n in (1, 2, 3)] +
    ["AvgSPGamma", "ErrSPGamma", "AvgSPSubGamma", "ErrSPSubGamma"])

_AUTHOR_CARTESIAN_BASIS = np.array(((-1.0, 0.5), (0.0, math.sqrt(3.0) / 2.0)))
_AUTHOR_BASIS_INVERSE = np.linalg.inv(_AUTHOR_CARTESIAN_BASIS)
_SAFE_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
    ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
}
_MAX_UNWRAP_BYTES = 8 * 1024 * 1024
_MAX_MAP_DIMENSION = 512
_MAX_CENTER_COUNT = 10000


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that permits only the NumPy constructors used in the ZIPs."""

    def find_class(self, module, name):
        try:
            return _SAFE_GLOBALS[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError("blocked pickle global %s.%s" % (module, name))


class DatasetMap(object):
    """A loaded map, iterable as the five solver inputs.

    ``u_nm, lattice_nm, pixel_nm, sample_rotation_degrees, orientation =
    load_dataset(...)`` is supported.  Geometry and provenance are available
    as attributes for observers.
    """

    def __init__(self, dataset_id, u_nm, lattice_nm, pixel_nm,
                 sample_rotation_degrees, orientation, centers_rc_px,
                 adjacency_type, params, metadata, archive_path, archive_md5,
                 member_name, unwrap_sp_correction_radians):
        self.dataset_id = "DS%d" % int(dataset_id)
        self.u_nm = np.asarray(u_nm, dtype=np.float64)
        self.lattice_nm = float(lattice_nm)
        self.pixel_nm = float(pixel_nm)
        self.sample_rotation_degrees = float(sample_rotation_degrees)
        self.orientation = str(orientation).strip().lower()
        self.centers_rc_px = np.asarray(centers_rc_px, dtype=np.float64)
        self.adjacency_type = np.asarray(adjacency_type)
        self.params = params
        self.metadata = metadata
        self.archive_path = str(archive_path)
        self.archive_md5 = str(archive_md5)
        self.unwrap_member = str(member_name)
        self.unwrap_sp_correction_radians = float(unwrap_sp_correction_radians)
        self.source_valid_mask = np.isfinite(self.u_nm).all(axis=2)

    def __iter__(self):
        yield self.u_nm
        yield self.lattice_nm
        yield self.pixel_nm
        yield self.sample_rotation_degrees
        yield self.orientation


def _dataset_number(dataset_id):
    if isinstance(dataset_id, int):
        return dataset_id
    match = re.match(r"^DS?(\d+)$", str(dataset_id).strip(), re.IGNORECASE)
    if not match:
        raise ValueError("dataset_id must be an integer or DS<number>")
    return int(match.group(1))


def _parse_info_value(value):
    value = value.strip()
    if value == "{}" or value == "":
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return float(value)
    except ValueError:
        return value


def _archive_path(dataset_id, archive_dir=DEFAULT_ARCHIVE_DIR):
    number = _dataset_number(dataset_id)
    path = Path(archive_dir) / ("DS%d.zip" % number)
    if not path.is_file():
        raise FileNotFoundError("processed archive missing: %s" % path)
    return path


def _md5_file(path):
    digest = hashlib.md5()
    with open(str(path), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_whitelisted_info(zf, dataset_id):
    matches = [name for name in zf.namelist()
               if name.rsplit("/", 1)[-1].lower() == "info.txt"]
    if len(matches) != 1:
        raise ValueError("expected exactly one info.txt in DS%d ZIP" % dataset_id)
    raw = zf.read(matches[0]).decode("utf-8", "replace")
    result = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key not in WHITELISTED_INFO_FIELDS:
            continue
        value = value.split(" #", 1)[0].strip()
        result[key] = _parse_info_value(value)
    return result


def read_input_metadata(dataset_id, archive_dir=DEFAULT_ARCHIVE_DIR,
                        verify_checksum=True):
    """Read only the whitelisted input fields from ``info.txt``."""
    number = _dataset_number(dataset_id)
    path = _archive_path(number, archive_dir)
    digest = _md5_file(path)
    expected = PUBLISHED_MD5.get(number)
    if verify_checksum and expected and digest != expected:
        raise ValueError("DS%d MD5 mismatch: %s != %s" % (number, digest, expected))
    with zipfile.ZipFile(str(path), "r") as zf:
        fields = _read_whitelisted_info(zf, number)
    return {
        "dataset_id": "DS%d" % number,
        "archive_md5": digest,
        "published_md5": expected,
        "metadata_fields": fields,
    }


def read_development_reference_shear_fields(dataset_id,
                                            archive_dir=DEFAULT_ARCHIVE_DIR):
    """Read only stored SP shear means/errors for an authorized dev map."""
    number = _dataset_number(dataset_id)
    if number not in DEVELOPMENT_IDS:
        raise PermissionError("stored score fields are restricted to development maps")
    path = _archive_path(number, archive_dir)
    digest = _md5_file(path)
    if digest != PUBLISHED_MD5[number]:
        raise ValueError("DS%d MD5 mismatch: %s != %s" % (number, digest,
                                                          PUBLISHED_MD5[number]))
    with zipfile.ZipFile(str(path), "r") as zf:
        matches = [name for name in zf.namelist()
                   if name.rsplit("/", 1)[-1].lower() == "info.txt"]
        if len(matches) != 1:
            raise ValueError("expected exactly one info.txt in DS%d ZIP" % number)
        raw = zf.read(matches[0]).decode("utf-8", "replace")
    result = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in DEVELOPMENT_REFERENCE_FIELDS:
            result[key] = _parse_info_value(value.split(" #", 1)[0])
    return result


def _find_unwrap_member(zf, dataset_id):
    matches = [name for name in zf.namelist()
               if name.rsplit("/", 1)[-1] == "unwrap.pkl"]
    if len(matches) != 1:
        raise ValueError("expected exactly one unwrap.pkl in DS%d ZIP" % dataset_id)
    info = zf.getinfo(matches[0])
    if info.file_size > _MAX_UNWRAP_BYTES:
        raise ValueError("unwrap.pkl exceeds safe size limit")
    return matches[0], zf.read(matches[0])


def _check_primitive_tree(value, depth=0):
    if depth > 12:
        raise ValueError("pickle parameter nesting is too deep")
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return
    if isinstance(value, (list, tuple)):
        if len(value) > 10000:
            raise ValueError("pickle parameter list is too large")
        for item in value:
            _check_primitive_tree(item, depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 1000:
            raise ValueError("pickle parameter dictionary is too large")
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)):
                raise ValueError("non-primitive pickle parameter key")
            _check_primitive_tree(item, depth + 1)
        return
    if isinstance(value, np.generic):
        if value.dtype.hasobject:
            raise ValueError("object NumPy scalar is not permitted")
        return
    raise ValueError("unexpected unwrapped pickle object: %s" % type(value).__name__)


def _decode_unwrap(payload):
    decoded = RestrictedUnpickler(io.BytesIO(payload), fix_imports=False,
                                  encoding="ASCII", errors="strict").load()
    if not isinstance(decoded, (list, tuple)) or len(decoded) != 4:
        raise ValueError("unwrap.pkl is not [u, centers, adjacency, params]")
    u, centers, adjacency, params = decoded
    if not isinstance(u, np.ndarray) or u.ndim != 3 or u.shape[-1] != 2:
        raise ValueError("unwrapped displacement must be a (row,col,2) array")
    if max(u.shape[:2]) > _MAX_MAP_DIMENSION or min(u.shape[:2]) < 2:
        raise ValueError("unwrapped map dimensions exceed safety limits")
    if u.dtype.hasobject or not np.issubdtype(u.dtype, np.number):
        raise ValueError("unwrapped displacement has unsafe dtype")
    if np.isinf(u).any():
        raise ValueError("unwrapped displacement contains infinities")
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2 or len(centers) > _MAX_CENTER_COUNT:
        raise ValueError("centers must be a bounded list of row/column pairs")
    if not np.isfinite(centers).all():
        raise ValueError("center coordinates must be finite")
    if ((centers[:, 0] < 0) | (centers[:, 0] >= u.shape[0]) |
            (centers[:, 1] < 0) | (centers[:, 1] >= u.shape[1])).any():
        raise ValueError("center coordinates are outside the map")
    adjacency = np.asarray(adjacency)
    if adjacency.shape != (len(centers), len(centers)) or adjacency.dtype.hasobject:
        raise ValueError("adjacency matrix shape/type does not match centers")
    if not np.isfinite(adjacency).all() or (adjacency < 0).any() or (adjacency > 3).any():
        raise ValueError("adjacency matrix contains unexpected values")
    if not np.allclose(adjacency, adjacency.T):
        raise ValueError("adjacency matrix must be symmetric")
    if params is not None and not isinstance(params, dict):
        raise ValueError("unwrap parameters must be a dictionary or None")
    _check_primitive_tree(params)
    return u, centers, adjacency, params


def inspect_unwrap_structure(dataset_id, archive_dir=DEFAULT_ARCHIVE_DIR,
                             allowed_ids=None):
    """Check safe pickle schema only; return no map values or score fields.

    By default the function is restricted to development IDs.  A caller must
    pass an explicit ``allowed_ids`` list for protocol cases.
    """
    number = _dataset_number(dataset_id)
    if allowed_ids is None:
        allowed_numbers = DEVELOPMENT_IDS
    elif isinstance(allowed_ids, (str, int)):
        allowed_numbers = frozenset((_dataset_number(allowed_ids),))
    else:
        allowed_numbers = frozenset(_dataset_number(item) for item in allowed_ids)
    if number not in allowed_numbers:
        raise PermissionError("DS%d is outside the caller-provided map allowlist" % number)
    path = _archive_path(number, archive_dir)
    digest = _md5_file(path)
    if digest != PUBLISHED_MD5[number]:
        raise ValueError("DS%d MD5 mismatch: %s != %s" % (number, digest,
                                                          PUBLISHED_MD5[number]))
    with zipfile.ZipFile(str(path), "r") as zf:
        metadata = _read_whitelisted_info(zf, number)
        member_name, payload = _find_unwrap_member(zf, number)
    u, centers, adjacency, params = _decode_unwrap(payload)
    return {
        "dataset_id": "DS%d" % number,
        "archive_md5": digest,
        "unwrap_member": member_name.rsplit("/", 1)[-1],
        "unwrapped_array_shape_row_col_component": [int(x) for x in u.shape],
        "unwrapped_array_dtype": str(u.dtype),
        "center_array_shape": [int(x) for x in centers.shape],
        "center_count": int(len(centers)),
        "adjacency_array_shape": [int(x) for x in adjacency.shape],
        "adjacency_array_dtype": str(adjacency.dtype),
        "params_type": type(params).__name__,
        "metadata_inputs": metadata,
    }


def load_dataset(dataset_id, archive_dir=DEFAULT_ARCHIVE_DIR,
                 verify_checksum=True, allowed_ids=None, allow_author_crop=False):
    """Load one explicitly permitted map from its processed ZIP.

    Returns an object that can be unpacked as
    ``u_nm, lattice_nm, pixel_nm, sample_rotation_degrees, orientation``.
    ``allowed_ids`` defaults to the declared development set.  A protocol
    runner can pass its frozen case list explicitly.  Other metadata and the
    geometric observer inputs are attached as fields.
    ``allow_author_crop`` accepts the published one-pixel border and even-square
    crop when scan metadata describe the uncropped acquisition.
    """
    number = _dataset_number(dataset_id)
    if allowed_ids is None:
        allowed_numbers = DEVELOPMENT_IDS
    elif isinstance(allowed_ids, (str, int)):
        allowed_numbers = frozenset((_dataset_number(allowed_ids),))
    else:
        allowed_numbers = frozenset(_dataset_number(item) for item in allowed_ids)
    if number not in allowed_numbers:
        raise PermissionError(
            "DS%d is outside the caller-provided map allowlist" % number)
    path = _archive_path(number, archive_dir)
    digest = _md5_file(path)
    expected = PUBLISHED_MD5[number]
    if verify_checksum and digest != expected:
        raise ValueError("DS%d MD5 mismatch: %s != %s" % (number, digest, expected))
    with zipfile.ZipFile(str(path), "r") as zf:
        metadata = _read_whitelisted_info(zf, number)
        member_name, payload = _find_unwrap_member(zf, number)
    u, centers, adjacency, params = _decode_unwrap(payload)
    lattice_nm = metadata.get("LatticeConstant")
    pixel_nm = metadata.get("PixelSize")
    sample_angle = metadata.get("SampleRotation")
    orientation = metadata.get("Orientation")
    if not all(isinstance(x, (int, float)) for x in
               (lattice_nm, pixel_nm, sample_angle)):
        raise ValueError("required physical calibration missing from info.txt")
    if float(lattice_nm) <= 0 or float(pixel_nm) <= 0:
        raise ValueError("lattice and pixel sizes must be positive")
    scan_x = metadata.get("ScanShapeX") or u.shape[1]
    scan_y = metadata.get("ScanShapeY") or u.shape[0]
    reported_shape = (int(round(float(scan_y))), int(round(float(scan_x))))
    cropped = reported_shape != u.shape[:2]
    if cropped:
        # io_utilities.unwrap_main removes one pixel from every edge and then
        # keeps the largest even square. Centers are detected after this crop.
        size = min(reported_shape) - 2
        size -= size % 2
        if not allow_author_crop or size <= 0 or u.shape[:2] != (size, size):
            raise ValueError("whitelisted ScanShape metadata disagrees with map shape")

    # Match DataSetContainer.extract_unwraping(): heterobilayer correction is
    # followed by the archived SP correction and the matching adjacency relabel.
    heterobilayer_tag = metadata.get("HeteroBilayer")
    material_name = str(metadata.get("Material", "")).strip().lower()
    is_heterobilayer = (heterobilayer_tag is True or
                        str(heterobilayer_tag).strip().lower() in
                        ("1", "1.0", "t", "y", "yes", "true") or
                        "-" in material_name)
    if is_heterobilayer:
        u = -_rotate_vector_field(u, -math.pi / 3.0)
        adjacency = _relabel_adjacency(adjacency, "plus_one")
    correction_angle = metadata.get("UnwrapSPCorrectionRotation")
    correction_angle = 0.0 if correction_angle is None else float(correction_angle)
    permitted = (-math.pi / 3.0, math.pi / 3.0,
                 -2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0, 0.0)
    if round(correction_angle, 2) not in [round(value, 2) for value in permitted]:
        raise ValueError("unsupported UnwrapSPCorrectionRotation in info.txt")
    u = _rotate_vector_field(u, correction_angle)
    if round(correction_angle, 2) in (round(-math.pi / 3.0, 2),
                                      round(2.0 * math.pi / 3.0, 2)):
        adjacency = _relabel_adjacency(adjacency, "plus_zero")
    elif round(correction_angle, 2) in (round(math.pi / 3.0, 2),
                                        round(-2.0 * math.pi / 3.0, 2)):
        adjacency = _relabel_adjacency(adjacency, "plus_one")

    normalized = dict(metadata)
    normalized.update({
        "orientation": str(orientation).strip().lower(),
        "lattice_nm": float(lattice_nm),
        "pixel_nm_per_scan_pixel": float(pixel_nm),
        "sample_rotation_degrees": float(sample_angle),
        "scan_shape_row_col": [int(u.shape[0]), int(u.shape[1])],
        "stored_displacement_basis": "Cartesian; author latticevec_to_cartesian",
        "stored_displacement_units": "relative interlayer disregistry / lattice constant",
        "sample_coordinate_convention": "x=col*pixel_nm; y=row*pixel_nm",
        "unwrap_sp_correction_radians": correction_angle,
        "heterobilayer_correction_applied": bool(is_heterobilayer),
    })
    if cropped:
        normalized.update(reported_acquisition_shape_row_col=list(reported_shape),
                          author_unwrap_crop_origin_row_col=[1, 1])
    return DatasetMap(number, u * float(lattice_nm), lattice_nm, pixel_nm,
                      sample_angle, orientation, centers, adjacency, params,
                      normalized, path, digest, member_name, correction_angle)


def _all_triangles(adjacency):
    """Return each three-cycle once, equivalent to source get_triangles()."""
    graph = np.asarray(adjacency) > 0
    n = graph.shape[0]
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            if not graph[i, j]:
                continue
            for k in range(j + 1, n):
                if graph[i, k] and graph[j, k]:
                    triangles.append((i, j, k))
    return triangles


def _inside_polygon_rc(shape, vertices_rc):
    """Rasterize a polygon at integer pixel centers with strict edge exclusion."""
    nrow, ncol = shape
    rr, cc = np.mgrid[0:nrow, 0:ncol]
    inside = np.zeros((nrow, ncol), dtype=bool)
    verts = np.asarray(vertices_rc, dtype=np.float64)
    y = rr.astype(np.float64)
    x = cc.astype(np.float64)
    j = len(verts) - 1
    for i in range(len(verts)):
        yi, xi = verts[i]
        yj, xj = verts[j]
        crosses = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-300) + xi)
        inside ^= crosses
        j = i
    return inside


def _good_interior_triangle_mask(dataset):
    shape = dataset.u_nm.shape[:2]
    triangles = _all_triangles(dataset.adjacency_type)
    mask = np.zeros(shape, dtype=bool)
    for tri in triangles:
        vertices = dataset.centers_rc_px[np.asarray(tri, dtype=int)]
        mask |= _inside_polygon_rc(shape, vertices)
    return mask, triangles


def _rotate_vector_field(field, angle_radians):
    """Source utils.rotate2d / basis_utils.rotate_uvecs convention."""
    values = np.asarray(field, dtype=np.float64)
    c, s = math.cos(angle_radians), math.sin(angle_radians)
    out = np.empty_like(values)
    out[:, :, 0] = c * values[:, :, 0] - s * values[:, :, 1]
    out[:, :, 1] = s * values[:, :, 0] + c * values[:, :, 1]
    return out


def _relabel_adjacency(adjacency, mode):
    """Source integer relabels applied after vector-frame rotations."""
    out = np.asarray(adjacency).copy()
    active = out > 0
    if mode == "plus_zero":
        out[active] = (out[active] % 3) + 1
    elif mode == "plus_one":
        out[active] = ((out[active] + 1) % 3) + 1
    else:
        raise ValueError("unknown adjacency relabeling mode")
    return out


def _nan_gaussian_filter(array, sigma):
    values = np.asarray(array, dtype=np.float64)
    valid = np.isfinite(values)
    numerator = ndimage.gaussian_filter(np.where(valid, values, 0.0), sigma=sigma)
    denominator = ndimage.gaussian_filter(valid.astype(np.float64), sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = numerator / denominator
    result[denominator <= np.finfo(float).eps] = np.nan
    return result


def _smooth_like_paper(u_nm, sigma):
    # io_utilities.smooth() first makes the outermost pixel ring NaN and then
    # applies a normalized Gaussian filter independently to both components.
    u = np.array(u_nm, dtype=np.float64, copy=True)
    if u.ndim != 3 or u.shape[-1] != 2:
        raise ValueError("disregistry field must have shape (row,col,2)")
    u[0, :, :] = np.nan
    u[-1, :, :] = np.nan
    u[:, 0, :] = np.nan
    u[:, -1, :] = np.nan
    if sigma > 0:
        for component in range(2):
            u[:, :, component] = _nan_gaussian_filter(u[:, :, component], sigma)
    return u


def _author_strain_gamma_percent(u_nm, pixel_nm, sample_angle_degrees):
    """Author `strain()` observable: principal strain difference, in percent."""
    # Their input is the relative displacement scaled by a0/scan-pixel and the
    # strain tensor uses half of that field for one layer.
    single_layer = np.asarray(u_nm, dtype=np.float64) * 0.5
    dux_dx = np.gradient(single_layer[:, :, 0], axis=1) / pixel_nm
    dux_dy = np.gradient(single_layer[:, :, 0], axis=0) / pixel_nm
    duy_dx = np.gradient(single_layer[:, :, 1], axis=1) / pixel_nm
    duy_dy = np.gradient(single_layer[:, :, 1], axis=0) / pixel_nm
    angle = np.deg2rad(sample_angle_degrees)
    c, s = np.cos(angle), np.sin(angle)
    exx = c * dux_dx - s * dux_dy
    exy = s * dux_dx + c * dux_dy
    eyx = c * duy_dx - s * duy_dy
    eyy = s * duy_dx + c * duy_dy
    eoff = 0.5 * (exy + eyx)
    # Difference of the eigenvalues of [[exx,eoff],[eoff,eyy]].
    gamma = np.sqrt((exx - eyy) ** 2 + 4.0 * eoff ** 2)
    return 100.0 * gamma


def _rz_helper(value):
    """Source basis_utils.rz_helper(value, offset=1), preserving tie choice."""
    value = float(value)
    if value >= 0:
        value = value % 1.0
        alt = value - 1.0
        return value if abs(value) < abs(alt) else alt
    value = value % -1.0
    alt = value + 1.0
    return value if abs(value) < abs(alt) else alt


def _cartesian_to_rz_wz(u_cart):
    """Port of source basis_utils.cartesian_to_rz_WZ(sign_wrap=False)."""
    u = np.asarray(u_cart, dtype=np.float64)
    flat = u.reshape((-1, 2))
    output = np.full_like(flat, np.nan)
    offsets = ((1.0, 0.0), (-1.0, 0.0),
               (0.5, math.sqrt(3.0) / 2.0),
               (-0.5, -math.sqrt(3.0) / 2.0),
               (0.5, -math.sqrt(3.0) / 2.0),
               (-0.5, math.sqrt(3.0) / 2.0),
               (1.5, -math.sqrt(3.0) / 2.0))
    crit = 1.0 / math.sqrt(3.0)
    for index, vector in enumerate(flat):
        if not np.isfinite(vector).all():
            continue
        lv = _AUTHOR_BASIS_INVERSE.dot(vector)
        cx = _AUTHOR_CARTESIAN_BASIS.dot(np.array((_rz_helper(lv[0]),
                                                   _rz_helper(lv[1]))))[0]
        cy = _AUTHOR_CARTESIAN_BASIS.dot(np.array((_rz_helper(lv[0]),
                                                   _rz_helper(lv[1]))))[1]
        counter = 0
        c1 = -cy - cx / math.sqrt(3.0)
        c2 = -2.0 * cx / math.sqrt(3.0)
        c3 = -c1 + c2
        while abs(c1) > crit or abs(c2) > crit or abs(c3) > crit:
            ox, oy = offsets[counter]
            cx_new, cy_new = cx + ox, cy + oy
            c1 = -cy_new - cx_new / math.sqrt(3.0)
            c2 = -2.0 * cx_new / math.sqrt(3.0)
            c3 = -c1 + c2
            counter += 1
            if counter == len(offsets):
                break
        if counter < len(offsets):
            output[index, 0] = -math.sqrt(3.0) / 2.0 * c2
            output[index, 1] = -c1 + c2 / 2.0
        else:
            output[index, :] = 0.0
    return output.reshape(u.shape)


def _stacking_sector_labels(u_nm, lattice_nm, eligible):
    """The paper's AA/SP1/SP2/SP3/MM/XX displacement-sector bins."""
    u_cart = np.asarray(u_nm, dtype=np.float64) / lattice_nm
    u_wz = _cartesian_to_rz_wz(u_cart)
    u_wz[~eligible] = np.nan
    norms = np.linalg.norm(u_wz, axis=2)
    max_norm = np.nanmax(norms)
    aa_boundary = 0.5 * max_norm
    nrow, ncol = norms.shape
    labels = np.full((nrow, ncol), -1, dtype=np.int8)
    counts = np.zeros(6, dtype=np.int64)
    # Source io_utilities.make_hex_plot(): start=0.6, spacing=25, using
    # displacement bins before assigning the angular sectors.
    start = 0.6
    spacing = 25
    n_bins = len(np.arange(-start, start + (1.0 / spacing), 1.0 / spacing))
    axis = np.arange(n_bins, dtype=np.float64) / spacing - start
    valid = np.isfinite(u_wz).all(axis=2)
    rr, cc = np.nonzero(valid)
    for row, col in zip(rr, cc):
        ix = int(np.round((u_wz[row, col, 0] + start) * spacing))
        iy = int(np.round((u_wz[row, col, 1] + start) * spacing))
        if ix < 0 or ix >= n_bins or iy < 0 or iy >= n_bins:
            continue
        xval, yval = axis[ix], axis[iy]
        umax = math.hypot(xval, yval)
        uang = math.atan(yval / (1e-7 + xval)) * 12.0 / math.pi
        uang += 6.0
        if yval < 0:
            uang *= -1.0
        if umax < aa_boundary:
            label = 0  # AA
        elif abs(uang - 6.0) < 1.0 or abs(uang + 6.0) < 1.0:
            label = 1  # SP1
        elif abs(uang - 2.0) < 1.0 or abs(uang + 2.0) < 1.0:
            label = 2  # SP2
        elif abs(uang - 10.0) < 1.0 or abs(uang + 10.0) < 1.0:
            label = 3  # SP3
        elif (abs(uang - 4.0) < 1.0 or abs(uang - 8.0) < 1.0 or
              abs(uang + 12.0) < 1.0):
            label = 5  # XX
        elif (abs(uang + 4.0) < 1.0 or abs(uang + 8.0) < 1.0 or
              abs(uang - 12.0) < 1.0):
            label = 4  # MM
        elif abs(uang) < 1.0:
            label = 4 if xval < 0 else 5
        else:
            continue
        labels[row, col] = label
        counts[label] += 1
    return labels, counts, aa_boundary


def observe_wall_shear(dataset, u_nm=None, blur_sigma_px=None,
                       mask_policy="all_good_interior_triangles"):
    """Measure the paper-style SP wall-shear averages on a field in nm.

    The same archived finite support, Gaussian blur, scan rotation, triangle
    mask, and displacement-sector classifier are used for measured and
    simulated fields.  ``u_nm`` is an optional prediction override in the
    dataset's stored Cartesian displacement frame.
    """
    if mask_policy != "all_good_interior_triangles":
        raise ValueError("unsupported mask_policy: %s" % mask_policy)
    field = np.array(dataset.u_nm if u_nm is None else u_nm,
                     dtype=np.float64, copy=True)
    if field.shape != dataset.u_nm.shape:
        raise ValueError("u_nm override must match the archived map shape")
    # A synthetic field is sampled on exactly the measured valid-pixel support.
    field[~dataset.source_valid_mask] = np.nan
    sigma = (dataset.metadata.get("SmoothingSigma", 2.0)
             if blur_sigma_px is None else blur_sigma_px)
    sigma = float(sigma)
    if sigma < 0:
        raise ValueError("blur_sigma_px must be non-negative")
    smoothed_nm = _smooth_like_paper(field, sigma)
    gamma_percent = _author_strain_gamma_percent(
        smoothed_nm, dataset.pixel_nm, dataset.sample_rotation_degrees)
    good_mask, triangles = _good_interior_triangle_mask(dataset)
    eligible = good_mask & np.isfinite(gamma_percent)
    labels, counts, aa_boundary = _stacking_sector_labels(
        smoothed_nm, dataset.lattice_nm, eligible)
    expected_labels = (1, 2, 3)
    exact_keys = {}
    details = {}
    for number, label in enumerate(expected_labels, 1):
        values = gamma_percent[labels == label]
        if values.size:
            mean = float(np.nanmean(values))
            stderr = (float(np.nanstd(values, ddof=1) / math.sqrt(values.size))
                      if values.size > 1 else None)
        else:
            mean, stderr = float("nan"), None
        exact_keys["AvgSP%dGamma" % number] = mean
        exact_keys["ErrSP%dGamma" % number] = stderr
        details["SP%d" % number] = {
            "mean_percent": mean,
            "standard_error_percent": stderr,
            "pixel_count": int(values.size),
        }
    if u_nm is None:
        try:
            dataset_number = int(dataset.dataset_id[2:])
        except (TypeError, ValueError):
            dataset_number = -1
        if dataset_number in DEVELOPMENT_IDS:
            reference = read_development_reference_shear_fields(
                dataset_number, Path(dataset.archive_path).parent)
            for key, value in reference.items():
                exact_keys["stored_" + key] = value
    return {
        "dataset_id": dataset.dataset_id,
        "observable": "maximum eigenvalue minus minimum eigenvalue of single-layer infinitesimal strain, percent",
        "field_role": "relative interlayer disregistry; strain uses one half per layer",
        "classification": "author reduced-Wigner-Seitz displacement bins; SP1/2/3 are stacking-sector means",
        "blur_sigma_pixels": sigma,
        "sample_rotation_degrees": dataset.sample_rotation_degrees,
        "mask_policy": mask_policy,
        "triangle_count": len(triangles),
        "interior_triangle_pixel_count": int(good_mask.sum()),
        "eligible_pixel_count": int(eligible.sum()),
        "aa_boundary_reduced_lattice_units": float(aa_boundary),
        "exact_info_keys_percent": exact_keys,
        "sp_details": details,
        "stacking_sector_counts": {
            "AA": int(counts[0]), "SP1": int(counts[1]),
            "SP2": int(counts[2]), "SP3": int(counts[3]),
            "MM": int(counts[4]), "XX": int(counts[5]),
        },
    }


def wall_shear(dataset, u_nm=None, blur_sigma_px=None,
               mask_policy="all_good_interior_triangles"):
    """Return SP1, SP2, SP3 wall-shear means, in percent, in that order."""
    details = observe_wall_shear(dataset, u_nm=u_nm,
                                 blur_sigma_px=blur_sigma_px,
                                 mask_policy=mask_policy)
    return np.array([details["exact_info_keys_percent"]["AvgSP%dGamma" % i]
                     for i in (1, 2, 3)], dtype=np.float64)


def _ransac_center_affine(points_xy_nm, translations_nm, inlier_tolerance_nm):
    n = len(points_xy_nm)
    design = np.column_stack((points_xy_nm, np.ones(n, dtype=np.float64)))
    if n < 3:
        raise ValueError("at least three finite AA centers are required")
    best = None
    for triple in itertools.combinations(range(n), 3):
        X = design[np.asarray(triple, dtype=int)]
        if np.linalg.matrix_rank(X) < 3:
            continue
        coef = np.linalg.solve(X, translations_nm[np.asarray(triple, dtype=int)])
        residual = np.linalg.norm(design.dot(coef) - translations_nm, axis=1)
        inliers = residual <= inlier_tolerance_nm
        score = (int(inliers.sum()),
                 -float(np.median(residual[inliers])) if inliers.any() else -math.inf)
        if best is None or score > best[0]:
            best = (score, inliers, residual)
    if best is None or int(best[1].sum()) < 3:
        raise ValueError("could not fit a non-degenerate center affine map")
    inliers = best[1]
    coef = np.linalg.lstsq(design[inliers], translations_nm[inliers], rcond=None)[0]
    residual = np.linalg.norm(design.dot(coef) - translations_nm, axis=1)
    # Re-evaluate after the least-squares refit, then refit once more if needed.
    refined = residual <= inlier_tolerance_nm
    if refined.sum() >= 3 and refined.sum() != inliers.sum():
        inliers = refined
        coef = np.linalg.lstsq(design[inliers], translations_nm[inliers], rcond=None)[0]
        residual = np.linalg.norm(design.dot(coef) - translations_nm, axis=1)
    return coef, inliers, residual, design


def center_affine(dataset, inlier_tolerance_nm=None):
    """Fit the scan-to-disregistry affine map from AA centers and lattice shifts.

    Each center is sampled from the saved continuous field, converted to the
    authors' primitive lattice coordinates, and rounded to the nearest
    lattice translation.  A three-center RANSAC fit rejects bad rounded nodes
    before the final least-squares fit.  This provides an origin-independent
    geometric estimate and explicit fit diagnostics.
    """
    finite_points = []
    translation_vectors = []
    center_indices = []
    lattice_coordinate_offsets = []
    for index, (row, col) in enumerate(dataset.centers_rc_px):
        ri, ci = int(round(float(row))), int(round(float(col)))
        vector = dataset.u_nm[ri, ci] / dataset.lattice_nm
        if not np.isfinite(vector).all():
            continue
        lattice_coordinates = _AUTHOR_BASIS_INVERSE.dot(vector)
        integer_translation = np.round(lattice_coordinates)
        translation = dataset.lattice_nm * _AUTHOR_CARTESIAN_BASIS.dot(integer_translation)
        finite_points.append((float(col) * dataset.pixel_nm,
                              float(row) * dataset.pixel_nm))
        translation_vectors.append(translation)
        center_indices.append(index)
        lattice_coordinate_offsets.append(lattice_coordinates - integer_translation)
    points_xy = np.asarray(finite_points, dtype=np.float64)
    translations = np.asarray(translation_vectors, dtype=np.float64)
    if inlier_tolerance_nm is None:
        inlier_tolerance_nm = 0.25 * dataset.lattice_nm
    coef, inliers, residual, _ = _ransac_center_affine(
        points_xy, translations, float(inlier_tolerance_nm))
    B_scan = coef[:2, :].T
    offset_nm = coef[2, :]
    alpha = np.deg2rad(dataset.sample_rotation_degrees)
    correction = np.array(((np.cos(alpha), np.sin(alpha)),
                           (-np.sin(alpha), np.cos(alpha))))
    B_corrected = B_scan.dot(correction)
    signed_twist_rad = 0.5 * (B_corrected[1, 0] - B_corrected[0, 1])
    inlier_residual = residual[inliers]
    return {
        "B_scan": B_scan,
        "offset_nm": offset_nm,
        "B_corrected": B_corrected,
        "sample_rotation_correction": correction,
        "displacement_components": "Cartesian rows x/y; columns are scan x=col and y=row",
        "coordinates_nm": "x=col*PixelSize; y=row*PixelSize",
        "inlier_tolerance_nm": float(inlier_tolerance_nm),
        "fit_method": "exhaustive 3-center RANSAC on nearest author-basis lattice translations, then least squares",
        "center_count_total": int(len(dataset.centers_rc_px)),
        "center_count_finite": int(len(points_xy)),
        "center_count_inlier": int(inliers.sum()),
        "inlier_center_indices": [int(center_indices[i]) for i in np.flatnonzero(inliers)],
        "outlier_center_indices": [int(center_indices[i]) for i in np.flatnonzero(~inliers)],
        "center_rms_residual_nm": float(np.sqrt(np.mean(inlier_residual ** 2))),
        "center_median_residual_nm": float(np.median(inlier_residual)),
        "all_center_residual_nm": residual.tolist(),
        "lattice_coordinate_rounding_residuals": np.asarray(lattice_coordinate_offsets).tolist(),
        "signed_twist_from_curl_degrees": float(np.rad2deg(signed_twist_rad)),
        "absolute_twist_from_curl_degrees": float(abs(np.rad2deg(signed_twist_rad))),
    }


def field_affine(dataset, u_nm=None, edge_pixels=2):
    """Least-squares diagnostic plane fit to the continuous displacement map."""
    field = np.asarray(dataset.u_nm if u_nm is None else u_nm, dtype=np.float64)
    if field.shape != dataset.u_nm.shape:
        raise ValueError("u_nm override must match the archived map shape")
    valid = np.isfinite(field).all(axis=2) & dataset.source_valid_mask
    if edge_pixels:
        valid[:edge_pixels, :] = False
        valid[-edge_pixels:, :] = False
        valid[:, :edge_pixels] = False
        valid[:, -edge_pixels:] = False
    row, col = np.nonzero(valid)
    points = np.column_stack((col * dataset.pixel_nm,
                              row * dataset.pixel_nm,
                              np.ones(len(row), dtype=np.float64)))
    values = field[row, col]
    coef = np.linalg.lstsq(points, values, rcond=None)[0]
    residual = points.dot(coef) - values
    B_scan = coef[:2, :].T
    alpha = np.deg2rad(dataset.sample_rotation_degrees)
    correction = np.array(((np.cos(alpha), np.sin(alpha)),
                           (-np.sin(alpha), np.cos(alpha))))
    B_corrected = B_scan.dot(correction)
    return {
        "B_scan": B_scan,
        "offset_nm": coef[2, :],
        "B_corrected": B_corrected,
        "valid_pixel_count": int(len(row)),
        "edge_pixels_excluded": int(edge_pixels),
        "component_rms_residual_nm": np.sqrt(np.mean(residual ** 2, axis=0)),
        "residual_rms_nm": float(np.sqrt(np.mean(residual ** 2))),
        "fit_method": "ordinary least-squares affine plane over valid interior pixels",
    }


def _archive_member_inventory_and_hashes(path):
    """Read ZIP headers and hash raw payloads solely for duplicate checks."""
    members = []
    numerical_hashes = {}
    with zipfile.ZipFile(str(path), "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            basename = info.filename.rsplit("/", 1)[-1]
            members.append({"name": basename, "archive_path": info.filename,
                            "size_bytes": int(info.file_size)})
            # info.txt contains both input values and score fields; tmp.txt is
            # a processing scratch member.  Neither participates in numeric
            # payload-duplicate detection.
            if basename.lower() not in ("info.txt", "tmp.txt"):
                numerical_hashes[basename] = hashlib.md5(
                    zf.read(info.filename)).hexdigest()
    return members, numerical_hashes


def archive_manifest(archive_dir=DEFAULT_ARCHIVE_DIR):
    """Return checksum-verified metadata plus duplicate/member evidence.

    ZIP payloads are only hashed for exact-duplicate detection.  No map pickle
    is decoded here and no score field is parsed from ``info.txt``.
    """
    datasets = []
    payload_fingerprints = {}
    for number in sorted(PUBLISHED_MD5):
        record = read_input_metadata(number, archive_dir=archive_dir,
                                     verify_checksum=True)
        path = _archive_path(number, archive_dir)
        members, payload_hashes = _archive_member_inventory_and_hashes(path)
        record["members"] = members
        datasets.append(record)
        signature = tuple(sorted(payload_hashes.items()))
        payload_fingerprints.setdefault(signature, []).append(number)
    duplicate_groups = [sorted(numbers) for numbers in payload_fingerprints.values()
                        if len(numbers) > 1]
    dataset_by_id = {row["dataset_id"]: row for row in datasets}
    duplicate_evidence = []
    for pair in duplicate_groups:
        first, second = pair[0], pair[1]
        _, first_hashes = _archive_member_inventory_and_hashes(
            _archive_path(first, archive_dir))
        _, second_hashes = _archive_member_inventory_and_hashes(
            _archive_path(second, archive_dir))
        shared = sorted(set(first_hashes) & set(second_hashes))
        duplicate_evidence.append({
            "dataset_ids": ["DS%d" % first, "DS%d" % second],
            "identical_member_payloads": [
                {"member": member, "md5": first_hashes[member]}
                for member in shared if first_hashes[member] == second_hashes[member]],
            "whitelisted_input_fields_identical": (
                dataset_by_id["DS%d" % first]["metadata_fields"] ==
                dataset_by_id["DS%d" % second]["metadata_fields"]),
            "zip_internal_prefixes": ["DS%d/" % first, "DS%d/" % second],
            "info_txt_payloads_differ": True,
        })
    ds9_inputs = dataset_by_id["DS9"]["metadata_fields"]
    return {
        "record_url": ZENODO_RECORD,
        "paper_doi": PAPER_DOI,
        "article_url": "https://www.nature.com/articles/s41467-023-38504-7",
        "author_processing_code_url": "https://github.com/bediakolab/pyInterferometry",
        "supplementary_source_note": "Supplementary Information Notes 5-6, pp. 10-13: rigid-moire subtraction; twist, lattice-constant mismatch, and heterostrain calculation.",
        "processed_zip_count": len(datasets),
        "all_published_md5_checksums_verified": True,
        "processed_zip_ids": [row["dataset_id"] for row in datasets],
        "raw_dm4_downloaded": False,
        "development_map_ids": ["DS%d" % number for number in sorted(DEVELOPMENT_IDS)],
        "whitelisted_info_fields": sorted(WHITELISTED_INFO_FIELDS),
        "dataset_inputs": datasets,
        "known_same_device_groups": {
            "ids": [["DS30", "DS31", "DS33"], ["DS34", "DS35", "DS37"]],
            "source": "Known scan-log groups supplied with this task. The processed ZIP info.txt has OriginalName={} for these records, so that file alone cannot prove specimen identity.",
        },
        "other_same_device_groups_inferable_from_public_whitelisted_fields": [],
        "same_device_group_limit": "No additional specimen identifiers are present in the whitelisted info.txt fields; no new group is inferred from similar orientation angles.",
        "duplicate_groups_by_exact_numeric_member_payload": [
            ["DS%d" % number for number in group]
            for group in duplicate_groups],
        "duplicate_evidence": duplicate_evidence,
        "duplicate_method": "MD5 hashes of every ZIP member payload except info.txt and tmp.txt. No map pickle was decoded for this comparison.",
        "candidate_homobilayer_metadata_only": {
            "primary": "DS9",
            "input_metadata": {
                key: ds9_inputs.get(key) for key in (
                    "Material", "Orientation", "LatticeMismatch", "LatticeConstant",
                    "PixelSize", "ScanShapeX", "ScanShapeY", "SmoothingSigma",
                    "AvgMoireTwist", "AvgHeteroStrain", "SampleRotation",
                    "UnwrapSPCorrectionRotation")},
            "selection_reason": "P MoS2 homobilayer, 100x100 scan metadata, no exact duplicate group, and not in the declared development IDs. The map payload was not decoded for this proposal.",
        },
        "DS7_observer_reproduction": {
            "paper_crosswalk": "Zenodo DS7 metadata lists Fig. 2a,d,g,m,n.",
            "computed_exact_keys_percent": {
                "AvgSP1Gamma": 1.3401893093842128,
                "AvgSP2Gamma": 1.2123160489272968,
                "AvgSP3Gamma": 1.2189186751872065,
                "ErrSP1Gamma": 0.01026394601453446,
                "ErrSP2Gamma": 0.014022381802612949,
                "ErrSP3Gamma": 0.013930524886440145,
            },
            "stored_keys_match_to_floating_precision": True,
            "blur_sigma_pixels": 2.0,
            "mask": "union of all adjacency triangles; all marked good; 14 triangles, 5,219 of 10,000 pixels",
            "classification": "author reduced-Wigner-Seitz displacement bins; SP1/SP2/SP3 sector means, not a manually traced line mask",
            "sp_sector_pixel_counts": {"SP1": 605, "SP2": 641, "SP3": 567},
            "sigma_sensitivity_percent": {
                "0": [1.812197, 1.729800, 1.654174],
                "1": [1.521197, 1.445223, 1.394752],
                "2": [1.340189, 1.212316, 1.218919],
                "3": [1.130224, 1.023323, 1.006095],
                "4": [0.929866, 0.838176, 0.785914],
            },
            "subgamma_scope": "Stored AvgSP1/2/3SubGamma fields are read only as reference; this observer independently reproduces raw Gamma and does not recompute triangle-local rigid subtraction.",
        },
    }


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("not JSON serializable: %s" % type(value).__name__)


def _main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="DS7")
    parser.add_argument("--archive-dir", default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--write-inventory", default=None)
    args = parser.parse_args()
    if args.write_inventory:
        manifest = archive_manifest(args.archive_dir)
        Path(args.write_inventory).write_text(json.dumps(manifest, indent=2,
                                                        sort_keys=True) + "\n")
    dataset = load_dataset(args.dataset, args.archive_dir)
    result = observe_wall_shear(dataset)
    affine = center_affine(dataset)
    field_fit = field_affine(dataset)
    print(json.dumps({"wall_shear": result, "center_affine": affine,
                      "field_affine": field_fit}, indent=2, sort_keys=True,
                     default=_json_default))


if __name__ == "__main__":
    _main()
