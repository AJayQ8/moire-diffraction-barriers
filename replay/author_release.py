"""Load the paper's archived pyInterferometry code for integrated-disk replay."""

from hashlib import md5
from importlib import import_module
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from zipfile import ZipFile


RELEASE_MD5 = "7891a65ffa50f72e8e2e1f5ffd86083a"
RELEASE_COMMIT = "85f21dd9134b600000e517ad1b22acc725571376"


def register_unused_import(name, package=False):
    """Satisfy legacy imports for raw-acquisition functions outside this replay."""
    if name in sys.modules:
        return sys.modules[name]
    module = ModuleType(name)
    module.__all__ = []
    if package:
        module.__path__ = []
    sys.modules[name] = module
    parent, separator, child = name.rpartition(".")
    if separator:
        setattr(sys.modules[parent], child, module)
    return module


def load_author_release(src, dependencies):
    """Import numerical routines from the checksum-verified 2023 source."""
    src = Path(src).resolve()
    dependencies = Path(dependencies).resolve()
    archive = src.parents[1] / "release.zip"
    if md5(archive.read_bytes()).hexdigest() != RELEASE_MD5:
        raise ValueError("The pyInterferometry release archive does not match Zenodo")
    with ZipFile(archive) as release:
        if release.comment.decode() != RELEASE_COMMIT:
            raise ValueError("The author release commit does not match the archived record")
        for member in release.namelist():
            path = Path(member)
            if path.parent.name == "src" and path.suffix == ".py":
                if (src / path.name).read_bytes() != release.read(member):
                    raise ValueError(f"Author source differs from release: {path.name}")

    import matplotlib

    matplotlib.use("Agg")
    for path in (dependencies, src):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    py4dstem = register_unused_import("py4DSTEM", package=True)
    py4dstem.__version__ = "raw input skipped"
    register_unused_import("py4DSTEM.io", package=True)
    register_unused_import("py4DSTEM.process", package=True)
    register_unused_import("py4DSTEM.process.diskdetection")
    register_unused_import("py4DSTEM.process.virtualimage")
    register_unused_import("ncempy", package=True)
    register_unused_import("ncempy.io", package=True)
    register_unused_import("ncempy.io.dm")
    register_unused_import("numba")
    modules = {key: import_module(name) for key, name in (
        ("fit", "interferometry_fitting"), ("io", "io_utilities"),
        ("unwrap", "unwrap_utils"), ("strain", "strain"),
        ("diskset", "diskset"), ("basis", "basis_utils"), ("utils", "utils"))}
    if any(Path(module.__file__).resolve().parent != src for module in modules.values()):
        raise ImportError("A module was loaded outside the pinned author release")
    return SimpleNamespace(**modules)
