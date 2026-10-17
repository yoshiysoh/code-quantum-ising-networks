"""Named arrays in HDF5, the format of every file in data/ that is not a
per-realisation table (class_fractions.write_h5).

Arrays are datasets, a list of arrays of different lengths is a group of
datasets "0", "1", ..., and scalars and strings are attributes of the
file. load_arrays returns all three in one dict.

Every file written here or by the scripts also carries a "provenance"
group whose attributes record how it was made (write_provenance);
load_arrays skips it.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

PROVENANCE = "provenance"
PACKAGES = ("numpy", "scipy", "h5py", "networkx", "matplotlib")


def _cpu_model() -> str:
    """CPU model name, or the platform's processor string."""
    try:
        if sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor() or "unknown"


def _git_commit() -> str:
    """Commit of the code that wrote the file, with "+dirty" if the
    working tree has changes, or "unknown" outside a git checkout."""
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("+dirty" if dirty.strip() else "")


def provenance() -> dict:
    """How the current process runs: code commit, command line, package
    versions, CPU, BLAS libraries and threads, and the UTC time."""
    from threadpoolctl import threadpool_info

    # Library file names only: full paths would record the user's home.
    blas = [
        {**lib, "filepath": Path(lib.get("filepath") or "").name}
        for lib in threadpool_info()
    ]
    info = {
        "git_commit": _git_commit(),
        # The home directory as "~", so paths in arguments stay anonymous.
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]).replace(
            str(Path.home()), "~"
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "blas": json.dumps(blas),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
    }
    for name in PACKAGES:
        try:
            info[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info[name] = "not installed"
    return info


def write_provenance(f: h5py.File) -> None:
    """Store provenance() as attributes of the group "provenance" of an
    open, writable file."""
    g = f.require_group(PROVENANCE)
    for key, value in provenance().items():
        g.attrs[key] = value


def save_arrays(path: Path, arrays: dict, attrs: dict | None = None) -> None:
    """Write `arrays` (name -> array or list of arrays) and `attrs`
    (name -> scalar or string) to `path`, atomically."""
    tmp = Path(path).with_suffix(".partial.h5")
    with h5py.File(tmp, "w") as f:
        for name, value in arrays.items():
            if isinstance(value, (list, tuple)):
                g = f.create_group(name)
                for i, a in enumerate(value):
                    g.create_dataset(
                        str(i), data=np.asarray(a), compression="gzip"
                    )
            else:
                f.create_dataset(
                    name, data=np.asarray(value), compression="gzip"
                )
        for name, value in (attrs or {}).items():
            f.attrs[name] = value
        write_provenance(f)
    tmp.replace(path)


def load_arrays(path: Path) -> dict:
    """Every array, list of arrays and attribute of a save_arrays file."""
    out = {}
    with h5py.File(path, "r") as f:
        for name, item in f.items():
            if name == PROVENANCE:
                continue
            if isinstance(item, h5py.Group):
                out[name] = [item[k][()] for k in sorted(item, key=int)]
            else:
                out[name] = item[()]
        for name, value in f.attrs.items():
            out[name] = value.decode() if isinstance(value, bytes) else value
    return out
