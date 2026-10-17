"""Tiny runs of the data scripts and of the HDF5 helpers.

Each script runs at the smallest size in a temporary directory and must
write its file with the expected contents and a "provenance" group. The
runs take seconds and catch a broken pipeline before a long run does.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from qising_networks import DATA_DIR, h5io
from qising_networks.scar_metrics import NULL_FILE

ROOT = Path(__file__).resolve().parent.parent
ENV = {
    **os.environ,
    "NUM_WORKERS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
# The class rule reads the revival null from data/.
needs_null = pytest.mark.skipif(
    not (DATA_DIR / NULL_FILE).exists(), reason="data/ not downloaded"
)
PROVENANCE_KEYS = {
    "git_commit",
    "command",
    "python",
    "numpy",
    "cpu",
    "blas",
    "created_utc",
}


def run_script(name: str, *args: str) -> None:
    """Run scripts/<name> from the repository root; fail on error."""
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *args],
        cwd=ROOT,
        env=ENV,
        check=True,
        capture_output=True,
    )


def assert_provenance(path: Path) -> None:
    with h5py.File(path) as f:
        assert PROVENANCE_KEYS <= set(f[h5io.PROVENANCE].attrs)


def test_save_and_load_arrays(tmp_path):
    path = tmp_path / "a.h5"
    arrays = {"x": np.arange(3), "ragged": [np.arange(2), np.arange(5)]}
    h5io.save_arrays(path, arrays, {"label": "test"})
    loaded = h5io.load_arrays(path)
    assert set(loaded) == {"x", "ragged", "label"}
    assert np.array_equal(loaded["x"], arrays["x"])
    assert [len(a) for a in loaded["ragged"]] == [2, 5]
    assert_provenance(path)


def load_script(name: str):
    """scripts/<name> as a module, without running its main()."""
    spec = importlib.util.spec_from_file_location(
        name.removesuffix(".py"), ROOT / "scripts" / name
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_revival_null():
    # The full script sweeps 40 dimensions up to 4096 whatever N is; the
    # smoke test calls its two steps at small size instead.
    null = load_script("build_revival_null.py")
    real = null.measure_real_extents(8, "ferro", 1)
    assert real.shape[1] == 2 and len(real)
    _dims, extents = null.extent_model(real)
    rng = np.random.default_rng(42)
    for after_dip in (False, True):
        vals = null.null_at(16, float(extents[-1]), 3, rng, after_dip)
        assert np.all((vals >= 0) & (vals <= 1))


@needs_null
def test_class_dataset(tmp_path):
    run_script(
        "generate_class_dataset.py",
        "--N",
        "8",
        "--nets",
        "1",
        "--model",
        "ferro",
        "--out-dir",
        str(tmp_path),
    )
    out = tmp_path / "classes_N8_ferro.h5"
    with h5py.File(out) as f:
        assert len(f["realisations"]) == 9  # one network, phi = 0..8/8
    assert_provenance(out)


def test_spectral_data(tmp_path):
    run_script(
        "generate_spectral_data.py",
        "--sizes",
        "8",
        "--nets",
        "1",
        "--model",
        "ferro",
        "--workers",
        "1",
        "--out-dir",
        str(tmp_path),
    )
    out = tmp_path / "spectral_data.h5"
    assert h5io.load_arrays(out)
    assert_provenance(out)


def test_hamiltonian_check(tmp_path):
    run_script(
        "verify_hamiltonian_approximation.py",
        "--N",
        "8",
        "--model",
        "ferro",
        "--gamma",
        "0.3",
        "--nsamples",
        "2",
        "--workers",
        "1",
        "--skip-self-test",
        "--out-dir",
        str(tmp_path),
    )
    out = tmp_path / "hamapprox_N8_ferro_gamma0.30.h5"
    with h5py.File(out) as f:
        assert len(f["samples"]["sample_idx"]) == 2
    assert_provenance(out)


@needs_null
@pytest.mark.parametrize("model", ["ferro", "pmJ"])
def test_ring_example(tmp_path, model):
    args = [
        "--N",
        "12",
        "--h",
        "2.0",
        "--model",
        model,
        "--out-dir",
        str(tmp_path),
    ]
    if model == "pmJ":
        args += ["--seed", "0"]
    run_script("generate_ring_example.py", *args)
    (out,) = tmp_path.glob(f"ring_N12_{model}_*.h5")
    assert_provenance(out)
