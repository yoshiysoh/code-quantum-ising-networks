"""Import smoke test for qising_networks and the driver scripts.

Catches a broken import or a stale path assumption immediately, without
running any real computation. Run it after any change to
src/qising_networks/, scripts/, or the repository layout.
"""

from __future__ import annotations

import importlib
import runpy
from pathlib import Path

import pytest

LIBRARY_MODULES = [
    "qising_networks.class_fractions",
    "qising_networks.examples",
    "qising_networks.fig_maker",
    "qising_networks.fig_maker_helper",
    "qising_networks.model",
    "qising_networks.plots.common",
    "qising_networks.plots.dynamics",
    "qising_networks.plots.fractions",
    "qising_networks.plots.spectral",
    "qising_networks.scar_metrics",
    "qising_networks.spectra",
]
SCRIPTS = sorted((Path(__file__).resolve().parent.parent / "scripts")
                 .glob("*.py"))


@pytest.mark.parametrize("module_name", LIBRARY_MODULES)
def test_library_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)


def test_path_constants_resolve() -> None:
    from qising_networks import DATA_DIR, FIGURES_DIR, PROJECT_ROOT

    assert PROJECT_ROOT.is_dir()
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert FIGURES_DIR == PROJECT_ROOT / "figures"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_script_imports(path: Path) -> None:
    """Load a driver script without running main(): run_name differs from
    "__main__", so its `if __name__ == "__main__":` guard never fires.
    Scripts import only from qising_networks, never from each other."""
    runpy.run_path(str(path), run_name=f"_smoketest_{path.stem}")
