"""Class fractions of the largest sector against flippability (Figs. 2 and
4, Extended Data Figs. 2, 4 and 8).

Reads generate_class_dataset.py's per-state records, classifies them with
scar_metrics.classify, and reduces them to the fraction of the sector in
each class at each flippability. Measured states are weighted by
comp_size / n_measured of their component, since large components are
sampled. Error bars are the standard deviation across networks, not the
standard error of the mean.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from . import DATA_DIR as RESULTS
from . import scar_metrics as sm
from .h5io import write_provenance

COLUMNS = (
    "fid_min",
    "fid_rev",
    "n_eff",
    "n_inv1",
    "comp_size",
    "support",
    "weight",
)
COL = {name: i for i, name in enumerate(COLUMNS)}
# Settings of generate_class_dataset.py.
TIME_MAX: float = 2000.0
MAX_COMPONENT_DEFAULT: int = 800
SAMPLE_SIZE_DEFAULT: int = 100


def write_h5(path: Path, rows: list[dict], params: dict) -> None:
    """Write per-realisation state tables (generate_class_dataset.py's
    format, read back by load()); atomic via a .partial.h5 rename."""
    tmp = path.with_suffix(".partial.h5")
    with h5py.File(tmp, "w") as f:
        write_provenance(f)
        p = f.create_group("params")
        for k, v in params.items():
            p.attrs[k] = v
        p.attrs["columns"] = np.array(COLUMNS, dtype=h5py.string_dtype())
        g = f.create_group("realisations")
        for i, r in enumerate(rows):
            grp = g.create_group(str(i))
            grp.create_dataset(
                "states",
                data=r["states"],
                compression="gzip",
                compression_opts=4,
            )
            for k, v in r.items():
                if k != "states":
                    grp.attrs[k] = v
    tmp.replace(path)


def load(
    num_spins: int, model: str, results: Path = RESULTS
) -> list[dict] | None:
    """Every realisation of one (N, model)."""
    path = results / f"classes_N{num_spins}_{model}.h5"
    if not path.exists():
        return None
    rows = []
    with h5py.File(path) as f:
        for key in sorted(f["realisations"], key=int):
            g = f["realisations"][key]
            rows.append({**dict(g.attrs), "states": g["states"][:]})
    return rows


def fractions(
    num_spins: int,
    model: str,
    giant_only: bool = False,
    part: str = "n_eff",
    part_frac: float = 0.5,
    normalize: bool = True,
    **kw,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(phi levels, mean fraction per class, SD across networks).

    `giant_only` restricts the population to the largest subspace of each
    sector -- the same classification asked of the states that actually share
    one connected, thermalizing Hilbert-space block.

    `normalize=False` skips the divide-by-total step and returns the
    weighted state *count* per class instead of its fraction of the
    population -- same weighting (comp_size / n_measured_in_component), just
    not renormalized to sum to one.
    """
    phi, frac = per_network_fractions(
        num_spins,
        model,
        giant_only=giant_only,
        part=part,
        part_frac=part_frac,
        normalize=normalize,
        **kw,
    )
    if phi is None:
        return None
    per_level: dict[float, list[np.ndarray]] = {}
    for p, f in zip(phi, frac):
        per_level.setdefault(round(float(p), 6), []).append(f)
    levels = np.array(sorted(per_level))
    mean = np.array([np.mean(per_level[x], axis=0) for x in levels])
    sd = np.array([np.std(per_level[x], axis=0) for x in levels])
    return levels, mean, sd


def per_network_fractions(
    num_spins: int,
    model: str,
    giant_only: bool = False,
    part: str = "n_eff",
    part_frac: float = 0.5,
    normalize: bool = True,
    **kw,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """(phi per network, fraction per class per network), one row per
    realisation -- the un-aggregated form `fractions()` reduces further.
    Exposed so callers (e.g. a bootstrap over networks) can resample rows
    within a phi level themselves rather than only seeing the mean/SD.
    """
    rows = load(num_spins, model, **kw)
    if rows is None:
        return None, None
    phi_list, frac_list = [], []
    for r in rows:
        s = r["states"]
        keep = (
            (s[:, COL["comp_size"]] == r["giant_size"])
            if giant_only
            else np.ones(len(s), dtype=bool)
        )
        if not keep.any():
            continue
        cls = sm.classify(
            s[keep, COL["fid_min"]],
            s[keep, COL["fid_rev"]],
            s[keep, COL[part]],
            num_spins,
            comp_size=s[keep, COL["comp_size"]],
            part_frac=part_frac,
        )
        w = s[keep, COL["weight"]].astype(float)
        frac = np.array(
            [w[cls == c].sum() for c in range(len(sm.CLASS_NAMES))]
        )
        if normalize:
            total = frac.sum()
            if total <= 0:
                continue
            frac = frac / total
        phi_list.append(float(r["phi_target"]))
        frac_list.append(frac)
    return np.array(phi_list), np.array(frac_list)
