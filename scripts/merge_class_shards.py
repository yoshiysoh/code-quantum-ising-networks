"""Merge partial runs of generate_class_dataset.py (--shard, --job-range or
checkpoints) into the file the unsharded run would have written.

Each realisation's seed is fixed by its position in the full job list, so
copies from different runs agree to round-off; the first is kept. The
output holds every (net, phi) realisation once, sorted by (phi, net).

Run: python3 merge_class_shards.py --out OUT.h5 IN1.h5 IN2.h5 ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from qising_networks.class_fractions import write_h5

SELECTION_KEYS = ("selection", "shard_index", "n_shards", "n_jobs_total")
RTOL, ATOL = 1e-8, 1e-10


def read_rows(path: Path) -> tuple[dict, list[dict]]:
    with h5py.File(path, "r") as f:
        params = dict(f["params"].attrs)
        rows = []
        for g in f["realisations"].values():
            r = dict(g.attrs)
            r["states"] = g["states"][()]
            rows.append(r)
    params = {k: v for k, v in params.items() if k not in SELECTION_KEYS}
    return params, rows


def same(a, b) -> bool:
    return np.array_equal(np.asarray(a), np.asarray(b))


def same_row(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)


def close_row(a: dict, b: dict) -> float | None:
    """Largest |a - b| over the per-state columns if the two copies of a
    realisation agree to round-off, else None."""
    if a.keys() != b.keys() or a["states"].shape != b["states"].shape:
        return None
    if not all(same(a[k], b[k]) for k in a if k != "states"):
        return None
    sa, sb = a["states"], b["states"]
    if not np.allclose(sa, sb, rtol=RTOL, atol=ATOL):
        return None
    return float(np.max(np.abs(sa - sb), initial=0.0))


def merge(inputs: list[Path],
          report_missing: bool = False) -> tuple[dict, list[dict]]:
    base: dict | None = None
    rows: dict[tuple[int, float], dict] = {}
    for path in inputs:
        params, new = read_rows(path)
        if base is None:
            base = params
        elif not same_row(params, base):
            raise ValueError(f"{path}: params differ from {inputs[0]}")
        n_dup, max_dev = 0, 0.0
        for r in new:
            key = (int(r["net_idx"]), float(r["phi_target"]))
            if key in rows:
                dev = close_row(r, rows[key])
                if dev is None:
                    raise ValueError(f"{path}: realisation {key} differs")
                n_dup, max_dev = n_dup + 1, max(max_dev, dev)
            else:
                rows[key] = r
        print(f"{path.name}: {len(new)} realisations ({n_dup} already seen, "
              f"max deviation {max_dev:.1e})")
    if base is None:
        raise ValueError("no inputs")

    expected = {(net, float(lvl)) for lvl in base["phi_levels"]
                for net in range(int(base["n_nets"]))}
    missing, extra = expected - rows.keys(), rows.keys() - expected
    if report_missing:
        levels = [float(v) for v in base["phi_levels"]]
        idx = sorted(levels.index(phi) * int(base["n_nets"]) + net
                     for net, phi in missing)
        print(":".join(map(str, idx)))
        return base, []
    if missing or extra:
        raise ValueError(f"{len(missing)} realisations missing "
                         f"(e.g. {sorted(missing)[:5]}), {len(extra)} extra")
    merged = sorted(rows.values(),
                    key=lambda r: (float(r["phi_target"]), int(r["net_idx"])))
    return base, merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--report-missing", action="store_true",
                    help="print the missing job indices (colon-separated) "
                         "instead of merging")
    args = ap.parse_args()
    if args.report_missing:
        merge(args.inputs, report_missing=True)
        return
    if args.out is None:
        raise SystemExit("--out is required")
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; not overwriting")
    try:
        params, rows = merge(args.inputs)
    except ValueError as err:
        raise SystemExit(f"not merged: {err}") from err
    write_h5(args.out, rows, params)
    print(f"wrote {args.out} ({len(rows)} realisations)")


if __name__ == "__main__":
    main()
