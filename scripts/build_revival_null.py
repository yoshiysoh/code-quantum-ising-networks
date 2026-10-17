"""Size-matched null for the fidelity revival (Extended Data Fig. 8).

For each subspace dimension D, F_rev of a random basis state under GOE
matrices rescaled to the median spectral extent of real subspaces of that
dimension. scar_metrics uses the table to judge a revival against what its
subspace gives by chance. With --rule early-dip the revival is the
earlier rule that Extended Data Fig. 8 compares with: the maximum after the
deepest dip in Jt <= scar_metrics.RELAX_TIME.

Writes revival_null.h5 (or revival_null_early_dip.h5) to
data/<date of the run>/ (or --out-dir).

Run:
    python3 -u build_revival_null.py --N 13 --model ferro --nets 12 --reps 200
    python3 -u build_revival_null.py --N 13 --model ferro --nets 12 --reps 200 \
        --rule early-dip
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from qising_networks import run_dir
from qising_networks import scar_metrics as sm
from qising_networks.h5io import save_arrays
from qising_networks.model import (
    GAMMA,
    build_network,
    degree_to_h_on_off,
    h_on_off_to_h,
    largest_sector,
)
from qising_networks.scar_metrics import make_time_grid

TIME_MAX = sm.DEFAULT_TIME_MAX
DIMS = np.unique(np.round(np.geomspace(2, 4096, 40)).astype(int))
PERCENTILES = (50, 90, 99)

log = logging.getLogger("null")


def measure_real_extents(num_spins, model, n_nets, seed0=0):
    """(dimension, spectral extent) of every connected subspace we can see."""
    out = []
    for net in range(n_nets):
        rng = np.random.default_rng(seed0 + net)
        im, degree = build_network(num_spins, model, rng)
        h_on, h_off = degree_to_h_on_off(degree)
        for k in range(num_spins + 1):
            h = h_on_off_to_h(h_on, h_off, k, rng)
            sec_idx, H = largest_sector(im, h, GAMMA)
            if sec_idx is None or len(sec_idx) < 2:
                continue
            A = sm.adjacency_from_H(H, GAMMA)
            ss = sm.states_from_indices(sec_idx, num_spins)
            split = sm.split_components(A, ss)
            for comp in np.unique(split.labels):
                member = np.flatnonzero(split.labels == comp)
                if len(member) < 2:
                    continue
                ev = np.linalg.eigvalsh(H[np.ix_(member, member)])
                out.append((len(member), float(ev[-1] - ev[0])))
    return np.array(out)


def extent_model(real):
    """Median spectral extent as a function of dimension, log-interpolated."""
    d, w = real[:, 0], real[:, 1]
    dims = np.unique(d)
    med = np.array([np.median(w[d == x]) for x in dims])
    keep = med > 0
    return dims[keep], med[keep]


def null_at(dim, extent, n_rep, rng, after_dip=False):
    """F_rev of a random basis state under a GOE of this dim and bandwidth."""
    vals = np.empty(n_rep)
    for i in range(n_rep):
        A = rng.normal(size=(dim, dim))
        A = (A + A.T) / 2.0
        ev, U = np.linalg.eigh(A)
        # rescale the spectrum to the real subspaces' extent at this size
        span = ev[-1] - ev[0]
        if span > 0:
            ev = ev * (extent / span)
        t, _w, _o = make_time_grid(ev, TIME_MAX)
        a = int(rng.integers(dim))
        amp = U @ (np.exp(-1j * np.outer(t, ev)) * U[a]).T
        F = np.abs(amp[a]) ** 2
        vals[i] = sm.revival(F, t, after_dip=after_dip)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=13)
    ap.add_argument("--model", default="ferro")
    ap.add_argument("--nets", type=int, default=12)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument(
        "--rule",
        choices=("relax", "early-dip"),
        default="relax",
        help="revival rule (see the module docstring)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: data/<today's date>/",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    log.info(
        "measuring real subspace bandwidths (N=%d, %d networks)",
        args.N,
        args.nets,
    )
    real = measure_real_extents(args.N, args.model, args.nets)
    dims_r, ext_r = extent_model(real)
    log.info(
        "  %d subspaces, dimensions %d..%d",
        len(real),
        int(real[:, 0].min()),
        int(real[:, 0].max()),
    )

    rng = np.random.default_rng(0)
    rows = []
    for D in DIMS:
        extent = float(np.interp(np.log(D), np.log(dims_r), ext_r))
        reps = max(20, int(args.reps * min(1.0, 64 / D)))
        v = null_at(
            int(D), extent, reps, rng, after_dip=args.rule == "early-dip"
        )
        rows.append([D, extent, reps, *np.percentile(v, PERCENTILES)])
        log.info(
            "  D=%5d extent=%.3f reps=%3d  p50=%.4f p90=%.4f p99=%.4f",
            D,
            extent,
            reps,
            *np.percentile(v, PERCENTILES),
        )

    name = (
        "revival_null.h5"
        if args.rule == "relax"
        else "revival_null_early_dip.h5"
    )
    out = (args.out_dir or run_dir()) / name
    save_arrays(
        out,
        {
            "table": np.array(rows),
            "percentiles": np.array(PERCENTILES),
            "real_dims": dims_r,
            "real_extent": ext_r,
        },
        attrs={
            "time_max": TIME_MAX,
            "source_N": args.N,
            "source_model": args.model,
            "rule": args.rule,
        },
    )
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
