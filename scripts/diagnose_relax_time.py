"""Compare two definitions of the revival F_rev (Extended Data Fig. 8).

Classifies the same sampled Fock states with F_rev measured after the
deepest dip in the first quarter of the window (earlier rule) and after
Jt = 50 (scar_metrics.revival), and stores the fidelity trace of every
state whose class changes.

Writes relax_diag_N<N>_<model>.h5 to data/<date of the run>/ (or --out-dir).

Run: python3 -u diagnose_relax_time.py --nets 32
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from qising_networks import run_dir
from qising_networks import scar_metrics as sm
from qising_networks.class_fractions import (
    MAX_COMPONENT_DEFAULT,
    SAMPLE_SIZE_DEFAULT,
    TIME_MAX,
)
from qising_networks.h5io import save_arrays
from qising_networks.model import (
    GAMMA,
    build_network,
    degree_to_h_on_off,
    h_on_off_to_h,
    largest_sector,
)
from qising_networks.scar_metrics import make_time_grid

NUM_SPINS = 12
PHI_LEVELS = (0.25, 0.5, 0.75, 1.0)
RELAX_TIME = 50.0
NET_SEED0 = 10_000

log = logging.getLogger("relax")


def fidelity_trace(H, sec_idx, alpha, time_list):
    """F(t) of Fock state `alpha` (position in the sector) on `time_list`,
    evolved in its own connected component."""
    A = sm.adjacency_from_H(H, GAMMA)
    split = sm.split_components(A, sm.states_from_indices(sec_idx, NUM_SPINS))
    member = np.flatnonzero(split.labels == split.labels[alpha])
    ev, U = np.linalg.eigh(H[np.ix_(member, member)])
    a = int(np.flatnonzero(member == alpha)[0])
    amp = (U[a] ** 2) @ np.exp(-1j * np.outer(ev, time_list))
    return np.abs(amp) ** 2


def run_sector(args):
    model, net, phi = args
    rng = np.random.default_rng(NET_SEED0 + net)
    im, degree = build_network(NUM_SPINS, model, rng)
    h_on, h_off = degree_to_h_on_off(degree)
    h = h_on_off_to_h(h_on, h_off, round(phi * NUM_SPINS), rng)
    sec, H = largest_sector(im, h, GAMMA)
    if sec is None or len(sec) < 2:
        return None
    res = {}
    for key, rt in (("quarter", None), ("relax", RELAX_TIME)):
        m = sm.measure_sector(
            sec,
            H,
            GAMMA,
            NUM_SPINS,
            time_max=TIME_MAX,
            max_component=MAX_COMPONENT_DEFAULT,
            sample_size=SAMPLE_SIZE_DEFAULT,
            sample_rng=np.random.default_rng(net),
            relax_time=rt,
        )
        res[key] = m
    mq, mr = res["quarter"], res["relax"]
    assert np.array_equal(mq.state_index, mr.state_index)
    cq = sm.classify(
        mq.fid_min, mq.fid_rev, mq.n_eff, NUM_SPINS, comp_size=mq.comp_size
    )
    cr = sm.classify(
        mr.fid_min, mr.fid_rev, mr.n_eff, NUM_SPINS, comp_size=mr.comp_size
    )
    switched = np.flatnonzero(cq != cr)
    ev = np.linalg.eigvalsh(H)
    time_list, _w, _o = make_time_grid(np.array([ev[0], ev[-1]]), TIME_MAX)
    traces = [
        fidelity_trace(H, sec, int(mq.state_index[i]), time_list)
        for i in switched
    ]
    return {
        "model": model,
        "net": net,
        "phi": phi,
        "dim": len(sec),
        "n_states": len(cq),
        "counts_quarter": np.bincount(cq, minlength=sm.N_CLASSES),
        "counts_relax": np.bincount(cr, minlength=sm.N_CLASSES),
        "cls_quarter": cq[switched],
        "cls_relax": cr[switched],
        "comp_size": mq.comp_size[switched],
        "fid_min": mq.fid_min[switched],
        "frev_quarter": mq.fid_rev[switched],
        "frev_relax": mr.fid_rev[switched],
        "anom_quarter": sm.anomaly_ratio(mq.fid_rev, mq.comp_size)[switched],
        "anom_relax": sm.anomaly_ratio(mr.fid_rev, mr.comp_size)[switched],
        "time": time_list,
        "traces": traces,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", type=int, default=32)
    ap.add_argument("--workers", type=int, default=16)
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
    for model in ("ferro", "pmJ"):
        tasks = [(model, n, p) for n in range(args.nets) for p in PHI_LEVELS]
        with ProcessPoolExecutor(args.workers) as ex:
            rows = [r for r in ex.map(run_sector, tasks) if r is not None]
        flat = {}
        for k in (
            "cls_quarter",
            "cls_relax",
            "comp_size",
            "fid_min",
            "frev_quarter",
            "frev_relax",
            "anom_quarter",
            "anom_relax",
        ):
            flat[k] = np.concatenate([r[k] for r in rows])
        flat["sector"] = np.concatenate(
            [np.full(len(r["cls_quarter"]), i) for i, r in enumerate(rows)]
        )
        traces = [tr for r in rows for tr in r["traces"]]
        times = [r["time"] for r in rows for _ in r["traces"]]
        out = (
            args.out_dir or run_dir()
        ) / f"relax_diag_N{NUM_SPINS}_{model}.h5"
        save_arrays(
            out,
            dict(
                **flat,
                counts_quarter=np.array([r["counts_quarter"] for r in rows]),
                counts_relax=np.array([r["counts_relax"] for r in rows]),
                sector_meta=np.array(
                    [
                        (r["net"], r["phi"], r["dim"], r["n_states"])
                        for r in rows
                    ]
                ),
                traces=traces,
                times=times,
            ),
            attrs={"relax_time": RELAX_TIME, "time_max": TIME_MAX},
        )
        log.info(
            "%s: %d sectors, %d switched states -> %s",
            model,
            len(rows),
            len(flat["cls_quarter"]),
            out,
        )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
