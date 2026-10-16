"""Dynamical class of every Fock state in the largest sector (Figs. 2, 4, S2,
S4, S8).

For each network and flippability phi = k/N, measures min_t F(t), the
revival F_rev and the participation number of each state of the largest
sector (scar_metrics.measure_sector), with the size of its subspace.
Components larger than --max-component are sampled; each state carries a
weight that restores the component's share.

Writes classes_N<N>_<model>.h5 to data/<date of the run>/ (or --out-dir).

Run:
    python3 -u generate_class_dataset.py --N 8 9 10 11 12 13 --nets 200
    python3 -u generate_class_dataset.py --N 14 15 --nets 100
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from qising_networks import run_dir
from qising_networks import scar_metrics as sm
from qising_networks.class_fractions import (MAX_COMPONENT_DEFAULT,
                                             SAMPLE_SIZE_DEFAULT, TIME_MAX,
                                             write_h5)
from qising_networks.model import GAMMA, P_GRAPH, build_network, degree_to_h_on_off, flipability, h_on_off_to_h, largest_sector

MASTER_SEED: int = 42
CHECKPOINT_EVERY: int = 25


log = logging.getLogger("classdata")

def measure_realisation(im, degree, h, num_spins, net_idx, phi_target,
                        max_component, sample_size, seed):
    sec_idx, H = largest_sector(im, h, GAMMA)
    if sec_idx is None or len(sec_idx) < 2:
        return None
    rng = np.random.default_rng(seed)
    m = sm.measure_sector(sec_idx, H, GAMMA, num_spins, time_max=TIME_MAX,
                          max_component=max_component,
                          sample_size=sample_size, sample_rng=rng)

    cols = np.column_stack([
        m.fid_min, m.fid_rev, m.n_eff, m.n_inv(1.0).astype(float),
        m.comp_size.astype(float), m.support.astype(float), m.weight(),
    ]).astype(np.float32)

    return dict(
        net_idx=int(net_idx),
        phi_target=float(phi_target),
        phi=flipability(degree, h),
        mean_degree=float(degree.mean()),
        sector_dim=int(len(sec_idx)),
        giant_size=int(m.comp_size.max()),
        n_measured=int(len(m.state_index)),
        states=cols,
    )


def _worker(args):
    (net_idx, num_spins, model, phi_level, net_seed,
     max_component, sample_size) = args
    try:
        rng = np.random.default_rng(net_seed)
        im, degree = build_network(num_spins, model, rng)
        h_on, h_off = degree_to_h_on_off(degree)
        k = int(round(phi_level * num_spins))
        h = h_on_off_to_h(h_on, h_off, k, rng)
        return measure_realisation(im, degree, h, num_spins, net_idx,
                                   phi_level, max_component, sample_size,
                                   net_seed)
    except Exception:                                        # noqa: BLE001
        log.exception("net %d (N=%d, %s, phi=%s) failed", net_idx, num_spins,
                      model, phi_level)
        return None


def run(num_spins: int, model: str, n_nets: int, out_dir: Path,
        max_component: int, sample_size: int, workers: int,
        phi_levels: list[float] | None = None,
        shard: tuple[int, int] | None = None,
        job_range: tuple[int, int] | None = None) -> Path:
    levels = (phi_levels if phi_levels
              else [k / num_spins for k in range(num_spins + 1)])
    seeds = np.random.SeedSequence(
        [MASTER_SEED, num_spins, 0 if model == "ferro" else 1, 7]
    ).spawn(n_nets * len(levels))

    jobs, s = [], 0
    for lvl in levels:
        for net in range(n_nets):
            jobs.append((net, num_spins, model, lvl, seeds[s],
                         max_component, sample_size))
            s += 1
    # --job-range and --shard (every k-th realisation) select from the full
    # list after the seeds are spawned, so merge_class_shards.py can
    # reassemble the unsharded dataset.
    tag = ""
    if job_range is not None:
        a, b = job_range
        jobs = jobs[a:b]
        tag += f"_jobs{a}to{b}"
    if shard is not None:
        i, k = shard
        jobs = jobs[i::k]
        tag += f"_shard{i}of{k}"

    suffix = ("_partial" if phi_levels else "") + tag
    out = out_dir / f"classes_N{num_spins}_{model}{suffix}.h5"
    params = dict(num_spins=num_spins, model=model, n_nets=n_nets, Gamma=GAMMA,
                  p_graph=P_GRAPH, master_seed=MASTER_SEED,
                  time_max=TIME_MAX, max_component=max_component,
                  sample_size=sample_size, phi_levels=np.array(levels),
                  drop_cut=sm.DROP_CUT, rev_cut=sm.REV_CUT)
    if tag:
        params.update(selection=tag, n_jobs_total=s)

    rows: list[dict] = []
    log.info("N=%d %s: %d realisations on %d workers",
             num_spins, model, len(jobs), workers)
    with mp.Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, jobs, chunksize=1)):
            if r is not None:
                rows.append(r)
            if (i + 1) % CHECKPOINT_EVERY == 0:
                write_h5(out, rows, params)
                log.info("  %d/%d done (%d kept)", i + 1, len(jobs), len(rows))
    write_h5(out, rows, params)
    log.info("wrote %s (%d realisations)", out, len(rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", required=True)
    ap.add_argument("--model", nargs="+", default=["ferro", "pmJ"])
    ap.add_argument("--nets", type=int, default=200)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: data/<today's date>/")
    ap.add_argument("--max-component", type=int, default=MAX_COMPONENT_DEFAULT)
    ap.add_argument("--sample-size", type=int, default=SAMPLE_SIZE_DEFAULT)
    ap.add_argument("--phi-levels", type=float, nargs="+", default=None)
    ap.add_argument("--shard", type=int, nargs=2, default=None,
                    metavar=("I", "K"),
                    help="run only realisations I, I+K, I+2K, ...")
    ap.add_argument("--job-range", type=int, nargs=2, default=None,
                    metavar=("START", "STOP"),
                    help="run only realisations START..STOP-1, before --shard")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir = args.out_dir or run_dir()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    workers = int(os.environ.get("NUM_WORKERS", mp.cpu_count()))

    # Positive control (the ring's scar) before any data is generated.
    sm.self_test()

    for n in args.N:
        for model in args.model:
            run(n, model, args.nets, args.out_dir, args.max_component,
                args.sample_size, workers, phi_levels=args.phi_levels,
                shard=tuple(args.shard) if args.shard else None,
                job_range=tuple(args.job_range) if args.job_range else None)


if __name__ == "__main__":
    main()
