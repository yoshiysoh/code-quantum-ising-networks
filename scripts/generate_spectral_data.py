"""Level statistics and atypical eigenstates of the largest sector (Fig. 2
and Extended Data Fig. 2).

For every N, model and flippability phi = k/N, per network
(spectra.level_statistics_worker, spectra.eigenstate_worker):

  levels    <r> of the whole sector and of its largest subspace, with
            the subspace sizes;
  atypical  the number of atypical (low-S2) eigenstates, split by whether
            the largest subspace hosts them.

--extra N instead adds N independent networks at phi = 1 (Fig. 2 e, h).

Writes spectral_data.h5, or with --extra spectral_data_extra.h5, to
data/<date of the run>/ (or --out-dir; pass the same one to resume --extra).

Run:
    python3 -u generate_spectral_data.py
    python3 -u generate_spectral_data.py --extra 200
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from qising_networks import run_dir, spectra
from qising_networks.h5io import load_arrays, save_arrays

MASTER_SEED = 42
SALT_LEVELS, SALT_ATYPICAL = 91, 55
SALT_LEVELS_EXTRA, SALT_ATYPICAL_EXTRA = 9100, 5500


def _spawn(num_spins, model, salt, count):
    return np.random.SeedSequence(
        [MASTER_SEED, num_spins, 0 if model == "ferro" else 1, salt]
    ).spawn(count)


def _sweep_jobs(n, model, salt, n_nets):
    seeds = _spawn(n, model, salt, n_nets * (n + 1))
    return [
        (net, n, model, k, seeds[k * n_nets + net])
        for k in range(n + 1)
        for net in range(n_nets)
    ]


def levels_data(sizes, models, n_nets, workers):
    """<r> whole sector vs <r> largest subspace, per phi."""
    out = {}
    for n in sizes:
        for model in models:
            jobs = _sweep_jobs(n, model, SALT_LEVELS, n_nets)
            with mp.Pool(workers) as pool:
                rows = [
                    r
                    for r in pool.map(spectra.level_statistics_worker, jobs)
                    if r is not None
                ]
            a = np.array(rows)
            # columns: phi, r_whole, r_giant, sector_dim, giant_dim, n_comp,
            #          giant_frac
            out[f"levels_{n}_{model}"] = a
            print(f"  levels N={n} {model}: {len(a)} rows")
    return out


def atypical_data(sizes, models, n_nets, workers):
    """n_atyp decomposed by host subspace, per phi: mean and SD over
    networks of the total, fragment-hosted and largest-subspace-hosted
    counts."""
    out = {}
    for n in sizes:
        for model in models:
            jobs = _sweep_jobs(n, model, SALT_ATYPICAL, n_nets)
            with mp.Pool(workers) as pool:
                res = [
                    r
                    for r in pool.map(spectra.eigenstate_worker, jobs)
                    if r is not None
                ]
            d = {kk: np.concatenate([o[kk] for o in res]) for kk in res[0]}
            nt = d["atyp"].astype(bool)
            in_giant = d["giant_wt"] > 0.5
            levels = np.unique(np.round(d["phi"], 6))
            tot, frag, gia, gfrac = [], [], [], []
            sd_t, sd_f, sd_g = [], [], []
            for lv in levels:
                sel = np.isclose(d["phi"], lv)
                per_t, per_g = [], []
                for net in np.unique(d["net"][sel]):
                    m = sel & (d["net"] == net)
                    per_t.append(nt[m].sum())
                    per_g.append((nt[m] & in_giant[m]).sum())
                per_t = np.array(per_t, dtype=float)
                per_g = np.array(per_g, dtype=float)
                tot.append(per_t.mean())
                gia.append(per_g.mean())
                frag.append((per_t - per_g).mean())
                sd_t.append(per_t.std())
                sd_g.append(per_g.std())
                sd_f.append((per_t - per_g).std())
                gfrac.append(d["giant_frac"][sel].mean())
            out[f"atypical_{n}_{model}"] = np.array(
                [levels, tot, frag, gia, gfrac]
            )
            out[f"atypical_sd_{n}_{model}"] = np.array([sd_t, sd_f, sd_g])
            print(f"  atypical N={n} {model}: {len(levels)} phi levels")
    return out


def extra_phi1(sizes, models, n_extra, workers, out_dir):
    """n_extra new networks at phi = 1, checkpointed after every (N, model);
    rerun with the same arguments to resume."""
    out_path = out_dir / "spectral_data_extra.h5"
    out: dict[str, np.ndarray] = {}
    if out_path.exists():
        out = load_arrays(out_path)
        print(f"resuming {out_path} ({len(out)} arrays already present)")
    for n in sizes:
        for model in models:
            key02, key03 = (
                f"levels_extra_{n}_{model}",
                f"atypical_extra_{n}_{model}",
            )
            if key02 in out and key03 in out:
                print(f"N={n} {model}: already done, skipping")
                continue
            t0 = time.time()
            seeds02 = _spawn(n, model, SALT_LEVELS_EXTRA, n_extra)
            seeds03 = _spawn(n, model, SALT_ATYPICAL_EXTRA, n_extra)
            with mp.Pool(workers) as pool:
                rows02 = [
                    r
                    for r in pool.map(
                        spectra.level_statistics_worker,
                        [
                            (net, n, model, n, seeds02[net])
                            for net in range(n_extra)
                        ],
                    )
                    if r is not None
                ]
                rows03 = [
                    r
                    for r in pool.map(
                        spectra.eigenstate_worker,
                        [
                            (net, n, model, n, seeds03[net])
                            for net in range(n_extra)
                        ],
                    )
                    if r is not None
                ]
            out[key02] = np.array(rows02)
            # per-eigenstate rows (net, atypical, hosted by the largest
            # subspace), so the analysis can re-reduce them
            if rows03:
                d = {
                    kk: np.concatenate([o[kk] for o in rows03])
                    for kk in rows03[0]
                }
                out[key03] = np.array(
                    [
                        d["net"],
                        d["atyp"].astype(float),
                        (d["giant_wt"] > 0.5).astype(float),
                    ]
                )
            else:
                out[key03] = np.zeros((3, 0))
            save_arrays(out_path, out)
            n_net03 = len(np.unique(out[key03][0])) if rows03 else 0
            print(
                f"N={n} {model}: {len(rows02)} level-statistics nets, {n_net03} eigenstate "
                f"nets, {time.time() - t0:.1f}s -- checkpointed"
            )
    print(f"done: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=list(range(8, 16)))
    ap.add_argument("--model", nargs="+", default=["ferro", "pmJ"])
    ap.add_argument("--nets", type=int, default=100)
    ap.add_argument(
        "--extra",
        type=int,
        default=None,
        metavar="N_EXTRA",
        help="instead of the sweep, add N_EXTRA networks at phi = 1",
    )
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: data/<today's date>/",
    )
    args = ap.parse_args()
    workers = args.workers or int(
        os.environ.get("NUM_WORKERS", mp.cpu_count())
    )
    out_dir = args.out_dir or run_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.extra is not None:
        extra_phi1(args.sizes, args.model, args.extra, workers, out_dir)
        return
    print("level statistics by subspace")
    data = levels_data(args.sizes, args.model, args.nets, workers)
    print("atypical eigenstates by host subspace")
    data.update(atypical_data(args.sizes, args.model, args.nets, workers))
    out = out_dir / "spectral_data.h5"
    save_arrays(out, data)
    print(f"wrote {out} ({len(data)} arrays)")


if __name__ == "__main__":
    main()
