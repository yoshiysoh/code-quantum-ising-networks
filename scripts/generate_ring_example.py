"""The ring examples of Fig. 3 and Extended Data Fig. 3: a ferromagnetic
ring with a uniform field h = 2J, and its ±J gauge (Mattis) transform.

Every sector of the ring is scanned (examples.scan_ring_sectors); the scar
column is the Néel state and the other columns come from its sector. A
class with no state there is stored as missing. The ±J ring has
J'_ij = tau_i tau_j and h'_i = tau_i h for a random tau; --verify-against
checks that its scar revives as the ferromagnetic one does.

Writes ring_N<N>_<model>_h<h>[_seed<s>].h5 to data/<date of the run>/
(or --out-dir).

Run:
    python3 -u generate_ring_example.py --N 12 --h 2.0
    python3 -u generate_ring_example.py --N 12 --h 2.0 --model pmJ --seed 0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

from qising_networks import examples as ex
from qising_networks import run_dir
from qising_networks import scar_metrics as sm
from qising_networks.h5io import write_provenance
from qising_networks.model import GAMMA, neel_index, ring
from qising_networks.scar_metrics import DISPLAY_TIME_MAX, evolve_for_display

log = logging.getLogger("ring")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=ex.DEFAULT_N)
    ap.add_argument(
        "--h",
        type=float,
        default=2.0,
        help="uniform longitudinal field of the ferromagnetic "
        "ring, in units of J",
    )
    ap.add_argument("--model", choices=("ferro", "pmJ"), default="ferro")
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for the random gauge tau (pmJ only)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: data/<today's date>/",
    )
    ap.add_argument(
        "--verify-against",
        type=Path,
        default=None,
        help="ferromagnetic-ring HDF5 whose scar fid_rev the "
        "pmJ scar must reproduce",
    )
    args = ap.parse_args()
    args.out_dir = args.out_dir or run_dir()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.model == "pmJ" and args.seed is None:
        ap.error("--model pmJ needs --seed")

    sm.self_test()
    if args.model == "ferro":
        im, _ = ring(args.N)
        h = np.full(args.N, args.h)
        neel = neel_index(args.N)
    else:
        im, h, tau = ex.gauge_ring(
            args.N, args.h, np.random.default_rng(args.seed)
        )
        log.info("gauge tau: %s", tau.astype(int).tolist())
        neel = ex.gauge_index(neel_index(args.N), tau)
    degree, per_sector, candidates = ex.scan_ring_sectors(im, h, args.N)

    picks = {}
    picks[sm.SCAR] = ex.pick_scar_neel(
        args.N, per_sector, neel
    ) or ex.pick_scar(args.N, per_sector, candidates)
    scar_sector = picks[sm.SCAR]["sector"] if picks[sm.SCAR] else None
    picks[sm.LOCALIZED] = ex.pick_localized(candidates, sector=scar_sector)
    picks[sm.POINCARE] = ex.pick_poincare(candidates, sector=scar_sector)
    picks[sm.PLATEAU_SCAR] = ex.pick_plateau_scar(
        candidates, sector=scar_sector
    )
    picks[sm.THERMALIZED] = ex.pick_thermalized(candidates, picks[sm.SCAR])

    missing = [
        sm.CLASS_NAMES[c]
        for c in range(len(sm.CLASS_NAMES))
        if picks[c] is None
    ]
    if missing:
        log.warning(
            "%s ring N=%d: no representative for class(es) %s -- "
            "written as N/A",
            args.model,
            args.N,
            missing,
        )

    if args.verify_against is not None:
        with h5py.File(args.verify_against) as f:
            ferro_fid_rev = float(f["class_2"].attrs["fid_rev"])
        fid_rev = per_sector[picks[sm.SCAR]["sector"]]["m"].fid_rev[
            picks[sm.SCAR]["pos"]
        ]
        log.info(
            "gauge-mapping check: ferro scar fid_rev %.9f vs %s %.9f "
            "(diff %.2e)",
            ferro_fid_rev,
            args.model,
            fid_rev,
            abs(ferro_fid_rev - fid_rev),
        )
        assert np.isclose(ferro_fid_rev, fid_rev, atol=1e-6), (
            "gauge-mapped scar does not reproduce the ferromagnetic one"
        )

    if args.model == "ferro":
        out = args.out_dir / f"ring_N{args.N}_ferro_h{args.h:.1f}.h5"
    else:
        out = args.out_dir / (
            f"ring_N{args.N}_pmJ_h{args.h:.1f}_seed{args.seed}.h5"
        )
    with h5py.File(out, "w") as f:
        write_provenance(f)
        p = f.create_group("params")
        if args.model == "ferro":
            p.attrs.update(
                num_spins=args.N,
                model="ring",
                h_uniform=args.h,
                Gamma=GAMMA,
                display_time_max=DISPLAY_TIME_MAX,
                classify_time_max=ex.CLASSIFY_TIME_MAX,
            )
        else:
            p.attrs.update(
                num_spins=args.N,
                model="pmJ_ring_gauge",
                h_uniform=args.h,
                Gamma=GAMMA,
                seed=args.seed,
                display_time_max=DISPLAY_TIME_MAX,
                classify_time_max=ex.CLASSIFY_TIME_MAX,
            )
        p.attrs["class_names"] = np.array(
            sm.CLASS_NAMES, dtype=h5py.string_dtype()
        )
        if args.model == "pmJ":
            p.create_dataset("tau", data=tau)
        f.create_dataset("im", data=im)
        f.create_dataset("h", data=h)
        f.create_dataset("degree", data=degree)

        for c in range(len(sm.CLASS_NAMES)):
            grp = f.create_group(f"class_{c}")
            grp.attrs["name"] = sm.CLASS_NAMES[c]
            pick = picks[c]
            grp.attrs["valid"] = pick is not None
            if pick is None:
                log.info("  %-15s N/A", sm.CLASS_NAMES[c])
                continue
            sec = per_sector[pick["sector"]]
            pos = pick["pos"]
            alpha = pick["alpha"]
            time_list, z_t, fid_t, occ = evolve_for_display(
                sec["H"], sec["sec_idx"], args.N, [alpha], DISPLAY_TIME_MAX
            )
            grp.attrs["sector_rank"] = sec["rank"]
            grp.attrs["alpha"] = alpha
            grp.attrs["comp_size"] = int(sec["m"].comp_size[pos])
            grp.attrs["fid_min"] = float(sec["m"].fid_min[pos])
            grp.attrs["fid_rev"] = float(sec["m"].fid_rev[pos])
            grp.attrs["n_eff"] = float(sec["m"].n_eff[pos])
            grp.create_dataset("sec_idx", data=sec["sec_idx"])
            grp.create_dataset(
                "adjacency", data=sec["A"].astype(np.int8), compression="gzip"
            )
            grp.create_dataset("comp_labels", data=sec["comp_labels"])
            grp.create_dataset("time", data=time_list)
            grp.create_dataset("z_of_t", data=z_t[0], compression="gzip")
            grp.create_dataset(
                "fidelity_of_t", data=fid_t[0], compression="gzip"
            )
            grp.create_dataset("occupation", data=occ[0], compression="gzip")
            log.info(
                "  %-15s sector rank %d  state %5d  fid_min %.3f  "
                "fid_rev %.3f  n_eff %.2f  comp %d",
                sm.CLASS_NAMES[c],
                sec["rank"],
                alpha,
                sec["m"].fid_min[pos],
                sec["m"].fid_rev[pos],
                sec["m"].n_eff[pos],
                sec["m"].comp_size[pos],
            )
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
