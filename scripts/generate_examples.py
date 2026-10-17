"""The network examples of Fig. 3, Extended Data Fig. 3 and Supplementary
Movies 1-3.

Finds a network whose largest sector holds a Fock state of every class (or
takes --fixed-net), picks one initial state per class
(qising_networks.examples), and writes the network, field, sector, chosen
states, their F(t) and <Z_i(t)>, and their time-averaged occupations.

Writes examples_N<N>_<model>_phi<phi>[_M<m>].h5 to data/<date of the run>/
(or --out-dir).

Run:
    python3 -u generate_examples.py --N 12 --phi 1.0 --model ferro \
        --fixed-net 641 --magnetization 0 --recurrence-spins 2 --scar-pxp
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
from qising_networks.model import GAMMA, flippability
from qising_networks.scar_metrics import DISPLAY_TIME_MAX, evolve_for_display

log = logging.getLogger("examples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=ex.DEFAULT_N)
    ap.add_argument("--model", default="ferro")
    ap.add_argument("--phi", type=float, default=ex.DEFAULT_PHI)
    ap.add_argument("--max-nets", type=int, default=200)
    ap.add_argument(
        "--best-of",
        type=int,
        default=8,
        help="scan this many qualifying networks and "
        "keep the one with the clearest scar",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: data/<today's date>/",
    )
    ap.add_argument(
        "--seed0",
        type=int,
        default=1000,
        help="base seed; net i uses seed0+i, so a disjoint "
        "--seed0 lets parallel jobs explore disjoint "
        "network ranges",
    )
    ap.add_argument(
        "--magnetization",
        type=int,
        default=None,
        help="restrict every displayed initial Fock state to "
        "this total magnetization sum_i s_i (e.g. 0)",
    )
    ap.add_argument(
        "--recurrence-spins",
        type=int,
        default=None,
        help="recurrence column with exactly this many moving "
        "spins (networks without one do not qualify)",
    )
    ap.add_argument(
        "--scar-pxp",
        action="store_true",
        help="scar column = the scar state with the most "
        "PXP-like F(t) (see examples.SCAR_PXP)",
    )
    ap.add_argument(
        "--fixed-net",
        type=int,
        default=None,
        help="use this network instead of searching "
        "(e.g. one chosen with select_example.py)",
    )
    args = ap.parse_args()
    args.out_dir = args.out_dir or run_dir()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ex.SCAR_PXP = args.scar_pxp
    ex.MAGNETIZATION = args.magnetization
    ex.RECURRENCE_SPINS = args.recurrence_spins

    sm.self_test()
    if args.fixed_net is not None:
        result = ex.evaluate_network(
            args.N, args.model, args.phi, args.seed0, args.fixed_net
        )
        hit = result[1] if result is not None else None
    else:
        hit = ex.find_realisation(
            args.N,
            args.model,
            args.phi,
            args.max_nets,
            seed0=args.seed0,
            best_of=args.best_of,
        )
    if hit is None:
        raise RuntimeError(
            f"no realisation with all four required classes at N={args.N}, "
            f"phi={args.phi} within {args.max_nets} networks"
        )

    m, cls = hit["metrics"], hit["cls"]
    scar_idx = hit["scar_idx"]
    typ_idx = hit.get("typ_idx")
    plateau_idx = hit.get("plateau_idx")
    allowed = ex.magnetization_mask(hit["sec_idx"], args.N)[m.state_index]
    alphas = ex.choose_states(
        m,
        cls,
        scar_idx=scar_idx,
        typ_idx=typ_idx,
        plateau_idx=plateau_idx,
        allowed=allowed,
    )
    if ex.SCAR_PXP:
        pool = np.flatnonzero((cls == sm.SCAR) & allowed)
        cand = m.state_index[pool]
        _t, _z, f_c, _o = evolve_for_display(
            hit["H"], hit["sec_idx"], args.N, cand, DISPLAY_TIME_MAX
        )
        shp = np.array([ex.pxp_shape(f) for f in f_c])
        zero, major, minor = shp[:, 0], shp[:, 1], shp[:, 2]
        ok = np.flatnonzero(major >= ex.REV_MIN_PEAKS)
        if ok.size:
            ok = ok[minor[ok] == minor[ok].min()]
            # Symmetry-related states tie: the lowest Fock index wins.
            best = min(ok, key=lambda j: (-ex._score(zero[j]), cand[j]))
            alphas[sm.SCAR] = int(cand[best])
            log.info(
                "scar column: state %d, %.2f of the window at F < %.2f, "
                "%d major / %d minor revival peaks, F_rev %.3f "
                "(most PXP-like of %d scar states)",
                cand[best],
                zero[best],
                ex.PXP_ZERO_CUT,
                major[best],
                minor[best],
                m.fid_rev[pool[best]],
                pool.size,
            )
        else:
            log.warning(
                "no scar state with %d major revivals; keeping the "
                "spectral pick",
                ex.REV_MIN_PEAKS,
            )
    # positions of the chosen states in m's arrays; -1 where a class has no
    # candidate (masked to NaN below)
    where = np.array(
        [
            int(np.flatnonzero(m.state_index == a)[0]) if a >= 0 else -1
            for a in alphas
        ]
    )
    time_list, z_t, fid_t, occ = evolve_for_display(
        hit["H"], hit["sec_idx"], args.N, alphas, DISPLAY_TIME_MAX
    )

    A = sm.adjacency_from_H(hit["H"], GAMMA)
    ss = sm.states_from_indices(hit["sec_idx"], args.N)
    split = sm.split_components(A, ss)

    mag_tag = "" if ex.MAGNETIZATION is None else f"_M{ex.MAGNETIZATION}"
    out = args.out_dir / (
        f"examples_N{args.N}_{args.model}_phi{args.phi:.2f}{mag_tag}.h5"
    )
    with h5py.File(out, "w") as f:
        write_provenance(f)
        p = f.create_group("params")
        p.attrs.update(
            num_spins=args.N,
            model=args.model,
            phi_target=args.phi,
            phi=flippability(hit["degree"], hit["h"]),
            Gamma=GAMMA,
            net=hit["net"],
            pick="relative",
            display_time_max=DISPLAY_TIME_MAX,
            classify_time_max=ex.CLASSIFY_TIME_MAX,
        )
        p.attrs["class_names"] = np.array(
            sm.CLASS_NAMES, dtype=h5py.string_dtype()
        )
        f.create_dataset("im", data=hit["im"])
        f.create_dataset("h", data=hit["h"])
        f.create_dataset("degree", data=hit["degree"])
        f.create_dataset("sec_idx", data=hit["sec_idx"])
        f.create_dataset(
            "adjacency", data=A.astype(np.int8), compression="gzip"
        )
        f.create_dataset("comp_labels", data=split.labels)
        f.create_dataset("alphas", data=alphas)
        f.create_dataset("time", data=time_list)
        f.create_dataset("z_of_t", data=z_t, compression="gzip")
        f.create_dataset("fidelity_of_t", data=fid_t, compression="gzip")
        f.create_dataset("occupation", data=occ, compression="gzip")
        valid = where >= 0
        safe = np.where(valid, where, 0)

        def masked(arr):
            out_ = arr[safe].astype(float)
            out_[~valid] = np.nan
            return out_

        f.create_dataset("fid_min", data=masked(m.fid_min))
        f.create_dataset("fid_rev", data=masked(m.fid_rev))
        f.create_dataset("n_eff", data=masked(m.n_eff))
        f.create_dataset("comp_size", data=masked(m.comp_size))
        f.create_dataset(
            "class_counts",
            data=np.bincount(cls, minlength=len(sm.CLASS_NAMES)),
        )
    log.info("wrote %s", out)
    for c, a in enumerate(alphas):
        j = where[c]
        if j < 0:
            log.info(
                "  %-15s -- no candidate in this sector", sm.CLASS_NAMES[c]
            )
            continue
        log.info(
            "  %-15s state %5d  fid_min %.3f  fid_rev %.3f  "
            "n_eff %.2f  comp %d",
            sm.CLASS_NAMES[c],
            a,
            m.fid_min[j],
            m.fid_rev[j],
            m.n_eff[j],
            m.comp_size[j],
        )


if __name__ == "__main__":
    main()
