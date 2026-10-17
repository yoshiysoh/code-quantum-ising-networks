"""Rank candidate networks for the examples of Fig. 3 and Extended Data
Fig. 3.

Scans networks start..stop-1 of generate_examples.py's candidate pool and
logs how good an illustration each qualifying network's scar is; the chosen
network is then regenerated with `generate_examples.py --fixed-net <net>`.

  --rank pxp     scar states scored by how much F(t) looks like the PXP
                 Néel scar (examples.pxp_shape). The example is the network
                 with a plateau-scar state ("plateau True") whose scar has
                 no minor peaks and the largest "zero" fraction: network
                 641 of 0-719 (ferro).
  --rank purity  the scar pick scored by the spectral purity of F(t)
                 (examples.spectral_purity). The example is the network of
                 highest purity among those with a plateau-scar state
                 ("plateau state True"): network 24 of 0-59 (±J).

Run (in blocks of networks A..B-1):
    python3 -u select_example.py --rank pxp --model ferro --start A \
        --stop B --magnetization 0 --recurrence-spins 2
    python3 -u select_example.py --rank purity --model pmJ --start A \
        --stop B --magnetization 0 --recurrence-spins 2
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from qising_networks import examples as ex
from qising_networks import scar_metrics as sm
from qising_networks.scar_metrics import DISPLAY_TIME_MAX, evolve_for_display

log = logging.getLogger("select")


def rank_pxp(args):
    for net in range(args.start, args.stop):
        result = ex.evaluate_network(
            args.N, args.model, args.phi, args.seed0, net
        )
        if result is None:
            continue
        _score, hit = result
        m, cls = hit["metrics"], hit["cls"]
        allowed = ex.magnetization_mask(hit["sec_idx"], args.N)[m.state_index]
        has_plateau = bool((cls[allowed] == sm.PLATEAU_SCAR).any())
        pool = np.flatnonzero((cls == sm.SCAR) & allowed)
        alphas = m.state_index[pool]
        _t, _z, fid, _o = evolve_for_display(
            hit["H"], hit["sec_idx"], args.N, alphas, DISPLAY_TIME_MAX
        )
        for j, a in enumerate(alphas):
            zero, major, minor = ex.pxp_shape(fid[j])
            if major >= 5 and zero >= 0.25:
                log.info(
                    "cand net %3d state %4d: zero %.2f  major %2d  "
                    "minor %2d  F_rev %.3f  plateau %s",
                    net,
                    a,
                    zero,
                    major,
                    minor,
                    m.fid_rev[pool[j]],
                    has_plateau,
                )
        log.info("net %3d qualifies: %d scar states scored", net, len(pool))


def rank_purity(args):
    rows = []
    for net in range(args.start, args.stop):
        result = ex.evaluate_network(
            args.N, args.model, args.phi, args.seed0, net
        )
        if result is None:
            continue
        _score, hit = result
        m, H, sec_idx = hit["metrics"], hit["H"], hit["sec_idx"]
        alpha = int(m.state_index[hit["scar_idx"]])
        _t, _z, fid_t, _occ = evolve_for_display(
            H, sec_idx, args.N, [alpha], DISPLAY_TIME_MAX
        )
        purity = ex.spectral_purity(fid_t[0])
        quiet = ex.quiet_fraction(fid_t[0])
        peaks = ex.fidelity_revival_peaks(fid_t[0])
        fid_rev = float(m.fid_rev[hit["scar_idx"]])
        has_plateau = hit["plateau_idx"] is not None
        allowed = ex.magnetization_mask(sec_idx, args.N)[m.state_index]
        any_plateau = bool((hit["cls"][allowed] == sm.PLATEAU_SCAR).any())
        rows.append(
            (
                net,
                purity,
                fid_rev,
                has_plateau,
                any_plateau,
                quiet,
                peaks,
                alpha,
            )
        )
        log.info(
            "net %3d: purity %.3f  fid_rev %.3f  plateau finalist %s  "
            "plateau state %s  quiet %.2f  peaks %d  scar state %d",
            *rows[-1],
        )
    rows.sort(key=lambda r: -r[1])
    log.info("=== ranked by purity (descending) ===")
    for row in rows:
        log.info(
            "net %3d: purity %.3f  fid_rev %.3f  plateau finalist %s  "
            "plateau state %s  quiet %.2f  peaks %d  scar state %d",
            *row,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", choices=("pxp", "purity"), required=True)
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--model", default="ferro")
    ap.add_argument("--phi", type=float, default=1.0)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=50)
    ap.add_argument(
        "--magnetization",
        type=int,
        default=None,
        help="as generate_examples.py --magnetization",
    )
    ap.add_argument(
        "--recurrence-spins",
        type=int,
        default=None,
        help="as generate_examples.py --recurrence-spins",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ex.MAGNETIZATION = args.magnetization
    ex.RECURRENCE_SPINS = args.recurrence_spins
    (rank_pxp if args.rank == "pxp" else rank_purity)(args)


if __name__ == "__main__":
    main()
