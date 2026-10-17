"""Check the effective Hamiltonian against the full one (Extended Data
Fig. 1).

The effective Hamiltonian keeps only the spin flips that leave the
classical energy unchanged, so it is block-diagonal in the degenerate
sectors. For random networks at flippability phi, both are diagonalized
and the rank-paired spectra compared:

    eigval_error = sum_i |E_exact_i - E_approx_i|
    eigvec_error = sum_i (1 - <psi_exact_i| P_i |psi_exact_i>)

with P_i the projector onto the Fock states of the classical energy of
the i-th approximate eigenstate.

Writes hamapprox_N<N>_<model>_gamma<Gamma>.h5 to data/<date of the run>/
(or --out-dir).

Run (one model and Gamma per call):
    python3 -u verify_hamiltonian_approximation.py --N 12 --model ferro \
        --gamma 0.3 --nsamples 10000 --seed0 42000 --skip-self-test
"""

from __future__ import annotations

import argparse
import gc
import itertools
import logging
import multiprocessing as mp
import os
from pathlib import Path

import h5py
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from qising_networks import run_dir
from qising_networks.h5io import write_provenance
from qising_networks.model import (
    build_network,
    degree_to_h_on_off,
    flippability,
    h_on_off_to_h,
    interaction_to_hamiltonian,
)

MASTER_SEED: int = 42
DEFAULT_PHI: float = 1.0
CHECKPOINT_EVERY: int = 25

log = logging.getLogger("hamapprox")


def build_approx_and_components(
    H_exact: np.ndarray, num_spins: int
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """H_approx: H_exact without the single-spin flips that change the
    classical energy, and the connected-component labels of the retained
    flips (one block of H_approx each).
    """
    dim = H_exact.shape[0]
    diag = np.diag(H_exact)
    idx = np.arange(dim)
    flip_masks = 1 << np.arange(num_spins - 1, -1, -1)

    H_approx = H_exact.copy()
    rows_list, cols_list = [], []
    for m in flip_masks:
        partner = idx ^ m
        floppy = diag == diag[partner]
        H_approx[idx[~floppy], partner[~floppy]] = 0.0
        keep = floppy & (idx < partner)  # one entry per undirected edge
        rows_list.append(idx[keep])
        cols_list.append(partner[keep])

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    adj = sp.coo_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)), shape=(dim, dim)
    )
    n_components, labels = connected_components(adj, directed=False)
    return H_approx, n_components, labels, diag.copy()


def diagonalize_blockdiag(
    H_approx: np.ndarray,
    n_components: int,
    labels: np.ndarray,
    diag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Ascending eigenvalues of the block-diagonal H_approx, one block at a
    time, with the classical energy of each eigenvalue's block.
    """
    dim = H_approx.shape[0]
    eigvals = np.empty(dim, dtype=np.float64)
    shell_energy = np.empty(dim, dtype=np.float64)

    write_pos = 0
    for c in range(n_components):
        members = np.nonzero(labels == c)[0]
        n = len(members)
        e = diag[members[0]]  # shared by every member, by construction
        if n == 1:
            eigvals[write_pos] = H_approx[members[0], members[0]]
        else:
            block = H_approx[np.ix_(members, members)]  # a copy
            w = sla.eigvalsh(block, overwrite_a=True, check_finite=False)
            eigvals[write_pos : write_pos + n] = w
        shell_energy[write_pos : write_pos + n] = e
        write_pos += n

    order = np.argsort(eigvals)
    return eigvals[order], shell_energy[order]


def energy_level_fidelity_error(
    eigvecs_exact: np.ndarray,
    shell_energy_approx: np.ndarray,
    diag_exact: np.ndarray,
) -> float:
    """sum_i (1 - <psi_exact_i| P_i |psi_exact_i>), P_i the projector onto
    all Fock states with the classical energy of the i-th approximate
    eigenstate. Unlike the overlap |<psi_exact_i|psi_approx_i>|^2 it does
    not depend on the basis chosen within degenerate levels, so it vanishes
    as Gamma -> 0.
    """
    error = 0.0
    for level in np.unique(shell_energy_approx):
        cols = np.nonzero(shell_energy_approx == level)[0]
        rows = np.nonzero(diag_exact == level)[0]
        weight = np.sum(eigvecs_exact[np.ix_(rows, cols)] ** 2, axis=0)
        error += float(np.sum(1.0 - weight))
    return error


def compare_exact_vs_approx(
    H_exact: np.ndarray, num_spins: int
) -> tuple[float, float]:
    """(eigval_error, eigvec_error), rank-paired between the ascending exact
    and approximate spectra. H_exact is diagonalized in place; pass it as an
    unnamed expression so that it can be freed.
    """
    H_approx, n_components, labels, diag = build_approx_and_components(
        H_exact, num_spins
    )

    eigvals_exact, eigvecs_exact = sla.eigh(
        H_exact, overwrite_a=True, check_finite=False
    )
    del H_exact
    gc.collect()

    eigvals_approx, shell_energy_approx = diagonalize_blockdiag(
        H_approx, n_components, labels, diag
    )
    del H_approx
    gc.collect()

    eigval_error = float(np.sum(np.abs(eigvals_exact - eigvals_approx)))
    eigvec_error = energy_level_fidelity_error(
        eigvecs_exact, shell_energy_approx, diag
    )
    return eigval_error, eigvec_error


def _worker(args: tuple) -> dict | None:
    sample_idx, num_spins, model, gamma, phi, seed = args
    try:
        rng = np.random.default_rng(seed)
        im, degree = build_network(num_spins, model, rng)
        h_on, h_off = degree_to_h_on_off(degree)
        k = round(phi * num_spins)
        h = h_on_off_to_h(h_on, h_off, k, rng)
        Gamma_vec = gamma * np.ones(num_spins)

        # Passed straight through so compare_exact_vs_approx can free it.
        eigval_error, eigvec_error = compare_exact_vs_approx(
            interaction_to_hamiltonian(im, Gamma_vec, h), num_spins
        )

        return {
            "sample_idx": int(sample_idx),
            "eigval_error": eigval_error,
            "eigvec_error": eigvec_error,
            "phi": flippability((im != 0).sum(axis=1), h),
            "mean_degree": float(degree.mean()),
        }
    except Exception:
        log.exception(
            "sample %d (N=%d, %s, Gamma=%s) failed",
            sample_idx,
            num_spins,
            model,
            gamma,
        )
        return None


def write_h5(path: Path, rows: list[dict], params: dict) -> None:
    tmp = path.with_suffix(".partial.h5")
    with h5py.File(tmp, "w") as f:
        write_provenance(f)
        p = f.create_group("params")
        for k, v in params.items():
            p.attrs[k] = v
        if not rows:
            tmp.replace(path)
            return
        cols = sorted(rows[0].keys())
        g = f.create_group("samples")
        for col in cols:
            g.create_dataset(
                col,
                data=np.array([r[col] for r in rows]),
                compression="gzip",
                compression_opts=4,
            )
    tmp.replace(path)


def run(
    num_spins: int,
    model: str,
    gamma: float,
    nsamples: int,
    phi: float,
    seed0: int,
    out_dir: Path,
    workers: int,
) -> Path:
    seeds = np.random.SeedSequence(
        [MASTER_SEED, num_spins, 0 if model == "ferro" else 1, seed0]
    ).spawn(nsamples)
    jobs = [
        (i, num_spins, model, gamma, phi, seeds[i]) for i in range(nsamples)
    ]

    out = out_dir / f"hamapprox_N{num_spins}_{model}_gamma{gamma:.2f}.h5"
    params = {
        "num_spins": num_spins,
        "model": model,
        "gamma": gamma,
        "nsamples": nsamples,
        "phi_target": phi,
        "seed0": seed0,
        "master_seed": MASTER_SEED,
    }

    rows: list[dict] = []
    log.info(
        "N=%d %s Gamma=%.2f: %d realisations on %d workers",
        num_spins,
        model,
        gamma,
        nsamples,
        workers,
    )
    with mp.Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, jobs, chunksize=1)):
            if r is not None:
                rows.append(r)
            if (i + 1) % CHECKPOINT_EVERY == 0:
                write_h5(out, rows, params)
                log.info("  %d/%d done (%d kept)", i + 1, nsamples, len(rows))
    write_h5(out, rows, params)
    log.info("wrote %s (%d realisations)", out, len(rows))
    return out


def self_test() -> None:
    """Brute-force cross-check of build_approx_and_components on a tiny
    system: for every pair of Fock states, H_approx should retain an
    off-diagonal element if and only if the two states are Hamming-distance
    1 AND share the same classical (diagonal) energy -- the
    delta[H(s), H(F_k s)] condition of the effective Hamiltonian (SI,
    Section I), checked here by an independent double loop rather than the vectorized bit-trick under
    test, and the retained-edge connected components must match
    H_approx's own nonzero block structure."""
    num_spins = 5
    rng = np.random.default_rng(0)
    im, degree = build_network(num_spins, "pmJ", rng)
    h_on, h_off = degree_to_h_on_off(degree)
    h = h_on_off_to_h(h_on, h_off, 3, rng)
    gamma = 0.37
    H_exact = interaction_to_hamiltonian(im, gamma * np.ones(num_spins), h)
    dim = H_exact.shape[0]
    diag = np.diag(H_exact)

    H_approx, n_components, labels, diag_returned = (
        build_approx_and_components(H_exact.copy(), num_spins)
    )
    assert np.array_equal(diag_returned, diag)

    expected = np.zeros((dim, dim))
    np.fill_diagonal(expected, diag)
    for s, sp_ in itertools.product(range(dim), repeat=2):
        if s == sp_:
            continue
        hamming = (s ^ sp_).bit_count()
        if hamming == 1 and diag[s] == diag[sp_]:
            expected[s, sp_] = -gamma

    assert np.allclose(H_approx, expected), "H_approx mismatch vs brute force"

    # Every retained edge must sit inside a single connected component.
    rows, cols = np.nonzero(np.triu(H_approx, 1))
    assert np.all(labels[rows] == labels[cols]), (
        "floppy edge crosses components"
    )

    # Diagonalizing block-wise must reproduce a plain dense eigh of H_approx.
    eigvals_block, shell_energy = diagonalize_blockdiag(
        H_approx, n_components, labels, diag
    )
    eigvals_dense = np.linalg.eigvalsh(H_approx)
    assert np.allclose(np.sort(eigvals_block), np.sort(eigvals_dense))
    # Every recorded shell energy must be a classical energy of the system.
    assert np.all(np.isin(shell_energy, diag))

    # Gamma = 0: H_exact and H_approx are identical, so both errors must be
    # exactly zero (no artefact from degenerate eigenbases).
    H0 = interaction_to_hamiltonian(im, np.zeros(num_spins), h)
    e_err0, v_err0 = compare_exact_vs_approx(H0, num_spins)
    assert e_err0 < 1e-9, f"Gamma=0 eigval_error should vanish, got {e_err0}"
    assert v_err0 < 1e-9, f"Gamma=0 eigvec_error should vanish, got {v_err0}"

    # Errors must be non-negative.
    e_err, v_err = compare_exact_vs_approx(H_exact.copy(), num_spins)
    assert e_err >= 0.0 and v_err >= -1e-9
    log.info("self_test passed (N=%d, %d components)", num_spins, n_components)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--model", choices=["ferro", "pmJ"], required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--nsamples", type=int, required=True)
    ap.add_argument("--phi", type=float, default=DEFAULT_PHI)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: data/<today's date>/",
    )
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--skip-self-test", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args.out_dir = args.out_dir or run_dir()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers or int(
        os.environ.get("NUM_WORKERS", mp.cpu_count())
    )

    if not args.skip_self_test:
        self_test()

    run(
        args.N,
        args.model,
        args.gamma,
        args.nsamples,
        args.phi,
        args.seed0,
        args.out_dir,
        workers,
    )


if __name__ == "__main__":
    main()
