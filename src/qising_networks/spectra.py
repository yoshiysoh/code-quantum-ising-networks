"""Eigenstate and level statistics of sector Hamiltonians (Fig. 2 and
Extended Data Figs. 5-8).

Also holds the two per-network workers behind the Fig. 2 data
(scripts/generate_spectral_data.py): level-spacing ratios of the largest
sector and of its largest subspace, and the participation entropy of every
eigenstate, subspace by subspace.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from . import model
from . import scar_metrics as sm

GAMMA = 0.1
MIN_BLOCK = 20  # below this, <r> over a block is meaningless
MIN_LEVELS = 10  # levels needed for a <r> value
MIN_EIG_SECTOR = 3  # smallest sector whose eigenstates are recorded
NUM_STD = 3.0  # atypical: S2 this many std below the largest subspace's
POISSON, GOE = 0.386, 0.531  # <r> of uncorrelated and of GOE levels
# A phi level enters the largest-subspace <r> only if at least this
# fraction of its networks has a largest subspace of >= MIN_BLOCK states.
MIN_QUALIFY_FRAC = 0.5
# phi_<r>: |GOE - <r>| < PHI_CHAOTIC_EPS * (GOE - POISSON).
PHI_CHAOTIC_EPS = 0.1


def level_spacing_ratios(evals):
    """min(s_n, s_{n+1}) / max(s_n, s_{n+1}) for sorted levels evals."""
    energy_diff = np.diff(evals) + 1e-8
    return np.min(
        (
            np.divide(energy_diff[:-1], energy_diff[1:]),
            np.divide(energy_diff[1:], energy_diff[:-1]),
        ),
        axis=0,
    )


def mean_level_ratio(levels: np.ndarray) -> float:
    """<r>; NaN with fewer than MIN_LEVELS levels."""
    if len(levels) < MIN_LEVELS:
        return np.nan
    return float(level_spacing_ratios(levels).mean())


def participation_entropy(evecs: np.ndarray, order: float = 2) -> np.ndarray:
    """Rényi participation entropy of every column of evecs."""
    inverse_participation_ratio = (np.abs(evecs) ** (2 * order)).sum(axis=0)
    return np.where(
        inverse_participation_ratio > 1e-15,
        -1.0 / (order - 1) * np.log(inverse_participation_ratio),
        0.0,
    )


def global_entanglement(
    evecs: np.ndarray, sector_state_indices: np.ndarray, num_spins: int
) -> np.ndarray:
    """Meyer-Wallach global entanglement Q of every column of evecs, whose
    rows are the Fock states sector_state_indices (sorted ascending)."""
    n_eig = evecs.shape[1]
    sector_dim = len(sector_state_indices)
    Q = np.zeros(n_eig)
    for i in range(num_spins):
        bp = num_spins - 1 - i
        bits_i = (sector_state_indices >> bp) & 1
        m0 = bits_i == 0
        rho_00 = (np.abs(evecs[m0, :]) ** 2).sum(axis=0)
        rho_11 = 1.0 - rho_00
        pairs = sector_state_indices[m0] ^ (1 << bp)
        b_idx = np.searchsorted(sector_state_indices, pairs)
        b_safe = np.minimum(b_idx, sector_dim - 1)
        valid = (b_idx < sector_dim) & (sector_state_indices[b_safe] == pairs)
        if valid.any():
            rho_01 = (
                evecs[m0][valid, :] * np.conj(evecs[b_safe[valid], :])
            ).sum(axis=0)
        else:
            rho_01 = np.zeros(n_eig)
        Q += 2.0 * (1.0 - rho_00**2 - 2 * np.abs(rho_01) ** 2 - rho_11**2)
    return Q / num_spins


# ── Fig. 2 workers ────────────────────────────────────────────────────────
def _network_sector(num_spins, model_name, k, seed):
    """Network with k flippable spins and its most degenerate sector."""
    rng = np.random.default_rng(seed)
    im, degree = model.build_network(num_spins, model_name, rng)
    h = model.h_on_off_to_h(*model.degree_to_h_on_off(degree), k, rng)
    sec_idx, H = model.most_degenerate_sector(im, h, GAMMA)
    return rng, sec_idx, H


def level_statistics_worker(args):
    """(phi, <r> of the sector, <r> of its largest subspace, sector dim,
    largest-subspace dim, number of subspaces, largest-subspace fraction)."""
    _net_idx, num_spins, model_name, k, seed = args
    rng, sec_idx, H = _network_sector(num_spins, model_name, k, seed)
    dim = len(sec_idx)
    if dim < MIN_BLOCK:
        return None
    try:
        A = sm.adjacency_from_H(H, GAMMA)
    except RuntimeError:
        return None
    split = sm.split_components(A, sm.states_from_indices(sec_idx, num_spins))
    sizes = np.bincount(split.labels)
    member = np.flatnonzero(split.labels == int(np.argmax(sizes)))

    # A small random diagonal field lifts trivial degeneracies.
    H = H + np.diag(
        model.sector_diagonal_noise(sec_idx, num_spins, GAMMA, rng)
    )
    r_whole = mean_level_ratio(np.linalg.eigvalsh(H))
    r_giant = np.nan
    if len(member) >= MIN_BLOCK:
        r_giant = mean_level_ratio(
            np.linalg.eigvalsh(H[np.ix_(member, member)])
        )
    return (
        k / num_spins,
        float(r_whole),
        float(r_giant),
        float(dim),
        float(len(member)),
        float(len(sizes)),
        len(member) / dim,
    )


def eigenstate_worker(args):
    """Participation entropy S2 of every eigenstate of the sector, with its
    host subspace and whether it is atypical (S2 more than NUM_STD standard
    deviations below the mean over the largest subspace's eigenstates)."""
    net_idx, num_spins, model_name, k, seed = args
    rng, sec_idx, H = _network_sector(num_spins, model_name, k, seed)
    dim = len(sec_idx)
    if dim < MIN_EIG_SECTOR:
        return None
    # Each block diagonalised on its own, so eigenvectors never mix blocks.
    ss = sm.states_from_indices(sec_idx, num_spins)
    try:
        A = sm.adjacency_from_H(H, GAMMA)
    except RuntimeError:
        return None
    H = H + np.diag(
        model.sector_diagonal_noise(sec_idx, num_spins, GAMMA, rng)
    )
    labels = sm.split_components(A, ss).labels
    n_comp = labels.max() + 1
    sizes = np.bincount(labels, minlength=n_comp)
    giant = int(np.argmax(sizes))

    s2_l, host_l = [], []
    for c in range(n_comp):
        member = np.flatnonzero(labels == c)
        _evals, evecs = np.linalg.eigh(H[np.ix_(member, member)])
        s2_l.append(participation_entropy(evecs, order=2))
        host_l.append(np.full(len(member), c))
    s2 = np.concatenate(s2_l)
    host = np.concatenate(host_l)

    ref = host == giant
    atyp = np.zeros(len(s2), dtype=bool)
    if ref.sum() >= 2 and s2[ref].std() > 0:
        z = (s2 - s2[ref].mean()) / s2[ref].std()
        atyp = ref & (z < -NUM_STD)

    return {
        "s2": s2,
        "atyp": atyp,
        "giant_wt": (host == giant).astype(float),
        "host_size": sizes[host].astype(float),
        "sector_dim": np.full(len(s2), float(dim)),
        "giant_frac": np.full(len(s2), sizes[giant] / dim),
        "n_comp": np.full(len(s2), float(n_comp)),
        "phi": np.full(len(s2), k / num_spins),
        "net": np.full(len(s2), float(net_idx)),
    }


# ── Eigenstates of one example sector (Extended Data Figs. 5-7) ──────────
Z_WINDOW = 0.5  # energy window, in units of Gamma, for local_z outliers


def half_entropy(V, sec_idx, N, A):
    """Von Neumann entanglement entropy between spins A and the rest for
    every column of V (eigenvectors on the Fock states sec_idx)."""
    n = V.shape[1]
    B = [i for i in range(N) if i not in A]
    full = np.zeros((2**N, n), dtype=V.dtype)
    full[sec_idx] = V
    psi = (
        full.T.reshape((n,) + (2,) * N)
        .transpose([0] + [1 + i for i in A] + [1 + i for i in B])
        .reshape(n, 2 ** len(A), -1)
    )
    p = np.linalg.svd(psi, compute_uv=False) ** 2
    return -(p * np.log(np.where(p > 0, p, 1))).sum(1)


def half_cut_entropy(V, sec_idx, N, ring):
    """Half-system entanglement entropy. Ring: averaged over the N/2
    contiguous half-ring cuts. Random network: the balanced cut whose
    sector-averaged entropy is the median over all C(N, N/2)/2 cuts."""
    if ring:
        cuts = [[(j + i) % N for i in range(N // 2)] for j in range(N // 2)]
        return (
            np.mean([half_entropy(V, sec_idx, N, A) for A in cuts], 0),
            None,
            None,
        )
    cuts = [[0, *c] for c in combinations(range(1, N), N // 2 - 1)]
    S = np.array([half_entropy(V, sec_idx, N, A) for A in cuts])
    k = np.argsort(S.mean(1))[len(cuts) // 2]
    return S[k], cuts[k], (S, cuts)


def local_z(E, y, window, min_count=10):
    """z-score of y against the other eigenstates within +-window in
    energy (NaN where fewer than min_count neighbours)."""
    z = np.full(len(y), np.nan)
    for a in range(len(y)):
        m = np.abs(E - E[a]) < window
        m[a] = False
        if m.sum() >= min_count:
            z[a] = (y[a] - y[m].mean()) / y[m].std()
    return z


def eigensystem(im, h, sec_idx, ring=False):
    """Nondegenerate eigenstates of the largest kinetically accessible
    subspace of the sector, with no added field (main-text Methods).

    Random network: the subspace is diagonalized on its own, and
    eigenstates in a degenerate eigenspace (the zero modes at E0, or
    accidental duplicates) are dropped, since their S2 is basis-dependent.
    Ring: translation T and reflection R are resolved first -- tiny
    symmetry-breaking terms make eigh return simultaneous eigenstates of
    H, T and R (as in Extended Data Fig. 6's scar tower) -- and only states
    still degenerate within the same (momentum, parity) labels are
    dropped.
    Returns (E, V, Fock indices of the subspace, E0).
    """
    N = len(h)
    H = model.sector_hamiltonian(im, h, sec_idx, GAMMA)
    E0 = float(np.diag(H).mean())  # classical energy of the equienergy sector
    labels = sm.split_components(
        sm.adjacency_from_H(H, GAMMA), sm.states_from_indices(sec_idx, N)
    ).labels
    member = np.flatnonzero(labels == np.argmax(np.bincount(labels)))
    H, sub = H[np.ix_(member, member)], sec_idx[member]
    if not ring:
        E, V = np.linalg.eigh(H)
        keep = model.nondegenerate_mask(E)
        return E[keep], V[:, keep], sub, E0
    bits = (sub[:, None] >> (N - 1 - np.arange(N))) & 1
    pos = {s: i for i, s in enumerate(sub)}

    def to_idx(b):
        return pos[int("".join(map(str, b)), 2)]

    T = np.zeros_like(H)
    T[[to_idx(np.roll(b, 1)) for b in bits], np.arange(len(sub))] = 1
    R = np.zeros_like(H)
    R[[to_idx(b[::-1]) for b in bits], np.arange(len(sub))] = 1
    assert np.abs(H @ T - T @ H).max() < 1e-12
    _, V = np.linalg.eigh(H + 1e-6 * (T + T.T) + 3e-7j * (T - T.T) + 2e-8 * R)
    E = np.einsum("ia,ij,ja->a", V.conj(), H, V).real
    m = (
        np.rint(
            np.angle(np.einsum("ia,ij,ja->a", V.conj(), T, V))
            / (2 * np.pi / N)
        ).astype(int)
        % N
    )
    p = np.sign(np.einsum("ia,ij,ja->a", V.conj(), R, V).real)
    p = np.where((m == 0) | (m == N // 2), p, 0)
    key = np.c_[m, p]
    keep = np.ones(len(E), dtype=bool)
    for a in range(len(E)):
        same = (np.abs(E - E[a]) < 1e-6) & (key == key[a]).all(1)
        keep[a] = same.sum() == 1
    order = np.argsort(E[keep])
    return E[keep][order], V[:, keep][:, order], sub, E0


def analyse_eigenstates(
    im, h, sec_idx, ring=False, num_std=NUM_STD, z_window=Z_WINDOW
):
    """S2, Q and S_half of the nondegenerate eigenstates of the largest
    subspace (see eigensystem); atypical = S2 more than num_std standard
    deviations below their mean; S_half outliers by local_z within
    +-z_window * GAMMA in energy."""
    N = len(h)
    E, V, sub, E0 = eigensystem(im, h, sec_idx, ring)
    s2 = participation_entropy(V, order=2)
    Q = global_entanglement(V, sub, N)
    z = (s2 - s2.mean()) / s2.std()
    atyp = z < -num_std
    cut = s2.mean() - num_std * s2.std()
    x = (E - E0) / (N * GAMMA)
    A = np.c_[np.ones_like(x), x**2]

    def resid(y):
        return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]

    r = float(np.corrcoef(s2, Q)[0, 1])
    r_partial = float(np.corrcoef(resid(s2), resid(Q))[0, 1])
    S_half, half_cut, all_cuts = half_cut_entropy(V, sub, N, ring)
    zS = local_z(E, S_half, z_window * GAMMA)
    zQ = local_z(E, Q, z_window * GAMMA)
    return {
        "N": N,
        "E": E,
        "E0": E0,
        "s2": s2,
        "Q": Q,
        "atyp": atyp,
        "cut": cut,
        "r": r,
        "r_partial": r_partial,
        "dim": len(sub),
        "n_eig": len(E),
        "S_half": S_half,
        "half_cut": half_cut,
        "all_cuts": all_cuts,
        "zS": zS,
        "zQ": zQ,
        "ent_outlier": zS < -num_std,
        "r_h": float(np.corrcoef(S_half, Q)[0, 1]),
        "r_h_partial": float(np.corrcoef(resid(S_half), resid(Q))[0, 1]),
    }
