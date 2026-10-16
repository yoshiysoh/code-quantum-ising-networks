"""The model: random Ising networks, fields, and degenerate-sector Hamiltonians.

H = -sum_{i<j} J_ij Z_i Z_j - sum_i h_i Z_i - Gamma sum_i X_i. For a weak
transverse field (Gamma << J) the dynamics stays inside one degenerate
manifold of the classical energy (a "sector"); within it the Hamiltonian is
E_s * I - Gamma * A_s, with A_s the single-flip adjacency of the sector's Fock
states. Fock state k has spin i = +1 where bit N-1-i of k is set.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

GAMMA: float = 0.1    # transverse field, in units of J
P_GRAPH: float = 0.5  # Erdős–Rényi edge probability


# ── Networks and fields ───────────────────────────────────────────────────
def generate_connected_graph(
    num_spins: int, p: float, rng: np.random.Generator
) -> nx.Graph:
    """Draw random Erdős–Rényi graphs until one is connected."""
    while True:
        seed = int(rng.integers(0, 2**31))
        G = nx.gnp_random_graph(num_spins, p, seed=seed)
        if nx.is_connected(G):
            return G


def build_network(num_spins: int, model: str, rng: np.random.Generator):
    """Coupling matrix and degrees of a connected ER network; ferromagnetic
    (J = +1 on every edge) or ±J (random sign per edge)."""
    G = generate_connected_graph(num_spins, P_GRAPH, rng)
    adj = nx.to_numpy_array(G)
    if model == "ferro":
        im = adj.copy()
    elif model == "pmJ":
        rows, cols = np.where(np.triu(adj, 1) > 0)
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(rows))
        im = np.zeros((num_spins, num_spins))
        im[rows, cols] = signs
        im[cols, rows] = signs
    else:
        raise ValueError(model)
    return im, (adj > 0).sum(axis=1).astype(int)


def ring(num_spins):
    """Ferromagnetic ring with PBC and a uniform h = +2J on every site."""
    im = np.zeros((num_spins, num_spins))
    for i in range(num_spins):
        im[i, (i + 1) % num_spins] = im[(i + 1) % num_spins, i] = 1.0
    h = np.full(num_spins, 2.0)
    return im, h


def neel_index(num_spins):
    """Bit pattern 1010... as an integer (spin 0 up, alternating)."""
    bits = [1 if i % 2 == 0 else 0 for i in range(num_spins)]
    return int("".join(str(b) for b in bits), 2)


def degree_to_h_on_off(degree: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Longitudinal fields that turn each spin's flippability on or off.

    h_on satisfies (degree_i + h_on_i) % 2 == 0, i.e. spin i is flippable;
    h_off is the complementary, non-flippable value. Assumes
    h_{i;z} in {1, 2}, matching this paper's convention.
    """
    h_on = np.where(degree % 2 == 0, 2, 1).astype(float)
    h_off = np.where(degree % 2 == 0, 1, 2).astype(float)
    return h_on, h_off


def h_on_off_to_h(
    h_on: np.ndarray,
    h_off: np.ndarray,
    num_flippable: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Longitudinal field vector with exactly num_flippable spins flippable."""
    num_spins = len(h_on)
    active = rng.choice(num_spins, size=num_flippable, replace=False)
    h = h_off.copy()
    h[active] = h_on[active]
    return h


def flipability(degree: np.ndarray, h: np.ndarray) -> float:
    """Fraction of spins with (degree + h) even, i.e. flippable."""
    return float(((degree + h.astype(int)) % 2 == 0).mean())


# ── Full Hamiltonian ──────────────────────────────────────────────────────
def generate_all_ising_states(num_dimensions):
    """All 2**N spin configurations (+-1), site 0 = most-significant bit."""
    vertices_ids = np.arange(2**num_dimensions)
    vertex_binaries = np.array(
        [[(number >> i) & 1 for i in reversed(range(num_dimensions))]
         for number in vertices_ids], order="C")
    return 2 * vertex_binaries - 1


def interaction_to_hamiltonian(
    interaction_matrix: np.ndarray,
    external_field_Gamma: np.ndarray,
    external_field_h: np.ndarray,
) -> np.ndarray:
    """
    Full (2**num_spins, 2**num_spins) dense Ising Hamiltonian:
    H = 0.5 * sum_ij (-J_ij Z_i Z_j) - sum_i Gamma_i X_i - sum_i h_i Z_i.

    Basis index k has Z_i = +1 where bit N-1-i of k is 0 (qutip's
    convention), the reverse of the sector functions below.
    """
    num_spins = len(interaction_matrix)
    dim = 2**num_spins
    idx = np.arange(dim)
    bits = (idx[:, None] >> np.arange(num_spins - 1, -1, -1)) & 1
    signs = 1 - 2 * bits

    diag = (
        -0.5 * np.einsum("ki,ij,kj->k", signs, interaction_matrix, signs)
        - signs @ external_field_h
    )
    hamiltonian = np.diag(diag).astype(np.float64)

    flip_masks = 1 << np.arange(num_spins - 1, -1, -1)
    for i in np.nonzero(external_field_Gamma)[0]:
        partner = idx ^ flip_masks[i]
        hamiltonian[idx, partner] += -external_field_Gamma[i]
    return hamiltonian


# ── Degenerate sectors ────────────────────────────────────────────────────
MIN_SECTOR_DIM = 2   # dim-1 sectors are static by construction


def classical_energies(interaction_matrix, external_field_h):
    """Classical energy of every Fock state, rounded to 1e-9 so that
    degenerate states compare equal."""
    num_spins = len(external_field_h)
    idx = np.arange(2**num_spins)
    bits = (idx[:, None] >> np.arange(num_spins - 1, -1, -1)[None, :]) & 1
    states = 2 * bits.astype(float) - 1
    energy = (-((states @ np.triu(interaction_matrix, 1)) * states).sum(1)
              - states @ external_field_h)
    return np.round(energy, 9)


def _sector_hamiltonian(sec_idx, e_sec, num_spins, Gamma):
    """E_s * I - Gamma * A_s on the sorted Fock states sec_idx."""
    d = len(sec_idx)
    H = np.zeros((d, d))
    np.fill_diagonal(H, e_sec)
    for bit in range(num_spins):
        flip = sec_idx ^ (1 << bit)
        b = np.searchsorted(sec_idx, flip)
        b_safe = np.minimum(b, d - 1)
        valid = (b < d) & (sec_idx[b_safe] == flip)
        H[np.flatnonzero(valid), b_safe[valid]] = -Gamma
    return H


def all_sectors(interaction_matrix, external_field_h, Gamma):
    """Yield (rank, energy, sector_state_indices, H) for every degenerate
    manifold of dimension >= MIN_SECTOR_DIM, largest first."""
    num_spins = len(external_field_h)
    energy = classical_energies(interaction_matrix, external_field_h)
    idx = np.arange(len(energy))
    unique_e, counts = np.unique(energy, return_counts=True)
    for rank, k in enumerate(np.argsort(-counts)):
        e_sec = unique_e[k]
        sec_idx = idx[energy == e_sec]
        if len(sec_idx) < MIN_SECTOR_DIM:
            continue
        yield rank, e_sec, sec_idx, _sector_hamiltonian(sec_idx, e_sec,
                                                        num_spins, Gamma)


def largest_sector(interaction_matrix, external_field_h, Gamma):
    """(sector_state_indices, H) of the most degenerate manifold: the first
    item of all_sectors, or (None, None) if every manifold is static."""
    for _rank, _e_sec, sec_idx, H in all_sectors(interaction_matrix,
                                                 external_field_h, Gamma):
        return sec_idx, H
    return None, None


def sector_hamiltonian(interaction_matrix, external_field_h, sec_idx,
                       Gamma=GAMMA):
    """The Hamiltonian of the manifold whose Fock states are sec_idx."""
    for _rank, _e, si, H in all_sectors(interaction_matrix, external_field_h,
                                        Gamma):
        if np.array_equal(si, sec_idx):
            return H
    raise RuntimeError("sector not found")


def most_degenerate_sector(interaction_matrix, external_field_h, Gamma):
    """(sector_state_indices, H) of the most degenerate manifold, with ties
    going to the lowest energy (np.argmax), as the level-statistics data of
    Fig. 2 were generated. largest_sector breaks ties by np.argsort and can
    pick a different manifold of the same dimension."""
    num_spins = len(external_field_h)
    energy = classical_energies(interaction_matrix, external_field_h)
    unique_e, counts = np.unique(energy, return_counts=True)
    e_sec = unique_e[np.argmax(counts)]
    sec_idx = np.arange(len(energy))[energy == e_sec]
    return sec_idx, _sector_hamiltonian(sec_idx, e_sec, num_spins, Gamma)


def sector_containing(interaction_matrix, external_field_h, Gamma, state_index):
    """(sector_state_indices, H, rank by size, number of manifolds) of the
    manifold holding one Fock state, e.g. the Néel state on the ring."""
    num_spins = len(external_field_h)
    energy = classical_energies(interaction_matrix, external_field_h)
    e_sec = energy[state_index]
    unique_e, counts = np.unique(energy, return_counts=True)
    order = np.argsort(-counts)
    rank = int(np.flatnonzero(unique_e[order] == e_sec)[0])
    sec_idx = np.arange(len(energy))[energy == e_sec]
    H = _sector_hamiltonian(sec_idx, e_sec, num_spins, Gamma)
    return sec_idx, H, rank, len(unique_e)


# ── Perturbations ─────────────────────────────────────────────────────────
DEGENERACY_TOL: float = 1e-8


def nondegenerate_mask(evals: np.ndarray,
                       tol: float = DEGENERACY_TOL) -> np.ndarray:
    """True for each eigenvalue (sorted ascending) with no other eigenvalue
    within `tol` -- its eigenvector is then unique up to a phase."""
    gap = np.diff(evals) > tol
    return np.r_[True, gap] & np.r_[gap, True]


NOISE_FRACTION: float = 0.1


def sector_diagonal_noise(sec_idx: np.ndarray, num_spins: int, gamma: float,
                          rng: np.random.Generator,
                          fraction: float = NOISE_FRACTION) -> np.ndarray:
    """Diagonal of a small random longitudinal field sum_i eps_i Z_i on the
    Fock states sec_idx, eps_i ~ U(-fraction*gamma, fraction*gamma).

    Added to a sector Hamiltonian, it lifts the trivial degeneracies (the
    zero modes at E0 of the bipartite hopping graph, copies of identical
    fragments) without coupling different kinetically accessible
    subspaces, since it is diagonal in the Fock basis.
    """
    eps = rng.uniform(-fraction * gamma, fraction * gamma, num_spins)
    bits = (np.asarray(sec_idx)[:, None]
            >> (num_spins - 1 - np.arange(num_spins))) & 1
    return (2.0 * bits - 1.0) @ eps
