"""Two-dimensional projection of the Fock-space hypercube (Fig. 1f)."""

from __future__ import annotations

import numpy as np

from .. import model


class WPCA:
    """Weighted principal components of the 2**N spin configurations."""

    def fit(self, data, weights):
        p = weights / weights.sum()
        mean_data = p @ data  # p already sums to 1: weighted sum, not np.mean
        c_data = data - mean_data
        cov = c_data.T * p @ c_data

        values, vectors = np.linalg.eigh(cov)
        values = values[::-1]
        vectors = vectors.T[::-1]
        vectors = vectors * (2 * (vectors.sum(axis=1, keepdims=True) >= 0) - 1)
        self.mean_ = mean_data
        self.components_ = vectors
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T


def hilbert_space_projection(interaction_matrix, external_field_h, seed=42):
    """(wpca, hypercube adjacency, its equal-energy part, energies column).

    The projection weights each Fock state by its number of equal-energy
    neighbours (plus |N(mean, 1)| jitter), so states with many allowed
    flips spread out in the plane.
    """
    rng = np.random.default_rng(seed)
    num_spins = len(interaction_matrix)
    energies = np.vstack(
        model.classical_energies(interaction_matrix, external_field_h)
    )
    dim = 2**num_spins
    idx = np.arange(dim)
    hyp = np.zeros((dim, dim))
    for i in range(num_spins):
        hyp[idx, idx ^ (1 << (num_spins - 1 - i))] += 1.0
    hyp_same = hyp * (energies.T == energies).astype(int)
    weights = np.sum(hyp_same, axis=0)
    weights = weights + np.abs(
        rng.normal(np.mean(weights), 1, size=len(weights))
    )
    wpca = WPCA().fit(model.generate_all_ising_states(num_spins), weights)
    return wpca, hyp, hyp_same, energies
