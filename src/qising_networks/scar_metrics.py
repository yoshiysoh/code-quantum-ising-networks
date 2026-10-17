"""Time-domain metrics and the five-class classification, shared by the
data generators, the analysis and the figures.

For an initial Fock state alpha with spins s_i(alpha):

    exc_i     max_t - min_t of s_i(alpha) <Z_i(t)>; 2 when the spin fully
              inverts.
    n_eff     (sum_i exc_i)^2 / sum_i exc_i^2, the number of spins that
              share the excursion.
    n_inv(c)  number of spins with exc_i > c.
    fid_min   minimum fidelity over the whole window.
    fid_rev   maximum fidelity after RELAX_TIME (the revival).
    support   number of spins that vary within the state's subspace.

The sector Hamiltonian is block-diagonal in the connected components of
the sector, so the dynamics is computed per component.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from . import DATA_DIR

# dt = NYQUIST_SAFETY * pi / (spectral width of the network's sector).
NYQUIST_SAFETY: float = 0.8


def make_time_grid(
    evals: np.ndarray, time_max: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(time_list, hann_window, omega) sized for this network's spectrum."""
    max_freq = float(evals[-1] - evals[0])
    if max_freq <= 0:
        raise RuntimeError("degenerate sector: zero spectral width")
    dt = NYQUIST_SAFETY * np.pi / max_freq
    n_time = int(np.ceil(time_max / dt)) + 1
    time_list = np.linspace(0.0, time_max, n_time)
    omega = 2 * np.pi * np.fft.rfftfreq(n_time, time_list[1] - time_list[0])
    return time_list, np.hanning(n_time), omega


DEFAULT_TIME_MAX = 2000.0
DEFAULT_THRESHOLDS = (0.5, 1.0, 1.5)
BATCH_BYTES_BUDGET = 200e6


def n_eff(exc: np.ndarray) -> np.ndarray:
    """Participation number of a per-spin excursion vector (last axis = spins)."""
    exc = np.asarray(exc, dtype=float)
    ssq = (exc**2).sum(axis=-1)
    tot = exc.sum(axis=-1)
    return np.divide(tot**2, ssq, out=np.zeros_like(ssq), where=ssq > 0)


def states_from_indices(idx: np.ndarray, num_spins: int) -> np.ndarray:
    """Fock-state bit patterns as +-1 spin vectors (spin i is bit N-1-i)."""
    pos = np.arange(num_spins - 1, -1, -1)
    return 2.0 * ((np.asarray(idx)[:, None] >> pos) & 1) - 1.0


def adjacency_from_H(H: np.ndarray, Gamma: float) -> np.ndarray:
    """A from H = E_sec*1 - Gamma*A, with the identity checked rather than assumed.

    On the flippable ensemble the sector Hamiltonian IS the adjacency matrix of
    the Hilbert-space diagram up to this affine map, which is the premise the
    whole component decomposition rests on.
    """
    e_sec = float(H[0, 0])
    A = (e_sec * np.eye(len(H)) - H) / Gamma
    A_int = np.rint(A)
    if not np.allclose(A, A_int, atol=1e-8):
        raise RuntimeError("H is not E*1 - Gamma*A with integer A")
    if A_int.min() < 0 or A_int.max() > 1:
        raise RuntimeError("recovered adjacency is not 0/1")
    if not np.allclose(np.diag(A_int), 0.0):
        raise RuntimeError("recovered adjacency has a nonzero diagonal")
    return A_int


@dataclass
class ComponentSplit:
    labels: np.ndarray  # (D,) component id per Fock state
    sizes: np.ndarray  # (D,) size of each state's component
    support: np.ndarray  # (D,) spins that vary in each state's component
    movable: np.ndarray  # (D, N) bool, which spins can move


def split_components(
    A: np.ndarray, sector_states: np.ndarray
) -> ComponentSplit:
    n_comp, labels = connected_components(csr_matrix(A), directed=False)
    sizes = np.bincount(labels, minlength=n_comp)
    num_spins = sector_states.shape[1]
    movable = np.zeros((n_comp, num_spins), dtype=bool)
    for c in range(n_comp):
        sel = sector_states[labels == c]
        movable[c] = sel.max(axis=0) != sel.min(axis=0)
    return ComponentSplit(
        labels=labels,
        sizes=sizes[labels],
        support=movable.sum(axis=1)[labels],
        movable=movable[labels],
    )


@dataclass
class StateMetrics:
    """Per Fock state; every array is indexed by the states actually measured."""

    state_index: np.ndarray  # position within the sector
    exc: np.ndarray  # (S, N) per-spin excursion
    fid_rev: np.ndarray  # (S,)
    fid_min: np.ndarray  # (S,)
    n_eff: np.ndarray  # (S,)
    exc_min_movable: np.ndarray
    support: np.ndarray
    comp_size: np.ndarray
    n_comp_measured: np.ndarray  # states measured in this state's component
    n_time: int

    def weight(self) -> np.ndarray:
        """Each measured state stands for comp_size / n_comp_measured states.

        Sampling is per component, so pooling the measured states unweighted
        silently under-represents whichever component was sampled -- always
        the giant one, which is exactly the population a sector fraction is
        supposed to be dominated by near phi = 1.
        """
        return self.comp_size / np.maximum(self.n_comp_measured, 1)

    def n_inv(self, cut: float) -> np.ndarray:
        return (self.exc > cut).sum(axis=1)

    def mb2(self, num_spins: int) -> np.ndarray:
        return (
            self.fid_rev
            * (self.exc_min_movable / 2.0)
            * (self.support / num_spins)
        )


# The revival is the largest fidelity after RELAX_TIME (units of 1/J), by
# which thermalizing states have relaxed (Extended Data Fig. 8;
# diagnose_relax_time.py).
RELAX_TIME: float = 50.0


def _collapse_window(time_list: np.ndarray, relax_time: float | None) -> int:
    """Number of leading grid points searched for the first collapse t_min:
    the first quarter of the grid (deepest-dip rule) or t <= relax_time."""
    n_time = len(time_list)
    if relax_time is None:
        return max(2, n_time // 4)
    return max(2, int(np.searchsorted(time_list, relax_time, side="right")))


def revival(
    fid: np.ndarray,
    time_list: np.ndarray,
    relax_time: float | None = RELAX_TIME,
    after_dip: bool = False,
) -> np.ndarray:
    """F_rev of fidelity traces `fid` (..., T) sampled on `time_list`.

    max_{t >= relax_time} F(t). Two earlier rules, which Extended Data
    Fig. 8 compares with, take the max after the deepest dip t_min instead:
    after_dip=True searches t_min in t <= relax_time (the early-dip rule),
    relax_time=None in the first quarter of the grid.
    """
    fid = np.asarray(fid)
    if relax_time is None or after_dip:
        flat = fid.reshape(-1, fid.shape[-1])
        window = _collapse_window(time_list, relax_time)
        k = np.argmin(flat[:, :window], axis=1)
        return np.array([f[j:].max() for f, j in zip(flat, k)]).reshape(
            fid.shape[:-1]
        )
    start = min(
        int(np.searchsorted(time_list, relax_time, side="left")),
        len(time_list) - 1,
    )
    return fid[..., start:].max(axis=-1)


def _measure_block(
    H_block: np.ndarray,
    states_block: np.ndarray,
    which: np.ndarray,
    time_list: np.ndarray,
    eig: tuple[np.ndarray, np.ndarray] | None = None,
    relax_time: float | None = RELAX_TIME,
):
    """Excursions and revivals of the `which` states of one connected
    component. `time_list` is the sector's common grid: a grid sized for one
    small component would be coarser and miss extrema.
    """
    dim, num_spins = states_block.shape
    evals, evecs = eig if eig is not None else np.linalg.eigh(H_block)
    n_time = len(time_list)

    if dim == 1 or (evals[-1] - evals[0]) < 1e-12:
        # An isolated Fock state does not evolve.
        s = len(which)
        return (np.zeros((s, num_spins)), np.ones(s), np.ones(s), n_time)

    phase = np.exp(-1j * np.outer(time_list, evals))

    per_state = 3 * n_time * dim * 16
    batch = max(1, int(BATCH_BYTES_BUDGET // max(per_state, 1)))

    exc = np.empty((len(which), num_spins))
    fid_rev = np.empty(len(which))
    fid_min = np.empty(len(which))
    for start in range(0, len(which), batch):
        chunk = which[start : start + batch]
        n_chunk = len(chunk)
        mixed = phase[:, :, None] * evecs[chunk, :].T[None, :, :]
        amps = (evecs @ mixed.transpose(1, 0, 2).reshape(dim, -1)).reshape(
            dim, n_time, n_chunk
        )
        prob = np.abs(amps) ** 2

        fid = prob[chunk, :, np.arange(n_chunk)]  # (S, T)
        fid_rev[start : start + n_chunk] = revival(fid, time_list, relax_time)
        fid_min[start : start + n_chunk] = fid.min(axis=1)

        spin = np.einsum("bi,bts->sit", states_block, prob)  # (S, N, T)
        signed = spin * states_block[chunk][:, :, None]  # +1 at t=0
        exc[start : start + n_chunk] = signed.max(axis=2) - signed.min(axis=2)
    return exc, fid_rev, fid_min, n_time


def measure_sector(
    sector_state_indices: np.ndarray,
    H: np.ndarray,
    Gamma: float,
    num_spins: int,
    time_max: float = DEFAULT_TIME_MAX,
    max_component: int | None = None,
    sample_rng: np.random.Generator | None = None,
    sample_size: int | None = None,
    force_positions: np.ndarray | None = None,
    relax_time: float | None = RELAX_TIME,
) -> StateMetrics:
    """Metrics of all (or sampled) Fock states of one sector, component by
    component. Components larger than `max_component` have `sample_size`
    states measured; `force_positions` (sector positions) are always
    measured.
    """
    ss = states_from_indices(sector_state_indices, num_spins)
    A = adjacency_from_H(H, Gamma)
    split = split_components(A, ss)

    # Pass 1: diagonalise every component and find the common spectral extent.
    blocks = []
    lo, hi = np.inf, -np.inf
    for comp in np.unique(split.labels):
        member = np.flatnonzero(split.labels == comp)
        H_block = H[np.ix_(member, member)]
        evals, evecs = np.linalg.eigh(H_block)
        lo, hi = min(lo, evals[0]), max(hi, evals[-1])
        blocks.append((member, H_block, evals, evecs))

    if hi - lo < 1e-12:
        time_list = np.array([0.0, time_max])
    else:
        time_list, _window, _omega = make_time_grid(
            np.array([lo, hi]), time_max
        )

    force = (
        np.asarray(force_positions, dtype=int)
        if force_positions is not None
        else np.array([], dtype=int)
    )

    # Pass 2: evolve, all components on the same grid.
    idx_out, exc_out, fid_out, min_out, nmeas_out = [], [], [], [], []
    for member, H_block, evals, evecs in blocks:
        local = np.arange(len(member))
        if (
            max_component is not None
            and len(member) > max_component
            and sample_size is not None
            and sample_size < len(member)
        ):
            rng = sample_rng or np.random.default_rng(0)
            local = np.sort(
                rng.choice(len(member), size=sample_size, replace=False)
            )
            forced_local = np.flatnonzero(np.isin(member, force))
            if forced_local.size:
                local = np.union1d(local, forced_local)
        exc, fid, fmin, _nt = _measure_block(
            H_block,
            ss[member],
            local,
            time_list,
            eig=(evals, evecs),
            relax_time=relax_time,
        )
        idx_out.append(member[local])
        exc_out.append(exc)
        fid_out.append(fid)
        min_out.append(fmin)
        nmeas_out.append(np.full(len(local), len(local)))
    ntime = len(time_list)

    order_index = np.concatenate(idx_out)
    exc_all = np.concatenate(exc_out, axis=0)
    fid_all = np.concatenate(fid_out)
    min_all = np.concatenate(min_out)
    nmeas_all = np.concatenate(nmeas_out)
    mov = split.movable[order_index]
    exc_min_mov = np.where(
        mov.any(axis=1), np.where(mov, exc_all, np.inf).min(axis=1), 0.0
    )
    return StateMetrics(
        state_index=order_index,
        exc=exc_all,
        fid_rev=fid_all,
        fid_min=min_all,
        n_eff=n_eff(exc_all),
        exc_min_movable=exc_min_mov,
        support=split.support[order_index],
        comp_size=split.sizes[order_index],
        n_comp_measured=nmeas_all,
        n_time=ntime,
    )


# ── Dynamical classification ──────────────────────────────────────────────

CLASS_NAMES = (
    "localized",
    "recurrence",
    "scar",
    "plateau scar",
    "thermalized",
)
LOCALIZED, POINCARE, SCAR, PLATEAU_SCAR, THERMALIZED = 0, 1, 2, 3, 4
N_CLASSES = len(CLASS_NAMES)

DROP_CUT = 0.1  # 1 - min_t F(t) below this: localized
REV_CUT = 0.1  # fixed revival cut, used only without comp_size
NULL_PERCENTILE = 99
ANOMALY_CUT = 3.0  # anomaly_ratio at or above this: scar
FLOOR_CUT = 0.1  # fid_min / fid_rev at or above this: plateau scar
POINCARE_CUT = 0.1  # GOE-null median revival above this (D <~ 180): recurrence

_NULL_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
NULL_FILE = "revival_null.h5"  # from scripts/build_revival_null.py


def revival_null(
    comp_size: np.ndarray,
    percentile: int = NULL_PERCENTILE,
    name: str = NULL_FILE,
) -> np.ndarray:
    """The `percentile` of F_rev a random state reaches by chance in a
    subspace of each state's dimension (table from build_revival_null.py,
    log-interpolated; 1 for dimension 1).
    """
    key = f"{name}:{percentile}"
    if key not in _NULL_CACHE:
        from .h5io import load_arrays

        z = load_arrays(DATA_DIR / name)
        table, pcts = z["table"], list(z["percentiles"])
        col = 3 + pcts.index(percentile)
        _NULL_CACHE[key] = (table[:, 0], table[:, col])
    dims, vals = _NULL_CACHE[key]
    d = np.asarray(comp_size, dtype=float)
    out = np.interp(np.log(np.maximum(d, 2.0)), np.log(dims), vals)
    return np.where(d < 2, 1.0, out)


def anomaly_ratio(fid_rev: np.ndarray, comp_size: np.ndarray) -> np.ndarray:
    """F_rev divided by the median F_rev of states in subspaces of the same
    dimension within the realisation. A fixed cut ignores that small
    subspaces revive by chance, and the GOE null underestimates how much
    these sparse subspaces recur; ANOMALY_CUT = 3 is well above the GOE
    99th/median ratio, at most 1.8 (Extended Data Fig. 8).
    """
    fid_rev = np.asarray(fid_rev, dtype=float)
    comp_size = np.asarray(comp_size)
    out = np.ones(fid_rev.shape)
    for d in np.unique(comp_size):
        m = comp_size == d
        med = np.median(fid_rev[m])
        out[m] = fid_rev[m] / med if med > 0 else 1.0
    return out


def classify(
    fid_min: np.ndarray,
    fid_rev: np.ndarray,
    part: np.ndarray,
    num_spins: int,
    comp_size: np.ndarray | None = None,
    drop_cut: float = DROP_CUT,
    rev_cut: float = REV_CUT,
    part_frac: float = 0.5,
    anomaly_cut: float = ANOMALY_CUT,
    poincare_cut: float = POINCARE_CUT,
    floor_cut: float = FLOOR_CUT,
) -> np.ndarray:
    """Class of each Fock state (CLASS_NAMES).

    Localized: 1 - fid_min < drop_cut. Recurrence: the GOE null median at the
    subspace dimension reaches poincare_cut. Scar: anomaly_ratio >=
    anomaly_cut, split into scar and plateau scar by fid_min / fid_rev at
    floor_cut. Everything else is thermalized. Without comp_size, a fixed
    revival cut rev_cut replaces the null and the anomaly ratio.
    """
    fid_min = np.asarray(fid_min, dtype=float)
    fid_rev = np.asarray(fid_rev, dtype=float)
    floor_ratio = np.divide(
        fid_min, fid_rev, out=np.zeros_like(fid_min), where=fid_rev > 0
    )
    out = np.full(fid_min.shape, THERMALIZED, dtype=np.int8)
    moved = (1.0 - fid_min) >= drop_cut
    out[~moved] = LOCALIZED
    if comp_size is None:
        returned = moved & (fid_rev >= rev_cut)
        out[returned & (floor_ratio < floor_cut)] = SCAR
        out[returned & (floor_ratio >= floor_cut)] = PLATEAU_SCAR
        return out

    # Recurrence is set by the subspace size (the GOE null), not by the
    # subspace's own median, which would make the ring's Neel scar recurrent.
    comp_size = np.asarray(comp_size)
    recurrent = moved & (
        revival_null(comp_size, percentile=50) >= poincare_cut
    )
    anomalous = (
        moved & ~recurrent & (anomaly_ratio(fid_rev, comp_size) >= anomaly_cut)
    )

    out[recurrent] = POINCARE
    out[anomalous & (floor_ratio < floor_cut)] = SCAR
    out[anomalous & (floor_ratio >= floor_cut)] = PLATEAU_SCAR
    return out


def is_many_body(
    part: np.ndarray, num_spins: int, part_frac: float = 0.5
) -> np.ndarray:
    """Whether each state's revival is carried by at least N/2 spins."""
    return np.asarray(part, dtype=float) >= part_frac * num_spins


# ── Self-test: the positive control comes first ──────────────────────────────


def self_test(Gamma: float = 0.1) -> None:
    """Checks the metrics and the class rule on states of known class,
    with the ring's Neel scar as the positive control."""
    import networkx as nx

    from .model import (
        generate_connected_graph,
        most_degenerate_sector,
        neel_index,
        ring,
        sector_containing,
    )

    # 1. block-diagonal decomposition reproduces whole-sector diagonalisation
    rng = np.random.default_rng(0)
    G = generate_connected_graph(10, 0.4, rng)
    adj = nx.to_numpy_array(G)
    deg = (adj > 0).sum(axis=1).astype(int)
    h = np.where(deg % 2 == 1, 1.0, 2.0)
    sec_idx, H = most_degenerate_sector(adj, h, Gamma)
    blocked = measure_sector(sec_idx, H, Gamma, 10, time_max=200.0)

    ss = states_from_indices(sec_idx, 10)
    evals_w, evecs_w = np.linalg.eigh(H)
    grid, _w, _o = make_time_grid(evals_w, 200.0)
    whole = _measure_block(
        H, ss, np.arange(len(sec_idx)), grid, eig=(evals_w, evecs_w)
    )
    order = np.argsort(blocked.state_index)
    dev = np.abs(blocked.exc[order] - whole[0]).max()
    assert dev < 1e-8, (
        f"block-diagonal decomposition disagrees with whole-sector evolution "
        f"(max deviation {dev:.2e})"
    )
    print(
        f"[ok] per-component evolution matches whole-sector (max dev {dev:.2e})"
    )
    assert np.abs(blocked.fid_rev[order] - whole[1]).max() < 1e-8, (
        "block-diagonal fidelity revival disagrees with whole-sector"
    )

    # 2. ring positive control: extensive scar, every spin participating
    # want_fid: largest revival after RELAX_TIME
    for num_spins, want_fid in ((10, 0.80), (12, 0.76)):
        im, hr = ring(num_spins)
        sec, Hr, _rank, _n = sector_containing(
            im, hr, Gamma, neel_index(num_spins)
        )
        m = measure_sector(sec, Hr, Gamma, num_spins)
        a = int(
            np.flatnonzero(
                m.state_index == np.searchsorted(sec, neel_index(num_spins))
            )[0]
        )
        # Not exactly N: the excursions are read off a finite time grid.
        assert m.n_eff[a] > num_spins - 0.05, (
            f"ring N={num_spins}: n_eff {m.n_eff[a]:.3f}, expected ~{num_spins}"
        )
        assert m.exc.min(axis=1)[a] > 1.9, "ring: some spin barely moves"
        assert abs(m.fid_rev[a] - want_fid) < 0.03, (
            f"ring N={num_spins}: fid_rev {m.fid_rev[a]:.3f}, expected ~{want_fid}"
        )
        assert m.n_inv(1.0)[a] == num_spins, "ring: n_inv(1.0) is not N"
        print(
            f"[ok] ring N={num_spins}: n_eff {m.n_eff[a]:.2f}/{num_spins}, "
            f"exc_min {m.exc.min(axis=1)[a]:.3f}, "
            f"fid_rev {m.fid_rev[a]:.3f}, mb2 {m.mb2(num_spins)[a]:.3f}"
        )

    # 3. threshold independence of the ring's participation count
    for cut in DEFAULT_THRESHOLDS:
        assert m.n_inv(cut)[a] == 12, f"ring: n_inv({cut}) = {m.n_inv(cut)[a]}"
    print(f"[ok] ring n_inv is N for every cut in {DEFAULT_THRESHOLDS}")

    # 4. the class rule on two states of known class: the ring's Neel state
    # (scar) and an isolated Fock state (localized).
    cls = classify(m.fid_min, m.fid_rev, m.n_eff, 12, comp_size=m.comp_size)
    ratio = anomaly_ratio(m.fid_rev, m.comp_size)[a]
    assert cls[a] == SCAR, (
        f"ring Neel classified {CLASS_NAMES[cls[a]]}, expected a scar "
        f"(fid_rev {m.fid_rev[a]:.3f}, n_eff {m.n_eff[a]:.2f}, "
        f"anomaly {ratio:.2f} against a cut of {ANOMALY_CUT})"
    )
    assert is_many_body(m.n_eff[a : a + 1], 12)[0], (
        "ring Neel is not many-body"
    )
    print(
        f"[ok] ring Neel is a {CLASS_NAMES[cls[a]]}: fid_rev "
        f"{m.fid_rev[a]:.3f} is {ratio:.1f}x the median of its own "
        f"{m.comp_size[a]}-state subspace (cut {ANOMALY_CUT}), and "
        f"many-body at n_eff {m.n_eff[a]:.1f}"
    )

    # A 4-state subspace in which every state returns to ~0.99 is recurrence.
    tiny_f = np.array([0.99, 0.98, 0.99, 0.97])
    tiny = classify(
        np.zeros(4), tiny_f, np.full(4, 6.0), 12, comp_size=np.full(4, 4)
    )
    assert (tiny == POINCARE).all(), (
        f"a uniformly-reviving 4-state subspace classified as "
        f"{[CLASS_NAMES[c] for c in tiny]}"
    )
    print(
        f"[ok] a 4-state subspace where every state returns to ~0.99 is a "
        f"{CLASS_NAMES[POINCARE]} for all four"
    )

    # An anomalous narrow state in a large subspace is a scar, not recurrence.
    big = np.full(9, 1005)
    frev = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.60])
    parts = np.array([11.0] * 8 + [5.0])
    got = classify(np.zeros(9), frev, parts, 12, comp_size=big)[-1]
    assert got == SCAR, (
        f"an anomalous narrow state in a 1005-state subspace classified "
        f"{CLASS_NAMES[got]}, expected {CLASS_NAMES[SCAR]}"
    )
    assert not is_many_body(parts[-1:], 12)[0], (
        "5 of 12 spins is not many-body"
    )
    print(
        f"[ok] anomalous but narrow state in a 1005-state subspace is a "
        f"{CLASS_NAMES[SCAR]} (not a {CLASS_NAMES[POINCARE]}), and is "
        f"reported as not many-body"
    )

    assert (
        classify(
            np.array([1.0]),
            np.array([1.0]),
            np.array([0.0]),
            12,
            comp_size=np.array([1]),
        )[0]
        == LOCALIZED
    ), "isolated Fock state is not localized"
    print(f"[ok] isolated Fock state is {CLASS_NAMES[LOCALIZED]}")

    # An anomalous state with an elevated floor between revivals: plateau scar.
    frev2 = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.60])
    fmin2 = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.55])
    parts2 = np.array([11.0] * 8 + [12.0])
    got2 = classify(fmin2, frev2, parts2, 12, comp_size=big)[-1]
    assert got2 == PLATEAU_SCAR, (
        f"a many-body state that never relaxes near background between "
        f"revivals classified {CLASS_NAMES[got2]}, expected "
        f"{CLASS_NAMES[PLATEAU_SCAR]}"
    )
    print(
        f"[ok] a many-body state with fid_min/fid_rev "
        f"{fmin2[-1] / frev2[-1]:.2f} (never relaxes between revivals) is "
        f"a {CLASS_NAMES[PLATEAU_SCAR]}, distinct from the "
        f"{CLASS_NAMES[SCAR]} (fid_min/fid_rev = 0) above"
    )


# ── Display ───────────────────────────────────────────────────────────────
DISPLAY_TIME_MAX = 400.0  # the grid the example heatmaps are drawn on


def evolve_for_display(H, sec_idx, num_spins, alphas, time_max):
    """F(t), <Z_i(t)> and the time-averaged occupation, for each state.

    alphas may contain -1 (see examples.choose_states) for a class with no
    candidate; that row is left NaN rather than evolved.
    """
    evals, evecs = np.linalg.eigh(H)
    time_list, _w, _o = make_time_grid(evals, time_max)
    ss = states_from_indices(sec_idx, num_spins)
    dim = len(sec_idx)
    phase = np.exp(-1j * np.outer(time_list, evals))  # (T, dim)

    z_t = np.full((len(alphas), num_spins, len(time_list)), np.nan)
    fid = np.full((len(alphas), len(time_list)), np.nan)
    occ = np.full((len(alphas), dim), np.nan)
    for j, a in enumerate(alphas):
        if a < 0:
            continue
        amps = evecs @ (phase * evecs[a, :][None, :]).T  # (dim, T)
        prob = np.abs(amps) ** 2
        z_t[j] = ss.T @ prob
        fid[j] = prob[a]
        occ[j] = prob.mean(axis=1)
    return time_list, z_t, fid, occ


if __name__ == "__main__":
    self_test()
