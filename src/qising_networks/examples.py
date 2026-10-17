"""Selection of the example networks and initial states (Fig. 3 and
Extended Data Fig. 3).

A random network qualifies when its largest sector holds a Fock state of
every class (evaluate_network); the columns are then chosen by the rules
below (choose_states). The ring examples instead scan every sector of one
fixed ring (scan_ring_sectors) and pick within the Néel state's sector.
The module-level settings MAGNETIZATION, RECURRENCE_SPINS and SCAR_PXP are
set by the command-line scripts that use this module.
"""

from __future__ import annotations

import logging

import numpy as np

from . import scar_metrics as sm
from . import spectra
from .model import (
    GAMMA,
    all_sectors,
    build_network,
    degree_to_h_on_off,
    h_on_off_to_h,
    largest_sector,
    nondegenerate_mask,
)
from .scar_metrics import DISPLAY_TIME_MAX, evolve_for_display
from .scar_metrics import LOCALIZED as LOCALIZED_C
from .spectra import NUM_STD as EIGENSTATE_NUM_STD

log = logging.getLogger("examples")

DEFAULT_PHI = 1.0
DEFAULT_N = 12
CLASSIFY_TIME_MAX = (
    sm.DEFAULT_TIME_MAX
)  # the grid every dataset is measured on
SCAN_MAX_COMPONENT = 400  # above this, sample during the scan
SCAN_SAMPLE_SIZE = 300
BREADTH_CUT = 0.6  # per-spin excursion counted as "visibly moving"
OSC_MIN_CYCLES = (
    4  # local extrema after the transient to count as "oscillating"
)
OSC_PROMINENCE = 0.7  # minimum peak/trough prominence in <Z_i(t)>
OSC_SKIP_FRAC = 0.05  # ignore this leading fraction of the window
OSC_TOP_K = 15  # anomaly-ranked candidates fully evolved per network
MIN_HOST_COMPONENT = 60  # skip atypicality check below this dimension
CENTRAL_ENERGY_BAND = (
    0.25,
    0.75,
)  # normalized energies of atypical eigenstates
SCAR_MIN_OSCILLATING = (
    9  # spins that must clear BREADTH_CUT for the scar panel's pick
)
SCAR_CANDIDATES_CAP = (
    40  # top atypical-eigenstate hosts checked for breadth, per network
)
REV_PEAK_PROMINENCE = (
    0.15  # minimum rise/fall in F(t) to count as a revival peak
)
REV_MIN_PEAKS = 5  # peaks required after the collapse to call F(t) periodic
SCAR_MIN_FID_REV = (
    0.4  # minimum peak revival F_rev the scar panel's pick must reach
)
REV_QUIET_CUT = 0.05  # "relaxed to near zero" -- F(t) below this
REV_QUIET_FRAC_MIN = 0.2  # minimum fraction of the window spent that quiet

# Settings of generate_examples.py / select_example.py:
# --magnetization: total magnetization of every displayed initial state.
MAGNETIZATION: int | None = None
TYPICAL_HOSTS_RANKED = 20  # typical eigenstates tried for the thermalized pick
# --recurrence-spins: number of moving spins in the recurrence column.
RECURRENCE_SPINS: int | None = None
# --scar-pxp: the scar state whose F(t) looks most like the PXP Neel scar:
# at least REV_MIN_PEAKS peaks above PXP_MAJOR_CUT, fewest peaks between
# PXP_MINOR_CUT and PXP_MAJOR_CUT, then the largest time below PXP_ZERO_CUT.
SCAR_PXP = False
PXP_ZERO_CUT = 0.01
PXP_MAJOR_CUT = 0.2
PXP_MINOR_CUT = 0.02
# Scores are compared at TIE_DECIMALS, so that states that score equal up to
# rounding (symmetry-related states) are tied on every machine; ties go to
# the lower Fock index.
TIE_DECIMALS = 9


def _score(value):
    return round(float(value), TIE_DECIMALS)


def _best(pool, values, alphas):
    """Position in `pool` with the largest value at TIE_DECIMALS, ties to
    the lowest Fock index."""
    return pool[
        min(
            range(len(pool)),
            key=lambda i: (-_score(values[pool[i]]), alphas[pool[i]]),
        )
    ]


def pxp_shape(fid):
    """(zero fraction, major peaks, minor peaks) of one fidelity trace."""
    from scipy.signal import find_peaks

    pk, _ = find_peaks(fid, prominence=0.01)
    h = fid[pk]
    return (
        float((fid < PXP_ZERO_CUT).mean()),
        int((h > PXP_MAJOR_CUT).sum()),
        int(((h > PXP_MINOR_CUT) & (h <= PXP_MAJOR_CUT)).sum()),
    )


def magnetization_mask(sec_idx, num_spins):
    """Boolean mask over sector positions: True where the Fock state has
    total magnetization MAGNETIZATION (all True when it is None)."""
    if MAGNETIZATION is None:
        return np.ones(len(sec_idx), dtype=bool)
    ss = sm.states_from_indices(sec_idx, num_spins)
    return ss.sum(axis=1) == MAGNETIZATION


def scar_pool(m, cls, num_spins, breadth_cut=BREADTH_CUT):
    """Scar-class states of which at least N/2 spins clear `breadth_cut`,
    else all scar-class states of the component.
    """
    pool = np.flatnonzero(
        (cls == sm.SCAR) & sm.is_many_body(m.n_eff, num_spins)
    )
    if not pool.size:
        return pool
    wide = pool[m.n_inv(breadth_cut)[pool] >= num_spins // 2]
    return wide if wide.size else pool


def _atypical_eigenstates_by_component(
    H,
    sec_idx,
    num_spins,
    num_std=EIGENSTATE_NUM_STD,
    central_band=CENTRAL_ENERGY_BAND,
):
    """Per component above MIN_HOST_COMPONENT states, eigenstates whose
    participation entropy S2 lies more than `num_std` standard deviations
    below the component's mean (the atypicality criterion of Methods) and
    whose normalized energy lies in `central_band` (edge states have fewer
    levels to mix with).

    Yields (member, z, host_local, atyp): the component's sector positions,
    every eigenstate's z-score, the position in `member` of each
    eigenstate's largest-overlap Fock state, and the mask of the central
    atypical ones.
    """
    A = sm.adjacency_from_H(H, GAMMA)
    ss = sm.states_from_indices(sec_idx, num_spins)
    split = sm.split_components(A, ss)
    allowed = magnetization_mask(sec_idx, num_spins)
    lo, hi = central_band
    for label in np.unique(split.labels):
        member = np.flatnonzero(split.labels == label)
        if len(member) < MIN_HOST_COMPONENT:
            continue
        H_g = H[np.ix_(member, member)]
        evals_g, evecs_g = np.linalg.eigh(H_g)
        # Eigenvectors of a degenerate eigenspace depend on the LAPACK
        # build, so only nondegenerate eigenstates enter (as in Methods).
        nondeg = nondegenerate_mask(evals_g)
        s2_g = spectra.participation_entropy(evecs_g, order=2)
        if not nondeg.any():
            continue
        s2_mean, s2_std = float(s2_g[nondeg].mean()), float(s2_g[nondeg].std())
        if s2_std <= 0:
            continue
        z = (s2_g - s2_mean) / s2_std
        e_span = max(evals_g[-1] - evals_g[0], 1e-12)
        e_pos = (evals_g - evals_g[0]) / e_span
        atyp = nondeg & (z < -num_std) & (e_pos >= lo) & (e_pos <= hi)
        if not atyp.any():
            continue
        # host: the allowed Fock state of largest overlap (MAGNETIZATION)
        ok = allowed[member]
        if not ok.any():
            continue
        host_local = np.argmax(np.where(ok[:, None], evecs_g**2, -1.0), axis=0)
        yield member, z, host_local, atyp


def atypical_host_states(
    H,
    sec_idx,
    num_spins,
    num_std=EIGENSTATE_NUM_STD,
    central_band=CENTRAL_ENERGY_BAND,
):
    """Sector positions of the Fock states that host a central atypical
    eigenstate (largest overlap).
    """
    host_positions = [
        member[np.unique(host_local[atyp])]
        for member, z, host_local, atyp in _atypical_eigenstates_by_component(
            H, sec_idx, num_spins, num_std, central_band
        )
    ]
    if not host_positions:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(host_positions))


def rank_atypical_hosts(
    H,
    sec_idx,
    num_spins,
    num_std=EIGENSTATE_NUM_STD,
    central_band=CENTRAL_ENERGY_BAND,
):
    """[(z, host_position)] for every central atypical eigenstate, lowest S2
    first, with the sector position of its largest-overlap Fock state.
    """
    ranked = []
    for member, z, host_local, atyp in _atypical_eigenstates_by_component(
        H, sec_idx, num_spins, num_std, central_band
    ):
        for j in np.flatnonzero(atyp):
            ranked.append((float(z[j]), int(member[host_local[j]])))
    ranked.sort(key=lambda t: t[0])
    return ranked


def typical_host_state(
    H,
    sec_idx,
    num_spins,
    target_alpha,
    central_band=CENTRAL_ENERGY_BAND,
    n_ranked=None,
):
    """The Fock state of largest overlap with a typical eigenstate (S2
    closest to the component's mean, in the central band) of
    target_alpha's component: the spectral opposite of the scar pick.
    """
    A = sm.adjacency_from_H(H, GAMMA)
    ss = sm.states_from_indices(sec_idx, num_spins)
    split = sm.split_components(A, ss)
    member = np.flatnonzero(split.labels == split.labels[target_alpha])
    H_g = H[np.ix_(member, member)]
    evals_g, evecs_g = np.linalg.eigh(H_g)
    s2_g = spectra.participation_entropy(evecs_g, order=2)
    nondeg = nondegenerate_mask(evals_g)
    s2_mean = (
        float(s2_g[nondeg].mean()) if nondeg.any() else float(s2_g.mean())
    )
    e_span = max(evals_g[-1] - evals_g[0], 1e-12)
    e_pos = (evals_g - evals_g[0]) / e_span
    lo, hi = central_band
    in_band = (e_pos >= lo) & (e_pos <= hi)
    if not in_band.any():
        in_band = np.ones_like(in_band)
    in_band &= nondeg  # see _atypical_eigenstates_by_component
    dist = np.where(in_band, np.abs(s2_g - s2_mean), np.inf)
    ok = magnetization_mask(sec_idx, num_spins)[member]
    weight = np.where(ok[:, None], evecs_g**2, -1.0)
    if n_ranked is None:
        return int(member[int(np.argmax(weight[:, int(np.argmin(dist))]))])
    # Hosts of the n_ranked most typical in-band eigenstates, most typical
    # first, without repeats.
    hosts = []
    for j in np.argsort(dist)[:n_ranked]:
        a = int(member[int(np.argmax(weight[:, j]))])
        if np.isfinite(dist[j]) and a not in hosts:
            hosts.append(a)
    return hosts


def spin_oscillation_counts(
    z_of_t, prominence=OSC_PROMINENCE, skip_frac=OSC_SKIP_FRAC
):
    """Local extrema per spin in <Z_i(t)>, skipping the initial quench.

    scar_pool's excursion test (max - min) does not distinguish a spin that
    swings back and forth from one that relaxes once to a new value and
    stays there -- both have a large excursion, but only the first is
    actually oscillating. A relaxed spin has at most one local extremum
    (the initial move) in this window; two or more means it came back and
    left again.
    """
    from scipy.signal import find_peaks

    n_spins, n_time = z_of_t.shape
    skip = max(1, int(skip_frac * n_time))
    counts = np.zeros(n_spins, dtype=int)
    for i in range(n_spins):
        trace = z_of_t[i, skip:]
        pk, _ = find_peaks(trace, prominence=prominence)
        tr, _ = find_peaks(-trace, prominence=prominence)
        counts[i] = len(pk) + len(tr)
    return counts


def n_oscillating(z_of_t, min_cycles=OSC_MIN_CYCLES, **kwargs):
    return int((spin_oscillation_counts(z_of_t, **kwargs) >= min_cycles).sum())


def fidelity_revival_peaks(
    fid_of_t, prominence=REV_PEAK_PROMINENCE, skip_frac=OSC_SKIP_FRAC
):
    """Number of peaks of F(t) after the initial collapse, each rising at
    least `prominence` above its surroundings. Tells a periodic revival from
    a single collapse onto an elevated plateau.
    """
    from scipy.signal import find_peaks

    n_time = len(fid_of_t)
    skip = max(1, int(skip_frac * n_time))
    pk, _ = find_peaks(fid_of_t[skip:], prominence=prominence)
    return len(pk)


def quiet_fraction(fid_of_t, cut=REV_QUIET_CUT, skip_frac=OSC_SKIP_FRAC):
    """Fraction of the window, after the initial collapse, that F(t) spends
    below `cut`.
    """
    n_time = len(fid_of_t)
    skip = max(1, int(skip_frac * n_time))
    trace = fid_of_t[skip:]
    return float((trace < cut).mean())


def evaluate_network(num_spins, model, phi, seed0, net):
    """One (seed0, net) realisation: None if it does not qualify, else
    (best_key, hit).

    Scar candidates are the Fock states closest to the central atypical
    eigenstates (rank_atypical_hosts), the first SCAR_CANDIDATES_CAP measured
    exactly. A finalist classifies as a scar, has at least
    SCAR_MIN_OSCILLATING spins clearing BREADTH_CUT, at least REV_MIN_PEAKS
    revival peaks, F_rev >= SCAR_MIN_FID_REV, and spends at least
    REV_QUIET_FRAC_MIN of the window below REV_QUIET_CUT. The finalist with
    the largest F_rev is kept (breadth breaks ties). The thermalized state
    is typical_host_state of the scar's subspace.
    """
    rng = np.random.default_rng(seed0 + net)
    im, degree = build_network(num_spins, model, rng)
    h_on, h_off = degree_to_h_on_off(degree)
    h = h_on_off_to_h(h_on, h_off, round(phi * num_spins), rng)
    sec_idx, H = largest_sector(im, h, GAMMA)
    if sec_idx is None or len(sec_idx) < 2:
        return None

    ranked = rank_atypical_hosts(H, sec_idx, num_spins)
    if not ranked:
        return None
    ranked = ranked[:SCAR_CANDIDATES_CAP]
    cand_alphas = np.array([a for _z, a in ranked])
    # With MAGNETIZATION, every state of that magnetization is measured exactly.
    force_all = cand_alphas
    if MAGNETIZATION is not None:
        force_all = np.union1d(
            cand_alphas, np.flatnonzero(magnetization_mask(sec_idx, num_spins))
        )

    # Sample large components; force_positions keeps the scar candidates exact.
    m = sm.measure_sector(
        sec_idx,
        H,
        GAMMA,
        num_spins,
        time_max=CLASSIFY_TIME_MAX,
        max_component=SCAN_MAX_COMPONENT,
        sample_size=SCAN_SAMPLE_SIZE,
        sample_rng=np.random.default_rng(seed0 + net),
        force_positions=force_all,
    )
    cls = sm.classify(
        m.fid_min, m.fid_rev, m.n_eff, num_spins, comp_size=m.comp_size
    )
    # A network qualifies only if every required class has an allowed state.
    allowed_m = magnetization_mask(sec_idx, num_spins)[m.state_index]
    counts = np.bincount(cls[allowed_m], minlength=len(sm.CLASS_NAMES))
    # Plateau scar is not required (the rarest class); if none is found its
    # panel is drawn N/A.
    required = (sm.LOCALIZED, sm.POINCARE, sm.SCAR, sm.THERMALIZED)
    if not all(counts[c] > 0 for c in required):
        return None
    if (
        RECURRENCE_SPINS is not None
        and not (
            (cls == sm.POINCARE) & allowed_m & (m.support == RECURRENCE_SPINS)
        ).any()
    ):
        return None

    # Breadth and periodicity are checked on the display grid, the window
    # actually drawn.
    _t_disp, z_disp, fid_disp, _occ_disp = evolve_for_display(
        H, sec_idx, num_spins, cand_alphas, DISPLAY_TIME_MAX
    )
    exc_disp = z_disp.max(axis=-1) - z_disp.min(axis=-1)  # (cand, num_spins)
    n_wide_disp = (exc_disp > BREADTH_CUT).sum(axis=-1)

    finalists = []
    for j, (z, alpha) in enumerate(ranked):
        pos = np.flatnonzero(m.state_index == alpha)
        if not pos.size:
            continue
        p = int(pos[0])
        if cls[p] != sm.SCAR:
            continue
        wide = int(n_wide_disp[j])
        if wide < SCAR_MIN_OSCILLATING:
            continue
        peaks = fidelity_revival_peaks(fid_disp[j])
        if peaks < REV_MIN_PEAKS:
            continue
        fid_rev = float(m.fid_rev[p])
        if fid_rev < SCAR_MIN_FID_REV:
            continue
        quiet = quiet_fraction(fid_disp[j])
        if quiet < REV_QUIET_FRAC_MIN:
            continue
        finalists.append(
            {
                "p": p,
                "z": z,
                "alpha": alpha,
                "wide": wide,
                "peaks": peaks,
                "fid_rev": fid_rev,
                "quiet": quiet,
            }
        )
    if not finalists:
        return None

    # Plateau-scar finalists: same pool and breadth bar, without the
    # periodicity and quiet-fraction bars (which plateau scars fail by
    # definition).
    plateau_finalists = []
    for j, (z, alpha) in enumerate(ranked):
        pos = np.flatnonzero(m.state_index == alpha)
        if not pos.size:
            continue
        p = int(pos[0])
        if cls[p] != sm.PLATEAU_SCAR:
            continue
        wide = int(n_wide_disp[j])
        if wide < SCAR_MIN_OSCILLATING:
            continue
        fid_rev = float(m.fid_rev[p])
        if fid_rev < SCAR_MIN_FID_REV:
            continue
        plateau_finalists.append(
            {"p": p, "z": z, "alpha": alpha, "wide": wide, "fid_rev": fid_rev}
        )

    def rank(c):
        return (-_score(c["fid_rev"]), -c["wide"], c["alpha"])

    best_plateau = (
        min(plateau_finalists, key=rank) if plateau_finalists else None
    )

    # The most strongly revived finalist, breadth as the tiebreak.
    best = min(finalists, key=rank)
    best_z, alpha_scar = best["z"], best["alpha"]
    n_wide_chosen, n_peaks_chosen = best["wide"], best["peaks"]
    quiet_chosen = best["quiet"]

    # Thermalized: the Fock state overlapping a typical eigenstate of the scar's
    # subspace (typical_host_state), measured with the same seed.
    if MAGNETIZATION is None:
        typ_hosts = [typical_host_state(H, sec_idx, num_spins, alpha_scar)]
    else:
        typ_hosts = typical_host_state(
            H, sec_idx, num_spins, alpha_scar, n_ranked=TYPICAL_HOSTS_RANKED
        )
    alpha_typ = typ_hosts[0]
    alpha_plateau = best_plateau["alpha"] if best_plateau is not None else None
    force = np.append(force_all, typ_hosts)
    if alpha_plateau is not None:
        force = np.append(force, alpha_plateau)
    m = sm.measure_sector(
        sec_idx,
        H,
        GAMMA,
        num_spins,
        time_max=CLASSIFY_TIME_MAX,
        max_component=SCAN_MAX_COMPONENT,
        sample_size=SCAN_SAMPLE_SIZE,
        sample_rng=np.random.default_rng(seed0 + net),
        force_positions=force,
    )
    cls = sm.classify(
        m.fid_min, m.fid_rev, m.n_eff, num_spins, comp_size=m.comp_size
    )
    allowed_m = magnetization_mask(sec_idx, num_spins)[m.state_index]
    counts = np.bincount(cls[allowed_m], minlength=len(sm.CLASS_NAMES))
    if not all(counts[c] > 0 for c in required):
        return None  # should not happen (superset of the first pass)
    scar_idx = int(np.flatnonzero(m.state_index == alpha_scar)[0])
    # With MAGNETIZATION the host of the most typical eigenstate may not
    # thermalize; take the most typical one that does.
    for a in typ_hosts:
        if cls[int(np.flatnonzero(m.state_index == a)[0])] == sm.THERMALIZED:
            alpha_typ = a
            break
    typ_idx = int(np.flatnonzero(m.state_index == alpha_typ)[0])
    plateau_idx = (
        int(np.flatnonzero(m.state_index == alpha_plateau)[0])
        if alpha_plateau is not None
        else None
    )
    if cls[typ_idx] != sm.THERMALIZED:
        log.warning(
            "net %d: typical-eigenstate host (state %d) "
            "classifies as %s, not thermalized -- using it "
            "anyway, since it is the typical-eigenstate "
            "construction, not a classification result",
            net,
            alpha_typ,
            sm.CLASS_NAMES[cls[typ_idx]],
        )

    ratio = sm.anomaly_ratio(m.fid_rev, m.comp_size)
    log.info(
        "net %d qualifies: sector %d, classes %s, scar state is the "
        "most many-body/most revived of %d finalists -- %d/%d spins "
        "> %.1f, F_rev %.2f (%.1fx anomaly), %d revival peaks, %.0f%% "
        "of window < %.2f, host eigenstate %.1f std below the "
        "subspace mean; thermalized state overlaps the typical eigenstate of the "
        "same subspace, F_rev %.3f, on the display grid; plateau "
        "scar %s (%d finalists)",
        net,
        len(sec_idx),
        counts,
        len(finalists),
        n_wide_chosen,
        num_spins,
        BREADTH_CUT,
        best["fid_rev"],
        ratio[scar_idx],
        n_peaks_chosen,
        100 * quiet_chosen,
        REV_QUIET_CUT,
        -best_z,
        m.fid_rev[typ_idx],
        (
            f"F_rev {best_plateau['fid_rev']:.2f}, "
            f"{best_plateau['wide']}/{num_spins} spins > {BREADTH_CUT}"
            if best_plateau is not None
            else "not found"
        ),
        len(plateau_finalists),
    )
    has_plateau = int(counts[sm.PLATEAU_SCAR] > 0)
    return (
        int(best_plateau is not None),
        has_plateau,
        best["fid_rev"],
        n_wide_chosen,
    ), {
        "net": net,
        "im": im,
        "degree": degree,
        "h": h,
        "sec_idx": sec_idx,
        "H": H,
        "metrics": m,
        "cls": cls,
        "scar_idx": scar_idx,
        "typ_idx": typ_idx,
        "plateau_idx": plateau_idx,
    }


def find_realisation(num_spins, model, phi, max_nets, seed0=1000, best_of=1):
    """A network whose largest sector holds every class. With best_of > 1,
    scan until best_of networks qualify and keep the one whose scar
    finalist revives most strongly.
    """
    found = []
    for net in range(max_nets):
        result = evaluate_network(num_spins, model, phi, seed0, net)
        if result is None:
            continue
        found.append(result)
        if len(found) >= best_of:
            break
    if not found:
        return None
    score, hit = max(found, key=lambda x: x[0])
    persist_found, _has_plateau, fid_rev_score, wide_score = score
    log.info(
        "using net %d (plateau scar %s, F_rev %.2f, %d/%d spins "
        "> BREADTH_CUT, of %d candidates)",
        hit["net"],
        "found" if persist_found else "not found",
        fid_rev_score,
        wide_score,
        num_spins,
        len(found),
    )
    return hit


def choose_states(
    m, cls, scar_idx=None, typ_idx=None, plateau_idx=None, allowed=None
):
    """One sector position per class.

    The recurrence state is ranked by scar_metrics.anomaly_ratio (the
    revival relative to the median of its subspace), not by the raw revival,
    which is largest in small subspaces. The scar and thermalized states are
    `scar_idx` and `typ_idx` from evaluate_network when given, else the most
    anomalous scar state and the thermalized state closest to the class
    median revival in the scar's subspace. Localized states are all
    equivalent (F(t) = 1).
    """
    rel = sm.anomaly_ratio(m.fid_rev, m.comp_size)
    chosen = {}
    if allowed is None:
        allowed = np.ones(len(cls), dtype=bool)
    cls = np.where(allowed, cls, -1)

    # localized: prefer a subspace of more than one state
    pool = np.flatnonzero(cls == LOCALIZED_C)
    nontrivial = pool[m.comp_size[pool] > 1]
    pool = nontrivial if nontrivial.size else pool
    chosen[0] = pool[int(np.argmax(m.comp_size[pool]))] if pool.size else None

    # recurrence: the smallest subspace in which at least two spins move,
    # else the second-largest subspace size; ties by anomaly ratio.
    pool = np.flatnonzero(cls == sm.POINCARE)
    if pool.size and RECURRENCE_SPINS is not None:
        exact = pool[m.support[pool] == RECURRENCE_SPINS]
        pool = exact if exact.size else pool
    if pool.size:
        multi = pool[m.support[pool] >= 2]
        if multi.size:
            at_size = multi[m.comp_size[multi] == m.comp_size[multi].min()]
        else:
            sizes = np.sort(np.unique(m.comp_size[pool]))[::-1]
            target = sizes[1] if sizes.size > 1 else sizes[0]
            at_size = pool[m.comp_size[pool] == target]
        chosen[sm.POINCARE] = _best(at_size, rel, m.state_index)

    # scar: find_realisation's oscillation-checked pick, if supplied.
    if scar_idx is not None:
        chosen[sm.SCAR] = scar_idx
    else:
        pool = np.flatnonzero(cls == sm.SCAR)
        if pool.size:
            wide = scar_pool(m, cls, m.exc.shape[1])
            if wide.size:
                pool = wide
            chosen[sm.SCAR] = _best(pool, rel, m.state_index)

    # plateau scar: find_realisation's pick, else the most anomalous one;
    # -1 when there is none.
    if plateau_idx is not None:
        chosen[sm.PLATEAU_SCAR] = plateau_idx
    else:
        pool = np.flatnonzero(cls == sm.PLATEAU_SCAR)
        if pool.size:
            chosen[sm.PLATEAU_SCAR] = _best(pool, rel, m.state_index)

    # thermalized: evaluate_network's typical-eigenstate pick, if supplied.
    if typ_idx is not None:
        chosen[sm.THERMALIZED] = typ_idx
    else:
        pool = np.flatnonzero(cls == sm.THERMALIZED)
        if pool.size and sm.SCAR in chosen:
            same = pool[m.comp_size[pool] == m.comp_size[chosen[sm.SCAR]]]
            if same.size:
                pool = same
            med = np.median(m.fid_rev[pool])
            chosen[sm.THERMALIZED] = pool[
                int(np.argmin(np.abs(m.fid_rev[pool] - med)))
            ]
        elif pool.size:
            chosen[sm.THERMALIZED] = pool[0]

    # -1 marks a class with no candidate (possible on the ring).
    alphas = np.full(len(sm.CLASS_NAMES), -1, dtype=int)
    for c in range(len(sm.CLASS_NAMES)):
        idx = chosen.get(c)
        if idx is not None:
            alphas[c] = int(m.state_index[idx])
    return alphas


# ── Ring examples ─────────────────────────────────────────────────────────
def gauge_ring(num_spins, h_uniform, rng):
    """Random gauge tau in {+-1}^N and the +-J ring it maps the uniform
    ferromagnetic ring onto: J'_ij = tau_i tau_j (ring edges only),
    h'_i = tau_i * h_uniform. Frustration-free for every tau, since the
    product of the couplings around the loop is 1."""
    tau = rng.choice([-1.0, 1.0], size=num_spins)
    im = np.zeros((num_spins, num_spins))
    for i in range(num_spins):
        j = (i + 1) % num_spins
        im[i, j] = im[j, i] = tau[i] * tau[j]
    loop_product = np.prod(
        [im[i, (i + 1) % num_spins] for i in range(num_spins)]
    )
    assert np.isclose(loop_product, 1.0), (
        "gauge-transformed ring is not frustration-free"
    )
    h = tau * h_uniform
    return im, h, tau


def gauge_index(index, tau):
    """Fock index under the same site-wise gauge flip as gauge_ring, in
    the package's bit convention (bit N-1-i of the index is spin i,
    +1 for a set bit -- see scar_metrics.states_from_indices)."""
    num_spins = len(tau)
    bits = [(index >> (num_spins - 1 - i)) & 1 for i in range(num_spins)]
    spins = 2.0 * np.array(bits, dtype=float) - 1.0
    spins_gauged = tau * spins
    bits_gauged = ((spins_gauged + 1.0) / 2.0).astype(int)
    out = 0
    for b in bits_gauged:
        out = (out << 1) | int(b)
    return out


def scan_ring_sectors(im, h, num_spins):
    """Every energy sector of a ring, measured exhaustively (all are small
    enough -- at most a few thousand states -- that no sampling is needed).

    Returns (degree, per_sector, candidates): per_sector[i] holds that
    sector's (sec_idx, H, m, cls, ratio, adjacency, comp_labels);
    candidates[c] is a flat list of dicts, one per Fock state classified
    into class c anywhere, each tagged with which sector it came from.
    """
    degree = (im != 0).sum(axis=1).astype(int)
    assert ((degree + h.astype(int)) % 2 == 0).all(), (
        "ring with this field is not fully flippable"
    )

    sectors = list(all_sectors(im, h, GAMMA))
    log.info(
        "ring N=%d: %d sectors, sizes %s",
        num_spins,
        len(sectors),
        [len(s[2]) for s in sectors],
    )

    per_sector = []
    candidates = {c: [] for c in range(len(sm.CLASS_NAMES))}
    for srank, e_sec, sec_idx, H in sectors:
        d = len(sec_idx)
        m = sm.measure_sector(
            sec_idx,
            H,
            GAMMA,
            num_spins,
            time_max=CLASSIFY_TIME_MAX,
            max_component=d,
            sample_size=d,
            sample_rng=np.random.default_rng(0),
        )
        cls = sm.classify(
            m.fid_min, m.fid_rev, m.n_eff, num_spins, comp_size=m.comp_size
        )
        ratio = sm.anomaly_ratio(m.fid_rev, m.comp_size)
        A = sm.adjacency_from_H(H, GAMMA)
        ss = sm.states_from_indices(sec_idx, num_spins)
        split = sm.split_components(A, ss)
        s_i = len(per_sector)
        per_sector.append(
            {
                "rank": srank,
                "sec_idx": sec_idx,
                "H": H,
                "m": m,
                "cls": cls,
                "ratio": ratio,
                "A": A,
                "comp_labels": split.labels,
            }
        )
        counts = np.bincount(cls, minlength=len(sm.CLASS_NAMES))
        log.info(
            "  sector rank %d (dim %d): %s",
            srank,
            d,
            dict(zip(sm.CLASS_NAMES, counts)),
        )
        for c in range(len(sm.CLASS_NAMES)):
            for pos in np.flatnonzero(cls == c):
                candidates[c].append(
                    {
                        "sector": s_i,
                        "pos": int(pos),
                        "alpha": int(m.state_index[pos]),
                        "comp_size": int(m.comp_size[pos]),
                        "comp_label": int(split.labels[pos]),
                        "fid_rev": float(m.fid_rev[pos]),
                        "fid_min": float(m.fid_min[pos]),
                        "n_eff": float(m.n_eff[pos]),
                        "ratio": float(ratio[pos]),
                    }
                )
    return degree, per_sector, candidates


def pick_localized(candidates, sector=None):
    """An isolated Fock state, restricted to the scar's own sector if given.

    The scar panel's initial state lives wherever the Neel state does
    (sector rank 5 at N = 12); showing a localized/recurrence panel from a
    DIFFERENT sector would silently compare states that never see the same
    Hilbert space. Returns None (written as N/A by main()) rather than
    reaching into another sector when this class has no representative here.
    """
    pool = candidates[sm.LOCALIZED]
    if sector is not None:
        pool = [c for c in pool if c["sector"] == sector]
    return pool[0] if pool else None


def pick_poincare(candidates, sector=None):
    """Second-largest distinct subspace size, restricted to the scar's own
    sector if given -- see pick_localized."""
    pool = candidates[sm.POINCARE]
    if sector is not None:
        pool = [c for c in pool if c["sector"] == sector]
    if not pool:
        return None
    sizes = sorted({c["comp_size"] for c in pool}, reverse=True)
    target = sizes[1] if len(sizes) > 1 else sizes[0]
    at_size = [c for c in pool if c["comp_size"] == target]
    return min(at_size, key=lambda c: (-_score(c["ratio"]), c["alpha"]))


def pick_plateau_scar(candidates, sector=None):
    """Best-anomaly plateau-scar candidate, restricted to the scar's own
    sector if given -- see pick_localized. The ring's scarring mechanism
    (confirmed at N = 10-14) is purely oscillatory: every ring sector
    checked has zero plateau-scar states, so this returns None and the
    panel is written N/A -- a genuine physical contrast with the network
    ensemble, not a search failure."""
    pool = candidates[sm.PLATEAU_SCAR]
    if sector is not None:
        pool = [c for c in pool if c["sector"] == sector]
    return (
        min(pool, key=lambda c: (-_score(c["ratio"]), c["alpha"]))
        if pool
        else None
    )


def pick_scar(num_spins, per_sector, candidates):
    """Same ranking as evaluate_network: intersect with the many-body/
    breadth pool and the atypical-eigenstate host set within each
    sector, then rank the pooled survivors by (n_oscillating, ratio).
    """
    eig_candidates = []
    for s_i, sec in enumerate(per_sector):
        wide = scar_pool(sec["m"], sec["cls"], num_spins)
        if not wide.size:
            continue
        hosts = atypical_host_states(sec["H"], sec["sec_idx"], num_spins)
        hosted = wide[np.isin(sec["m"].state_index[wide], hosts)]
        pool = hosted if hosted.size else wide
        for pos in pool:
            eig_candidates.append(
                {
                    "sector": s_i,
                    "pos": int(pos),
                    "alpha": int(sec["m"].state_index[pos]),
                    "ratio": float(sec["ratio"][pos]),
                    "hosted": bool(hosted.size),
                }
            )
    if not eig_candidates:
        return None
    order = sorted(
        eig_candidates, key=lambda c: (-_score(c["ratio"]), c["alpha"])
    )[:OSC_TOP_K]
    best, best_key = None, (-1, -1.0)
    for c in order:
        sec = per_sector[c["sector"]]
        _, z_disp, _, _ = evolve_for_display(
            sec["H"], sec["sec_idx"], num_spins, [c["alpha"]], DISPLAY_TIME_MAX
        )
        key = (n_oscillating(z_disp[0]), _score(c["ratio"]))
        if key > best_key:
            best_key, best = key, c
    log.info(
        "ring scar: sector rank %d, %d/%d spins oscillating, %.1fx "
        "anomaly (%s)",
        per_sector[best["sector"]]["rank"],
        best_key[0],
        num_spins,
        best_key[1],
        "eigenstate-hosted" if best["hosted"] else "dynamical fallback",
    )
    return {
        "sector": best["sector"],
        "pos": best["pos"],
        "alpha": best["alpha"],
    }


def pick_scar_neel(num_spins, per_sector, target_index):
    """The Neel (staggered) state target_index -- or its gauge image on the
    ±J ring -- if it classifies as a scar.

    This is the canonical PXP-like scar the ring is included to illustrate,
    so the scar column shows it directly rather than whatever pick_scar's
    oscillation ranking prefers in the same subspace. Returns None (and the
    caller falls back to pick_scar) if it is not classified as a scar.
    """
    for s_i, sec in enumerate(per_sector):
        sector_pos = np.flatnonzero(sec["sec_idx"] == target_index)
        if not sector_pos.size:
            continue
        m_pos = np.flatnonzero(sec["m"].state_index == sector_pos[0])
        if not m_pos.size:
            continue
        pos = int(m_pos[0])
        if sec["cls"][pos] != sm.SCAR:
            log.warning(
                "Neel state found (sector rank %d) but "
                "classifies as %s, not scar -- falling back to "
                "the ranked search",
                sec["rank"],
                sm.CLASS_NAMES[sec["cls"][pos]],
            )
            return None
        log.info(
            "ring scar: Neel state, sector rank %d, "
            "fid_rev %.6f, n_eff %.2f/%d",
            sec["rank"],
            sec["m"].fid_rev[pos],
            sec["m"].n_eff[pos],
            num_spins,
        )
        return {"sector": s_i, "pos": pos, "alpha": int(sector_pos[0])}
    log.warning(
        "Neel state not found in any sector -- falling "
        "back to the ranked search"
    )
    return None


def pick_thermalized(candidates, scar_pick):
    """A bulk-like thermalized Fock state in the scar's component.

    Prefer low fid_rev over the class median: on the ring the median still
    shows strong finite-size revivals in the display window, which reads as
    atypical for the thermalized panel.
    """
    pool = candidates[sm.THERMALIZED]
    if not pool:
        return None
    if scar_pick is not None:
        same = [
            c
            for c in pool
            if c["sector"] == scar_pick["sector"]
            and c["comp_label"]
            == next(
                cc["comp_label"]
                for cc in candidates[sm.SCAR]
                if cc["sector"] == scar_pick["sector"]
                and cc["pos"] == scar_pick["pos"]
            )
        ]
        if same:
            pool = same
    return min(pool, key=lambda c: (_score(c["fid_rev"]), c["alpha"]))


# ── Scans ─────────────────────────────────────────────────────────────────
def spectral_purity(fid_t: np.ndarray) -> float:
    """Fraction of the AC power of F(t) in its single largest FFT bin: 1 for
    a single-frequency decaying oscillation, lower when several frequencies
    compete."""
    x = fid_t - fid_t.mean()
    spec = np.abs(np.fft.rfft(x)) ** 2
    spec[0] = 0.0  # exclude any residual DC leakage
    total = spec.sum()
    if total <= 0:
        return 0.0
    return float(spec.max() / total)
