"""Finite-size extrapolation data of Figs. 2e-h and 4b (and Extended Data
Figs. 2 and 4).

Reduces the outputs of generate_spectral_data.py and
generate_class_dataset.py, for N = 8-15, to the quantities plotted against
1/N: the largest subspace's share of the sector and its <r> at phi = 1,
phi_<r> (where the largest subspace's <r> comes within PHI_CHAOTIC_EPS of
GOE), the number of atypical eigenstates at phi = 1, and phi' (where the
recurrence fraction falls to the thermalized one). Realisations are
seeded independently per (network, phi), so the crossings are bootstrapped
over networks.

Reads data/spectral_data*.h5 and data/classes_N*.h5; writes
spectral_extrapolation.h5 to data/<date of the run>/.

Run: python3 -u extrapolate_spectral_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qising_networks import DATA_DIR as RESULTS
from qising_networks import class_fractions as acf
from qising_networks import run_dir, spectra
from qising_networks import scar_metrics as sm
from qising_networks.h5io import load_arrays, save_arrays

N_VALUES = list(range(8, 16))
MODELS = ("ferro", "pmJ")
MIN_BLOCK = spectra.MIN_BLOCK
MIN_QUALIFY_FRAC = spectra.MIN_QUALIFY_FRAC
POISSON, GOE = spectra.POISSON, spectra.GOE
R_MID = (POISSON + GOE) / 2
PHI_CHAOTIC_EPS = spectra.PHI_CHAOTIC_EPS
N_BOOT = 2000
RNG_SEED = 42


def load_spectral_data(results: Path = RESULTS) -> dict[str, np.ndarray]:
    """generate_spectral_data.py's output, sweep and --extra merged."""
    merged: dict[str, np.ndarray] = {}
    for name in ("spectral_data.h5", "spectral_data_extra.h5"):
        merged.update(load_arrays(results / name))
    return merged


def phi1_giant_frac(
    a: np.ndarray, extra: np.ndarray | None = None
) -> tuple[float, float]:
    """(mean, SD) of giant_frac across networks at phi = 1. `extra`, if
    given, is a phi=1-only levels_extra_* array (same 7 columns) of
    additional independent networks, pooled in directly."""
    phi = a[:, 0]
    sel = np.isclose(phi, 1.0)
    vals = a[sel, 6]
    if extra is not None:
        vals = np.concatenate([vals, extra[:, 6]])
    return float(vals.mean()), float(vals.std())


def phi1_r_giant(
    a: np.ndarray, extra: np.ndarray | None = None
) -> tuple[float, float]:
    """(mean, SD) of r_giant across networks at phi = 1, MIN_BLOCK-masked.
    `extra`, if given, is a phi=1-only levels_extra_* array pooled in."""
    phi, r_giant, giant_dim = a[:, 0], a[:, 2], a[:, 4]
    sel = (
        np.isclose(phi, 1.0) & (giant_dim >= MIN_BLOCK) & np.isfinite(r_giant)
    )
    vals = r_giant[sel]
    if extra is not None:
        ephi, er_giant, egiant_dim = extra[:, 0], extra[:, 2], extra[:, 4]
        esel = (
            np.isclose(ephi, 1.0)
            & (egiant_dim >= MIN_BLOCK)
            & np.isfinite(er_giant)
        )
        vals = np.concatenate([vals, er_giant[esel]])
    return float(vals.mean()), float(vals.std())


def pooled_stats(
    n1: int, mean1: float, sd1: float, vals2: np.ndarray
) -> tuple[float, float]:
    """(mean, SD) of the union of a summarised group (n1, mean1, sd1,
    population SD) and a second group given as raw per-network values --
    used for panel h, where only the original 100 networks' aggregate is
    on hand (spectral_data.h5 stores no per-network counts), while the
    extra 200 networks' raw per-network counts are."""
    n2 = len(vals2)
    mean2 = float(vals2.mean())
    var1, var2 = sd1**2, float(vals2.var())
    mean_total = (n1 * mean1 + n2 * mean2) / (n1 + n2)
    ss = n1 * (var1 + (mean1 - mean_total) ** 2) + n2 * (
        var2 + (mean2 - mean_total) ** 2
    )
    return mean_total, float(np.sqrt(ss / (n1 + n2)))


def extra_giant_atyp(extra3: np.ndarray) -> np.ndarray:
    """Per-network giant-hosted atypical-eigenstate counts from a
    atypical_extra_* array of raw [net, atyp, in_giant] eigenstate rows --
    same reduction as generate_spectral_data.py's atypical_data."""
    net, atyp, in_giant = extra3
    atyp, in_giant = atyp > 0.5, in_giant > 0.5
    nets = np.unique(net)
    return np.array(
        [float((atyp[net == nn] & in_giant[net == nn]).sum()) for nn in nets]
    )


def phi_prime_bootstrap(
    a: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT
) -> tuple[float, float, int]:
    """(mean, SD, n_valid_draws) of the phi where r_whole crosses R_MID."""
    phi, r_whole = a[:, 0], a[:, 1]
    levels = np.unique(np.round(phi, 6))
    per_level = []
    for lv in levels:
        sel = np.isclose(phi, lv) & np.isfinite(r_whole)
        per_level.append(r_whole[sel])
    crossings = []
    for _ in range(n_boot):
        curve = np.array(
            [
                rng.choice(vals, size=len(vals), replace=True).mean()
                if len(vals)
                else np.nan
                for vals in per_level
            ]
        )
        ok = np.isfinite(curve)
        if ok.sum() < 2:
            continue
        lv_ok, curve_ok = levels[ok], curve[ok]
        above = curve_ok >= R_MID
        if not above.any() or above.all():
            continue
        j = int(np.argmax(above))  # first level at/above R_MID
        if j == 0:
            continue
        x0, x1 = lv_ok[j - 1], lv_ok[j]
        y0, y1 = curve_ok[j - 1], curve_ok[j]
        crossings.append(x0 + (R_MID - y0) * (x1 - x0) / (y1 - y0))
    crossings = np.array(crossings)
    return float(crossings.mean()), float(crossings.std()), len(crossings)


def phi_chaotic_bootstrap(
    a: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
    eps: float = PHI_CHAOTIC_EPS,
) -> tuple[float, float, int]:
    """(mean, SD, n_valid_draws) of the first flippability level at which the
    largest subspace's <r> satisfies |GOE - <r>| / (GOE - POISSON) < eps,
    bootstrapped over networks. Only levels where at least MIN_QUALIFY_FRAC
    of the networks (and at least 5) have a subspace of MIN_BLOCK states
    count. A size that never reaches eps gives NaN.
    """
    phi, r_giant, giant_dim = a[:, 0], a[:, 2], a[:, 4]
    levels = np.unique(np.round(phi, 6))
    per_level, valid_levels = [], []
    for lv in levels:
        at = np.isclose(phi, lv)
        sel = at & (giant_dim >= MIN_BLOCK) & np.isfinite(r_giant)
        if sel.sum() >= max(5, MIN_QUALIFY_FRAC * at.sum()):
            per_level.append(r_giant[sel])
            valid_levels.append(lv)
    valid_levels = np.array(valid_levels)
    goe_range = GOE - POISSON
    crossings = []
    for _ in range(n_boot):
        z = np.empty(len(valid_levels))
        for i, vals in enumerate(per_level):
            idx = rng.integers(0, len(vals), size=len(vals))
            resampled = vals[idx]
            z[i] = abs(GOE - resampled.mean()) / goe_range
        below = z < eps
        if not below.any():
            continue
        j = int(np.argmax(below))  # first level with z < eps
        crossings.append(valid_levels[j])
    crossings = np.array(crossings)
    return float(crossings.mean()), float(crossings.std()), len(crossings)


def phi_recur_therm_bootstrap(
    num_spins: int,
    model: str,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> tuple[float, float, int]:
    """(mean, SD, n_valid_draws) of the phi where the largest subspace's
    recurrence fraction crosses its thermalized fraction (Fig. 4b). Same bootstrap-over-networks-per-level logic as
    phi_prime_bootstrap, but resampling the (recurrence, thermalized)
    pair together per network, since both come from one realisation.
    """
    phi, frac = acf.per_network_fractions(num_spins, model, giant_only=True)
    # phi = 0 excluded: both fractions vanish there, a trivial tie.
    levels = np.unique(np.round(phi, 6))
    levels = levels[levels > 0]
    per_level = []
    for lv in levels:
        sel = np.isclose(phi, lv)
        per_level.append((frac[sel, sm.POINCARE], frac[sel, sm.THERMALIZED]))
    crossings = []
    for _ in range(n_boot):
        diff_curve = []
        for recur_vals, therm_vals in per_level:
            if len(recur_vals) == 0:
                diff_curve.append(np.nan)
                continue
            idx = rng.integers(0, len(recur_vals), size=len(recur_vals))
            diff_curve.append(recur_vals[idx].mean() - therm_vals[idx].mean())
        diff_curve = np.array(diff_curve)
        ok = np.isfinite(diff_curve)
        if ok.sum() < 2:
            continue
        lv_ok, diff_ok = levels[ok], diff_curve[ok]
        # recurrence starts above thermalized (diff > 0) and ends below
        # (diff < 0); find the sign change.
        below = diff_ok <= 0
        if not below.any() or below.all():
            continue
        j = int(np.argmax(below))  # first level at/below 0
        if j == 0:
            continue
        x0, x1 = lv_ok[j - 1], lv_ok[j]
        y0, y1 = diff_ok[j - 1], diff_ok[j]
        crossings.append(x0 - y0 * (x1 - x0) / (y1 - y0))
    crossings = np.array(crossings)
    return float(crossings.mean()), float(crossings.std()), len(crossings)


def main() -> None:
    d = extra = load_spectral_data()
    rng = np.random.default_rng(RNG_SEED)
    out: dict[str, np.ndarray] = {}
    for model in MODELS:
        gf_m, gf_s = [], []
        rg_m, rg_s = [], []
        pp_m, pp_s = [], []
        pc_m, pc_s = [], []
        nt_m, nt_s = [], []
        rt_m, rt_s = [], []
        for n in N_VALUES:
            a = d[f"levels_{n}_{model}"]
            a_extra = extra.get(f"levels_extra_{n}_{model}")
            m, s = phi1_giant_frac(a, a_extra)
            gf_m.append(m)
            gf_s.append(s)
            m, s = phi1_r_giant(a, a_extra)
            rg_m.append(m)
            rg_s.append(s)
            m, s, n_valid = phi_prime_bootstrap(a, rng)
            pp_m.append(m)
            pp_s.append(s)
            print(
                f"  phi' N={n} {model}: {m:.4f} +- {s:.4f} "
                f"({n_valid}/{N_BOOT} valid draws)"
            )
            m, s, n_valid = phi_chaotic_bootstrap(a, rng)
            pc_m.append(m)
            pc_s.append(s)
            print(
                f"  phi_chaotic(giant) N={n} {model}: {m:.4f} +- {s:.4f} "
                f"({n_valid}/{N_BOOT} valid draws)"
            )
            m, s, n_valid = phi_recur_therm_bootstrap(n, model, rng)
            rt_m.append(m)
            rt_s.append(s)
            print(
                f"  phi_recur_therm N={n} {model}: {m:.4f} +- {s:.4f} "
                f"({n_valid}/{N_BOOT} valid draws)"
            )
            atyp = d.get(f"atypical_{n}_{model}")
            atyp_sd = d.get(f"atypical_sd_{n}_{model}")
            if atyp is None:
                nt_m.append(np.nan)
                nt_s.append(np.nan)
                continue
            phi3 = atyp[0]
            sel3 = np.isclose(phi3, 1.0)
            m0, s0 = float(atyp[3, sel3][0]), float(atyp_sd[2, sel3][0])
            n0 = int(np.isclose(a[:, 0], 1.0).sum())
            b_extra = extra.get(f"atypical_extra_{n}_{model}")
            if b_extra is not None and b_extra.shape[1] > 0:
                m, s = pooled_stats(n0, m0, s0, extra_giant_atyp(b_extra))
            else:
                m, s = m0, s0
            print(
                f"  n_atyp(giant) N={n} {model}: {m0:.2f}+-{s0:.2f} "
                f"(n={n0}) -> {m:.2f}+-{s:.2f} (pooled)"
            )
            nt_m.append(m)
            nt_s.append(s)
        out[f"n_values_{model}"] = np.array(N_VALUES, dtype=float)
        out[f"giant_frac_phi1_{model}"] = np.array([gf_m, gf_s])
        out[f"r_giant_phi1_{model}"] = np.array([rg_m, rg_s])
        out[f"phi_prime_{model}"] = np.array([pp_m, pp_s])
        out[f"phi_chaotic_giant_{model}"] = np.array([pc_m, pc_s])
        out[f"n_atyp_giant_phi1_{model}"] = np.array([nt_m, nt_s])
        out[f"phi_recur_therm_{model}"] = np.array([rt_m, rt_s])

    out_path = run_dir() / "spectral_extrapolation.h5"
    save_arrays(out_path, out)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    # No options; the parser makes --help exit without overwriting output.
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
