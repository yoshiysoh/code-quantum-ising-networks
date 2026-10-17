"""Fig. 2 and Extended Data Fig. 2: flippability, fragmentation and level
statistics.

Top row at N = 15: a, the natural flippability of an untuned random network;
b-d, against flippability, the largest subspace's share of the sector, the
mean level-spacing ratio <r>, and the number of atypical eigenstates hosted
by the largest subspace. Bottom row, e-h: a's mean and b-d's phi = 1 (for
g, chaos-onset) values against 1/N with a straight-line fit. Error bars are
the standard deviation across networks (g: across bootstrap resamples); the
N -> infinity intercept is marked with the fit's standard error. A bounded
quantity whose fit overshoots its bound (f, and h when its intercept is
negative) shows the N where the fit reaches the bound instead.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .. import DATA_DIR as RESULTS
from .. import spectra
from ..h5io import load_arrays
from .common import MONO, fit_1_over_N, line_crosses
from .figure import MAX_FIG_WIDTH, align_panel_labels_to_yaxis, journal_labels

POISSON, GOE = spectra.POISSON, spectra.GOE
MIN_BLOCK = spectra.MIN_BLOCK
N_DETAIL = 15  # panels a-d show this system size
N_FIT_MIN = 10  # N = 8, 9 sit off the 1/N trend; every fit uses N >= 10
NATURAL_P = 0.5
NATURAL_N_SAMPLES = 10000
NATURAL_SEED = 42
REFERENCE_LINE = {"f": 1.0}


def load_spectral_data(results=RESULTS):
    """generate_spectral_data.py's output, sweep and --extra merged."""
    merged = {}
    for name in ("spectral_data.h5", "spectral_data_extra.h5"):
        merged.update(load_arrays(results / name))
    return merged


def panel_ab_curves(d, N, model):
    """Per-phi mean and SD of the largest-subspace share, <r> of the whole
    sector, and <r> of the largest subspace (only where at least
    spectra.MIN_QUALIFY_FRAC of the networks have a largest subspace of
    MIN_BLOCK states or more)."""
    a = d[f"levels_{N}_{model}"]
    phi, r_whole, r_giant, giant_dim, gfrac = (
        a[:, 0],
        a[:, 1],
        a[:, 2],
        a[:, 4],
        a[:, 6],
    )
    levels = np.unique(np.round(phi, 6))
    gf_m, gf_s = [], []
    w_m, w_s, g_m, g_s, ok = [], [], [], [], []
    for lv in levels:
        s = np.isclose(phi, lv)
        gf_m.append(gfrac[s].mean())
        gf_s.append(gfrac[s].std())
        fin = s & np.isfinite(r_whole)
        w_ok = fin.sum() >= 5
        w_m.append(r_whole[fin].mean() if w_ok else np.nan)
        w_s.append(r_whole[fin].std() if w_ok else np.nan)
        big = s & (giant_dim >= MIN_BLOCK) & np.isfinite(r_giant)
        good = big.sum() >= max(5, spectra.MIN_QUALIFY_FRAC * s.sum())
        ok.append(good)
        g_m.append(r_giant[big].mean() if good else np.nan)
        g_s.append(r_giant[big].std() if good else np.nan)
    return (
        levels,
        np.array(gf_m),
        np.array(gf_s),
        np.array(w_m),
        np.array(w_s),
        np.array(g_m),
        np.array(g_s),
        np.array(ok),
    )


def first_level_below_eps(levels, curve, eps=spectra.PHI_CHAOTIC_EPS):
    """First level where |GOE - curve| / (GOE - POISSON) < eps: the chaos
    criterion of extrapolate_spectral_data.phi_chaotic_bootstrap, on one
    mean curve."""
    z = np.abs(GOE - curve) / (GOE - POISSON)
    below = z < eps
    if not below.any():
        return None
    return float(levels[np.argmax(below)])


def random_connected_graph(n: int, p: float, seed: int) -> nx.Graph:
    rng_seed = np.random.default_rng(seed)
    while True:
        G = nx.gnp_random_graph(n, p, seed=int(rng_seed.integers(0, 2**31)))
        if nx.is_connected(G):
            return G


def sample_natural_flippability(
    n: int, p: float, n_samples: int, seed: int
) -> np.ndarray:
    """Flippability of n_samples ER(n, p) networks with an untuned field
    h_i ~ Uniform({1, 2})."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_samples)
    for i in range(n_samples):
        G = random_connected_graph(n, p, seed=int(rng.integers(0, 2**31)))
        degree = np.array([deg for _, deg in sorted(G.degree())])
        h = rng.integers(1, 3, size=n)
        out[i] = ((degree + h) % 2 == 0).mean()
    return out


def _print_series(n_vals, mean, sd):
    print(
        "    per-N values: "
        + ", ".join(
            f"N={int(n)}:{m:.5g}+-{s:.4g}" for n, m, s in zip(n_vals, mean, sd)
        )
    )


def spectral_figure(model: str):
    """Fig. 2 (model="ferro") or Extended Data Fig. 2 (model="pmJ"); prints
    the numbers
    quoted in the text."""
    d = load_spectral_data()
    ext = load_arrays(RESULTS / "spectral_extrapolation.h5")

    lv_ab, gf_m, gf_s, w_m, w_s, g_m, g_s, ok_ab = panel_ab_curves(
        d, N_DETAIL, model
    )
    # largest subspace's chaos onset at N = 15, panel c's marker for g
    phi_r_n15 = first_level_below_eps(lv_ab[ok_ab], g_m[ok_ab])
    print(
        f"c (phi_<r>, N={N_DETAIL}, first level within the GOE band): "
        f"{phi_r_n15}"
    )

    atyp = d[f"atypical_{N_DETAIL}_{model}"]
    atyp_sd = d[f"atypical_sd_{N_DETAIL}_{model}"]
    lv_c = atyp[0]
    c_m, c_s = atyp[3] / 2**N_DETAIL, atyp_sd[2] / 2**N_DETAIL

    n_vals = ext[f"n_values_{model}"]
    n_vals = n_vals[n_vals >= N_FIT_MIN]
    extrap = {
        "f": ("giant_frac_phi1", r"Largest-subspace fraction"),
        "g": (
            "phi_chaotic_giant",
            r"$\phi_{\langle r \rangle}$ (largest subspace)",
        ),
        "h": (
            "n_atyp_giant_phi1",
            r"Largest-subspace-hosted $n_{\mathrm{atyp}}/2^N$",
        ),
    }
    extrap_data = {}
    for letter, (key, ylabel) in extrap.items():
        mean, sd = ext[f"{key}_{model}"]
        # stored for N = 8..15: the last len(n_vals) entries are N >= 10
        mean, sd = mean[-len(n_vals) :], sd[-len(n_vals) :]
        if letter == "h":
            norm = 2.0**n_vals
            mean, sd = mean / norm, sd / norm
        x_fit, y_fit, intercept, intercept_se = fit_1_over_N(n_vals, mean, sd)
        bound = REFERENCE_LINE.get(letter)
        extrap_data[letter] = {
            "n_vals": n_vals,
            "mean": mean,
            "sd": sd,
            "x_fit": x_fit,
            "y_fit": y_fit,
            "intercept": intercept,
            "intercept_se": intercept_se,
            "ylabel": ylabel,
            "bound": bound,
        }
        if letter in REFERENCE_LINE:
            print(
                f"{letter} ({key}): fit shown, N -> infinity point "
                f"estimate suppressed (bounded quantity, straight-line "
                f"fit overshoots the bound of {bound}) -- see the "
                f"fit-crosses-bound N marked in the figure instead"
            )
        else:
            print(
                f"{letter} ({key}): N->inf = {intercept:.6f} +- "
                f"{intercept_se:.6f}"
            )
        _print_series(n_vals, mean, sd)

    natural_flip_n15 = sample_natural_flippability(
        N_DETAIL, NATURAL_P, NATURAL_N_SAMPLES, seed=NATURAL_SEED
    )
    print(
        f"a (natural flippability, N={N_DETAIL}): "
        f"mean={natural_flip_n15.mean():.4f}, "
        f"sd={natural_flip_n15.std():.4f}"
    )
    natural_flip_mean, natural_flip_sd = [], []
    for N in n_vals:
        samp = sample_natural_flippability(
            int(N), NATURAL_P, NATURAL_N_SAMPLES, seed=NATURAL_SEED + int(N)
        )
        natural_flip_mean.append(samp.mean())
        natural_flip_sd.append(samp.std())
    natural_flip_mean = np.array(natural_flip_mean)
    natural_flip_sd = np.array(natural_flip_sd)
    x_fit_nat, y_fit_nat, intercept_nat, intercept_se_nat = fit_1_over_N(
        n_vals, natural_flip_mean, natural_flip_sd
    )
    print(
        f"e (natural flippability): N->inf = {intercept_nat:.6f} +- "
        f"{intercept_se_nat:.6f}"
    )
    _print_series(n_vals, natural_flip_mean, natural_flip_sd)

    fig = plt.figure(figsize=(MAX_FIG_WIDTH, 4.2))
    subfig_top, subfig_bottom = fig.subfigures(2, 1, height_ratios=[1, 1.1])

    def label_N(ax, xy=(0.05, 0.92)):
        ax.annotate(f"$N={N_DETAIL}$", xy=xy, xycoords="axes fraction")

    def label_phi1(ax, xy=(0.05, 0.92), ha="left"):
        ax.annotate(r"$\phi=1$", xy=xy, xycoords="axes fraction", ha=ha)

    axes_top = subfig_top.subplots(1, 4)
    ax_a, ax_b, ax_c, ax_d = axes_top

    # a: probability of each grid value k/N over the untuned draws
    ax = ax_a
    levels_nat = np.arange(N_DETAIL + 1) / N_DETAIL
    counts_nat = np.bincount(
        np.round(natural_flip_n15 * N_DETAIL).astype(int),
        minlength=N_DETAIL + 1,
    )
    ax.bar(
        levels_nat,
        counts_nat / counts_nat.sum(),
        width=0.8 / N_DETAIL,
        color=MONO,
    )
    ax.set_ylabel("Probability")
    label_N(ax)

    ax = ax_b
    ax.errorbar(
        lv_ab,
        gf_m,
        yerr=gf_s,
        marker="o",
        ms=2.5,
        lw=0.4,
        capsize=1.5,
        color=MONO,
    )
    ax.set_ylabel("Largest-subspace fraction")
    ax.set_ylim(-0.03, 1.03)
    label_N(ax)

    ax = ax_c
    ax.axhline(POISSON, ls="--", lw=0.4, color="0.5")
    ax.axhline(GOE, ls=":", lw=0.4, color="0.5")
    ax.errorbar(
        lv_ab,
        w_m,
        yerr=w_s,
        marker="s",
        ms=2.5,
        mfc="none",
        lw=0.4,
        capsize=1.5,
        color=MONO,
        label="Whole sector",
    )
    ax.errorbar(
        lv_ab[ok_ab],
        g_m[ok_ab],
        yerr=g_s[ok_ab],
        marker="o",
        ms=2.5,
        lw=0.4,
        capsize=1.5,
        color=MONO,
        label="Largest subspace",
    )
    if phi_r_n15 is not None:
        ax.axvline(phi_r_n15, ls=":", lw=0.4, color="0.3")
        ax.annotate(
            r"$\phi_{\langle r \rangle}$",
            xy=(phi_r_n15, 0.94),
            xycoords=ax.get_xaxis_transform(),
            xytext=(5, 0),
            textcoords="offset points",
            color="0.3",
            va="top",
        )
    ax.set_ylabel(r"Mean level-spacing ratio $\langle r \rangle$")
    ax.set_ylim(0.20, 0.60)  # room below the data for the legend
    handles, labels = ax.get_legend_handles_labels()
    # white, borderless frame: hides the phi_<r> guide line behind the text
    ax.legend(
        handles[::-1],
        labels[::-1],
        frameon=True,
        framealpha=1,
        facecolor="white",
        edgecolor="none",
        loc="lower left",
    )
    label_N(ax)

    # d: same flippability levels as c's largest-subspace curve
    ax = ax_d
    good = (c_m > 0) & np.isin(np.round(lv_c, 6), np.round(lv_ab[ok_ab], 6))
    ax.errorbar(
        lv_c[good],
        c_m[good],
        yerr=[np.minimum(c_s[good], 0.9 * c_m[good]), c_s[good]],
        marker="o",
        ms=2.5,
        lw=0.4,
        capsize=1.5,
        color=MONO,
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"Largest-subspace-hosted $n_{\mathrm{atyp}}/2^N$")
    label_N(ax)

    for ax in axes_top:
        ax.set_xlabel(r"Flippability $\phi$")
        ax.set_xlim(-0.03, 1.03)

    axes_bottom = subfig_bottom.subplots(1, 4)
    ax_e, ax_f, ax_g, ax_h = axes_bottom
    xlim = (-0.01, 1.0 / n_vals.min() * 1.15 + 0.01)

    def intercept_marker(ax, value, se):
        ax.errorbar(
            [0],
            [value],
            yerr=[se],
            marker="x",
            ms=6,
            mec=MONO,
            elinewidth=0.5,
            capsize=2,
            color=MONO,
            zorder=4,
        )

    ax = ax_e
    ax.errorbar(
        1.0 / n_vals,
        natural_flip_mean,
        yerr=natural_flip_sd,
        marker="o",
        ms=2.5,
        lw=0,
        elinewidth=0.4,
        capsize=1.5,
        color=MONO,
    )
    ax.plot(x_fit_nat, y_fit_nat, ls="--", lw=0.4, color="0.5")
    intercept_marker(ax, intercept_nat, intercept_se_nat)
    ax.set_ylabel(r"Natural flippability $\overline{\phi}$")

    # f: the fit overshoots the bound 1, so mark where it reaches it
    ax = ax_f
    info = extrap_data["f"]
    ax.errorbar(
        1.0 / info["n_vals"],
        info["mean"],
        yerr=info["sd"],
        marker="o",
        ms=2.5,
        lw=0,
        elinewidth=0.4,
        capsize=1.5,
        color=MONO,
    )
    ax.plot(info["x_fit"], info["y_fit"], ls="--", lw=0.4, color="0.5")
    ax.axhline(info["bound"], ls=":", lw=0.4, color="0.5")
    x0_f = line_crosses(info["x_fit"], info["y_fit"], info["bound"])
    if x0_f is not None:
        ylim_f = ax.get_ylim()
        arrow_bottom = info["bound"] - 0.125 * (ylim_f[1] - ylim_f[0])
        ax.annotate(
            "",
            xy=(x0_f, info["bound"]),
            xytext=(x0_f, arrow_bottom),
            arrowprops={"arrowstyle": "->", "lw": 0.5, "color": MONO},
        )
        ax.annotate(
            rf"$N \approx {1.0 / x0_f:.0f}$",
            xy=(x0_f, arrow_bottom),
            xytext=(0, -2),
            textcoords="offset points",
            color=MONO,
            ha="center",
            va="top",
        )
    ax.set_ylabel(info["ylabel"])
    label_phi1(ax, xy=(0.95, 0.92), ha="right")

    ax = ax_g
    info = extrap_data["g"]
    ax.errorbar(
        1.0 / info["n_vals"],
        info["mean"],
        yerr=info["sd"],
        marker="o",
        ms=2.5,
        lw=0,
        elinewidth=0.4,
        capsize=1.5,
        color=MONO,
    )
    ax.plot(info["x_fit"], info["y_fit"], ls="--", lw=0.4, color="0.5")
    intercept_marker(ax, info["intercept"], info["intercept_se"])
    ax.set_ylabel(info["ylabel"])

    # h: non-negative, so the lower error bar is clipped as in d; if the
    # fit crosses 0, mark that N instead of the (negative) intercept
    ax = ax_h
    info = extrap_data["h"]
    h_lo = np.minimum(info["sd"], 0.9 * info["mean"])
    ax.errorbar(
        1.0 / info["n_vals"],
        info["mean"],
        yerr=[h_lo, info["sd"]],
        marker="o",
        ms=2.5,
        lw=0,
        elinewidth=0.4,
        capsize=1.5,
        color=MONO,
    )
    ax.plot(info["x_fit"], info["y_fit"], ls="--", lw=0.4, color="0.5")
    ax.axhline(0.0, ls=":", lw=0.4, color="0.5")
    x0_h = line_crosses(info["x_fit"], info["y_fit"], 0.0)
    if x0_h is not None:
        ylim_h = ax.get_ylim()
        arrow_top = 0.125 * (ylim_h[1] - ylim_h[0])
        ax.annotate(
            "",
            xy=(x0_h, 0),
            xytext=(x0_h, arrow_top),
            arrowprops={"arrowstyle": "->", "lw": 0.5, "color": MONO},
        )
        ax.annotate(
            rf"$N \approx {1.0 / x0_h:.0f}$",
            xy=(x0_h, arrow_top),
            xytext=(0, 2),
            textcoords="offset points",
            color=MONO,
            ha="center",
        )
    else:
        intercept_marker(ax, info["intercept"], info["intercept_se"])
    ax.set_ylabel(info["ylabel"])
    label_phi1(ax)

    for ax in axes_bottom:
        ax.set_xlim(xlim)
        ax.set_xlabel("Reciprocal system size $1/N$")

    align_panel_labels_to_yaxis(
        fig,
        [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h],
        journal_labels("abcdefgh"),
    )
    return fig
