"""Figs. 4 and S4: dynamical classes of the largest subspace.

a, fraction of the largest subspace's Fock states in each class against
flippability at N = 15, with phi' where the recurrence fraction falls to
the thermalized one; b, phi' against 1/N; c, the phi = 1 fractions of the
thermalized, scar and plateau-scar classes against 1/N. Fits as in Fig. 2:
a quantity whose fit overshoots its bound (phi' >= 0, fraction <= 1) shows
the N where the fit reaches the bound instead of the intercept.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .. import DATA_DIR as RESULTS
from .. import class_fractions as acf
from .. import fig_maker as fm
from .. import fig_maker_helper as fmh
from .. import scar_metrics as sm
from .common import CLASS_COLORS, MONO, fit_1_over_N, line_crosses

N_DETAIL = 15
N_FIT_MIN = 10
CLASS_MARKERS = ("o", "s", "^", "v", "D")  # by class index
CLASS_ORDER = [0, 1, 4, 2, 3]  # Fig. 3's column order
PANEL_C_CLASSES = [(sm.THERMALIZED, "Thermalized"), (sm.SCAR, "Scar"),
                   (sm.PLATEAU_SCAR, "Plateau scar")]
PANEL_C_BOUND = {"Thermalized": 1.0}  # the only series that can overshoot


def crossing_phi_falling(levels, curve_above, curve_below):
    """Flippability where the falling curve_above first drops to the rising
    curve_below, linearly interpolated; phi = 0, where both vanish, is
    excluded."""
    keep = levels > 0
    levels, diff = levels[keep], (curve_above - curve_below)[keep]
    below = diff <= 0
    if not below.any() or below.all():
        return None
    j = int(np.argmax(below))
    if j == 0:
        return None
    x0, x1 = levels[j - 1], levels[j]
    y0, y1 = diff[j - 1], diff[j]
    return x0 - y0 * (x1 - x0) / (y1 - y0)


def class_frac_phi1(N, model, cls_idx):
    lv, mean, sd = acf.fractions(N, model, giant_only=True, normalize=True)
    j = int(np.argmin(np.abs(lv - 1.0)))
    return mean[j, cls_idx], sd[j, cls_idx]


def class_fraction_figure(model: str):
    """Fig. 4 (model="ferro") or Fig. S4 (model="pmJ"); prints the numbers
    quoted in the text."""
    ext = np.load(RESULTS / "spectral_extrapolation.npz")
    lv_d, mean_d, sd_d = acf.fractions(N_DETAIL, model, giant_only=True,
                                       normalize=True)
    phi_prime_n15 = crossing_phi_falling(lv_d, mean_d[:, sm.POINCARE],
                                         mean_d[:, sm.THERMALIZED])
    print(f"a (phi', N={N_DETAIL}): {phi_prime_n15}")

    n_vals = ext[f"n_values_{model}"]
    n_vals = n_vals[n_vals >= N_FIT_MIN]
    mean_h, sd_h = ext[f"phi_recur_therm_{model}"]
    mean_h, sd_h = mean_h[-len(n_vals):], sd_h[-len(n_vals):]
    ok = np.isfinite(mean_h)
    n_use_h, mean_use_h, sd_use_h = n_vals[ok], mean_h[ok], sd_h[ok]
    x_fit_h, y_fit_h, intercept_h, intercept_se_h = fit_1_over_N(
        n_use_h, mean_use_h, sd_use_h)
    print(f"b (phi_recur_therm): N->inf = {intercept_h:.6f} +- "
          f"{intercept_se_h:.6f}")

    panel_c_data = {}
    for cls_idx, name in PANEL_C_CLASSES:
        means, sds = [], []
        for N in n_vals:
            m, s = class_frac_phi1(int(N), model, cls_idx)
            means.append(m)
            sds.append(s)
        means, sds = np.array(means), np.array(sds)
        x_fit, y_fit, intercept, intercept_se = fit_1_over_N(n_vals, means,
                                                             sds)
        panel_c_data[name] = dict(cls_idx=cls_idx, mean=means, sd=sds,
                                  x_fit=x_fit, y_fit=y_fit,
                                  intercept=intercept,
                                  intercept_se=intercept_se)
        print(f"c ({name}): N->inf = {intercept:.6f} +- {intercept_se:.6f}")

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(fm.MAX_FIG_WIDTH, 1.56 * 1.4))

    # a: plotted (and legend-ordered) in Fig. 3's column order
    ax = ax_a
    for cls in CLASS_ORDER:
        v, e = mean_d[:, cls], sd_d[:, cls]
        ax.errorbar(lv_d, v, yerr=[np.minimum(e, 0.9 * v), e],
                    marker=CLASS_MARKERS[cls], ms=2.5, mew=0.7, lw=0.4,
                    capsize=1.5, color=CLASS_COLORS[cls],
                    label=fmh.CLASS_NAMES[cls])
    if phi_prime_n15 is not None:
        ax.axvline(phi_prime_n15, ls="--", lw=0.4, color="0.3")
        ax.annotate(r"$\phi'$", xy=(phi_prime_n15, 1.02),
                    xycoords=ax.get_xaxis_transform(), xytext=(1, 0),
                    textcoords="offset points", color="0.3", va="top")
    ax.set_xlabel(r"Flippability $\phi$")
    ax.set_ylabel("Fraction of states (largest subspace)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(top=1.32)  # headroom so the phi' label clears the error bars
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.08, 0.5))
    ax.annotate(f"$N={N_DETAIL}$", xy=(0.05, 0.92), xycoords="axes fraction")

    xlim = (-0.01, 1.0 / n_vals.min() * 1.15 + 0.01)

    # b: if the fit crosses phi = 0, mark that N.
    ax = ax_b
    ax.errorbar(1.0 / n_use_h, mean_use_h, yerr=sd_use_h, marker="o",
                ms=2.5, lw=0, elinewidth=0.4, capsize=1.5, color=MONO)
    ax.plot(x_fit_h, y_fit_h, ls="--", lw=0.4, color="0.5")
    ax.axhline(0.0, ls=":", lw=0.4, color="0.5")
    x0 = line_crosses(x_fit_h, y_fit_h, 0.0)
    if x0 is not None:
        arrow_top = mean_use_h.max() * 0.3
        ax.annotate("", xy=(x0, 0), xytext=(x0, arrow_top),
                    arrowprops=dict(arrowstyle="->", lw=0.5, color=MONO))
        label_x = 0.4 * line_crosses(x_fit_h, y_fit_h, arrow_top)
        ax.annotate(rf"$N \approx {1.0 / x0:.0f}$", xy=(label_x, arrow_top),
                    xytext=(0, 2), textcoords="offset points", color=MONO,
                    ha="center", va="bottom")
    else:
        ax.errorbar([0], [intercept_h], yerr=[intercept_se_h], marker="x",
                    ms=6, mec=MONO, elinewidth=0.5, capsize=2, color=MONO,
                    zorder=4)
    ax.set_ylabel(r"$\phi'$")
    ax.set_xlabel("Reciprocal system size $1/N$")
    ax.set_xlim(xlim)

    # c: panel a's class colours
    marker_of = {sm.THERMALIZED: "D", sm.SCAR: "^", sm.PLATEAU_SCAR: "v"}
    ax = ax_c
    for name, info in panel_c_data.items():
        color = CLASS_COLORS[info["cls_idx"]]
        ax.errorbar(1.0 / n_vals, info["mean"], yerr=info["sd"],
                    marker=marker_of[info["cls_idx"]], ms=2.5, mew=0.7, lw=0,
                    elinewidth=0.4, capsize=1.5, color=color, label=name)
        bound = PANEL_C_BOUND.get(name)
        overshoots = bound is not None and info["intercept"] > bound
        x0_i = (line_crosses(info["x_fit"], info["y_fit"], bound)
                if overshoots else None)
        if x0_i is not None:
            keep = info["x_fit"] >= x0_i
            ax.plot(info["x_fit"][keep], info["y_fit"][keep], ls="--",
                    lw=0.4, color=color)
            ylim_i = ax.get_ylim()
            arrow_bottom = bound - 0.125 * (ylim_i[1] - ylim_i[0])
            ax.annotate("", xy=(x0_i, bound), xytext=(x0_i, arrow_bottom),
                        arrowprops=dict(arrowstyle="->", lw=0.5, color=color))
            ax.annotate(rf"$N \approx {1.0 / x0_i:.0f}$",
                        xy=(x0_i, arrow_bottom), xytext=(0, -2),
                        textcoords="offset points", color=color,
                        ha="center", va="top")
        else:
            ax.plot(info["x_fit"], info["y_fit"], ls="--", lw=0.4,
                    color=color)
            ax.errorbar([0], [info["intercept"]], yerr=[info["intercept_se"]],
                        marker="x", ms=6, mec=color, elinewidth=0.5,
                        capsize=2, color=color, zorder=4)
    ax.set_ylabel("Fraction (largest subspace)")
    ax.set_xlabel("Reciprocal system size $1/N$")
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(0.98, 0.5))
    ax.annotate(r"$\phi=1$", xy=(0.05, 0.92), xycoords="axes fraction")
    ax.set_xlim(xlim)

    fm.align_panel_labels_to_yaxis(fig, [ax_a, ax_b, ax_c],
                                   fm.journal_labels("abc"))
    return fig
