"""Helpers shared by the plotting modules."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np

MONO = "0.15"

# Dark2, in Fig. 3's column order.
_dark2 = mpl.colormaps["Dark2"].colors
CLASS_COLORS = (_dark2[0], _dark2[1], _dark2[3], _dark2[4], _dark2[2])


def fit_1_over_N(n_values, mean, sd):
    """Weighted straight-line fit of mean against 1/N: (x_fit, y_fit,
    intercept, intercept standard error)."""
    x = 1.0 / n_values
    w = 1.0 / np.maximum(sd, 1e-6 * np.max(np.abs(mean)))
    coeffs, cov = np.polyfit(x, mean, 1, w=w, cov=True)
    slope, intercept = coeffs
    intercept_se = np.sqrt(cov[1, 1])
    x_fit = np.linspace(0, x.max() * 1.08, 50)
    y_fit = slope * x_fit + intercept
    return x_fit, y_fit, intercept, intercept_se


def line_crosses(x_fit, y_fit, target):
    """x where a piecewise-linear curve crosses y = target (None if it
    does not)."""
    diff = y_fit - target
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        return None
    j = sign_change[0]
    x0, x1 = x_fit[j], x_fit[j + 1]
    y0, y1 = diff[j], diff[j + 1]
    return x0 - y0 * (x1 - x0) / (y1 - y0)
