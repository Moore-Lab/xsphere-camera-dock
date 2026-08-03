
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def fit_linear_model_second_half(n_droplets):
    """
    Fit a linear model to the second half of the data and return the fitted values and coefficients.
    Returns
    -------
    y_fit_full : np.ndarray
        Linear fit evaluated over the full x-range.
    coeffs : (m, b)
        Slope and intercept of the fitted line.
    """
    distance_thresholds = np.arange(len(n_droplets))
    half = len(distance_thresholds) // 2
    x_fit = distance_thresholds[half:]
    y_fit = n_droplets[half:]

    if len(x_fit) > 1:
        m, b = np.polyfit(x_fit, y_fit, 1)  # slope, intercept
        y_fit_full = m * distance_thresholds + b
    else:
        m, b = 0.0, float(y_fit[0] if len(y_fit) else 0.0)
        y_fit_full = np.full_like(distance_thresholds, b, dtype=float)
    return y_fit_full, (m, b)


def line_plus_exponential(x, m, b, A, tau):
    """
    Model: linear baseline + exponential bump (starting at x=0).
    y(x) = m*x + b + A * exp(-x / tau)
    """
    return (m * x + b) + A * np.exp(-x / np.maximum(tau, 1e-12))


def fit_exponential_plus_line_fixed_line(n_droplets):
    """
    Fit A and tau while keeping m and b fixed from the second-half linear fit.
    Uses weighted least-squares with weights = 1/sqrt(y) (safe for zeros).
    Returns
    -------
    y_model : np.ndarray
        Best-fit model over all x.
    params : dict
        {'m':..., 'b':..., 'A':..., 'tau':..., 'success': bool}
    """
    x = np.arange(len(n_droplets))
    y = n_droplets.astype(float)
    y_line, (m, b) = fit_linear_model_second_half(y)

    # Initial guesses (as requested)
    tau0 = np.average(x, weights=np.maximum(y, 1e-12))  # weighted mean
    A0 = float(np.sum(y))  # amplitude guess as the sum of the distribution

    def residuals(theta):
        A, tau = theta
        y_model = line_plus_exponential(x, m, b, A, tau)
        yerr = np.sqrt(np.maximum(y, 1.0))  # avoid zeros
        return (y - y_model) / yerr

    tau_upper = max(5.0, x.max() * 10.0 + 10.0)
    bounds = ([0.0, 1e-6], [np.inf, tau_upper])

    res = least_squares(
        residuals,
        x0=[A0, tau0],
        bounds=bounds,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
        max_nfev=2000,
    )

    A_fit, tau_fit = res.x
    y_model = line_plus_exponential(x, m, b, A_fit, tau_fit)
    return y_model, {"m": m, "b": b, "A": A_fit, "tau": tau_fit, "success": res.success}


def fit_exponential_plus_line_joint(n_droplets):
    """
    Fit (m, b, A, tau) jointly via weighted least-squares.
    Returns
    -------
    y_model : np.ndarray
        Best-fit model over all x.
    params : dict
        {'m':..., 'b':..., 'A':..., 'tau':..., 'success': bool}
    """
    x = np.arange(len(n_droplets))
    y = n_droplets.astype(float)

    # Initial guesses
    y_line0, (m0, b0) = fit_linear_model_second_half(y)
    tau0 = np.average(x, weights=np.maximum(y, 1e-12))  # weighted mean
    A0 = float(np.sum(y))  # amplitude guess

    def residuals(theta):
        m, b, A, tau = theta
        y_model = line_plus_exponential(x, m, b, A, tau)
        yerr = np.sqrt(np.maximum(y, 1.0))
        return (y - y_model) / yerr

    tau_upper = max(5.0, x.max() * 10.0 + 10.0)
    bounds_lower = [-np.inf, -np.inf, 0.0, 1e-6]
    bounds_upper = [np.inf, np.inf, np.inf, tau_upper]

    res = least_squares(
        residuals,
        x0=[m0, b0, A0, tau0],
        bounds=(bounds_lower, bounds_upper),
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
        max_nfev=5000,
    )

    m, b, A, tau = res.x
    y_model = line_plus_exponential(x, m, b, A, tau)
    return y_model, {"m": m, "b": b, "A": A, "tau": tau, "success": res.success}


def threshold_2sigma_from_line_right_to_left(n_droplets, y_line, k_sigma=2.0, two_sided=False):
    """
    Determine the threshold by scanning from right to left and returning the first index
    where the data deviates from the linear baseline by >= k_sigma * errorbar.

    Errorbar matches the plot convention: yerr = sqrt(max(y, 1.0)).

    Parameters
    ----------
    n_droplets : array-like
        Observed counts.
    y_line : array-like
        Linear baseline evaluated over the full x range.
    k_sigma : float
        Number of error bars (default 2.0).
    two_sided : bool
        If True, uses absolute deviation; otherwise uses upward-only deviation.

    Returns
    -------
    idx : int
        Index of the first crossing encountered scanning from right to left.
        If none found, returns a fallback based on the elbow heuristic.
    """
    y = n_droplets.astype(float)
    yerr = np.sqrt(np.maximum(y, 1.0))
    if two_sided:
        mask = np.abs(y - y_line) >= k_sigma * yerr
    else:
        mask = (y - y_line) >= k_sigma * yerr  # upward-only

    # scan from right to left; return the first index where the condition holds
    for i in range(len(y) - 1, -1, -1):
        if mask[i]:
            return i

    # Fallback: elbow by max first difference (previous behavior)
    diffs = np.diff(n_droplets)
    return 0 if len(diffs) == 0 else int(np.argmax(diffs) + 1)


def plot_droplets_vs_distance(data=None, filename="droplets_vs_distance.npy", show_plot=True, fit_mode="joint",
                              k_sigma=2.0, two_sided=False):
    """
    Load droplets_vs_distance.npy and plot number of droplets vs distance threshold.
    Fit line + exponential and compute threshold where, scanning right->left,
    the data is first >= k_sigma errorbars from the linear baseline.

      - fit_mode="fixed_line" -> fit only (A, tau) with (m, b) fixed from 2nd half
      - fit_mode="joint"      -> fit (m, b, A, tau) jointly (default)
    """
    if data is not None:
        n_droplets = data
    else:
        n_droplets = np.load(filename)
    x = np.arange(len(n_droplets))

    # Linear fit for baseline and for visualization
    y_line, (m_lin, b_lin) = fit_linear_model_second_half(n_droplets)

    # Fit the exponential + line model (for plotting & diagnostics)
    if fit_mode == "fixed_line":
        y_model, params = fit_exponential_plus_line_fixed_line(n_droplets)
        n_params = 2
    else:
        y_model, params = fit_exponential_plus_line_joint(n_droplets)
        n_params = 4

    # Compute the new threshold
    thr_idx = threshold_2sigma_from_line_right_to_left(
        n_droplets, y_line, k_sigma=float(k_sigma), two_sided=bool(two_sided)
    )

    if show_plot:
        fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                gridspec_kw={'height_ratios': [2, 1]})
        ax1, ax2 = axs

        y = n_droplets.astype(float)
        yerr = np.sqrt(np.maximum(y, 1.0))

        ax1.errorbar(x, y, yerr=yerr, fmt='o', ls='-', label='Data', capsize=3)
        ax1.step(x, y, where='mid', color='tab:blue', alpha=0.7)

        ax1.plot(x, y_line, 'r--', label='Linear fit (2nd half, extrapolated)')
        ax1.plot(x, y_model, 'g-', label=f'Line + Exponential (τ={params["tau"]:.2f}, A={params["A"]:.1f})')

        # Mark the threshold
        ax1.axvline(thr_idx, color='k', linestyle='--', alpha=0.8,
                    label=f'2σ-from-line threshold @ {thr_idx}')
        ax1.plot([thr_idx], [y[thr_idx]], 'ko')

        ax1.set_ylabel('Number of Droplets Tracked')
        ax1.set_title('Droplets Tracked vs Linking Distance Threshold')
        ax1.grid(True)
        ax1.legend()

        # Residuals w.r.t. the model and reduced chi^2
        residuals = (y - y_model) / yerr
        dof = max(len(y) - n_params, 1)
        chi2 = np.sum(residuals ** 2)
        red_chi2 = chi2 / dof
        ax2.plot(x, residuals, marker='s', color='tab:purple',
                 label=f'Normalized residuals\nReduced $\\chi^2$ = {red_chi2:.2f}')
        ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax2.set_xlabel('Distance Threshold for Linking')
        ax2.set_ylabel('Residual/Error')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    return int(thr_idx), int(n_droplets[thr_idx])


if __name__ == "__main__":
    # === USER SETTINGS ===
    VIDEO_BASENAME = "water_constexp2"
    import os as _os
    _root = _os.path.dirname(_os.path.abspath(__file__))
    _filename = _os.path.join(_root, "data", VIDEO_BASENAME, f"{VIDEO_BASENAME}_droplets_vs_distance.npy")

    best_threshold, n_tracked = plot_droplets_vs_distance(
        filename=_filename, fit_mode="joint", k_sigma=2.0, two_sided=False
    )
    print(f"Suggested best threshold: {best_threshold} (tracks {n_tracked} droplets)")
