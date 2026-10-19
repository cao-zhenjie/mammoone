"""Calibration and selective-prediction evaluation.

ECE (15 equal-width bins), Brier score, temperature scaling fitted on
the validation split, and risk-coverage curves for selective
prediction driven by deep-ensemble disagreement.
"""
import numpy as np


def expected_calibration_error(y_true, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y_true[m].mean())
    return float(ece)


def brier(y_true, p):
    return float(np.mean((p - y_true) ** 2))


def fit_temperature(logits_val, y_val, grid=None):
    """Scalar temperature minimizing validation NLL."""
    if grid is None:
        grid = np.linspace(0.5, 5.0, 91)
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        p = 1.0 / (1.0 + np.exp(-logits_val / t))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        nll = -np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))
        if nll < best_nll:
            best_nll, best_t = nll, t
    return float(best_t)


def risk_coverage(y_true, p_mean, p_std):
    """Sweep uncertainty threshold; return coverage, selective risk."""
    cov, risk = [], []
    for tau in np.quantile(p_std, np.linspace(0, 1, 101)):
        keep = p_std <= tau
        if keep.sum() < 10:
            continue
        cov.append(float(keep.mean()))
        risk.append(float(np.mean((p_mean[keep] > 0.5) != y_true[keep])))
    return np.array(cov), np.array(risk)
