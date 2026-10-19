"""Evaluation entry points: discrimination and ordinal severity metrics.

- Malignancy AUC with patient-level 1000-sample bootstrap 95% CIs;
  DeLong test for pairwise model comparisons.
- BI-RADS severity: MAE, quadratic-weighted kappa (QWK), within-one
  accuracy on the [1,5] ordinal continuum.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def patient_bootstrap_auc(y_true, y_score, patient_ids, n_boot=1000,
                          seed=0):
    """Cluster bootstrap at the patient level (resample patients)."""
    rng = np.random.default_rng(seed)
    patients = np.unique(patient_ids)
    aucs = []
    for _ in range(n_boot):
        draw = rng.choice(patients, size=len(patients), replace=True)
        idx = np.isin(patient_ids, draw)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return roc_auc_score(y_true, y_score), (lo, hi)


def quadratic_weighted_kappa(y_true, y_pred, n_cat=5):
    """QWK over BI-RADS categories 1..5."""
    O = np.zeros((n_cat, n_cat))
    for t, p in zip(y_true, y_pred):
        O[int(round(t)) - 1, int(round(p)) - 1] += 1
    W = (np.subtract.outer(np.arange(n_cat), np.arange(n_cat)) ** 2
         ) / (n_cat - 1) ** 2
    hist_t = O.sum(1) / O.sum()
    hist_p = O.sum(0) / O.sum()
    E = np.outer(hist_t, hist_p) * O.sum()
    return 1.0 - (W * O).sum() / (W * E).sum()


def severity_metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))
    qwk = quadratic_weighted_kappa(y_true, y_pred)
    within1 = float(np.mean(np.abs(np.asarray(y_pred)
                                   - np.asarray(y_true)) <= 1.0))
    return dict(MAE=mae, QWK=qwk, within_one=within1)


def delong_test(y_true, score_a, score_b):
    """DeLong test for the difference of two correlated AUCs (Sec. 4.3).
    Returns (auc_a, auc_b, two-sided p) with H0: equal AUC."""
    from scipy.stats import norm
    y_true = np.asarray(y_true)
    xa, xb = np.asarray(score_a)[y_true == 1], np.asarray(score_b)[y_true == 1]
    ya, yb = np.asarray(score_a)[y_true == 0], np.asarray(score_b)[y_true == 0]
    m, n = len(xa), len(ya)

    def psi(p, q):
        return (p > q).astype(float) + 0.5 * (p == q)

    def components(x, y):
        v10 = psi(x[:, None], y[None, :]).mean(axis=1)  # per positive
        v01 = psi(x[:, None], y[None, :]).mean(axis=0)  # per negative
        return v10, v01

    va10, va01 = components(xa, ya)
    vb10, vb01 = components(xb, yb)
    auc_a, auc_b = va10.mean(), vb10.mean()
    c10 = np.cov(np.vstack([va10, vb10]))
    c01 = np.cov(np.vstack([va01, vb01]))
    var = (c10[0, 0] + c10[1, 1] - 2 * c10[0, 1]) / m \
        + (c01[0, 0] + c01[1, 1] - 2 * c01[0, 1]) / n
    z = (auc_a - auc_b) / np.sqrt(max(var, 1e-12))
    return float(auc_a), float(auc_b), float(2 * norm.sf(abs(z)))
