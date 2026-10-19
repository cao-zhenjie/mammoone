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
