"""Subgroup analysis: per-stratum delta-AUC vs. the dense control with
patient-level bootstrap CIs (reported in Fig. 8 / Table 9)."""
import numpy as np
from sklearn.metrics import roc_auc_score


def subgroup_delta_auc(y_true, p_ours, p_dense, patient_ids, mask,
                       n_boot=1000, seed=0):
    """AUC(ours) - AUC(dense) within the subgroup defined by mask."""
    y, a, b = y_true[mask], p_ours[mask], p_dense[mask]
    pid = patient_ids[mask]
    rng = np.random.default_rng(seed)
    patients = np.unique(pid)
    deltas = []
    for _ in range(n_boot):
        draw = rng.choice(patients, size=len(patients), replace=True)
        idx = np.isin(pid, draw)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], a[idx])
                      - roc_auc_score(y[idx], b[idx]))
    point = roc_auc_score(y, a) - roc_auc_score(y, b)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return point, (lo, hi)
