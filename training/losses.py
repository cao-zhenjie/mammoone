"""Composite training objectives (Sec. 3.3, 3.4).

Stage I   : mae 1.0
Stage II  : cls/sev/vqa 1.0, mae 0.5
Stage III : lambda_i(t) curriculum shifting weight from MAE to clinical tasks
All stages: + load-balancing aux loss (weight 0.01, Eq. 9).
Severity  : L_severity = L_sev + lambda_ord * L_ord, lambda_ord = 0.3 (Eq. 14).
"""
import torch
import torch.nn.functional as F

LAMBDA_ORD, ORD_MARGIN = 0.3, 0.1


def ordinal_consistency_loss(scores, targets):
    """Pairwise hinge on within-batch study pairs (Eq. 13):
    L_ord = mean over P of max(0, s_i - s_j + delta), P = {(i,j): s_i* < s_j*}."""
    s, t = scores.view(-1), targets.view(-1)
    si, sj = s.unsqueeze(0), s.unsqueeze(1)
    ti, tj = t.unsqueeze(0), t.unsqueeze(1)
    pair_mask = ti < tj
    if not pair_mask.any():
        return scores.new_zeros(())
    viol = F.relu(si - sj + ORD_MARGIN)
    return viol[pair_mask].mean()


def severity_loss(scores, targets):
    return F.l1_loss(scores, targets) + LAMBDA_ORD * ordinal_consistency_loss(scores, targets)


def mae_loss(z_hat, z):
    return F.mse_loss(z_hat, z)


def stage_weights(stage: int, step: int, total_steps: int) -> dict:
    """lambda_i(t) curriculum (Table 2)."""
    if stage == 1:
        return {"mae": 1.0}
    if stage == 2:
        return {"cls": 1.0, "sev": 1.0, "vqa": 1.0, "mae": 0.5}
    frac = step / max(total_steps, 1)
    return {"mae": 0.5 * (1 - frac), "cls": 1.0, "sev": 1.0, "vqa": 1.0}
