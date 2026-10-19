"""Composite training objectives (Sec. 3.3, 3.4).

Stage I   : mae 1.0
Stage II  : cls/sev/vqa 1.0, mae 0.5
Stage III : lambda_i(t) linear curriculum, renormalized each step
            (mae 0.40->0.15, cls 0.20->0.30, sev 0.25->0.35,
             vqa 0.15->0.30), shifting capacity from representation
            learning to clinical tasks.
Stage IV  : Stage-III setting repeated with five seeds.
All stages: + load-balancing aux loss (weight 0.01, Eq. 9).
Severity  : L_severity = L1 + lambda_ord * L_ord (Eq. 14, lambda=0.3).
"""
import torch
import torch.nn.functional as F

LAMBDA_ORD, ORD_MARGIN = 0.3, 0.1

# Stage-III curriculum endpoints (start, end); renormalized per step.
CURRICULUM = {"mae": (0.40, 0.15), "cls": (0.20, 0.30),
              "sev": (0.25, 0.35), "vqa": (0.15, 0.30)}


def ordinal_consistency_loss(scores, targets):
    """Pairwise hinge on within-batch study pairs (Eq. 13):
    L_ord = mean_P max(0, s_i - s_j + delta), P = {(i,j): s_i* < s_j*}."""
    s, t = scores.view(-1), targets.view(-1)
    si, sj = s.unsqueeze(0), s.unsqueeze(1)
    ti, tj = t.unsqueeze(0), t.unsqueeze(1)
    pair_mask = ti < tj
    if not pair_mask.any():
        return scores.new_zeros(())
    viol = F.relu(si - sj + ORD_MARGIN)
    return viol[pair_mask].mean()


def severity_loss(scores, targets):
    return (F.l1_loss(scores, targets)
            + LAMBDA_ORD * ordinal_consistency_loss(scores, targets))


def mae_loss(z_hat, z):
    return F.mse_loss(z_hat, z)


def stage_weights(stage: int, step: int, total_steps: int) -> dict:
    """lambda_i(t) loss weights (Table 2)."""
    if stage == 1:
        return {"mae": 1.0}
    if stage == 2:
        return {"cls": 1.0, "sev": 1.0, "vqa": 1.0, "mae": 0.5}
    t = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    raw = {k: a + (b - a) * t for k, (a, b) in CURRICULUM.items()}
    z = sum(raw.values())
    return {k: v / z for k, v in raw.items()}  # renormalized each step
