"""Expert-knockout ablation: zero selected experts' gates at inference
and renormalize the remaining top-k weights (Sec. 4.x)."""
import torch


@torch.no_grad()
def knockout_forward(moe_ffn, x, knocked_out):
    """moe_ffn: models.moe.moe_ffn.MoEFFN. knocked_out: iterable of
    expert indices whose router logits are set to -inf."""
    logits = moe_ffn.router(x)
    for k in knocked_out:
        logits[..., k] = float("-inf")
    return moe_ffn.forward_with_logits(logits, x)
