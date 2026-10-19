"""Four-stage training driver for MammoOne (Sec. 3.4, Table 2).

  I   Anatomical pretraining (MAE)      LR 1e-4, 30K steps, mae 1.0
  II  Multi-task SFT                    LR 5e-5, 30K steps,
                                        cls/sev/vqa 1.0, mae 0.5
  III Synergistic alignment             LR 2e-5, 10K steps,
                                        lambda_i(t) curriculum
  IV  Deep ensemble (x5 seeds)          Stage-III setting repeated on
                                        top of the shared I-III trunk.

Common recipe (Sec. 4.2): AdamW(betas=(0.9, 0.95), weight_decay=0.1),
cosine decay with 3% linear warmup, gradient clipping 1.0, bf16 with
DeepSpeed ZeRO-3, global batch 256, one node of 8x A100-80GB.
"""
import argparse
import math
import random

import torch

from training.losses import stage_weights

STAGE_CFG = {
    1: dict(lr=1e-4, steps=30_000),
    2: dict(lr=5e-5, steps=30_000),
    3: dict(lr=2e-5, steps=10_000),
    4: dict(lr=2e-5, steps=10_000),  # repeats the Stage-III setting
}
GLOBAL_BATCH = 256
WARMUP_FRAC = 0.03
GRAD_CLIP = 1.0


def cosine_lr(step, total, base_lr, warmup_frac=WARMUP_FRAC):
    warm = int(total * warmup_frac)
    if step < warm:
        return base_lr * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


def build_model(args):
    """Instantiate backbone + MoE upcycling + diagnostic heads.

    Dense Qwen3-8B weights are loaded from --init-ckpt and every FFN is
    upcycled to MoE (models.moe.convert.upcycle_to_moe); attention,
    token embeddings, RoPE, and normalization layers remain frozen.
    """
    raise NotImplementedError(
        "Wire up the backbone checkpoint path for your environment."
    )


def build_loaders(args):
    """Patient-disjoint loaders per frozen split manifest
    (data/splits.py, scripts/freeze_splits.py)."""
    raise NotImplementedError


def train_stage(stage, model, loaders, device):
    cfg = STAGE_CFG[stage]
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"],
        betas=(0.9, 0.95), weight_decay=0.1)
    model.train()
    for step in range(cfg["steps"]):
        lr = cosine_lr(step, cfg["steps"], cfg["lr"])
        for g in opt.param_groups:
            g["lr"] = lr
        w = stage_weights(min(stage, 3), step, cfg["steps"])
        batch = next(loaders)
        losses = model.compute_losses(batch, weights=w)
        loss = sum(losses.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 100 == 0:
            msg = " ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
            print(f"[stage {stage}] step {step} lr={lr:.2e} {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=0,
                    help="ensemble member seed (Stage IV, five seeds)")
    ap.add_argument("--init-ckpt", type=str, default=None)
    ap.add_argument("--manifest", type=str, default="splits.json")
    ap.add_argument("--data-root", type=str, required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args)
    loaders = build_loaders(args)
    train_stage(args.stage, model, loaders, device)


if __name__ == "__main__":
    main()
