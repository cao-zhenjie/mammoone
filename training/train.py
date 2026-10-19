"""Four-stage training driver for MammoOne (Sec. 3.5, Table 2).

Stages:
  1  MAE pretraining on in-house cohort (frozen text tower inputs optional)
  2  Joint multi-task tuning (classification / severity / VQA / MAE)
  3  Curriculum: MAE weight annealed to zero over the stage
  4  Deep-ensemble instantiation: five instances fine-tuned with
     different seeds and data orders; VQA branch adapted with LoRA
     (r=64, alpha=128, dropout=0.05 on q_proj/v_proj/lm_head).

Default hyperparameters (Table 2):
  AdamW(betas=(0.9, 0.95), weight_decay=0.1), cosine schedule with
  3% linear warmup, gradient clipping 1.0, bf16 with ZeRO-3,
  global batch 256, LR 1e-4 / 1e-4 / 5e-5 / 2e-5 for stages 1-4,
  30K / 30K / 10K / 4K steps respectively.
"""
import argparse
import math
import random

import torch

from training.losses import stage_weights

STAGE_CFG = {
    1: dict(lr=1e-4, steps=30_000, frac=0.0),
    2: dict(lr=1e-4, steps=30_000, frac=0.0),
    3: dict(lr=5e-5, steps=10_000, frac=0.0),
    4: dict(lr=2e-5, steps=4_000,  frac=0.0),
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

    Dense ViT weights are loaded from --init-ckpt and upcycled to MoE
    (models.moe.convert.upcycle_to_moe) before stage 1.
    """
    raise NotImplementedError(
        "Wire up the backbone checkpoint path for your environment."
    )


def build_loaders(args):
    """Patient-disjoint loaders per split manifest (data/splits.py)."""
    raise NotImplementedError


def train_stage(stage, model, loaders, device):
    cfg = STAGE_CFG[stage]
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"],
        betas=(0.9, 0.95), weight_decay=0.1,
    )
    model.train()
    for step in range(cfg["steps"]):
        lr = cosine_lr(step, cfg["steps"], cfg["lr"])
        for g in opt.param_groups:
            g["lr"] = lr
        frac = step / max(1, cfg["steps"] - 1)
        w = stage_weights(stage, frac)
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
                    help="ensemble member seed (stage 4)")
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
