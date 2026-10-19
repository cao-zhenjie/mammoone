"""Dense-to-MoE conversion of a pretrained LLM backbone (Sec. 3.2).

Every FFN in the Qwen3-8B trunk is replaced by an MoEFFN whose eight
experts are all initialized from the original dense FFN. Symmetry is broken
by the load-balancing loss acting on router logits and by the randomly
initialized router (Sec. 3.2, symmetry discussion; Fig. 5).
"""
import torch.nn as nn
from .moe_ffn import MoEFFN


def find_ffn_modules(model):
    for name, module in model.named_modules():
        if hasattr(module, "mlp"):
            yield name, module


def upcycle_to_moe(model, n_experts=8, top_k=2):
    n_converted = 0
    for _, block in find_ffn_modules(model):
        dense = block.mlp
        dim = dense.w1.in_features if hasattr(dense, "w1") else model.config.hidden_size
        inter = dense.w1.out_features if hasattr(dense, "w1") else model.config.intermediate_size
        block.mlp = MoEFFN(dim, inter, n_experts=n_experts, top_k=top_k)
        for expert in block.mlp.experts:
            expert.load_state_dict(dense.state_dict())
        nn.init.normal_(block.mlp.router.weight, std=0.02)
        n_converted += 1
    return model, n_converted


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    return {"total_B": round(total / 1e9, 1)}  # 46.8B total; 13.6B active per token (top-2 of 8)
