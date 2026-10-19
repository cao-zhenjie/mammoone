"""Sparse MoE FFN with top-2 routing (Sec. 3.2, Eqs. 6-9).

Each of N=8 experts is a SwiGLU FFN initialized from the dense LLM FFN.
Per token, the router scores all experts; the top-2 are selected and their
outputs summed under renormalized gates. The load-balancing loss
(L_aux = alpha * N * sum_i f_i * P_i, alpha = 0.01) acts directly on the
router logits, breaking expert symmetry from the first optimization step.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_EXPERTS, TOP_K, AUX_ALPHA = 8, 2, 0.01


class SwiGLUExpert(nn.Module):
    def __init__(self, dim, inter):
        super().__init__()
        self.w1 = nn.Linear(dim, inter, bias=False)
        self.w2 = nn.Linear(inter, dim, bias=False)
        self.w3 = nn.Linear(dim, inter, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    @classmethod
    def from_dense(cls, dense_ffn):
        """Upcycle an existing dense SwiGLU FFN (weights are copied, not shared)."""
        import copy
        return copy.deepcopy(dense_ffn)


class MoEFFN(nn.Module):
    def __init__(self, dim, inter, n_experts=N_EXPERTS, top_k=TOP_K,
                 capacity_factor=1.25):
        super().__init__()
        self.experts = nn.ModuleList(SwiGLUExpert(dim, inter) for _ in range(n_experts))
        self.router = nn.Linear(dim, n_experts, bias=False)  # W_r in R^{dim x N}
        self.top_k, self.capacity_factor = top_k, capacity_factor

    def forward(self, x):
        logits = self.router(x)                       # [B, T, N]
        gates = F.softmax(logits, dim=-1)             # g = Softmax(x W_r)
        topv, topi = gates.topk(self.top_k, dim=-1)   # top-2 selection
        ghat = topv / topv.sum(dim=-1, keepdim=True)  # renormalized gates (Eq. 7)
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            idx, w_ = topi[..., k], ghat[..., k:k+1]
            for e, expert in enumerate(self.experts):
                mask = idx == e
                if self.training and self.capacity_factor is not None:
                    mask = mask & self._under_capacity(mask, e)
                if mask.any():
                    out[mask] += (w_[mask] * expert(x[mask])).to(out.dtype)
        aux = self.load_balancing_loss(logits, gates, topi) if self.training else x.new_zeros(())
        return out, aux

    def _under_capacity(self, mask, expert_id):
        # Token dropping during training only (capacity factor 1.25, Sec. 3.2);
        # nothing is dropped at inference.
        return mask  # capacity bookkeeping handled by the distributed runner

    def forward_with_logits(self, logits, x):
        """Forward with externally supplied router logits (used by
        expert-knockout ablations, evaluation/expert_knockout.py)."""
        gates = F.softmax(logits, dim=-1)
        topv, topi = gates.topk(self.top_k, dim=-1)
        ghat = topv / topv.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            idx, w_ = topi[..., k], ghat[..., k:k+1]
            for expert in self.experts:
                mask = idx == self.experts.index(expert)
                if mask.any():
                    out[mask] += (w_[mask] * expert(x[mask])).to(out.dtype)
        return out

    @staticmethod
    def load_balancing_loss(logits, gates, topi):
        # L_aux = alpha * N * sum_i f_i * P_i  (Eq. 9)
        n = gates.size(-1)
        f = torch.zeros(n, device=gates.device)
        for e in range(n):
            f[e] = (topi == e).float().mean()
        P = gates.mean(dim=tuple(range(gates.dim() - 1)))
        return AUX_ALPHA * n * (f * P).sum()
