"""Memory-efficient serving via expert sharding.

Active (top-2 routed) experts are kept on GPU in bf16; non-active
experts are offloaded to CPU RAM as INT8 and dequantized on demand.
This is the protocol behind the Table 15 memory budget.
"""
import torch


def quantize_int8(weight):
    scale = weight.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    q = torch.round(weight / scale).to(torch.int8)
    return q, scale


def dequantize_int8(q, scale):
    return q.to(torch.float32) * scale


class ExpertShard:
    """One offloaded expert: INT8 weights on CPU, bf16 compute on GPU."""

    def __init__(self, expert):
        self.qparams = {}
        for name, p in expert.named_parameters():
            self.qparams[name] = quantize_int8(p.detach().cpu())

    def load_bf16(self, device):
        return {n: dequantize_int8(q, s).to(device).to(torch.bfloat16)
                for n, (q, s) in self.qparams.items()}
