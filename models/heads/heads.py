"""Diagnostic tokens and task heads (Sec. 3.3).

Four learnable diagnostic tokens condition the backbone on the task:
[CLS_MAL] malignancy classification, [QA_VQA] question answering,
[REC_ANATOMY] anatomical reconstruction (Stage I only), [REG_SEV] severity.
"""
import torch
import torch.nn as nn

DIAGNOSTIC_TOKENS = ["[CLS_MAL]", "[QA_VQA]", "[REC_ANATOMY]", "[REG_SEV]"]


class DiagnosticTokens(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(len(DIAGNOSTIC_TOKENS), dim) * 0.02)

    def forward(self, batch_size):
        return self.tokens.unsqueeze(0).expand(batch_size, -1, -1)


class ClassificationHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def forward(self, h):
        return self.proj(h).squeeze(-1)  # malignancy logit


class SeverityHead(nn.Module):
    """Continuous BI-RADS severity on [1,5] (Sec. 3.3; 0/6 excluded)."""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def forward(self, h):
        return self.proj(h).squeeze(-1).clamp(1.0, 5.0)


class ReconstructionHead(nn.Module):
    """MAE decoder for Stage-I anatomical pretraining (Sec. 3.3):
    reconstructs frozen DINOv2 features of 40% masked global-view tokens.
    W1 in R^{2048x4096}, W2 in R^{768x2048}."""
    def __init__(self, dim=4096, hidden=2048, target_dim=768):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, target_dim)
        self.act = nn.GELU()

    def forward(self, h_masked):
        return self.w2(self.act(self.w1(h_masked)))


VQA_LORA = dict(rank=64, alpha=128, dropout=0.05,
                targets=["q_proj", "v_proj", "lm_head"])  # Sec. 3.3
