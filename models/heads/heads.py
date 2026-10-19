"""Diagnostic tokens and task heads (Sec. 3.3).

Four learnable diagnostic tokens condition the backbone on the task:
[CLS_MALIGNANCY] malignancy classification, [REG_SEVERITY] continuous
BI-RADS severity on [1,5], [QA_VQA] question answering,
[REC_ANATOMY] anatomical reconstruction (Stage I only).
"""
import torch
import torch.nn as nn

DIAGNOSTIC_TOKENS = ["[CLS_MALIGNANCY]", "[REG_SEVERITY]",
                     "[QA_VQA]", "[REC_ANATOMY]"]


class DiagnosticTokens(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.tokens = nn.Parameter(
            torch.randn(len(DIAGNOSTIC_TOKENS), dim) * 0.02)

    def forward(self, batch_size):
        return self.tokens.unsqueeze(0).expand(batch_size, -1, -1)


class ClassificationHead(nn.Module):
    """Two-layer MLP: W1 in R^{512x4096}, W2 in R^{2x512}; GELU +
    dropout (p=0.1). Trained with class-weighted cross-entropy
    (inverse class frequencies of the training set)."""

    def __init__(self, dim=4096, hidden=512, n_classes=2, p_drop=0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, n_classes)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p_drop)

    def forward(self, h):
        return self.w2(self.drop(self.act(self.w1(h))))  # logits


class SeverityHead(nn.Module):
    """MLP followed by sigmoid scaling: s_hat = 1 + 4 * sigmoid(MLP(h0)),
    a continuous score on the ordinal [1,5] scale (BI-RADS 0/6 excluded).
    Trained with L_severity = L1 + lambda_ord * L_ord (Eq. 14)."""

    def __init__(self, dim=4096, hidden=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, h):
        return 1.0 + 4.0 * torch.sigmoid(self.mlp(h).squeeze(-1))


class ReconstructionHead(nn.Module):
    """MAE decoder for Stage-I anatomical pretraining (Sec. 3.3):
    reconstructs frozen DINOv2 features of the 40% randomly masked
    global mammogram tokens (ROI tokens excluded).
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
