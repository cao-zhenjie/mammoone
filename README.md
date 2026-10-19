# MammoOne

Official implementation of **"MammoOne: A Large Multimodal Model with Task-Adaptive Experts for Breast Cancer Diagnosis"** (under review at Medical Image Analysis).

MammoOne transforms the dense FFNs of a Qwen3-8B backbone into sparse mixture-of-experts modules (8 experts, top-2 routing, 46.8B total / 13.6B activated parameters per token), and unifies malignancy classification, ordinal-aware continuous BI-RADS severity regression, and visual question answering in a four-stage training recipe.

## Repository layout
- `configs/` — stage-wise training configurations (Stages I–IV)
- `models/moe/` — MoE FFN conversion, router, load-balancing loss
- `models/heads/` — diagnostic tokens, classification / severity / VQA / MAE-reconstruction heads
- `training/` — four-stage training pipeline (Stage I anatomical MAE pretraining, Stage II multi-task SFT, Stage III synergistic alignment with ordinal-consistency loss, Stage IV deep-ensemble adaptation)
- `evaluation/` — frozen patient-level evaluation protocol, bootstrap CIs, DeLong tests, calibration (ECE/Brier, temperature scaling), subgroup and expert-knockout analyses
- `inference/` — expert-sharding serving (bf16 active experts on-device, inactive experts off-device/INT8), two-stage ROI procedure, peak-memory measurement script
- `scripts/` — data preparation and split-freezing utilities

## Status
Code is being synchronized with the camera-ready revision and will be uploaded here. Review-time access can be granted by the corresponding author upon request.

## Data
The in-house cohort cannot be publicly released due to patient privacy and institutional data-governance requirements. INbreast, CBIS-DDSM, CMMD, and KAU-BCMD are available from their respective distributors.
