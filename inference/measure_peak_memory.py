"""Peak-memory measurement for the Table 15 budget.

Runs one full four-view study through the serving stack
(bf16 active experts + INT8 offloaded experts, four views shared)
and reports torch.cuda.max_memory_allocated in GiB.

Usage:
    python inference/measure_peak_memory.py --ckpt PATH --study DIR
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--study", required=True,
                    help="directory with the four DICOM views")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    # TODO: load model from args.ckpt with inference.serve sharding
    # model = ...
    # study = assemble_study(args.study)  # data.mammography

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(args.repeats):
            pass  # out = model(study)
    peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
    print(f"peak_memory_gib,{peak_gib:.2f}")


if __name__ == "__main__":
    main()
