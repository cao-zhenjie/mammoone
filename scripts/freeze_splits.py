"""Freeze patient-level split manifests for all cohorts.

Writes a single JSON manifest (e.g. splits.json) consumed by
training/train.py and evaluation. Run once before training; the frozen
manifest must be kept with the evaluation protocol.
"""
import argparse
import csv

from data.splits import (assign_cmmd_split, assign_inhouse_split,
                         freeze_manifest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inhouse-csv", required=True,
                    help="CSV with a patient_id column")
    ap.add_argument("--cmmd-csv", default=None)
    ap.add_argument("--out", default="splits.json")
    args = ap.parse_args()

    manifest = {}
    with open(args.inhouse_csv) as f:
        ids = [row["patient_id"] for row in csv.DictReader(f)]
    manifest["inhouse"] = assign_inhouse_split(ids)
    if args.cmmd_csv:
        with open(args.cmmd_csv) as f:
            ids = [row["patient_id"] for row in csv.DictReader(f)]
        manifest["cmmd"] = assign_cmmd_split(ids)
    freeze_manifest(manifest, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
