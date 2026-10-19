"""Patient-level frozen splits (Sec. 4.1).

All partitions are at the patient level: every examination of one patient
(all views, laterality, follow-ups) falls into exactly one partition.
The manifest is frozen before any model development; every reported number
is produced under this single frozen partition.
"""
import hashlib, json


def patient_key(patient_id: str, salt: str = "mammoone-v1") -> float:
    h = hashlib.sha256(f"{salt}:{patient_id}".encode()).hexdigest()
    return int(h[:12], 16) / float(0xFFFFFFFFFFFF)


def assign_inhouse_split(patient_id: str) -> str:
    """Deterministic 8:1:1 train/val/test assignment."""
    u = patient_key(patient_id)
    if u < 0.8:
        return "train"
    return "val" if u < 0.9 else "test"


def assign_cmmd_split(patient_id: str) -> str:
    """Deterministic 80/20 assignment; the 20% is never used in any training stage."""
    return "pretrain80" if patient_key(patient_id, "cmmd-v1") < 0.8 else "heldout20"


def freeze_manifest(patient_ids, assign_fn, out_path):
    manifest = {pid: assign_fn(pid) for pid in sorted(patient_ids)}
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def load_manifest(path):
    with open(path) as f:
        return json.load(f)
