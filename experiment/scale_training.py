"""Test the leading family on every eligible training row, with fixed evaluation IDs."""
import json
from pathlib import Path

import numpy as np

from experiment.refine_candidates import execute
from experiment.train_candidates import ROOT, save_registry
from framework.training.tracking import prepared_identity, write_json


def main():
    base = ROOT / "artifacts/training/datasense_protocols_v1"
    prepared = ROOT / "dataset/prepared/datasense_full_protocols"
    output = ROOT / "artifacts/training/datasense_full_training"
    original = json.loads((base / "experiments.json").read_text())
    anchor = max([r for r in original if r["candidate"] == "xgboost" and r["seed"] == 42 and r["status"] == "completed"], key=lambda r: r["metrics"]["macro_f1"])
    parent = Path(anchor["prepared"])
    for split in ["validation", "test"]:
        assert np.array_equal(np.load(parent / f"row_ids_{split}.npy"), np.load(prepared / f"row_ids_{split}.npy"))
        assert (parent / f"y_{split}.npy").read_bytes() == (prepared / f"y_{split}.npy").read_bytes()
    if not np.isin(np.load(parent / "row_ids_train.npy"), np.load(prepared / "row_ids_train.npy")).all():
        raise ValueError("Expanded training data must contain the original training sample")
    specs = [
        {"id": "full_rows", "parameters": anchor["config"], "purpose": "Use every eligible training row at fixed parameters and evaluation IDs", "changes": {"training_rows": "all eligible"}},
        {"id": "full_sqrt_weights", "parameters": {**anchor["config"], "weight_strategy": "sqrt_balanced"}, "purpose": "Check weighting with expanded training data", "changes": {"training_rows": "all eligible", "weight_strategy": "sqrt_balanced"}},
    ]
    output.mkdir(parents=True, exist_ok=True)
    plan = {"anchor": anchor["run_id"], "specs": specs, "test_evaluated": False, "evaluation_ids_unchanged": True}
    plan.update(prepared_identity(prepared))
    plan_file = output / "scale_plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise ValueError("Scaling plan changed")
    write_json(plan_file, plan)
    registry = output / "experiments.json"
    records = json.loads(registry.read_text()) if registry.exists() else []
    for spec in specs:
        execute(output, prepared, "xgboost", spec, 42, "search", records)
    selected = max([r for r in records if r["seed"] == 42 and r["status"] == "completed"], key=lambda r: r["metrics"]["macro_f1"])
    spec = next(s for s in specs if s["id"] == selected["config_id"])
    for seed in [123, 2025]:
        execute(output, prepared, "xgboost", spec, seed, "repeat_selected", records)
    save_registry(output, records)
    print(f"FULL TRAINING STUDY COMPLETE: {output}; test remains locked", flush=True)
    return int(any(r["status"] != "completed" for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
