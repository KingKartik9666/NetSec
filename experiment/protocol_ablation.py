"""Test observed protocol identities on unchanged DataSense row partitions."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiment.refine_candidates import execute
from experiment.train_candidates import ROOT, save_registry
from framework.training.tracking import prepared_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "artifacts/training/datasense_gpu_refinement")
    parser.add_argument("--prepared", type=Path, default=ROOT / "dataset/prepared/datasense_protocols_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/training/datasense_protocols_v1")
    args = parser.parse_args()
    original = json.loads((args.base / "experiments.json").read_text())
    prepared = args.prepared.resolve()
    manifest = json.loads((prepared / "manifest.json").read_text())
    candidates = ["random_forest", "xgboost", "kan", "hem"]
    anchors = {name: max([r for r in original if r["candidate"] == name and r["seed"] == 42 and r["status"] == "completed"], key=lambda r: r["metrics"]["macro_f1"]) for name in candidates}
    for anchor in anchors.values():
        parent = Path(anchor["prepared"])
        for split in ["train", "validation", "test"]:
            if not np.array_equal(np.load(parent / f"row_ids_{split}.npy"), np.load(prepared / f"row_ids_{split}.npy")):
                raise ValueError("Feature comparison requires identical partition row IDs")
        if anchor["dataset_sha256"] != manifest["source_sha256"]:
            raise ValueError("Feature comparison source data differs")
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    specs = {name: {"id": "protocol_presence", "parameters": anchor["config"],
                    "purpose": f"Add train-vocabulary protocol identities at fixed parameters from {anchor['run_id']}",
                    "changes": {"feature_strategy": "numeric plus observed protocol presence"}} for name, anchor in anchors.items()}
    plan = {"anchors": {k: v["run_id"] for k, v in anchors.items()}, "specs": specs, "source_sha256": manifest["source_sha256"], "test_evaluated": False}
    plan.update(prepared_identity(prepared))
    plan_file = output / "feature_plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise ValueError("Feature plan changed; use another output directory")
    write_json(plan_file, plan)
    records = json.loads((output / "experiments.json").read_text()) if (output / "experiments.json").exists() else []
    for name in candidates:
        execute(output, prepared, name, specs[name], 42, "search", records)
    # One stronger XGBoost capacity setting follows the representation change.
    extra = {"id": "protocol_capacity", "parameters": {**anchors["xgboost"]["config"], "max_depth": 8, "n_estimators": 300, "lr": .08, "early_stopping_rounds": 25},
             "purpose": "Test additional capacity after adding observable protocol features", "changes": {"max_depth": 8, "n_estimators": 300, "lr": .08}}
    execute(output, prepared, "xgboost", extra, 42, "search", records)
    for name in candidates:
        trials = [r for r in records if r["candidate"] == name and r["seed"] == 42 and r["status"] == "completed"]
        if not trials:
            continue
        best = max(trials, key=lambda r: r["metrics"]["macro_f1"])
        if best["metrics"]["macro_f1"] > anchors[name]["metrics"]["macro_f1"]:
            spec = extra if best["config_id"] == "protocol_capacity" else specs[name]
            for seed in [123, 2025]:
                execute(output, prepared, name, spec, seed, "repeat_selected", records)
    save_registry(output, records)
    text = "# Protocol identity feature experiment\n\nSame rows, labels and split; vocabulary fit on training only. Test remains locked.\n\n| Model | Numeric anchor macro F1 | Protocol macro F1 | Numeric accuracy | Protocol accuracy |\n|---|---:|---:|---:|---:|\n"
    for name in candidates:
        trials = [r for r in records if r["candidate"] == name and r["seed"] == 42 and r["config_id"] == "protocol_presence" and r["status"] == "completed"]
        if trials:
            old, new = anchors[name]["metrics"], trials[0]["metrics"]
            text += f"| {name} | {old['macro_f1']:.5f} | {new['macro_f1']:.5f} | {old['accuracy']:.5f} | {new['accuracy']:.5f} |\n"
    (output / "feature_effects.md").write_text(text, encoding="utf-8")
    print(f"PROTOCOL ABLATION COMPLETE: {output}; test remains locked", flush=True)
    return int(any(r["status"] != "completed" for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
