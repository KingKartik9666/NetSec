"""Evidence-guided second round: isolate parameters, then test interactions.

Uses validation only. The first-round configuration selected by seed-42 macro F1
is the anchor. All four single changes and the paired interaction are recorded
before their results are measured; after five trials, produce an effect report.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from experiment.train_candidates import ROOT, save_registry
from framework.training.tracking import choose_best, completed_trial, prepared_identity, write_json


def proposals(candidate, anchor):
    base = anchor["config"]
    if candidate == "random_forest":
        changes = [{"max_depth": int(base["max_depth"]) + 2}, {"num_parallel_tree": 96},
                   {"colsample_bynode": .6}, {"min_child_weight": 3}]
        paired = {**changes[0], **changes[1]}
    elif candidate == "xgboost":
        changes = [{"n_estimators": 240}, {"max_depth": int(base["max_depth"]) + 2},
                   {"lr": float(base["lr"]) / 2}, {"max_bin": 256}]
        paired = {**changes[0], **changes[2]}
    elif candidate == "kan":
        changes = [{"lr": float(base["lr"]) / 2}, {"hidden": [128, 64]},
                   {"grid_size": 7}, {"weight_strategy": "none"}]
        paired = {**changes[0], **changes[1]}
    else:
        changes = [{"pca_components": 8}, {"knn_k": 1},
                   {"knn_max_rows": 40000}, {"epochs": 20}]
        paired = {**changes[1], **changes[2]}
    specs = []
    for index, delta in enumerate(changes + [paired], 1):
        specs.append({"id": f"refine_{index}", "parameters": {**base, **delta}, "changes": delta,
                      "purpose": f"Controlled change from {anchor['config_id']}: {delta}" if index < 5 else f"Measured interaction from {anchor['config_id']}: {delta}"})
    return specs


def execute(output, prepared, candidate, spec, seed, phase, records):
    completed = completed_trial(records, candidate, spec["id"], seed, spec["parameters"], prepared)
    if completed:
        return completed
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{candidate}_{spec['id']}_s{seed}_{stamp}"
    directory = output / "runs" / run_id
    directory.mkdir(parents=True)
    request = {"run_id": run_id, "candidate": candidate, "config_id": spec["id"], "config": spec["parameters"],
               "purpose": spec["purpose"], "changes_from_anchor": spec.get("changes"), "seed": seed,
               "phase": phase, "prepared": str(prepared), "timestamp_utc": stamp}
    request.update(prepared_identity(prepared))
    request_file = directory / "request.json"
    write_json(request_file, request)
    env = {**os.environ, "PYTHONHASHSEED": str(seed), "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "4"}
    print(f"REFINE {candidate}/{spec['id']} seed={seed}: {spec.get('changes', 'repeat')}", flush=True)
    with (directory / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-u", "-m", "experiment.train_candidates", "--worker", str(request_file)],
                                   cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        for line in process.stdout:
            print(line, end="", flush=True); log.write(line); log.flush()
        process.wait()
    result_file = directory / "result.json"
    result = json.loads(result_file.read_text()) if result_file.exists() else {**request, "status": "failed", "error": f"Process exited {process.returncode}"}
    records.append(result)
    save_registry(output, records)
    best_path = output / "best" / f"{candidate}.json"
    best = None
    for row in records:
        if row["candidate"] == candidate:
            best = choose_best(best, row)
    if best is not None:
        write_json(best_path, best)
    return result


def effects_report(output, anchors, records):
    effects = []
    text = "# Optimization patterns after controlled trials\n\nAll effects below use seed 42 and the same validation split. They are local observations, not monotonic laws or final test results.\n\n"
    for candidate, anchor in anchors.items():
        text += f"## {candidate}\n\nAnchor: {anchor['config_id']}; macro F1 {anchor['metrics']['macro_f1']:.5f}, accuracy {anchor['metrics']['accuracy']:.5f}, FPR {anchor['metrics']['false_positive_rate']:.5f}.\n\n"
        text += "| Change | Macro F1 delta | Accuracy delta | FPR delta |\n|---|---:|---:|---:|\n"
        trials = [r for r in records if r["candidate"] == candidate and r["config_id"].startswith("refine_") and r["seed"] == 42 and r["status"] == "completed"]
        for r in trials:
            effect = {"candidate": candidate, "run_id": r["run_id"], "anchor_run_id": anchor["run_id"], "changes": r["changes_from_anchor"]}
            for metric in ["macro_f1", "accuracy", "false_positive_rate"]:
                effect[metric + "_delta"] = r["metrics"][metric] - anchor["metrics"][metric]
            effects.append(effect)
            text += f"| {r['changes_from_anchor']} | {effect['macro_f1_delta']:+.5f} | {effect['accuracy_delta']:+.5f} | {effect['false_positive_rate_delta']:+.5f} |\n"
    text += "\n## Accuracy target\n\nUser target: at least one model reaches 99.97% accuracy on the unchanged 84-class task. Validation attainment is not test attainment.\n\n"
    successful = [r for r in records if r["status"] == "completed"]
    best_accuracy = max(successful, key=lambda r: r["metrics"]["accuracy"])
    text += f"Highest measured validation accuracy: {best_accuracy['metrics']['accuracy']:.6%} ({best_accuracy['candidate']}, {best_accuracy['config_id']}, seed {best_accuracy['seed']}). Target reached on validation: {best_accuracy['metrics']['accuracy'] >= .9997}. Test remains locked.\n"
    write_json(output / "optimization_effects.json", effects)
    (output / "optimization_effects.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "artifacts/training/datasense_gpu_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/training/datasense_gpu_refinement")
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    base_records = json.loads((args.base / "experiments.json").read_text())
    study = json.loads((args.base / "study.json").read_text())
    prepared = Path(study["prepared"])
    if hashlib.sha256((prepared / "manifest.json").read_bytes()).hexdigest() != study["manifest_sha256"]:
        raise ValueError("Prepared manifest changed since the base study")
    anchors = {candidate: max([r for r in base_records if r["candidate"] == candidate and r["seed"] == 42 and r["status"] == "completed"], key=lambda r: r["metrics"]["macro_f1"])
               for candidate in study["candidates"]}
    plan = {"anchors": {k: v["run_id"] for k, v in anchors.items()}, "proposals": {k: proposals(k, v) for k, v in anchors.items()},
            "accuracy_target": .9997, "test_evaluated": False, "selection_primary": "validation macro F1; report accuracy and false-positive tradeoffs",
            "stopping_rule": "one controlled block of five changes per candidate, then two repeat seeds for the selected configuration"}
    plan.update(prepared_identity(prepared))
    path = output / "refinement_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Refinement plan differs; choose a new output directory")
    write_json(path, plan)
    registry = output / "experiments.json"
    records = json.loads(registry.read_text()) if registry.exists() else base_records.copy()
    for candidate, anchor in anchors.items():
        for spec in plan["proposals"][candidate]:
            execute(output, prepared, candidate, spec, 42, "search", records)
        effects_report(output, anchors, records)
        print(f"Five controlled trials analyzed for {candidate}; see optimization_effects.md", flush=True)
    for candidate in anchors:
        best = max([r for r in records if r["candidate"] == candidate and r["seed"] == 42 and r["status"] == "completed"], key=lambda r: r["metrics"]["macro_f1"])
        spec = {"id": best["config_id"], "parameters": best["config"], "purpose": "Repeat validation-selected setting to measure training-seed sensitivity"}
        for seed in [123, 2025]:
            execute(output, prepared, candidate, spec, seed, "repeat_selected", records)
    save_registry(output, records)
    effects_report(output, anchors, records)
    print(f"REFINEMENT COMPLETE: {output}; test remains locked", flush=True)
    return int(any(r["status"] != "completed" for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
