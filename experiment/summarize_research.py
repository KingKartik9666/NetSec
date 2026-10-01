"""Summarize measured validation studies without evaluating the locked test set."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from framework.training.tracking import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "artifacts/training"
    records = {}
    for name in ["datasense_gpu_v1", "datasense_gpu_refinement", "datasense_protocols_v1", "datasense_full_training"]:
        path = directory / name / "experiments.json"
        if path.exists():
            for record in json.loads(path.read_text()):
                records[record["run_id"]] = record
    completed = [r for r in records.values() if r["status"] == "completed"]
    groups = defaultdict(list)
    for r in completed:
        groups[(r["candidate"], r["prepared"], json.dumps(r["config"], sort_keys=True))].append(r)
    finalists = []
    for candidate in ["random_forest", "xgboost", "kan", "hem"]:
        anchors = [r for r in completed if r["candidate"] == candidate and r["seed"] == 42]
        if not anchors:
            continue
        anchor = max(anchors, key=lambda r: r["metrics"]["macro_f1"])
        runs = groups[(candidate, anchor["prepared"], json.dumps(anchor["config"], sort_keys=True))]
        row = {"candidate": candidate, "backend": anchor["backend"], "config_id": anchor["config_id"],
               "training_rows": anchor["training_rows"],
               "feature_count": len(anchor["features"]), "seed_42_run_id": anchor["run_id"],
               "checkpoint_path": anchor["checkpoint_path"], "prepared": anchor["prepared"],
               "seeds": sorted(r["seed"] for r in runs)}
        for metric in ["macro_f1", "accuracy", "false_positive_rate", "weighted_f1", "balanced_accuracy"]:
            values = [r["metrics"][metric] for r in runs]
            row[metric + "_mean"] = float(np.mean(values))
            row[metric + "_std"] = float(np.std(values))
        finalists.append(row)
        write_json(directory / "selected" / f"{candidate}.json", {"selection": row, "seed_42_record": anchor})
    highest = max(completed, key=lambda r: r["metrics"]["accuracy"])
    total_seconds = sum(r["training_seconds"] for r in completed)
    summary = {"dataset": "User-supplied DataSense combined CSV", "target_accuracy": .9997,
               "target_reached_on_validation": highest["metrics"]["accuracy"] >= .9997,
               "highest_single_run_validation_accuracy": highest["metrics"]["accuracy"],
               "highest_accuracy_run_id": highest["run_id"], "test_evaluated": False,
               "completed_trials": len(completed), "failed_trials": len(records) - len(completed),
               "sum_training_seconds": total_seconds,
               "all_completed_trials_cuda_verified": all(r["device"].startswith("cuda") for r in completed),
               "all_saved_checkpoints_reload_verified": all(r["checkpoint_reload_verified"] for r in completed),
               "selection_method": "Highest seed-42 validation macro F1 within each family; summarize repeats on same split",
               "candidates": finalists}
    write_json(directory / "RESULTS.json", summary)
    text = "# DataSense GPU training results\n\n"
    text += f"Completed **{len(completed)} real-data CUDA training trials**, failed **{summary['failed_trials']}**. Total recorded fitting time: {total_seconds / 60:.1f} minutes (not a standardized speed benchmark).\n\n"
    text += f"**99.97% accuracy target reached: {summary['target_reached_on_validation']}.** Highest measured single-run validation accuracy: **{highest['metrics']['accuracy']:.4%}**, {highest['candidate']}, {highest['config_id']}, seed {highest['seed']}.\n\n"
    text += "## Repeated-seed results\n\nConfigurations are selected by seed-42 macro F1, then repeated across seeds on the same split. Standard deviations describe training randomness, not independent-capture uncertainty.\n\n"
    text += "| Candidate | Training rows | Inputs | Seeds | Accuracy mean | Macro F1 mean +/- SD | Benign FPR mean |\n|---|---:|---:|---|---:|---:|---:|\n"
    for row in finalists:
        text += f"| {row['candidate']} | {row['training_rows']} | {row['feature_count']} | {row['seeds']} | {row['accuracy_mean']:.4%} | {row['macro_f1_mean']:.5f} +/- {row['macro_f1_std']:.5f} | {row['false_positive_rate_mean']:.4%} |\n"
    text += "\n## What the numbers mean\n\n- All trials use real source data. Initial searches use 120,000 training rows; the scaling study uses every eligible training row. All share the same 30,000 validation rows from the inspected 685,671-row CSV.\n- Fine-grained classification has 84 label4 classes, not just benign versus attack. One class has no independent validation support after overlap/duplicate exclusions.\n- Class-conditional temporal purging prevents overlapping windows across partitions. Results are within-capture, not independent-capture/device validation.\n- Test data remains locked: none of these numbers are final test accuracy. No decisive production winner has been promoted.\n- Forest means XGBoost standalone CUDA forest. KAN is genuine B-spline KAN. HEM is a documented DataSense closed-set adaptation using original neural classes, CUDA KNN and source-equivalent voting.\n- Default run_demo.py remains a separate synthetic DemoDetector and offline TemplateProvider. No XAI/LLM execution is claimed by this experiment.\n"
    text += "\n## Optimization evidence\n\nSee `datasense_gpu_refinement/optimization_effects.md` for controlled parameter changes and interactions, and `datasense_protocols_v1/feature_effects.md` for observed protocol feature comparisons. The registries retain all settings, class weights, seeds, per-class metrics, confusion matrices and resource measurements. Best model pointers are in `selected/`.\n\n"
    text += "Cleanup is deferred: the requested accuracy target and a final model selection have not been established. No original data or code was deleted.\n"
    (directory / "RESULTS.md").write_text(text, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
