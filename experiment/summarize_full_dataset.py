"""Report the six-family full-training study using validation results only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiment.train_full_dataset import CANDIDATES, OUTPUT
from framework.training.tracking import write_json


def has_complete_coverage(plan, completed):
    """Require each planned search and each selected repeat exactly once."""
    expected = {(name, spec["id"], plan["search_seed"])
                for name, specs in plan["configurations"].items() for spec in specs}
    actual = [(r["candidate"], r["config_id"], r["seed"]) for r in completed]
    for name in plan["configurations"]:
        search = [r for r in completed if r["candidate"] == name and r["seed"] == plan["search_seed"]]
        if not search:
            return False
        selected = max(search, key=lambda r: r["metrics"]["macro_f1"])
        expected.update((name, selected["config_id"], seed) for seed in plan["repeat_seeds"])
    return len(actual) == len(set(actual)) and set(actual) == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=OUTPUT)
    args = parser.parse_args()
    directory = args.study.resolve()
    plan = json.loads((directory / "study.json").read_text())
    records = json.loads((directory / "experiments.json").read_text())
    completed = [r for r in records if r["status"] == "completed"]
    if not completed:
        raise ValueError("No completed full-data trials to summarize")
    expected_searches = sum(len(specs) for specs in plan["configurations"].values())
    expected_trials = expected_searches + len(CANDIDATES) * len(plan["repeat_seeds"])
    candidates = []
    for name in CANDIDATES:
        search = [r for r in completed if r["candidate"] == name and r["seed"] == plan["search_seed"]]
        if not search:
            continue
        anchor = max(search, key=lambda r: r["metrics"]["macro_f1"])
        runs = [r for r in completed if r["candidate"] == name and r["config"] == anchor["config"]
                and r["prepared_manifest_sha256"] == anchor["prepared_manifest_sha256"]]
        by_seed = {r["seed"]: r for r in runs}
        if len(by_seed) != len(runs):
            raise ValueError("Duplicate completed seed/configuration records require reconciliation")
        row = {"candidate": name, "backend": anchor["backend"], "config_id": anchor["config_id"],
               "training_rows": anchor["training_rows"], "feature_count": len(anchor["features"]),
               "seeds": sorted(by_seed), "seed_42_run_id": anchor["run_id"],
               "knn_reference_rows": anchor.get("knn_reference_rows")}
        for metric in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "false_positive_rate"):
            values = [r["metrics"][metric] for r in runs]
            row[metric + "_mean"] = float(np.mean(values))
            row[metric + "_std"] = float(np.std(values))
        row["best_single_accuracy"] = max(r["metrics"]["accuracy"] for r in completed if r["candidate"] == name)
        candidates.append(row)
        write_json(directory / "selected" / f"{name}.json", {"selection": row, "seed_42_record": anchor})
    highest = max(completed, key=lambda r: r["metrics"]["accuracy"])
    summary = {"study": "All eligible DataSense training rows, six GPU families", "test_evaluated": False,
               "source_rows": plan["source_rows"], "training_rows": plan["training_rows"],
               "validation_rows": plan["validation_rows"], "classes": len(highest["classes"]),
               "completed_trials": len(completed), "failed_attempts": len(records) - len(completed),
               "planned_completed_trials": expected_trials, "study_complete": has_complete_coverage(plan, completed),
               "highest_validation_accuracy": highest["metrics"]["accuracy"],
               "highest_accuracy_run_id": highest["run_id"], "highest_accuracy_candidate": highest["candidate"],
               "accuracy_96_percent_reached": highest["metrics"]["accuracy"] >= .96,
               "accuracy_99_97_percent_reached": highest["metrics"]["accuracy"] >= .9997,
               "all_trials_used_full_training_rows": all(r["training_rows"] == plan["training_rows"] for r in completed),
               "all_checkpoints_reload_verified": all(r["checkpoint_reload_verified"] for r in completed),
               "all_training_gpu_verified": all(r.get("gpu_training_verified", r["device"].startswith("cuda")) for r in completed),
               "candidates": candidates}
    if not summary["all_trials_used_full_training_rows"]:
        raise AssertionError("A completed trial did not use all eligible training rows")
    write_json(directory / "RESULTS.json", summary)
    text = "# Full-data GPU candidate results\n\n"
    text += f"Completed {len(completed)} of {expected_trials} planned trials; {summary['failed_attempts']} failed attempts retained. Every completed fit used **{plan['training_rows']:,} training rows**, drawn from the {plan['source_rows']:,}-row source after chronological splitting and leakage exclusions. Evaluation rows are not training data.\n\n"
    text += f"Highest single-run validation accuracy: **{highest['metrics']['accuracy']:.4%}**, {highest['candidate']}, {highest['config_id']}, seed {highest['seed']}. 96% reached: **{summary['accuracy_96_percent_reached']}**; 99.97% reached: **{summary['accuracy_99_97_percent_reached']}**.\n\n"
    text += "Configurations were selected by seed-42 unweighted macro F1 and repeated with seeds 123 and 2025. The highest accuracy above includes all measured settings; the table summarizes the macro-F1-selected setting per family.\n\n"
    text += "| Candidate | Features | Seeds | Mean accuracy | Mean macro F1 +/- SD | Mean benign FPR |\n|---|---:|---|---:|---:|---:|\n"
    for row in candidates:
        text += f"| {row['candidate']} | {row['feature_count']} | {row['seeds']} | {row['accuracy_mean']:.4%} | {row['macro_f1_mean']:.5f} +/- {row['macro_f1_std']:.5f} | {row['false_positive_rate_mean']:.4%} |\n"
    text += "\nAll candidates use the same training/validation row IDs and 84 fine-grained labels. KAN/HEM retain their previously selected 71 numeric inputs; the trees use 170 numeric/protocol inputs. HEM uses all training rows as KNN references. CatBoost receives the same numeric inputs, without invented categorical fields. LightGBM uses NVIDIA OpenCL histogram acceleration with CPU orchestration; the other training backends use CUDA. CatBoost/LightGBM saved-model prediction runs on CPU.\n\n"
    text += "Test scoring remains locked. These are within-capture validation experiments, not unseen-device/capture generalization or final research test results. One class has zero validation support after exclusions; all 84 remain in macro F1. Seed variability describes training randomness. GPU floating-point reduction order can introduce small nondeterminism.\n\n"
    text += "The forest is XGBoost's standalone GPU forest. KAN is the original efficient-kan B-spline implementation. HEM is a DataSense closed-set adaptation using upstream neural classes and source-equivalent voting; no unknown-attack reproduction is claimed. The synthetic demo and TemplateProvider remain separate. No XAI or LLM runs in this study.\n\n"
    text += "Per-trial settings, checkpoints, class-wise metrics, confusion matrices and measured resource use are retained in `experiments.json` and `runs/`. `selected/` contains provisional model pointers. Original data and earlier experiments were preserved.\n"
    (directory / "RESULTS.md").write_text(text, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
