"""Train six GPU families on every eligible training row; keep evaluation held out."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiment.refine_candidates import execute
from experiment.train_candidates import ROOT, save_registry
from framework.training.tracking import prepared_identity, validate_manifest, write_json

CANDIDATES = ("random_forest", "kan", "hem", "xgboost", "catboost", "lightgbm")
OUTPUT = ROOT / "artifacts/training/datasense_all_models_full_v1"


def study_plan():
    """Freeze parameter proposals before measuring the expanded study."""
    selected = ROOT / "artifacts/training/selected"
    anchors = {name: json.loads((selected / f"{name}.json").read_text())["seed_42_record"]
               for name in ("random_forest", "kan", "hem", "xgboost")}
    numeric = ROOT / "dataset/prepared/datasense_full_train"
    protocols = ROOT / "dataset/prepared/datasense_full_protocols"
    numeric_manifest = json.loads((numeric / "manifest.json").read_text())
    n_train = sum(numeric_manifest["eligible_class_counts"]["train"].values())
    prepared = {name: numeric if name in {"kan", "hem"} else protocols for name in CANDIDATES}
    for directory in {numeric, protocols}:
        manifest = json.loads((directory / "manifest.json").read_text())
        validate_manifest(manifest)
        if manifest["source_sha256"] != numeric_manifest["source_sha256"]:
            raise ValueError("Prepared representations have different source data")
        if len(np.load(directory / "y_train.npy", mmap_mode="r")) != n_train:
            raise ValueError("This study requires every eligible training row")
        if manifest["classes"] != numeric_manifest["classes"]:
            raise ValueError("Class ordering differs between representations")
    for split in ("train", "validation"):
        for prefix in ("row_ids", "y"):
            if not np.array_equal(np.load(numeric / f"{prefix}_{split}.npy"),
                                  np.load(protocols / f"{prefix}_{split}.npy")):
                raise ValueError("Representations must use identical rows and labels")

    configs = {name: [{"id": "full_previous_best", "parameters": dict(anchors[name]["config"]),
                      "purpose": "Retrain previous selected configuration on all eligible training rows"}]
               for name in ("random_forest", "kan", "hem")}
    configs["hem"][0]["parameters"].update(knn_max_rows=n_train, knn_batch_size=64)
    xgb_base = {"max_bin": 128, "weight_strategy": "none", "early_stopping_rounds": 40,
                "verbose_eval": 25, "subsample": .8, "colsample_bytree": .9,
                "colsample_bynode": .8, "reg_alpha": 0., "reg_lambda": 1., "gamma": 0.}
    xgb_changes = [
        {"max_depth": 6, "n_estimators": 500, "lr": .08, "min_child_weight": 1},
        {"max_depth": 8, "n_estimators": 750, "lr": .06, "min_child_weight": 2, "reg_lambda": 3, "gamma": .1},
        {"max_depth": 10, "n_estimators": 1000, "lr": .05, "min_child_weight": 5, "reg_lambda": 5, "reg_alpha": .2},
        {"max_depth": 12, "n_estimators": 1500, "lr": .04, "min_child_weight": 10, "reg_lambda": 10, "gamma": .5},
        {"max_depth": 8, "n_estimators": 1000, "lr": .05, "min_child_weight": 3, "subsample": .95, "colsample_bytree": .7, "reg_alpha": .1},
        {"max_depth": 6, "n_estimators": 1500, "lr": .03, "min_child_weight": 2, "reg_lambda": 3, "gamma": .05},
    ]
    configs["xgboost"] = [{"id": f"joint_{i}", "parameters": {**xgb_base, **change},
                           "purpose": "Joint depth, rounds, learning rate, sampling and regularization search"}
                          for i, change in enumerate(xgb_changes, 1)]
    configs["catboost"] = [
        {"id": "depth6", "parameters": {"depth": 6, "iterations": 750, "learning_rate": .08,
         "l2_leaf_reg": 3, "border_count": 128, "early_stopping_rounds": 40, "weight_strategy": "none"},
         "purpose": "GPU CatBoost baseline on full prepared features"},
        {"id": "depth8", "parameters": {"depth": 8, "iterations": 1200, "learning_rate": .05,
         "l2_leaf_reg": 5, "border_count": 128, "early_stopping_rounds": 40, "weight_strategy": "none"},
         "purpose": "Greater CatBoost capacity with stronger regularization"},
    ]
    configs["lightgbm"] = [
        {"id": "leaves31", "parameters": {"num_leaves": 31, "n_estimators": 500, "learning_rate": .08,
         "max_bin": 63, "min_data_in_leaf": 30, "lambda_l2": 1, "early_stopping_rounds": 40, "weight_strategy": "none"},
         "purpose": "NVIDIA OpenCL LightGBM baseline on all training rows"},
        {"id": "leaves63", "parameters": {"num_leaves": 63, "n_estimators": 1000, "learning_rate": .05,
         "max_bin": 63, "min_data_in_leaf": 50, "lambda_l2": 3, "lambda_l1": .1,
         "feature_fraction": .8, "bagging_fraction": .9, "bagging_freq": 1,
         "early_stopping_rounds": 40, "weight_strategy": "none"},
         "purpose": "Larger LightGBM with sampling and regularization"},
    ]
    return {"source_sha256": numeric_manifest["source_sha256"], "source_rows": numeric_manifest["source_rows"],
            "training_rows": n_train, "validation_rows": numeric_manifest["rows"]["validation"],
            "prepared_by_candidate": {name: prepared_identity(path) for name, path in prepared.items()},
            "anchors": {name: value["run_id"] for name, value in anchors.items()},
            "configurations": configs, "search_seed": 42, "repeat_seeds": [123, 2025],
            "primary_selection": "unweighted validation macro F1 over all 84 classes",
            "accuracy_targets": [.96, .9997], "test_evaluated": False,
            "complete_dataset_definition": "All eligible training rows after temporal purging; evaluation rows are held out",
            "representation_note": "Trees use 170 numeric/protocol inputs; KAN and HEM retain their selected 71 numeric inputs"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument("--search-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = study_plan()
    plan_file = output / "study.json"
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise ValueError("Full-data study identity changed; use a new output directory")
    write_json(plan_file, plan)
    registry = output / "experiments.json"
    records = json.loads(registry.read_text()) if registry.exists() else []
    for name in args.candidates:
        prepared = Path(plan["prepared_by_candidate"][name]["prepared"])
        for spec in plan["configurations"][name]:
            execute(output, prepared, name, spec, 42, "search", records)
    if not args.search_only:
        for name in args.candidates:
            search = [r for r in records if r["candidate"] == name and r["seed"] == 42 and r["status"] == "completed"]
            if not search:
                continue
            selected = max(search, key=lambda r: r["metrics"]["macro_f1"])
            spec = next(s for s in plan["configurations"][name] if s["id"] == selected["config_id"])
            for seed in plan["repeat_seeds"]:
                execute(output, Path(plan["prepared_by_candidate"][name]["prepared"]), name, spec, seed, "repeat_selected", records)
    save_registry(output, records)
    print(f"FULL-DATA STUDY saved to {output}; every fit uses {plan['training_rows']} training rows; test not evaluated", flush=True)
    return int(any(r["status"] != "completed" for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
