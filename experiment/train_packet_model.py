"""Train the separate experimental seven-count, ten-second packet model on CUDA.

Only existing training/validation arrays are read. Test data remains locked.
Original prepared arrays, the 170-feature checkpoint and its records are not
modified. Completed attempts can be resumed only against the identical plan.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from framework.detection.packet_xgboost import (
    CLASSES, FEATURE_NAMES, LIMITATIONS, STATS_KEYS, WINDOW_SECONDS, file_sha256,
)
from framework.training.tracking import classification_metrics, write_json
from framework.training.trees import predict_tree, train_tree

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "dataset/prepared/datasense_full_train"
OUTPUT = ROOT / "artifacts/training/packet_live_v1"
BUNDLE = ROOT / "framework/models/packet_live/bundle.json"
CONFIGS = [
    {"id": "depth3", "max_depth": 3, "learning_rate": .08},
    {"id": "depth5", "max_depth": 5, "learning_rate": .06},
    {"id": "depth7", "max_depth": 7, "learning_rate": .05},
]


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


def array_digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def load_counts():
    """Use every eligible exact ten-second row from the frozen source split."""
    manifest = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))
    names = manifest["feature_names"]
    indices = [names.index(name) for name in FEATURE_NAMES]
    if manifest["classes"][manifest["benign_index"]] != "benign":
        raise ValueError("Unexpected source benign mapping")
    arrays, counts = {}, {}
    for split in ["train", "validation"]:
        start = np.load(PREPARED / f"window_start_{split}.npy", mmap_mode="r")
        end = np.load(PREPARED / f"window_end_{split}.npy", mmap_mode="r")
        labels = np.load(PREPARED / f"y_{split}.npy", mmap_mode="r")
        original = np.load(PREPARED / f"X_{split}.npy", mmap_mode="r")
        row_ids = np.load(PREPARED / f"row_ids_{split}.npy", mmap_mode="r")
        if not (len(start) == len(end) == len(labels) == len(original) == len(row_ids)):
            raise ValueError("Source arrays have inconsistent row counts")
        selected = np.flatnonzero(end - start == WINDOW_SECONDS * 1_000_000_000)
        if not len(selected):
            raise ValueError(f"No exact ten-second {split} windows; refusing a substitute duration")
        X = np.asarray(original[np.ix_(selected, indices)], dtype=np.float32)
        y = np.asarray(labels[selected] != manifest["benign_index"], dtype=np.int32)
        if (not np.isfinite(X).all() or (X < 0).any() or (X != np.floor(X)).any()
                or (X[:, 1:] > X[:, :1]).any() or len(np.unique(y)) != 2):
            raise ValueError(f"{split} packet counts or binary support are invalid")
        arrays[split] = (X, y)
        support = np.bincount(y, minlength=2)
        counts[split] = {
            "original_prepared_rows": int(len(labels)), "ten_second_rows": int(len(selected)),
            "normal_rows": int(support[0]), "suspicious_rows": int(support[1]),
            "zero_packet_rows": int((X[:, 0] == 0).sum()),
            "features_sha256": array_digest(X), "binary_labels_sha256": array_digest(y),
            "selected_row_ids_sha256": array_digest(row_ids[selected]),
            "window_duration_ns": WINDOW_SECONDS * 1_000_000_000,
        }
    train_vectors = np.ascontiguousarray(arrays["train"][0]).view(
        np.dtype((np.void, len(FEATURE_NAMES) * np.dtype(np.float32).itemsize))).ravel()
    validation_vectors = np.ascontiguousarray(arrays["validation"][0]).view(train_vectors.dtype).ravel()
    overlap = np.isin(validation_vectors, np.unique(train_vectors))
    counts["projection_diagnostics"] = {
        "validation_rows_with_identical_seven_counts_in_training": int(overlap.sum()),
        "note": "The existing split purges overlapping windows and original numeric duplicates. Reducing to seven counts can produce equal vectors from different windows; source partitions are retained.",
    }
    return manifest, arrays, counts


def plan(manifest, counts):
    common = {"n_estimators": 400, "early_stopping_rounds": 40, "max_bin": 128,
              "weight_strategy": "none", "verbose_eval": 25, "subsample": .8,
              "colsample_bytree": 1., "colsample_bynode": 1., "min_child_weight": 1.,
              "reg_lambda": 1., "reg_alpha": 0., "gamma": 0.}
    return {
        "schema_version": 1, "name": "Experimental packet-only binary XGBoost",
        "feature_schema_version": "datasense_header_counts_v1", "window_seconds": WINDOW_SECONDS,
        "seed": 42, "classes": CLASSES, "feature_names": FEATURE_NAMES, "stats_keys": STATS_KEYS,
        "binary_mapping": "normal=label4 benign; suspicious=all other label4 classes",
        "prepared": PREPARED.relative_to(ROOT).as_posix(),
        "prepared_manifest_sha256": file_sha256(PREPARED / "manifest.json"),
        "source_sha256": manifest["source_sha256"], "source_rows": manifest["source_rows"],
        "source_split_method": manifest["split_method"], "counts": counts,
        "configs": [{"id": config["id"], "parameters": {**common, **{k: v for k, v in config.items() if k != "id"}}}
                    for config in CONFIGS],
        "selection": "maximum unweighted validation macro F1; keep earlier config on ties",
        "test_evaluated": False, "live_validation": False, "limitations": LIMITATIONS,
        "primary_sources": ["https://www.unb.ca/cic/datasets/iiot-dataset-2025.html",
                            "https://doi.org/10.3390/electronics14204095",
                            "https://xgboost.readthedocs.io/en/stable/gpu/index.html",
                            "https://xgboost.readthedocs.io/en/stable/prediction.html"],
        "extraction_scope": "Seven packet/TCP-flag counts described by DataSense Table 5; SYN+ACK contributes to both counts. Publisher extractor equivalence and Windows generalization are not validated.",
    }


def run_trial(spec, arrays, study):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"xgboost_packet_{spec['id']}_s42_{stamp}"
    directory = OUTPUT / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    request = {"run_id": run_id, "config_id": spec["id"], "config": spec["parameters"],
               "seed": 42, "source_sha256": study["source_sha256"],
               "prepared_manifest_sha256": study["prepared_manifest_sha256"],
               "training_code_sha256": file_sha256(Path(__file__)),
               "inference_code_sha256": file_sha256(ROOT / "framework/detection/packet_xgboost.py"),
               "tree_training_code_sha256": file_sha256(ROOT / "framework/training/trees.py"),
               "test_evaluated": False, "live_validation": False}
    write_json(directory / "request.json", request)
    with (directory / "console.log").open("x", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                print(f"Real CUDA training: {run_id}; training rows {len(arrays['train'][1]):,}; validation rows {len(arrays['validation'][1]):,}", flush=True)
                trained = train_tree("xgboost", *arrays["train"], *arrays["validation"],
                                     len(CLASSES), spec["parameters"], 42, directory)
                metrics = classification_metrics(arrays["validation"][1], trained["predictions"], CLASSES, 0)
                reloaded = predict_tree(trained["checkpoint_path"], arrays["validation"][0])
                if not np.array_equal(reloaded, trained["predictions"]):
                    raise RuntimeError("Saved checkpoint does not reproduce validation predictions")
                np.save(directory / "validation_predictions.npy", reloaded)
                result = {**request, "status": "completed", "metrics": metrics,
                          "device": trained["device"], "device_evidence": trained["device_evidence"],
                          "backend": trained["backend"], "xgboost_version": trained["xgboost_version"],
                          "training_seconds": trained["training_seconds"],
                          "train_rows": trained["train_rows"], "validation_rows": trained["validation_rows"],
                          "saved_rounds": trained["saved_rounds"], "best_iteration": trained["best_iteration"],
                          "checkpoint": (directory / "model.ubj").relative_to(ROOT).as_posix(),
                          "checkpoint_sha256": file_sha256(directory / "model.ubj"), "reload_verified": True}
                print(f"Completed: accuracy={metrics['accuracy']:.6f}; macro F1={metrics['macro_f1']:.6f}; FPR={metrics['false_positive_rate']:.6f}; FNR={metrics['attack_missed_as_benign_rate']:.6f}", flush=True)
            except Exception as exc:
                traceback.print_exc()
                result = {**request, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    write_json(directory / "result.json", result)
    return result


def package(best, study, X_val):
    checkpoint = ROOT / best["checkpoint"]
    bundle = {"schema_version": 1, "model": "Experimental packet XGBoost",
              "checkpoint": Path(os.path.relpath(checkpoint, BUNDLE.parent)).as_posix(),
              "checkpoint_sha256": best["checkpoint_sha256"], "source_sha256": study["source_sha256"],
              "prepared_manifest_sha256": study["prepared_manifest_sha256"],
              "feature_schema_version": study["feature_schema_version"], "feature_names": FEATURE_NAMES,
              "stats_keys": STATS_KEYS, "classes": CLASSES, "window_seconds": WINDOW_SECONDS,
              "run_id": best["run_id"], "config": best["config"], "validation_metrics": best["metrics"],
              "counts": study["counts"], "live_validation": False, "test_evaluated": False,
              "limitations": LIMITATIONS, "primary_sources": study["primary_sources"]}
    if BUNDLE.exists():
        if json.loads(BUNDLE.read_text(encoding="utf-8")) != bundle:
            raise FileExistsError("A different packet bundle already exists; preserve it and use a new study/version")
    else:
        write_json(BUNDLE, bundle)
    from framework.detection.packet_xgboost import PacketXGBoost
    detector = PacketXGBoost(BUNDLE)
    proof = detector.predict(dict(zip(STATS_KEYS, X_val[np.flatnonzero(X_val[:, 0] > 0)[0]].tolist())))
    proof["check"] = "CPU saved-checkpoint inference and exact native TreeSHAP margin reconstruction"
    write_json(OUTPUT / "cpu_xai_proof.json", proof)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    manifest, arrays, counts = load_counts()
    study = plan(manifest, counts)
    study_file = OUTPUT / "study.json"
    if study_file.exists():
        if json.loads(study_file.read_text(encoding="utf-8")) != study:
            raise ValueError("Study inputs or configurations changed; use a new version instead of overwriting")
    else:
        write_json(study_file, study)
        (OUTPUT / "red_signal.txt").write_text(
            "Before implementation: PacketXGBoost import/predict failed with ModuleNotFoundError.\n"
            "Alternative post-change proof: real CUDA three-trial fit, saved-model reload on all validation rows, CPU inference and exact TreeSHAP additivity, root-owned integration tests.\n", encoding="utf-8")
    registry = OUTPUT / "experiments.json"
    records = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else []
    for spec in study["configs"]:
        existing = [r for r in records if r["config_id"] == spec["id"] and r["status"] == "completed"]
        if existing:
            if (len(existing) != 1 or existing[0]["config"] != spec["parameters"]
                    or existing[0]["prepared_manifest_sha256"] != study["prepared_manifest_sha256"]
                    or file_sha256(ROOT / existing[0]["checkpoint"]) != existing[0]["checkpoint_sha256"]):
                raise ValueError("Completed packet trial identity or checkpoint changed")
            continue
        records.append(run_trial(spec, arrays, study))
        write_json(registry, records)
    completed = [r for r in records if r["status"] == "completed"]
    if not completed:
        raise RuntimeError("No real CUDA packet trials completed; no model can be packaged")
    best = max(completed, key=lambda record: record["metrics"]["macro_f1"])
    summary = {"study": study["name"], "completed_trials": len(completed),
               "failed_trials": sum(r["status"] == "failed" for r in records), "best": best,
               "counts": counts, "test_evaluated": False, "live_validation": False,
               "limitations": LIMITATIONS}
    write_json(OUTPUT / "RESULTS.json", summary)
    text = "# Experimental packet-only XGBoost results\n\n"
    text += "Binary classification of genuine DataSense ten-second windows using seven packet/TCP-flag counts. Separate from the original 84-class app model. Test remains locked; Windows live performance is unvalidated.\n\n"
    text += f"Training: **{counts['train']['ten_second_rows']:,}** rows (every eligible exact ten-second training window); validation: **{counts['validation']['ten_second_rows']:,}** rows.\n\n"
    text += "| Setting | Status | Validation accuracy | Macro F1 | False-positive rate | Missed-attack rate |\n|---|---|---:|---:|---:|---:|\n"
    for record in records:
        metrics = record.get("metrics")
        scores = " | ".join(f"{metrics[key]:.6f}" for key in ["accuracy", "macro_f1", "false_positive_rate", "attack_missed_as_benign_rate"]) if metrics else "— | — | — | —"
        text += f"| {record['config_id']} | {record['status']} | {scores} |\n"
    text += f"\nSelected by validation macro F1: **{best['config_id']}**, seed 42.\n\n"
    text += "\n".join(f"- {limitation}" for limitation in LIMITATIONS)
    text += f"\n- Seven-count projection overlap: {counts['projection_diagnostics']['validation_rows_with_identical_seven_counts_in_training']:,} validation rows have count vectors also observed in training; these can represent distinct windows.\n"
    (OUTPUT / "RESULTS.md").write_text(text + "\n", encoding="utf-8")
    package(best, study, arrays["validation"][0])
    print(f"Selected {best['config_id']}; CPU packet model bundle: {BUNDLE.relative_to(ROOT)}", flush=True)
    return int(len(completed) != len(study["configs"]))


if __name__ == "__main__":
    raise SystemExit(main())
