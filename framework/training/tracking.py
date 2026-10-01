"""Measured classification results and append-only experiment provenance."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix


def validate_manifest(manifest: dict) -> None:
    """Reject target/identifier columns before any candidate sees the arrays."""
    features = manifest["feature_names"]
    forbidden = [name for name in features if name.lower().startswith("label")
                 or name.lower().startswith("timestamp")
                 or name.lower() in {"device_name", "device_mac"}]
    if forbidden:
        raise ValueError(f"Potential label/identifier leakage: {forbidden}")
    if not features or len(features) != len(set(features)):
        raise ValueError("Feature schema is empty or contains duplicates")
    classes = manifest["classes"]
    if len(classes) < 2 or len(classes) != len(set(classes)):
        raise ValueError("At least two distinct classes required")
    if not 0 <= manifest["benign_index"] < len(classes):
        raise ValueError("Invalid benign class index")


def classification_metrics(y_true, y_pred, classes: list[str], benign_index: int) -> dict:
    """Compute unweighted validation metrics on all declared target classes."""
    truth = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if truth.ndim != 1 or truth.shape != pred.shape or truth.size == 0:
        raise ValueError("Nonempty matching one-dimensional predictions required")
    if min(truth.min(), pred.min()) < 0 or max(truth.max(), pred.max()) >= len(classes):
        raise ValueError("Prediction/target outside declared class range")
    labels = np.arange(len(classes))
    report = classification_report(truth, pred, labels=labels, target_names=classes,
                                   output_dict=True, zero_division=0)
    benign = truth == benign_index
    attack = ~benign
    return {
        "accuracy": float(np.mean(truth == pred)),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "classes_without_evaluation_support": [name for name in classes if report[name]["support"] == 0],
        "false_positive_rate": float(np.mean(pred[benign] != benign_index)) if benign.any() else None,
        "attack_missed_as_benign_rate": float(np.mean(pred[attack] == benign_index)) if attack.any() else None,
        "per_class": {name: report[name] for name in classes},
        "confusion_matrix": confusion_matrix(truth, pred, labels=labels).tolist(),
        "rows": int(len(truth)),
    }


def choose_best(current: dict | None, challenger: dict) -> dict | None:
    """Keep the incumbent on ties or failures; use validation macro F1 only."""
    if challenger.get("status") != "completed":
        return current
    if current is None or challenger["metrics"]["macro_f1"] > current["metrics"]["macro_f1"]:
        return challenger
    return current


def prepared_identity(prepared: Path) -> dict:
    prepared = Path(prepared).resolve()
    return {"prepared": str(prepared),
            "prepared_manifest_sha256": hashlib.sha256((prepared / "manifest.json").read_bytes()).hexdigest()}


def completed_trial(records, candidate, config_id, seed, config, prepared):
    """Reuse only an identical configuration and prepared-data identity."""
    identity = prepared_identity(prepared)
    matches = [r for r in records if r["candidate"] == candidate and r["config_id"] == config_id
               and r["seed"] == seed and r["status"] == "completed"]
    for record in matches:
        if record["config"] != config or any(record.get(k) != v for k, v in identity.items()):
            raise ValueError("Completed trial has different parameters or prepared identity; use a new study directory")
        if not Path(record["checkpoint_path"]).is_file():
            raise FileNotFoundError("Completed trial checkpoint is missing; restore artifacts before resuming")
    return matches[-1] if matches else None


def write_json(path: Path, value: dict | list) -> None:
    """Replace a small report atomically; checkpoints remain in unique run dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
