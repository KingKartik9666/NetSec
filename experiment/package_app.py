"""Freeze the highest-accuracy full-study checkpoint for the offline app.

Original weights, records and test data are never modified. The small bundle
references the original checkpoint with a portable relative path and checksum.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from framework.detection.trained_xgboost import file_sha256, json_features, PROTOCOL_FIELDS

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "artifacts/training/datasense_all_models_full_v1"


def local_artifact(value: str) -> Path:
    """Relocate historical absolute records beneath this project root."""
    path = Path(value)
    if path.is_file() or path.is_dir():
        return path
    normalized = value.replace("\\", "/")
    for folder in ["artifacts/", "dataset/"]:
        if folder in normalized:
            return ROOT / normalized[normalized.index(folder):]
    raise FileNotFoundError(f"Cannot locate saved artifact: {value}")


def package(output: Path, study: Path = STUDY) -> dict:
    import xgboost as xgb
    if output.exists():
        raise FileExistsError(f"Bundle already exists: {output}. Keep it frozen or choose a new --output.")
    records = json.loads((study / "experiments.json").read_text())
    eligible = [r for r in records if r["status"] == "completed" and r["candidate"] == "xgboost"]
    if not eligible:
        raise ValueError("No completed XGBoost full-data run exists")
    record = max(eligible, key=lambda r: r["metrics"]["accuracy"])
    prepared = local_artifact(record["prepared"])
    manifest_path = prepared / "manifest.json"
    if file_sha256(manifest_path) != record["prepared_manifest_sha256"]:
        raise ValueError("Prepared manifest checksum differs from the training record")
    manifest = json.loads(manifest_path.read_text())
    checkpoint = local_artifact(record["checkpoint_path"])
    metadata = json.loads((checkpoint.parent / "tree_metadata.json").read_text())
    if (manifest["rows"]["train"] != 480405 or len(manifest["classes"]) != 84
            or metadata["train_rows"] != manifest["rows"]["train"]
            or metadata["n_features"] != len(manifest["feature_names"])
            or record["classes"] != manifest["classes"]):
        raise ValueError("Expected the verified full-data 84-class schema and training rows")
    if "protocol_vocabulary" not in manifest:
        raise ValueError("Expected the frozen numeric/protocol feature representation")
    flag_count = sum(len(manifest["protocol_vocabulary"][f]) + 1 for f in PROTOCOL_FIELDS)
    numeric = manifest["feature_names"][:-flag_count]
    expected = numeric + [f"{field}__present_{token}" for field in PROTOCOL_FIELDS
                          for token in manifest["protocol_vocabulary"][field] + ["UNSEEN"]]
    if expected != manifest["feature_names"]:
        raise ValueError("Protocol vocabulary disagrees with recorded feature order")
    booster = xgb.Booster(model_file=str(checkpoint))
    booster.set_param({"device": "cpu", "nthread": 4})
    if booster.num_boosted_rounds() != metadata["saved_rounds"]:
        raise ValueError("Saved checkpoint rounds disagree with training metadata")
    X = np.load(prepared / "X_validation.npy", mmap_mode="r")
    labels = np.load(prepared / "y_validation.npy", mmap_mode="r")
    source_rows = np.load(prepared / "row_ids_validation.npy", mmap_mode="r")
    predictions = booster.predict(xgb.DMatrix(X, nthread=4), strict_shape=True).argmax(axis=1)
    cpu_accuracy = float(np.mean(predictions == labels))
    if abs(cpu_accuracy - record["metrics"]["accuracy"]) > 1e-6:
        raise ValueError("CPU checkpoint accuracy differs from the recorded validation result")
    indexes = sorted(int(i) for label in np.unique(labels) for i in np.flatnonzero(labels == label)[:2])
    samples = [{"id": i, "label": manifest["classes"][int(labels[row])], "source_row": int(source_rows[row]),
                "validation_index": row, "features": json_features(X[row])} for i, row in enumerate(indexes)]
    model = {"name": "XGBoost", "run_id": record["run_id"],
             "validation_accuracy": record["metrics"]["accuracy"], "macro_f1": record["metrics"]["macro_f1"],
             "training_rows": metadata["train_rows"], "features": metadata["n_features"],
             "classes": metadata["n_classes"], "seed": record["seed"],
             "selection_basis": "Highest single-run validation accuracy in the full-data study; provisional. "
                                "The study's macro-F1-selected model is a separate checkpoint.",
             "saved_rounds": metadata["saved_rounds"], "cpu_validation_accuracy": cpu_accuracy}
    bundle = {"schema_version": 1, "checkpoint": Path(os.path.relpath(checkpoint, output.parent)).as_posix(),
              "checkpoint_sha256": file_sha256(checkpoint), "prepared_manifest_sha256": file_sha256(manifest_path),
              "source_sha256": manifest["source_sha256"], "feature_names": manifest["feature_names"],
              "numeric_features": numeric, "protocol_vocabulary": manifest["protocol_vocabulary"],
              "classes": manifest["classes"], "benign_index": manifest["benign_index"], "model": model,
              "samples": samples, "sample_selection": "First two validation observations per supported class; no correctness filtering",
              "test_evaluated": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(bundle, stream, indent=2, allow_nan=False)
    return {"bundle": str(output), "checkpoint_sha256": bundle["checkpoint_sha256"],
            "samples": len(samples), "model": model, "test_evaluated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "framework/models/xgboost_app/bundle.json")
    args = parser.parse_args()
    print(json.dumps(package(args.output), indent=2))


if __name__ == "__main__":
    main()
