"""Run a saved research checkpoint on one real validation observation."""
import argparse
import json
from pathlib import Path

import numpy as np

from framework.training.tracking import prepared_identity

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT / "artifacts/training/selected/xgboost.json")
    parser.add_argument("--index", type=int, default=0, help="Validation-row index; no test access")
    args = parser.parse_args()
    saved = json.loads(args.record.read_text())
    record = saved.get("seed_42_record", saved)
    prepared = Path(record["prepared"])
    if record.get("status") != "completed":
        parser.error("The record must describe a successfully completed training run")
    if any(record.get(key) != value for key, value in prepared_identity(prepared).items()):
        parser.error("Prepared data identity differs from the saved training record")
    X = np.load(prepared / "X_validation.npy", mmap_mode="r")
    if not 0 <= args.index < len(X):
        parser.error(f"index must be in [0, {len(X) - 1}]")
    if record["candidate"] in {"random_forest", "xgboost"}:
        from framework.training.trees import predict_tree as predict
    elif record["candidate"] in {"catboost", "lightgbm"}:
        from framework.training.boosters import predict_booster as predict
    else:
        from framework.training.neural import predict_neural as predict
    predicted = int(predict(Path(record["checkpoint_path"]), X[args.index:args.index + 1])[0])
    actual = int(np.load(prepared / "y_validation.npy", mmap_mode="r")[args.index])
    source_row = int(np.load(prepared / "row_ids_validation.npy", mmap_mode="r")[args.index])
    print(json.dumps({"mode": "Real-data saved-checkpoint inference; validation-selected research model, not final NIDS",
                      "backend": record["backend"], "checkpoint": record["checkpoint_path"],
                      "validation_index": args.index, "source_row_index_zero_based": source_row,
                      "prediction": record["classes"][predicted], "dataset_label": record["classes"][actual],
                      "matches_dataset_label": predicted == actual, "test_evaluated": False,
                      "llm_executed": False, "xai_executed": False}, indent=2))


if __name__ == "__main__":
    main()
