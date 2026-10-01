"""Explicit CUDA tree candidates for the offline research comparison.

``random_forest`` is XGBoost's standalone forest: one boosting round,
parallel trees, row/column subsampling, and no replacement sampling. It is
not scikit-learn's conventional RandomForestClassifier. Both candidates
use XGBoost's second-order objective. Missing numeric values remain NaN.

References:
https://xgboost.readthedocs.io/en/stable/tutorials/rf.html
https://xgboost.readthedocs.io/en/stable/gpu/index.html
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np


def class_weights(y_train: np.ndarray, n_classes: int, strategy: str) -> np.ndarray:
    """Training-frequency weights, normalized by sample mean then capped at 20.

    Clipping is deliberately after normalization, so the final mean may be
    below one. All candidates use this same definition and training labels.
    """
    labels = _labels(y_train, n_classes, "y_train")
    if strategy not in {"none", "sqrt_balanced", "balanced"}:
        raise ValueError(f"Unknown weight_strategy: {strategy}")
    counts = np.bincount(labels, minlength=n_classes)
    if np.any(counts == 0):
        raise ValueError("Each encoded class must occur in training data")
    power = {"none": 0.0, "sqrt_balanced": 0.5, "balanced": 1.0}[strategy]
    weights = (len(labels) / (n_classes * counts.astype(np.float64))) ** power
    weights /= np.average(weights, weights=counts)
    return np.minimum(weights, 20.0).astype(np.float32)


def _labels(y: np.ndarray, n_classes: int, name: str) -> np.ndarray:
    values = np.asarray(y)
    if n_classes < 2 or values.ndim != 1 or len(values) == 0:
        raise ValueError(f"{name} must contain labels for at least two classes")
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite integer class indices")
    if np.any(values != np.floor(values)) or np.any(values < 0) or np.any(values >= n_classes):
        raise ValueError(f"{name} must contain integer indices in [0, n_classes)")
    return values.astype(np.int32, copy=False)


def _features(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=np.float32, order="C")
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("Features must be a nonempty two-dimensional numeric array")
    if np.any(np.isinf(values)):
        raise ValueError("Infinite features must be converted to missing values before training")
    return values


def _require_cuda(model: Any) -> str:
    """Inspect actual runtime configuration; never accept CPU fallback."""
    device = str(json.loads(model.save_config())["learner"]["generic_param"]["device"])
    if not device.startswith("cuda"):
        raise RuntimeError(f"GPU execution required, but XGBoost selected {device!r}")
    return device


def _parameters(candidate: str, n_classes: int, config: dict, seed: int) -> tuple[dict, int]:
    if candidate not in {"random_forest", "xgboost"}:
        raise ValueError(f"Unknown tree candidate: {candidate}")
    if config.get("device", "cuda") not in {"cuda", "cuda:0"}:
        raise ValueError("Research tree training requires device='cuda' or 'cuda:0'")
    params = {
        "booster": "gbtree", "tree_method": "hist", "device": "cuda:0",
        "objective": "multi:softprob", "num_class": n_classes,
        "eval_metric": "mlogloss", "seed": int(seed), "nthread": 4,
        "max_depth": 6, "max_bin": 128, "min_child_weight": 1.0,
        "reg_lambda": 1.0, "reg_alpha": 0.0, "subsample": 0.8,
        "colsample_bytree": 1.0, "colsample_bynode": 0.8,
        "learning_rate": 0.1, "num_parallel_tree": 1,
    }
    for name in (
        "max_depth", "max_bin", "min_child_weight", "reg_lambda", "reg_alpha", "gamma",
        "subsample", "colsample_bytree", "colsample_bynode", "learning_rate", "nthread",
    ):
        if name in config:
            params[name] = config[name]
    if "lr" in config:
        params["learning_rate"] = config["lr"]
    if int(params["max_depth"]) < 1 or int(params["max_bin"]) < 2:
        raise ValueError("max_depth must be positive and max_bin at least two")
    for name in ("subsample", "colsample_bytree", "colsample_bynode"):
        if not 0 < float(params[name]) <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    if candidate == "random_forest":
        if config.get("n_estimators", 1) != 1 or config.get("num_boost_round", 1) != 1:
            raise ValueError("Standalone random_forest uses exactly one boosting round; set num_parallel_tree")
        params["num_parallel_tree"] = int(config.get("num_parallel_tree", 64))
        params["learning_rate"] = 1.0
        if params["num_parallel_tree"] < 1:
            raise ValueError("num_parallel_tree must be positive")
        if params["subsample"] >= 1 or min(params["colsample_bytree"], params["colsample_bynode"]) >= 1:
            raise ValueError("Standalone forest requires both row and column subsampling below one")
        rounds = 1
    else:
        if config.get("num_parallel_tree", 1) != 1:
            raise ValueError("The boosted candidate uses one tree per class per round")
        rounds = int(config.get("num_boost_round", config.get("n_estimators", 160)))
        if rounds < 1:
            raise ValueError("Number of boosting rounds must be positive")
    return params, rounds


def _probabilities(predictions: np.ndarray, n_rows: int, n_classes: int) -> np.ndarray:
    values = np.asarray(predictions)
    if values.size != n_rows * n_classes:
        raise ValueError(f"Unexpected multiclass prediction shape: {values.shape}")
    return values.reshape(n_rows, n_classes)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    counts = np.bincount(n_classes * y_true + y_pred, minlength=n_classes**2).reshape(n_classes, n_classes)
    denominator = counts.sum(axis=0) + counts.sum(axis=1)
    scores = np.divide(2 * np.diag(counts), denominator, out=np.zeros(n_classes), where=denominator != 0)
    return float(scores.mean())


def train_tree(
    candidate: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    config: dict,
    seed: int,
    output_dir: str | Path,
) -> dict:
    """Fit one CUDA trial and save its validation-selected checkpoint.

    Test data is never accepted here. Numeric column order and label names
    belong to the experiment manifest. Training seconds include DMatrix
    construction and fitting, excluding checkpoint and prediction work.
    Output files are exclusively created and cannot overwrite prior trials.
    """
    import xgboost as xgb

    params, rounds = _parameters(candidate, n_classes, config, seed)
    X_train, X_val = _features(X_train), _features(X_val)
    y_train = _labels(y_train, n_classes, "y_train")
    y_val = _labels(y_val, n_classes, "y_val")
    if len(X_train) != len(y_train) or len(X_val) != len(y_val) or X_train.shape[1] != X_val.shape[1]:
        raise ValueError("Train/validation arrays have incompatible shapes")
    weights = class_weights(y_train, n_classes, config.get("weight_strategy", "none"))
    if not xgb.build_info().get("USE_CUDA", False):
        raise RuntimeError("Installed XGBoost was built without CUDA support")
    directory = Path(output_dir)
    checkpoint = directory / "model.ubj"
    metadata_path = directory / "tree_metadata.json"
    if checkpoint.exists() or metadata_path.exists():
        raise FileExistsError(f"Trial artifacts already exist in {directory}")

    class CudaGuard(xgb.callback.TrainingCallback):
        def before_training(self, model):
            _require_cuda(model)
            return model

        def after_training(self, model):
            _require_cuda(model)
            return model

    def macro_f1(predictions, data):
        truth = data.get_label().astype(np.int32)
        predicted = _probabilities(predictions, len(truth), n_classes).argmax(axis=1)
        return "macro_f1", _macro_f1(truth, predicted, n_classes)

    callbacks = [CudaGuard()]
    if candidate == "xgboost":
        patience = int(config.get("early_stopping_rounds", 20))
        if patience < 1:
            raise ValueError("early_stopping_rounds must be positive")
        callbacks.append(xgb.callback.EarlyStopping(
            rounds=patience, metric_name="macro_f1", data_name="validation",
            maximize=True, save_best=True,
        ))
    history: dict = {}
    started = time.perf_counter()
    training = xgb.QuantileDMatrix(
        X_train, label=y_train, weight=weights[y_train], max_bin=params["max_bin"], nthread=params["nthread"],
    )
    validation = xgb.QuantileDMatrix(
        X_val, label=y_val, max_bin=params["max_bin"], ref=training, nthread=params["nthread"],
    )
    model = xgb.train(
        params, training, num_boost_round=rounds, evals=[(validation, "validation")],
        custom_metric=macro_f1, callbacks=callbacks, evals_result=history,
        verbose_eval=config.get("verbose_eval", False),
    )
    training_seconds = time.perf_counter() - started
    device = _require_cuda(model)
    probabilities = _probabilities(model.predict(validation), len(y_val), n_classes)
    best_iteration = int(getattr(model, "best_iteration", model.num_boosted_rounds() - 1))
    backend = "XGBoost CUDA standalone random forest" if candidate == "random_forest" else "XGBoost CUDA gradient-boosted trees"
    metadata = {
        "candidate": candidate, "backend": backend, "xgboost_version": xgb.__version__,
        "seed": int(seed), "parameters": params, "requested_config": config,
        "training_seconds": training_seconds, "device": device,
        "device_evidence": json.loads(model.save_config())["learner"]["generic_param"],
        "best_iteration": best_iteration, "trained_rounds": len(history["validation"]["macro_f1"]),
        "saved_rounds": model.num_boosted_rounds(), "n_classes": n_classes,
        "train_rows": len(y_train), "validation_rows": len(y_val), "n_features": X_train.shape[1],
        "weight_strategy": config.get("weight_strategy", "none"), "class_weights": weights.tolist(),
        "preprocessing": "Numeric float32 input; native missing-value handling; quantile cuts fit on training only",
        "selection_metric": "unweighted validation macro F1 over all encoded classes",
        "history": history,
        "rf_caveat": "XGBoost forest uses no replacement subsampling and a second-order objective; not sklearn RandomForest" if candidate == "random_forest" else None,
    }
    directory.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("xb") as stream:
        stream.write(model.save_raw(raw_format="ubj"))
    with metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return {
        **metadata, "predictions": probabilities.argmax(axis=1).astype(np.int32),
        "probabilities": probabilities, "checkpoint_path": str(checkpoint),
        "metadata_path": str(metadata_path),
    }


def predict_tree(checkpoint_path: str | Path, X: np.ndarray) -> np.ndarray:
    """Reload a saved tree model and produce class indices using CUDA."""
    import xgboost as xgb

    model = xgb.Booster(model_file=str(checkpoint_path))
    model.set_param({"device": "cuda:0", "nthread": 4})
    _require_cuda(model)
    predictions = model.predict(xgb.DMatrix(_features(X), nthread=4))
    if predictions.ndim != 2:
        raise ValueError("Expected a saved multiclass softprob model")
    return predictions.argmax(axis=1).astype(np.int32)
