"""One frozen conventional RF control, with no optimization or GPU substitution.

The base paper specifies the Random Forest family but omits its parameters.
These explicitly chosen sklearn defaults are a documented dataset/platform
adaptation, not an exact reproduction of its unpublished implementation.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any


FIXED_PARAMETERS: dict[str, Any] = {
    "n_estimators": 100, "criterion": "gini", "max_depth": None,
    "min_samples_split": 2, "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0., "max_features": "sqrt", "max_leaf_nodes": None,
    "min_impurity_decrease": 0., "bootstrap": True, "oob_score": False,
    "n_jobs": 2, "random_state": 42, "verbose": 0, "warm_start": False,
    "class_weight": None, "ccp_alpha": 0., "max_samples": None,
    "monotonic_cst": None,
}


def train_fixed_random_forest(
    X_train, y_train, X_val, y_val, n_classes: int,
    config: dict[str, Any], seed: int, output_dir: Path,
) -> dict[str, Any]:
    """Fit all supplied training rows once and retain CPU reload evidence.

No test partition is accepted. The median imputer fits training only. RF has
no validation stopping or parameter search. Its CPU execution is explicit:
sklearn RandomForestClassifier has no native CUDA training backend.
"""
    if seed != 42 or not isinstance(config, dict):
        raise ValueError("The fixed RF baseline requires seed 42 and an object configuration.")
    for parameter, value in config.items():
        if parameter not in FIXED_PARAMETERS or value != FIXED_PARAMETERS[parameter]:
            raise ValueError(f"The fixed RF baseline cannot tune parameter {parameter!r}.")
    directory = Path(output_dir)
    checkpoint = directory / "model.joblib"
    details_path = directory / "training_details.json"
    if checkpoint.exists() or details_path.exists():
        raise FileExistsError(f"Fixed RF artifacts already exist in {directory}.")
    import joblib
    import numpy as np
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    train = np.asarray(X_train, dtype=np.float32)
    validation = np.asarray(X_val, dtype=np.float32)
    if (train.ndim != 2 or validation.ndim != 2 or len(train) == 0 or len(validation) == 0
            or train.shape[1] == 0 or train.shape[1] != validation.shape[1]
            or np.isinf(train).any() or np.isinf(validation).any()):
        raise ValueError("Fixed RF requires aligned, nonempty numeric train/validation arrays without infinity.")
    if not isinstance(n_classes, int) or isinstance(n_classes, bool) or n_classes < 2:
        raise ValueError("Fixed RF requires at least two declared classes.")

    def labels(values, count: int, name: str):
        encoded = np.asarray(values)
        if (encoded.ndim != 1 or len(encoded) != count or not np.issubdtype(encoded.dtype, np.number)
                or not np.isfinite(encoded).all() or np.any(encoded != np.floor(encoded))
                or np.any(encoded < 0) or np.any(encoded >= n_classes)):
            raise ValueError(f"{name} must contain one valid integer class index per row.")
        return encoded.astype(np.int64)

    train_labels = labels(y_train, len(train), "Training labels")
    validation_labels = labels(y_val, len(validation), "Validation labels")
    if len(np.unique(train_labels)) != n_classes:
        raise ValueError("Every declared RF class must occur in training.")
    fitted = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("classifier", RandomForestClassifier(**FIXED_PARAMETERS)),
    ])
    started = time.perf_counter()
    fitted.fit(train, train_labels)
    training_seconds = time.perf_counter() - started
    predicted_at = time.perf_counter()
    probabilities = fitted.predict_proba(validation)
    predictions = probabilities.argmax(axis=1).astype(np.int32)
    inference_seconds = time.perf_counter() - predicted_at
    directory.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("xb") as stream:
        joblib.dump(fitted, stream, compress=3)
    reload_started = time.perf_counter()
    reloaded = joblib.load(checkpoint)
    reload_rows = min(128, len(validation))
    reload_probabilities = reloaded.predict_proba(validation[:reload_rows])
    equal_labels = np.array_equal(reload_probabilities.argmax(1), predictions[:reload_rows])
    equal_scores = np.allclose(reload_probabilities, probabilities[:reload_rows], rtol=1e-12, atol=1e-12)
    if not equal_labels or not equal_scores:
        raise RuntimeError("Fixed RF saved checkpoint failed CPU prediction reload verification.")
    result = {
        "candidate": "random_forest", "backend": "sklearn RandomForestClassifier fixed baseline",
        "parameters": dict(FIXED_PARAMETERS), "requested_config": dict(config),
        "seed": seed, "device": "cpu", "device_evidence": {"backend": "sklearn CPU", "n_jobs": 2},
        "sklearn_version": sklearn.__version__, "training_seconds": training_seconds,
        "inference_seconds": inference_seconds, "inference_rows": len(validation),
        "train_rows": len(train), "training_rows": len(train), "validation_rows": len(validation_labels),
        "n_features": train.shape[1], "n_classes": n_classes,
        "checkpoint_path": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_reload_verified": True, "reload_prediction_rows": reload_rows,
        "reload_probability_tolerance": 1e-12,
        "reload_and_predict_seconds": time.perf_counter() - reload_started,
        "tree_count": len(fitted.named_steps["classifier"].estimators_),
        "total_nodes": sum(tree.tree_.node_count for tree in fitted.named_steps["classifier"].estimators_),
        "preprocessing": "train-only median imputation; no scaling, selection, weighting or augmentation",
        "baseline_status": "fixed conventional RF; paper hyperparameters undisclosed, selected defaults documented",
        "unknown_attack_detection": False, "test_evaluated": False,
        "predictions": predictions, "probabilities": probabilities,
    }
    with details_path.open("x", encoding="utf-8") as stream:
        json.dump({key: value for key, value in result.items() if key not in {"predictions", "probabilities"}}, stream, indent=2)
    return result
