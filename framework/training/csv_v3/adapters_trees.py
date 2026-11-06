"""Cascade v3 adapter for the tree families: CUDA XGBoost and the fixed sklearn RandomForest control.

Families
--------
``xgboost``
    Trained by ``framework.training.csv_xgboost.train_csv_xgboost`` (CUDA enforced by the trainer before
    training and after every round). The checkpoint is the round with the best tune accuracy (macro F1 over
    all declared classes breaks ties), saved as ``model.ubj``. ``predict_cpu`` reloads it with
    ``xgboost.Booster`` and ``set_param(device='cpu', nthread=2)``, the same pattern as
    ``experiment.train_csv_models.reload_outputs``; it never touches CUDA.
``random_forest``
    The fixed sklearn control, trained by ``framework.training.fixed_random_forest.train_fixed_random_forest``
    on CPU (sklearn has no CUDA backend; this is the documented exception in ``contract.py``). The recipe is
    frozen: 100 trees, unlimited depth, min_samples_leaf 1, sqrt features, no class weight, n_jobs 2,
    random_state 42, train-only median imputation. Seed must be 42. No overrides are possible.

Outputs of ``predict_cpu`` are genuine class probabilities for both families (XGBoost ``multi:softprob``;
RandomForest mean leaf class frequencies). Neither family is HEM-derived, so no one-hot vote is ever returned.

Recipes
-------
``default_config`` reads ``artifacts/training/all_models_wrapup_20261023_v1/selected.json`` at call time and
returns that family's frozen ``config`` unchanged except:

* XGBoost ``runtime_seconds`` is 3600 instead of 1500 (the frozen fit hit the cap at 215 of 1000 rounds with
  its best round being the last one). Every other value, including ``early_stopping_rounds`` 60 and the
  1000-round cap, is kept. ``selection_metric`` is kept for the record; the XGBoost trainer ignores it.

The returned dict also carries a ``_source_recipe`` provenance entry. Keys starting with ``_`` are stripped
by ``fit`` before the trainer sees the config.

Task "8" (recon specialist, no benign class)
--------------------------------------------
Only ``xgboost`` supports it. ``TaskData.benign_index`` is ``None`` there, and the trainer's ``benign_index``
is used only for per-round telemetry (``csv_monitoring.snapshot``), never for fitting or selection. This
adapter passes ``benign_index=-1``: ``progress.json`` / ``history.json`` then show
``false_positive_rate=None`` and ``attack_missed_as_benign_rate=0.0``; both are meaningless for task "8" and
must not be reported. ``random_forest`` raises ``NotImplementedError`` for task "8".

Files written in ``out_dir``
----------------------------
``model.ubj`` or ``model.joblib`` (checkpoint), trainer side files (``progress.json`` and
``actual_xgboost_config.json`` for XGBoost; ``training_details.json`` for RF), ``model_schema.json``
(feature names, classes, widths, checkpoint sha256: used by ``predict_cpu``), ``tune_predictions.npy`` and
``tune_probabilities.npy`` (the trainer's own CPU outputs on ``X_tune``), ``history.json`` (XGBoost per-round
telemetry) and ``fit_record.json`` (the dict returned by ``fit``).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contract import TaskData

ROOT = Path(__file__).resolve().parents[3]
SELECTED_RELATIVE = "artifacts/training/all_models_wrapup_20261023_v1/selected.json"
SELECTED = ROOT / SELECTED_RELATIVE

FAMILIES = ("xgboost", "random_forest")
CHECKPOINTS = {"xgboost": "model.ubj", "random_forest": "model.joblib"}
SCHEMA_FILE = "model_schema.json"
RECORD_FILE = "fit_record.json"
XGBOOST_RUNTIME_SECONDS = 3600          # frozen recipe: 1500 (binding; 215/1000 rounds)
RF_SEED = 42
# Files whose presence means a previous fit already used this directory.
_GUARDED = ("model.ubj", "model.joblib", SCHEMA_FILE, RECORD_FILE, "training_details.json",
            "tune_predictions.npy", "tune_probabilities.npy")
# Bulky entries removed from the trainer result before it is returned (saved as files where useful).
_DROP = ("predictions", "probabilities", "history", "class_weights", "requested_config")


def _family(family: str) -> str:
    if family not in FAMILIES:
        raise ValueError(f"adapters_trees handles {FAMILIES}, not {family!r}")
    return family


def _check_task(family: str, task: str) -> None:
    if task not in ("77", "8", "84"):
        raise ValueError(f"unknown task {task!r}")
    if task == "8" and family != "xgboost":
        raise NotImplementedError(f"{family} is not provided for the recon specialist task '8'")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance(family: str, entry: dict, selected_sha: str) -> dict:
    keys = ("family", "arm", "seed", "phase", "trial_number", "job", "recipe", "study", "checkpoint",
            "run_directory", "source_result", "source_result_sha256")
    record = {key: entry[key] for key in keys if key in entry}
    record.update(selected_json=SELECTED_RELATIVE, selected_json_sha256=selected_sha,
                  frozen_config=copy.deepcopy(entry["config"]))
    if "selection_metrics" in entry:
        metrics = entry["selection_metrics"]
        record["frozen_tune_metrics"] = {k: metrics[k] for k in ("accuracy", "macro_f1", "balanced_accuracy", "rows")
                                         if k in metrics}
    if family == "xgboost":
        details = entry.get("details", {})
        record["frozen_outcome"] = {k: details.get(k) for k in ("stop_reason", "saved_rounds", "best_iteration",
                                                                 "training_seconds", "xgboost_version")}
        record["changes"] = {"runtime_seconds": {
            "from": entry["config"]["runtime_seconds"], "to": XGBOOST_RUNTIME_SECONDS,
            "reason": "frozen fit stopped on the runtime cap at 215/1000 rounds with best_iteration = last round"}}
    else:
        record["changes"] = {}
    return record


def default_config(family: str, task: str) -> dict:
    """Frozen selected recipe for ``family`` (read now from selected.json) plus ``_source_recipe`` provenance."""
    _family(family)
    _check_task(family, task)
    raw = SELECTED.read_bytes()
    selected = json.loads(raw)
    if family not in selected:
        raise KeyError(f"{SELECTED_RELATIVE} has no entry for {family!r}")
    entry = selected[family]
    if entry.get("family") != family or entry.get("status") != "completed" or not isinstance(entry.get("config"), dict):
        raise ValueError(f"{SELECTED_RELATIVE}[{family!r}] is not a completed recipe with a config")
    config = copy.deepcopy(entry["config"])
    if family == "xgboost":
        if config.get("runtime_seconds") != 1500:
            raise ValueError("selected XGBoost recipe changed: expected the recorded 1500 s cap")
        config["runtime_seconds"] = XGBOOST_RUNTIME_SECONDS
    else:
        from framework.training.fixed_random_forest import FIXED_PARAMETERS
        for key, value in config.items():
            if FIXED_PARAMETERS.get(key, object()) != value:
                raise ValueError(f"selected RF recipe disagrees with FIXED_PARAMETERS on {key!r}")
        if entry.get("seed") != RF_SEED:
            raise ValueError("selected RF recipe is not seed 42")
    config["_source_recipe"] = _provenance(family, entry, hashlib.sha256(raw).hexdigest())
    return config


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.size > 256:
            raise ValueError("refusing to put a large array in the fit record")
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())          # recurse so a numpy NaN/inf also becomes None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _tune_scores(y: np.ndarray, pred: np.ndarray, n_classes: int) -> dict:
    present = np.unique(y)
    matrix = np.bincount(n_classes * y + pred, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    actual, proposed, correct = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    f1 = 2 * correct[present] / (actual[present] + proposed[present])
    return dict(tune_accuracy=float(np.mean(pred == y)), tune_macro_f1=float(f1.mean()),
                tune_classes_present=int(present.size),
                tune_classes_absent=[int(c) for c in np.setdiff1d(np.arange(n_classes), present)])


def _refuse_existing(out_dir: Path) -> None:
    if out_dir.exists():
        found = [name for name in _GUARDED if (out_dir / name).exists()]
        if found:
            raise FileExistsError(f"{out_dir} already holds fit artifacts {found}; use a new directory")


def _require_xgboost_cuda() -> None:
    import xgboost as xgb
    if not xgb.build_info().get("USE_CUDA", False):
        raise RuntimeError("installed XGBoost has no CUDA support")
    try:
        import torch
    except ImportError:          # the trainer still verifies the actual device every round
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; XGBoost training refuses to run on CPU")


def fit(family: str, data: TaskData, config: dict, seed: int, out_dir: Path) -> dict:
    """Train one model into ``out_dir`` and return a JSON-serialisable record (see module docstring)."""
    _family(family)
    data.validate()
    _check_task(family, data.task)
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")
    out_dir = Path(out_dir)
    _refuse_existing(out_dir)
    if data.y_tune.size == 0 or data.y_fit.size == 0:
        raise ValueError("fit and tune partitions must be non-empty")
    if len(data.X_fit) != len(data.y_fit) or len(data.X_tune) != len(data.y_tune):
        raise ValueError("feature and label row counts differ")
    n_classes, n_features = int(data.n_classes), len(data.feature_names)

    fresh = default_config(family, data.task)
    provenance = fresh.pop("_source_recipe")
    supplied = config.get("_source_recipe")
    if supplied is not None and _jsonable(supplied) != _jsonable(provenance):
        raise ValueError("config carries a _source_recipe that no longer matches selected.json")
    trainer_config = {k: copy.deepcopy(v) for k, v in config.items() if not str(k).startswith("_")}
    overrides = {k: {"default": fresh.get(k), "used": trainer_config.get(k)}
                 for k in sorted(set(fresh) | set(trainer_config)) if fresh.get(k, "<absent>") != trainer_config.get(k, "<absent>")}
    provenance = dict(provenance, overrides_vs_default=overrides)

    out_dir.mkdir(parents=True, exist_ok=True)
    X_fit = np.ascontiguousarray(data.X_fit, dtype=np.float32)
    X_tune = np.ascontiguousarray(data.X_tune, dtype=np.float32)
    benign_passed = None
    if family == "xgboost":
        _require_xgboost_cuda()
        if trainer_config.get("device", "cuda") not in ("cuda", "cuda:0"):
            raise ValueError("XGBoost must train on CUDA")
        from framework.training.csv_xgboost import train_csv_xgboost
        benign_passed = -1 if data.benign_index is None else int(data.benign_index)
        result = train_csv_xgboost(X_fit, data.y_fit, X_tune, data.y_tune, n_classes, trainer_config,
                                   int(seed), out_dir, benign_index=benign_passed,
                                   row_weight=data.extras.get("row_weight"))
        if not str(result.get("device", "")).startswith("cuda") or result.get("gpu_training_verified") is not True:
            raise RuntimeError(f"XGBoost did not report CUDA training: {result.get('device')!r}")
        device, stop_reason = str(result["device"]), str(result["stop_reason"])
        (out_dir / "history.json").write_text(json.dumps(_jsonable(result["history"]), indent=1), encoding="utf-8")
    else:
        if int(seed) != RF_SEED:
            raise ValueError("the fixed RandomForest control only runs with seed 42")
        from framework.training.fixed_random_forest import train_fixed_random_forest
        result = train_fixed_random_forest(X_fit, data.y_fit, X_tune, data.y_tune, n_classes, trainer_config,
                                           RF_SEED, out_dir)
        if result.get("device") != "cpu":
            raise RuntimeError("unexpected RF device")
        device, stop_reason = "cpu", "fixed_recipe_complete"

    probabilities = np.asarray(result["probabilities"], dtype=np.float32)
    predictions = np.asarray(result["predictions"], dtype=np.int64)
    if probabilities.shape != (len(data.y_tune), n_classes) or not np.isfinite(probabilities).all():
        raise RuntimeError(f"trainer returned invalid tune probabilities {probabilities.shape}")
    if not np.array_equal(probabilities.argmax(1), predictions):
        raise RuntimeError("trainer predictions disagree with its probabilities")
    np.save(out_dir / "tune_predictions.npy", predictions)
    np.save(out_dir / "tune_probabilities.npy", probabilities)

    checkpoint = CHECKPOINTS[family]
    schema = dict(family=family, task=data.task, checkpoint=checkpoint,
                  checkpoint_sha256=_sha256(out_dir / checkpoint), n_features=n_features, n_classes=n_classes,
                  classes=list(data.classes), feature_names=list(data.feature_names),
                  benign_index=data.benign_index, trainer_benign_index=benign_passed)
    with (out_dir / SCHEMA_FILE).open("x", encoding="utf-8") as stream:
        json.dump(schema, stream, indent=1)

    replay = predict_cpu(family, out_dir, X_tune)
    if not np.array_equal(replay.argmax(1), predictions) or not np.allclose(replay, probabilities, rtol=0, atol=1e-6):
        raise RuntimeError("CPU reload of the checkpoint does not reproduce the fit-time tune predictions")

    extras = {k: v for k, v in result.items() if k not in _DROP and k not in ("device", "stop_reason", "training_seconds")}
    extras["checkpoint_path"] = checkpoint
    record = dict(
        family=family, task=data.task, seed=int(seed), checkpoint=checkpoint, config_used=trainer_config,
        training_seconds=float(result["training_seconds"]), device=device, stop_reason=stop_reason,
        **_tune_scores(data.y_tune, predictions, n_classes),
        n_features=n_features, n_classes=n_classes, fit_rows=int(len(data.y_fit)), tune_rows=int(len(data.y_tune)),
        trainer_benign_index=benign_passed,
        telemetry_note=("benign_index=-1 passed for telemetry only (no benign class): false_positive_rate and "
                        f"attack_missed_as_benign_rate in history are meaningless for task {data.task!r}")
        if benign_passed == -1 else None,
        cpu_reload_verified=True, cpu_reload_rows=int(len(data.y_tune)),
        checkpoint_sha256=schema["checkpoint_sha256"], source_recipe=provenance, trainer=extras,
    )
    record = _jsonable(record)
    json.dumps(record, allow_nan=False)         # strict JSON: raises rather than writing NaN/Infinity
    with (out_dir / RECORD_FILE).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=1, allow_nan=False)
    return record


def predict_cpu(family: str, out_dir: Path, X: np.ndarray) -> np.ndarray:
    """Reload the checkpoint in ``out_dir`` on CPU and return float32 probabilities (n, n_classes)."""
    _family(family)
    out_dir = Path(out_dir)
    schema = json.loads((out_dir / SCHEMA_FILE).read_text(encoding="utf-8"))
    if schema["family"] != family:
        raise ValueError(f"{out_dir} holds a {schema['family']} model, not {family}")
    n_features, n_classes = int(schema["n_features"]), int(schema["n_classes"])
    checkpoint = out_dir / schema["checkpoint"]
    if _sha256(checkpoint) != schema["checkpoint_sha256"]:
        raise RuntimeError(f"{checkpoint} changed since fit")
    X = np.array(X, dtype=np.float32, order="C", copy=True)
    if X.ndim != 2 or X.shape[1] != n_features or len(X) == 0:
        raise ValueError(f"X must be non-empty (n, {n_features}), got {X.shape}")
    if np.isinf(X).any():
        raise ValueError("X contains inf; the contract allows NaN only")

    if family == "xgboost":
        import xgboost as xgb
        booster = xgb.Booster(model_file=str(checkpoint))
        booster.set_param({"device": "cpu", "nthread": 2})
        learner = json.loads(booster.save_config())["learner"]
        if learner["generic_param"]["device"] != "cpu":
            raise RuntimeError("XGBoost booster is not on CPU")
        if learner["objective"]["name"] != "multi:softprob":
            raise RuntimeError("checkpoint is not a multi:softprob model")
        if int(learner["learner_model_param"]["num_class"]) != n_classes or booster.num_features() != n_features:
            raise RuntimeError("checkpoint class or feature count differs from model_schema.json")
        probabilities = booster.predict(xgb.DMatrix(X, nthread=2))
    else:
        import joblib
        model = joblib.load(checkpoint)
        forest = model.steps[-1][1]
        if list(forest.classes_) != list(range(n_classes)) or model.n_features_in_ != n_features:
            raise RuntimeError("RF checkpoint class or feature layout differs from model_schema.json")
        probabilities = model.predict_proba(X)

    probabilities = np.ascontiguousarray(probabilities, dtype=np.float32)
    if probabilities.shape != (len(X), n_classes):
        raise RuntimeError(f"unexpected probability shape {probabilities.shape}")
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1.0, atol=1e-4):
        raise RuntimeError("non-finite or unnormalised probabilities")
    return probabilities
