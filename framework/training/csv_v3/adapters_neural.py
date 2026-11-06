"""Cascade v3 adapters for the neural families ``kan`` (efficient-KAN) and ``hem`` (HEM-derived).

Both families are trained by the existing ``framework.training.neural.train_neural`` on CUDA; nothing
is reimplemented here. ``default_config`` returns the frozen selected recipe from
``artifacts/training/all_models_wrapup_20261023_v1/selected.json`` (read at call time) with the
recorded caps unchanged (KAN 1500 s, HEM 3300 s). Only tasks "77" and "84" are supported; task "8"
(recon specialist without a benign class) raises ``NotImplementedError`` for both families.

Config layout: a flat dict of ``train_neural`` keys plus one reserved key ``source_recipe`` (provenance
of the frozen recipe). ``fit`` removes ``source_recipe`` before training and reports every difference
between the recorded recipe and the config actually used under ``config_changes_vs_recorded``.

Synthetic rows: if the runner appends synthetic (augmented) rows after the real ones, it must pass
``data.extras["original_training_rows"]`` (the length of the real prefix of ``X_fit``). The value is
forwarded to ``train_neural`` so the HEM KNN reference set (and the training monitor sample) use only
real rows. Without that key every fitting row is treated as real.

CPU replay (``predict_cpu``):

* ``kan`` -- ``experiment.train_csv_models.reload_outputs`` (CPU, ``map_location='cpu'``); returns
  raw, uncalibrated softmax probabilities as float32 ``(n, n_classes)``.
* ``hem`` -- HEM's prediction is a hard closed-set vote of three components (binary head,
  multiclass head, exact KNN), so the ensemble has NO class probabilities. ``predict_cpu`` returns a
  float32 ONE-HOT ENCODING OF THE VOTE. Rows sum to 1 but the values are not probabilities and must
  not be calibrated, averaged or reported as confidence. ``hem_cpu_outputs`` exposes the component
  outputs (multiclass softmax, binary sigmoid, KNN ids) for diagnostics. The CPU reload replicates
  ``framework.training.day_worker.hem_cpu_outputs`` using only ``framework.training.neural`` functions,
  because importing ``day_worker`` pulls in modules that pin the v2 DATA path.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from framework.training.csv_v3.contract import TaskData

FAMILIES = ("kan", "hem")
SUPPORTED_TASKS = ("77", "84")
ROOT = Path(__file__).resolve().parents[3]
SELECTED = ROOT / "artifacts/training/all_models_wrapup_20261023_v1/selected.json"
CHECKPOINT = "model.pt"
RECORD = "fit_record.json"
TUNE_PREDICTIONS = "tune_predictions.npy"
# Files whose presence means out_dir already holds (part of) a fit.
_FIT_FILES = (CHECKPOINT, "preprocessing.joblib", "knn_references.npz", "training_details.json", RECORD,
              TUNE_PREDICTIONS, "progress.json")
# Keys the CPU reload / serving adapters read directly; they must be explicit in the config.
_REQUIRED_KEYS = {
    "kan": ("hidden", "grid_size", "spline_order", "selection_metric", "runtime_seconds"),
    "hem": ("hidden", "binary_hidden", "pca_components", "knn_k", "knn_batch_size", "knn_max_rows",
            "selection_metric", "runtime_seconds"),
}
_RETURN_DROP = ("predictions", "history", "class_weights", "binary_class_weights", "checkpoint_path")
CPU_RELOAD_CHECK_ROWS = 128
HEM_OUTPUT_MEANING = ("one-hot encoding of the HEM hard closed-set vote (binary head + multiclass head + "
                      "exact KNN); not class probabilities")


def _check_family_task(family: str, task: str) -> None:
    if family not in FAMILIES:
        raise ValueError(f"unknown neural family {family!r}; expected one of {FAMILIES}")
    if task == "8":
        raise NotImplementedError(f"{family} does not support the no-benign recon specialist task '8'")
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"unknown task {task!r}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------- config
def default_config(family: str, task: str) -> dict:
    """Frozen selected recipe for ``family`` with its provenance under ``source_recipe``."""
    _check_family_task(family, task)
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    entry = selected[family]
    if entry.get("family") != family:
        raise ValueError(f"selected.json entry for {family} names family {entry.get('family')!r}")
    recorded = dict(entry["config"])
    missing = [k for k in _REQUIRED_KEYS[family] if k not in recorded]
    if missing:
        raise ValueError(f"frozen {family} recipe lacks explicit keys {missing}")
    # Deep copies: the returned config must not alias the provenance copy (nested lists such as
    # ``hidden``), otherwise an in-place edit would also rewrite ``recorded_config`` and hide the change.
    config = copy.deepcopy(recorded)
    source_result = entry.get("source_result")
    source_path = ROOT / source_result if source_result else None
    config["source_recipe"] = {
        "selected_json": SELECTED.relative_to(ROOT).as_posix(),
        "selected_json_sha256": _sha256(SELECTED),
        "family": family,
        "trial_number": entry.get("trial_number"),
        "job": entry.get("job"),
        "recipe": entry.get("recipe"),
        "study": entry.get("study"),
        "phase": entry.get("phase"),
        "seed": entry.get("seed"),
        "arm": entry.get("arm"),
        "checkpoint": entry.get("checkpoint"),
        "run_directory": entry.get("run_directory"),
        "source_result": source_result,
        "source_result_sha256": (_sha256(source_path) if source_path is not None and source_path.is_file()
                                 else None),
        "recorded_source_result_sha256": entry.get("source_result_sha256"),
        "recorded_config": copy.deepcopy(recorded),
        "changes_from_recorded": [],
        "arm_note": "recipe was selected on the v2 'strict' row arm; the equivalent row filter is the "
                    "runner's data preparation, not this adapter",
    }
    return config


def _config_changes(recorded: dict, used: dict) -> list[dict]:
    changes = []
    for key in sorted(set(recorded) | set(used)):
        if recorded.get(key, "<absent>") != used.get(key, "<absent>"):
            changes.append({"key": key, "recorded": recorded.get(key, "<absent>"),
                            "used": used.get(key, "<absent>")})
    return changes


# ---------------------------------------------------------------------------------------------- fit
def _require_cuda_torch():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to fit kan/hem; no CPU fallback")
    return torch


def fit(family: str, data: TaskData, config: dict, seed: int, out_dir: Path) -> dict:
    """Train one kan/hem model on CUDA via ``train_neural``; write everything inside ``out_dir``."""
    data.validate()
    _check_family_task(family, data.task)
    if data.benign_index is None:
        raise ValueError(f"{family} requires a benign class (benign_index is None)")
    if len(data.y_tune) == 0 or len(data.X_tune) != len(data.y_tune) or len(data.X_fit) != len(data.y_fit):
        raise ValueError("fit/tune features and labels must match and tune must be non-empty")
    if np.isinf(data.X_fit).any() or np.isinf(data.X_tune).any():
        raise ValueError("inf in features")
    out_dir = Path(out_dir)
    existing = [name for name in _FIT_FILES if (out_dir / name).exists()]
    if existing:
        raise FileExistsError(f"{out_dir} already holds fit output {existing}; refusing to overwrite")

    config = dict(config)
    source_recipe = config.pop("source_recipe", None)
    if source_recipe is None:
        raise ValueError("config must carry 'source_recipe' (start from default_config)")
    if source_recipe.get("family") != family:
        raise ValueError("config source_recipe belongs to another family")
    missing = [k for k in _REQUIRED_KEYS[family] if k not in config]
    if missing:
        raise ValueError(f"{family} config must set {missing} explicitly")
    if config["selection_metric"] != "accuracy":
        raise ValueError("frozen neural recipes select on tune accuracy; selection_metric must be 'accuracy'")

    original_rows = int(data.extras.get("original_training_rows", len(data.y_fit)))
    if not 1 <= original_rows <= len(data.y_fit):
        raise ValueError("extras['original_training_rows'] must be in [1, n_fit]")
    if "original_training_rows" in config and int(config["original_training_rows"]) != original_rows:
        raise ValueError("config original_training_rows disagrees with data.extras")
    config["original_training_rows"] = original_rows
    if family == "hem":
        real = data.y_fit[:original_rows]
        if np.unique(real).size != data.n_classes:
            raise ValueError("HEM KNN references need every class among the real (non-synthetic) rows")
        if int(config["knn_k"]) > min(original_rows, int(config["knn_max_rows"])):
            raise ValueError("knn_k exceeds the number of KNN reference rows")

    torch = _require_cuda_torch()
    from framework.training.neural import train_neural

    out_dir.mkdir(parents=True, exist_ok=True)
    result = train_neural(family, data.X_fit, data.y_fit, data.X_tune, data.y_tune, data.n_classes,
                          data.benign_index, config, seed, out_dir)
    checkpoint = Path(result["checkpoint_path"])
    if checkpoint.resolve() != (out_dir / CHECKPOINT).resolve() or not checkpoint.is_file():
        raise RuntimeError(f"train_neural did not leave {CHECKPOINT} in {out_dir}")
    devices = result["device_evidence"]["parameter_devices"]
    if len(devices) != 1 or not devices[0].startswith("cuda"):
        raise RuntimeError(f"parameters were not on a single CUDA device: {devices}")

    predictions = np.asarray(result["predictions"], dtype=np.int64)
    if predictions.shape != data.y_tune.shape:
        raise RuntimeError("train_neural tune predictions have the wrong shape")
    np.save(out_dir / TUNE_PREDICTIONS, predictions)
    from sklearn.metrics import f1_score

    present = np.unique(data.y_tune)
    tune_accuracy = float(np.mean(predictions == data.y_tune))
    tune_macro_f1 = float(f1_score(data.y_tune, predictions, labels=present, average="macro", zero_division=0))
    tune_macro_f1_all = float(f1_score(data.y_tune, predictions, labels=np.arange(data.n_classes),
                                       average="macro", zero_division=0))

    # CPU reload check on a bounded prefix of the tune rows (mirrors train_csv_models.fit).
    rows = min(CPU_RELOAD_CHECK_ROWS, len(data.y_tune))
    cpu_ids = _predict_cpu(family, out_dir, data.X_tune[:rows], int(data.X_fit.shape[1]),
                           int(data.n_classes)).argmax(1)
    cpu_agreement = int(np.sum(cpu_ids == predictions[:rows]))

    record = {k: v for k, v in result.items() if k not in _RETURN_DROP}
    record.update({
        "family": family, "task": data.task, "seed": int(seed), "checkpoint": CHECKPOINT,
        "config_used": config, "source_recipe": source_recipe,
        "config_changes_vs_recorded": _config_changes(source_recipe["recorded_config"], config),
        # train_neural reports a fixed "cuda:0" string; use the parameter-device evidence instead.
        "training_seconds": float(result["training_seconds"]), "device": devices[0],
        "device_name": result["device_name"], "stop_reason": result["stop_reason"],
        "tune_accuracy": tune_accuracy, "tune_macro_f1": tune_macro_f1,
        "tune_macro_f1_labels": "classes present in y_tune",
        "tune_classes_present": int(present.size),
        "tune_macro_f1_all_classes": tune_macro_f1_all,
        "n_features": int(data.X_fit.shape[1]), "n_classes": int(data.n_classes),
        "benign_index": int(data.benign_index), "classes": list(data.classes),
        "feature_names_sha256": hashlib.sha256("\n".join(data.feature_names).encode()).hexdigest(),
        "n_fit": int(len(data.y_fit)), "n_tune": int(len(data.y_tune)),
        "original_training_rows": original_rows, "synthetic_training_rows": int(len(data.y_fit) - original_rows),
        "tune_predictions": TUNE_PREDICTIONS,
        "cpu_reload_check": {"rows": rows, "agreeing_rows": cpu_agreement, "matches": cpu_agreement == rows},
        "fit_deadline_env": os.environ.get("IMDAI_FIT_DEADLINE"),
        "output_meaning": ("raw uncalibrated softmax probabilities" if family == "kan" else HEM_OUTPUT_MEANING),
        "torch_version": torch.__version__,
    })
    record = json.loads(json.dumps(record, allow_nan=False))  # enforce strict JSON (no NaN/inf)
    (out_dir / RECORD).write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    return record


# ------------------------------------------------------------------------------------------ predict
def _check_X(X: np.ndarray, n_features: int) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != n_features:
        raise ValueError(f"X must be 2-D with {n_features} columns, got {X.shape}")
    X = X.astype(np.float32, copy=False)
    if np.isinf(X).any():
        raise ValueError("X contains inf")
    return X


def _load_record(family: str, out_dir: Path) -> dict:
    record = json.loads((Path(out_dir) / RECORD).read_text(encoding="utf-8"))
    if record["family"] != family:
        raise ValueError(f"{out_dir} holds a {record['family']} fit, not {family}")
    return record


def hem_cpu_outputs(checkpoint: Path, X: np.ndarray) -> tuple[np.ndarray, dict]:
    """CPU-only HEM reload: (vote ids, component outputs). Replicates day_worker.hem_cpu_outputs."""
    import joblib
    import torch
    from scipy.special import expit, softmax

    from framework.training.neural import _clean, _models, cuda_knn_predict, hem_closed_set_vote

    checkpoint = Path(checkpoint)
    torch.set_num_threads(2)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["candidate"] != "hem":
        raise ValueError("checkpoint is not a HEM model")
    transformed = joblib.load(checkpoint.parent / "preprocessing.joblib").transform(_clean(X)).astype(np.float32)
    if not np.isfinite(transformed).all():
        raise FloatingPointError("non-finite values after HEM preprocessing")
    models = _models("hem", saved["input_size"], saved["n_classes"], saved["config"], torch, device="cpu")
    logits = {}
    for name, model in models.items():
        model.load_state_dict(saved["state_dicts"][name])
        model.eval()
        if any(p.device.type != "cpu" for p in model.parameters()):
            raise RuntimeError("HEM reload is not on CPU")
        with torch.inference_mode():
            logits[name] = np.concatenate([model(torch.from_numpy(transformed[i:i + 256])).numpy()
                                           for i in range(0, len(transformed), 256)])
    multi = softmax(logits["multiclass"], axis=1)
    binary = expit(logits["binary"][:, 0])
    with np.load(checkpoint.parent / "knn_references.npz") as refs:
        knn = cuda_knn_predict(transformed, refs["X"], refs["y"], int(saved["n_classes"]),
                               int(saved["config"]["knn_k"]), int(saved["config"]["knn_batch_size"]), device="cpu")
    vote = hem_closed_set_vote(multi.argmax(1), binary >= .5, knn, saved["benign_index"])
    return vote, {"multiclass_probabilities": multi, "binary_probabilities": binary, "knn_predictions": knn}


def predict_cpu(family: str, out_dir: Path, X: np.ndarray) -> np.ndarray:
    """CPU replay. kan: float32 softmax (n, C). hem: float32 one-hot of the hard vote (see module doc).

    Requires the ``fit_record.json`` written by a completed ``fit``: a bare checkpoint left by an
    interrupted fit is refused, and the record's feature/class counts are cross-checked.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown neural family {family!r}")
    out_dir = Path(out_dir)
    if not (out_dir / RECORD).is_file():
        raise FileNotFoundError(f"{out_dir / RECORD} missing; out_dir does not hold a completed fit")
    record = _load_record(family, out_dir)
    return _predict_cpu(family, out_dir, X, int(record["n_features"]), int(record["n_classes"]))


def _predict_cpu(family: str, out_dir: Path, X: np.ndarray, n_features_expected: int,
                 n_classes_expected: int) -> np.ndarray:
    out_dir = Path(out_dir)
    checkpoint = out_dir / CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    import joblib

    n_features = int(joblib.load(out_dir / "preprocessing.joblib").n_features_in_)
    if n_features != n_features_expected:
        raise ValueError("fit record and preprocessing disagree on the feature count")
    X = _check_X(X, n_features)
    import torch

    torch.set_num_threads(2)
    saved_meta = torch.load(checkpoint, map_location="cpu", weights_only=True)
    n_classes = int(saved_meta["n_classes"])
    if saved_meta["candidate"] != family:
        raise ValueError(f"checkpoint holds {saved_meta['candidate']}, not {family}")
    if n_classes != n_classes_expected:
        raise ValueError("fit record and checkpoint disagree on the class count")
    if len(X) == 0:
        return np.empty((0, n_classes), dtype=np.float32)
    if family == "kan":
        from experiment.train_csv_models import reload_outputs

        _, probabilities = reload_outputs("kan", checkpoint, X)
        probabilities = np.asarray(probabilities, dtype=np.float32)
    else:
        vote, _ = hem_cpu_outputs(checkpoint, X)
        if vote.min() < 0 or vote.max() >= n_classes:
            raise ValueError("HEM vote outside the class range")
        probabilities = np.zeros((len(X), n_classes), dtype=np.float32)
        probabilities[np.arange(len(X)), vote] = 1.0
    if probabilities.shape != (len(X), n_classes):
        raise ValueError(f"output shape {probabilities.shape} != {(len(X), n_classes)}")
    if not np.isfinite(probabilities).all():
        raise FloatingPointError(f"non-finite {family} CPU outputs")
    return probabilities
