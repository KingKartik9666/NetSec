"""CUDA research candidates: upstream B-spline KAN and a DataSense HEM adaptation.

HEM's original neural classes are imported without editing upstream. Its closed-set
vote is vectorized here and checked against upstream in tests. This is a new-data
adaptation, not a reproduction of published UQ/IoT-NID or unknown-attack results.
CPU preprocessing is intentional; neural optimization and neighbor search use CUDA.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, FunctionTransformer


ROOT = Path(__file__).resolve().parents[2]
HEM_SHA = "8513ad39c904dbe427727ee6f8470ee8a1ce8e0d"
KAN_SHA = "7b6ce1c87f18c8bc90c208f6b494042344216b11"


def class_weights(y: np.ndarray, n_classes: int, strategy: str) -> np.ndarray:
    """Train-only inverse-frequency weights, expectation normalized then capped.

The cap deliberately takes precedence over exact post-cap unit expectation.
Missing training classes are an error rather than silently assigned zero weight.
"""
    powers = {"none": 0.0, "sqrt_balanced": 0.5, "balanced": 1.0}
    if strategy not in powers:
        raise ValueError(f"Unknown weight_strategy: {strategy}")
    y = np.asarray(y, dtype=np.int64)
    if y.ndim != 1 or len(y) == 0 or np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError("Training labels must be a nonempty vector in [0,n_classes).")
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Every requested class must occur in training data.")
    weights = (len(y) / (n_classes * counts)) ** powers[strategy]
    weights /= np.dot(weights, counts) / len(y)
    return np.minimum(weights, 20.0).astype(np.float32)


def _source_module(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required upstream source is missing: {path}")
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path.resolve():
            raise ImportError(f"Unexpected module already loaded under {name}")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_classes(candidate: str):
    if candidate == "kan":
        source = ROOT / "external/efficient-kan/src/efficient_kan/kan.py"
        return (_source_module("_imdai_upstream_efficient_kan", source).KAN,)
    directory = ROOT / "external/HEM/model_evaluation"
    binary = _source_module("_imdai_hem_binary", directory / "binary_classifier.py")
    multi = _source_module("_imdai_hem_multiclass", directory / "multiclass_classifier.py")
    return binary.BinaryClassifier, multi.ExpandableMultiClassClassifier


def _require_cuda():
    # Must precede CUDA context creation for deterministic CUDA matrix multiplication.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these research candidates; no CPU fallback.")
    return torch


def _seed(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def hem_closed_set_vote(
    multi: np.ndarray, binary: np.ndarray, knn: np.ndarray, benign_index: int
) -> np.ndarray:
    """Source-equivalent HEM vote with an arbitrary explicit benign class index.

The binary label remains 0=benign, 1=attack. No novelty thresholds are used.
"""
    multi, binary, knn = (np.asarray(a).reshape(-1) for a in (multi, binary, knn))
    if not (multi.shape == binary.shape == knn.shape):
        raise ValueError("Submodel prediction lengths must match.")
    benign_vote = (binary == 0) & ((multi == benign_index) | (knn == benign_index))
    attack_vote = np.where(knn != benign_index, knn, multi)
    return np.where(benign_vote, benign_index, attack_vote).astype(np.int64)


def cuda_knn_predict(
    X: np.ndarray, references: np.ndarray, labels: np.ndarray,
    n_classes: int, k: int, batch_size: int = 256, *, device: str = "cuda",
) -> np.ndarray:
    """Exact Euclidean KNN over the supplied training reference subset on CUDA.

Distance ties use earlier stored reference index, class-vote ties use lowest
class index. Distances are computed directly to avoid cancellation of small
distances in the matrix multiplication identity.
"""
    if device == "cpu":
        import torch
        torch.set_num_threads(2)
        batch_size = min(batch_size, 16)
    elif device == "cuda":
        torch = _require_cuda()
    else:
        raise ValueError("Explicit inference device must be cpu or cuda")
    if k < 1 or k > len(references) or batch_size < 1:
        raise ValueError("Invalid KNN k or batch_size.")
    reference_tensor = torch.as_tensor(np.asarray(references), dtype=torch.float32, device=device)
    label_tensor = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=device)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            query = torch.as_tensor(np.asarray(X[start:start + batch_size]), dtype=torch.float32, device=device)
            distances = torch.cdist(query, reference_tensor, compute_mode="donot_use_mm_for_euclid_dist")
            # argmin preserves the first-reference tie rule without allocating a
            # complete sorted index matrix for the full-training k=1 study.
            neighbors = (torch.argmin(distances, dim=1, keepdim=True) if k == 1
                         else torch.argsort(distances, dim=1, stable=True)[:, :k])
            neighbor_labels = label_tensor[neighbors]
            counts = (neighbor_labels[:, :, None] == torch.arange(n_classes, device=device)[None, None, :]).sum(1)
            outputs.append(counts.argmax(1).cpu().numpy())
    del reference_tensor, label_tensor
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.int64)


def _clean(X: np.ndarray) -> np.ndarray:
    X = np.array(X, dtype=np.float32, copy=True)
    X[~np.isfinite(X)] = np.nan
    return X


def signed_log1p(values: np.ndarray) -> np.ndarray:
    """Compress heavy-tailed measurements without inventing values or losing signs."""
    return np.sign(values) * np.log1p(np.abs(values))


def _fit_preprocessing(candidate, X_train, X_val, config, seed):
    scaling = config.get("scaling", "minmax")
    if scaling not in {"minmax", "log_minmax", "quantile"}:
        raise ValueError("Unknown neural scaling strategy")
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scaling == "log_minmax":
        steps.append(("log", FunctionTransformer(signed_log1p)))
    elif scaling == "quantile":
        steps.append(("quantile", QuantileTransformer(n_quantiles=min(1000, len(X_train)),
                      output_distribution="uniform", subsample=10000, random_state=seed)))
    steps.append(("scaler", MinMaxScaler(feature_range=(-1, 1) if candidate == "kan" else (0, 1), clip=True)))
    if candidate == "hem":
        dimensions = min(int(config.get("pca_components", 32)), X_train.shape[1], len(X_train) - 1)
        if dimensions < 1:
            raise ValueError("At least two training rows are required for HEM PCA.")
        steps.append(("pca", PCA(n_components=dimensions, svd_solver="full", random_state=seed)))
    pipeline = Pipeline(steps)
    train = pipeline.fit_transform(_clean(X_train)).astype(np.float32)
    val = pipeline.transform(_clean(X_val)).astype(np.float32)
    return pipeline, train, val


def _models(candidate, input_size, n_classes, config, torch, device="cuda"):
    if candidate == "kan":
        (KAN,) = _load_classes(candidate)
        return {"kan": KAN(
            [input_size, *config.get("hidden", [64, 32]), n_classes],
            grid_size=int(config.get("grid_size", 5)), spline_order=int(config.get("spline_order", 3)),
        ).to(device)}
    Binary, Multiclass = _load_classes(candidate)
    return {
        "binary": Binary(input_size, config.get("binary_hidden", [1024, 256, 64, 16])).to(device),
        "multiclass": Multiclass(input_size, config.get("hidden", [512, 256, 32, 16]), n_classes).to(device),
    }


def _neural_predictions(models, X, batch_size, torch):
    results = {name: [] for name in models}
    device = next(next(iter(models.values())).parameters()).device
    for model in models.values():
        model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            for name, model in models.items():
                logits = model(batch)
                predictions = (logits[:, 0] >= 0).long() if name == "binary" else logits.argmax(1)
                results[name].append(predictions.cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in results.items()}


def _reference_indices(y, max_rows, seed, *, original_training_rows=None):
    y = np.asarray(y)
    if original_training_rows is not None:
        if not 1 <= original_training_rows <= len(y):
            raise ValueError("original_training_rows must identify a nonempty original prefix.")
        original = y[:original_training_rows]
        if set(np.unique(original)) != set(np.unique(y)):
            raise ValueError("Every training class must occur in original observations.")
        y = original
    if len(y) <= max_rows:
        return np.arange(len(y), dtype=np.int64)
    if max_rows < len(np.unique(y)):
        raise ValueError("knn_max_rows must accommodate every training class.")
    # StratifiedShuffleSplit approximates original training proportions.
    try:
        selection, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=max_rows, random_state=seed).split(np.zeros(len(y)), y))
    except ValueError:
        selection = np.empty(0, dtype=np.int64)
    if set(np.unique(y[selection])) != set(np.unique(y)):
        # Singleton classes cannot use StratifiedShuffleSplit. Reserve one real
        # observation per class, then sample the remaining original rows.
        rng = np.random.default_rng(seed)
        required = np.array([rng.choice(np.flatnonzero(y == label)) for label in np.unique(y)])
        remaining = np.setdiff1d(np.arange(len(y)), required)
        selection = np.concatenate([required, rng.choice(remaining, max_rows - len(required), replace=False)])
    return np.sort(selection)


def _training_monitor_indices(total_rows, original_training_rows, seed):
    """A fixed small original-observation sample, separate from augmentation."""
    if not 1 <= original_training_rows <= total_rows:
        raise ValueError("original_training_rows must identify a nonempty original prefix.")
    if original_training_rows <= 4096:
        return np.arange(original_training_rows, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(original_training_rows, 4096, replace=False))


def _make_scheduler(optimizer, name, epochs, torch):
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=2)
    raise ValueError("scheduler must be none, cosine or plateau")


def train_neural(
    candidate: str, X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray, n_classes: int,
    benign_index: int, config: dict, seed: int, output_dir: Path,
) -> dict[str, Any]:
    """Train one CUDA candidate; retain and return validation-best predictions.

No test set is accepted. HEM's two neural components are checkpointed jointly
at the epoch with highest complete ensemble validation macro-F1.
"""
    candidate = candidate.lower()
    if candidate not in {"kan", "hem"}:
        raise ValueError("candidate must be kan or hem")
    if n_classes < 2 or not 0 <= benign_index < n_classes:
        raise ValueError("Require >=2 classes and a valid benign_index.")
    epochs, batch_size = int(config.get("epochs", 12)), int(config.get("batch_size", 512))
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    y_train, y_val = np.asarray(y_train, dtype=np.int64), np.asarray(y_val, dtype=np.int64)
    weights = class_weights(y_train, n_classes, config.get("weight_strategy", "none"))
    if len(X_train) != len(y_train) or len(X_val) != len(y_val) or not len(y_val):
        raise ValueError("Features/labels must match and validation cannot be empty.")
    original_rows = int(config.get("original_training_rows", len(y_train)))
    monitor_ids = _training_monitor_indices(len(y_train), original_rows, seed)
    torch = _require_cuda()
    _seed(torch, seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    runtime_seconds = min(float(config.get("runtime_seconds", float("inf"))),
                          float(os.environ.get("IMDAI_FIT_DEADLINE", "inf")) - time.time() - 90)
    deadline = start + runtime_seconds
    if runtime_seconds <= 0:
        raise TimeoutError("Insufficient fit budget before any validated checkpoint exists")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    preprocessing, train, val = _fit_preprocessing(candidate, X_train, X_val, config, seed)
    joblib.dump(preprocessing, output_dir / "preprocessing.joblib")
    models = _models(candidate, train.shape[1], n_classes, config, torch)
    parameters = [p for model in models.values() for p in model.parameters()]
    device = parameters[0].device
    optimizer = torch.optim.AdamW(parameters, lr=float(config.get("lr", 0.001)), weight_decay=float(config.get("weight_decay", 0.0001)))
    selection = config.get("selection_metric", "macro_f1")
    if selection not in {"macro_f1", "accuracy"}:
        raise ValueError("selection_metric must be accuracy or macro_f1")
    scheduler_name = config.get("scheduler", "none")
    scheduler = _make_scheduler(optimizer, scheduler_name, epochs, torch)
    from .research_losses import multiclass_loss, binary_loss
    loss_options = {"loss": config.get("loss", "native_ce"), "smoothing": float(config.get("smoothing", .05)),
                    "gamma": float(config.get("focal_gamma", config.get("gamma", 2.)))}
    weight_tensor = torch.as_tensor(weights, device=device)
    binary_y = (y_train != benign_index).astype(np.int64)
    binary_weights = class_weights(binary_y, 2, config.get("weight_strategy", "none")) if candidate == "hem" else None
    binary_weight_tensor = torch.as_tensor(binary_weights, device=device) if binary_weights is not None else None
    knn_val, knn_monitor, reference_ids = None, None, None
    if time.perf_counter() >= deadline:
        raise TimeoutError("Fit deadline reached during setup before any validated checkpoint exists")
    if candidate == "hem":
        reference_ids = _reference_indices(y_train, int(config.get("knn_max_rows", 20000)), seed,
                                           original_training_rows=original_rows)
        np.savez(output_dir / "knn_references.npz", X=train[reference_ids], y=y_train[reference_ids], train_indices=reference_ids)
        print(f"[HEM seed={seed}] CUDA KNN: {len(reference_ids)} training references, {len(val)} validation rows", flush=True)
        knn_val = cuda_knn_predict(val, train[reference_ids], y_train[reference_ids], n_classes,
                                   int(config.get("knn_k", 3)), int(config.get("knn_batch_size", 256)))
        knn_monitor = cuda_knn_predict(train[monitor_ids], train[reference_ids], y_train[reference_ids], n_classes,
                                       int(config.get("knn_k", 3)), int(config.get("knn_batch_size", 256)))
    rng = np.random.default_rng(seed)
    history, best_score, best_epoch, stale_epochs = [], -1.0, 0, 0
    best_pair = (-1., -1.)
    best_predictions = None
    stop_reason = "epochs_complete"
    patience = int(config.get("patience", 3))
    for epoch in range(1, epochs + 1):
        for model in models.values():
            model.train()
        total_loss = 0.0
        order = rng.permutation(len(train))
        for offset in range(0, len(train), batch_size):
            if time.perf_counter() >= deadline:
                if best_predictions is None:
                    raise TimeoutError(f"Fit deadline reached in {candidate} epoch {epoch} before any validated checkpoint exists")
                stop_reason = "deadline"
                break
            indexes = order[offset:offset + batch_size]
            batch = torch.as_tensor(train[indexes], dtype=torch.float32, device=device)
            target = torch.as_tensor(y_train[indexes], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            if candidate == "kan":
                loss = multiclass_loss(models["kan"](batch), target, weight_tensor, **loss_options)
                regularization = float(config.get("spline_regularization", 0.0))
                if regularization:
                    loss = loss + regularization * models["kan"].regularization_loss()
            else:
                binary_target = torch.as_tensor(binary_y[indexes], dtype=torch.long, device=device)
                loss = (multiclass_loss(models["multiclass"](batch), target, weight_tensor, **loss_options)
                        + binary_loss(models["binary"](batch)[:, 0], binary_target, binary_weight_tensor, **loss_options))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss in {candidate} epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indexes)
        if stop_reason == "deadline":
            break
        if time.perf_counter() >= deadline:
            if best_predictions is None:
                raise TimeoutError(f"Fit deadline reached after {candidate} epoch {epoch} batches before any validated checkpoint exists")
            stop_reason = "deadline"
            break
        predicted = _neural_predictions(models, val, batch_size, torch)
        predictions = predicted["kan"] if candidate == "kan" else hem_closed_set_vote(predicted["multiclass"], predicted["binary"], knn_val, benign_index)
        score = float(f1_score(y_val, predictions, labels=np.arange(n_classes), average="macro", zero_division=0))
        accuracy = float(np.mean(predictions == y_val))
        monitor = _neural_predictions(models, train[monitor_ids], batch_size, torch)
        monitor_predictions = (monitor["kan"] if candidate == "kan" else hem_closed_set_vote(
            monitor["multiclass"], monitor["binary"], knn_monitor, benign_index))
        training_accuracy = float(np.mean(monitor_predictions == y_train[monitor_ids]))
        history.append({"epoch": epoch, "train_loss": total_loss / len(train), "validation_macro_f1": score,
                        "validation_accuracy": accuracy, "training_accuracy": training_accuracy,
                        "training_monitor_rows": len(monitor_ids), "original_training_rows": original_rows,
                        "learning_rate": optimizer.param_groups[0]['lr']})
        if selection == 'accuracy':
            from .csv_monitoring import snapshot
            from .tracking import write_json
            telemetry = snapshot(y_val, predictions, n_classes, benign_index)
            history[-1].update(validation_macro_recall=telemetry['macro_recall'],
                               validation_macro_precision=telemetry['macro_precision'],
                               false_positive_rate=telemetry['false_positive_rate'],
                               attack_missed_as_benign_rate=telemetry['attack_missed_as_benign_rate'])
            write_json(output_dir / 'progress.json', history[-1])
        print(f"[{candidate.upper()} seed={seed}] epoch {epoch}/{epochs} loss={total_loss / len(train):.5f} training_accuracy={training_accuracy:.5f} validation_accuracy={accuracy:.5f} validation_macro_f1={score:.5f}", flush=True)
        pair = (accuracy, score)
        improved = pair > best_pair if selection == "accuracy" else score > best_score + 1e-8
        if improved:
            best_score, best_epoch, stale_epochs = score, epoch, 0
            best_pair = pair
            best_predictions = predictions.copy()
            torch.save({
                "candidate": candidate, "config": dict(config), "seed": seed,
                "input_size": train.shape[1], "n_classes": n_classes, "benign_index": benign_index,
                "best_epoch": best_epoch, "validation_macro_f1": best_score,
                "validation_accuracy": accuracy,
                "selection_criterion": f"complete_candidate_validation_{selection}; accuracy uses macro F1 tie-break; earliest exact tie",
                "state_dicts": {name: {k: v.detach().cpu() for k, v in model.state_dict().items()} for name, model in models.items()},
                "upstream_commit": KAN_SHA if candidate == "kan" else HEM_SHA,
            }, checkpoint_path)
        else:
            stale_epochs += 1
        if (output_dir / "PRUNE").exists():
            stop_reason = "pruned"
            break
        if stale_epochs >= patience:
            stop_reason = "early_stopping"
            break
        if scheduler:
            if scheduler_name == "plateau":
                scheduler.step(accuracy if selection == "accuracy" else score)
            else:
                scheduler.step()
        if time.perf_counter() >= deadline:
            stop_reason = "deadline"
            break
    torch.cuda.synchronize()
    result = {
        "predictions": best_predictions, "training_seconds": time.perf_counter() - start,
        "device": "cuda:0", "device_name": torch.cuda.get_device_name(0),
        "device_evidence": {"parameter_devices": sorted({str(p.device) for p in parameters}), "cuda_version": torch.version.cuda},
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()), "best_epoch": best_epoch,
        "checkpoint_path": str(checkpoint_path), "history": history,
        "class_weights": weights.tolist(), "binary_class_weights": binary_weights.tolist() if binary_weights is not None else None,
        "preprocessing": "train-only median imputer + " + config.get("scaling", "minmax") + (" [-1,1]" if candidate == "kan" else " [0,1] + PCA (CPU)"),
        "selection_criterion": f"complete_candidate_validation_{selection}; HEM component weights saved jointly",
        "backend": "efficient-kan B-spline KAN (fixed grids)" if candidate == "kan" else "HEM DataSense closed-set adaptation: upstream neural classes + CUDA KNN + source-equivalent vote",
        "upstream_commit": KAN_SHA if candidate == "kan" else HEM_SHA,
        "knn_reference_rows": len(reference_ids) if reference_ids is not None else None,
        "knn_reference_selection": (("all original training rows; exact search" if len(reference_ids) == original_rows
                                      else "class-covered original-training subset; exact search within subset")
                                     if candidate == "hem" else None),
        "original_training_rows": original_rows, "training_monitor_rows": len(monitor_ids),
        "training_monitor_selection": "fixed seeded sample of at most 4096 original training observations",
        "loss": loss_options, "scheduler": scheduler_name,
        "pruned": stop_reason == "pruned", "timed_out": stop_reason == "deadline", "stop_reason": stop_reason,
        "unknown_attack_detection": False,
    }
    (output_dir / "training_details.json").write_text(json.dumps({k: v for k, v in result.items() if k != "predictions"}, indent=2), encoding="utf-8")
    del models, parameters, optimizer, train, val
    gc.collect()
    torch.cuda.empty_cache()
    return result


def predict_neural(checkpoint_path: Path, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Reload a saved candidate with its fitted preprocessing and reference data."""
    torch = _require_cuda()
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    candidate, config = checkpoint["candidate"], checkpoint["config"]
    preprocessing = joblib.load(checkpoint_path.parent / "preprocessing.joblib")
    transformed = preprocessing.transform(_clean(X)).astype(np.float32)
    models = _models(candidate, checkpoint["input_size"], checkpoint["n_classes"], config, torch)
    for name, model in models.items():
        model.load_state_dict(checkpoint["state_dicts"][name])
    predicted = _neural_predictions(models, transformed, batch_size, torch)
    if candidate == "kan":
        result = predicted["kan"]
    else:
        with np.load(checkpoint_path.parent / "knn_references.npz") as reference:
            knn = cuda_knn_predict(transformed, reference["X"], reference["y"], checkpoint["n_classes"],
                                   int(config.get("knn_k", 3)), int(config.get("knn_batch_size", 256)))
        result = hem_closed_set_vote(predicted["multiclass"], predicted["binary"], knn, checkpoint["benign_index"])
    del models, transformed
    gc.collect()
    torch.cuda.empty_cache()
    return result
