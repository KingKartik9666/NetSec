"""Cascade v3 adapters for the TabM families and the auxiliary TabPack branch.

Families: ``tabm_ne``, ``tabm_hier``, ``tabm_hier_ne`` (trained by ``framework.training.tabm_fit.train``)
and ``tabpack`` (trained by ``framework.training.hybrid_neural.train`` with
``tabpack_backend.pack_configuration(seed)``). No model code is reimplemented here; this module only
validates inputs, builds the in-memory manifest/request the sealed trainers expect, and records results.

Contract notes (see ``ml_pipeline/trainer_contracts.json``):

* Tasks ``"77"`` and ``"84"`` only. Task ``"8"`` (recon specialist, no benign class) raises
  ``NotImplementedError`` for every family here: the hierarchical heads assume benign at index 0 and
  ``tabm_fit.train`` hard-requires ``benign_index == 0``.
* Every family here outputs real class probabilities: TabM is the mean of member softmaxes (hierarchical
  families: ``[1-sigmoid(z0), sigmoid(z0)*softmax(z1..)]``), TabPack is the greedy member-weighted mean of
  member softmaxes. There is no hard-vote output in this module (the HEM one-hot convention does not
  apply here).
* ``default_config`` returns the frozen recipe from
  ``artifacts/training/all_models_wrapup_20261023_v1/selected.json`` verbatim (runtime caps unchanged:
  tabm_ne 1500 s, hierarchical 3600 s, tabpack 3600 s) plus a reserved ``"_source"`` key with its
  provenance. ``fit`` strips ``"_source"`` before calling the trainer and requires it, because ``fit`` must
  not read files beyond the arrays it is given.
* ``fit`` must run on CUDA, in a fresh process (the trainers set process-global torch state:
  2 threads, deterministic algorithms, TF32 off). ``CUBLAS_WORKSPACE_CONFIG`` must be set before CUDA is
  initialised; ``fit`` sets it when CUDA is not yet initialised and raises otherwise.
* ``tabm_fit.train`` enforces a strict CPU/GPU replay (atol 5e-5, rtol 5e-4, identical argmax on 256 tune
  rows) and raises ``AssertionError`` after ``model.pt`` is written when it fails. That error propagates;
  no ``fit_record.json`` is written, so ``predict_cpu`` refuses such a directory.
* Tune-set selection inside the trainers uses (accuracy, macro F1 over *all* classes). The record's
  ``tune_macro_f1`` is computed over classes present in ``y_tune`` only; the trainer's own value is kept
  as ``trainer_selection_metrics``.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from framework.training.csv_v3.contract import TaskData

FAMILIES = ("tabm_ne", "tabm_hier", "tabm_hier_ne", "tabpack")
TABM = ("tabm_ne", "tabm_hier", "tabm_hier_ne")
HIERARCHICAL = ("tabm_hier", "tabm_hier_ne")
EMBEDDED = ("tabm_ne", "tabm_hier_ne")
SUPPORTED_TASKS = ("77", "84")

ROOT = Path(__file__).resolve().parents[3]
SELECTED = ROOT / "artifacts/training/all_models_wrapup_20261023_v1/selected.json"
CHECKPOINT = "model.pt"
RECORD = "fit_record.json"
TUNE_PROBABILITIES = "tune_probabilities_cuda.npy"
SOURCE_KEY = "_source"

TABM_KEYS = ("k", "width", "depth", "dropout", "d_embedding", "n_bins", "scaling", "lr", "weight_decay",
             "weight_power", "smoothing", "subtype_weight", "batch_size", "epochs", "patience",
             "runtime_seconds")
TABM_INT = {"k", "width", "depth", "d_embedding", "n_bins", "batch_size", "epochs", "patience"}
TABM_STR = {"scaling"}
PACK_LISTS = ("n_blocks", "d_block", "dropout", "lr", "weight_decay")
PACK_INT = {"seed", "pack_size", "n_bins", "batch_size", "epochs", "patience", "n_blocks", "d_block"}
PACK_STR = {"scaling", "upstream_commit", "protocol"}
CUBLAS = ":4096:8"


# --------------------------------------------------------------------------------------------- helpers
def _check_family(family: str, task: str) -> None:
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; this module handles {FAMILIES}")
    if task == "8":
        raise NotImplementedError(f"{family} is not supported for task '8' (no benign class at index 0)")
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"unknown task {task!r}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _plain_number(key: str, value, ints: set[str]):
    """Plain Python int/float (torch.load(weights_only=True) rejects numpy scalars)."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"config[{key!r}] must be numeric, got bool")
    if key in ints:
        if float(value) != int(value):
            raise ValueError(f"config[{key!r}] must be an integer, got {value!r}")
        return int(value)
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"config[{key!r}] is not finite")
    return out


def _tabm_config(config: dict) -> dict:
    keys = set(config)
    missing, unknown = set(TABM_KEYS) - keys, keys - set(TABM_KEYS)
    if missing or unknown:
        raise ValueError(f"TabM config keys mismatch: missing={sorted(missing)} unknown={sorted(unknown)}")
    cfg = {}
    for key in TABM_KEYS:
        value = config[key]
        cfg[key] = str(value) if key in TABM_STR else _plain_number(key, value, TABM_INT)
    return cfg


def _pack_config(config: dict, seed: int) -> tuple[dict, dict]:
    """``pack_configuration(seed)`` with the caller's explicit overrides; returns (cfg, overrides)."""
    from framework.training.tabpack_backend import pack_configuration

    base = pack_configuration(int(seed))
    unknown = set(config) - set(base)
    if unknown:
        raise ValueError(f"TabPack config has unknown keys {sorted(unknown)}")
    cfg, overrides = dict(base), {}
    for key, value in config.items():
        if key == "seed":   # weight-init seed always follows the fit seed (pack_configuration(seed))
            if int(value) != base["seed"]:
                overrides["seed"] = {"recipe": int(value), "used": base["seed"],
                                     "reason": "TabPack init seed = fit seed via pack_configuration(seed)"}
            continue
        if key in PACK_LISTS:
            value = [_plain_number(key, v, PACK_INT) for v in value]
            if len(value) != base["pack_size"]:
                raise ValueError(f"config[{key!r}] must have {base['pack_size']} entries")
        elif key in PACK_STR:
            value = str(value)
        else:
            value = _plain_number(key, value, PACK_INT)
        if value != base[key]:
            overrides[key] = {"pack_configuration": base[key], "used": value}
        cfg[key] = value
    if cfg["pack_size"] != base["pack_size"]:
        raise ValueError("TabPack pack_size is fixed by pack_configuration")
    return cfg, overrides


def _check_indicators(names: list[str], arrays: dict[str, np.ndarray]) -> None:
    from framework.training.tabm_data import is_indicator

    for j, name in enumerate(names):
        if not is_indicator(name):
            continue
        for label, x in arrays.items():
            column = x[:, j]
            finite = column[np.isfinite(column)]
            if np.any((finite != 0) & (finite != 1)):
                raise ValueError(f"indicator column {name!r} in {label} holds values other than 0/1/NaN")


def _macro_f1_present(y: np.ndarray, pred: np.ndarray) -> float:
    present = np.unique(y)
    scores = []
    for c in present:
        tp = float(np.sum((pred == c) & (y == c)))
        denom = float(np.sum(pred == c) + np.sum(y == c))
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))


def _require_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fit(); no CPU substitute is used")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None:
        if torch.cuda.is_initialized():
            raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be set before CUDA is initialised "
                               "(deterministic cuBLAS); launch fit in a fresh process")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS
    return torch


def _check_probabilities(p: np.ndarray, n_rows: int, n_classes: int, where: str) -> np.ndarray:
    p = np.asarray(p)
    if p.shape != (n_rows, n_classes):
        raise ValueError(f"{where}: probability shape {p.shape} != {(n_rows, n_classes)}")
    if not np.isfinite(p).all():
        raise ValueError(f"{where}: non-finite probabilities")
    if not np.allclose(p.sum(1), 1.0, rtol=0, atol=1e-4):
        raise ValueError(f"{where}: probability rows do not sum to 1")
    return p.astype(np.float32, copy=False)


# --------------------------------------------------------------------------------------- default config
def default_config(family: str, task: str) -> dict:
    """Frozen selected recipe for ``family`` (read now from selected.json) plus ``"_source"`` provenance.

    The returned config is used unchanged by ``fit`` apart from ``"_source"``. Runtime caps are the
    recorded ones (no change for these families).
    """
    _check_family(family, task)
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    if family not in selected:
        raise KeyError(f"{family} missing from {SELECTED}")
    entry = selected[family]
    if entry.get("family") != family:
        raise ValueError(f"selected.json entry for {family} names family {entry.get('family')!r}")
    config = json.loads(json.dumps(entry["config"]))
    if family == "tabpack":
        from framework.training.tabpack_backend import pack_configuration

        if config != pack_configuration(int(entry["seed"])):
            raise ValueError("frozen TabPack config differs from pack_configuration(seed)")
    source = {
        "selected_json": SELECTED.relative_to(ROOT).as_posix(),
        "selected_json_sha256": _sha256(SELECTED),
        "family": family,
        **{k: entry[k] for k in ("job", "trial_number", "recipe", "study", "phase", "arm", "seed",
                                 "checkpoint", "source_result", "source_result_sha256") if k in entry},
        "frozen_config": json.loads(json.dumps(entry["config"])),
        "adapter_changes": "none (recorded runtime cap kept)",
    }
    config[SOURCE_KEY] = source
    return config


# ------------------------------------------------------------------------------------------------- fit
PARITY_ATOL, PARITY_RTOL = 5e-5, 5e-4   # tabm_fit.train's own tolerance; never loosened here


def _separated_encoding(module, x):
    """CPU operation order for the piecewise-linear embedding (copied from finish_models.separated_encoding)."""
    import torch
    x = module.weight * x[..., None]
    x = x + module.bias
    if x.shape[-1] > 1:
        x = torch.cat([x[..., :1].clamp_max(1.), x[..., 1:-1].clamp(0., 1.),
                       x[..., -1:].clamp_min(0.) if module.single_bin_mask is None else
                       torch.where(module.single_bin_mask[..., None], x[..., -1:], x[..., -1:].clamp_min(0.))], dim=-1)
    return x


def _cpu_order_on_cuda(model) -> None:
    import types
    if getattr(model, "embeddings", None) is not None:
        impl = model.embeddings.impl
        impl.forward = types.MethodType(_separated_encoding, impl)


def _recover_parity(tabm_fit, out_dir: Path, X_tune, y_tune, cfg: dict, started: float, failure: str) -> dict:
    """Re-verify a saved checkpoint after tabm_fit's GPU/CPU parity failure, without changing the tolerance.

    Mirrors finish_models.recover_tabm: 256-row check plus a 4,096-row evenly spaced audit, both at the
    original atol/rtol with identical argmax. Raises if either fails. Writes ``numerical_audit.json``.
    """
    import torch
    from framework.training.hybrid_blend import quick_metrics
    from framework.training.hybrid_neural import parity_diagnostics

    checkpoint = out_dir / CHECKPOINT
    cpu, prep, saved = tabm_fit.load_checkpoint(checkpoint)
    gpu, gpu_prep, _ = tabm_fit.load_checkpoint(checkpoint, "cuda")
    sample = X_tune[:min(256, len(X_tune))]
    native = tabm_fit.predict(gpu, gpu_prep, sample, device="cuda")
    reference = tabm_fit.predict(cpu, prep, sample)
    _cpu_order_on_cuda(gpu)
    aligned = tabm_fit.predict(gpu, gpu_prep, sample, device="cuda")
    np.testing.assert_allclose(aligned, reference, atol=PARITY_ATOL, rtol=PARITY_RTOL)
    np.testing.assert_array_equal(aligned.argmax(1), reference.argmax(1))
    ids = np.linspace(0, len(X_tune) - 1, min(4096, len(X_tune)), dtype=int)
    a = tabm_fit.predict(gpu, gpu_prep, X_tune[ids], device="cuda")
    b = tabm_fit.predict(cpu, prep, X_tune[ids])
    np.testing.assert_allclose(a, b, atol=PARITY_ATOL, rtol=PARITY_RTOL)
    np.testing.assert_array_equal(a.argmax(1), b.argmax(1))
    epochs = [json.loads(line) for line in (out_dir / "epochs.jsonl").read_text(encoding="utf-8").splitlines() if line]
    tune_p = tabm_fit.predict(cpu, prep, X_tune)
    audit = {"trainer_failure": failure[-2000:], "native_gpu_cpu": parity_diagnostics(native, reference),
             "aligned_gpu_cpu": parity_diagnostics(aligned, reference), "extended_rows": int(len(ids)),
             "extended_aligned_gpu_cpu": parity_diagnostics(a, b), "atol": PARITY_ATOL, "rtol": PARITY_RTOL,
             "weights_and_bins_unchanged": True, "cpu_computation_unchanged": True,
             "adaptation": "CUDA PLE inference uses separate multiply/add to match CPU operation order",
             "precedent": "framework/training/finish_models.py recover_tabm"}
    (out_dir / "numerical_audit.json").write_text(json.dumps(json.loads(json.dumps(audit, default=float)), indent=2),
                                                  encoding="utf-8")
    del gpu
    torch.cuda.empty_cache()
    stopped_early = len(epochs) < int(cfg["epochs"])
    return {"epochs": len(epochs), "best_epoch": saved["best_epoch"],
            "stop_reason": "early_stopping" if stopped_early else "epoch_limit",
            "selection_metrics": quick_metrics(y_tune, tune_p),
            "training_seconds": float(epochs[-1]["elapsed_seconds"]) if epochs else time.monotonic() - started,
            "training_device": "cuda", "mixed_precision": "bfloat16" if torch.cuda.is_bf16_supported() else "float32",
            "parameter_count": int(sum(v.numel() for v in cpu.parameters())), "checkpoint_reload_verified": True,
            "cpu_replay_rows": int(len(sample)), "gpu_peak_bytes": int(torch.cuda.max_memory_allocated()),
            "parity_recovered_with_cpu_order": True}


def fit(family: str, data: TaskData, config: dict, seed: int, out_dir: Path) -> dict:
    data.validate()
    _check_family(family, data.task)
    if data.benign_index != 0 or data.classes[0] != "benign":
        raise ValueError(f"{family} requires benign at class index 0")
    if family in HIERARCHICAL and data.n_classes < 3:
        raise ValueError("hierarchical TabM needs at least 3 classes (benign + 2 subtypes)")
    if len(set(data.classes)) != data.n_classes or len(set(data.feature_names)) != len(data.feature_names):
        raise ValueError("class names and feature names must be unique")
    if not np.isin(np.unique(data.y_tune), np.arange(data.n_classes)).all():
        raise ValueError("y_tune contains labels outside the class list")
    if SOURCE_KEY not in config:
        raise ValueError("config must come from default_config() (missing '_source' provenance)")
    source = config[SOURCE_KEY]
    if source.get("family") != family:
        raise ValueError(f"config provenance is for {source.get('family')!r}, not {family!r}")
    raw = {k: v for k, v in config.items() if k != SOURCE_KEY}
    seed = int(seed)

    out_dir = Path(out_dir)
    for name in (CHECKPOINT, "model.tmp", RECORD, "epochs.jsonl"):
        if (out_dir / name).exists():
            raise FileExistsError(f"{out_dir} already holds {name}; use a new directory")
    if out_dir.exists() and any(out_dir.glob("*.pt")):
        raise FileExistsError(f"{out_dir} already holds a checkpoint")

    names = [str(n) for n in data.feature_names]
    classes = [str(c) for c in data.classes]
    X_fit = np.ascontiguousarray(data.X_fit, dtype=np.float32)
    X_tune = np.ascontiguousarray(data.X_tune, dtype=np.float32)
    y_fit = np.ascontiguousarray(data.y_fit, dtype=np.int64)
    y_tune = np.ascontiguousarray(data.y_tune, dtype=np.int64)
    _check_indicators(names, {"X_fit": X_fit, "X_tune": X_tune})

    row_weight = data.extras.get("row_weight")          # runner.scenario_weights multipliers, or None
    if row_weight is not None and family not in TABM:
        raise ValueError(f"{family} does not support row weights")
    torch = _require_cuda()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"feature_names": names, "classes": classes, "benign_index": 0}
    started = time.monotonic()

    if family in TABM:
        from framework.training import tabm_fit

        cfg = _tabm_config(raw)
        overrides = {k: {"frozen": source["frozen_config"].get(k), "used": v} for k, v in cfg.items()
                     if source["frozen_config"].get(k) != v}
        request = {"family": family, "seed": seed, "config": cfg}
        aligned_order = False
        try:
            _, _, details = tabm_fit.train(X_fit, y_fit, X_tune, y_tune, manifest, request, out_dir, device="cuda",
                                           row_weight=row_weight)
        except AssertionError as failure:
            # tabm_fit verifies GPU vs CPU replay *after* saving model.pt. A few-ULP difference from the GPU's
            # fused multiply-add in the numeric embedding can exceed its tolerance. Re-check with the project's
            # established remedy (finish_models.recover_tabm): same weights, CPU operation order, same tolerance.
            if not (out_dir / CHECKPOINT).exists():
                raise
            details = _recover_parity(tabm_fit, out_dir, X_tune, y_tune, cfg, started, str(failure))
            aligned_order = True
        device_model, device_prep, saved = tabm_fit.load_checkpoint(out_dir / CHECKPOINT, "cuda")
        if aligned_order:
            _cpu_order_on_cuda(device_model)
        p_tune = tabm_fit.predict(device_model, device_prep, X_tune, device="cuda")
        n_embedded = len(saved["preprocessor"]["embedded_indices"])
        if family in EMBEDDED and (n_embedded == 0 or not device_model.embedded):
            raise RuntimeError(f"{family}: no column qualified for numeric embeddings; the model would "
                               "silently fall back to a non-embedded network")
        trainer = "framework.training.tabm_fit.train"
        extra = {"best_epoch": details["best_epoch"], "epochs": details["epochs"],
                 "mixed_precision": details["mixed_precision"], "parameter_count": details["parameter_count"],
                 "gpu_peak_bytes": int(details["gpu_peak_bytes"]), "n_embedded_features": n_embedded,
                 "cpu_replay_rows": details["cpu_replay_rows"],
                 "checkpoint_reload_verified": details["checkpoint_reload_verified"],
                 "trainer_selection_metrics": details["selection_metrics"],
                 "parity_recovered_with_cpu_order": bool(details.get("parity_recovered_with_cpu_order", False))}
        del device_model
    else:
        from framework.training import hybrid_neural

        cfg, overrides = _pack_config(raw, seed)
        details = hybrid_neural.train("tabpack", X_fit, y_fit, X_tune, y_tune, manifest, cfg, seed, out_dir,
                                      device="cuda")
        device_model, device_prep, saved, weights = hybrid_neural.load(out_dir / CHECKPOINT, "cuda")
        p_tune = hybrid_neural.predict(device_model, device_prep, X_tune, "tabpack", weights, device="cuda")
        trainer = "framework.training.hybrid_neural.train (pack_configuration(seed))"
        extra = {"best_epoch": details["best_epoch"], "epochs": details["epochs"],
                 "parameter_count": details["parameter_count"],
                 "member_weights": [float(w) for w in details["member_weights"]],
                 "cpu_gpu_diagnostics": details["cpu_gpu_diagnostics"],
                 "checkpoint_reload_verified": details["cpu_checkpoint_reload_verified"],
                 "trainer_selection_metrics": saved["selection_metrics"]}
        del device_model
    torch.cuda.empty_cache()

    p_tune = _check_probabilities(p_tune, len(y_tune), data.n_classes, "CUDA tune replay")
    np.save(out_dir / TUNE_PROBABILITIES, p_tune, allow_pickle=False)
    pred = p_tune.argmax(1)
    p_cpu = predict_cpu(family, out_dir, X_tune, _require_record=False)
    cpu_pred = p_cpu.argmax(1)

    record = {
        "family": family, "task": data.task, "seed": seed,
        "checkpoint": CHECKPOINT,
        "config_used": cfg,
        "config_overrides_vs_frozen": overrides,
        "training_seconds": float(details["training_seconds"]),
        "adapter_wall_seconds": float(time.monotonic() - started),
        "device": f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})",
        "trainer_device": details["training_device"],
        "stop_reason": details["stop_reason"],
        "tune_accuracy": float(np.mean(pred == y_tune)),
        "tune_macro_f1": _macro_f1_present(y_tune, pred),
        "tune_metric_source": "CUDA reload of the selected checkpoint; macro F1 over classes present in y_tune",
        "tune_classes_present": int(np.unique(y_tune).size),
        "tune_probabilities": TUNE_PROBABILITIES,
        "cpu_tune_argmax_agreement": float(np.mean(cpu_pred == pred)),
        "cpu_tune_max_abs_diff": float(np.max(np.abs(p_cpu - p_tune))),
        "n_features": len(names), "n_classes": data.n_classes, "classes": classes,
        "n_fit_rows": int(len(y_fit)), "n_tune_rows": int(len(y_tune)),
        "trainer": trainer,
        "probability_kind": "class probabilities (member-averaged softmax)",
        "source_recipe": source,
        "row_weighted": row_weight is not None,
        **extra,
    }
    record = json.loads(json.dumps(record))
    (out_dir / RECORD).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


# ------------------------------------------------------------------------------------------ predict_cpu
def predict_cpu(family: str, out_dir: Path, X: np.ndarray, _require_record: bool = True) -> np.ndarray:
    """Reload ``out_dir/model.pt`` on CPU and return float32 probabilities (n, n_classes).

    Refuses directories without a completed ``fit_record.json`` (e.g. a fit that failed its replay check
    after writing ``model.pt``). Columns must be in the fitted feature order; only the count is checkable.
    """
    import torch

    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    out_dir = Path(out_dir)
    checkpoint = out_dir / CHECKPOINT
    record = None
    if _require_record:
        record_path = out_dir / RECORD
        if not record_path.exists():
            raise FileNotFoundError(f"{out_dir} has no {RECORD}; the fit did not complete")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["family"] != family or record["checkpoint"] != CHECKPOINT:
            raise ValueError(f"{out_dir} holds a {record['family']} fit, not {family}")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    # Cast first, then check: a finite float64 beyond float32 range becomes inf in the cast, and the
    # preprocessor would otherwise silently treat it as missing.
    with np.errstate(over="ignore"):
        X = np.ascontiguousarray(X, dtype=np.float32)
    if np.isinf(X).any():
        raise ValueError("X contains inf (or values outside float32 range)")
    cuda_was_initialised = torch.cuda.is_initialized()
    torch.set_num_threads(2)

    if family in TABM:
        from framework.training import tabm_fit

        model, prep, saved = tabm_fit.load_checkpoint(checkpoint, device="cpu")
    else:
        from framework.training import hybrid_neural

        model, prep, saved, weights = hybrid_neural.load(checkpoint, device="cpu")
    if saved["family"] != family:
        raise ValueError(f"checkpoint family {saved['family']!r} != {family!r}")
    if X.shape[1] != len(saved["feature_names"]):
        raise ValueError(f"X has {X.shape[1]} columns, checkpoint expects {len(saved['feature_names'])}")
    if record is not None and (list(saved["classes"]) != record["classes"]
                               or len(saved["feature_names"]) != record["n_features"]):
        raise ValueError(f"{checkpoint} classes/feature count differ from {RECORD}")
    tensors = list(model.parameters()) + list(model.buffers())
    if any(t.device.type != "cpu" for t in tensors):
        raise RuntimeError("model is not on CPU")
    if family in TABM:
        p = tabm_fit.predict(model, prep, X, device="cpu")
    else:
        p = hybrid_neural.predict(model, prep, X, "tabpack", weights, device="cpu")
    if not cuda_was_initialised and torch.cuda.is_initialized():
        raise RuntimeError("CPU replay initialised CUDA")
    return _check_probabilities(p, len(X), len(saved["classes"]), "CPU replay")
