"""Optional CatBoost CUDA and NVIDIA OpenCL LightGBM research backends.

Windows LightGBM accelerates histogram construction on the GPU; other stages
and prediction use the CPU. Both candidates select checkpoints on unweighted
validation macro F1. No test inputs are accepted and CPU training fallback fails.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import json
import logging
import os
from pathlib import Path
import time

import numpy as np

from framework.training.tracking import write_json
from framework.training.trees import _features, _labels, _macro_f1, class_weights


def nvidia_opencl_device() -> dict:
    """Find an actual NVIDIA GPU and its OpenCL platform/device indices."""
    library = ctypes.WinDLL("OpenCL.dll") if os.name == "nt" else ctypes.CDLL(ctypes.util.find_library("OpenCL") or "libOpenCL.so.1")
    pointer, uint, size = ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t
    library.clGetPlatformIDs.argtypes = [uint, ctypes.POINTER(pointer), ctypes.POINTER(uint)]
    library.clGetDeviceIDs.argtypes = [pointer, ctypes.c_ulonglong, uint, ctypes.POINTER(pointer), ctypes.POINTER(uint)]
    library.clGetDeviceInfo.argtypes = [pointer, uint, size, pointer, ctypes.POINTER(size)]
    count = uint()
    if library.clGetPlatformIDs(0, None, ctypes.byref(count)) != 0 or not count.value:
        raise RuntimeError("No OpenCL platform available; no CPU fallback")
    platforms = (pointer * count.value)()
    if library.clGetPlatformIDs(count, platforms, None) != 0:
        raise RuntimeError("Cannot enumerate OpenCL platforms")
    for platform_id, platform in enumerate(platforms):
        devices_count = uint()
        if library.clGetDeviceIDs(platform, 0xFFFFFFFF, 0, None, ctypes.byref(devices_count)) != 0:
            continue
        devices = (pointer * devices_count.value)()
        if library.clGetDeviceIDs(platform, 0xFFFFFFFF, devices_count, devices, None) != 0:
            continue
        for device_id, device in enumerate(devices):
            vendor, name = ctypes.create_string_buffer(1024), ctypes.create_string_buffer(1024)
            device_type = ctypes.c_ulonglong()
            statuses = [library.clGetDeviceInfo(device, field, ctypes.sizeof(buffer), ctypes.byref(buffer), None)
                        for field, buffer in [(0x102C, vendor), (0x102B, name), (0x1000, device_type)]]
            if all(value == 0 for value in statuses) and b"NVIDIA" in vendor.value.upper() and device_type.value & 4:
                return {"gpu_platform_id": platform_id, "gpu_device_id": device_id,
                        "device_name": name.value.decode(), "vendor": vendor.value.decode()}
    raise RuntimeError("No NVIDIA OpenCL GPU detected; no CPU fallback")


def require_lightgbm_gpu(lines: list[str]) -> str:
    """Require positive backend and NVIDIA-device proof from the native runtime."""
    text = "\n".join(lines)
    devices = [line.strip() for line in lines if "Using GPU Device:" in line and "NVIDIA" in line.upper()]
    if "This is the GPU trainer" not in text or not devices:
        raise RuntimeError("LightGBM did not confirm NVIDIA GPU execution; CPU fallback rejected")
    return devices[0]


class _LightGBMLog:
    def __init__(self):
        self.lines = []
        self.no_split_messages = 0

    def info(self, message):
        if "No further splits with positive gain" in message:
            self.no_split_messages += 1
            return
        self.lines.append(str(message))
        print(message, flush=True)

    def warning(self, message):
        self.info(message)


def _catboost(X, y, V, v, n_classes, config, seed, directory, weights):
    import catboost as cb
    from catboost.utils import get_gpu_device_count

    if config.get("task_type", "GPU") != "GPU" or config.get("devices", "0") != "0":
        raise ValueError("CatBoost research training requires GPU device 0")
    if get_gpu_device_count() < 1:
        raise RuntimeError("CatBoost CUDA device unavailable; no CPU fallback")
    params = {"loss_function": "MultiClass", "classes_count": n_classes,
              "eval_metric": "TotalF1:average=Macro;use_weights=false",
              "task_type": "GPU", "devices": "0", "random_seed": seed,
              "thread_count": 4, "allow_writing_files": False, "gpu_ram_part": .65,
              "bootstrap_type": "Bernoulli", "subsample": .8,
              "iterations": 750, "depth": 6, "learning_rate": .08,
              "l2_leaf_reg": 3, "border_count": 128}
    for key in ("iterations", "depth", "learning_rate", "l2_leaf_reg", "border_count", "subsample", "random_strength", "gpu_ram_part"):
        if key in config:
            params[key] = config[key]
    start = time.perf_counter()
    model = cb.CatBoostClassifier(**params)
    print("CatBoost log 'test' means the validation eval_set; the locked test partition is not loaded.", flush=True)
    model.fit(cb.Pool(X, label=y, weight=weights[y]), eval_set=cb.Pool(V, label=v),
              use_best_model=True, early_stopping_rounds=int(config.get("early_stopping_rounds", 40)), verbose=25)
    seconds = time.perf_counter() - start
    actual = model.get_all_params()
    if actual.get("task_type") != "GPU":
        raise RuntimeError("Fitted CatBoost model does not report GPU training")
    pred = np.asarray(model.predict(V, prediction_type="Class", thread_count=4)).reshape(-1).astype(np.int32)
    score = _macro_f1(v, pred, n_classes)
    history = model.get_evals_result()
    metric = next(key for key in history["validation"] if key.startswith("TotalF1"))
    best_iteration = model.get_best_iteration()
    if not np.isclose(history["validation"][metric][best_iteration], score, atol=1e-8, rtol=0):
        raise AssertionError("CatBoost stopping metric differs from unweighted all-class macro F1")
    checkpoint = directory / "model.cbm"
    model.save_model(str(checkpoint))
    return {"predictions": pred, "training_seconds": seconds, "device": "cuda:0",
            "backend": "CatBoost CUDA multiclass gradient boosting", "library_version": cb.__version__,
            "checkpoint_path": str(checkpoint), "parameters": actual, "history": history,
            "best_iteration": best_iteration, "saved_rounds": model.tree_count_,
            "validation_macro_f1": score, "device_evidence": {"fitted_task_type": actual["task_type"], "requested_devices": "0"},
            "gpu_training_verified": True, "prediction_device": "cpu",
            "determinism_note": "GPU floating-point reductions can vary despite a fixed seed"}


def _lightgbm(X, y, V, v, n_classes, config, seed, directory, weights):
    import lightgbm as lgb

    if config.get("device_type", "gpu") != "gpu":
        raise ValueError("LightGBM research training requires the GPU backend")
    device = nvidia_opencl_device()
    params = {"objective": "multiclass", "num_class": n_classes, "metric": "None",
              "device_type": "gpu", "gpu_platform_id": device["gpu_platform_id"],
              "gpu_device_id": device["gpu_device_id"], "gpu_use_dp": False,
              "seed": seed, "feature_fraction_seed": seed, "bagging_seed": seed,
              "data_random_seed": seed, "num_threads": 4, "verbosity": 1,
              "max_bin": 63, "num_leaves": 31, "learning_rate": .08,
              "min_data_in_leaf": 30, "lambda_l2": 1, "feature_pre_filter": False}
    for key in ("max_bin", "num_leaves", "max_depth", "learning_rate", "min_data_in_leaf",
                "lambda_l1", "lambda_l2", "min_gain_to_split", "feature_fraction", "bagging_fraction", "bagging_freq"):
        if key in config:
            params[key] = config[key]
    logger = _LightGBMLog()
    lgb.register_logger(logger)
    history = {}

    def macro_f1(probabilities, data):
        values = np.asarray(probabilities)
        if values.ndim == 1:
            values = values.reshape(-1, n_classes, order="F")
        return "macro_f1", _macro_f1(data.get_label().astype(np.int32), values.argmax(axis=1), n_classes), True

    start = time.perf_counter()
    try:
        training = lgb.Dataset(X, label=y, weight=weights[y], params=params)
        validation = lgb.Dataset(V, label=v, reference=training)
        model = lgb.train(params, training, num_boost_round=int(config.get("n_estimators", 500)),
                          valid_sets=[validation], valid_names=["validation"], feval=macro_f1,
                          callbacks=[lgb.early_stopping(int(config.get("early_stopping_rounds", 40)), first_metric_only=True),
                                     lgb.record_evaluation(history), lgb.log_evaluation(25)])
    finally:
        (directory / "lightgbm_runtime.log").write_text("\n".join(logger.lines), encoding="utf-8")
        lgb.register_logger(logging.getLogger("lightgbm"))
    seconds = time.perf_counter() - start
    evidence = require_lightgbm_gpu(logger.lines)
    pred = model.predict(V, num_iteration=model.best_iteration, num_threads=4).argmax(axis=1).astype(np.int32)
    score = _macro_f1(v, pred, n_classes)
    if not np.isclose(history["validation"]["macro_f1"][model.best_iteration - 1], score, atol=1e-8, rtol=0):
        raise AssertionError("LightGBM selected checkpoint does not match its stopping metric")
    checkpoint = directory / "model.lgb.txt"
    model.save_model(str(checkpoint), num_iteration=model.best_iteration)
    return {"predictions": pred, "training_seconds": seconds,
            "device": f"opencl:{device['gpu_platform_id']}:{device['gpu_device_id']}",
            "backend": "LightGBM NVIDIA OpenCL GPU histograms with CPU tree orchestration",
            "library_version": lgb.__version__, "checkpoint_path": str(checkpoint), "parameters": params,
            "best_iteration": model.best_iteration - 1, "saved_rounds": model.best_iteration,
            "history": history, "validation_macro_f1": score, "device_evidence": {**device, "native_runtime": evidence},
            "gpu_training_verified": True, "prediction_device": "cpu",
            "no_further_split_messages": logger.no_split_messages,
            "determinism_note": "OpenCL histogram reductions can vary despite a fixed seed"}


def train_booster(candidate, X_train, y_train, X_val, y_val, n_classes, config, seed, output_dir):
    """Train a GPU backend; reject artifact replacement and incompatible inputs."""
    if candidate not in {"catboost", "lightgbm"}:
        raise ValueError(f"Unknown boosting candidate: {candidate}")
    X, V = _features(X_train), _features(X_val)
    y, v = _labels(y_train, n_classes, "y_train"), _labels(y_val, n_classes, "y_val")
    if len(X) != len(y) or len(V) != len(v) or X.shape[1] != V.shape[1]:
        raise ValueError("Incompatible training/validation shapes")
    weights = class_weights(y, n_classes, config.get("weight_strategy", "none"))
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ("model.cbm", "model.lgb.txt", "booster_metadata.json")):
        raise FileExistsError(f"Booster artifacts already exist in {directory}")
    function = _catboost if candidate == "catboost" else _lightgbm
    result = function(X, y, V, v, n_classes, config, seed, directory, weights)
    result.update(candidate=candidate, class_weights=weights.tolist(), seed=int(seed),
                  training_rows=len(y), validation_rows=len(v), n_features=X.shape[1], n_classes=n_classes,
                  selection_criterion="unweighted validation macro F1 over all declared classes",
                  preprocessing="Numeric float32; native missing values; quantization learned from training only")
    write_json(directory / "booster_metadata.json", {key: value for key, value in result.items() if key != "predictions"})
    return result


def predict_booster(checkpoint_path, X):
    """Reload an actual fitted research model; these libraries use CPU prediction."""
    checkpoint = Path(checkpoint_path)
    metadata = json.loads((checkpoint.parent / "booster_metadata.json").read_text())
    X = _features(X)
    if X.shape[1] != metadata["n_features"]:
        raise ValueError("Feature count differs from fitted model")
    if metadata["candidate"] == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(str(checkpoint))
        return np.asarray(model.predict(X, prediction_type="Class", thread_count=4)).reshape(-1).astype(np.int32)
    if metadata["candidate"] == "lightgbm":
        from lightgbm import Booster
        return Booster(model_file=str(checkpoint)).predict(X, num_threads=4).argmax(axis=1).astype(np.int32)
    raise ValueError("Unknown saved booster backend")
