"""Bounded, validation-only CUDA improvements; the application model is unchanged.

Run from the project root with ``python -m experiment.improve_detector``.
Every XGBoost fit includes all eligible real training rows. No held-out test
array is accepted or opened. GAN rows, when enabled, are explicitly synthetic.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from experiment.train_candidates import ResourceSampler
from framework.training.tracking import classification_metrics, prepared_identity, validate_manifest, write_json
from framework.training.trees import _macro_f1, _parameters, _require_cuda, class_weights

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "dataset/prepared/datasense_full_protocols"
OUTPUT = ROOT / "artifacts/training/improvement_v1"
ANCHOR = ROOT / "artifacts/training/datasense_all_models_full_v1/runs/xgboost_joint_1_s42_20260929T182104981183Z"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def proposals() -> list[dict]:
    """Freeze contrasts before measuring them; avoid a validation fishing sweep."""
    base = json.loads((ANCHOR / "tree_metadata.json").read_text())["requested_config"]
    return [
        {"id": "baseline_refit", "changes": {}, "purpose": "Reproduce the existing full-data CUDA baseline"},
        {"id": "gradient_sampling", "changes": {"sampling_method": "gradient_based", "subsample": .6},
         "purpose": "Prioritize difficult training gradients at fixed depth and weighting"},
        {"id": "leafwise_64", "changes": {"grow_policy": "lossguide", "max_depth": 0, "max_leaves": 64},
         "purpose": "Compare leaf-wise growth with a 64-leaf cap instead of depth-wise growth"},
        {"id": "sqrt_weights", "changes": {"weight_strategy": "sqrt_balanced"},
         "purpose": "Isolate capped square-root inverse-frequency training weights"},
        {"id": "gan_augmentation", "changes": {},
         "purpose": "A/B genuine conditional GAN training-only augmentation against baseline_refit"},
    ], base


def train_trial(X, y, V, v, classes, benign_index, config, directory: Path) -> dict:
    import xgboost as xgb
    if not xgb.build_info().get("USE_CUDA"):
        raise RuntimeError("XGBoost CUDA build required; no CPU training fallback")
    adapted = dict(config)
    # Existing parameter validation requires positive depth. Unlimited depth is
    # separately allowed only with a strict leaf budget for the new comparison.
    leafwise = adapted.get("grow_policy") == "lossguide"
    if leafwise:
        if adapted.get("max_depth") != 0 or not 2 <= int(adapted.get("max_leaves", 0)) <= 128:
            raise ValueError("Leaf-wise comparison requires depth zero and a bounded leaf budget")
        adapted["max_depth"] = 6
    params, rounds = _parameters("xgboost", len(classes), adapted, 42)
    for name in ("sampling_method", "grow_policy", "max_leaves"):
        if name in config:
            params[name] = config[name]
    if leafwise:
        params["max_depth"] = 0
    weights = class_weights(y, len(classes), config["weight_strategy"])

    class CudaGuard(xgb.callback.TrainingCallback):
        def before_training(self, model):
            _require_cuda(model)
            return model

        def after_training(self, model):
            _require_cuda(model)
            return model

    class RuntimeCap(xgb.callback.TrainingCallback):
        def __init__(self):
            self.triggered = False
            self.limit_seconds = float(config.get("runtime_limit_seconds", 720))

        def after_iteration(self, model, epoch, evals_log):
            if time.perf_counter() - started >= self.limit_seconds:
                self.triggered = True
                print(f"RUNTIME CAP after round {epoch + 1}; validation-selected checkpoint retained", flush=True)
                return True
            return False

    def macro_f1(probabilities, data):
        truth = data.get_label().astype(np.int32)
        return "macro_f1", _macro_f1(truth, probabilities.reshape(-1, len(classes)).argmax(1), len(classes))

    history = {}
    started = time.perf_counter()
    runtime = RuntimeCap()
    with ResourceSampler() as resources:
        training = xgb.QuantileDMatrix(X, label=y, weight=weights[y], max_bin=params["max_bin"], nthread=4)
        validation = xgb.QuantileDMatrix(V, label=v, ref=training, max_bin=params["max_bin"], nthread=4)
        model = xgb.train(params, training, num_boost_round=rounds, evals=[(validation, "validation")],
                          custom_metric=macro_f1, evals_result=history, verbose_eval=25,
                          callbacks=[CudaGuard(), runtime, xgb.callback.EarlyStopping(
                              rounds=config["early_stopping_rounds"], metric_name="macro_f1",
                              data_name="validation", maximize=True, save_best=True)])
        seconds = time.perf_counter() - started
        runtime_device = _require_cuda(model)
        probabilities = model.predict(validation)
        predictions = probabilities.argmax(1).astype(np.int32)
        model.save_model(directory / "model.ubj")
        reload = xgb.Booster(model_file=str(directory / "model.ubj"))
        reload.set_param({"device": "cuda:0", "nthread": 4})
        _require_cuda(reload)
        reloaded = reload.predict(xgb.DMatrix(np.asarray(V[:128]), nthread=4)).argmax(1)
        if not np.array_equal(reloaded, predictions[:128]):
            raise AssertionError("Saved checkpoint did not reproduce its validation predictions")
    np.save(directory / "validation_probabilities.npy", probabilities)
    np.save(directory / "validation_predictions.npy", predictions)
    return {"status": "completed", "metrics": classification_metrics(v, predictions, classes, benign_index),
            "training_seconds": seconds, "parameters": params, "requested_config": config,
            "device": runtime_device, "device_evidence": json.loads(model.save_config())["learner"]["generic_param"],
            "gpu_training_verified": True, "seed": 42, "checkpoint_reload_verified": True,
            "validation_probabilities_sha256": sha256(directory / "validation_probabilities.npy"),
            "validation_predictions_sha256": sha256(directory / "validation_predictions.npy"),
            "checkpoint_sha256": sha256(directory / "model.ubj"), "saved_rounds": model.num_boosted_rounds(),
            "best_iteration": int(model.best_iteration), "trained_rounds": len(history["validation"]["macro_f1"]),
            "history": history, "training_rows": len(y), "validation_rows": len(v),
            "runtime_limit_seconds": runtime.limit_seconds, "runtime_capped": runtime.triggered,
            "stopping_reason": "12-minute runtime budget, checked between rounds" if runtime.triggered else "validation macro F1 patience or round budget",
            "peak_process_rss_bytes_sampled": resources.peak_rss_bytes,
            "peak_device_used_mib_sampled": resources.device_peak_used_mib,
            "memory_note": "VRAM sampling is device-wide and includes other apps; not exact process peak",
            "selection_metric": "Unweighted validation macro F1 over all 84 declared classes", "xgboost_version": xgb.__version__}


def audit_trial(directory: Path, result: dict, V, v, manifest: dict, verify_checkpoint: bool = False) -> dict:
    """Validate cached scores and arrays; old first-run artifacts get a replay audit."""
    if result["checkpoint_sha256"] != sha256(directory / "model.ubj"):
        raise ValueError("Previously completed checkpoint was modified")
    probabilities = np.load(directory / "validation_probabilities.npy")
    predictions = np.load(directory / "validation_predictions.npy")
    if probabilities.shape != (len(v), len(manifest["classes"])) or predictions.shape != (len(v),):
        raise ValueError("Cached validation array schema changed")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Cached validation probabilities are invalid")
    if not np.allclose(probabilities.sum(1), 1, atol=1e-5, rtol=0) or not np.array_equal(probabilities.argmax(1), predictions):
        raise ValueError("Cached validation probabilities/predictions disagree")
    metrics = classification_metrics(v, predictions, manifest["classes"], manifest["benign_index"])
    if result["metrics"] != metrics:
        raise ValueError("Cached scores differ from current real validation labels and predictions")
    digests = {"validation_probabilities_sha256": sha256(directory / "validation_probabilities.npy"),
               "validation_predictions_sha256": sha256(directory / "validation_predictions.npy")}
    legacy = not all(key in result for key in digests)
    if not legacy and any(result[key] != value for key, value in digests.items()):
        raise ValueError("Cached validation arrays were modified")
    if verify_checkpoint or legacy:
        import xgboost as xgb
        model = xgb.Booster(model_file=str(directory / "model.ubj"))
        model.set_param({"device": "cuda:0", "nthread": 4})
        _require_cuda(model)
        replay = model.predict(xgb.DMatrix(V, nthread=4))
        if not np.array_equal(replay.argmax(1), predictions) or not np.allclose(replay, probabilities, atol=1e-6, rtol=0):
            raise ValueError("Saved checkpoint does not reproduce all cached validation rows")
        result["full_validation_checkpoint_replay_verified"] = True
    result.update(digests)
    write_json(directory / "result.json", result)
    return result


def summarize(output: Path, records: list[dict], baseline: dict, ensemble: dict | None = None) -> None:
    completed = [r for r in records if r.get("status") == "completed"]
    contenders = [baseline, *completed, *([ensemble] if ensemble else [])]
    best_accuracy = max(contenders, key=lambda r: r["metrics"]["accuracy"])
    best_macro = max(contenders, key=lambda r: r["metrics"]["macro_f1"])
    report = {"completed_new_trials": len(completed), "planned_new_trials": 5,
              "baseline": baseline, "trials": records, "ensemble": ensemble, "best_accuracy_id": best_accuracy["id"],
              "best_validation_accuracy": best_accuracy["metrics"]["accuracy"], "best_macro_f1_id": best_macro["id"],
              "best_validation_macro_f1": best_macro["metrics"]["macro_f1"], "test_evaluated": False,
              "app_checkpoint_changed": False, "accuracy_99_97_percent_reached": best_accuracy["metrics"]["accuracy"] >= .9997,
              "accuracy_96_percent_reached": best_accuracy["metrics"]["accuracy"] >= .96}
    write_json(output / "RESULTS.json", report)
    lines = ["# Bounded CUDA detector improvement study", "", "Validation only. Held-out test remains locked. App checkpoint unchanged.", "",
             "| Method | Accuracy | Macro F1 | Benign FPR | Missed as benign | Training seconds |", "|---|---:|---:|---:|---:|---:|"]
    for r in contenders:
        m = r["metrics"]
        lines.append(f"| {r['id']} | {m['accuracy']:.6f} | {m['macro_f1']:.6f} | {m['false_positive_rate']:.6f} | {m['attack_missed_as_benign_rate']:.6f} | {r.get('training_seconds', 0):.1f} |")
    lines += ["", "All fits use every eligible real training row (480,405). GAN rows are additional synthetic training examples; validation stays real and unchanged.",
              "These are within-capture development results, not independent device/capture performance or confirmed live-PC accuracy.",
              "Class labels include fine-grained attack ports; no unknown-attack evaluation or independent final test scoring is performed."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--only", nargs="+", choices=[s["id"] for s in proposals()[0]])
    parser.add_argument("--audit-only", action="store_true", help="Replay completed checkpoints and verify cached metrics; do not fit")
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU research training is not authorized as a substitute")
    torch.set_num_threads(4)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PREPARED / "manifest.json").read_text())
    validate_manifest(manifest)
    specs, base = proposals()
    identity = {**prepared_identity(PREPARED), "source_sha256": manifest["source_sha256"],
                "seed": 42, "specs": specs, "base_parameters": base, "test_evaluated": False,
                "baseline_checkpoint_sha256": sha256(ANCHOR / "model.ubj"),
                "real_training_rows": 480405, "validation_rows": 30000,
                "gpu_name": torch.cuda.get_device_name(0), "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "ensemble": "Equal-weight probability mean of saved baseline + gradient_sampling + leafwise_64; frozen before scores"}
    study = output / "study.json"
    if study.exists() and json.loads(study.read_text()) != identity:
        raise ValueError("Study identity changed; use a different output directory")
    write_json(study, identity)
    # This path intentionally names training and validation arrays only.
    X = np.load(PREPARED / "X_train.npy", mmap_mode="r")
    y = np.load(PREPARED / "y_train.npy", mmap_mode="r")
    V = np.load(PREPARED / "X_validation.npy", mmap_mode="r")
    v = np.load(PREPARED / "y_validation.npy", mmap_mode="r")
    if len(y) != 480405 or len(v) != 30000:
        raise ValueError("Prepared rows differ from the frozen full-data comparison")
    import xgboost as xgb
    original = xgb.Booster(model_file=str(ANCHOR / "model.ubj"))
    original.set_param({"device": "cuda:0", "nthread": 4})
    _require_cuda(original)
    probabilities = original.predict(xgb.DMatrix(V, nthread=4))
    baseline = {"id": "saved_app_baseline", "metrics": classification_metrics(v, probabilities.argmax(1), manifest["classes"], manifest["benign_index"]),
                "checkpoint_sha256": identity["baseline_checkpoint_sha256"], "replay_only": True,
                "training_seconds": json.loads((ANCHOR / "tree_metadata.json").read_text())["training_seconds"]}
    np.save(output / "baseline_probabilities.npy", probabilities)
    del original, probabilities
    records = []
    for spec in specs:
        directory = output / "runs" / spec["id"]
        result_path = directory / "result.json"
        if result_path.exists():
            prior = json.loads(result_path.read_text())
            if prior["status"] == "completed":
                prior = audit_trial(directory, prior, V, v, manifest, verify_checkpoint=args.audit_only)
                if spec["id"] == "gan_augmentation":
                    from framework.training.augmentation import train_conditional_gan
                    _, _, verified_augmentation = train_conditional_gan(X, y, manifest["feature_names"], output / "gan", seed=42)
                    if verified_augmentation != prior["augmentation"]:
                        raise ValueError("Saved detector GAN provenance differs from verified augmentation artifacts")
                records.append(prior)
                print(f"REUSE {spec['id']}", flush=True)
                continue
        if args.audit_only:
            continue
        if args.only and spec["id"] not in args.only:
            continue
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "model.ubj").exists():
            raise FileExistsError("Refusing to overwrite a previous incomplete checkpoint")
        request = {"id": spec["id"], "purpose": spec["purpose"], "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "real_training_rows": len(y), "test_evaluated": False, "source_sha256": manifest["source_sha256"]}
        write_json(directory / "request.json", request)
        print(f"START {spec['id']} real_training_rows={len(y)} device=cuda:0", flush=True)
        A, a = X, y
        augmentation = None
        try:
            if spec["id"] == "gan_augmentation":
                from framework.training.augmentation import train_conditional_gan
                S, s, augmentation = train_conditional_gan(X, y, manifest["feature_names"], output / "gan", seed=42)
                A, a = np.concatenate((X, S)), np.concatenate((y, s))
            result = {**request, **train_trial(A, a, V, v, manifest["classes"], manifest["benign_index"], {**base, **spec["changes"]}, directory),
                      "synthetic_training_rows": len(a) - len(y), "augmentation": augmentation}
        except Exception:
            result = {**request, "status": "failed", "error": traceback.format_exc()}
            print(result["error"], flush=True)
        write_json(result_path, result)
        records.append(result)
        summarize(output, records, baseline)
        if result["status"] == "completed":
            print(f"COMPLETED {spec['id']} accuracy={result['metrics']['accuracy']:.6f} macro_F1={result['metrics']['macro_f1']:.6f} seconds={result['training_seconds']:.1f}", flush=True)
        del A, a
        torch.cuda.empty_cache()
    summarize(output, records, baseline)
    needed = [output / "baseline_probabilities.npy", output / "runs/gradient_sampling/validation_probabilities.npy", output / "runs/leafwise_64/validation_probabilities.npy"]
    if all(p.is_file() for p in needed):
        mean = sum(np.load(p) for p in needed) / len(needed)
        pred = mean.argmax(1)
        ensemble = {"id": "equal_weight_ensemble", "constituents": ["saved_app_baseline", "gradient_sampling", "leafwise_64"],
                    "weights": [1 / 3] * 3, "fit_to_validation_labels": False,
                    "metrics": classification_metrics(v, pred, manifest["classes"], manifest["benign_index"]),
                    "test_evaluated": False, "app_checkpoint_changed": False}
        np.save(output / "ensemble_probabilities.npy", mean)
        np.save(output / "ensemble_predictions.npy", pred.astype(np.int32))
        ensemble["validation_probabilities_sha256"] = sha256(output / "ensemble_probabilities.npy")
        ensemble["validation_predictions_sha256"] = sha256(output / "ensemble_predictions.npy")
        write_json(output / "ensemble.json", ensemble)
        summarize(output, records, baseline, ensemble)
        print(f"ENSEMBLE accuracy={ensemble['metrics']['accuracy']:.6f} macro_F1={ensemble['metrics']['macro_f1']:.6f}", flush=True)
    return int(any(r["status"] != "completed" for r in records))


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    raise SystemExit(main())
