"""Run a bounded CUDA candidate search and repeat selected settings on new seeds.

Only training and validation arrays are accepted. The prepared test partition is
never opened. Each trial runs in a separate process to release CUDA allocations.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

from framework.training.tracking import classification_metrics, choose_best, completed_trial, prepared_identity, validate_manifest, write_json

ROOT = Path(__file__).resolve().parents[1]


def configurations():
    """Controlled weighting comparison plus a second architecture/parameter setting."""
    baseline = {
        "random_forest": {"max_depth": 6, "num_parallel_tree": 32, "max_bin": 128, "subsample": .8, "colsample_bynode": .8},
        "xgboost": {"max_depth": 4, "n_estimators": 100, "lr": .1, "max_bin": 128, "early_stopping_rounds": 15},
        "kan": {"hidden": [64], "grid_size": 3, "spline_order": 3, "epochs": 8, "batch_size": 512, "lr": .001, "patience": 3},
        "hem": {"hidden": [512, 256, 32, 16], "binary_hidden": [1024, 256, 64, 16], "pca_components": 16,
                "knn_k": 3, "knn_max_rows": 20000, "epochs": 8, "batch_size": 512, "lr": .001, "patience": 3},
    }
    tuned = {
        "random_forest": {"max_depth": 8, "num_parallel_tree": 64, "colsample_bynode": .65, "min_child_weight": 2},
        "xgboost": {"max_depth": 6, "n_estimators": 160, "lr": .08, "min_child_weight": 2, "reg_lambda": 2},
        "kan": {"hidden": [96, 48], "grid_size": 5, "epochs": 12, "lr": .002, "weight_decay": .001},
        "hem": {"pca_components": 32, "knn_k": 7, "epochs": 12, "lr": .0007},
    }
    return {name: [
        {"id": "baseline", "purpose": "Unweighted baseline", "parameters": {**base, "weight_strategy": "none"}},
        {"id": "sqrt_weights", "purpose": "Isolate square-root inverse-frequency weights", "parameters": {**base, "weight_strategy": "sqrt_balanced"}},
        {"id": "tuned_sqrt", "purpose": "Change model hyperparameters at fixed weighting", "parameters": {**base, **tuned[name], "weight_strategy": "sqrt_balanced"}},
        {"id": "tuned_balanced", "purpose": "Isolate capped inverse-frequency weights on tuned setting", "parameters": {**base, **tuned[name], "weight_strategy": "balanced"}},
    ] for name, base in baseline.items()}


class ResourceSampler:
    """Sample process RAM and device-wide VRAM; the latter includes other apps."""
    def __init__(self):
        self.stop = threading.Event()
        self.peak_rss_bytes = 0
        self.device_peak_used_mib = None
        self.samples = 0

    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        last_gpu = 0
        while not self.stop.is_set():
            self.peak_rss_bytes = max(self.peak_rss_bytes, self.process.memory_info().rss)
            self.samples += 1
            if time.monotonic() - last_gpu > 5:
                try:
                    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                            capture_output=True, text=True, timeout=3,
                                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    value = int(result.stdout.strip().splitlines()[0])
                    self.device_peak_used_mib = max(self.device_peak_used_mib or 0, value)
                except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                    pass
                last_gpu = time.monotonic()
            self.stop.wait(.5)

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)


def trial_worker(request_file: Path):
    request = json.loads(request_file.read_text(encoding="utf-8"))
    prepared, output = Path(request["prepared"]), request_file.parent
    identity = prepared_identity(prepared)
    if request.get("prepared_manifest_sha256") != identity["prepared_manifest_sha256"]:
        raise ValueError("Prepared manifest changed after trial was scheduled")
    request.update(identity)
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest)
    # Deliberately no test filename anywhere in this data-loading path.
    X = np.load(prepared / "X_train.npy", mmap_mode="r")
    y = np.load(prepared / "y_train.npy", mmap_mode="r")
    V = np.load(prepared / "X_validation.npy", mmap_mode="r")
    v = np.load(prepared / "y_validation.npy", mmap_mode="r")
    if X.shape[1] != len(manifest["feature_names"]) or V.shape[1] != X.shape[1]:
        raise ValueError("Prepared feature schema does not match manifest")
    record = {**request, "status": "running", "training_rows": len(y), "validation_rows": len(v),
              "features": manifest["feature_names"], "classes": manifest["classes"],
              "dataset_sha256": manifest["source_sha256"], "evaluation_partition": "validation",
              "test_evaluated": False, "python": platform.python_version(), "platform": platform.platform()}
    try:
        with ResourceSampler() as resources:
            if request["candidate"] in {"random_forest", "xgboost"}:
                from framework.training.trees import train_tree, predict_tree
                result = train_tree(request["candidate"], X, y, V, v, len(manifest["classes"]), request["config"], request["seed"], output)
                reload_predict = predict_tree
            elif request["candidate"] in {"catboost", "lightgbm"}:
                from framework.training.boosters import train_booster, predict_booster
                result = train_booster(request["candidate"], X, y, V, v, len(manifest["classes"]), request["config"], request["seed"], output)
                reload_predict = predict_booster
            else:
                from framework.training.neural import train_neural, predict_neural
                result = train_neural(request["candidate"], X, y, V, v, len(manifest["classes"]), manifest["benign_index"], request["config"], request["seed"], output)
                reload_predict = predict_neural
            predictions = result.pop("predictions")
            result.pop("probabilities", None)
            np.save(output / "validation_predictions.npy", predictions)
            # Real saved-checkpoint round trip, on a small fixed validation slice.
            count = min(128, len(V))
            started = time.perf_counter()
            reloaded = reload_predict(Path(result["checkpoint_path"]), V[:count])
            reload_seconds = time.perf_counter() - started
            if not np.array_equal(reloaded, predictions[:count]):
                raise AssertionError("Saved checkpoint predictions differ from selected model")
            record.update(result)
            record.update(status="completed", metrics=classification_metrics(v, predictions, manifest["classes"], manifest["benign_index"]),
                          checkpoint_reload_verified=True, reload_and_predict_seconds=reload_seconds, reload_prediction_rows=count)
        record["peak_process_rss_bytes_sampled"] = resources.peak_rss_bytes
        record["peak_device_used_mib_sampled"] = resources.device_peak_used_mib
        record["memory_note"] = "RAM sampled every 0.5s; VRAM device-wide every 5s includes other applications, not exact process peak."
        record["model_directory_bytes"] = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
        print(f"COMPLETED {request['run_id']} macro_F1={record['metrics']['macro_f1']:.5f} accuracy={record['metrics']['accuracy']:.5f}", flush=True)
    except Exception:
        record.update(status="failed", error=traceback.format_exc())
        print(record["error"], flush=True)
    write_json(output / "result.json", record)
    return 0 if record["status"] == "completed" else 1


def save_registry(output, records):
    write_json(output / "experiments.json", records)
    fields = ["run_id", "candidate", "config_id", "phase", "seed", "status", "macro_f1", "accuracy", "weighted_f1", "false_positive_rate", "training_seconds", "device"]
    with (output / "experiments.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get("metrics", {}).get(k, r.get(k)) for k in fields})
    text = "# Initial DataSense GPU candidate comparison\n\nValidation only. Test is locked. These are bounded within-capture experiments, not final research conclusions.\n\n"
    text += "| Candidate | Setting | Seed | Macro F1 | Accuracy | FPR | Seconds |\n|---|---|---:|---:|---:|---:|---:|\n"
    for r in records:
        if r["status"] == "completed":
            m = r["metrics"]
            text += f"| {r['candidate']} | {r['config_id']} | {r['seed']} | {m['macro_f1']:.5f} | {m['accuracy']:.5f} | {m['false_positive_rate']:.5f} | {r['training_seconds']:.1f} |\n"
        else:
            text += f"| {r['candidate']} | {r['config_id']} | {r['seed']} | FAILED | | | |\n"
    text += "\n## Selected configurations repeated across seeds\n\n"
    summaries = []
    for candidate in sorted({r["candidate"] for r in records}):
        search = [r for r in records if r["candidate"] == candidate and r["phase"] == "search" and r["status"] == "completed"]
        if not search:
            continue
        best = max(search, key=lambda r: r["metrics"]["macro_f1"])
        matching = [r for r in records if r["candidate"] == candidate and r["config_id"] == best["config_id"] and r["status"] == "completed"]
        values = [r["metrics"]["macro_f1"] for r in matching]
        summary = {"candidate": candidate, "selected_config": best["config_id"], "seeds": [r["seed"] for r in matching],
                   "macro_f1_mean": float(np.mean(values)), "macro_f1_std": float(np.std(values)),
                   "macro_f1_min": float(np.min(values)), "macro_f1_max": float(np.max(values))}
        summaries.append(summary)
        text += f"- {candidate}: {best['config_id']}, seeds {summary['seeds']}, macro F1 {summary['macro_f1_mean']:.5f} +/- {summary['macro_f1_std']:.5f}.\n"
    write_json(output / "comparison.json", summaries)
    (output / "comparison.md").write_text(text, encoding="utf-8")


def run_study(args):
    prepared = args.prepared.resolve()
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configs = configurations()
    study = {"prepared": str(prepared), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
             "configurations": configs, "candidates": args.candidates, "search_seed": 42, "repeat_seeds": [123, 2025],
             "budget": "Four configurations per candidate plus selected configuration repeated twice; epoch/tree caps and validation early stopping",
             "test_evaluated": False}
    study_file = output / "study.json"
    if study_file.exists() and json.loads(study_file.read_text()) != study:
        raise ValueError("Study identity differs; use a new output directory")
    write_json(study_file, study)
    records = json.loads((output / "experiments.json").read_text()) if (output / "experiments.json").exists() else []

    def execute(candidate, spec, seed, phase):
        completed = completed_trial(records, candidate, spec["id"], seed, spec["parameters"], prepared)
        if completed:
            print(f"Already completed: {candidate}/{spec['id']}/{seed}", flush=True)
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"{candidate}_{spec['id']}_s{seed}_{stamp}"
        directory = output / "runs" / run_id
        directory.mkdir(parents=True)
        request = {"run_id": run_id, "candidate": candidate, "config_id": spec["id"], "config": spec["parameters"],
                   "purpose": spec["purpose"], "seed": seed, "phase": phase, "prepared": str(prepared), "timestamp_utc": stamp}
        request.update(prepared_identity(prepared))
        request_file = directory / "request.json"
        write_json(request_file, request)
        print(f"START {candidate}/{spec['id']} seed={seed}", flush=True)
        env = {**os.environ, "PYTHONHASHSEED": str(seed), "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "4"}
        with (directory / "run.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", "experiment.train_candidates", "--worker", str(request_file)],
                                       cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line); log.flush()
            process.wait()
        result_path = directory / "result.json"
        record = json.loads(result_path.read_text()) if result_path.exists() else {**request, "status": "failed", "error": f"Worker exited {process.returncode}; inspect run.log"}
        records.append(record)
        best_path = output / "best" / f"{candidate}.json"
        incumbent = json.loads(best_path.read_text()) if best_path.exists() else None
        best = choose_best(incumbent, record)
        if best is record:
            write_json(best_path, record)
        save_registry(output, records)

    for candidate in args.candidates:
        for spec in configs[candidate]:
            execute(candidate, spec, 42, "search")
    if not args.search_only:
        for candidate in args.candidates:
            search = [r for r in records if r["candidate"] == candidate and r["phase"] == "search" and r["status"] == "completed"]
            if not search:
                continue
            selected = max(search, key=lambda r: r["metrics"]["macro_f1"])
            spec = next(s for s in configs[candidate] if s["id"] == selected["config_id"])
            for seed in [123, 2025]:
                execute(candidate, spec, seed, "repeat_selected")
    failures = [r for r in records if r["status"] != "completed"]
    print(f"Study saved to {output}; completed={len(records)-len(failures)}, failed={len(failures)}; TEST NOT EVALUATED", flush=True)
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=ROOT / "dataset/prepared/datasense_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/training/datasense_gpu_v1")
    parser.add_argument("--candidates", nargs="+", choices=list(configurations()), default=list(configurations()))
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    return trial_worker(args.worker) if args.worker else run_study(args)


if __name__ == "__main__":
    raise SystemExit(main())
