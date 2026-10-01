"""Prepare, review, then run a bounded extension of the frozen partial study.

The default invocation prepares a plan only. GPU work additionally requires
``--run --reviewed-plan-sha256 HASH`` and a completed, promoted base study.
No held-out test array is loaded. Original study code/artifacts remain unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from experiment import train_partial_xgboost as base
from framework.detection.packet_xgboost import file_sha256
from framework.training.partial_augmentation import AugmentationBatch, audit_augmentation, augment_packet_minority, array_sha256
from framework.training.tracking import write_json
from framework.training.trees import _macro_f1, _require_cuda, class_weights

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/training/partial_enhancements_v1"
INCUMBENT = ROOT / "framework/models/xgboost_partial/bundle.json"
BUNDLE = ROOT / "framework/models/xgboost_partial_enhanced/bundle.json"
MAX_ROUNDS, PATIENCE, RUNTIME_CAP_SECONDS, SOFT_STOP_SECONDS = 500, 40, 600, 580
SEEDS = (42, 17, 73)
AUGMENTATION_SEED = 73219
SOURCES = (
    "https://imbalanced-learn.org/stable/over_sampling.html",
    "https://imbalanced-learn.org/stable/common_pitfalls.html",
    "https://imbalanced-learn.org/stable/auto_examples/over-sampling/plot_comparison_over_sampling.html",
    "https://xgboost.readthedocs.io/en/release_3.1.0/parameter.html",
    "https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html",
)
CODE_FILES = (
    "experiment/train_partial_enhancements.py", "framework/training/partial_augmentation.py",
    "evaluation/test_partial_augmentation.py", "experiment/train_partial_xgboost.py",
    "framework/training/partial_objectives.py", "framework/training/trees.py",
    "framework/training/tracking.py", "framework/detection/packet_xgboost.py",
    "experiment/train_candidates.py",
)


def configurations():
    return [
        {"id": "continuation_control", "augmentation": "none", "overrides": {}},
        {"id": "random_oversampling", "augmentation": "random_oversampling", "overrides": {}},
        {"id": "constrained_smote", "augmentation": "constrained_smote", "overrides": {}},
        {"id": "conservative_leaf_update", "augmentation": "none", "overrides": {"max_delta_step": 1.}},
        {"id": "gradient_sampling", "augmentation": "none", "overrides": {"sampling_method": "gradient_based"}},
    ]


def xgboost_parameters(incumbent_config, treatment, seed, num_classes):
    excluded = {"id", "warm_start", "loss", "smoothing", "weight_strategy"}
    params = {key: value for key, value in incumbent_config.items() if key not in excluded}
    params.update(treatment["overrides"])
    params.update(objective="multi:softprob", num_class=num_classes, tree_method="hist",
                  device="cuda:0", nthread=4, seed=seed, disable_default_eval_metric=True)
    return params


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_file_hash(path, expected):
    if file_sha256(Path(path)) != expected:
        raise ValueError(f"Frozen file changed: {path}")


def code_hashes():
    return {name: file_sha256(ROOT / name) for name in CODE_FILES}


def resolve_incumbent():
    result_path = base.OUTPUT / "RESULTS.json"
    if not result_path.is_file() or not INCUMBENT.is_file():
        raise RuntimeError("Base partial study is not complete/promoted; no enhancement fit may start")
    result, bundle = read_json(result_path), read_json(INCUMBENT)
    if (result.get("status") != "completed" or result.get("completed_fits") != 16
            or not result.get("promoted") or not bundle.get("promoted")
            or not bundle.get("promotion", {}).get("improved")
            or bundle.get("schema_version") != 2 or not bundle.get("partial_input_supported")):
        raise RuntimeError("A completed, explicitly promoted 16-fit base study is required")
    selected = next((run for run in result["runs"] if run["run_id"] == result["selected_run"]), None)
    if not selected or selected["run_id"] != bundle["model"]["run_id"]:
        raise ValueError("Incumbent bundle and final study selection disagree")
    checkpoint = (INCUMBENT.parent / bundle["checkpoint"]).resolve()
    require_file_hash(checkpoint, bundle["checkpoint_sha256"])
    if bundle["checkpoint_sha256"] != selected["checkpoint_sha256"]:
        raise ValueError("Incumbent checkpoint differs from the selected base run")
    return bundle, selected, checkpoint


def verify_plan(plan, plan_path=None):
    for name, expected in plan["source_code_sha256"].items():
        require_file_hash(ROOT / name, expected)
    require_file_hash(INCUMBENT, plan["incumbent_bundle_sha256"])
    require_file_hash(ROOT / plan["incumbent_checkpoint"], plan["incumbent_checkpoint_sha256"])
    require_file_hash(base.OUTPUT / "RESULTS.json", plan["base_results_sha256"])
    require_file_hash(base.PREPARED / "manifest.json", plan["prepared_manifest_sha256"])
    for asset in plan["augmentation_assets"].values():
        require_file_hash(ROOT / asset["path"], asset["sha256"])
        require_file_hash(ROOT / asset["metadata_path"], asset["metadata_sha256"])
    if plan_path is not None:
        require_file_hash(plan_path, plan["reviewed_plan_sha256"])


def prepare_plan():
    """Freeze code, existing partitions and generated training-only NPZs before GPU."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUTPUT / "plan.json"
    if plan_path.is_file():
        plan = read_json(plan_path)
        verify_plan(plan)
        print(f"Existing unchanged plan SHA256: {file_sha256(plan_path)}", flush=True)
        return plan
    bundle, chosen, checkpoint = resolve_incumbent()
    manifest, original, arrays, profiles = base.load_data()
    if profiles != read_json(base.OUTPUT / "plan.json")["source_profiles"]:
        raise ValueError("Base partition or missingness masks changed")
    if len(arrays["train"]["y"]) != 480405 or len(manifest["feature_names"]) != 170 or len(manifest["classes"]) != 84:
        raise ValueError("This extension requires the unchanged full 480405-row, 170-feature, 84-class study")
    if bundle["feature_names"] != original["feature_names"] or bundle["classes"] != original["classes"]:
        raise ValueError("Incumbent schema differs from the original frozen schema")
    hashes = code_hashes()
    assets = {}
    train = arrays["train"]
    for seed in (AUGMENTATION_SEED,):
        for method in ("random_oversampling", "constrained_smote"):
            batch = augment_packet_minority(train["live"], train["live_y"], train["row_ids"][train["exact"]],
                                            manifest["feature_names"], allowed_training_ids=train["row_ids"],
                                            num_classes=len(manifest["classes"]), method=method, seed=seed)
            audit_augmentation(batch, train["live"], train["live_y"], train["row_ids"][train["exact"]], manifest["feature_names"])
            key = f"{method}_s{seed}"
            directory = OUTPUT / "training_views" / key
            directory.mkdir(parents=True, exist_ok=True)
            filename = directory / "augmentation.npz"
            np.savez_compressed(filename, values=batch.values, labels=batch.labels, **batch.provenance)
            metadata = directory / "metadata.json"
            write_json(metadata, batch.summary)
            assets[key] = {"path": filename.relative_to(ROOT).as_posix(), "sha256": file_sha256(filename),
                           "metadata_path": metadata.relative_to(ROOT).as_posix(),
                           "metadata_sha256": file_sha256(metadata), "summary": batch.summary}
    plan = {"schema_version": 1, "study": "Bounded partial-input XGBoost enhancements",
            "incumbent_bundle_sha256": file_sha256(INCUMBENT),
            "incumbent_checkpoint": checkpoint.relative_to(ROOT).as_posix(),
            "incumbent_checkpoint_sha256": bundle["checkpoint_sha256"],
            "incumbent_config": chosen["config"], "incumbent_metrics": chosen["profile_metrics"],
            "base_results_sha256": file_sha256(base.OUTPUT / "RESULTS.json"),
            "prepared_manifest_sha256": file_sha256(base.PREPARED / "manifest.json"),
            "source_sha256": manifest["source_sha256"], "source_profiles": profiles,
            "source_code_sha256": hashes, "augmentation_assets": assets,
            "augmentation_seed": AUGMENTATION_SEED,
            "augmentation_repeat_policy": "Identical saved augmented views across seeds42/17/73; only XGBoost training randomness varies",
            "configurations": configurations(), "initial_seed": 42, "repeat_seeds": [17, 73],
            "maximum_fits": 9, "max_additional_rounds": MAX_ROUNDS, "patience": PATIENCE,
            "runtime_cap_seconds": RUNTIME_CAP_SECONDS, "soft_stop_seconds": SOFT_STOP_SECONDS,
            "runtime_cap_scope": "Entire GPU child process; hard supervisor timeout. CPU validation follows separately.",
            "selection_matrix": "Raw DMatrix; never QuantileDMatrix validation cuts",
            "include_incumbent_as_round_zero": True,
            "profile_weights": {"full_original": 1., "packet_counts_7": 8., "random_partial": 2., "augmented_packet_views": 8.},
            "class_weight_source": "Unchanged original full training labels only, using incumbent weight_strategy",
            "mask_seed": base.MASK_SEED, "random_observed_rates": list(base.PARTIAL_RATES),
            "repeat_gate": "Best seed42 treatment accuracy strictly exceeds control and unchanged incumbent",
            "promotion_gate": "Same-three-seed treatment mean accuracy exceeds control AND selected treatment checkpoint accuracy exceeds incumbent; macro F1 breaks ties",
            "test_evaluated": False, "live_windows_validation": False,
            "primary_sources": list(SOURCES),
            "limitations": ["Source-partition validation only; not live-PC accuracy or held-out test evidence.",
                            "Validation has support for only 60 of 84 labels in the exact-ten-second view.",
                            "Synthetic counts inherit labels; physical bounds do not prove attack semantics.",
                            "Nearest-neighbor interpolation uses training-only log1p counts, with no fitted normalization.",
                            "Full CSV validation regression is disclosed even if the primary seven-count metric improves.",
                            "The existing GAN study reduced full-feature validation accuracy; GAN is not a default extension."]}
    write_json(plan_path, plan)
    for name in CODE_FILES:
        destination = OUTPUT / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        require_file_hash(destination, hashes[name])
    print(f"Prepared CPU-only plan; no GPU fit executed. Review SHA256: {file_sha256(plan_path)}", flush=True)
    return plan


def build_training_views(train, plan, treatment, seed):
    """Preserve every original/full/live/random value and append additional views."""
    spec = plan["incumbent_config"]
    values = [train["X"], train["live"], train["random"]]
    labels = [train["y"], train["live_y"], np.tile(train["live_y"], len(base.PARTIAL_RATES))]
    profile = [np.ones(len(labels[0])), np.full(len(labels[1]), 8.), np.full(len(labels[2]), 2.)]
    summary = {"method": "none", "added_rows": 0, "synthetic_rows": 0, "duplicate_rows": 0}
    if treatment["augmentation"] != "none":
        asset = plan["augmentation_assets"][f"{treatment['augmentation']}_s{plan['augmentation_seed']}"]
        require_file_hash(ROOT / asset["path"], asset["sha256"])
        with np.load(ROOT / asset["path"], allow_pickle=False) as data:
            batch = AugmentationBatch(data["values"].copy(), data["labels"].copy(),
                                      {name: data[name].copy() for name in data.files if name not in {"values", "labels"}}, asset["summary"])
        audit_augmentation(batch, train["live"], train["live_y"], train["row_ids"][train["exact"]],
                           read_json(INCUMBENT)["feature_names"])
        values.append(batch.values)
        labels.append(batch.labels)
        profile.append(np.full(len(labels[-1]), 8.))
        summary = asset["summary"]
    x = np.concatenate(values).astype(np.float32, copy=False)
    y = np.concatenate(labels).astype(np.int32, copy=False)
    class_weight = class_weights(train["y"], len(plan["source_profiles"]["train"]["full_support"]), spec["weight_strategy"])
    weight = np.concatenate(profile).astype(np.float32)*class_weight[y]
    return x, y, weight, class_weight, summary


def trial(request_path):
    """Only the supervised GPU fitting phase; no test or full validation scoring."""
    import xgboost as xgb
    from framework.training.partial_objectives import CudaSmoothedObjective
    request = read_json(request_path)
    directory = Path(request_path).parent
    started = time.perf_counter()
    try:
        plan = read_json(OUTPUT / "plan.json")
        verify_plan(plan)
        require_file_hash(OUTPUT / "plan.json", request["plan_sha256"])
        if not xgb.build_info().get("USE_CUDA"):
            raise RuntimeError("A CUDA-enabled XGBoost build is required; no CPU fallback")
        manifest, _, arrays, profiles = base.load_data()
        if profiles != plan["source_profiles"]:
            raise ValueError("Training/validation arrays or original masks changed")
        train, val = arrays["train"], arrays["validation"]
        x, y, weight, class_weight, augmentation = build_training_views(train, plan, request["config"], request["seed"])
        spec = plan["incumbent_config"]
        params = xgboost_parameters(spec, request["config"], request["seed"], len(manifest["classes"]))
        objective = CudaSmoothedObjective(y, weight, spec["smoothing"]) if spec["loss"] == "label_smoothed_ce" else None
        training = xgb.QuantileDMatrix(x, label=y, weight=weight, max_bin=params["max_bin"], nthread=4)
        validation = xgb.DMatrix(val["live"], label=val["live_y"], nthread=4)
        history = []
        progress = {"request": request, "status": "fitting", "rounds_executed": 0,
                    "additional_best_iteration": 0, "original_training_rows": len(train["y"]),
                    "effective_training_rows": len(y), "augmentation": augmentation,
                    "class_weights": class_weight.tolist(), "weight_sum": float(weight.sum(dtype=np.float64)),
                    "training_values_sha256": array_sha256(x), "training_labels_sha256": array_sha256(y),
                    "training_weights_sha256": array_sha256(weight), "xgboost_version": xgb.__version__,
                    "test_evaluated": False, "live_windows_validation": False}

        class Selection(xgb.callback.TrainingCallback):
            def __init__(self):
                self.best = (-1., -1.)
                self.wait = 0

            def assess(self, model, iteration):
                _require_cuda(model)
                probability = model.predict(validation, strict_shape=True)
                predicted = probability.argmax(axis=1)
                pair = (float(np.mean(predicted == val["live_y"])),
                        _macro_f1(val["live_y"], predicted, len(manifest["classes"])))
                elapsed = time.perf_counter()-started
                history.append({"additional_round": iteration, "accuracy": pair[0], "macro_f1": pair[1], "seconds": elapsed})
                if pair > self.best:
                    self.best, self.wait = pair, 0
                    model.save_model(directory / "model.ubj")
                    np.save(directory / "selected_live_probabilities.npy", probability)
                    progress.update(additional_best_iteration=iteration, selection_accuracy=pair[0],
                                    selection_macro_f1=pair[1], checkpoint_sha256=file_sha256(directory / "model.ubj"))
                elif iteration:
                    self.wait += 1
                progress.update(rounds_executed=iteration, training_seconds=elapsed,
                                soft_runtime_cap_triggered=elapsed >= SOFT_STOP_SECONDS,
                                custom_objective_calls=objective.calls if objective else 0)
                write_json(directory / "history.json", history)
                write_json(directory / "fit_progress.json", progress)
                if iteration % 10 == 0 or pair == self.best:
                    print(f"round={iteration} accuracy={pair[0]:.6f} macro_f1={pair[1]:.6f} seconds={elapsed:.1f}", flush=True)
                return elapsed >= SOFT_STOP_SECONDS or self.wait >= PATIENCE

            def before_training(self, model):
                _require_cuda(model)
                actual = json.loads(model.save_config())
                write_json(directory / "actual_xgboost_config.json", actual)
                self.assess(model, 0)
                return model

            def after_iteration(self, model, epoch, evals_log):
                return self.assess(model, epoch+1)

            def after_training(self, model):
                _require_cuda(model)
                return model

        xgb.train(params, training, num_boost_round=MAX_ROUNDS, obj=objective,
                  xgb_model=str(ROOT / plan["incumbent_checkpoint"]), callbacks=[Selection()])
        progress.update(status="fit_completed", training_seconds=time.perf_counter()-started)
        write_json(directory / "fit_progress.json", progress)
        return 0
    except Exception as exc:
        traceback.print_exc()
        write_json(directory / "fit_failure.json", {"error": f"{type(exc).__name__}: {exc}",
                   "seconds": time.perf_counter()-started})
        return 1


def completed_run(directory, request):
    """Reject ambiguous resumes and changed completed checkpoints."""
    directory = Path(directory)
    if (directory / "request.json").is_file() and read_json(directory / "request.json") != request:
        raise ValueError("Frozen run identity changed; refusing ambiguous resume")
    if not (directory / "result.json").is_file():
        return None
    result = read_json(directory / "result.json")
    if result.get("status") != "completed" or result.get("request") != request:
        raise ValueError("Existing failed or mismatched run must be preserved and reviewed")
    require_file_hash(directory / "model.ubj", result["checkpoint_sha256"])
    return result


def finalize_fit(directory, request, plan, *, hard_cap=False):
    """CPU-only validation after GPU child exit, using the selected raw-input model."""
    import xgboost as xgb
    progress = read_json(directory / "fit_progress.json")
    if not (directory / "actual_xgboost_config.json").is_file() or progress["request"] != request:
        raise ValueError("Missing verified GPU fitting identity")
    require_file_hash(directory / "model.ubj", progress["checkpoint_sha256"])
    actual = read_json(directory / "actual_xgboost_config.json")
    device = actual["learner"]["generic_param"]["device"]
    if not device.startswith("cuda"):
        raise RuntimeError("The saved fit did not execute CUDA; no CPU substitution")
    manifest, _, arrays, profiles = base.load_data()
    if profiles != plan["source_profiles"]:
        raise ValueError("Validation source identities changed")
    model = xgb.Booster(model_file=str(directory / "model.ubj"))
    metrics, probabilities = base.evaluate(model, arrays["validation"], manifest, directory)
    selected = np.load(directory / "selected_live_probabilities.npy", allow_pickle=False)
    cpu = probabilities["packet_counts_7"]
    if not np.array_equal(selected.argmax(axis=1), cpu.argmax(axis=1)) or not np.allclose(selected, cpu, atol=2e-5, rtol=2e-5):
        raise RuntimeError("Selected CUDA and reloaded CPU raw-input predictions differ")
    if metrics["packet_counts_7"]["accuracy"] != progress["selection_accuracy"]:
        raise RuntimeError("Reloaded model differs from selected validation accuracy")
    result = {**progress, "status": "completed", "run_id": directory.name,
              "profile_metrics": metrics, "saved_rounds": model.num_boosted_rounds(),
              "checkpoint": (directory / "model.ubj").relative_to(ROOT).as_posix(),
              "device": device, "gpu_training_verified": True, "reload_verified": True,
              "hard_runtime_cap_triggered": hard_cap, "runtime_cap_seconds": RUNTIME_CAP_SECONDS,
              "validation_device": "cpu", "test_evaluated": False, "live_windows_validation": False}
    write_json(directory / "result.json", result)
    return result


def run_one(treatment, seed, plan, plan_sha):
    verify_plan(plan)
    request = {"config": treatment, "seed": seed, "plan_sha256": plan_sha,
               "incumbent_checkpoint_sha256": plan["incumbent_checkpoint_sha256"],
               "source_code_sha256": plan["source_code_sha256"], "test_evaluated": False}
    directory = OUTPUT / "runs" / f"{treatment['id']}_s{seed}"
    prior = completed_run(directory, request)
    if prior:
        print("RESUME " + directory.name, flush=True)
        return prior
    if directory.exists():
        raise RuntimeError(f"Interrupted or failed trial retained at {directory}; review before retrying")
    directory.mkdir(parents=True)
    write_json(directory / "request.json", request)
    hard_cap = False
    started = time.monotonic()
    with (directory / "console.log").open("w", encoding="utf-8") as log:
        try:
            process = subprocess.run([sys.executable, "-u", "-m", "experiment.train_partial_enhancements",
                                      "--trial", str(directory / "request.json")], cwd=ROOT, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=RUNTIME_CAP_SECONDS,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except subprocess.TimeoutExpired:
            hard_cap = True
    wall_seconds = time.monotonic()-started
    if not hard_cap and process.returncode:
        raise RuntimeError(f"GPU trial failed; see {directory / 'console.log'}")
    if not (directory / "fit_progress.json").is_file():
        raise RuntimeError("Fit ended without a complete selected checkpoint; no result substituted")
    write_json(directory / "supervisor.json", {"hard_cap": hard_cap, "gpu_child_wall_seconds": wall_seconds,
                                               "limit_seconds": RUNTIME_CAP_SECONDS})
    return finalize_fit(directory, request, plan, hard_cap=hard_cap)


def ranking(result):
    values = result["profile_metrics"]["packet_counts_7"]
    return values["accuracy"], values["macro_f1"]


def repeated_summary(rows):
    if {row["request"]["seed"] for row in rows} != set(SEEDS) or len(rows) != len(SEEDS):
        raise ValueError("Treatment and control require the same three distinct seeds")
    return {"seeds": sorted(SEEDS), "accuracy_mean": float(np.mean([ranking(row)[0] for row in rows])),
            "accuracy_std": float(np.std([ranking(row)[0] for row in rows], ddof=1)),
            "macro_f1_mean": float(np.mean([ranking(row)[1] for row in rows])),
            "macro_f1_std": float(np.std([ranking(row)[1] for row in rows], ddof=1))}


def promotion_decision(control, treatment, incumbent_accuracy):
    control_summary, treatment_summary = repeated_summary(control), repeated_summary(treatment)
    best = max(treatment, key=ranking)
    promoted = (treatment_summary["accuracy_mean"] > control_summary["accuracy_mean"]
                and ranking(best)[0] > incumbent_accuracy)
    return promoted, best, {"control": control_summary, "treatment": treatment_summary}


def package(best, plan, summaries):
    """Versioned bundle only; the incumbent and default-selector code are untouched."""
    require_file_hash(INCUMBENT, plan["incumbent_bundle_sha256"])
    if (summaries["treatment"]["accuracy_mean"] <= summaries["control"]["accuracy_mean"]
            or ranking(best)[0] <= plan["incumbent_metrics"]["packet_counts_7"]["accuracy"]):
        raise ValueError("Enhancement did not pass both promotion gates")
    bundle = read_json(INCUMBENT)
    full, live = best["profile_metrics"]["full"], best["profile_metrics"]["packet_counts_7"]
    bundle.update(schema_version=2, promoted=True, partial_input_supported=True,
                  checkpoint=Path(os.path.relpath(ROOT / best["checkpoint"], BUNDLE.parent)).as_posix(),
                  checkpoint_sha256=best["checkpoint_sha256"],
                  parent_checkpoint_sha256=plan["incumbent_checkpoint_sha256"],
                  profile_metrics=best["profile_metrics"], repeated_seed_summary=summaries,
                  live_profile_validation={"profile": "packet_counts_7", "accuracy": live["accuracy"],
                      "macro_f1": live["macro_f1"], "top3_accuracy": live["top3_accuracy"],
                      "validation_rows": live["rows"], "classes_with_support": live["classes_with_support"]},
                  promotion={"improved": True, "candidate_live_accuracy": live["accuracy"],
                      "baseline_live_accuracy": plan["incumbent_metrics"]["packet_counts_7"]["accuracy"],
                      "matched_control_mean": summaries["control"]["accuracy_mean"],
                      "treatment_mean": summaries["treatment"]["accuracy_mean"],
                      "full_csv_accuracy_change": full["accuracy"]-plan["incumbent_metrics"]["full"]["accuracy"]},
                  training_study="artifacts/training/partial_enhancements_v1",
                  augmentation=best["augmentation"], test_evaluated=False, live_validation=False)
    bundle["model"].update(version="partial-enhanced-v1-"+best["run_id"], run_id=best["run_id"],
                           validation_accuracy=full["accuracy"], cpu_validation_accuracy=full["accuracy"],
                           macro_f1=full["macro_f1"], live_validation_accuracy=live["accuracy"],
                           live_macro_f1=live["macro_f1"], saved_rounds=best["saved_rounds"],
                           seed=best["request"]["seed"], training_rows=best["original_training_rows"],
                           effective_training_rows=best["effective_training_rows"],
                           additional_synthetic_rows=best["augmentation"]["synthetic_rows"],
                           additional_duplicate_rows=best["augmentation"]["duplicate_rows"],
                           selection_basis="Five matched seed42 continuations; best improving treatment and control repeated at17/73. Promotion requires improved three-seed mean and a checkpoint above the immutable incumbent. Test remains locked.")
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    for name in ("xai_reference.npz", "xai_reference.json"):
        source, destination = INCUMBENT.parent / name, BUNDLE.parent / name
        shutil.copy2(source, destination)
        require_file_hash(destination, file_sha256(source))
    write_json(BUNDLE, bundle)


def run_study(reviewed_hash):
    plan_path = OUTPUT / "plan.json"
    if not plan_path.is_file() or not reviewed_hash:
        raise ValueError("Prepare and review plan.json, then supply --reviewed-plan-sha256")
    require_file_hash(plan_path, reviewed_hash)
    plan = read_json(plan_path)
    verify_plan(plan)
    resolve_incumbent()
    lock = OUTPUT / "running.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "started": datetime.now(timezone.utc).isoformat()}))
    try:
        rows = [run_one(config, 42, plan, reviewed_hash) for config in configurations()]
        control, treatment = rows[0], max(rows[1:], key=ranking)
        incumbent_accuracy = plan["incumbent_metrics"]["packet_counts_7"]["accuracy"]
        promoted, summaries, selected = False, None, treatment
        if ranking(treatment)[0] > max(ranking(control)[0], incumbent_accuracy):
            chosen = treatment["request"]["config"]
            for seed in (17, 73):
                rows.extend([run_one(configurations()[0], seed, plan, reviewed_hash),
                             run_one(chosen, seed, plan, reviewed_hash)])
            controls = [row for row in rows if row["request"]["config"]["id"] == "continuation_control"]
            treatments = [row for row in rows if row["request"]["config"]["id"] == chosen["id"]]
            promoted, selected, summaries = promotion_decision(controls, treatments, incumbent_accuracy)
            if promoted:
                package(selected, plan, summaries)
        result = {"status": "completed", "completed_fits": len(rows), "promoted": promoted,
                  "selected_run": selected["run_id"] if promoted else None,
                  "best_treatment_run": selected["run_id"], "repeat_summary": summaries,
                  "incumbent_metrics": plan["incumbent_metrics"], "best_treatment_metrics": selected["profile_metrics"],
                  "full_csv_accuracy_change": selected["profile_metrics"]["full"]["accuracy"]-plan["incumbent_metrics"]["full"]["accuracy"],
                  "plan_sha256": reviewed_hash, "runs": rows,
                  "test_evaluated": False, "live_windows_validation": False}
        write_json(OUTPUT / "RESULTS.json", result)
        print(json.dumps({"completed_fits": len(rows), "promoted": promoted, "selected_run": result["selected_run"]}), flush=True)
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--reviewed-plan-sha256")
    args = parser.parse_args()
    if args.trial:
        return trial(args.trial)
    if args.run:
        run_study(args.reviewed_plan_sha256)
    else:
        prepare_plan()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
