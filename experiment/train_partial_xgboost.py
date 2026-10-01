"""Frozen 12-configuration + four repeat CUDA study; held-out test stays locked.

Run ``python -m experiment.train_partial_xgboost``. Each fit has a separate
process/checkpoint; an interrupted study resumes completed identical requests.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from framework.detection.packet_xgboost import FEATURE_NAMES as LIVE_FEATURES, file_sha256
from framework.training.tracking import classification_metrics, validate_manifest, write_json
from framework.training.trees import _macro_f1, _require_cuda, class_weights

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "dataset/prepared/datasense_full_protocols"
OUTPUT = ROOT / "artifacts/training/partial_xgboost_v1"
ORIGINAL = ROOT / "framework/models/xgboost_app/bundle.json"
BUNDLE = ROOT / "framework/models/xgboost_partial/bundle.json"
MASK_SEED = 61429
PARTIAL_RATES = (.15, .4, .7)
MAX_ROUNDS = 500
PATIENCE = 40
RUNTIME_CAP_SECONDS = 720


def configurations():
    common = dict(max_bin=128, subsample=.8, colsample_bytree=1., colsample_bynode=.9,
                  min_child_weight=3., reg_lambda=3., reg_alpha=.05, gamma=0.,
                  weight_strategy="none", loss="native_ce", smoothing=0.)
    settings = [
        ("warm_d4_ce", True, 4, .08, {}),
        ("warm_d6_ce", True, 6, .06, {}),
        ("warm_d6_sqrt", True, 6, .06, {"weight_strategy": "sqrt_balanced"}),
        ("warm_d6_balanced", True, 6, .06, {"weight_strategy": "balanced"}),
        ("warm_d6_smooth", True, 6, .06, {"weight_strategy": "sqrt_balanced", "loss": "label_smoothed_ce", "smoothing": .05}),
        ("warm_d8_ce", True, 8, .04, {"reg_lambda": 6., "subsample": .7, "colsample_bynode": .8}),
        ("fresh_d4_ce", False, 4, .1, {}),
        ("fresh_d6_ce", False, 6, .08, {}),
        ("fresh_d6_sqrt", False, 6, .08, {"weight_strategy": "sqrt_balanced"}),
        ("fresh_d6_balanced", False, 6, .08, {"weight_strategy": "balanced"}),
        ("fresh_d6_smooth", False, 6, .08, {"weight_strategy": "sqrt_balanced", "loss": "label_smoothed_ce", "smoothing": .05}),
        ("fresh_d8_smooth", False, 8, .05, {"loss": "label_smoothed_ce", "smoothing": .02, "reg_lambda": 6., "subsample": .7, "colsample_bynode": .8}),
    ]
    return [{"id": name, "warm_start": warm, "max_depth": depth, "learning_rate": rate,
             **common, **changes} for name, warm, depth, rate, changes in settings]


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def masked_live(rows, indices):
    result = np.full(rows.shape, np.nan, dtype=np.float32)
    result[:, indices] = rows[:, indices]
    return result


def random_partial(rows, feature_names, observed_rate, seed):
    """Mask independently by raw field; a protocol field's flags stay together."""
    if not 0 < observed_rate < 1:
        raise ValueError("Observed feature rate must be between zero and one")
    keys = [name.split("__present_")[0] for name in feature_names]
    groups = list(dict.fromkeys(keys))
    group_index = np.array([groups.index(key) for key in keys])
    rng = np.random.default_rng(seed)
    present = rng.random((len(rows), len(groups))) < observed_rate
    empty = ~present.any(axis=1)
    present[np.flatnonzero(empty), rng.integers(len(groups), size=empty.sum())] = True
    result = np.asarray(rows, dtype=np.float32).copy()
    result[~present[:, group_index]] = np.nan
    return result


def load_data():
    """Explicit allowlist: only train/validation arrays, never any test array."""
    manifest = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if manifest["feature_names"] != original["feature_names"] or manifest["classes"] != original["classes"]:
        raise ValueError("Prepared and app feature/class order differ")
    if file_sha256(PREPARED / "manifest.json") != original["prepared_manifest_sha256"]:
        raise ValueError("Prepared manifest changed from original app")
    indices = [manifest["feature_names"].index(name) for name in LIVE_FEATURES]
    arrays, profiles = {}, {}
    for split in ("train", "validation"):
        X = np.load(PREPARED / f"X_{split}.npy", mmap_mode="r")
        y = np.load(PREPARED / f"y_{split}.npy", mmap_mode="r")
        row_ids = np.load(PREPARED / f"row_ids_{split}.npy", mmap_mode="r")
        start = np.load(PREPARED / f"window_start_{split}.npy", mmap_mode="r")
        end = np.load(PREPARED / f"window_end_{split}.npy", mmap_mode="r")
        if X.shape != (len(y), len(manifest['feature_names'])) or len(row_ids) != len(y):
            raise ValueError("Prepared row/schema mismatch")
        exact = np.flatnonzero(end-start == 10_000_000_000)
        selected = np.asarray(X[exact], dtype=np.float32)
        live = masked_live(selected, indices)
        counts = live[:, indices]
        if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any() or (counts[:, 1:] > counts[:, :1]).any():
            raise ValueError("Exact-ten-second source packet counts are invalid")
        random = np.concatenate([random_partial(selected, manifest['feature_names'], rate, MASK_SEED+i)
                                 for i, rate in enumerate(PARTIAL_RATES)])
        arrays[split] = dict(X=X, y=np.asarray(y, dtype=np.int32), row_ids=row_ids,
                             exact=exact, live=live, random=random,
                             live_y=np.asarray(y[exact], dtype=np.int32))
        profiles[split] = {
            "full_rows": len(y), "exact_ten_second_rows": len(exact),
            "full_support": np.bincount(y, minlength=len(manifest['classes'])).tolist(),
            "live_support": np.bincount(y[exact], minlength=len(manifest['classes'])).tolist(),
            "live_classes_with_support": int(len(np.unique(y[exact]))),
            "source_row_ids_sha256": digest(row_ids), "selected_row_ids_sha256": digest(row_ids[exact]),
            "features_sha256": file_sha256(PREPARED / f"X_{split}.npy"),
            "labels_sha256": file_sha256(PREPARED / f"y_{split}.npy"),
            "live_masked_features_sha256": digest(live), "random_masked_features_sha256": digest(random),
            "live_observed_columns": LIVE_FEATURES, "random_partial_rows": len(random),
        }
    if np.intersect1d(arrays['train']['row_ids'], arrays['validation']['row_ids']).size:
        raise ValueError("Train and validation source-row identities overlap")
    if len(np.unique(arrays['train']['y'])) != len(manifest['classes']):
        raise ValueError("Original training rows do not support every class")
    a = np.ascontiguousarray(arrays['train']['live'][:, indices]).view(np.dtype((np.void, len(indices)*4))).ravel()
    b = np.ascontiguousarray(arrays['validation']['live'][:, indices]).view(a.dtype).ravel()
    profiles['projection_overlap'] = {"validation_live_vectors_seen_in_train": int(np.isin(b, np.unique(a)).sum()),
        "note": "Source-row partitions are disjoint; identical projected count vectors can recur. This is not a cross-capture generalization study."}
    return manifest, original, arrays, profiles


def evaluate(model, data, manifest, directory):
    import xgboost as xgb
    model.set_param({"device": "cpu", "nthread": 4})
    profiles = {
        "full": (data['X'], data['y']),
        "packet_counts_7": (data['live'], data['live_y']),
        "random_partial": (data['random'], np.tile(data['live_y'], len(PARTIAL_RATES))),
    }
    metrics, probabilities = {}, {}
    for name, (features, labels) in profiles.items():
        prob = model.predict(xgb.DMatrix(features, nthread=4), strict_shape=True)
        if prob.shape != (len(labels), len(manifest['classes'])) or not np.isfinite(prob).all():
            raise RuntimeError("Invalid multiclass probability shape/values")
        if (prob < 0).any() or (prob > 1).any() or not np.allclose(prob.sum(axis=1), 1, atol=2e-5):
            raise RuntimeError("Multiclass probability normalization failed")
        metric = classification_metrics(labels, prob.argmax(axis=1), manifest['classes'], manifest['benign_index'])
        top3 = np.argpartition(prob, -3, axis=1)[:, -3:]
        metric.update(top3_accuracy=float(np.mean((top3 == labels[:, None]).any(axis=1))),
                      classes_with_support=int(len(np.unique(labels))), probability_sum_max_error=float(np.max(np.abs(prob.sum(axis=1)-1))))
        np.save(directory / f"{name}_validation_probabilities.npy", prob)
        metrics[name], probabilities[name] = metric, prob
    return metrics, probabilities


def trial(request_path):
    import xgboost as xgb
    from experiment.train_candidates import ResourceSampler
    from framework.training.partial_objectives import CudaSmoothedObjective
    request = json.loads(request_path.read_text(encoding="utf-8"))
    directory = request_path.parent
    start = time.perf_counter()
    try:
        manifest, original, arrays, profiles = load_data()
        if profiles != request['source_profiles']:
            raise ValueError("Prepared arrays or masks changed from frozen request")
        if not xgb.build_info().get('USE_CUDA'):
            raise RuntimeError("A CUDA-enabled XGBoost build is required")
        train, val, spec = arrays['train'], arrays['validation'], request['config']
        X = np.concatenate([train['X'], train['live'], train['random']])
        y = np.concatenate([train['y'], train['live_y'], np.tile(train['live_y'], len(PARTIAL_RATES))])
        profile_weight = np.concatenate([np.ones(len(train['y'])), np.full(len(train['live_y']), 8.),
                                         np.full(len(train['random']), 2.)]).astype(np.float32)
        class_weight = class_weights(train['y'], len(manifest['classes']), spec['weight_strategy'])
        weight = profile_weight * class_weight[y]
        params = {k: v for k, v in spec.items() if k not in {'id', 'warm_start', 'loss', 'smoothing', 'weight_strategy'}}
        params.update(objective="multi:softprob", num_class=len(manifest['classes']), tree_method="hist",
                      device="cuda:0", nthread=4, seed=request['seed'], disable_default_eval_metric=True)
        parent = (ORIGINAL.parent / original['checkpoint']).resolve()
        if file_sha256(parent) != original['checkpoint_sha256']:
            raise ValueError("Parent checkpoint checksum mismatch")
        if not spec['warm_start']:
            params['base_score'] = .5  # Explicit shared intercept for native/custom CE comparisons.
        objective = CudaSmoothedObjective(y, weight, spec['smoothing']) if spec['loss'] == 'label_smoothed_ce' else None
        history = []

        class GuardAndSelection(xgb.callback.TrainingCallback):
            def __init__(self):
                self.best = (-1., -1.)
                self.best_iteration = -1
                self.wait = 0
                self.cap_triggered = False

            def before_training(self, model):
                _require_cuda(model)
                return model

            def after_iteration(self, model, epoch, evals_log):
                _require_cuda(model)
                probability = model.predict(validation, strict_shape=True)
                prediction = probability.argmax(axis=1)
                pair = (float(np.mean(prediction == val['live_y'])), _macro_f1(val['live_y'], prediction, len(manifest['classes'])))
                elapsed = time.perf_counter()-start
                history.append({'round': epoch+1, 'accuracy': pair[0], 'macro_f1': pair[1], 'seconds': elapsed})
                if pair > self.best:
                    self.best, self.best_iteration, self.wait = pair, epoch, 0
                    model.save_model(directory / 'model.ubj')
                    np.save(directory / 'selected_live_probabilities.npy', probability)
                else:
                    self.wait += 1
                if epoch % 10 == 0 or pair == self.best:
                    print(f"round={epoch+1} live_accuracy={pair[0]:.6f} macro_f1={pair[1]:.6f} seconds={elapsed:.1f}", flush=True)
                self.cap_triggered = elapsed >= RUNTIME_CAP_SECONDS
                return self.cap_triggered or self.wait >= PATIENCE

            def after_training(self, model):
                _require_cuda(model)
                return model

        callback = GuardAndSelection()
        print(f"FIT {spec['id']} seed={request['seed']} original_rows={len(train['y'])} effective_rows={len(y)} loss={spec['loss']} CUDA", flush=True)
        with ResourceSampler() as resources:
            training = xgb.QuantileDMatrix(X, label=y, weight=weight, max_bin=spec['max_bin'], nthread=4)
            # Raw values must govern selection. QuantileDMatrix's new training
            # cuts may route old warm-start splits differently from app inputs.
            validation = xgb.DMatrix(val['live'], label=val['live_y'], nthread=4)
            fitted = xgb.train(params, training, num_boost_round=MAX_ROUNDS, obj=objective,
                               xgb_model=str(parent) if spec['warm_start'] else None, callbacks=[callback])
            actual = json.loads(fitted.save_config())
            training_seconds = time.perf_counter()-start
            del fitted, training, validation, X
            if objective is not None:
                import torch
                custom_calls = objective.calls
                del objective
                torch.cuda.empty_cache()
            else:
                custom_calls = 0
            model = xgb.Booster(model_file=str(directory / 'model.ubj'))
            metrics, probabilities = evaluate(model, val, manifest, directory)
            reloaded = xgb.Booster(model_file=str(directory / 'model.ubj'))
            reloaded.set_param({'device': 'cpu', 'nthread': 4})
            reload_probability = reloaded.predict(xgb.DMatrix(val['live'], nthread=4), strict_shape=True)
            if not np.array_equal(reload_probability, probabilities['packet_counts_7']):
                raise RuntimeError("Checkpoint reload predictions differ")
            selected_probability = np.load(directory / 'selected_live_probabilities.npy')
            if not np.array_equal(selected_probability.argmax(axis=1), reload_probability.argmax(axis=1)):
                raise RuntimeError("Selected per-row CUDA predictions differ from reloaded CPU checkpoint")
            if not np.allclose(selected_probability, reload_probability, atol=2e-5, rtol=2e-5):
                raise RuntimeError("Selected per-row CUDA probabilities differ from reloaded CPU checkpoint")
            if not np.isclose(metrics['packet_counts_7']['accuracy'], callback.best[0], atol=0, rtol=0):
                raise RuntimeError("Saved live validation accuracy differs from selection")
        result = {**request, 'status': 'completed', 'profile_metrics': metrics,
                  'training_seconds': training_seconds, 'total_seconds': time.perf_counter()-start,
                  'saved_rounds': model.num_boosted_rounds(), 'additional_best_iteration': callback.best_iteration,
                  'rounds_executed': len(history), 'runtime_cap_triggered': callback.cap_triggered,
                  'runtime_cap_seconds': RUNTIME_CAP_SECONDS, 'runtime_cap_granularity': 'checked after each complete multiclass boosting iteration',
                  'device': actual['learner']['generic_param']['device'], 'gpu_training_verified': True,
                  'custom_objective_device': 'torch cuda:0 with explicit NumPy ABI transfers' if custom_calls else None,
                  'custom_objective_calls': custom_calls, 'xgboost_version': xgb.__version__,
                  'checkpoint': (directory / 'model.ubj').relative_to(ROOT).as_posix(),
                  'checkpoint_sha256': file_sha256(directory / 'model.ubj'), 'reload_verified': True,
                  'effective_training_rows': len(y), 'original_training_rows': len(train['y']),
                  'masked_training_rows': len(y)-len(train['y']), 'class_weights': class_weight.tolist(),
                  'weight_sum': float(weight.sum(dtype=np.float64)),
                  'profile_weight_sums': {'full': float(weight[:len(train['y'])].sum(dtype=np.float64)),
                      'packet_counts_7': float(weight[len(train['y']):len(train['y'])+len(train['live_y'])].sum(dtype=np.float64)),
                      'random_partial': float(weight[len(train['y'])+len(train['live_y']):].sum(dtype=np.float64))},
                  'peak_rss_bytes': resources.peak_rss_bytes, 'device_peak_used_mib': resources.device_peak_used_mib,
                  'test_evaluated': False, 'live_windows_validation': False}
        write_json(directory / 'history.json', history)
        write_json(directory / 'actual_xgboost_config.json', actual)
    except Exception as exc:
        traceback.print_exc()
        result = {**request, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'total_seconds': time.perf_counter()-start}
    write_json(directory / 'result.json', result)
    return 0 if result['status'] == 'completed' else 1


def preflight():
    """Finite differences and actual CUDA custom/native ABI before any real fit."""
    import torch
    import xgboost as xgb
    from framework.training.partial_objectives import CudaSmoothedObjective, smoothed_derivatives
    code = subprocess.run([sys.executable, '-m', 'pytest', 'evaluation/test_partial_training.py', '-v'],
                          cwd=ROOT, capture_output=True, text=True)
    (OUTPUT / 'preflight_tests.log').write_text(code.stdout + code.stderr, encoding='utf-8')
    if code.returncode:
        raise RuntimeError('Custom-loss finite-difference/masking tests failed; see preflight_tests.log')
    if not torch.cuda.is_available() or not xgb.build_info().get('USE_CUDA'):
        raise RuntimeError('Actual CUDA required; no CPU fallback')
    rng = np.random.default_rng(17)
    X = rng.normal(size=(84, 7)).astype(np.float32)
    y = np.arange(84, dtype=np.int32) % 4
    weights = rng.uniform(.1, 2, len(y)).astype(np.float32)
    margins = rng.normal(size=(len(y), 4)).astype(np.float32)
    objective = CudaSmoothedObjective(y, weights, .05)
    g, h = objective(margins, None)
    expected_g, _, expected_h = smoothed_derivatives(margins, y, weights, .05)
    np.testing.assert_allclose(g, expected_g, atol=5e-7, rtol=2e-5)
    np.testing.assert_allclose(h, expected_h, atol=5e-7, rtol=2e-5)
    training = xgb.DMatrix(X, label=y, weight=weights, nthread=4)
    params = dict(objective='multi:softprob', num_class=4, base_score=.5, device='cuda:0',
                  tree_method='hist', max_depth=2, nthread=4, seed=42)
    native = xgb.train(params, training, num_boost_round=2)
    zero_smoothing = CudaSmoothedObjective(y, weights, 0.)
    custom = xgb.train(params, training, num_boost_round=2, obj=zero_smoothing)
    _require_cuda(native)
    _require_cuda(custom)
    a, b = native.predict(training), custom.predict(training)
    np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-6)
    write_json(OUTPUT / 'preflight.json', dict(numerical_tests_passed=True, cuda_gradient_agreement=True,
               zero_smoothing_native_ce_agreement=True, native_custom_max_error=float(np.max(np.abs(a-b))),
               device=torch.cuda.get_device_name(0), torch_version=torch.__version__, xgboost_version=xgb.__version__,
               preflight_data='Generated numerical-test matrices only; not research results'))
    del objective, zero_smoothing, training, native, custom
    torch.cuda.empty_cache()


def run_one(spec, seed, study, code_hashes):
    request_base = dict(config=spec, seed=seed, source_profiles=study['source_profiles'],
                        parent_checkpoint_sha256=study['parent_checkpoint_sha256'],
                        prepared_manifest_sha256=study['prepared_manifest_sha256'],
                        source_code_sha256=code_hashes, test_evaluated=False)
    directory = OUTPUT / 'runs' / f"{spec['id']}_s{seed}"
    request_path = directory / 'request.json'
    if request_path.exists():
        prior = json.loads(request_path.read_text(encoding='utf-8'))
        if any(prior.get(k) != v for k, v in request_base.items()):
            raise ValueError('Frozen trial identity changed; refusing ambiguous resume')
        if (directory / 'result.json').exists():
            result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
            if result['status'] == 'completed':
                if file_sha256(ROOT / result['checkpoint']) != result['checkpoint_sha256']:
                    raise ValueError('Completed checkpoint changed')
                print('RESUME completed ' + directory.name, flush=True)
                return result
            raise RuntimeError(f"Existing trial failed: {directory}; preserve evidence and repair explicitly")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        write_json(request_path, {**request_base, 'run_id': directory.name, 'started_utc': datetime.now(timezone.utc).isoformat()})
    print('START ' + directory.name, flush=True)
    with (directory / 'console.log').open('a', encoding='utf-8') as log:
        process = subprocess.run([sys.executable, '-u', '-m', 'experiment.train_partial_xgboost', '--trial', str(request_path)],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if not (directory / 'result.json').exists():
        raise RuntimeError(f'Trial ended without result (exit {process.returncode}); console retained at {directory}')
    result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
    if result['status'] != 'completed':
        raise RuntimeError(f"Trial failed: {result['error']}")
    m = result['profile_metrics']['packet_counts_7']
    print(f"DONE {directory.name} live_accuracy={m['accuracy']:.6f} macro_f1={m['macro_f1']:.6f} seconds={result['training_seconds']:.1f}", flush=True)
    return result


def ranking(result):
    m = result['profile_metrics']['packet_counts_7']
    return m['accuracy'], m['macro_f1']


def package(best, original, baseline, summaries):
    live = best['profile_metrics']['packet_counts_7']
    if live['accuracy'] <= baseline['profile_metrics']['packet_counts_7']['accuracy']:
        return False
    bundle = json.loads(json.dumps(original))
    bundle.update(schema_version=2, promoted=True, partial_input_supported=True,
                  checkpoint=Path(os.path.relpath(ROOT / best['checkpoint'], BUNDLE.parent)).as_posix(),
                  checkpoint_sha256=best['checkpoint_sha256'], parent_checkpoint_sha256=original['checkpoint_sha256'],
                  live_profile_validation={'profile': 'packet_counts_7', 'accuracy': live['accuracy'], 'macro_f1': live['macro_f1'],
                                           'top3_accuracy': live['top3_accuracy'], 'validation_rows': live['rows'],
                                           'classes_with_support': live['classes_with_support']},
                  profile_metrics=best['profile_metrics'],
                  promotion={'baseline_live_accuracy': baseline['profile_metrics']['packet_counts_7']['accuracy'],
                             'candidate_live_accuracy': live['accuracy'], 'improved': True},
                  training_study='artifacts/training/partial_xgboost_v1', repeated_seed_summary=summaries,
                  live_validation=False, test_evaluated=False)
    full = best['profile_metrics']['full']
    bundle['model'].update(version='partial-v1-' + best['run_id'], run_id=best['run_id'],
                           validation_accuracy=full['accuracy'], cpu_validation_accuracy=full['accuracy'], macro_f1=full['macro_f1'],
                           live_validation_accuracy=live['accuracy'], live_macro_f1=live['macro_f1'],
                           seed=best['seed'], saved_rounds=best['saved_rounds'], training_rows=best['original_training_rows'],
                           additional_masked_rows=best['masked_training_rows'], effective_training_rows=best['effective_training_rows'],
                           selection_basis='Top two seed-42 configurations repeated at seeds 17 and 73; choose largest mean seven-count validation accuracy, macro F1 tie-break, then best checkpoint in that configuration. Test remains locked.')
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    for filename in ('xai_reference.npz', 'xai_reference.json'):
        source, destination = ORIGINAL.parent / filename, BUNDLE.parent / filename
        shutil.copy2(source, destination)
        if file_sha256(source) != file_sha256(destination):
            raise RuntimeError('Frozen training-only XAI reference copy differs')
    write_json(BUNDLE, bundle)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.trial:
        return trial(args.trial)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    preflight()
    if args.preflight_only:
        return 0
    manifest, original, arrays, profiles = load_data()
    parent = (ORIGINAL.parent / original['checkpoint']).resolve()
    if file_sha256(parent) != original['checkpoint_sha256']:
        raise ValueError('Original checkpoint checksum mismatch')
    source_files = [Path(__file__), ROOT / 'framework/training/partial_objectives.py', ROOT / 'evaluation/test_partial_training.py']
    hashes = {file.relative_to(ROOT).as_posix(): file_sha256(file) for file in source_files}
    study = dict(schema_version=1, name='84-class partial-input XGBoost', configs=configurations(),
                 first_seed=42, repeat_seeds=[17, 73], repeat_top_configurations=2, planned_fits=16,
                 max_additional_rounds=MAX_ROUNDS, early_stopping_rounds=PATIENCE, runtime_cap_seconds=RUNTIME_CAP_SECONDS,
                 prepared_manifest_sha256=file_sha256(PREPARED / 'manifest.json'), source_sha256=manifest['source_sha256'],
                 parent_checkpoint_sha256=original['checkpoint_sha256'], source_profiles=profiles,
                 mask_seed=MASK_SEED, random_observed_rates=list(PARTIAL_RATES), missing_value='native IEEE NaN',
                 training_profile_weights={'full': 1, 'packet_counts_7': 8, 'random_partial_each': 2},
                 class_weight_source='Original training labels only; normalized by original sample frequency, capped at 20',
                 profile_weight_rationale='Explicit fixed weighting keeps full-row supervision and gives the live deployment profile substantial training loss mass; views are not independent observations.',
                 selection='Lexicographic seven-count validation accuracy, then all-84-class macro F1. Repeat top two at seeds17/73 and select configuration by three-seed mean, checkpoint by best validation pair.',
                 source_code_sha256=hashes, test_evaluated=False, live_windows_validation=False,
                 limitations=['DataSense within-capture validation only; no labelled Windows live accuracy.',
                              'Seven observed counts cannot distinguish all 84 labels; 24 labels lack support in exact-ten-second validation.',
                              'Exact publisher packet-extractor equivalence is unavailable.',
                              'GPU floating-point reductions can vary despite fixed seeds; seed repeats are validation results.'],
                 primary_sources=['https://xgboost.readthedocs.io/en/stable/python/examples/custom_softmax.html',
                                  'https://xgboost.readthedocs.io/en/stable/tutorials/advanced_custom_obj.html'])
    plan_path = OUTPUT / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8')) != study:
        raise ValueError('Frozen plan differs; do not mutate a started study')
    write_json(plan_path, study)
    for source in source_files:
        destination = OUTPUT / 'source_snapshot' / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    baseline_path = OUTPUT / 'baseline.json'
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    else:
        import xgboost as xgb
        directory = OUTPUT / 'baseline'
        directory.mkdir(exist_ok=True)
        metrics, _ = evaluate(xgb.Booster(model_file=str(parent)), arrays['validation'], manifest, directory)
        baseline = dict(checkpoint_sha256=original['checkpoint_sha256'], profile_metrics=metrics,
                        source_profiles=profiles, test_evaluated=False, evaluation_device='cpu')
        write_json(baseline_path, baseline)
    print('ORIGINAL MASKED BASELINE ' + json.dumps({name: {'accuracy': m['accuracy'], 'macro_f1': m['macro_f1']} for name, m in baseline['profile_metrics'].items()}), flush=True)
    del arrays
    results = []
    for spec in configurations():
        results.append(run_one(spec, 42, study, hashes))
        write_json(OUTPUT / 'progress.json', {'completed': len(results), 'planned': 16,
                   'runs': [{'run_id': r['run_id'], 'accuracy': ranking(r)[0], 'macro_f1': ranking(r)[1]} for r in results]})
    top = sorted(results, key=ranking, reverse=True)[:2]
    for chosen in top:
        for seed in (17, 73):
            results.append(run_one(chosen['config'], seed, study, hashes))
            write_json(OUTPUT / 'progress.json', {'completed': len(results), 'planned': 16,
                       'runs': [{'run_id': r['run_id'], 'accuracy': ranking(r)[0], 'macro_f1': ranking(r)[1]} for r in results]})
    summaries = []
    for chosen in top:
        rows = [r for r in results if r['config']['id'] == chosen['config']['id']]
        summaries.append({'config_id': chosen['config']['id'], 'seeds': [r['seed'] for r in rows],
                          'accuracy_mean': float(np.mean([ranking(r)[0] for r in rows])),
                          'accuracy_std': float(np.std([ranking(r)[0] for r in rows], ddof=1)),
                          'macro_f1_mean': float(np.mean([ranking(r)[1] for r in rows])),
                          'macro_f1_std': float(np.std([ranking(r)[1] for r in rows], ddof=1))})
    winning_config = max(summaries, key=lambda s: (s['accuracy_mean'], s['macro_f1_mean']))['config_id']
    best = max([r for r in results if r['config']['id'] == winning_config], key=ranking)
    promoted = package(best, original, baseline, summaries)
    write_json(OUTPUT / 'RESULTS.json', {'status': 'completed', 'completed_fits': len(results),
               'selected_run': best['run_id'], 'promoted': promoted, 'repeat_summary': summaries,
               'baseline_metrics': baseline['profile_metrics'], 'selected_metrics': best['profile_metrics'],
               'runs': [{k: r[k] for k in ('run_id', 'config', 'seed', 'profile_metrics', 'training_seconds',
                         'runtime_cap_triggered', 'checkpoint', 'checkpoint_sha256', 'saved_rounds')} for r in results],
               'test_evaluated': False, 'live_windows_validation': False})
    print(f"COMPLETE selected={best['run_id']} promoted={promoted}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
