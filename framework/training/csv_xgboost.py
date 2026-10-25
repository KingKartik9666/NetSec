"""Full-observation CUDA XGBoost with accuracy-selected CPU-compatible checkpoints."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np

from .trees import _features, _labels, _macro_f1, _parameters, _require_cuda, class_weights


def train_csv_xgboost(X_train, y_train, X_val, y_val, n_classes: int,
                      config: dict, seed: int, output_dir: Path, *, benign_index: int = 0) -> dict:
    """Fit supplied CSV rows; optional synthetic additions follow real originals."""
    import xgboost as xgb
    from .csv_monitoring import snapshot
    from .tracking import write_json
    X, V = _features(X_train), _features(X_val)
    y, v = _labels(y_train, n_classes, 'training'), _labels(y_val, n_classes, 'validation')
    if len(X) != len(y) or len(V) != len(v) or X.shape[1] != V.shape[1]:
        raise ValueError('CSV feature and label dimensions differ')
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / 'model.ubj'
    if checkpoint.exists():
        raise FileExistsError('Preserve previous checkpoint; use a new attempt')
    params, rounds = _parameters('xgboost', n_classes, config, seed)
    params.update(disable_default_eval_metric=True, base_score=.5)
    weights = class_weights(y, n_classes, config.get('weight_strategy', 'none'))
    loss = config.get('loss', 'native_ce')
    if loss not in {'native_ce', 'label_smoothed_ce'}:
        raise ValueError('Unsupported CSV XGBoost loss')
    objective = None
    if loss == 'label_smoothed_ce':
        from .partial_objectives import CudaSmoothedObjective
        objective = CudaSmoothedObjective(y, weights[y], config.get('smoothing', .05))
    history = []
    started = time.perf_counter()
    runtime = min(float(config.get('runtime_seconds', 1050)),
                  float(os.environ.get('IMDAI_FIT_DEADLINE', 'inf')) - time.time() - 90)
    if runtime <= 0:
        raise TimeoutError('Insufficient remaining fit budget')
    training = xgb.QuantileDMatrix(X, label=y, weight=weights[y], max_bin=params['max_bin'], nthread=4)
    validation = xgb.DMatrix(V, label=v, nthread=4)
    original_rows = int(config.get('original_training_rows', len(y)))
    if not 0 < original_rows <= len(y):
        raise ValueError('Original training prefix is outside supplied rows')
    monitor_ids = np.random.default_rng(seed).choice(original_rows, min(4096, original_rows), replace=False)
    monitor = xgb.DMatrix(X[monitor_ids], nthread=2)

    class Selection(xgb.callback.TrainingCallback):
        def __init__(self):
            self.best = (-1., -1.)
            self.best_iteration = -1
            self.stale = 0
            self.runtime_limited = False
            self.pruned = False
            self.training_accuracy = None

        def before_training(self, model):
            _require_cuda(model)
            return model

        def after_iteration(self, model, epoch, evals_log):
            _require_cuda(model)
            pred = model.predict(validation).argmax(1)
            score = (float(np.mean(pred == v)), _macro_f1(v, pred, n_classes))
            scores = snapshot(v, pred, n_classes, benign_index)
            if epoch % 10 == 0:
                self.training_accuracy = float(np.mean(model.predict(monitor).argmax(1) == y[monitor_ids]))
            progress = dict(round=epoch + 1, validation_accuracy=score[0], validation_macro_f1=score[1],
                            validation_macro_recall=scores['macro_recall'], validation_macro_precision=scores['macro_precision'],
                            false_positive_rate=scores['false_positive_rate'], attack_missed_as_benign_rate=scores['attack_missed_as_benign_rate'])
            progress.update(training_accuracy=self.training_accuracy, training_monitor_rows=len(monitor_ids))
            history.append(progress)
            write_json(directory / 'progress.json', progress)
            if score > self.best:
                self.best, self.best_iteration, self.stale = score, epoch, 0
                model.save_model(checkpoint)
            else:
                self.stale += 1
            if epoch % 10 == 0:
                print(f'CSV XGBoost round={epoch + 1} accuracy={score[0]:.6f} macro_f1={score[1]:.6f}', flush=True)
            self.runtime_limited = time.perf_counter() - started >= runtime
            self.pruned = (directory / 'PRUNE').exists()
            return self.pruned or self.runtime_limited or self.stale >= int(config.get('early_stopping_rounds', 50))

    selection = Selection()
    fitted = xgb.train(params, training, num_boost_round=rounds, obj=objective, callbacks=[selection])
    actual = json.loads(fitted.save_config())
    device = _require_cuda(fitted)
    model = xgb.Booster(model_file=str(checkpoint))
    model.set_param({'device': 'cpu', 'nthread': 2})
    probability = model.predict(validation)
    predictions = probability.argmax(1)
    if (float(np.mean(predictions == v)), _macro_f1(v, predictions, n_classes)) != selection.best:
        raise RuntimeError('CPU checkpoint predictions differ from CUDA-selected validation result')
    (directory / 'actual_xgboost_config.json').write_text(json.dumps(actual, indent=2))
    return dict(predictions=predictions, probabilities=probability, checkpoint_path=str(checkpoint),
                device=device, gpu_training_verified=True, device_evidence=actual['learner']['generic_param'],
                backend='XGBoost CUDA; complete CSV observations only', history=history,
                training_seconds=time.perf_counter() - started, saved_rounds=model.num_boosted_rounds(),
                best_iteration=selection.best_iteration, runtime_limited=selection.runtime_limited,
                pruned=selection.pruned, stop_reason=('pruned' if selection.pruned else 'runtime_limit' if selection.runtime_limited
                         else 'early_stopping' if selection.stale >= int(config.get('early_stopping_rounds', 50)) else 'round_limit'),
                selection_criterion='full_csv_validation_accuracy; macro_f1 tie-break',
                effective_training_rows=len(y), masked_training_rows=0, class_weights=weights.tolist(),
                xgboost_version=xgb.__version__)
