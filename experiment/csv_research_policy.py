"""Training-only full-CSV proposals and evidence gates; importing needs no ML stack.

Proposals are not measured gains. HEM proposals adapt the upstream neural
classes in the existing closed-set experiment, not its novel-attack protocol.
Optuna is imported only when the explicitly requested research tuner is used.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

FAMILIES = ('xgboost', 'kan', 'hem')
TARGET = .95
SEEDS = (42, 43, 44)
PIPELINE = dict(feature_selection='none', feature_engineering=False,
                resampling='none', order='selection_then_resampling')
_BUDGETS = {'runtime_seconds', 'watchdog_seconds', 'timeout_seconds'}
_AUGMENTATION_DEFAULTS = dict(max_extra_fraction=.2, max_per_class_multiplier=2,
                              k_neighbors=5, ctgan_epochs=20)


def _family(family: str) -> None:
    if family not in FAMILIES:
        raise ValueError('Only XGBoost, efficient-KAN and upstream-derived HEM are tunable; RF is fixed')


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if k not in _BUDGETS}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def effective_recipe(request: dict, *, include_seed: bool = True) -> dict:
    """Normalize known inactive knobs while retaining all active model/pipeline knobs."""
    config = _clean(deepcopy(request['config']))
    config.setdefault('weight_strategy', 'none')
    config.setdefault('loss', 'native_ce')
    if config['loss'] == 'native':
        config['loss'] = 'native_ce'
    if config['loss'] != 'label_smoothed_ce':
        config.pop('smoothing', None)
        config.pop('label_smoothing', None)
    else:
        config.setdefault('smoothing', .05)
    if config['loss'] != 'focal':
        config.pop('focal_gamma', None)
        config.pop('focal_alpha', None)
    else:
        config.setdefault('focal_gamma', 2.)
    if request['model_id'] != 'xgboost':
        config.setdefault('scheduler', 'none')
    if config.get('scheduler', 'none') == 'none':
        for key in list(config):
            if key.startswith(('scheduler_', 'plateau_', 'cosine_')):
                del config[key]
    pipeline = _clean({**PIPELINE, **deepcopy(request.get('pipeline', {}))})
    if pipeline['resampling'] == 'none':
        for key in list(pipeline):
            if key.startswith(('resampling_', 'smote_', 'ctgan_', 'adasyn_')):
                del pipeline[key]
    else:
        parameters = {**_AUGMENTATION_DEFAULTS, **pipeline.get('resampling_params', {})}
        if pipeline['resampling'] != 'ctgan':
            parameters.pop('ctgan_epochs', None)
        pipeline['resampling_params'] = parameters
    if pipeline['feature_selection'] == 'none':
        pipeline.pop('selection_params', None)
    if not pipeline['feature_engineering']:
        pipeline.pop('feature_engineering_params', None)
    if pipeline['resampling'] == 'none' or pipeline['feature_selection'] == 'none':
        pipeline['order'] = 'selection_then_resampling'
    effective = dict(model_id=request['model_id'], config=config, pipeline=pipeline,
                     input_profile=request.get('input_profile', 'full_csv'))
    # Data/split/protocol identities matter; labels, telemetry and budgets do not.
    for key, value in request.items():
        if key in {'input', 'profile', 'protocol', 'protocol_id', 'research_policy',
                   'split_hash', 'source_sha256', 'prepared', 'prepared_manifest_sha256',
                   'train_row_ids_sha256', 'validation_row_ids_sha256'} or key.startswith(('input_', 'dataset_', 'split_')):
            effective[key] = _clean(value)
    if include_seed:
        effective['seed'] = int(request.get('seed', 42))
    return effective


def _digest(value: dict) -> str:
    payload = json.dumps(_clean(value), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def fingerprint(request: dict) -> str:
    """Effective fit identity, including its deterministic seed."""
    return _digest(effective_recipe(request))


def recipe_key(request: dict) -> str:
    """Family/data/protocol-specific recipe identity used to match seed repeats."""
    return _digest(effective_recipe(request, include_seed=False))


def _request(family: str, control: dict) -> dict:
    _family(family)
    if 'config' in control:
        if control.get('model_id', family) != family:
            raise ValueError('A model family cannot be changed by relabelling its configuration')
        row = {k: deepcopy(v) for k, v in control.items()
               if k not in {'metrics', 'execution', 'verification', 'checkpoint_path',
                            'checkpoint_reload_verified', 'cuda_verified', 'optuna_trial_number',
                            'optuna_study_name', 'completed_at', 'decision', 'fingerprint', 'recipe_key'}}
    else:
        row = dict(config=deepcopy(control))
    row.update(model_id=family, variant='control', seed=42)
    row.setdefault('input_profile', 'full_csv')
    row.setdefault('research_policy', 'csv_research_v3')
    row['pipeline'] = {**PIPELINE, **deepcopy(row.get('pipeline', {}))}
    return row


def _factors(family: str) -> list[dict]:
    factors = [dict(name=selection, group='feature_selection', pipeline={'feature_selection': selection})
               for selection in ('mi64', 'mi128')]
    factors.append(dict(name='feature_engineering', group='feature_engineering', pipeline={'feature_engineering': True}))
    factors.extend(dict(name=sampler, group='resampling', pipeline={'resampling': sampler})
                   for sampler in ('smotenc', 'borderline_smote', 'adasyn', 'ctgan'))
    # Both existing tree/neural implementations support these train-label-only
    # weighting strategies. They are one mutually exclusive ablation factor.
    factors.extend(dict(name=f'weight_{strategy}', group='weight_strategy',
                        config={'weight_strategy': strategy})
                   for strategy in ('none', 'sqrt_balanced', 'balanced'))
    losses = ('native_ce', 'label_smoothed_ce') if family == 'xgboost' else ('native_ce', 'focal', 'label_smoothed_ce')
    factors.extend(dict(name=loss, group='loss', config={'loss': loss, **({'smoothing': .05} if loss == 'label_smoothed_ce' else {})}) for loss in losses)
    if family != 'xgboost':
        factors.extend(dict(name=f'scheduler_{scheduler}', group='scheduler', config={'scheduler': scheduler})
                       for scheduler in ('none', 'cosine', 'plateau'))
    return factors


def build_recipes(controls: dict) -> tuple[list[dict], list[dict]]:
    """Controls, solo changes and compatible pairs; exclusions are auditable.

    Selection plus resampling has both measured execution orders. Two values of
    the same operation (especially two resamplers) are mutually exclusive.
    """
    rows, rejected, seen = [], [], set()
    for family in FAMILIES:
        base = _request(family, controls[family])
        rows.append(base)
        seen.add(fingerprint(base))
        factors = _factors(family)
        for group in [(factor,) for factor in factors] + list(combinations(factors, 2)):
            names = [factor['name'] for factor in group]
            if len(group) == 2 and group[0]['group'] == group[1]['group']:
                rejected.append(dict(model_id=family, factors=names, reason='mutually_exclusive'))
                continue
            orders = ('selection_then_resampling', 'resampling_then_selection') if {factor['group'] for factor in group} == {'feature_selection', 'resampling'} else ('selection_then_resampling',)
            for order in orders:
                row = deepcopy(base)
                row['variant'] = ('solo_' if len(group) == 1 else 'pair_') + '__'.join(names)
                row['pipeline']['order'] = order
                if len(orders) > 1:
                    row['variant'] += '__' + order
                for factor in group:
                    row['config'].update(factor.get('config', {}))
                    row['pipeline'].update(factor.get('pipeline', {}))
                row['pipeline'] = effective_recipe(row)['pipeline']
                key = fingerprint(row)
                if key in seen:
                    rejected.append(dict(model_id=family, factors=names, order=order,
                                         reason='duplicate_effective_recipe', fingerprint=key))
                    continue
                seen.add(key)
                rows.append(row)
    # Emit in control/solo/pair order so seed-42 independent evidence comes first.
    rows.sort(key=lambda row: (0 if row['variant'] == 'control' else 1 if row['variant'].startswith('solo_') else 2,
                               FAMILIES.index(row['model_id'])))
    return rows, rejected


def capacity_trial(family: str, leader: dict, stage: int) -> dict:
    """Three bounded capacity tiers, followed by tier rotation and unique tuning."""
    _family(family)
    if stage < 0:
        raise ValueError('Capacity stage must be nonnegative')
    row = _request(family, leader)
    row['variant'] = f'capacity_{stage:04d}'
    tier, cycle = stage % 3, stage // 3
    row['watchdog_seconds'] = (1200, 3600, 14400)[tier]
    config = row['config']
    config['runtime_seconds'] = row['watchdog_seconds'] - 180
    if family == 'xgboost':
        config.update(max_depth=(8, 12, 16)[tier], num_boost_round=(1500, 3000, 6000)[tier])
    elif family == 'kan':
        config.update(hidden=([128, 64], [256, 128, 64], [512, 256, 128])[tier],
                      grid_size=(5, 9, 15)[tier], epochs=(64, 128, 256)[tier])
    else:
        config.update(hidden=([768, 384, 64, 32], [1024, 512, 128, 64], [1536, 768, 256, 128])[tier],
                      binary_hidden=([1024, 512, 128, 32], [1536, 768, 256, 64], [2048, 1024, 256, 128])[tier],
                      pca_components=(32, 64, 128, 170)[stage % 4], epochs=(64, 128, 256)[tier])
        row['architecture_provenance'] = 'HEM-derived upstream neural classes; modified widths/PCA; closed-set CSV'
    if cycle:
        # Capacity does not grow indefinitely. A bounded, distinct learning rate
        # makes repeated tiers real trials rather than relabelled duplicate fits.
        key = 'learning_rate' if family == 'xgboost' else 'lr'
        config[key] = float(leader['config'].get(key, .001)) / (1. + .05 * cycle)
    return row


def score(result: dict) -> tuple[float, float]:
    return float(result['metrics']['accuracy']), float(result['metrics']['macro_f1'])


def _metric(metrics: dict, name: str) -> float | None:
    value = metrics.get(name)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 0. <= value <= 1. else None


def success_report(rows: list[dict], controls: dict) -> dict:
    """Require >=95% for the identical recipe at seeds 42/43/44 and verified fits.

    Control macro recall/FPR must exist. Missing evidence fails closed. This
    function reports eligibility; it does not promote or modify any checkpoint.
    """
    groups, candidates, rejections, regressions = {}, [], [], []
    for row in rows:
        if row.get('model_id') in FAMILIES and int(row.get('seed', 42)) in SEEDS:
            try:
                key = recipe_key(row)
            except (TypeError, ValueError, KeyError):
                rejections.append(dict(model_id=row.get('model_id'), reason='invalid_effective_recipe'))
                continue
            groups.setdefault(key, []).append(row)
    for key, repeats in groups.items():
        family = repeats[0]['model_id']
        present = sorted({int(row.get('seed', 42)) for row in repeats})
        reasons = []
        if present != list(SEEDS):
            reasons.append('missing_identical_recipe_seeds')
        control = controls.get(family, {})
        if isinstance(control, list):
            control = next((row for row in control if row.get('variant') == 'control'), {})
        control_metrics = control.get('metrics', {})
        reference_recall = _metric(control_metrics, 'macro_recall')
        reference_fpr = _metric(control_metrics, 'false_positive_rate')
        if reference_recall is None or reference_fpr is None:
            reasons.append('missing_control_metrics')
        for row in repeats:
            metrics = row.get('metrics', {})
            accuracy, recall, fpr = (_metric(metrics, name) for name in ('accuracy', 'macro_recall', 'false_positive_rate'))
            if accuracy is None or _metric(metrics, 'macro_f1') is None or recall is None or fpr is None:
                reasons.append('invalid_or_missing_metrics')
            elif accuracy < TARGET:
                reasons.append('accuracy_below_target')
            verification = row.get('verification', {})
            if row.get('checkpoint_reload_verified', verification.get('cpu_reload_verified')) is not True:
                reasons.append('cpu_reload_unverified')
            if row.get('cuda_verified', verification.get('cuda_verified')) is not True:
                reasons.append('cuda_unverified')
            flags = []
            if recall is not None and reference_recall is not None and reference_recall - recall > .01 + 1e-12:
                flags.append('macro_recall_decline_gt_1pp')
            if fpr is not None and reference_fpr is not None and fpr - reference_fpr > .005 + 1e-12:
                flags.append('false_positive_rate_rise_gt_0_5pp')
            if flags:
                reasons.append('material_regression')
                regressions.append(dict(model_id=family, recipe_key=key, seed=row['seed'], flags=flags))
        record = dict(model_id=family, recipe_key=key, seeds=present,
                      target=TARGET, reasons=sorted(set(reasons)))
        if reasons:
            rejections.append(record)
        else:
            record.update(min_accuracy=min(score(row)[0] for row in repeats),
                          min_macro_f1=min(score(row)[1] for row in repeats),
                          fingerprints=[fingerprint(row) for row in repeats])
            candidates.append(record)
    candidates.sort(key=lambda row: (row['min_accuracy'], row['min_macro_f1']), reverse=True)
    return dict(success=bool(candidates), target=TARGET, required_seeds=list(SEEDS),
                candidates=candidates, rejections=rejections, material_regressions=regressions,
                automatic_promotion=False)


def _study(study_path, family: str, index: int = 0):
    _family(family)
    import optuna
    path = Path(study_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(study_name=f'csv_research_v3_{family}',
                              storage='sqlite:///' + path.as_posix(), load_if_exists=True,
                              direction='maximize',
                              sampler=optuna.samplers.TPESampler(seed=42000 + 1000 * FAMILIES.index(family) + index),
                              pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5))


def _spaces(family: str, pipeline: dict | None = None) -> dict:
    if family == 'xgboost':
        spaces = dict(learning_rate=[.02, .04, .06, .08, .1], max_depth=[6, 8, 12, 16],
                    reg_lambda=[1., 3., 6., 10.], min_child_weight=[1., 3., 6.],
                    max_bin=[128, 256, 512], subsample=[.7, .8, 1.],
                    loss=['native_ce', 'label_smoothed_ce'],
                    weight_strategy=['none', 'sqrt_balanced', 'balanced'])
    else:
        spaces = dict(lr=[.00025, .0005, .001, .002, .004], weight_decay=[.0001, .001, .01],
                  scheduler=['none', 'cosine', 'plateau'], loss=['native_ce', 'focal', 'label_smoothed_ce'],
                  batch_size=[256, 512, 1024], weight_strategy=['none', 'sqrt_balanced', 'balanced'])
        if family == 'kan':
            spaces.update(grid_size=[5, 9, 15], spline_regularization=[0., .0001, .001])
        else:
            spaces.update(pca_components=[32, 64, 128, 170], knn_k=[1, 3, 5])
    pipeline = {**PIPELINE, **(pipeline or {})}
    if pipeline['feature_selection'] != 'none':
        spaces['pipeline.feature_selection'] = ['mi64', 'mi128']
    if pipeline['resampling'] != 'none':
        spaces.update({'pipeline.resampling_params.max_extra_fraction': [.1, .2, .4],
                       'pipeline.resampling_params.max_per_class_multiplier': [1.5, 2, 3],
                       'pipeline.resampling_params.k_neighbors': [3, 5, 9]})
        if pipeline['resampling'] == 'ctgan':
            spaces['pipeline.resampling_params.ctgan_epochs'] = [10, 20, 40, 80]
    if pipeline['feature_engineering']:
        spaces['pipeline.feature_engineering'] = [False, True]
    if pipeline['feature_selection'] != 'none' and pipeline['resampling'] != 'none':
        spaces['pipeline.order'] = ['selection_then_resampling', 'resampling_then_selection']
    return spaces


def propose_tpe(study_path, family: str, leader: dict, completed: list[dict], index: int) -> dict:
    """Persist a seed-42 TPE ask before execution, recovering an unfinished index.

    Initial proposals change one knob. Subsequent proposals tune two. Active
    MI sizes, engineering/order and compatible augmentation parameters are
    tunable; the resampler type and genuine model family remain those of the leader.
    Duplicate proposals are pruned and never sent to a training worker.
    """
    import optuna
    study = _study(study_path, family, index)
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.RUNNING and trial.user_attrs.get('proposal_index') == index:
            saved = trial.user_attrs.get('request')
            if saved is not None:
                return deepcopy(saved)
    base = _request(family, leader)
    base['seed'] = 42
    seen = {fingerprint(row) for row in completed if 'config' in row and 'model_id' in row}
    seen.add(fingerprint(base))
    seen.update(trial.user_attrs['fingerprint'] for trial in study.trials if 'fingerprint' in trial.user_attrs)
    spaces = _spaces(family, base['pipeline'])
    for attempt in range(256):
        trial = study.ask()
        row = deepcopy(base)
        # Stable solo selection ensures genuine independent ablations before
        # multi-parameter search. Values are chosen by the durable TPE study.
        names = list(spaces)
        # Retry no-op draws on the intended factor before moving to another
        # factor. Otherwise a default-valued draw could skip recipe tuning.
        position = (index + attempt // 8) % len(names)
        factor = names[position]
        knobs = [factor] if index < len(names) else [factor, names[(position + 1) % len(names)]]
        for key in knobs:
            # The expanded weighting choices coexist with historical persisted
            # Optuna distributions, whose categorical choices are immutable.
            parameter_name = 'weight_strategy_v2' if key == 'weight_strategy' else key
            value = trial.suggest_categorical(parameter_name, spaces[key])
            if key.startswith('pipeline.resampling_params.'):
                parameter = key.rsplit('.', 1)[1]
                row['pipeline'].setdefault('resampling_params', {})[parameter] = value
            elif key.startswith('pipeline.'):
                row['pipeline'][key.split('.', 1)[1]] = value
            else:
                row['config'][key] = value
        row['pipeline'] = effective_recipe(row)['pipeline']
        key = fingerprint(row)
        if key in seen:
            trial.set_user_attr('rejection', 'duplicate_effective_recipe')
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            continue
        row.update(variant=f'tpe_{index:04d}', optuna_trial_number=trial.number,
                   optuna_study_name=study.study_name)
        trial.set_user_attr('proposal_index', index)
        trial.set_user_attr('fingerprint', key)
        trial.set_user_attr('request', row)
        return row
    raise RuntimeError('TPE could not produce a unique effective proposal after 256 attempts')


def report_trial(study_path, family: str, trial_number: int, step: int, value: float) -> bool:
    """Persist intermediate accuracy and return the configured pruning decision."""
    import optuna
    study = _study(study_path, family)
    frozen = next(t for t in study.trials if t.number == trial_number)
    if frozen.state != optuna.trial.TrialState.RUNNING:
        return frozen.state == optuna.trial.TrialState.PRUNED
    trial = optuna.trial.Trial(study, frozen._trial_id)
    trial.report(float(value), int(step))
    return trial.should_prune()


def tell_trial(study_path, family: str, trial_number: int, value: float | None,
               state: str = 'complete') -> None:
    """Persist measured accuracy or a terminal pruned/fail state, idempotently."""
    import optuna
    states = dict(complete=optuna.trial.TrialState.COMPLETE,
                  pruned=optuna.trial.TrialState.PRUNED, fail=optuna.trial.TrialState.FAIL)
    if state not in states:
        raise ValueError('Trial state must be complete, pruned or fail')
    if state == 'complete' and (value is None or not math.isfinite(float(value))):
        raise ValueError('Completed trials need a finite measured score')
    study = _study(study_path, family)
    frozen = next(t for t in study.trials if t.number == trial_number)
    if frozen.state.is_finished():
        if frozen.state == states[state] and (state != 'complete' or frozen.value == float(value)):
            return
        raise ValueError('A persisted terminal trial cannot be rewritten')
    study.tell(trial_number, float(value) if state == 'complete' else None, state=states[state])
