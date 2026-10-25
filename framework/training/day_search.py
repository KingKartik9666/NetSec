"""Persistent multivariate Bayesian TPE for seven branches, with epoch pruning."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from framework.training.hybrid_hpo import sample_recipe
from framework.training.study_supervisor import FAMILIES


def load_study(folder, family):
    import optuna
    if family not in FAMILIES:
        raise ValueError('RF is fixed; unknown search family')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    storage = 'sqlite:///'+(folder/'optuna.sqlite3').resolve().as_posix()
    summaries = optuna.study.get_all_study_summaries(storage)
    count = next((s.n_trials for s in summaries if s.study_name == family), 0)
    sampler = optuna.samplers.TPESampler(seed=20261020+1000*FAMILIES.index(family)+count,
        n_startup_trials=6, multivariate=True, group=True)
    minimum = 60 if family == 'xgboost' else 6
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=minimum, reduction_factor=3)
    return optuna.create_study(study_name=family, storage=storage, load_if_exists=True,
        direction='maximize', sampler=sampler, pruner=pruner)


def sample(trial, family, base):
    if family in ('tabm_ne','xgboost','kan','tabpack'):
        request = sample_recipe(trial, family, base)
        cfg = request['config']
        if family == 'kan':
            cfg['spline_order'] = trial.suggest_categorical('spline_order', [2,3,4])
        if family == 'tabm_ne':
            cfg['k'] = trial.suggest_categorical('k',[8,16,32])
            cfg['batch_size'] = trial.suggest_categorical('batch_size',[256,512,1024])
            cfg['weight_power'] = trial.suggest_categorical('weight_power',[0.,.25,.5])
        if family == 'xgboost':
            cfg['gamma'] = trial.suggest_float('gamma', 1e-8, 2., log=True)
            cfg['weight_strategy'] = trial.suggest_categorical('weight_strategy', ['none','sqrt_balanced'])
    else:
        request = deepcopy(base)
        cfg = request['config']
        if family.startswith('tabm_'):
            cfg.update(k=trial.suggest_categorical('k',[8,16,32]),
                width=trial.suggest_categorical('width',[128,256,384,512]),
                depth=trial.suggest_int('depth',1,4),
                dropout=trial.suggest_float('dropout',0.,.35),
                d_embedding=trial.suggest_categorical('d_embedding',[8,12,16,24]),
                n_bins=trial.suggest_categorical('n_bins',[8,16,32,48]),
                scaling=trial.suggest_categorical('scaling',['log_robust','quantile']),
                lr=trial.suggest_float('lr',.0003,.003,log=True),
                weight_decay=trial.suggest_float('weight_decay',1e-6,.01,log=True),
                weight_power=trial.suggest_categorical('weight_power',[0.,.25,.5]),
                smoothing=trial.suggest_float('smoothing',0.,.05),
                subtype_weight=trial.suggest_float('subtype_weight',.5,2.))
        elif family == 'hem':
            cfg.update(hidden=[int(v) for v in trial.suggest_categorical('hidden',
                    ['256,128,64,32','512,256,64,32','768,384,128,32']).split(',')],
                binary_hidden=[int(v) for v in trial.suggest_categorical('binary_hidden',
                    ['256,128,32,16','512,128,32,16']).split(',')],
                pca_components=trial.suggest_categorical('pca_components',[16,32,64,96]),
                knn_k=trial.suggest_categorical('knn_k',[1,3,5]),knn_batch_size=32,
                lr=trial.suggest_float('lr',.0002,.003,log=True),
                weight_decay=trial.suggest_float('weight_decay',1e-6,.01,log=True),
                batch_size=trial.suggest_categorical('batch_size',[256,512,1024]),
                scaling=trial.suggest_categorical('scaling',['log_minmax','quantile']),
                weight_strategy=trial.suggest_categorical('weight_strategy',['none','sqrt_balanced']))
        request['arm'] = trial.suggest_categorical('arm',['full','strict'])
    # Hard per-fit bounds are part of each effective recipe, not implicit policy.
    cfg['runtime_seconds'] = 1500
    cfg['patience'] = 8
    if family != 'xgboost':
        cfg['epochs'] = 60
    request.update(family=family, seed=42)
    return request


def ask_request(folder, family, base):
    import optuna
    study = load_study(folder, family)
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.RUNNING:
            if 'request' not in trial.user_attrs:
                study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            else:
                return trial.user_attrs['request']
    trial = study.ask()
    # First fit is a measured warm start using the exact selected historic recipe.
    request = deepcopy(base) if trial.number == 0 else sample(trial, family, base)
    request.update(family=family, seed=42, phase='search', trial_number=trial.number,
        job=f'{family}_trial{trial.number:04d}')
    fingerprint = hashlib.sha256(json.dumps({k:request[k] for k in ('family','arm','seed','config')},
        sort_keys=True).encode()).hexdigest()
    if any(t.user_attrs.get('fingerprint') == fingerprint for t in study.trials if t.number != trial.number):
        study.tell(trial.number, state=optuna.trial.TrialState.PRUNED)
        return None
    trial.set_user_attr('fingerprint', fingerprint)
    trial.set_user_attr('request', request)
    return request


def tell(folder, family, number, value=None, state='COMPLETE'):
    import optuna
    study = load_study(folder, family)
    trial = next(t for t in study.trials if t.number == number)
    if not trial.state.is_finished():
        study.tell(number, value, state=getattr(optuna.trial.TrialState, state))

