"""Persisted TPE proposals; explicit ask/tell liveness belongs to the controller."""
from __future__ import annotations

from pathlib import Path
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from .protocol import HYBRIDS, read, relative, fingerprint


def load(study, family):
    path = Path(study)/'optuna.sqlite3'
    storage = optuna.storages.RDBStorage('sqlite:///'+path.as_posix(),
        engine_kwargs={'connect_args': {'timeout': 60}})
    trials = optuna.get_all_study_summaries(storage)
    count = next((s.n_trials for s in trials if s.study_name == family), 0)
    return optuna.create_study(study_name=family, storage=storage, direction='maximize', load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42+1009*count, n_startup_trials=4),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=60 if family == 'xgboost' else 8))


def configuration(t, family, smoke=False):
    cat, flt, integer = t.suggest_categorical, t.suggest_float, t.suggest_int
    runtime = 120 if smoke else 1500
    if family == 'random_forest':
        return dict(n_estimators=24 if smoke else cat('trees',[200,400,600]),
            max_depth=cat('depth',[12,24,None]), min_samples_leaf=cat('min_leaf',[1,2,4]),
            max_features=cat('max_features',['sqrt',.7,1.]), weight_strategy=cat('weight_strategy',['none','sqrt_balanced','balanced']),
            n_jobs=2, runtime_seconds=runtime)
    if family == 'xgboost':
        return dict(n_estimators=12 if smoke else 650, early_stopping_rounds=6 if smoke else 65,
            max_depth=integer('max_depth',3,8), learning_rate=flt('learning_rate',.02,.2,log=True),
            min_child_weight=flt('min_child_weight',.5,12,log=True), subsample=flt('subsample',.65,1),
            colsample_bytree=flt('colsample_bytree',.7,1), reg_lambda=flt('reg_lambda',.1,20,log=True),
            reg_alpha=flt('reg_alpha',1e-5,2,log=True), max_bin=128, nthread=2,
            weight_strategy=cat('weight_strategy',['none','sqrt_balanced','balanced']), runtime_seconds=runtime)
    if family in HYBRIDS:
        kind=cat('kind',['global','classwise'])
        return dict(kind=kind,penalty=flt('penalty',.1,100,log=True) if kind=='classwise' else None)
    common = dict(epochs=2 if smoke else 100, patience=2 if smoke else 12, batch_size=256,
        runtime_seconds=runtime, lr=flt('lr',1e-4,3e-3,log=True), weight_decay=flt('weight_decay',1e-6,.01,log=True))
    if family in ('kan','hem'):
        width=cat('width',[32,64,128])
        common.update(scaling=cat('scaling',['log_minmax','quantile']),
            weight_strategy=cat('weight_strategy',['none','sqrt_balanced','balanced']),
            selection_metric='accuracy', scheduler='plateau', loss='native_ce')
        if family == 'kan':
            common.update(hidden=[width]*integer('depth',1,3), grid_size=cat('grid_size',[3,5,8]),
                spline_order=cat('spline_order',[2,3]), spline_regularization=flt('spline_regularization',1e-7,1e-3,log=True))
        else:
            common.update(hidden=[width*2,width,max(16,width//2),16], binary_hidden=[width*2,width,max(16,width//2),16],
                pca_components=cat('pca_components',[5,7]), knn_k=cat('knn_k',[1,3,5,7]),
                knn_max_rows=20000, knn_batch_size=128)
        return common
    if family.startswith('tabm_'):
        common.update(k=cat('k',[4,8,16]), width=cat('width',[64,128,192]), depth=integer('depth',1,3),
            dropout=cat('dropout',[0.,.1,.2]), d_embedding=cat('d_embedding',[4,8]),
            n_bins=cat('n_bins',[8,16]), scaling=cat('scaling',['quantile','log_robust']),
            weight_power=cat('weight_power',[0.,.25,.5]), smoothing=cat('smoothing',[0.,.025,.05]),
            subtype_weight=flt('subtype_weight',.5,2) if family != 'tabm_ne' else 1.)
        return common
    if family == 'tabpack':
        from framework.training.tabpack_backend import pack_configuration
        cfg = pack_configuration(42)
        width=cat('width',[64,128,192]); depth=integer('depth',1,3)
        cfg.update(common, n_blocks=[max(1,depth-(i%2)) for i in range(8)],
            d_block=[width if i%2 else max(32,width//2) for i in range(8)],
            dropout=[cat('dropout',[0.,.1,.2])]*8,
            lr=[common['lr']*(.75 if i%2 else 1.25) for i in range(8)],
            weight_decay=[common['weight_decay']] * 8,
            scaling=cat('scaling',['quantile','log_robust']), n_bins=cat('n_bins',[8,16]))
        return cfg
    raise ValueError('Unknown live model family')


def proposal(study, family):
    settings = read(Path(study)/'study.json'); search = load(study, family)
    running = [t for t in search.trials if t.state == optuna.trial.TrialState.RUNNING]
    if len(running) > 1: raise RuntimeError('More than one pending proposal for a single worker')
    if running:
        trial = optuna.trial.Trial(search, running[0]._trial_id)
        if 'request' in running[0].user_attrs: return search, trial, running[0].user_attrs['request']
    else:
        if not search.trials and family in HYBRIDS:
            search.enqueue_trial({'kind':'global'})
        trial = search.ask()
    config = configuration(trial, family, settings['smoke'])
    request = dict(study=relative(study), family=family, phase='search', trial_number=trial.number,
        seed=42, config=config, schema_fingerprint=settings['schema_fingerprint'],
        protocol_fingerprint=settings['protocol_fingerprint'])
    request['request_fingerprint'] = fingerprint(request)
    trial.set_user_attr('request', request)
    return search, trial, request
