"""Versioned broad-input search spaces with persisted Optuna ask/tell proposals."""
from pathlib import Path
import optuna

from framework.training.live_study.protocol import HYBRIDS, read, relative, fingerprint
from framework.training.live_study.search import configuration as established_configuration

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load(study, family):
    storage = optuna.storages.RDBStorage('sqlite:///' + (Path(study) / 'optuna.sqlite3').as_posix(),
                                       engine_kwargs={'connect_args': {'timeout': 60}})
    count = next((s.n_trials for s in optuna.get_all_study_summaries(storage)
                  if s.study_name == family), 0)
    # A fresh, trial-count-derived sampler seed is intentional. Persisted pending
    # proposals resume exactly; we do not claim SQLite serializes sampler RNG.
    return optuna.create_study(study_name=family, storage=storage, direction='maximize', load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42 + 1009*count, n_startup_trials=8,
                                           multivariate=True, group=True, constant_liar=False),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8,
                                          n_warmup_steps=60 if family == 'xgboost' else 8))


def configuration(trial, family, smoke=False):
    cat, number, integer = trial.suggest_categorical, trial.suggest_float, trial.suggest_int
    if family in HYBRIDS:
        if trial.number == 0:
            return dict(kind='global', penalty=None)
        return dict(kind='classwise', penalty=number('penalty', .01, 100, log=True))
    if family == 'hem':
        width = cat('width', [64, 128, 256])
        return dict(epochs=2 if smoke else 100, patience=2 if smoke else 12,
                    runtime_seconds=120 if smoke else 1500, batch_size=256,
                    lr=number('lr', 1e-4, 3e-3, log=True),
                    weight_decay=number('weight_decay', 1e-6, .01, log=True),
                    hidden=[width*2, width, max(16, width//2), 16],
                    binary_hidden=[width*2, width, max(16, width//2), 16],
                    pca_components=cat('pca_components', [16, 32, 64, 128]),
                    knn_k=cat('knn_k', [1, 3, 5, 9, 15]), knn_max_rows=20000, knn_batch_size=128,
                    scaling=cat('scaling', ['log_minmax', 'quantile']),
                    weight_strategy=cat('weight_strategy', ['none', 'sqrt_balanced', 'balanced']),
                    selection_metric='accuracy', scheduler='plateau', loss='native_ce')
    # These trainers already derive their input dimension from the manifest and
    # fit preprocessing only on fitting rows. Reuse their tested parameter map.
    return established_configuration(trial, family, smoke)


def proposal(study, family):
    settings = read(Path(study) / 'study.json')
    optimizer = load(study, family)
    pending = [t for t in optimizer.trials if t.state == optuna.trial.TrialState.RUNNING]
    if len(pending) > 1:
        raise RuntimeError('Only one pending proposal is allowed for each family')
    if pending:
        trial = optuna.trial.Trial(optimizer, pending[0]._trial_id)
        if 'request' in pending[0].user_attrs:
            return optimizer, trial, pending[0].user_attrs['request']
    else:
        trial = optimizer.ask()
    config = configuration(trial, family, settings['smoke'])
    request = dict(study=relative(study), family=family, phase='search', trial_number=trial.number,
                   seed=42, config=config, schema_fingerprint=settings['schema_fingerprint'],
                   protocol_fingerprint=settings['protocol_fingerprint'])
    request['request_fingerprint'] = fingerprint(request)
    trial.set_user_attr('request', request)
    return optimizer, trial, request
