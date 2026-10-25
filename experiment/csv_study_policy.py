"""Research policy for complete CSV observations; no live-profile objective."""
from __future__ import annotations

from copy import deepcopy
import random

FAMILIES = ('xgboost', 'kan', 'hem')
TARGET = .96


def ranking(result: dict) -> tuple[float, float]:
    metrics = result['metrics']
    return float(metrics['accuracy']), float(metrics['macro_f1'])


def controls() -> dict[str, dict]:
    """Fresh CSV controls preserve prior capacity, without masked examples."""
    return {
        'xgboost': dict(max_depth=6, learning_rate=.08, max_bin=128, subsample=.8,
                        colsample_bytree=1., colsample_bynode=.9, min_child_weight=3.,
                        reg_lambda=3., reg_alpha=.05, gamma=0., weight_strategy='sqrt_balanced',
                        loss='label_smoothed_ce', smoothing=.05, num_boost_round=800,
                        early_stopping_rounds=50, runtime_seconds=1050),
        'kan': dict(hidden=[96, 48], grid_size=5, spline_order=3, lr=.002,
                    weight_decay=.001, weight_strategy='sqrt_balanced', epochs=24,
                    batch_size=512, patience=5, spline_regularization=0.,
                    scaling='minmax', scheduler='none', selection_metric='accuracy', runtime_seconds=1050),
        'hem': dict(hidden=[512, 256, 32, 16], binary_hidden=[1024, 256, 64, 16],
                    pca_components=16, knn_k=1, knn_max_rows=480405, knn_batch_size=64,
                    lr=.001, weight_decay=.0001, weight_strategy='none', epochs=16,
                    batch_size=512, patience=5, scaling='minmax', scheduler='none',
                    selection_metric='accuracy', runtime_seconds=1050),
    }


def request(family: str, variant: str, config: dict, seed: int = 42) -> dict:
    if family not in FAMILIES:
        raise ValueError('Only the three upgraded families participate; RF remains fixed')
    return dict(model_id=family, variant=variant, config=deepcopy(config), seed=seed,
                input_profile='full_csv', research_policy='csv_v2')


def independent_trials() -> list[dict]:
    """One changed factor per matched control, interleaved across families."""
    changes = {
        'xgboost': [('loss', 'native_ce'), ('weight_strategy', 'none'), ('max_depth', 8),
                    ('max_bin', 256), ('reg_lambda', 6.), ('min_child_weight', 1.),
                    ('learning_rate', .04), ('subsample', 1.), ('max_depth', 10)],
        'kan': [('scaling', 'quantile'), ('weight_strategy', 'none'), ('scaling', 'log_minmax'),
                ('scheduler', 'cosine'), ('grid_size', 7), ('hidden', [128, 64]),
                ('lr', .001), ('spline_regularization', .0001)],
        'hem': [('pca_components', 32), ('scaling', 'quantile'), ('scaling', 'log_minmax'),
                ('pca_components', 64), ('scheduler', 'cosine'), ('pca_components', 128),
                ('lr', .0005), ('hidden', [768, 384, 128, 64])],
    }
    base, trials = controls(), []
    for index in range(max(map(len, changes.values()))):
        for family in FAMILIES:
            if index < len(changes[family]):
                key, value = changes[family][index]
                config = {**deepcopy(base[family]), key: value}
                trials.append(request(family, f'ablation_{index:02d}_{key}', config))
    return trials


def supported_combination(family: str, control: dict, results: list[dict]) -> dict | None:
    """Combine only independently improved factors; keep the best value per key."""
    config, changes = deepcopy(control['config']), {}
    for result in sorted(results, key=ranking, reverse=True):
        if not result['variant'].startswith('ablation_') or ranking(result) <= ranking(control):
            continue
        differences = {k: v for k, v in result['config'].items() if v != control['config'].get(k)}
        if len(differences) == 1:
            for key, value in differences.items():
                changes.setdefault(key, value)
    if len(changes) < 2:
        return None
    config.update(changes)
    return request(family, 'supported_combination', config)


def adaptive_trial(family: str, leader: dict, index: int, seen: set[str]) -> dict | None:
    """Deterministic local search around measured leaders, not Bayesian search."""
    import json
    spaces = {
        'xgboost': dict(max_depth=[4, 6, 8, 10], max_bin=[128, 256], learning_rate=[.03, .06, .1],
                        weight_strategy=['none', 'sqrt_balanced'], loss=['native_ce', 'label_smoothed_ce'],
                        reg_lambda=[1., 3., 6., 10.], subsample=[.8, 1.]),
        'kan': dict(scaling=['minmax', 'log_minmax', 'quantile'], lr=[.0005, .001, .002],
                    weight_strategy=['none', 'sqrt_balanced'], grid_size=[5, 7, 9],
                    scheduler=['none', 'cosine'], hidden=[[96, 48], [128, 64], [192, 96]]),
        'hem': dict(scaling=['minmax', 'log_minmax', 'quantile'], pca_components=[24, 32, 64, 128],
                    lr=[.0005, .001, .002], scheduler=['none', 'cosine'], weight_decay=[.0001, .001, .01]),
    }
    rng = random.Random(42000 + 1000 * FAMILIES.index(family) + index)
    for _ in range(100):
        config = deepcopy(leader['config'])
        for key in rng.sample(list(spaces[family]), k=1 if index % 2 == 0 else 2):
            config[key] = rng.choice(spaces[family][key])
        if json.dumps(config, sort_keys=True) not in seen:
            return request(family, f'adaptive_{index:03d}', config)
    return None
