"""Versioned capacity experiments over the preserved four-model trainers.

Original upstream repositories and previous study trainers are never rewritten.
Modified HEM widths/PCA/KNN settings are HEM-derived experiments, not upstream
architecture reproductions. RF does not accept capacity overrides here.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def partial_options(config: dict) -> tuple[int, int, dict]:
    """Remove controller options before forwarding native XGBoost parameters."""
    params = copy.deepcopy(config)
    rounds = params.pop('num_boost_round', 500)
    patience = params.pop('early_stopping_rounds', 40)
    for name, value, maximum in [('num_boost_round', rounds, 1200),
                                 ('early_stopping_rounds', patience, 120)]:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(f'{name} must be an integer in [1, {maximum}]')
    return rounds, patience, params


def fit(request_path: Path) -> None:
    from experiment import train_four_models as original
    request = json.loads(request_path.read_text(encoding='utf-8'))
    if request['model_id'] == 'xgboost':
        from experiment import train_partial_xgboost as partial
        rounds, patience, _ = partial_options(request['config'])
        previous_trial = partial.trial
        previous_limits = partial.MAX_ROUNDS, partial.PATIENCE

        def capacity_trial(path: Path) -> int:
            trial_request = json.loads(path.read_text(encoding='utf-8'))
            _, _, trial_request['config'] = partial_options(trial_request['config'])
            sanitized = path.parent / 'capacity_request.json'
            sanitized.write_text(json.dumps(trial_request, indent=2), encoding='utf-8')
            return previous_trial(sanitized)

        partial.trial = capacity_trial
        partial.MAX_ROUNDS, partial.PATIENCE = rounds, patience
        try:
            original.fit(request_path)
        finally:
            partial.trial = previous_trial
            partial.MAX_ROUNDS, partial.PATIENCE = previous_limits
    else:
        original.fit(request_path)
    result_path = request_path.parent / 'result.json'
    result = json.loads(result_path.read_text(encoding='utf-8'))
    # The preserved trainer merges its inner sanitized result over the request.
    # Reinstate controller-only capacity options for immutable resume/repeats.
    result['config'] = copy.deepcopy(request['config'])
    if request['model_id'] == 'xgboost':
        result['effective_trainer_limits'] = {'num_boost_round': rounds,
                                             'early_stopping_rounds': patience,
                                             'inner_runtime_cap_seconds': partial.RUNTIME_CAP_SECONDS,
                                             'outer_runtime_cap_seconds': 1200}
    if request.get('architecture_modified'):
        result['architecture_scope'] = (
            'HEM-derived capacity/PCA/KNN experiment using original upstream neural classes; '
            'not an unchanged upstream architecture or unknown-attack reproduction'
            if request['model_id'] == 'hem' else f"Modified {request['model_id']} architecture experiment")
        if request['model_id'] == 'hem':
            result['backend'] = 'HEM-derived DataSense closed-set architecture variant'
    from experiment.optimize_models import write
    write(result_path, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    fit(parser.parse_args().fit)
