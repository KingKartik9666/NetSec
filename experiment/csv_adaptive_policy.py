"""Evidence-driven local search and documented responses to diminishing returns."""
from __future__ import annotations

from copy import deepcopy
import json

from experiment.csv_study_policy import adaptive_trial, ranking, request


def adaptive_decision(family: str, rows: list[dict], index: int, seen: set[str]) -> dict | None:
    """Monitor the last six fits; expand training/capacity only on a plateau.

    This is deterministic heuristic search, not Bayesian optimization or proof
    that the proposed change improves performance. All proposals still need fits.
    """
    ordered = sorted(rows, key=lambda r: r.get('completed_at', 0))
    leader = max(ordered, key=ranking)
    old_best = max((ranking(r)[0] for r in ordered[:-6]), default=ranking(ordered[0])[0])
    recent_best = max(ranking(r)[0] for r in ordered[-6:])
    gain = max(0., recent_best - old_best)
    plateau = len(ordered) >= 7 and gain < .001
    metrics = leader['metrics']
    control = next((r for r in ordered if r['variant'] == 'control'), ordered[0])
    tradeoff = (metrics.get('macro_recall', 1.) < control['metrics'].get('macro_recall', 0.) - .01
                or (metrics.get('false_positive_rate') or 0.) > (control['metrics'].get('false_positive_rate') or 0.) + .005)
    reason = 'Explore one or two changes around the measured full-CSV leader'
    proposal = None
    if plateau or tradeoff:
        # Each expansion remains within the original family and runtime budget.
        expansion = {
            'xgboost': [('weight_strategy', 'sqrt_balanced'), ('loss', 'native_ce'),
                        ('max_depth', 12), ('learning_rate', .02), ('max_bin', 512),
                        ('reg_lambda', 10.), ('early_stopping_rounds', 100), ('num_boost_round', 1200)],
            'kan': [('weight_strategy', 'sqrt_balanced'), ('scaling', 'quantile'),
                    ('hidden', [192, 96, 48]), ('grid_size', 11), ('epochs', 48),
                    ('scheduler', 'cosine'), ('lr', .0005), ('batch_size', 1024)],
            'hem': [('pca_components', 170), ('scaling', 'quantile'),
                    ('hidden', [1024, 512, 128, 64]), ('epochs', 32), ('scheduler', 'cosine'),
                    ('weight_strategy', 'sqrt_balanced'), ('lr', .00025), ('batch_size', 1024)],
        }[family]
        for offset in range(len(expansion)):
            key, value = expansion[(index + offset) % len(expansion)]
            config = deepcopy(leader['config'])
            config[key] = value
            if json.dumps(config, sort_keys=True) not in seen:
                proposal = request(family, f'adaptive_{index:03d}', config)
                reason = f"{'Recall/false-alarm tradeoff' if tradeoff else 'Accuracy plateau'}: independently test {key}={value}"
                break
    if proposal is None:
        proposal = adaptive_trial(family, leader, index, seen)
    if proposal:
        proposal['decision'] = dict(reason=reason, plateau=plateau, recent_accuracy_gain=gain,
             monitored_metrics={key: metrics.get(key) for key in
                                ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
                                 'false_positive_rate', 'attack_missed_as_benign_rate')},
             recall_or_false_alarm_tradeoff=tradeoff, leader_checkpoint=leader.get('checkpoint_path'),
             model_selection='full CSV accuracy first; macro F1 tie-break; tradeoffs are flagged, not hidden')
    return proposal
