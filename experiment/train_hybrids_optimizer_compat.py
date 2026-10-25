"""Recorded continuation-budget adapter; preserve the sealed hybrid study.

Only classwise L-BFGS-B's maximum iterations changes from 150 to 500.
Success/convergence tolerances, training rows, recipes and existing results stay
the same. Every adopted execution records this adapter and its source digest.
"""
from __future__ import annotations

import sys

from experiment import train_hybrids as base
from experiment.datasense_quality import digest
from experiment.train_verified_clean import read, write
from framework.training import hybrid_blend, hybrid_evaluation
from framework.training.hybrid_protocol import ROOT, STUDY, verify_protocol

MODULE = 'experiment.train_hybrids_optimizer_compat'
ADAPTER = STUDY/'controller_adapters/class_blend_iterations_v1.json'
SOURCE = 'experiment/train_hybrids_optimizer_compat.py'


def bounded_minimize(original, objective, initial, **kwargs):
    if kwargs.get('method') == 'L-BFGS-B':
        options = dict(kwargs.get('options', {}))
        if options.get('maxiter') != 150:
            raise ValueError('Unexpected original class-blend iteration budget')
        options['maxiter'] = 500
        kwargs['options'] = options
    return original(objective, initial, **kwargs)


def adoption():
    protocol = read(STUDY/'study.json'); verify_protocol(protocol)
    identity = dict(name='class_blend_iterations_v1', source=SOURCE,
        source_sha256=digest(ROOT/SOURCE), original_protocol_sha256=digest(STUDY/'study.json'),
        previous_limit=150, new_limit=500, tolerance_changes=False,
        reason='Three-branch classwise blend reached the original iteration limit after six completed fits.',
        preserves_completed_jobs=True, preserves_completed_hybrids=True, test_loaded=False)
    if ADAPTER.exists() and read(ADAPTER) != identity:
        raise ValueError('Adapter identity changed; create a new adoption revision')
    if not ADAPTER.exists():
        write(ADAPTER, identity)
        snapshot = STUDY/'source_snapshot'/SOURCE
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT/SOURCE).read_bytes())
    return identity


def main():
    identity = adoption()
    original_minimize = hybrid_blend.minimize
    hybrid_blend.minimize = lambda objective, initial, **kwargs: bounded_minimize(
        original_minimize, objective, initial, **kwargs)
    original_write = base.write

    def recorded_write(path, value):
        if digest(ROOT/SOURCE) != identity['source_sha256']:
            raise ValueError('Active adapter source changed')
        if isinstance(value, dict) and path.name in ('state.json', 'result.json'):
            value = dict(value, runtime_adapter=dict(path=ADAPTER.relative_to(ROOT).as_posix(),
                                                     sha256=digest(ADAPTER)))
        return original_write(path, value)

    base.write = recorded_write
    hybrid_evaluation.write = recorded_write
    base.MODULE = MODULE
    # Starting via the adapter also makes every subsequent worker use it.
    base.main()


if __name__ == '__main__':
    main()
