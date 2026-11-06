"""Persisted broad-profile proposals and bounded recovery contracts."""
from pathlib import Path

import optuna
import pytest

from framework.training.broad_live.search import configuration, load, proposal
from framework.training.broad_live.controller import finished_counts
from framework.training.live_study.protocol import atomic


def test_broad_proposal_resume_keeps_the_exact_pending_request(tmp_path, monkeypatch):
    from framework.training.broad_live import search
    atomic(tmp_path / 'study.json', dict(smoke=True, schema_fingerprint='252-schema', protocol_fingerprint='frozen'))
    monkeypatch.setattr(search, 'relative', lambda path: str(path))
    _, first, request = proposal(tmp_path, 'xgboost')
    _, again, recovered = proposal(tmp_path, 'xgboost')
    assert first.number == again.number == 0
    assert request == recovered
    assert len(load(tmp_path, 'xgboost').trials) == 1


def test_hem_search_is_adapted_to_broad_inputs_and_weighted_training():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))
    config = configuration(study.ask(), 'hem', smoke=True)
    assert config['pca_components'] in (16, 32, 64, 128)
    assert config['weight_strategy'] in ('none', 'sqrt_balanced', 'balanced')
    assert config['epochs'] == 2


def test_hybrid_global_reference_is_not_repeated_as_duplicate_trials():
    study = optuna.create_study()
    first = configuration(study.ask(), 'tabm_xgboost', smoke=True)
    second = configuration(study.ask(), 'tabm_xgboost', smoke=True)
    assert first == {'kind': 'global', 'penalty': None}
    assert second['kind'] == 'classwise' and second['penalty'] > 0


def test_terminal_trial_counts_do_not_count_running_work_as_finished():
    assert finished_counts(dict(complete=3, pruned=2, fail=1, running=1)) == 6
def test_running_trial_is_visible_in_heartbeat_before_worker_wait(tmp_path, monkeypatch):
    import time
    from types import SimpleNamespace
    import pytest
    from framework.training.broad_live import controller
    from framework.training.live_study.protocol import read
    state = {'deadline': time.time() + 60}
    monkeypatch.setattr(controller, 'counts', lambda *_: {'random_forest': dict(complete=0, pruned=0, fail=0, running=0)})
    monkeypatch.setattr(controller.search, 'proposal', lambda *_: (object(), SimpleNamespace(number=0), {}))
    def worker(study, state, *args):
        controller.pulse(study, state)
        assert read(study / 'state.json')['trial_counts']['random_forest']['running'] == 1
        raise controller.Halt('stopped')
    monkeypatch.setattr(controller, 'run_job', worker)
    with pytest.raises(controller.Halt):
        controller.optimize(tmp_path, state, ['random_forest'], 1)

