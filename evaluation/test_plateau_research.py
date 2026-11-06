"""Research integrity boundaries for the isolated representation pilot."""
import numpy as np
import pytest


def test_hashing_preserves_missingness_and_canonicalizes_signed_zero():
    from framework.training.plateau.data import row_hashes
    rows = np.array([[np.nan, -0.], [np.nan, 0.], [0., 0.]], dtype='float32')
    hashes = row_hashes(rows)
    assert hashes[0] == hashes[1]
    assert hashes[0] != hashes[2]
    assert np.isnan(rows[0, 0])


def test_duplicate_selection_keeps_constants_and_distinct_missingness():
    from framework.training.plateau.data import duplicate_columns
    values = np.array([[1, 1, 1, 0, np.nan], [2, 2, np.nan, 0, np.nan],
                       [3, 3, 3, 0, np.nan]], dtype='float32')
    keep, groups = duplicate_columns(values)
    assert keep == [0, 2, 3, 4]
    assert groups == [[0, 1]]


def test_projection_collisions_are_purged_for_every_arm():
    from framework.training.plateau.data import common_evaluation_mask
    fit = np.array([[1, 10], [2, 20]], dtype='float32')
    evaluate = np.array([[1, 11], [3, 20], [4, 40]], dtype='float32')
    keep, report = common_evaluation_mask(fit, evaluate,
                                          {'all': [0, 1], 'a': [0], 'b': [1]})
    assert keep.tolist() == [False, False, True]
    assert report['purged_union'] == 2
    assert report['collisions_by_arm'] == {'all': 0, 'a': 1, 'b': 1}


def test_temporal_check_rejects_touching_windows():
    from framework.training.plateau.data import check_roles
    with pytest.raises(ValueError, match='Temporal'):
        check_roles(np.array([0, 0]), np.array([0, 10]), np.array([10, 20]),
                    np.array([0]), np.array([1]))


def test_temporal_check_rejects_same_row_in_both_roles():
    from framework.training.plateau.data import check_roles
    with pytest.raises(ValueError, match='overlap'):
        check_roles(np.array([0, 0]), np.array([0, 11]), np.array([10, 20]),
                    np.array([0]), np.array([0]))


def test_score_keeps_declared_classes_and_reports_false_alarms():
    from framework.training.plateau.metrics import measure
    p = np.array([[.1, .8, .1], [.1, .8, .1]])
    result = measure(np.array([0, 1]), p, ['benign', 'attack_a', 'attack_b'], 0)
    assert result['accuracy'] == .5
    assert result['macro_f1_declared'] == pytest.approx(2/9)
    assert result['macro_f1_supported'] == pytest.approx(1/3)
    assert result['false_positive_rate'] == 1.
    assert result['supported_classes'] == 2


def test_source_pin_change_fails_closed(tmp_path):
    from framework.training.plateau.data import sha256, verify_pins
    source = tmp_path/'source.py'
    source.write_text('original')
    pins = {'source.py': sha256(source)}
    verify_pins(pins, tmp_path)
    source.write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        verify_pins(pins, tmp_path)


def test_train_only_loader_rejects_validation_and_test_names(tmp_path):
    from framework.training.plateau.data import load_training_array
    for name in ('X_validation.npy', 'X_test.npy', '../X_train.npy'):
        with pytest.raises(ValueError, match='allowlist'):
            load_training_array(tmp_path, name)


def test_real_cpu_fit_saves_equal_budget_predictions_and_replays(tmp_path):
    from framework.training.plateau.runner import train_one
    rng = np.random.default_rng(7)
    X = rng.normal(size=(90, 4)).astype('float32')
    y = np.tile(np.arange(3), 30)
    result = train_one(tmp_path, X[:60], y[:60], X[60:], y[60:],
                       ['a', 'b', 'c', 'd'], ['benign', 'a', 'b'], 0,
                       seed=42, rounds=4, checkpoints=[2, 4], threads=1,
                       stop_path=tmp_path/'STOP', deadline=float('inf'))
    assert result['status'] == 'completed'
    assert result['rounds'] == 4
    assert result['cpu_reload_verified'] is True
    assert set(result['metrics_by_round']) == {'2', '4'}
    assert np.load(tmp_path/'probabilities.npy').shape == (30, 3)


def test_stop_never_produces_a_completed_fit(tmp_path):
    from framework.training.plateau.runner import train_one
    (tmp_path/'STOP').touch()
    X = np.array([[0.], [1.], [2.], [3.]], dtype='float32')
    with pytest.raises(InterruptedError):
        train_one(tmp_path, X, np.array([0, 1, 0, 1]), X,
                  np.array([0, 1, 0, 1]), ['a'], ['benign', 'a'], 0,
                  seed=42, rounds=4, checkpoints=[2, 4], threads=1,
                  stop_path=tmp_path/'STOP', deadline=float('inf'))
    assert not (tmp_path/'result.json').exists()


def test_resume_rejects_unbound_completed_result(tmp_path):
    from framework.training.plateau.data import write_json
    from framework.training.plateau.runner import completed_results
    protocol = dict(jobs=[dict(arm='all_features', seed=42)], rounds=4,
                    feature_names=['a'], arms={'all_features': [0]})
    folder = tmp_path/'runs/all_features_s42/attempt_0000'
    write_json(folder/'result.json', dict(status='completed', seed=42, rounds=4,
        feature_names=['a'], artifact_hashes={}, metrics_by_round={'4': {'accuracy': 1.}}))
    with pytest.raises(ValueError, match='binding|artifact'):
        completed_results(tmp_path, protocol)


def test_stop_during_training_keeps_attempt_incomplete(tmp_path):
    from framework.training.plateau.runner import train_one
    X = np.arange(20, dtype='float32').reshape(-1, 1)
    y = np.tile([0, 1], 10)

    def stop_after_first_round(*_):
        (tmp_path/'STOP').touch()

    with pytest.raises(InterruptedError):
        train_one(tmp_path, X, y, X, y, ['a'], ['benign', 'attack'], 0,
                  seed=42, rounds=8, checkpoints=[4, 8], threads=1,
                  stop_path=tmp_path/'STOP', deadline=float('inf'),
                  heartbeat=stop_after_first_round)
    assert not (tmp_path/'result.json').exists()


def test_expired_budget_during_scoring_cannot_publish_completion(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from framework.training.plateau import runner
    clock = [0.]
    original_measure = runner.measure

    def expire_during_measure(*args):
        result = original_measure(*args)
        clock[0] = 100.
        return result

    monkeypatch.setattr(runner, 'time', SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(runner, 'measure', expire_during_measure)
    X = np.arange(20, dtype='float32').reshape(-1, 1)
    y = np.tile([0, 1], 10)
    with pytest.raises(InterruptedError):
        runner.train_one(tmp_path, X, y, X, y, ['a'], ['benign', 'attack'], 0,
                         seed=42, rounds=2, checkpoints=[1, 2], threads=1,
                         stop_path=tmp_path/'STOP', deadline=10.)
    assert not (tmp_path/'result.json').exists()


def test_unsealed_bound_result_is_an_incomplete_attempt(tmp_path):
    from framework.training.plateau.data import write_json, sha256
    from framework.training.plateau.runner import completed_results
    protocol = dict(jobs=[dict(arm='all_features', seed=42)], rounds=4,
                    feature_names=['a'], arms={'all_features': [0]})
    write_json(tmp_path/'protocol.json', protocol)
    folder = tmp_path/'runs/all_features_s42/attempt_0000'
    write_json(folder/'request.json', dict(arm='all_features', seed=42,
        protocol_sha256=sha256(tmp_path/'protocol.json')))
    write_json(folder/'result.json', dict(status='completed', seed=42, rounds=4,
        feature_names=['a'], artifact_hashes={}, metrics_by_round={'4': {'accuracy': 1.}}))
    assert completed_results(tmp_path, protocol) == {}
