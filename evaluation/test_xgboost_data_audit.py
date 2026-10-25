"""Data-experiment boundaries: parsing, train-only vocabulary and true durations."""
import numpy as np
import pytest
import json

from experiment.audit_xgboost_data import parse_ports, port_vector, choose_vocabulary, duration_features


def test_ports_are_measurements_not_executable_text():
    assert parse_ports("['80', '443', '80']") == (80, 443)
    assert parse_ports('[0, 65535]') == (0, 65535)
    assert parse_ports('[]') == ()
    for text in ("__import__('os').getcwd()", '[65536]', '[-1]', '[true]', '[1.5]', "['tcp']"):
        with pytest.raises(ValueError):
            parse_ports(text)


def test_unseen_ports_do_not_expand_training_vocabulary():
    counts = np.zeros(65536, dtype=np.int64)
    counts[[80, 443, 22]] = [100, 10, 10]
    vocabulary = choose_vocabulary(counts, maximum=2)
    assert vocabulary == [80, 22]
    np.testing.assert_equal(port_vector((80, 65000), vocabulary), [1, 0, 1])
    np.testing.assert_equal(port_vector((), vocabulary), [0, 0, 0])


def test_rates_use_elapsed_time_without_absolute_time_identity():
    counts = np.array([[100, 60, 40, 10], [0, 0, 0, 0]], dtype=np.float32)
    start = np.array([10, 30], dtype=np.int64) * 10**9
    end = np.array([20, 30], dtype=np.int64) * 10**9
    a = duration_features(counts, start, end)
    np.testing.assert_allclose(a[0], [10, 10, 6, 4, 1])
    assert np.isnan(a[1]).all()
    np.testing.assert_equal(a, duration_features(counts, start + 10**15, end + 10**15))


def test_cpu_objective_matches_existing_loss_definition():
    from experiment.audit_xgboost_data import cpu_smoothed_objective
    from framework.training.partial_objectives import smoothed_derivatives
    margins = np.random.default_rng(42).normal(size=(10, 5)).astype(np.float32)
    labels = np.arange(10) % 5
    g, h = cpu_smoothed_objective(labels)(margins, None)
    expected, _, bound = smoothed_derivatives(margins, labels)
    np.testing.assert_allclose(g, expected, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(h, bound, atol=1e-7, rtol=1e-6)


def test_cpu_trial_keeps_validation_intact_and_reloads(tmp_path, monkeypatch):
    import experiment.audit_xgboost_data as audit
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    manifest = {'feature_names': ['count_a', 'count_b'], 'classes': ['benign', 'attack'], 'benign_index': 0}
    (prepared / 'manifest.json').write_text(json.dumps(manifest))
    (prepared / 'feature_transform.json').write_text(json.dumps({'extra_feature_names': ['port_present']}))
    X = np.array([[0, 0], [1, 0], [3, 2], [0, 0], [2, 1], [6, 3]], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    for split in ('train', 'validation'):
        for key, value in {'X': X, 'y': y, 'row_ids': np.arange(6), 'window_start': np.arange(6),
                           'window_end': np.arange(6)+1, 'feature_hash': np.arange(6),
                           'extra': np.ones((6, 1), dtype=np.float32)}.items():
            np.save(prepared / f'{key}_{split}.npy', value)
    np.save(prepared / 'training_keep.npy', np.any(X != 0, axis=1))
    fake_baseline = tmp_path / 'baseline'
    fake_baseline.mkdir()
    (fake_baseline / 'result.json').write_text(json.dumps({'config': {
        'max_depth': 2, 'learning_rate': .08, 'max_bin': 16, 'subsample': .8,
        'colsample_bytree': 1., 'colsample_bynode': .9, 'min_child_weight': 1.,
        'reg_lambda': 3., 'reg_alpha': .05, 'gamma': 0.}}))
    monkeypatch.setattr(audit, 'BASELINE', fake_baseline)
    load = np.load
    def guarded_load(path, *args, **kwargs):
        assert not str(path).endswith('_test.npy'), 'Held-out test access'
        return load(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', guarded_load)
    audit.fit(tmp_path, 'cleaned_enriched', rounds=2, threads=1)
    output = tmp_path / 'runs/cleaned_enriched'
    request = json.loads((output / 'request.json').read_text())
    result = json.loads((output / 'result.json').read_text())
    assert request['training_rows'] == 4
    assert result['metrics']['rows'] == 6
    assert result['checkpoint_reload_verified']
    assert result['device'] == 'cpu'
    assert len(result['feature_names']) == 3
