"""Research boundaries: feature isolation, purged cross-fitting and replay."""
import json

import numpy as np
import pytest

from experiment import xgboost_followup as followup
from experiment.report_xgboost_followup import calibrate, fit_temperature, summarize_arms
from experiment.run_xgboost_followup import verify_result


def test_feature_ablations_use_only_their_named_measurements():
    manifest = {'feature_names': ['count', 'protocol']}
    transform = {'extra_feature_names': ['network_ports_src__present_80', *followup.DURATION_NAMES]}
    inputs = {'X': np.arange(12).reshape(6, 2), 'extra': np.arange(36).reshape(6, 6)+100}
    port_indexes, port_names = followup.feature_indexes('ports', manifest, transform)
    rate_indexes, rate_names = followup.feature_indexes('rates', manifest, transform)
    assert port_names == ['count', 'protocol', 'network_ports_src__present_80']
    assert rate_names == ['count', 'protocol', *followup.DURATION_NAMES]
    np.testing.assert_equal(followup.matrix(inputs, np.array([1, 4]), port_indexes),
                            np.column_stack([inputs['X'][[1, 4]], inputs['extra'][[1, 4], 0]]))
    assert set(port_indexes[2:]).isdisjoint(rate_indexes[2:])


def test_cross_fit_purges_touching_windows_and_copies_with_full_held_coverage():
    y = np.repeat([0, 1], 30)
    start = np.tile(np.arange(30)*3, 2)
    end = start + 4
    hashes = np.arange(60)
    hashes[17] = hashes[1]
    blocks = followup.temporal_folds(y, start, end, hashes)
    for fitting, held in blocks:
        assert not np.intersect1d(fitting, held).size
        assert not np.intersect1d(hashes[fitting], hashes[held]).size
        assert set(y[fitting]) == {0, 1}
        for label in (0, 1):
            a, b = fitting[y[fitting] == label], held[y[held] == label]
            assert np.all((end[a] < start[b].min()) | (start[a] > end[b].max()))
    np.testing.assert_array_equal(np.sort(np.concatenate([b for _, b in blocks])), np.arange(60))


def test_temperature_fits_only_supplied_rows_and_preserves_decisions():
    probability = np.tile([[.999, .001], [.001, .999]], (50, 1))
    truth = np.tile([0, 1], 50)
    truth[::5] = 1-truth[::5]
    fitted = fit_temperature(probability, truth)
    assert fitted['temperature'] > 1
    assert fitted['oof_nll_after'] < fitted['oof_nll_before']
    np.testing.assert_array_equal(calibrate(probability, fitted['temperature']).argmax(1), probability.argmax(1))
    np.testing.assert_allclose(calibrate(probability, 1), probability, atol=1e-12)
    with pytest.raises(ValueError):
        calibrate(probability, -1)


@pytest.fixture
def small_source(tmp_path):
    source = tmp_path / 'source'
    prepared = source / 'prepared'
    prepared.mkdir(parents=True)
    manifest = {'feature_names': ['count', 'protocol'], 'classes': ['benign', 'attack'], 'benign_index': 0}
    (prepared / 'manifest.json').write_text(json.dumps(manifest))
    (prepared / 'feature_transform.json').write_text(json.dumps(
        {'extra_feature_names': ['network_ports_src__present_80', *followup.DURATION_NAMES]}))
    rng = np.random.default_rng(19)
    for split, n, offset in [('train', 120, 0), ('validation', 20, 1000)]:
        X = rng.normal(size=(n, 2)).astype(np.float32)
        y = np.arange(n) % 2
        X[:, 0] += y*2
        if split == 'train':
            X[:10] = 0
            np.save(prepared / 'training_keep.npy', np.any(X != 0, axis=1))
        values = dict(X=X, y=y, row_ids=np.arange(n)+offset,
                      window_start=np.arange(n)*10, window_end=np.arange(n)*10+5,
                      feature_hash=np.arange(n)+offset, extra=rng.random((n, 6)).astype(np.float32))
        for key, value in values.items():
            np.save(prepared / f'{key}_{split}.npy', value)
    return source


def tiny_request(seed=43):
    request = followup.recipe('cleaned_enriched', seed)
    request.update(rounds=3, threads=1)
    request['parameters'].update(max_depth=2, max_bin=16)
    return request


@pytest.mark.parametrize('seed', [43, 44])
def test_real_cpu_fit_retains_validation_seed_and_reload(small_source, tmp_path, monkeypatch, seed):
    original_load = np.load
    def guarded(path, *args, **kwargs):
        assert '_test.' not in str(path), 'Locked-test access'
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', guarded)
    output = tmp_path / f'fit{seed}'
    result = followup.fit(small_source, output, tiny_request(seed))
    request = json.loads((output / 'request.json').read_text())
    assert request['parameters']['seed'] == seed
    assert request['training_rows'] == 110
    assert result['metrics']['rows'] == 20
    assert result['checkpoint_reload_verified']
    np.testing.assert_array_equal(np.load(output / 'scoring_row_ids.npy'), np.arange(20)+1000)
    verify_result(output)
    with (output / 'probabilities.npy').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='Changed result artifact'):
        verify_result(output)


def test_calibration_fit_never_opens_validation_or_test_arrays(small_source, tmp_path, monkeypatch):
    original_load = np.load
    def guarded(path, *args, **kwargs):
        assert '_validation.' not in str(path) and '_test.' not in str(path)
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', guarded)
    output = tmp_path / 'fold'
    followup.fit(small_source, output, dict(tiny_request(42), fold=1))
    fitting = np.load(output / 'fit_row_ids.npy')
    held = np.load(output / 'scoring_row_ids.npy')
    assert not np.intersect1d(fitting, held).size
    assert held.max() < 120
    assert json.loads((output / 'request.json').read_text())['scoring_split'] == 'train'


def test_selection_rejects_accuracy_gain_with_recall_regression():
    records = {}
    for arm in followup.ARMS:
        for seed in followup.SEEDS:
            metrics = dict(accuracy=.79, macro_precision=.76, macro_recall=.67, macro_f1=.68,
                           false_positive_rate=.001, attack_missed_as_benign_rate=.04)
            if arm == 'ports':
                metrics.update(accuracy=.90, macro_recall=.50)
            if arm == 'cleaned_enriched':
                metrics.update(accuracy=.81, macro_recall=.75, macro_f1=.77)
            records[f'{arm}_s{seed}_base'] = dict(metrics=metrics, feature_names=['a', 'b'])
    summary, chosen = summarize_arms(records)
    assert chosen == 'cleaned_enriched'
    assert not summary['ports']['guards_pass']


def test_initialization_pins_portable_input_paths_and_rejects_mutation(small_source, tmp_path, monkeypatch):
    import experiment.run_xgboost_followup as runner
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    for name in ('experiment/xgboost_followup.py', 'experiment/run_xgboost_followup.py',
                 'experiment/report_xgboost_followup.py', 'experiment/audit_xgboost_data.py',
                 'framework/training/tracking.py', 'docs/XGBOOST_FOLLOWUP_20261013.md'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic fixture for source identity')
    prepared = small_source / 'prepared'
    manifest = json.loads((prepared / 'manifest.json').read_text())
    manifest['input_hashes'] = {f'X_{split}.npy': followup.digest(prepared / f'X_{split}.npy')
                              for split in ('train', 'validation')}
    (prepared / 'manifest.json').write_text(json.dumps(manifest))
    output = tmp_path / 'study'
    output.mkdir()
    runner.initialize(output, small_source)
    pinned = json.loads((output / 'protocol.json').read_text())
    assert 'prepared/X_train.npy' in pinned['input_hashes']
    assert all('\\' not in key for key in pinned['input_hashes'])
    runner.initialize(output, small_source)
    with (prepared / 'X_train.npy').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='Original prepared input changed'):
        runner.initialize(output, small_source)
