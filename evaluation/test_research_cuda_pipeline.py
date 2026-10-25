"""Small synthetic execution checks, never reported as research accuracy."""
import json

import numpy as np
import pytest


def require_cuda():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA research execution check')


def test_actual_conditional_ctgan_preserves_context_and_real_prefix(tmp_path):
    require_cuda()
    pytest.importorskip('ctgan')
    from framework.training.research_augmentation import augment_training
    rng = np.random.default_rng(42)
    names = ['network_packets_all_count', 'network_tcp-flags-syn_count',
             'network_packet-size_min', 'network_packet-size_avg', 'network_packet-size_max',
             'network_protocols_all__present_tcp', 'network_interval-packets', 'unobserved']
    X = np.column_stack((rng.integers(100, 500, 192), rng.integers(1, 30, 192),
                         np.full(192, 10), np.full(192, 20), np.full(192, 30),
                         np.ones(192), np.full(192, 10), np.full(192, np.nan))).astype(np.float32)
    y = np.array([0]*128 + [1]*64, dtype=np.int32)
    A, labels, evidence = augment_training(X, y, np.arange(192), names,
                {'method': 'ctgan', 'ctgan_epochs': 1}, 42, tmp_path)
    np.testing.assert_array_equal(A[:192], X)
    np.testing.assert_array_equal(labels[:192], y)
    assert len(A) > 192
    assert evidence['quality']['gpu_training_verified'] is True
    assert evidence['quality']['backend'] == 'ctgan.CTGAN'
    assert evidence['audit']['passed'] is True
    assert (tmp_path / 'ctgan.pkl').is_file()
    assert evidence['validation_used'] is False
    assert np.isnan(A[192:, -1]).all()
    assert (A[192:, 5] == 1).all() and (A[192:, 6] == 10).all()
    assert (A[192:, 1] <= A[192:, 0]).all()


@pytest.mark.parametrize('family', ['xgboost', 'kan', 'hem'])
def test_same_family_cuda_ensemble_cpu_reload(family, tmp_path, monkeypatch):
    require_cuda()
    from experiment import train_csv_research as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    rng = np.random.default_rng(42)
    y = np.tile(np.arange(3), 30)
    X = rng.normal(size=(90, 4)).astype(np.float32)
    X[:, 0] += 4*y
    manifest = {'feature_names': ['a', 'b', 'c', 'd'], 'classes': ['a', 'b', 'c'], 'benign_index': 0}
    request = {'model_id': family, 'seed': 42, 'pipeline': {'ensemble': 'seed_average'},
               'config': {'max_depth': 2, 'num_boost_round': 2, 'epochs': 1,
                          'hidden': [8], 'binary_hidden': [8], 'pca_components': 3,
                          'batch_size': 32, 'knn_k': 1, 'knn_max_rows': 60}}
    result = module.fit_ensemble(request, manifest, X[:60], y[:60], np.arange(60), X[60:], y[60:], tmp_path)
    cpu, probability = module.reload_research({'model_id': family, 'checkpoint_path': result['checkpoint_path']}, X[60:])
    np.testing.assert_array_equal(cpu, result['predictions'])
    assert result['checkpoint_reload_verified'] is True
    assert [m['seed'] for m in result['members']] == [42, 43, 44]
    assert all(m['details']['original_training_rows'] == 60 for m in result['members'])
    assert (probability is None) == (family == 'hem')
    assert json.loads((tmp_path / 'ensemble.json').read_text())['kind'] == 'same_family_seed_ensemble'
