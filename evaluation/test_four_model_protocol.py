"""The four-family study must retain one set of observations, never open tests."""
import numpy as np
import pytest

from experiment.train_four_models import verify_common_partitions, study_configurations


def test_row_identity_and_label_order_required(tmp_path):
    for name, ids in [('numeric', [3, 5]), ('protocol', [3, 5])]:
        p = tmp_path / name
        p.mkdir()
        for split in ['train', 'validation']:
            np.save(p / f'row_ids_{split}.npy', ids if split == 'train' else [7, 9])
            np.save(p / f'y_{split}.npy', [0, 1])
    assert verify_common_partitions(tmp_path / 'numeric', tmp_path / 'protocol')['test_loaded'] is False
    np.save(tmp_path / 'protocol/row_ids_validation.npy', [5, 3])
    with pytest.raises(ValueError, match='row'):
        verify_common_partitions(tmp_path / 'numeric', tmp_path / 'protocol')


def test_ablation_changes_only_one_parameter_and_rf_is_fixed():
    configs = study_configurations({'xgboost': {'reg_lambda': 1}, 'kan': {}, 'hem': {}, 'random_forest': {'max_depth': 6}})
    for family in ['xgboost', 'kan', 'hem']:
        original, enhanced = [x['config'] for x in configs if x['model_id'] == family]
        assert len([key for key in set(original) | set(enhanced) if original.get(key) != enhanced.get(key)]) == 1
    assert len([x for x in configs if x['model_id'] == 'random_forest']) == 1
