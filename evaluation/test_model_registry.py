"""Actual spawned workers exercise crash containment and recovery."""
import json
import sys

import pytest

from framework.core.contracts import CapabilityError
from framework.core.registry import ModelRegistry


def config():
    return {'schema_version': 1, 'request_timeout_seconds': 5,
            'models': {name: {'enabled': True, 'adapter': 'evaluation.modular_worker_fixtures:FakeModelAdapter',
                              'label': name, 'dependencies': []}
                       for name in ['random_forest', 'xgboost', 'kan', 'hem']},
            'plugins': {'explanation': True, 'calibration': False, 'conformal': False}}


def test_registry_import_and_metadata_do_not_import_models(tmp_path):
    before = set(sys.modules)
    with ModelRegistry(tmp_path, configuration=config()) as registry:
        assert len(registry.describe()) == 4
    assert not {'torch', 'xgboost', 'efficient_kan'} & (set(sys.modules) - before)


@pytest.mark.parametrize('value,expected', [(13, 'failed'), (14, 'failed'), (15, 'failed')])
def test_crash_hang_exception_isolated_and_recoverable(tmp_path, value, expected):
    with ModelRegistry(tmp_path, configuration=config()) as registry:
        assert registry.call('kan', 'predict', {'rows': [[1]]})[0]['label'] == 'benign'
        registry.timeout_seconds = 0.4
        with pytest.raises(CapabilityError) as failure:
            registry.call('kan', 'predict', {'rows': [[value]]})
        assert failure.value.model_id == 'kan' and failure.value.status == expected
        registry.timeout_seconds = 5
        assert registry.call('xgboost', 'predict', {'rows': [[1]]})[0]['label'] == 'benign'
        assert registry.call('kan', 'predict', {'rows': [[1]]})[0]['label'] == 'benign'


def test_missing_optional_explanation_does_not_disable_prediction(tmp_path):
    with ModelRegistry(tmp_path, configuration=config()) as registry:
        with pytest.raises(CapabilityError) as failure:
            registry.call('hem', 'explain', {'rows': [[1]]})
        assert failure.value.capability == 'explain'
        assert registry.call('hem', 'predict', {'rows': [[1]]})


def test_disabled_plugins_and_models_are_scoped(tmp_path):
    settings = config()
    settings['models']['kan']['enabled'] = False
    settings['plugins']['explanation'] = False
    with ModelRegistry(tmp_path, configuration=settings) as registry:
        assert next(x for x in registry.describe() if x['model_id'] == 'kan')['status'] == 'unavailable'
        with pytest.raises(CapabilityError):
            registry.call('kan', 'predict', {'rows': [[1]]})
        with pytest.raises(CapabilityError):
            registry.call('xgboost', 'explain', {'rows': [[1]]})
        assert registry.call('xgboost', 'predict', {'rows': [[1]]})
        assert registry.comparison()['rows']


def test_config_rejects_extra_family_and_nonfinite_timeout(tmp_path):
    settings = config()
    settings['models']['extra'] = {}
    with pytest.raises(ValueError):
        ModelRegistry(tmp_path, configuration=settings)


def test_close_during_queued_request_cannot_spawn_worker(tmp_path):
    registry = ModelRegistry(tmp_path, configuration=config())
    class ClosingLock:
        def acquire(self, timeout):
            registry._closed = True  # Close wins while this request waits.
            return True
        def release(self):
            pass
    registry._locks['kan'] = ClosingLock()
    try:
        with pytest.raises(CapabilityError, match='closed'):
            registry.call('kan', 'predict', {'rows': [[1]]})
        assert not registry._workers
    finally:
        registry._workers.clear()
        registry.close()


def test_catalog_survives_concurrently_closed_process(tmp_path):
    registry = ModelRegistry(tmp_path, configuration=config())
    class ClosedProcess:
        def is_alive(self):
            raise ValueError('process object is closed')
    registry._workers['kan'] = (ClosedProcess(), None)
    try:
        assert len(registry.describe()) == 4
    finally:
        registry._workers.clear()
        registry.close()
    settings = config()
    settings['request_timeout_seconds'] = float('nan')
    with pytest.raises(ValueError):
        ModelRegistry(tmp_path, configuration=settings)
