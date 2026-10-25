"""Optional services use independent processes and do not load on shutdown."""
import pytest

from framework.core.capabilities import IsolatedLanguageService, IsolatedMonitorService, _IsolatedService
from framework.core.contracts import CapabilityError
from framework.core.registry import ModelRegistry
from evaluation.test_model_registry import config


def test_disabled_language_does_not_load_or_affect_models(tmp_path):
    settings = config()
    settings['plugins']['language'] = False
    with ModelRegistry(tmp_path, configuration=settings) as registry:
        language = IsolatedLanguageService(registry)
        with pytest.raises(CapabilityError, match='disabled'):
            language.status()
        assert language._process is None
        assert registry.call('kan', 'predict', {'rows': [[1]]})
        language.close()


def test_stopping_unused_monitor_does_not_start_dependencies(tmp_path):
    with ModelRegistry(tmp_path, configuration=config()) as registry:
        monitor = IsolatedMonitorService(registry)
        assert monitor.stop()['running'] is False
        assert monitor._process is None
        monitor.close()


def test_optional_worker_failure_leaves_model_prediction_available(tmp_path):
    with ModelRegistry(tmp_path, configuration=config()) as registry:
        broken = _IsolatedService(registry, 'invalid_test_capability')
        with pytest.raises(CapabilityError, match='Unknown optional service'):
            broken._call('status')
        assert registry.call('hem', 'predict', {'rows': [[1]]})
        broken.close()
