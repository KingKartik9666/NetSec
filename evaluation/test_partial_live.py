"""Window identity and input fidelity, without opening a capture device."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.test_partial_inputs import partial_detector, detector
from framework.app.server import create_app
from framework.data.feature_schema import PACKET_COLUMNS
from framework.data.live_monitor import LiveMonitor
from framework.detection.trained_xgboost import transform_csv


def test_multiclass_window_preserves_only_measured_counts():
    names = [*PACKET_COLUMNS, "log_unavailable"]
    seen = []
    def analyze(values, source):
        seen.append((values, source))
        return {"features": values, "prediction": {"label": "fixture"}, "source": source}
    model = SimpleNamespace(feature_names=names, partial_input_supported=True, analyze=analyze)
    monitor = LiveMonitor(detector=model)
    stats = {"seconds": 10, "packets": 10, "ack": 4, "fin": 0, "psh": 2, "rst": 0, "syn_all": 3, "urg": 0}
    result = monitor._predict_multiclass_window(stats, "window-id", "timestamp")
    assert seen[0][0] == [10, 4, 0, 2, 0, 3, 0, None]
    assert result["observation_id"] == "window-id" and result["observed_window_seconds"] == 10
    assert monitor._predict_multiclass_window({**stats, "seconds": 12}, "late", "timestamp") is None
    assert monitor._predict_multiclass_window({**stats, "packets": 0}, "empty", "timestamp") is None


def test_live_csv_requires_current_observation_and_reproduces_prediction(partial_detector):
    monitor = LiveMonitor(detector=partial_detector)
    report = partial_detector.analyze([0, None, None, 1, None], {"kind": "Fixture observation"})
    report.update(observation_id="abc", observed_at="timestamp", observed_window_seconds=10)
    monitor._data.update(running=True, mode="packets", multiclass_prediction=report)
    monitor._last_observation = time.monotonic()
    client = create_app(partial_detector, monitor=monitor).test_client()
    assert client.get('/api/live/features.csv').status_code == 400
    assert client.get('/api/live/features.csv?observation_id=old').status_code == 400
    exported = client.get('/api/live/features.csv?observation_id=abc')
    assert exported.status_code == 200
    X = transform_csv(exported.data, partial_detector.bundle)
    np.testing.assert_equal(X[0], [0, np.nan, np.nan, 1, np.nan])
    assert partial_detector.predict_batch(X)[0] == report['prediction']
    monitor.stop()
    assert monitor.snapshot()['multiclass_prediction'] is None
    assert client.get('/api/live/features.csv?observation_id=abc').status_code == 400
    assert client.post('/api/xai', json={'kind': 'live', 'observation_id': 'abc'}).status_code == 400


@pytest.mark.parametrize('condition', ['stale', 'error', 'stopped'])
def test_no_stale_multiclass_verdict_on_lifecycle_changes(condition):
    monitor = LiveMonitor()
    monitor._data.update(running=True, mode='packets', multiclass_prediction={'observation_id':'old'})
    if condition == 'stale':
        monitor._last_observation = time.monotonic() - 30
    elif condition == 'error':
        monitor._data['error'] = 'capture failed'
    else:
        monitor._data['running'] = False
    assert monitor.snapshot()['multiclass_prediction'] is None


def test_binary_comparison_is_never_used_as_multiclass_evidence(partial_detector):
    monitor = LiveMonitor(detector=partial_detector)
    monitor._data.update(running=True, mode='packets', model_prediction={'score': .9, 'observation_id': 'binary-only'})
    client = create_app(partial_detector, monitor=monitor).test_client()
    for route in ['/api/xai', '/api/assistant/explain']:
        response = client.post(route, json={'kind': 'live', 'observation_id': 'binary-only'})
        assert response.status_code == 400 and 'multiclass' in response.json['error']
