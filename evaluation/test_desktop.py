"""Desktop authentication, process ownership and truthful live reporting."""
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

from evaluation.test_ids_app import detector
from framework.app.server import create_app


def test_desktop_session_protects_every_route(detector):
    client = create_app(detector, session_token="a" * 64).test_client()
    assert client.get("/").status_code == 403
    assert client.get("/api/live/status").status_code == 403
    assert client.post("/api/live/stop", json={}).status_code == 403
    headers = {"X-IMDAI-Session": "a" * 64}
    assert client.get("/api/status", headers=headers).status_code == 200
    assert client.get("/api/live/status", headers=headers).json["running"] is False
    assert client.post("/api/live/stop", json={}, headers={**headers, "Origin": "https://foreign.example"}).status_code == 403


def test_live_report_requires_observations_and_distinguishes_rules(detector):
    from framework.data.live_monitor import LiveMonitor
    monitor = LiveMonitor()
    client = create_app(detector, monitor=monitor).test_client()
    assert client.get("/api/live/report").status_code == 400
    # Explicit software fixture, not real traffic or a model-accuracy result.
    monitor._data.update(mode="connections", interface="test adapter", state="monitoring", running=True)
    monitor._data["metrics"]["packets_observed"] = 25
    report = client.get("/api/live/report").json
    assert report["source"]["kind"] == "Local adapter observation"
    assert report["reasoning"]["llm_executed"] is False
    assert report["model_prediction"] is None
    assert report["source"]["packet_headers_captured"] is False


def test_stale_live_xai_cannot_be_presented_as_the_current_observation(detector):
    from framework.data.live_monitor import LiveMonitor
    monitor = LiveMonitor()
    # Explicit fixture for lifecycle semantics, not a real packet observation.
    monitor._data.update(running=True, mode="packets", state="monitoring",
                         multiclass_prediction={"prediction": {"score": .7}, "observation_id": "new-window"})
    client = create_app(detector, monitor=monitor).test_client()
    for route in ["/api/xai", "/api/assistant/explain"]:
        response = client.post(route, json={"kind": "live", "observation_id": "old-window"})
        assert response.status_code == 400
        assert "observation changed" in response.json["error"]


def test_desktop_backend_exits_when_parent_pipe_closes(detector):
    token = "b" * 64
    child = subprocess.Popen([sys.executable, "-B", "-u", "-m", "framework.app.desktop_backend",
                              "--bundle", str(detector.bundle_path)],
                             cwd=Path(__file__).resolve().parents[1],
                             env={**os.environ, "IMDAI_SESSION_TOKEN": token},
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        line = child.stdout.readline()
        assert line, child.stderr.read()
        ready = json.loads(line)
        assert ready["type"] == "ready" and ready["port"] > 0
        url = f"http://127.0.0.1:{ready['port']}/api/live/status"
        with pytest.raises(urllib.error.HTTPError) as rejected:
            urllib.request.urlopen(url, timeout=3)
        assert rejected.value.code == 403
        request = urllib.request.Request(url, headers={"X-IMDAI-Session": token})
        with urllib.request.urlopen(request, timeout=3) as response:
            assert json.load(response)["running"] is False
        child.stdin.close()
        assert child.wait(timeout=10) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
