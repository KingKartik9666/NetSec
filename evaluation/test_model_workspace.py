"""Four-family HTTP contracts and deliberately unavailable optional capabilities.

The lightweight registry double replaces expensive/native workers only. Flask,
request validation, model routing, authentication and monitor behavior stay real.
"""
from __future__ import annotations

import builtins
import io
import time
from types import SimpleNamespace

import pytest

from framework.app.server import create_app
from evaluation.test_packet_model import packet_model, counts
from evaluation.test_ids_app import detector


class RegistryFixture:
    """Complete small transport fixture; not a research detector or score source."""

    ids = ("random_forest", "xgboost", "kan", "hem")

    def __init__(self):
        self.broken = set()
        self.explanation_broken = set()
        self.received = {}

    def describe(self):
        return [{"model_id": name, "name": name, "status": "ready",
                 "capabilities": {"predict": True, "explain": name != "hem"}}
                for name in self.ids]

    def call(self, model_id, operation, payload):
        if model_id not in self.ids:
            raise ValueError("Unknown model family")
        if model_id in self.broken:
            raise RuntimeError("Injected unavailable checkpoint")
        if operation == "metadata":
            return {"model_id": model_id, "name": model_id, "version": "fixture-v1",
                    "feature_names": ["network_b", "network_a"] if model_id == "kan" else ["network_a", "network_b"],
                    "classes": ["benign", "attack"], "capabilities": {"predict": True, "explain": model_id != "hem"},
                    "dataset": "Synthetic transport test fixture", "metrics": None}
        if operation == "snapshot":
            return {"feature_names": ["network_a", "network_b"], "classes": ["benign", "attack"],
                    "samples": [{"id": 0, "label": "benign", "source_row": 17, "features": [2, 7]}]}
        if operation == "predict":
            self.received[model_id] = payload["rows"]
            return [{"label": "benign", "score": .6, "is_attack": False,
                     "top_classes": [{"label": "benign", "score": .6}], "novelty": "not evaluated"}
                    for _ in payload["rows"]]
        if operation == "explain":
            if model_id in self.explanation_broken:
                raise ImportError("Injected unavailable explanation plugin")
            return {"method": "Transport test fixture", "important_features": []}
        raise ValueError("Unknown fixture operation")

    def comparison(self):
        return {"models": [{"model_id": name, "metrics": None, "status": "unavailable",
                            "reason": "No comparable stored metrics"} for name in self.ids],
                "comparable": False, "reason": "Transport fixture has no research metrics"}

    def close(self):
        pass


@pytest.fixture
def workspace():
    registry = RegistryFixture()
    app = create_app(registry=registry)
    return app.test_client(), registry


def test_four_model_pages_boot_when_the_primary_model_is_broken(workspace):
    client, registry = workspace
    registry.broken.add("xgboost")
    for route in ("/", "/models", "/live", "/csv", "/samples"):
        assert client.get(route).status_code == 200
    response = client.get("/api/models")
    assert response.status_code == 200
    assert {entry["model_id"] for entry in response.json["models"]} == set(registry.ids)
    assert client.post("/api/models/random_forest/predict", json={"rows": [[1, 2]]}).status_code == 200
    assert client.post("/api/models/xgboost/predict", json={"rows": [[1, 2]]}).status_code >= 400


def test_failed_model_is_scoped_and_other_prediction_routes_remain_available(workspace):
    client, registry = workspace
    registry.broken.add("kan")
    result = client.post("/api/models/kan/predict", json={"rows": [[1, 2]]})
    assert result.status_code >= 400
    assert result.json["model_id"] == "kan"
    assert result.json["status"] in {"failed", "unavailable", "incompatible"}
    assert result.json["error"]
    good = client.post("/api/models/random_forest/predict", json={"rows": [[1, 2]]})
    assert good.status_code == 200
    assert good.json["predictions"][0]["label"] == "benign"
    assert client.get("/api/models").status_code == 200


def test_explanation_failure_does_not_disable_prediction(workspace):
    client, registry = workspace
    registry.explanation_broken.add("xgboost")
    assert client.post("/api/models/xgboost/explain", json={"rows": [[1, 2]]}).status_code >= 400
    response = client.post("/api/models/xgboost/predict", json={"rows": [[1, 2]]})
    assert response.status_code == 200 and response.json["predictions"]


def test_common_sample_uses_feature_names_instead_of_assuming_model_order(workspace):
    client, registry = workspace
    result = client.post("/api/models/kan/predict", json={"sample_id": 0})
    assert result.status_code == 200
    assert registry.received["kan"] == [[7, 2]]
    assert result.json["source"]["source_row"] == 17


@pytest.mark.parametrize("payload", [{"rows": []}, {"rows": [[True, 2]]}, {"rows": [[float("inf"), 2]]},
                                     {"rows": [[None, None]]}, {"sample_id": True}, {"rows": "bad"}])
def test_invalid_model_request_returns_scoped_validation_error(workspace, payload):
    client, _ = workspace
    response = client.post("/api/models/random_forest/predict", json=payload)
    assert response.status_code == 400


def test_missing_metrics_do_not_break_comparison(workspace):
    client, _ = workspace
    response = client.get("/api/models/comparison")
    assert response.status_code == 200
    assert response.json["comparable"] is False
    assert all(entry["metrics"] is None for entry in response.json["models"])


def test_model_workspace_keeps_authentication_and_security_headers():
    token = "e" * 64
    client = create_app(registry=RegistryFixture(), session_token=token).test_client()
    for path in ("/models", "/api/models", "/api/models/comparison"):
        assert client.get(path).status_code == 403
        response = client.get(path, headers={"X-IMDAI-Session": token})
        assert response.status_code == 200
        assert "script-src 'self'" in response.headers["Content-Security-Policy"]
    response = client.post("/api/models/random_forest/predict", json={"rows": [[1, 2]]},
                           headers={"X-IMDAI-Session": token, "Origin": "https://outside.example"})
    assert response.status_code == 403


def test_deleted_monitor_and_language_modules_do_not_prevent_boot(monkeypatch):
    original = builtins.__import__
    def missing_optional(name, *args, **kwargs):
        if name in {"framework.data.live_monitor", "framework.llm.local_model"}:
            raise ImportError("Injected optional capability removal")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing_optional)
    client = create_app(registry=RegistryFixture()).test_client()
    assert client.get("/").status_code == 200
    assert client.get("/models").status_code == 200
    assert client.get("/api/assistant/status").json["available"] is False
    assert client.get("/api/live/capabilities").json["connections_available"] is False
    assert client.post("/api/models/random_forest/predict", json={"rows": [[1, 2]]}).status_code == 200


def test_multiclass_window_failure_returns_unavailable_without_stopping_observation():
    from framework.data.live_monitor import LiveMonitor
    from framework.data.feature_schema import PACKET_COLUMNS
    class BrokenDetector:
        partial_input_supported = True
        feature_names = list(PACKET_COLUMNS)
        def analyze(self, *args):
            raise RuntimeError("Injected model inference error")
    monitor = LiveMonitor(detector=BrokenDetector())
    monitor._data.update(running=True, mode="packets", interface="fixture")
    stats = {"seconds": 10, "packets": 10, "ack": 4, "fin": 0, "psh": 2, "rst": 0, "syn_all": 3, "urg": 0}
    result = monitor._predict_multiclass_window(stats, "window-id", "timestamp")
    assert result["available"] is False
    assert result["observation_id"] == "window-id"
    assert monitor.snapshot()["running"] is True
    assert monitor.snapshot()["error"] is None


def test_binary_checkpoint_failure_remains_a_model_error(monkeypatch):
    from framework.data.live_monitor import LiveMonitor
    import framework.detection.packet_xgboost as packet
    def broken(*args, **kwargs):
        raise RuntimeError("Injected binary checkpoint error")
    monkeypatch.setattr(packet, "PacketXGBoost", broken)
    monitor = LiveMonitor()
    result = monitor._predict_window({"seconds": 10, "packets": 10})
    assert result["available"] is False
    assert "model" in result["label"].lower()
    assert monitor.snapshot()["error"] is None


def test_packet_prediction_survives_deleted_xai_module(monkeypatch, packet_model):
    original = builtins.__import__
    def missing_xai(name, *args, **kwargs):
        if name == "framework.xai.tree_shap":
            raise ImportError("Injected TreeSHAP removal")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing_xai)
    result = packet_model.predict(counts())
    assert result["is_attack"] in (True, False)
    assert 0 <= result["score"] <= 1
    assert result["explanation"] is None
    assert result["capability_errors"]["explanation"]


def test_missing_lime_does_not_prevent_assistant_using_real_tree_evidence(monkeypatch, detector):
    from framework.llm.local_model import LocalLanguageModel
    original = builtins.__import__
    def missing_lime(name, *args, **kwargs):
        if name.startswith("lime"):
            raise ImportError("Injected LIME removal")
        return original(name, *args, **kwargs)
    def simulated_generation(self, evidence):
        return {"llm_executed": False, "mode": "Language transport test fixture",
                "source": evidence["source"], "xai_method": evidence["xai"]["method"], "lime": evidence["lime"]}
    monkeypatch.setattr(builtins, "__import__", missing_lime)
    monkeypatch.setattr(LocalLanguageModel, "generate", simulated_generation)
    result = create_app(detector).test_client().post("/api/assistant/explain", json={"kind": "sample", "sample_id": 0})
    assert result.status_code == 200
    assert result.json["xai_method"] == "Exact TreeSHAP (native XGBoost)"
    assert result.json["lime"] is None
    assert result.json["source"]["source_row"] == 17


def test_exit_stop_succeeds_when_monitor_module_is_unavailable(monkeypatch):
    original = builtins.__import__
    def missing_monitor(name, *args, **kwargs):
        if name == "framework.data.live_monitor":
            raise ImportError("Injected capture removal")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing_monitor)
    client = create_app(registry=RegistryFixture()).test_client()
    client.get("/api/live/capabilities")
    response = client.post("/api/live/stop", json={})
    assert response.status_code == 200
    assert response.json["running"] is False


def test_prepared_csv_reorders_for_the_selected_family_and_ignores_labels(workspace):
    client, registry = workspace
    response = client.post("/api/models/kan/upload", data={"file": (io.BytesIO(b"label4,network_a,network_b\nattack,2,7\n"), "traffic.csv")})
    assert response.status_code == 200
    assert registry.received["kan"] == [[7, 2]]
    assert response.json["predictions"][0]["label"] == "benign"
    assert response.json["source"]["kind"] == "Prepared traffic CSV"


@pytest.mark.parametrize("content", [b"network_a,network_a\n1,2\n", b"network_a,network_b\n1,no\n",
                                     b"network_a,network_b\n,\n", b"network_a\n1\n",
                                     b"network_a,network_b\n1,2,3\n"])
def test_prepared_csv_rejects_duplicate_invalid_empty_incomplete_and_malformed_rows(workspace, content):
    client, _ = workspace
    response = client.post("/api/models/kan/upload", data={"file": (io.BytesIO(content), "traffic.csv")})
    assert response.status_code == 400


def test_capture_loop_continues_counting_when_a_model_raises(monkeypatch):
    """Real loop, header-window state and rules; replace only OS/capture I/O."""
    import psutil
    from framework.data.live_monitor import LiveMonitor
    monitor = LiveMonitor()
    class CaptureFixture:
        running = True
        def stop(self, **kwargs):
            self.running = False
    class TickFixture:
        ticks = 0
        ended = False
        def wait(self, seconds):
            self.ticks += 1
            if self.ticks > 2:
                self.ended = True
                return True
            monitor._window.update(packets=40, bytes=400, ack=0, fin=0, psh=0, rst=0, syn_all=0, urg=0)
            return False
        def is_set(self):
            return self.ended
    def begin_capture(interface):
        monitor._sniffer = CaptureFixture()
    def broken_binary(stats):
        raise RuntimeError("Injected binary inference failure")
    counters = SimpleNamespace(packets_sent=0, packets_recv=0, bytes_sent=0, bytes_recv=0)
    monkeypatch.setattr(psutil, "net_io_counters", lambda **kwargs: {"fixture": counters})
    monkeypatch.setattr(monitor, "_begin_capture", begin_capture)
    monkeypatch.setattr(monitor, "_connections", lambda: [])
    monkeypatch.setattr(monitor, "_predict_window", broken_binary)
    monitor._stop = TickFixture()
    monitor._started = time.monotonic()
    monitor._data.update(running=True, mode="packets", interface="fixture")
    monitor._run("fixture", "packets")
    snapshot = monitor.snapshot()
    assert snapshot["error"] is None
    assert snapshot["metrics"]["packets_observed"] == 80
    assert len(snapshot["history"]) == 2
    assert all(entry["status"] == "benign_looking" for entry in snapshot["history"])
    assert snapshot["model_prediction"] is None
