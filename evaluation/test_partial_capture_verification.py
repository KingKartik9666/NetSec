"""Synthetic header/CPU model fixtures test verification without live capture."""
import hashlib
import json
import subprocess
import sys
import time

import numpy as np
import pytest

from experiment import verify_partial_capture as verification
from framework.data.feature_schema import packet_vector
from framework.data.live_monitor import LiveMonitor
from framework.detection.packet_xgboost import FEATURE_NAMES
from framework.detection.trained_xgboost import XGBoostDetector, json_features


@pytest.fixture
def capture_detector(tmp_path):
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(42)
    names = ["log_unavailable", *FEATURE_NAMES,
             *[f"network_protocols_{group}__present_UNSEEN" for group in ("all", "src", "dst")]]
    X = rng.integers(0, 10, size=(90, len(names))).astype(np.float32)
    X[:, -3:] %= 2
    model = xgb.train({"objective": "multi:softprob", "num_class": 3, "device": "cpu",
                       "seed": 42, "nthread": 2, "max_depth": 2},
                      xgb.DMatrix(X, label=np.arange(90) % 3, nthread=2), num_boost_round=3)
    checkpoint = tmp_path / "fixture.ubj"
    model.save_model(checkpoint)
    bundle = {"schema_version": 2, "partial_input_supported": True, "promoted": True,
              "promotion": {"improved": True}, "checkpoint": checkpoint.name,
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "feature_names": names, "numeric_features": names[:8],
              "protocol_vocabulary": {f"network_protocols_{group}": [] for group in ("all", "src", "dst")},
              "classes": ["benign", "synthetic_probe", "synthetic_flood"], "benign_index": 0,
              "samples": [], "model": {"name": "Synthetic verification fixture; not research results"}}
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    return XGBoostDetector(path)


def header_packets():
    pytest.importorskip("scapy")
    from scapy.layers.inet import IP, TCP
    from scapy.layers.l2 import Ether, ARP
    return [Ether(src="02:00:00:00:00:01") / IP(src="192.0.2.10", dst="203.0.113.4") / TCP(flags="SA"),
            Ether(src="02:00:00:00:00:01") / IP(src="192.0.2.10", dst="203.0.113.4") / TCP(flags="FPRU"),
            Ether(src="02:00:00:00:00:01") / ARP(psrc="192.0.2.10", pdst="192.0.2.11"),
            Ether(src="02:00:00:00:00:02") / IP(src="192.0.2.12", dst="203.0.113.4") / TCP(flags="S")]


def test_independent_oracle_tracks_completed_window_and_csv_checkpoint_roundtrip(capture_detector, tmp_path):
    monitor = verification.VerifiedMonitor(capture_detector)
    monitor._local_addresses = {"192.0.2.10"}
    monitor._local_macs = {"02:00:00:00:00:01"}
    monitor._window = monitor._empty_window()
    for packet in header_packets():
        monitor._packet(packet)
    stats = {**monitor._window, "seconds": 10.0}
    assert stats["packets"] == 3 and stats["syn_all"] == stats["syn_ack"] == 1 and stats["syn"] == 0
    monitor._window = monitor._empty_window()  # Production rotates before inference.
    report = monitor._predict_multiclass_window(stats, "fixture-observation", "fixture-time")
    proof = verification.verify_report(capture_detector, report, stats, tmp_path, "fixture")
    assert proof["observed_count"] == 7 and proof["missing_count"] == 4
    assert proof["checkpoint_predictions_equal"] and proof["shap_additivity_verified"]
    assert monitor.completed_evidence["fixture-observation"]["seven_count_header_oracle_match"]
    monitor._data.update(running=True, multiclass_prediction=report, model_prediction={"label": "fixture"})
    monitor._started = time.monotonic()
    stopped = monitor.stop()
    assert stopped["multiclass_prediction"] is None and stopped["model_prediction"] is None


def test_oracle_rejects_incorrect_recorded_counts(capture_detector):
    monitor = verification.VerifiedMonitor(capture_detector)
    monitor._local_addresses = {"192.0.2.10"}
    monitor._window = monitor._empty_window()
    monitor._packet(header_packets()[0])
    stats = {**monitor._window, "seconds": 10.0, "syn_all": 0}
    monitor._window = monitor._empty_window()
    with pytest.raises(AssertionError, match="oracle disagrees"):
        monitor._predict_multiclass_window(stats, "bad-fixture", "fixture-time")


def test_replay_uses_real_callback_but_preserves_synthetic_fixture_provenance(capture_detector, tmp_path, monkeypatch):
    pytest.importorskip("scapy")
    from scapy.utils import wrpcap
    packets = header_packets()
    for index, packet in enumerate(packets):
        packet.time = 1000 + index
    boundary = packets[0].copy()
    boundary.time = 1010
    capture = tmp_path / "synthetic_headers.pcap"
    wrpcap(str(capture), [*packets, boundary])
    monkeypatch.setattr(LiveMonitor, "_predict_window", lambda self, stats: {
        "available": False, "label": "Explicitly disabled binary model in synthetic unit test"})
    result = verification.verify_replay(capture_detector, tmp_path, capture)
    assert result["status"] == "passed" and result["capture_started"] is False
    assert result["seven_count_header_oracle_match"] and result["stats"]["packets"] == 3
    assert result["source_kind"] == "Supplied PCAP recording; provenance and per-packet labels unverified"
    assert result["multiclass_prediction"]["source"]["mode"] == "pcap_replay"
    # A fresh interpreter has no scapy layers registered by header_packets().
    # The production verifier must register dissectors before opening the PCAP.
    child_directory = tmp_path / "fresh_process"
    child_directory.mkdir()
    code = ("from pathlib import Path; from experiment.verify_partial_capture import verify_replay; "
            "from framework.detection.trained_xgboost import XGBoostDetector; "
            "from framework.data.live_monitor import LiveMonitor; "
            "LiveMonitor._predict_window=lambda self,stats: {'available':False}; "
            f"result=verify_replay(XGBoostDetector(Path({str(capture_detector.bundle_path)!r})), "
            f"Path({str(child_directory)!r}), Path({str(capture)!r})); "
            "assert result['stats']['packets']==3 and result['status']=='passed'")
    completed = subprocess.run([sys.executable, "-c", code], cwd=verification.ROOT,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_changed_feature_or_score_is_rejected(capture_detector, tmp_path):
    stats = {"packets": 12, "ack": 7, "fin": 0, "psh": 3, "rst": 0, "syn_all": 2, "urg": 0, "seconds": 10}
    vector = packet_vector(stats, capture_detector.feature_names)
    report = capture_detector.analyze(json_features(vector), {"kind": "Synthetic unit fixture"})
    report["prediction"]["score"] += .01
    with pytest.raises(AssertionError, match="score differs"):
        verification.verify_report(capture_detector, report, stats, tmp_path, "invalid")


def test_missing_promoted_bundle_records_unavailable_and_preserves_existing_proof(monkeypatch, tmp_path):
    original = tmp_path / "original.json"
    original.write_text(json.dumps({"schema_version": 1}))
    old = tmp_path / "previous_proof.json"
    old.write_bytes(b"preserve this exact proof")
    monkeypatch.setattr(verification, "default_bundle", lambda: original)
    monkeypatch.setattr(verification, "OLD_PROOF", old)
    monkeypatch.setattr(verification, "verify_live", lambda *args: pytest.fail("Unavailable model must not start capture"))
    root = tmp_path / "proofs"
    assert verification.main(["--mode", "both", "--output-root", str(root)]) == 2
    assert verification.main(["--output-root", str(root)]) == 2
    paths = list(root.glob("*/verification.json"))
    assert len(paths) == 2
    for path in paths:
        proof = json.loads(path.read_text())
        assert proof["status"] == "unavailable" and proof["previous_replay_proof_preserved"]
        assert "no substitute" in proof["reason"]
    assert old.read_bytes() == b"preserve this exact proof"


def test_live_unavailability_never_becomes_a_capture_success(capture_detector, tmp_path, monkeypatch):
    monkeypatch.setattr(verification.platform, "system", lambda: "Windows")
    monkeypatch.setattr(verification.VerifiedMonitor, "capabilities", lambda self: {
        "packet_capture_available": False, "packet_capture_message": "Npcap unavailable fixture"})
    monkeypatch.setattr(verification.VerifiedMonitor, "start", lambda *args: pytest.fail("Unavailable driver must not start"))
    result = verification.verify_live(capture_detector, tmp_path)
    assert result["status"] == "unavailable" and result["capture_started"] is False


def test_failed_async_start_does_not_claim_capture_started_and_always_stops(capture_detector, tmp_path, monkeypatch):
    monkeypatch.setattr(verification.platform, "system", lambda: "Windows")
    stopped = []

    class FailedCapture:
        def __init__(self, detector):
            pass

        def capabilities(self):
            return {"packet_capture_available": True}

        def start(self, interface, mode):
            assert interface == "Wi-Fi" and mode == "packets"

        def snapshot(self):
            return {"state": "error", "error": "Synthetic permission-denied start failure"}

        def stop(self):
            stopped.append(True)
            return {"running": False, "state": "stopped", "model_prediction": None,
                    "multiclass_prediction": None, "assessment": {"status": "not_assessed"}}

    monkeypatch.setattr(verification, "VerifiedMonitor", FailedCapture)
    result = verification.verify_live(capture_detector, tmp_path)
    assert result["status"] == "failed" and result["capture_requested"] is True
    assert result["capture_started"] is False and result["stop_clears_predictions"]
    assert stopped == [True]
