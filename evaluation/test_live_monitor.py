"""Real behavior rules, readable labels, opt-in monitoring and honest capture errors."""
import time
import threading

import pytest

from framework.app.labels import display_label, sample_names
from framework.detection.live_rules import assess_connections, assess_packets
from framework.data.live_monitor import LiveMonitor


def test_names_are_readable_and_do_not_expose_source_rows():
    assert display_label("benign") == "Normal network activity"
    assert "SYN flooding" in display_label("ddos_syn-flood-port-80")
    assert "Port scanning" in display_label("recon_port-scan")
    samples = sample_names([{"id": 0, "label": "benign", "source_row": 285252},
                            {"id": 1, "label": "benign", "source_row": 285253}])
    assert samples[0]["display_name"] == "Normal network activity — Example 1"
    assert samples[1]["display_name"].endswith("Example 2")
    assert "285252" not in samples[0]["display_name"]


def test_normal_connections_and_connection_attempt_burst_are_distinguished():
    normal = [{"status": "ESTABLISHED", "remote": "203.0.113.2:443"}]
    assert assess_connections(normal, 12)["status"] == "benign_looking"
    burst = [{"status": "SYN_SENT", "remote": f"203.0.113.2:{port}"} for port in range(45)]
    result = assess_connections(burst, 100)
    assert result["status"] == "suspicious"
    assert result["evidence"] and result["method"] == "Live behavior rules"
    assert "model" in result["limitations"].lower()
    assert assess_connections([], 0)["status"] == "idle"


def test_packet_rules_explain_flood_and_scan_without_treating_rate_as_attack():
    normal = {"packets": 30, "syn": 3, "syn_ack": 3, "ack": 20, "unique_ports_per_source": {}, "seconds": 5}
    assert assess_packets(normal)["status"] == "benign_looking"
    flood = {**normal, "packets": 700, "syn": 680, "syn_ack": 0, "ack": 0}
    assert assess_packets(flood)["status"] == "suspicious"
    assert assess_packets({**normal, "unique_ports_per_source": {"203.0.113.1": 30}})["status"] == "suspicious"
    heavy = {**normal, "packets": 20000, "syn": 0, "ack": 19000}
    assert assess_packets(heavy)["status"] == "benign_looking"


def test_monitor_defaults_stopped_and_is_explicitly_started_and_stopped():
    monitor = LiveMonitor()
    assert monitor.snapshot()["running"] is False
    assert monitor.snapshot()["assessment"]["status"] == "not_assessed"
    with pytest.raises(ValueError):
        monitor.start("nonexistent-interface", "connections")
    with pytest.raises(ValueError):
        monitor.start("", "pretend_capture")
    assert monitor.stop()["running"] is False


def test_missing_capture_driver_never_becomes_fake_benign(monkeypatch):
    monitor = LiveMonitor()
    monkeypatch.setattr(monitor, "capabilities", lambda: {"interfaces": [{"id": "test", "addresses": ["10.0.0.1"]}],
                                                        "connections_available": True,
                                                        "packet_capture_available": False,
                                                        "packet_capture_message": "Install Npcap"})
    with pytest.raises(ValueError, match="Npcap"):
        monitor.start("test", "packets")
    assert monitor.snapshot()["assessment"]["status"] == "not_assessed"


def test_new_start_cannot_overtake_a_stop_in_progress(monkeypatch):
    monitor = LiveMonitor()
    entered, release, started = threading.Event(), threading.Event(), threading.Event()
    class FinishingWorker:
        def is_alive(self):
            return False
        def join(self, timeout):
            entered.set()
            release.wait(2)
    monitor._thread = FinishingWorker()
    monkeypatch.setattr(monitor, "capabilities", lambda: {"interfaces": [{"id": "test", "addresses": []}],
                                                        "connections_available": True, "packet_capture_available": False})
    monkeypatch.setattr(monitor, "_run", lambda interface, mode: monitor._stop.wait(2))
    stopper = threading.Thread(target=monitor.stop)
    stopper.start()
    assert entered.wait(1)
    def start():
        monitor.start("test", "connections")
        started.set()
    starter = threading.Thread(target=start)
    starter.start()
    try:
        assert not started.wait(.1), "A new monitor started before the previous stop finished"
    finally:
        release.set()
        stopper.join(2)
        starter.join(2)
        monitor.stop()


def test_stalled_monitor_does_not_keep_a_healthy_verdict():
    monitor = LiveMonitor()
    monitor._data.update(running=True, state="monitoring", mode="connections",
                         assessment=assess_connections([{"status": "ESTABLISHED"}], 1),
                         model_prediction={"label": "normal"})
    monitor._started = time.monotonic() - 20
    monitor._last_observation = time.monotonic() - 20
    result = monitor.snapshot()
    assert result["state"] == "error"
    assert result["assessment"]["status"] == "not_assessed"
    assert result["model_prediction"] is None
    assert "fresh" in result["error"]


def test_packet_counts_flags_independently_and_only_our_host(monkeypatch):
    pytest.importorskip("scapy")
    from scapy.layers.inet import IP, TCP
    from scapy.layers.l2 import Ether, ARP
    monitor = LiveMonitor()
    monitor._local_addresses = {"10.0.0.1"}
    monitor._local_macs = {"02:00:00:00:00:01"}
    monitor._window = monitor._empty_window()
    monitor._packet(IP(src="10.0.0.1", dst="203.0.113.1") / TCP(flags="SA"))
    monitor._packet(IP(src="10.0.0.1", dst="203.0.113.1") / TCP(flags="FPRU"))
    monitor._packet(Ether(src="02:00:00:00:00:01") / ARP(psrc="10.0.0.1"))
    monitor._packet(IP(src="10.0.0.2", dst="203.0.113.1") / TCP(flags="S"))
    assert monitor._window["packets"] == 3
    assert monitor._window["syn"] == 0 and monitor._window["syn_all"] == 1
    for flag in ["ack", "fin", "psh", "rst", "urg"]:
        assert monitor._window[flag] == 1
    assert len(monitor.snapshot()["recent_packets"]) == 3


def test_delayed_window_skips_model_without_replacing_real_capture():
    result = LiveMonitor()._predict_window({"seconds": 12, "packets": 100})
    assert result["available"] is False
    assert "continue" in result["limitations"]


def test_session_expiry_clears_old_assessment_and_prediction(monkeypatch):
    import sys
    from types import SimpleNamespace
    monitor = LiveMonitor()
    monitor._started = time.monotonic() - 3601
    monitor._stop = SimpleNamespace(wait=lambda period: False)
    monitor._data.update(running=True, mode="connections",
                         assessment=assess_connections([{"status": "ESTABLISHED"}], 1),
                         model_prediction={"label": "old model result"})
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(net_io_counters=lambda **kwargs: {"test": object()}))
    monitor._run("test", "connections")
    result = monitor.snapshot()
    assert result["running"] is False and result["state"] == "stopped"
    assert result["assessment"]["status"] == "not_assessed"
    assert result["model_prediction"] is None


def test_failed_stop_does_not_keep_a_health_verdict():
    class UnresponsiveWorker:
        def is_alive(self):
            return True
        def join(self, timeout):
            pass
    monitor = LiveMonitor()
    monitor._thread = UnresponsiveWorker()
    monitor._data.update(running=True, state="monitoring",
                         assessment=assess_connections([{"status": "ESTABLISHED"}], 1),
                         model_prediction={"label": "old model result"})
    result = monitor.stop()
    assert result["state"] == "error"
    assert result["running"] is True  # Do not pretend the worker stopped.
    assert result["assessment"]["status"] == "not_assessed"
    assert result["model_prediction"] is None
