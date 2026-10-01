"""Optional real-upstream tests using tiny synthetic PCAP test fixtures."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scapy", reason="Install requirements-hem-features.txt for upstream extraction tests")
pytest.importorskip("pyximport", reason="Original HEM feature engine imports Cython")

from scapy.layers.inet import IP, UDP
from scapy.layers.l2 import Ether
from scapy.utils import wrpcap

from framework.data.hem_features import HEM_ROOT, extract_capture, load_hem_engine


@pytest.fixture
def capture(tmp_path):
    packets = []
    for stamp in (1.0, 1.0, 2.0):
        packet = Ether(src="02:00:00:00:00:01", dst="02:00:00:00:00:02") / IP(
            src="192.0.2.1", dst="192.0.2.2"
        ) / UDP(sport=5000, dport=6000)
        packet.time = stamp
        packets.append(packet)
    path = tmp_path / "synthetic_test.pcap"
    wrpcap(str(path), packets)
    return path


def test_original_engine_executes_and_preserves_state(capture, tmp_path):
    output = tmp_path / "features.csv"
    result = extract_capture(capture, output, limit=3)
    data = pd.read_csv(output)
    features = data[[f"kit_fe_{i}" for i in range(100)]]
    assert features.shape == (3, 100)
    assert np.isfinite(features.to_numpy()).all()
    assert features.iloc[0, 0] == pytest.approx(1)
    assert features.iloc[1, 0] == pytest.approx(2)
    assert Path(result["engine_source"]).resolve() == (
        HEM_ROOT / "feature_extraction/after_image/net_stat.py"
    ).resolve()
    assert len(result["feature_names"]) == 100
    assert result["protocol_counts"] == {"UDP": 3}
    assert output.with_suffix(".metadata.json").is_file()
    # Compare with calls to the original class, not a mocked feature generator.
    engine, _ = load_hem_engine()
    expected = [engine.updateGetStats(
        0, "02:00:00:00:00:01", "02:00:00:00:00:02", "192.0.2.1", "5000",
        "192.0.2.2", "6000", stamp, 42,
    ) for stamp in (1.0, 1.0, 2.0)]
    np.testing.assert_allclose(features, expected)


def test_packet_limit_and_repeated_runs_are_deterministic(capture, tmp_path):
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    assert extract_capture(capture, first, limit=2)["packets_read"] == 2
    extract_capture(capture, second, limit=2)
    assert first.read_bytes() == second.read_bytes()


def test_existing_artifacts_are_preserved(capture, tmp_path):
    output = tmp_path / "existing.csv"
    output.write_text("preserve me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        extract_capture(capture, output)
    assert output.read_text() == "preserve me"


def test_out_of_order_packets_are_counted(tmp_path):
    packets = []
    for stamp in (2.0, 1.0, 3.0):
        packet = Ether() / IP(src="192.0.2.1", dst="192.0.2.2") / UDP()
        packet.time = stamp
        packets.append(packet)
    path = tmp_path / "out_of_order.pcap"
    wrpcap(str(path), packets)
    result = extract_capture(path, tmp_path / "ordered.csv", 3)
    assert result["packets_read"] == 3
    assert result["feature_rows"] == 2
    assert result["skipped_packets"] == result["out_of_order_packets"] == 1


def test_invalid_limit(capture, tmp_path):
    with pytest.raises(ValueError, match="positive"):
        extract_capture(capture, tmp_path / "never.csv", limit=0)


def test_missing_capture(tmp_path):
    with pytest.raises(FileNotFoundError, match="PCAP"):
        extract_capture(tmp_path / "absent.pcap", tmp_path / "never.csv")
