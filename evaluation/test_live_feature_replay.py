"""Independent packet arithmetic; native checks skip explicitly until built."""
import struct

import numpy as np
import pytest

from experiment.verify_live_feature_replay import (
    DEFAULT_BINARY, FLAG_BITS, WITHHELD, compare, fixture_requests, load_port_names,
    native_replay, oracle, read_pcap_window,
)


@pytest.fixture(scope="module")
def port_names():
    return load_port_names()


def test_oracle_hand_calculated_mixed_packet_facts(port_names):
    expected = oracle(fixture_requests()["mixed_protocols_and_signed_time"], port_names)
    values = expected["features"]
    assert len(values) == 79 and len(expected["port_presence"]) == 192
    assert expected["legacy_counts"] == [6, 2, 1, 1, 0, 1, 0]
    assert values["network_packets_src_count"] == 3
    assert values["network_packets_dst_count"] == 3
    assert values["network_packet-size_avg"] == pytest.approx(np.mean([100, 60, 70, 60, 70, 100]))
    assert values["network_packet-size_std_deviation"] == pytest.approx(np.std([100, 60, 70, 60, 70, 100], ddof=0))
    assert values["network_ttl_avg"] == 80  # IPv4 64, 128, 32, 96; IPv6 255 excluded.
    assert expected["sample_counts"]["network_ttl"] == 4
    assert values["network_tcp-flags_avg"] == 21.5
    assert values["network_tcp-flags_std_deviation"] == 3.5
    assert values["network_mss_avg"] == 1460
    assert expected["sample_counts"]["network_mss"] == 1
    assert values["network_window-size_min"] == 0  # Measured zero is preserved.
    assert values["network_time-delta_min"] == pytest.approx(-0.000001)
    assert expected["quality"]["negative_time_deltas"] == 1
    assert values["network_payload-length_avg"] == pytest.approx((12 + 0 + 8 + 4 + 8) / 5)
    assert values["network_payload_fraction"] == pytest.approx(((12 + 0 + 8 + 4 + 8) / 5) / (460 / 6))
    assert values["network_src_packet_fraction"] == .5
    for flag, expected_count in zip(FLAG_BITS, [2, 1, 1, 0, 1, 0]):
        assert values[f"network_tcp-flags-{flag}__fraction_of_packets"] == pytest.approx(expected_count / 6)
    assert all(values[name] is None for name in WITHHELD)


def test_missing_support_and_true_zero_remain_distinct(port_names):
    missing = oracle(fixture_requests()["tcp_without_mss"], port_names)
    assert missing["features"]["network_mss_avg"] is None
    assert missing["features"]["network_window-size_avg"] == 0
    assert missing["features"]["network_time-delta_avg"] is None
    ipv6 = oracle(fixture_requests()["ipv6_excludes_ipv4_families"], port_names)
    assert ipv6["features"]["network_ttl_avg"] is None
    assert ipv6["features"]["network_ips_all_count"] == 0
    assert ipv6["features"]["network_ports_all_count"] == 2


def test_port_membership_uses_literal_endpoint_and_unknown_absence(port_names):
    complete = oracle(fixture_requests()["mixed_protocols_and_signed_time"], port_names)
    assert complete["port_presence"]["network_ports_src__present_1883"] == 1
    assert complete["port_presence"]["network_ports_dst__present_1883"] == 1
    incomplete = oracle(fixture_requests()["positive_port_survives_incomplete"], port_names)
    assert incomplete["port_presence"]["network_ports_src__present_1883"] == 1
    assert incomplete["port_presence"]["network_ports_dst__present_1883"] is None
    assert incomplete["features"]["network_ports_all_count"] is None
    for case in ("capture_loss", "empty_window", "invalid_duration"):
        result = oracle(fixture_requests()[case], port_names)
        assert all(value is None for value in result["features"].values())
        assert all(value is None for value in result["port_presence"].values())


def test_fragment_and_self_traffic_contract(port_names):
    initial = oracle(fixture_requests()["initial_tcp_fragment"], port_names)
    assert initial["features"]["network_fragmented-packets"] == 1
    assert initial["features"]["network_payload-length_avg"] is None
    assert initial["features"]["network_tcp-flags-syn_count"] == 1
    later = oracle(fixture_requests()["noninitial_tcp_fragment"], port_names)
    assert later["features"]["network_tcp-flags-syn_count"] is None
    assert later["features"]["network_ports_all_count"] is None
    self_traffic = oracle(fixture_requests()["local_self_traffic"], port_names)
    assert self_traffic["features"]["network_packets_src_count"] == 1
    assert self_traffic["features"]["network_packets_dst_count"] == 1


@pytest.mark.parametrize("case", list(fixture_requests()))
def test_native_matches_independent_oracle(case, port_names):
    if not DEFAULT_BINARY.is_file():
        pytest.skip("Native offline feature_replay binary not built; no implicit compilation during tests")
    report = compare(fixture_requests()[case], port_names)
    assert report["passed"], "\n".join(report["failures"])


def test_missing_native_is_an_error_for_verification(tmp_path):
    with pytest.raises(FileNotFoundError, match="required"):
        native_replay(fixture_requests()["empty_window"], tmp_path / "missing.exe")


@pytest.mark.parametrize("endian,resolution", [("<", 1000), (">", 1000), ("<", 1), (">", 1)])
def test_bounded_pcap_reader_excludes_exact_boundary(tmp_path, endian, resolution):
    magics = {("<", 1000): b"\xd4\xc3\xb2\xa1", (">", 1000): b"\xa1\xb2\xc3\xd4",
              ("<", 1): b"\x4d\x3c\xb2\xa1", (">", 1): b"\xa1\xb2\x3c\x4d"}
    packet = bytes.fromhex(fixture_requests()["tcp_without_mss"]["frames"][0]["hex"])
    payload = magics[endian, resolution] + struct.pack(endian + "HHIIII", 2, 4, 0, 0, 65535, 1)
    for seconds in (100, 109, 110):
        payload += struct.pack(endian + "IIII", seconds, 123, len(packet), len(packet)) + packet
    source = tmp_path / "fixture.pcap"
    source.write_bytes(payload)
    frames, metadata = read_pcap_window(source)
    assert len(frames) == 2
    assert metadata["records_read_including_boundary"] == 3
    assert metadata["window_start_ns"] == 100_000_000_000 + 123 * resolution
    assert metadata["source_file_span_bytes"] == 24 + 2 * (16 + len(packet)) + 16
    assert metadata["source_unchanged_size_and_mtime"]
    assert source.read_bytes() == payload


def test_pcap_parser_rejects_unbounded_or_truncated_input(tmp_path):
    source = tmp_path / "invalid.pcap"
    source.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="classic PCAP"):
        read_pcap_window(source)
