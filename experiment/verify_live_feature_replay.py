"""Independent offline packet oracle for the candidate native measurement contract.

Only NumPy statistics and stdlib packet decoding are used; no native extraction,
training derivation, model inference, capture driver, or external service is used.
The real PCAP run reads a bounded first ten-second window and stores aggregates
only. Cross-dataset replay does not establish DataSense exporter parity or accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "rust/netsec/target/debug/examples" / (
    "feature_replay.exe" if sys.platform == "win32" else "feature_replay")
DEFAULT_PORTS = ROOT / "rust/netsec/crates/runtime/src/live/port_features.json"
DEFAULT_PCAP = ROOT / "external/HEM/UQIOT2022/UQIOT2022/data/benign_samples/whole_week.pcap"
DEFAULT_REPORT = ROOT / "artifacts/training/live_features_20261027_v1/verification/replay_report.json"
GROUPS = ("header-length", "ip-flags", "ip-length", "mss", "packet-size",
          "payload-length", "tcp-flags", "time-delta", "ttl", "window-size")
SUFFIXES = ("avg", "max", "min", "std_deviation")
FLAG_BITS = {"ack": 16, "fin": 1, "psh": 8, "rst": 4, "syn": 2, "urg": 32}
WITHHELD = {"network_fragmentation-score", "network_interval-packets", "network_macs_dst_count",
            *(f"network_protocols_{side}_count" for side in ("all", "src", "dst"))}
LOCAL_IP = "192.0.2.10"
REMOTE_IP = "198.51.100.20"
LOCAL_V6 = "2001:db8::10"
REMOTE_V6 = "2001:db8::20"
LOCAL_MAC = "02:00:00:00:00:10"
REMOTE_MAC = "02:00:00:00:00:20"


@dataclass
class PacketFacts:
    """Values independently decoded from one frame; missing fields remain absent."""
    length: int
    timestamp: int | None
    src_mac: str
    dst_mac: str
    src_ip: str | None = None
    dst_ip: str | None = None
    version: int | None = None
    outgoing: bool = False
    incoming: bool = False
    protocol: str = "Other"
    fragmented: bool = False
    src_port: int | None = None
    dst_port: int | None = None
    flags: int | None = None
    values: dict[str, float] = field(default_factory=dict)
    errors: set[str] = field(default_factory=set)


def _u16(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset:offset + 2], "big")


def decode_frame(frame: dict[str, Any], local_ips: list[str], local_mac: str | None) -> PacketFacts | None:
    """Small, separately implemented Ethernet/IP/TCP/UDP/ICMP reference parser.

    The contract admits an endpoint by local IP or Ethernet MAC. Bounds use both
    retained bytes and the declared original wire length. Checksums are outside
    the contract. Fragment payload is never reconstructed.
    """
    data = bytes.fromhex(frame["hex"])
    wire_length = frame["original_length"]
    if len(data) < 14:
        return None
    mac = lambda raw: ":".join(f"{byte:02x}" for byte in raw)
    packet = PacketFacts(wire_length, frame.get("timestamp_ns"), mac(data[6:12]), mac(data[:6]))
    local_mac = local_mac.lower() if local_mac else None
    offset, ether_type = 14, _u16(data, 12)

    def unknown_transport(protocol: int | None) -> None:
        packet.errors.add("incomplete_transport_headers")
        if protocol in (None, 6):
            packet.errors.add("incomplete_tcp_headers")
        if protocol in (None, 6, 17):
            packet.errors.add("incomplete_port_headers")

    tags = 0
    while ether_type in (0x8100, 0x88A8, 0x9100):
        tags += 1
        if tags > 2 or offset + 4 > len(data):
            packet.errors.add("incomplete_link_headers")
            unknown_transport(None)
            break
        ether_type = _u16(data, offset + 2)
        offset += 4

    protocol, transport_length, transport_offset, noninitial = None, 0, offset, False
    if not packet.errors:
        if ether_type == 0x0800:
            packet.version, packet.protocol = 4, "IPv4"
            if len(data) < offset + 20:
                packet.errors.add("invalid_ipv4_headers")
                unknown_transport(None)
            else:
                packet.src_ip = str(ipaddress.ip_address(data[offset + 12:offset + 16]))
                packet.dst_ip = str(ipaddress.ip_address(data[offset + 16:offset + 20]))
                ihl, total, frag = (data[offset] & 15) * 4, _u16(data, offset + 2), _u16(data, offset + 6)
                protocol = data[offset + 9]
                packet.fragmented, noninitial = bool(frag & 0x3FFF), bool(frag & 0x1FFF)
                if data[offset] >> 4 != 4 or ihl < 20 or total < ihl or offset + ihl > len(data) or offset + total > wire_length:
                    packet.errors.add("invalid_ipv4_headers")
                    unknown_transport(protocol)
                else:
                    packet.values.update({"header-length": ihl, "ip-length": total,
                                          "ip-flags": frag >> 13, "ttl": data[offset + 8]})
                    transport_offset, transport_length = offset + ihl, total - ihl
        elif ether_type == 0x86DD:
            packet.version, packet.protocol = 6, "IPv6"
            if len(data) < offset + 40:
                packet.errors.add("invalid_ipv6_headers")
                unknown_transport(None)
            else:
                packet.src_ip = str(ipaddress.ip_address(data[offset + 8:offset + 24]))
                packet.dst_ip = str(ipaddress.ip_address(data[offset + 24:offset + 40]))
                total = 40 + _u16(data, offset + 4)
                protocol, transport_offset = data[offset + 6], offset + 40
                if data[offset] >> 4 != 6 or total == 40 or offset + total > wire_length:
                    packet.errors.add("invalid_ipv6_headers")
                    unknown_transport(None)
                else:
                    for _ in range(8):
                        if protocol not in (0, 43, 44, 51, 60):
                            break
                        if transport_offset + 8 > min(len(data), offset + total):
                            packet.errors.add("invalid_ipv6_headers")
                            unknown_transport(None)
                            break
                        next_protocol = data[transport_offset]
                        extension_length = 8 if protocol == 44 else (
                            (data[transport_offset + 1] + 2) * 4 if protocol == 51 else
                            (data[transport_offset + 1] + 1) * 8)
                        if protocol == 44:
                            fragment = _u16(data, transport_offset + 2)
                            packet.fragmented = bool(fragment & 0xFFF9)
                            noninitial = bool(fragment & 0xFFF8)
                        transport_offset += extension_length
                        protocol = next_protocol
                        if transport_offset > min(len(data), offset + total):
                            packet.errors.add("invalid_ipv6_headers")
                            unknown_transport(None)
                            break
                    if not packet.errors:
                        packet.values.update({"ipv6_header-length": transport_offset - offset,
                                              "ipv6_ip-length": total, "ipv6_hop-limit": data[offset + 7]})
                        transport_length = total - (transport_offset - offset)
        elif ether_type == 0x0806:
            packet.protocol = "ARP"

    packet.outgoing = packet.src_ip in local_ips or (local_mac is not None and packet.src_mac == local_mac)
    packet.incoming = packet.dst_ip in local_ips or (local_mac is not None and packet.dst_mac == local_mac)
    if not (packet.outgoing or packet.incoming):
        return None
    if packet.errors:
        return packet
    if noninitial:
        if protocol in (6, 17, 1, 58):
            unknown_transport(protocol)
        return packet
    if protocol is None:
        return packet
    packet.protocol = {6: "TCP", 17: "UDP", 1: "ICMP", 58: "ICMPv6"}.get(protocol, packet.protocol)
    start = transport_offset
    available = min(len(data) - start, transport_length)
    if protocol == 6:
        header_length = ((data[start + 12] >> 4) * 4) if available >= 20 else 0
        if header_length < 20 or available < header_length:
            unknown_transport(protocol)
            return packet
        packet.src_port, packet.dst_port = _u16(data, start), _u16(data, start + 2)
        packet.flags = ((data[start + 12] & 1) << 8) | data[start + 13]
        packet.values.update({"tcp-flags": packet.flags, "window-size": _u16(data, start + 14)})
        option, stop = start + 20, start + header_length
        while option < stop:
            kind = data[option]
            if kind == 0:
                break
            if kind == 1:
                option += 1
                continue
            if option + 2 > stop or data[option + 1] < 2 or option + data[option + 1] > stop:
                packet.errors.add("malformed_tcp_options")
                break
            size = data[option + 1]
            if kind == 2:
                if size != 4:
                    packet.errors.add("malformed_tcp_options")
                    break
                packet.values["mss"] = _u16(data, option + 2)
            option += size
        if packet.fragmented:
            packet.errors.add("incomplete_transport_headers")
        else:
            packet.values["payload-length"] = transport_length - header_length
    elif protocol == 17:
        if available < 8:
            unknown_transport(protocol)
            return packet
        udp_length = _u16(data, start + 4)
        if udp_length < 8 or (udp_length > transport_length and not packet.fragmented):
            unknown_transport(protocol)
            return packet
        packet.src_port, packet.dst_port = _u16(data, start), _u16(data, start + 2)
        if packet.fragmented:
            packet.errors.add("incomplete_transport_headers")
        else:
            packet.values["payload-length"] = udp_length - 8
    elif protocol in (1, 58):
        if available < 8:
            unknown_transport(protocol)
        elif packet.fragmented:
            packet.errors.add("incomplete_transport_headers")
        else:
            packet.values["payload-length"] = transport_length - 8
    return packet


def load_port_names(path: Path = DEFAULT_PORTS) -> list[str]:
    names = json.loads(path.read_text(encoding="utf-8"))
    if len(names) != 192 or len(set(names)) != 192 or not all(
            re.fullmatch(r"network_ports_(src|dst)__present_\d+", name) for name in names):
        raise ValueError("Expected exactly 192 unique numeric source/destination port feature names")
    return names


def oracle(request: dict[str, Any], port_names: list[str]) -> dict[str, Any]:
    """Vectorized, two-pass population summaries independent of native Welford."""
    packets = [packet for frame in request["frames"] if (packet := decode_frame(
        frame, request["local_ips"], request.get("local_mac"))) is not None]
    groups: dict[str, list[float]] = {name: [] for name in GROUPS}
    errors: Counter[str] = Counter(error for packet in packets for error in packet.errors)
    endpoints = {family: {side: set() for side in ("all", "src", "dst")}
                 for family in ("ips", "macs", "ports", "ipv6_ips")}
    for index, packet in enumerate(packets):
        groups["packet-size"].append(packet.length)
        for key, value in packet.values.items():
            if key in groups:
                groups[key].append(value)
        if packet.timestamp is None:
            errors["missing_timestamps"] += 1
        elif index and packets[index - 1].timestamp is not None:
            delta = (packet.timestamp - packets[index - 1].timestamp) / 1_000_000_000
            groups["time-delta"].append(delta)
            errors["negative_time_deltas"] += delta < 0
        if packet.flags is not None and "mss" not in packet.values and "malformed_tcp_options" not in packet.errors:
            errors["tcp_packets_without_mss_option"] += 1
        for family, attribute in (("ips", "ip"), ("macs", "mac"), ("ports", "port")):
            if family == "ips" and packet.version not in (4, 6):
                continue
            target = "ipv6_ips" if family == "ips" and packet.version == 6 else family
            for side in ("src", "dst"):
                value = getattr(packet, f"{side}_{attribute}")
                if value is not None:
                    endpoints[target][side].add(value)
                    endpoints[target]["all"].add(value)

    invalid_ip = errors["invalid_ipv4_headers"] or errors["incomplete_link_headers"]
    invalid_groups = {
        **{key: invalid_ip for key in ("header-length", "ip-length", "ip-flags", "ttl")},
        "mss": errors["incomplete_tcp_headers"] or errors["malformed_tcp_options"],
        "payload-length": errors["incomplete_transport_headers"],
        "tcp-flags": errors["incomplete_tcp_headers"], "window-size": errors["incomplete_tcp_headers"],
        "time-delta": errors["missing_timestamps"],
    }
    features: dict[str, float | None] = {}
    for name, values in groups.items():
        array = np.asarray(values, dtype=np.float64)
        stats = (float(array.mean()), float(array.max()), float(array.min()), float(array.std(ddof=0))) if len(array) else (None,) * 4
        for suffix, value in zip(SUFFIXES, stats):
            features[f"network_{name}_{suffix}"] = None if invalid_groups.get(name) else value
    count = len(packets)
    features.update(network_packets_all_count=float(count),
                    network_packets_src_count=float(sum(packet.outgoing for packet in packets)),
                    network_packets_dst_count=float(sum(packet.incoming for packet in packets)))
    features["network_fragmented-packets"] = None if invalid_ip else float(sum(
        packet.version == 4 and packet.fragmented for packet in packets))
    for family in ("ips", "macs", "ports"):
        invalid = invalid_ip if family == "ips" else errors["incomplete_port_headers"] if family == "ports" else False
        for side in ("all", "src", "dst"):
            features[f"network_{family}_{side}_count"] = None if invalid else float(len(endpoints[family][side]))
    for flag, mask in FLAG_BITS.items():
        features[f"network_tcp-flags-{flag}_count"] = None if errors["incomplete_tcp_headers"] else float(sum(
            bool(packet.flags & mask) for packet in packets if packet.flags is not None))
    features.update(dict.fromkeys(WITHHELD))
    assert len(features) == 64

    def divide(top: float | None, bottom: float | None) -> float | None:
        return top / bottom if top is not None and bottom is not None and math.isfinite(bottom) and bottom > 0 else None

    seconds = request["seconds"]
    for side in ("all", "src", "dst"):
        key = f"network_packets_{side}_count"
        features[f"{key}__per_second"] = divide(features[key], seconds)
    for flag in FLAG_BITS:
        features[f"network_tcp-flags-{flag}__fraction_of_packets"] = divide(features[f"network_tcp-flags-{flag}_count"], count)
    for side in ("src", "dst"):
        features[f"network_{side}_packet_fraction"] = divide(features[f"network_packets_{side}_count"], count)
    features["network_packet_size_coefficient_of_variation"] = divide(features["network_packet-size_std_deviation"], features["network_packet-size_avg"])
    features["network_payload_fraction"] = divide(features["network_payload-length_avg"], features["network_packet-size_avg"])
    features["network_packets_per_reported_ip"] = divide(count, features["network_ips_all_count"])
    features["network_packets_per_reported_port"] = divide(count, features["network_ports_all_count"])
    assert len(features) == 79
    global_invalid = not count or request.get("capture_drops", 0) > 0 or not math.isfinite(seconds) or seconds <= 0
    presence: dict[str, int | None] = {}
    for name in port_names:
        match = re.fullmatch(r"network_ports_(src|dst)__present_(\d+)", name)
        assert match
        side, port = match.group(1), int(match.group(2))
        presence[name] = None if global_invalid else (1 if port in endpoints["ports"][side] else
                         None if errors["incomplete_port_headers"] else 0)
    if global_invalid:
        features = dict.fromkeys(features)
    legacy_ok = count > 0 and math.isfinite(seconds) and abs(seconds - 10) <= 1 and not (
        request.get("capture_drops", 0) or errors["incomplete_tcp_headers"])
    return {"features": features, "port_presence": presence,
            "sample_counts": {f"network_{key}": len(values) for key, values in groups.items()},
            "quality": {"packets_observed": count, "capture_drops": request.get("capture_drops", 0),
                        **{name: errors[name] for name in (
                            "missing_timestamps", "negative_time_deltas", "invalid_ipv4_headers", "invalid_ipv6_headers",
                            "incomplete_link_headers", "incomplete_transport_headers", "incomplete_tcp_headers",
                            "incomplete_port_headers", "malformed_tcp_options", "tcp_packets_without_mss_option")}},
            "legacy_counts": [count, *(features[f"network_tcp-flags-{flag}_count"] for flag in FLAG_BITS)] if legacy_ok else None,
            "not_admitted_frames": len(request["frames"]) - count,
            "protocols": dict(Counter(packet.protocol for packet in packets)),
            "original_wire_bytes": sum(packet.length for packet in packets)}


def native_replay(request: dict[str, Any], binary: Path = DEFAULT_BINARY) -> dict[str, Any]:
    if not binary.is_file():
        raise FileNotFoundError(f"Native offline feature replay binary is required: {binary}")
    result = subprocess.run([str(binary)], input=json.dumps(request), text=True, capture_output=True,
                            check=True, timeout=30)
    return json.loads(result.stdout)


def compare(request: dict[str, Any], port_names: list[str], binary: Path = DEFAULT_BINARY) -> dict[str, Any]:
    expected, actual = oracle(request, port_names), native_replay(request, binary)
    observation, failures = actual["feature_observation"], []
    for section in ("features", "port_presence", "sample_counts", "quality"):
        if section in ("features", "port_presence") and set(observation.get(section, {})) != set(expected[section]):
            failures.append(f"{section}: field set differs")
        for name, wanted in expected[section].items():
            got = observation.get(section, {}).get(name, "MISSING")
            good = got is None if wanted is None else isinstance(got, (int, float)) and math.isclose(got, wanted, rel_tol=1e-9, abs_tol=1e-9)
            if not good:
                failures.append(f"{section}.{name}: expected {wanted!r}, received {got!r}")
    for key in ("legacy_counts", "not_admitted_frames"):
        if actual.get(key) != expected[key]:
            failures.append(f"{key}: expected {expected[key]!r}, received {actual.get(key)!r}")
    if actual.get("offline_replay") is not True or actual.get("capture_performed") is not False:
        failures.append("Offline replay provenance is missing")
    if observation.get("publisher_equivalence_verified") is not False:
        failures.append("Publisher equivalence must remain unverified")
    if expected["legacy_counts"] is None and not actual.get("legacy_error"):
        failures.append("An unavailable legacy vector requires an explicit error")
    return {"passed": not failures, "failures": failures, "expected": expected,
            "native_observation": observation, "native_legacy_counts": actual.get("legacy_counts"),
            "native_legacy_error": actual.get("legacy_error")}


def _ether(payload: bytes, kind: int = 0x0800, inbound: bool = False, vlan: bool = False) -> bytes:
    a, b = bytes.fromhex(LOCAL_MAC.replace(":", "")), bytes.fromhex(REMOTE_MAC.replace(":", ""))
    return (a + b if inbound else b + a) + (struct.pack("!HHH", 0x8100, 7, kind) if vlan else struct.pack("!H", kind)) + payload


def _ipv4(payload: bytes, protocol: int, inbound: bool = False, *, extra: int = 0,
          flags_offset: int = 0x4000, ttl: int = 64) -> bytes:
    a, b = ipaddress.ip_address(LOCAL_IP).packed, ipaddress.ip_address(REMOTE_IP).packed
    return struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + len(payload) + extra, 7, flags_offset,
                       ttl, protocol, 0, b if inbound else a, a if inbound else b) + payload


def _tcp(source: int = 1883, destination: int = 50000, flags: int = 18,
         window: int = 8192, options: bytes = b"\x02\x04\x05\xb4") -> bytes:
    assert len(options) % 4 == 0
    return struct.pack("!HHIIBBHHH", source, destination, 1, 0, (5 + len(options) // 4) << 4,
                       flags, window, 0, 0) + options


def _udp(source: int, destination: int, payload_size: int = 0) -> bytes:
    return struct.pack("!HHHH", source, destination, 8 + payload_size, 0)


def _frame(data: bytes, timestamp: int | None = 1_000_000_000, original: int | None = None) -> dict[str, Any]:
    return {"hex": data.hex(), "original_length": max(len(data), original or 0), "timestamp_ns": timestamp}


def fixture_requests() -> dict[str, dict[str, Any]]:
    """Hand-constructed packet bytes, never copied from real traffic."""
    tcp = _ether(_ipv4(_tcp(), 6, extra=12))
    inbound_tcp = _ether(_ipv4(_tcp(50000, 1883, 25, 0, b""), 6, True, ttl=128), inbound=True)
    udp = _ether(_ipv4(_udp(5353, 53, 8), 17, extra=8, ttl=32))
    arp = _ether(struct.pack("!HHBBH", 1, 0x0800, 6, 4, 2) + bytes(20), 0x0806, inbound=True)
    vlan = _ether(_ipv4(_udp(123, 62000, 4), 17, True, extra=4, ttl=96), inbound=True, vlan=True)
    ipv6 = _ether(struct.pack("!IHBB16s16s", 6 << 28, 16, 17, 255,
                             ipaddress.ip_address(LOCAL_V6).packed, ipaddress.ip_address(REMOTE_V6).packed) + _udp(443, 8080, 8), 0x86DD)
    base = {"local_ips": [LOCAL_IP, LOCAL_V6], "local_mac": LOCAL_MAC, "seconds": 10.0, "capture_drops": 0}
    mixed = [_frame(tcp, 1_000_000_000, 100), _frame(inbound_tcp, 1_100_000_000, 60),
             _frame(udp, 1_099_999_000, 70), _frame(arp, 1_300_000_000, 60),
             _frame(vlan, 1_400_000_000, 70), _frame(ipv6, 1_600_000_000, 100)]
    cases = {"mixed_protocols_and_signed_time": {**base, "frames": mixed},
             "tcp_without_mss": {**base, "frames": [_frame(inbound_tcp, original=60)]},
             "ipv6_excludes_ipv4_families": {**base, "frames": [_frame(ipv6, original=100)]},
             "arp_excludes_ip_and_transport": {**base, "frames": [_frame(arp, original=60)]},
             "capture_loss": {**base, "frames": mixed, "capture_drops": 1},
             "empty_window": {**base, "frames": []},
             "invalid_duration": {**base, "frames": mixed, "seconds": 0},
             "missing_timestamp": {**base, "frames": [_frame(tcp, None, 100), _frame(inbound_tcp, original=60)]},
             "truncated_tcp": {**base, "frames": [_frame(tcp[:42], original=100)]},
             "noninitial_tcp_fragment": {**base, "frames": [_frame(_ether(_ipv4(bytes(8), 6, flags_offset=1)), original=60)]},
             "initial_tcp_fragment": {**base, "frames": [_frame(_ether(_ipv4(_tcp(), 6, flags_offset=0x2000)), original=60)]},
             "positive_port_survives_incomplete": {**base, "frames": [mixed[0], _frame(tcp[:42], original=100)]},
             "malformed_tcp_option": {**base, "frames": [_frame(_ether(_ipv4(_tcp(options=b"\x02\x03\x05\x00"), 6)), original=60)]}}
    self_data = bytearray(inbound_tcp)
    self_data[26:30] = ipaddress.ip_address(LOCAL_IP).packed
    self_data[6:12] = bytes.fromhex(LOCAL_MAC.replace(":", ""))
    cases["local_self_traffic"] = {**base, "frames": [_frame(bytes(self_data), original=60)]}
    cases["unrelated_frame_ignored"] = {**base, "local_ips": [], "local_mac": "02:00:00:00:00:99", "frames": mixed}
    return cases


def read_pcap_window(path: Path, seconds: float = 10, max_records: int = 100_000,
                     max_file_bytes: int = 64 * 1024 * 1024) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read classic Ethernet PCAP only; stop at the first exclusive boundary.

    Retain at most 192 transient bytes per frame. Neither frame prefixes nor
    payload samples are written into reports. No full-file digest/scan is done.
    """
    before = path.stat()
    formats = {b"\xd4\xc3\xb2\xa1": ("<", 1000), b"\xa1\xb2\xc3\xd4": (">", 1000),
               b"\x4d\x3c\xb2\xa1": ("<", 1), b"\xa1\xb2\x3c\x4d": (">", 1)}
    frames, start, boundary, records = [], None, None, 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        header = stream.read(24)
        if len(header) != 24 or header[:4] not in formats:
            raise ValueError("Expected a classic PCAP header")
        endian, resolution = formats[header[:4]]
        major, minor, _, _, snaplen, link = struct.unpack(endian + "HHIIII", header[4:])
        if (major, minor) != (2, 4) or link != 1:
            raise ValueError("Only classic PCAP v2.4 Ethernet link type 1 is supported")
        digest.update(header)
        for _ in range(max_records):
            record_header = stream.read(16)
            if not record_header:
                break
            if len(record_header) != 16:
                raise ValueError("Truncated PCAP record header")
            sec, fraction, captured, original = struct.unpack(endian + "IIII", record_header)
            records += 1
            stamp = sec * 1_000_000_000 + fraction * resolution
            if fraction * resolution >= 1_000_000_000 or captured > snaplen or captured > original or original > 4 * 1024 * 1024:
                raise ValueError("Invalid or excessive PCAP record")
            if start is None:
                start = stamp
            if stamp >= start + int(seconds * 1_000_000_000):
                boundary = stamp
                break
            if stream.tell() + captured > min(before.st_size, max_file_bytes):
                raise ValueError("PCAP input is truncated or exceeds bounded source-read scope")
            prefix = stream.read(min(captured, 192))
            if len(prefix) != min(captured, 192):
                raise ValueError("Truncated PCAP frame")
            digest.update(record_header)
            digest.update(prefix)
            stream.seek(captured - len(prefix), 1)
            frames.append(_frame(prefix, stamp, original))
        else:
            raise ValueError("PCAP window exceeds explicit record bound")
        consumed = stream.tell()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("PCAP source metadata changed during read-only replay")
    return frames, {"source_bytes": before.st_size, "source_unchanged_size_and_mtime": True,
                    "records_read_including_boundary": records, "records_inside_window": len(frames),
                    "source_file_span_bytes": consumed, "window_start_ns": start,
                    "window_end_exclusive_ns": start + int(seconds * 1_000_000_000) if start is not None else None,
                    "first_boundary_timestamp_ns": boundary, "source_timestamp_resolution_ns": resolution,
                    "bounded_prefix_sha256": digest.hexdigest(),
                    "digest_scope": "PCAP global header and admitted-time record headers plus at most 192 captured bytes per record; not a full-source hash",
                    "payload_samples_saved": False, "header_prefixes_saved": False}


def run_verification(binary: Path, port_names: list[str], pcap: Path | None) -> dict[str, Any]:
    cases = {name: compare(request, port_names, binary) for name, request in fixture_requests().items()}
    real = None
    if pcap is not None:
        frames, source = read_pcap_window(pcap)
        request = {"frames": frames, "local_ips": ["192.168.0.111"], "local_mac": "a2:bd:fa:b5:89:92",
                   "seconds": 10, "capture_drops": 0}
        real = compare(request, port_names, binary)
        real["source"] = source
        real["source"]["path"] = str(pcap.relative_to(ROOT)) if pcap.is_relative_to(ROOT) else str(pcap)
        facts = real["expected"]
        for key, expected in (("records_read_including_boundary", 220), ("window_start_ns", 1619420399285747000)):
            if source[key] != expected:
                real["failures"].append(f"Historical bounded-PCAP anchor {key}: {source[key]} != {expected}")
        for key, expected in (("legacy_counts", [108, 7, 1, 3, 0, 0, 0]), ("protocols", {"TCP": 7, "UDP": 99, "ARP": 2})):
            if facts[key] != expected:
                real["failures"].append(f"Independent real-PCAP anchor {key}: {facts[key]} != {expected}")
        real["passed"] = not real["failures"]
    all_results = [*cases.values(), *([real] if real is not None else [])]
    coverage = {}
    for section in ("features", "port_presence"):
        for name in all_results[0]["expected"][section]:
            values = [result["expected"][section][name] for result in all_results]
            coverage[name] = {"compared_cases": len(values), "numeric_cases": sum(value is not None for value in values),
                              "null_cases": values.count(None), "zero_cases": values.count(0), "one_cases": values.count(1)}
    return {"schema_version": "independent-native-replay-verification-v1", "passed": all(result["passed"] for result in all_results),
            "oracle": "Independent stdlib protocol parsing and NumPy float64 two-pass population statistics; no native/model derivation imports",
            "numeric_tolerance": {"absolute": 1e-9, "relative": 1e-9}, "raw_fields": 64, "derived_fields": 15,
            "port_presence_fields": len(port_names), "fixture_cases": cases, "real_pcap": real, "per_feature_coverage": coverage,
            "native_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "oracle_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limits": {"publisher_equivalence_verified": False, "datasense_paired_pcap_comparison": False,
                       "detection_accuracy_evaluated": False, "capture_performed": False, "model_inference_performed": False,
                       "training_performed": False, "locked_test_read": False, "source_pcap_mutated": False,
                       "payload_samples_saved": False, "live_timestamp_accuracy_verified": False,
                       "pcap_capture_loss_known": False,
                       "pcap_loss_note": "Classic PCAP has no drop telemetry; capture_drops=0 is an explicit replay assumption, not proof of losslessness"}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--port-schema", type=Path, default=DEFAULT_PORTS)
    parser.add_argument("--pcap", type=Path, default=DEFAULT_PCAP)
    parser.add_argument("--fixtures-only", action="store_true")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error(f"Native offline replay binary required; build it explicitly first: {args.binary}")
    report = run_verification(args.binary, load_port_names(args.port_schema), None if args.fixtures_only else args.pcap)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "fixture_cases": len(report["fixture_cases"]),
                      "real_pcap_checked": report["real_pcap"] is not None, "report": str(args.report)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
