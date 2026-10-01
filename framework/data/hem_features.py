"""Run the original HEM/Kitsune statistics engine on offline packet captures.

Only packet decoding and file handling live here. The 100-feature calculation
executes external/HEM/feature_extraction/after_image/net_stat.py unchanged.
This module performs no classification, model training or LLM inference.
"""

import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np

from framework.data.loader import PROJECT_ROOT

HEM_ROOT = PROJECT_ROOT / "external" / "HEM"
CAPTURE_NAMES = {
    "syn": "SYN_Flooding_Smart_Clock_1.pcap",
    "scan": "Port_Scanning_Smart_Clock_1.pcap",
    "benign": "whole_week.pcap",
}


def find_capture(sample: str) -> Path:
    """Locate a user's existing UQ capture without moving or downloading it."""
    if sample not in CAPTURE_NAMES:
        raise ValueError(f"Unknown capture choice: {sample}")
    matches = sorted({path.resolve() for base in (PROJECT_ROOT / "dataset", HEM_ROOT)
                      for path in base.rglob(CAPTURE_NAMES[sample])})
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {CAPTURE_NAMES[sample]}, found {len(matches)}. "
            "Specify its exact location with --pcap PATH."
        )
    return matches[0]


def load_hem_engine():
    """Import the inspected upstream engine, refusing an unrelated name collision."""
    source = HEM_ROOT / "feature_extraction" / "after_image" / "net_stat.py"
    if not source.is_file():
        raise FileNotFoundError(f"Original HEM feature engine missing: {source}")
    sys.path.insert(0, str(source.parents[1]))
    try:
        module = importlib.import_module("after_image.net_stat")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "HEM feature dependencies missing. Run python -m pip install "
            "-r requirements-hem-features.txt."
        ) from exc
    finally:
        sys.path.pop(0)
    if Path(module.__file__).resolve() != source.resolve():
        raise RuntimeError("A different after_image package is loaded; use a fresh Python process.")
    # Match the upstream FE wrapper's limits. These are lazy dictionary caps,
    # not allocations; the CLI separately bounds the packets it reads.
    return module.netStat(HostLimit=100_000_000_000, HostSimplexLimit=100_000_000_000), source


def extract_capture(pcap: Path, output: Path, limit: int = 1000) -> dict:
    """Stream at most limit packets through the unchanged upstream feature engine.

    Ethernet IPv4, IPv6 and ARP records are supported. Each call starts fresh
    feature history. Capture filenames are provenance, never per-packet labels.
    Existing outputs are refused so research artifacts cannot be overwritten.
    """
    if limit < 1:
        raise ValueError("limit must be positive.")
    pcap, output = Path(pcap).resolve(), Path(output).resolve()
    metadata_path = output.with_suffix(".metadata.json")
    if not pcap.is_file():
        raise FileNotFoundError(f"PCAP not found: {pcap}")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"Output already exists; choose a new --output path: {output}")
    try:
        from scapy.layers.inet import ICMP, IP, TCP, UDP
        from scapy.layers.inet6 import IPv6
        from scapy.layers.l2 import ARP, Ether
        from scapy.utils import PcapReader
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install requirements-hem-features.txt to read PCAP files.") from exc

    engine, source = load_hem_engine()
    headers = engine.getNetStatHeaders()
    if len(headers) != 100:
        raise RuntimeError(f"Unexpected upstream schema: {len(headers)} features, expected 100.")
    rows, skipped, packets_read, previous_time = [], 0, 0, None
    out_of_order = 0
    protocols: dict[str, int] = {}
    with PcapReader(str(pcap)) as reader:
        for _ in range(limit):
            try:
                packet = next(reader)
            except StopIteration:
                break
            packets_read += 1
            if Ether not in packet or not any(layer in packet for layer in (IP, IPv6, ARP)):
                skipped += 1
                continue
            timestamp = float(packet.time)
            if not np.isfinite(timestamp):
                raise ValueError("Supported packets must have finite timestamps.")
            if previous_time is not None and timestamp < previous_time:
                # Keep the stateful decay clock monotonic. Account explicitly
                # for every skipped record rather than silently sorting a prefix.
                skipped += 1
                out_of_order += 1
                continue
            previous_time = timestamp
            src_mac, dst_mac = packet[Ether].src, packet[Ether].dst
            if ARP in packet:
                address = packet[ARP]
                ip_type, src_ip, dst_ip = 0, address.psrc, address.pdst
                src_mac, dst_mac = address.hwsrc, address.hwdst
                src_port = dst_port = "arp"
                protocol = "ARP"
            else:
                network = packet[IP] if IP in packet else packet[IPv6]
                ip_type, src_ip, dst_ip = (0 if IP in packet else 1), network.src, network.dst
                transport = packet[TCP] if TCP in packet else packet[UDP] if UDP in packet else None
                if transport is not None:
                    src_port, dst_port = str(transport.sport), str(transport.dport)
                    protocol = "TCP" if TCP in packet else "UDP"
                elif ICMP in packet:
                    src_port = dst_port = "icmp"
                    protocol = "ICMP"
                else:
                    src_port = dst_port = ""
                    protocol = "IP_OTHER"
            values = np.asarray(engine.updateGetStats(
                ip_type, src_mac, dst_mac, src_ip, src_port, dst_ip, dst_port,
                timestamp, len(packet),
            ), dtype=float)
            if values.shape != (100,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid upstream features at packet {packets_read}.")
            rows.append([timestamp, len(packet), protocol, *values.tolist()])
            protocols[protocol] = protocols.get(protocol, 0) + 1
    if not rows:
        raise ValueError("No supported Ethernet IPv4/IPv6/ARP packets found in the selected prefix.")

    # Write only after extraction succeeds; neither raw captures nor upstream
    # sources are modified. Bound memory by the explicit packet limit.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "length", "protocol", *[f"kit_fe_{i}" for i in range(100)]])
        writer.writerows(rows)
    metadata = {
        "input_capture": str(pcap), "input_bytes": pcap.stat().st_size,
        "packet_limit": limit, "packets_read": packets_read,
        "feature_rows": len(rows), "skipped_packets": skipped,
        "out_of_order_packets": out_of_order,
        "feature_count": 100, "protocol_counts": protocols,
        "engine_source": str(source),
        "engine_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "feature_names": {f"kit_fe_{i}": name for i, name in enumerate(headers)},
        "output_csv": str(output),
        "scope": "Original HEM/Kitsune feature extraction only; no trained HEM inference.",
        "label_policy": "No packet labels inferred from capture filename; no accuracy claim.",
        "preprocessing": "Raw features; no fitted scaling, protocol vocabulary or PCA applied.",
        "timestamp_policy": "Skip and count packets older than the previous accepted packet; fresh state per file.",
    }
    with metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
    return metadata
