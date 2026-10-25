"""Freeze offline packet/rule and lexical RAG behavior for the Rust migration.

All packets and evidence below are invented regression inputs. This helper never
starts capture, reads host interfaces, loads a model, trains or generates prose.
Private callback access is intentional: these fixtures characterize the current
Python implementation before any runtime refactor changes its boundaries.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace
from unittest.mock import patch

from framework.data import live_monitor
from framework.detection.live_rules import assess_connections, assess_packets
from framework.rag.grounding import check_grounding
from framework.rag.retrieval import LocalKnowledgeBase


_LIMITATIONS = (
    "Synthetic offline regression fixtures only; no real traffic, capture, model "
    "inference, training or LLM generation. Rule decisions and lexical checks "
    "do not establish detection accuracy, safety or semantic faithfulness."
)
_LOCAL_IP = "192.0.2.10"
_REMOTE_IP = "198.51.100.20"
_LOCAL_MAC = "02:00:00:00:00:10"
_REMOTE_MAC = "02:00:00:00:00:20"


def _write_json(path: Path, document: dict) -> None:
    # Exclusive creation and explicit LF make byte-level regeneration portable.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _packet_cases() -> list[tuple[str, float, list]]:
    from scapy.layers.inet import ICMP, IP, TCP, UDP
    from scapy.layers.inet6 import IPv6
    from scapy.layers.l2 import ARP, Ether

    def tcp(flags="S", port=443):
        return (Ether(src=_LOCAL_MAC, dst=_REMOTE_MAC)
                / IP(src=_LOCAL_IP, dst=_REMOTE_IP)
                / TCP(sport=40000, dport=port, flags=flags))

    mixed = [tcp("S"), tcp("SA"), tcp("A"), tcp("FPRU"),
             Ether(src=_LOCAL_MAC, dst=_REMOTE_MAC) / IP(src=_LOCAL_IP, dst=_REMOTE_IP)
             / UDP(sport=40000, dport=53),
             Ether(src=_LOCAL_MAC, dst=_REMOTE_MAC) / IP(src=_LOCAL_IP, dst=_REMOTE_IP) / ICMP(),
             Ether(src=_LOCAL_MAC, dst=_REMOTE_MAC)
             / IPv6(src="2001:db8::10", dst="2001:db8::20")
             / TCP(sport=40000, dport=443, flags="A"),
             Ether(src=_LOCAL_MAC, dst=_REMOTE_MAC)
             / ARP(hwsrc=_LOCAL_MAC, hwdst=_REMOTE_MAC, psrc=_LOCAL_IP, pdst=_REMOTE_IP),
             Ether(src=_LOCAL_MAC, dst=_REMOTE_MAC, type=0x88B5)]
    unrelated = [Ether(src="02:00:00:00:00:30", dst="02:00:00:00:00:40")
                 / IP(src="203.0.113.30", dst="203.0.113.40")
                 / TCP(sport=40000, dport=443, flags="S"),
                 Ether(src="02:00:00:00:00:30", dst="02:00:00:00:00:40")
                 / ARP(hwsrc="02:00:00:00:00:30", hwdst="02:00:00:00:00:40",
                       psrc="203.0.113.30", pdst="203.0.113.40")]
    return [("mixed_headers", 10.0, mixed),
            ("syn_burst", 2.0, [tcp() for _ in range(100)]),
            ("syn_below_count", 2.0, [tcp() for _ in range(99)]),
            ("syn_below_rate", 2.1, [tcp() for _ in range(100)]),
            ("port_sweep", 10.0, [tcp(port=port) for port in range(1000, 1025)]),
            ("port_below_threshold", 10.0, [tcp(port=port) for port in range(1000, 1024)]),
            ("zero_packets", 10.0, []),
            ("unrelated_packets", 10.0, unrelated)]


def _live_goldens(directory: Path) -> dict:
    from scapy.utils import PcapReader

    packet_cases = _packet_cases()
    pcap_path = directory / "synthetic.pcap"
    # Explicit little-endian, microsecond-resolution Ethernet PCAP. All frames
    # are header-only and use documentation IP ranges and locally assigned MACs.
    with pcap_path.open("xb") as stream:
        stream.write(struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for case_index, (_, seconds, packets) in enumerate(packet_cases):
            epoch = 1704067200 + case_index * 60
            for index, packet in enumerate(packets):
                offset = int((index + 1) * seconds * 1_000_000 / (len(packets) + 1))
                raw = bytes(packet)
                stream.write(struct.pack("<IIII", epoch + offset // 1_000_000,
                                         offset % 1_000_000, len(raw), len(raw)))
                stream.write(raw)
    with PcapReader(str(pcap_path)) as reader:
        replay = list(reader)

    cases, start = [], 0
    for case_index, (identifier, seconds, packets) in enumerate(packet_cases):
        monitor = live_monitor.LiveMonitor()
        monitor._local_addresses = {_LOCAL_IP, "2001:db8::10"}
        monitor._local_macs = {_LOCAL_MAC}
        monitor._window = monitor._empty_window()
        for packet in replay[start:start + len(packets)]:
            # Replace only this module's datetime binding during the callback.
            # No process-wide time function, worker thread or timer is changed.
            timestamp = float(packet.time)
            clock = SimpleNamespace(now=lambda tz, value=timestamp: datetime.fromtimestamp(value, tz))
            with patch.object(live_monitor, "datetime", clock):
                monitor._packet(packet)
        stats = {**monitor._window, "seconds": seconds,
                 "unique_ports_per_source": {key: len(ports) for key, ports in monitor._ports.items()}}
        cases.append({"id": identifier,
                      "input": {"pcap": "live/synthetic.pcap", "packet_start": start,
                                "packet_count": len(packets), "window_seconds": seconds,
                                "window_start_epoch_seconds": 1704067200 + case_index * 60,
                                "local_addresses": sorted(monitor._local_addresses),
                                "local_macs": sorted(monitor._local_macs)},
                      "output": {"stats": stats, "assessment": assess_packets(stats),
                                 "recent_packets": deepcopy(monitor._data["recent_packets"])}})
        start += len(packets)

    connection_cases = []
    for identifier, count, rate in [("idle", 0, 0.0), ("throughput_only", 0, 10000.0),
                                    ("pending_below_threshold", 39, 0.0),
                                    ("pending_at_threshold", 40, 0.0)]:
        connections = [{"status": "SYN_SENT"} for _ in range(count)]
        connection_cases.append({"id": identifier,
                                 "input": {"connections": connections, "packet_rate": rate},
                                 "output": assess_connections(connections, rate)})
    _write_json(directory / "windows.json", {
        "schema_version": 1, "synthetic": True, "limitations": _LIMITATIONS,
        "execution": "Offline PCAP replay through LiveMonitor._packet; no monitor start or model calls",
        "clock_policy": "Recent header times use the corresponding PCAP timestamp in UTC",
        "pcap_sha256": hashlib.sha256(pcap_path.read_bytes()).hexdigest(),
        "pcap_format": {"byte_order": "little", "timestamp_resolution": "microseconds", "link_type": 1},
        "cases": cases, "connection_cases": connection_cases,
    })
    return {"live_case_count": len(cases), "connection_case_count": len(connection_cases),
            "pcap_packet_count": len(replay)}


def _evidence() -> dict:
    return {"source": {"kind": "Synthetic packet observation", "mode": "packets", "synthetic": True},
            "prediction": {"label": "Suspicious traffic pattern", "is_attack": True},
            "xai": {"method": "Exact native TreeSHAP", "features": [
                {"feature": "network_packets_all_count", "value": 100, "shap_value": 0.4},
                {"feature": "network_tcp-flags-syn_count", "value": 20, "shap_value": -0.2}]},
            "rules": None}


def _rag_goldens(directory: Path) -> dict:
    original = Path(__file__).resolve().parents[1] / "framework" / "rag" / "knowledge.json"
    corpus = original.read_bytes()
    with (directory / "knowledge.json").open("xb") as stream:
        stream.write(corpus)
    knowledge = LocalKnowledgeBase(directory)
    if not knowledge.status()["available"]:
        raise ValueError("The checked local knowledge corpus is unavailable; no RAG golden was substituted")

    disagreement = _evidence()
    disagreement["rules"] = {"status": "benign_looking"}
    weak = _evidence()
    weak["prediction"] = {"label": "Quiet activity", "is_attack": False}
    weak["xai"] = {"method": "arbor explanation", "features": [{"feature": "opaque_a", "shap_value": 0.4}]}
    requests = [
        ("packet_disagreement", "retrieve", {"evidence": disagreement}),
        ("csv_scope", "retrieve_query", {"query": "DataSense CSV sensor log features validation", "scope": "csv"}),
        ("weak_supported_evidence", "retrieve", {"evidence": weak}),
        ("unrelated_query", "retrieve_query", {"query": "banana cheesecake lunar agriculture", "scope": "packet"}),
        ("empty_query", "retrieve_query", {"query": "", "scope": "common"}),
        ("unsupported_source", "retrieve", {"evidence": {"source": {"kind": "Synthetic weather record"},
                                                        "prediction": {"label": "Sunny"}}}),
        ("bounded_results", "retrieve_query", {
            "query": "SHAP LIME fidelity gain cover calibrated scores human verification uncertainty",
            "scope": "csv", "max_results": 50}),
    ]
    retrieval_cases = []
    for identifier, method, inputs in requests:
        output = getattr(knowledge, method)(**inputs)
        retrieval_cases.append({"id": identifier, "method": method, "input": inputs, "output": output})
    _write_json(directory / "retrieval.json", {
        "schema_version": 1, "synthetic_inputs": True, "limitations": _LIMITATIONS,
        "knowledge": {"file": "rag/knowledge.json", "source": "framework/rag/knowledge.json",
                      "sha256": hashlib.sha256(corpus).hexdigest(), "status": knowledge.status()},
        "cases": retrieval_cases,
    })

    good = ("The detector predicted Suspicious traffic pattern. The packets all count feature supported "
            "this prediction. The TCP syn count feature opposed this prediction. "
            "This prediction needs human verification and does not confirm an attack or guarantee safety.")
    partial = _evidence()
    partial["feature_coverage"] = {"observed": 7, "missing": 163, "total": 170, "validated_profile": False}
    partial["xai"]["features"][1].update(value=None, observed=False, is_missing=True)
    partial_text = good.replace("The TCP syn count feature opposed",
                               "The unavailable TCP syn count feature uses missing-value default routing and opposed")
    partial_text += " Input is partial and limited; this profile has not been validated."
    cited_passages = retrieval_cases[0]["output"]["passages"]
    citation = cited_passages[0]["id"]
    samples = [
        ("faithful", good, _evidence(), [], "stop"),
        ("valid_citation", good + f" [{citation}]", _evidence(), cited_passages, "stop"),
        ("partial_missing_routing", partial_text, partial, [], "stop"),
        ("rule_disagreement", good + " The rules did not flag this pattern and disagree with the model.", disagreement, [], "stop"),
        ("unsupported_percentage", good + " It is 99.97% certain.", _evidence(), [], "stop"),
        ("wrong_label", good.replace("Suspicious traffic pattern", "Normal traffic"), _evidence(), [], "stop"),
        ("reversed_direction", good.replace("count feature supported", "count feature opposed"), _evidence(), [], "stop"),
        ("invented_citation", good + " [invented-evidence]", _evidence(), cited_passages, "stop"),
        ("missing_routing_warning", good + " Input is partial and limited; this profile has not been validated.", partial, [], "stop"),
        ("missing_disagreement", good, disagreement, [], "stop"),
        ("truncation", good, _evidence(), [], "length"),
        ("missing_uncertainty", good.split("This prediction needs")[0], _evidence(), [], "stop"),
        ("instruction_echo", "Rewrite the facts. " + good, _evidence(), [], "stop"),
    ]
    grounding_cases = []
    for identifier, text, evidence, passages, finish_reason in samples:
        inputs = {"text": text, "evidence": evidence, "passages": passages, "finish_reason": finish_reason}
        grounding_cases.append({"id": identifier, "input": inputs, "output": check_grounding(**inputs)})
    _write_json(directory / "grounding.json", {
        "schema_version": 1, "synthetic_inputs": True, "limitations": _LIMITATIONS,
        "input_note": "Handwritten prose and fabricated attribution values; no model or LLM executed",
        "cases": grounding_cases,
    })
    return {"retrieval_case_count": len(retrieval_cases), "grounding_case_count": len(grounding_cases)}


def generate_observation_goldens(output_dir: Path) -> dict:
    """Write owned live/ and rag/ children without overwriting any existing child.

    The orchestrator may already have created output_dir for other feature files.
    Call serially in a dedicated generator process because packet replay briefly
    patches the live-monitor module's datetime binding for deterministic headers.
    """
    output_dir = Path(output_dir)
    owned = [output_dir / name for name in ("live", "rag")]
    for directory in owned:
        if directory.exists() or directory.is_symlink():
            raise FileExistsError(f"Refusing to overwrite observation fixture child: {directory.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in owned:
        directory.mkdir()
    live = _live_goldens(owned[0])
    rag = _rag_goldens(owned[1])
    return {"schema_version": 1, "synthetic": True, "limitations": _LIMITATIONS,
            "files": ["live/synthetic.pcap", "live/windows.json", "rag/knowledge.json",
                      "rag/retrieval.json", "rag/grounding.json"], **live, **rag}
