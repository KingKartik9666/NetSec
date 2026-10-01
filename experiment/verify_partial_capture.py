"""Independent partial-model replay and opt-in Windows capture verification.

Default: offline replay of the existing benign-source UQ-IoT PCAP. Pass
``--mode both --interface Wi-Fi`` to explicitly start a bounded passive capture.
Every invocation writes a new proof directory; old proofs are never replaced.
These are execution checks on unlabelled observations, never accuracy results.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
from pathlib import Path
import platform
import time
import uuid

import numpy as np

from framework.app.labels import display_label
from framework.data.feature_schema import packet_vector, vector_csv
from framework.data.live_monitor import LiveMonitor
from framework.detection.packet_xgboost import FEATURE_NAMES, STATS_KEYS
from framework.detection.trained_xgboost import XGBoostDetector, default_bundle, file_sha256, transform_csv

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "external/HEM/UQIOT2022/UQIOT2022/data/benign_samples/whole_week.pcap"
OLD_PROOF = ROOT / "artifacts/live_verification/pcap_replay_confirmed.json"
SCOPE = "Unlabelled execution evidence only; no incident confirmation, live accuracy or held-out test evaluation."


def header_oracle(packet, addresses, macs):
    """Count host-relevant TCP header bits independently of the capture callback."""
    from scapy.layers.inet import IP, TCP
    from scapy.layers.inet6 import IPv6
    from scapy.layers.l2 import ARP, Ether
    ip = packet.getlayer(IP) or packet.getlayer(IPv6)
    arp, ether = packet.getlayer(ARP), packet.getlayer(Ether)
    endpoints = {ip.src.split("%")[0], ip.dst.split("%")[0]} if ip is not None else set()
    endpoints |= {arp.psrc, arp.pdst} if arp is not None else set()
    hardware = {str(ether.src).lower(), str(ether.dst).lower()} if ether is not None else set()
    if not (endpoints & addresses or hardware & macs):
        return None
    tcp = packet.getlayer(TCP)
    bits = int(tcp.flags) if tcp is not None else 0
    return {"packets": 1, "bytes": len(packet), "ack": int(bool(bits & 16)),
            "fin": int(bool(bits & 1)), "psh": int(bool(bits & 8)), "rst": int(bool(bits & 4)),
            "syn_all": int(bool(bits & 2)), "urg": int(bool(bits & 32)),
            "syn_ack": int((bits & 18) == 18), "syn": int(bool(bits & 2) and not bool(bits & 16))}


class VerifiedMonitor(LiveMonitor):
    """Observe production callbacks; retain counters only, never packet objects."""
    def __init__(self, detector):
        super().__init__(detector=detector)
        self._oracle_windows = deque(maxlen=4)
        self.completed_evidence = {}

    def _empty_window(self):
        window = super()._empty_window()
        self._oracle_windows.append((window, Counter()))
        return window

    def _packet(self, packet):
        with self._lock:
            contribution = header_oracle(packet, self._local_addresses, self._local_macs)
            super()._packet(packet)
            if contribution is not None:
                oracle = next(counter for window, counter in self._oracle_windows if window is self._window)
                oracle.update(contribution)

    def _predict_multiclass_window(self, stats, observation_id, observed_at):
        with self._lock:
            completed = [(window, counter) for window, counter in self._oracle_windows if window is not self._window]
            if not completed:
                raise AssertionError("No completed callback window was retained for the independent oracle")
            window, counter = completed[-1]
            for key in self._empty_count_keys():
                if stats[key] != window[key] or stats[key] != counter[key]:
                    raise AssertionError(f"Independent live header oracle disagrees for {key}")
            evidence = {"stats": dict(stats), "oracle": dict(counter), "seven_count_header_oracle_match": True}
        report = super()._predict_multiclass_window(stats, observation_id, observed_at)
        self.completed_evidence[observation_id] = evidence
        while len(self.completed_evidence) > 4:
            del self.completed_evidence[next(iter(self.completed_evidence))]
        return report

    @staticmethod
    def _empty_count_keys():
        return [*STATS_KEYS, "bytes", "syn", "syn_ack"]


def private_endpoint(packet):
    from scapy.layers.inet import IP
    from scapy.layers.inet6 import IPv6
    from scapy.layers.l2 import ARP, Ether
    ip = packet.getlayer(IP) or packet.getlayer(IPv6)
    arp, ether = packet.getlayer(ARP), packet.getlayer(Ether)
    pairs = [(ip.src, "src"), (ip.dst, "dst")] if ip is not None else (
        [(arp.psrc, "src"), (arp.pdst, "dst")] if arp is not None else [])
    for value, direction in pairs:
        address = ipaddress.ip_address(value.split("%")[0])
        if address.is_private and not (address.is_loopback or address.is_multicast or address.is_unspecified):
            return str(address), str(getattr(ether, direction)).lower() if ether is not None else None
    return None


def verify_report(detector, report, stats, directory, prefix):
    """Check the same observation through live vectors, CSV and a reloaded model."""
    import xgboost as xgb
    values = detector.validate_features(report["features"])
    np.testing.assert_equal(values, packet_vector(stats, detector.feature_names))
    content = vector_csv(values, detector.feature_names)
    imported = transform_csv(content, detector.bundle)
    np.testing.assert_equal(imported, values.reshape(1, -1))
    expected = detector.predict_batch(imported)[0]
    assert report["prediction"]["label"] == expected["label"], "CSV prediction label differs"
    assert report["prediction"]["score"] == expected["score"], "CSV prediction score differs"
    assert report["prediction"]["top_classes"] == expected["top_classes"], "CSV top-three ordering differs"
    reloaded = xgb.Booster(model_file=str(detector.checkpoint))
    reloaded.set_param({"device": "cpu", "nthread": 2})
    matrix = xgb.DMatrix(imported, nthread=2)
    scores = reloaded.predict(matrix, strict_shape=True)[0]
    original_scores = detector.booster.predict(matrix, strict_shape=True)[0]
    np.testing.assert_array_equal(scores, original_scores)
    assert np.isfinite(scores).all() and np.isclose(scores.sum(), 1, atol=2e-5)
    explanation = report["explanation"]
    assert len(explanation["all_features"]) == len(detector.feature_names)
    total = explanation["bias"] + sum(row["shap_value"] for row in explanation["all_features"])
    assert np.isfinite(total) and np.isclose(total, explanation["margin"], atol=1e-4, rtol=1e-5)
    assert sum(value is None for value in report["features"]) == len(values) - 7
    assert report["feature_coverage"]["observed"] == 7
    assert report["feature_coverage"]["profile"] == "packet_counts_7"
    with (directory / f"{prefix}_features.csv").open("xb") as stream:
        stream.write(content)
    return {"vector_csv_round_trip_equal": True, "checkpoint_predictions_equal": True,
            "shap_additivity_verified": True, "shap_reconstruction_error": abs(total-explanation["margin"]),
            "cpu_inference": True, "observed_count": 7, "missing_count": len(values)-7,
            "feature_csv": f"{prefix}_features.csv", "feature_csv_sha256": hashlib.sha256(content).hexdigest(),
            "top_three": [{**entry, "display_label": display_label(entry["label"])} for entry in expected["top_classes"]]}


def verify_replay(detector, directory, capture=CAPTURE):
    # Register Ethernet/IP dissectors before PcapReader decodes its first frame.
    # Importing only scapy.utils can otherwise silently yield Raw packets in a
    # fresh process that has never imported any packet layers.
    from scapy.all import PcapReader
    if not capture.is_file():
        return {"status": "unavailable", "reason": "Existing real UQ-IoT PCAP was not found", "capture_started": False}
    monitor = LiveMonitor(detector=detector)
    monitor._window = monitor._empty_window()
    monitor._data["interface"] = "Offline recorded endpoint; no Windows adapter capture"
    oracle = Counter()
    digest = hashlib.sha256()
    endpoint = mac = started = boundary = last = None
    packets_read = 0
    with PcapReader(str(capture)) as reader:
        for packets_read, packet in enumerate(reader, 1):
            if packets_read > 1_000_000:
                raise RuntimeError("Replay read bound reached before a complete ten-second window")
            stamp = float(packet.time)
            if endpoint is None:
                selected = private_endpoint(packet)
                if selected is None:
                    continue
                endpoint, mac = selected
                started = stamp
                monitor._local_addresses = {endpoint}
                monitor._local_macs = {mac} if mac else set()
            if last is not None and stamp < last:
                raise ValueError("Replay timestamps are out of order")
            if stamp >= started + 10:
                boundary = stamp
                break
            last = stamp
            counted = header_oracle(packet, monitor._local_addresses, monitor._local_macs)
            monitor._packet(packet)
            if counted is not None:
                oracle.update(counted)
                digest.update(json.dumps([stamp, counted], sort_keys=True).encode("utf-8"))
    if boundary is None or not oracle["packets"]:
        raise ValueError("No complete nonempty ten-second replay window was available")
    stats = {**monitor._window, "seconds": 10.0}
    assert all(stats[key] == oracle[key] for key in VerifiedMonitor._empty_count_keys())
    observation_id = "pcap-replay-" + uuid.uuid4().hex
    observed_at = datetime.now(timezone.utc).isoformat()
    report = monitor._predict_multiclass_window(stats, observation_id, observed_at)
    if report is None:
        raise ValueError("The promoted model did not produce a replay prediction")
    source_kind = ("Existing real benign-source UQ-IoT recording; per-packet labels unverified"
                   if capture.resolve() == CAPTURE.resolve() else "Supplied PCAP recording; provenance and per-packet labels unverified")
    report["source"].update(kind=source_kind, mode="pcap_replay")
    binary = monitor._predict_window(stats)
    binary.update(observation_id=observation_id, observed_at=observed_at)
    checks = verify_report(detector, report, stats, directory, "replay")
    return {"status": "passed", "source_kind": source_kind,
            "capture": str(capture.relative_to(ROOT)) if capture.is_relative_to(ROOT) else str(capture),
            "capture_bytes": capture.stat().st_size, "capture_started": False, "payload_stored": False,
            "endpoint": endpoint, "endpoint_mac": mac, "capture_start_seconds": started,
            "capture_end_seconds": started+10, "boundary_packet_seconds": boundary, "packets_read": packets_read,
            "stats": stats, "oracle": dict(oracle), "seven_count_header_oracle_match": True,
            "header_facts_sha256": digest.hexdigest(), "checksum_scope": "Selected header/count facts, not payloads or full PCAP",
            "observation_id": observation_id, "multiclass_prediction": report,
            "binary_comparison": binary, "checks": checks, "scope": SCOPE}


def verify_live(detector, directory, interface="Wi-Fi", timeout=45):
    if platform.system() != "Windows":
        return {"status": "unavailable", "reason": "This check specifically requires Windows Npcap", "capture_started": False}
    monitor = VerifiedMonitor(detector)
    capabilities = monitor.capabilities()
    if not capabilities["packet_capture_available"]:
        return {"status": "unavailable", "reason": capabilities["packet_capture_message"], "capture_started": False}
    result = {"status": "failed", "interface": interface, "capture_requested": False,
              "capture_started": False, "payload_stored": False, "scope": SCOPE}
    try:
        result["capture_requested"] = True
        monitor.start(interface, "packets")
        deadline = time.monotonic() + timeout
        snapshot = None
        while time.monotonic() < deadline:
            snapshot = monitor.snapshot()
            if snapshot["state"] == "monitoring":
                result["capture_started"] = True
            if snapshot.get("error"):
                raise RuntimeError(snapshot["error"])
            if snapshot.get("multiclass_prediction"):
                break
            time.sleep(.1)
        if not snapshot or not snapshot.get("multiclass_prediction"):
            raise RuntimeError("No completed nonempty model observation arrived within the capture bound")
        report = snapshot["multiclass_prediction"]
        observation_id = report["observation_id"]
        measured = monitor.completed_evidence[observation_id]
        binary = snapshot["model_prediction"]
        if binary is not None:
            assert binary["observation_id"] == observation_id
        checks = verify_report(detector, report, measured["stats"], directory, "windows_live")
        result.update(status="passed", observation_id=observation_id, completed_window_seconds=report["observed_window_seconds"],
                      measured_headers=measured, multiclass_prediction=report, binary_comparison=binary,
                      rule_assessment=snapshot["assessment"], checks=checks)
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        stopped = monitor.stop()
        cleared = (not stopped["running"] and stopped["model_prediction"] is None
                   and stopped["multiclass_prediction"] is None and stopped["assessment"]["status"] == "not_assessed")
        result.update(stop_clears_predictions=cleared, stopped_state=stopped["state"])
        if not cleared:
            result.update(status="failed", stop_error=stopped.get("error"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "live", "both"), default="replay")
    parser.add_argument("--interface", default="Wi-Fi")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--capture", type=Path, default=CAPTURE)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/partial_verification")
    args = parser.parse_args(argv)
    if not 15 <= args.timeout <= 120:
        parser.error("Capture timeout must be 15 to 120 seconds")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    directory = args.output_root / run_id
    directory.mkdir(parents=True, exist_ok=False)
    prior_hash = file_sha256(OLD_PROOF) if OLD_PROOF.is_file() else None
    proof = {"verified_utc": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
             "test_evaluated": False, "live_validation": False, "payload_stored": False, "scope": SCOPE,
             "source_code_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in [
                 Path(__file__), ROOT / "framework/data/live_monitor.py", ROOT / "framework/data/feature_schema.py",
                 ROOT / "framework/detection/trained_xgboost.py", ROOT / "framework/xai/tree_shap.py"]}}
    try:
        selected = default_bundle()
        bundle = json.loads(selected.read_text(encoding="utf-8"))
        if not (bundle.get("schema_version") == 2 and bundle.get("partial_input_supported") is True
                and bundle.get("promoted") is True and bundle.get("promotion", {}).get("improved") is True):
            proof.update(status="unavailable", reason="No promoted partial-input schema-2 bundle is available; no substitute was executed")
        else:
            detector = XGBoostDetector(selected)
            if len(detector.feature_names) != 170 or len(detector.classes) != 84:
                raise ValueError("Promoted model does not match the 170-feature, 84-class study contract")
            proof.update(bundle=str(selected.relative_to(ROOT)), checkpoint_sha256=bundle["checkpoint_sha256"],
                         model_version=bundle["model"].get("version"), feature_names=detector.feature_names)
            for stage in (("replay", "live") if args.mode == "both" else (args.mode,)):
                try:
                    proof[stage] = (verify_replay(detector, directory, args.capture) if stage == "replay"
                                    else verify_live(detector, directory, args.interface, args.timeout))
                except Exception as error:
                    proof[stage] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            states = [proof[stage]["status"] for stage in ("replay", "live") if stage in proof]
            proof["status"] = "passed" if all(state == "passed" for state in states) else "failed" if "failed" in states else "unavailable"
    except Exception as error:
        proof.update(status="failed", error=f"{type(error).__name__}: {error}")
    proof["previous_replay_proof_preserved"] = prior_hash == (file_sha256(OLD_PROOF) if OLD_PROOF.is_file() else None)
    proof["previous_replay_proof_sha256"] = prior_hash
    if not proof["previous_replay_proof_preserved"]:
        proof.update(status="failed", error="The original replay proof changed during verification")
    with (directory / "verification.json").open("x", encoding="utf-8") as stream:
        json.dump(proof, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": proof["status"], "proof": str(directory / "verification.json"),
                      "reason": proof.get("reason"), "error": proof.get("error")}))
    return 0 if proof["status"] == "passed" else 2 if proof["status"] == "unavailable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
