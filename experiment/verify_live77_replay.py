"""Offline replay parity for the live77 model row (integration plan step 5).

The native ``feature_replay`` example (Rust extractor, ``Live77Features``) and the Python reference
(``framework/training/csv_v3/live77_reference.py`` Layer B) project the same frames to the 252 live77
inputs; every input must agree exactly (identical NaN positions, equal float32 values). Inputs:

* the shared hand-built fixtures ``evaluation/fixtures/live77/frames_v1.json``, also against their
  hand-computed expected vectors;
* the bounded first ten-second window of the real benign PCAP used by ``verify_live_feature_replay.py``
  (at most 192 transient bytes per frame; only aggregates are reported, no frame bytes are saved).

No capture is started, no model is executed, no dataset partition is opened. Parity is with the frozen
contract, not with the unpublished DataSense exporter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.verify_live_feature_replay import DEFAULT_PCAP, read_pcap_window  # noqa: E402
from framework.training.csv_v3 import live77_reference as ref  # noqa: E402

# Built with CARGO_TARGET_DIR=artifacts/app_v0_6_20261102/cargo-target (never rust/netsec/target).
DEFAULT_BINARY = ROOT / "artifacts/app_v0_6_20261102/cargo-target/debug/examples" / (
    "feature_replay.exe" if sys.platform == "win32" else "feature_replay")
FIXTURES = ROOT / "evaluation/fixtures/live77/frames_v1.json"
DEFAULT_REPORT = ROOT / "artifacts/app_v0_6_20261102/live77_replay_report.json"
# Device MAC of the real-PCAP window, as in verify_live_feature_replay.run_verification.
PCAP_DEVICE_MAC = "a2:bd:fa:b5:89:92"
PCAP_RECORDS_ANCHOR = 220


def expected_vector(spec: dict) -> np.ndarray:
    """Sparse fixture vector: default (0; null = NaN), then nan_prefixes, then nan names, then values."""
    d = spec.get("default", 0.0)
    out = np.full(len(ref.NAMES), np.nan if d is None else d, dtype=np.float32)
    for p in spec.get("nan_prefixes", []):
        out[[i for i, n in enumerate(ref.NAMES) if n.startswith(p)]] = np.nan
    for n in spec.get("nan", []):
        out[ref.NAMES.index(n)] = np.nan
    for n, v in spec.get("values", {}).items():
        out[ref.NAMES.index(n)] = np.nan if v is None else np.float32(v)
    return out


def request(frames: list[dict], device_mac: str, capture_drop: bool) -> dict[str, Any]:
    """Replay input for the native example; frames are ``{"t", "hex", "wire_length"?}``."""
    return {"frames": [{"hex": f["hex"], "original_length": int(f.get("wire_length", len(f["hex"]) // 2)),
                        "timestamp_ns": int(round(f["t"] * 1e9))} for f in frames],
            "local_ips": [], "local_mac": device_mac, "seconds": ref.NOMINAL_SECONDS,
            "capture_drops": int(capture_drop), "live77_names": ref.NAMES}


def native_row(req: dict[str, Any], binary: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if not binary.is_file():
        raise FileNotFoundError(f"Native offline feature replay binary is required: {binary}")
    out = json.loads(subprocess.run([str(binary)], input=json.dumps(req), text=True, capture_output=True,
                                    check=True, timeout=60).stdout)
    live = out["live77"]
    if live["names"] != ref.NAMES or live["row"] is None:
        raise RuntimeError(f"native live77 row unavailable: {live.get('error')}")
    row = np.array([np.nan if v is None else v for v in live["row"]], dtype=np.float32)
    return row, out


def differences(got: np.ndarray, want: np.ndarray) -> dict[str, list]:
    bad = {}
    for n, x, y in zip(ref.NAMES, got, want):
        if not ((np.isnan(x) and np.isnan(y)) or x == y):
            bad[n] = [None if np.isnan(x) else float(x), None if np.isnan(y) else float(y)]
    return bad


def compare(frames: list[dict], device_mac: str, capture_drop: bool, binary: Path,
            expected: np.ndarray | None = None) -> dict[str, Any]:
    req = request(frames, device_mac, capture_drop)
    native, out = native_row(req, binary)
    reference = ref.frames_vector(frames, device_mac, capture_drop=capture_drop)
    failures = {"native_vs_reference": differences(native, reference)}
    if expected is not None:
        failures["native_vs_fixture"] = differences(native, expected)
        failures["reference_vs_fixture"] = differences(reference, expected)
    if out.get("offline_replay") is not True or out.get("capture_performed") is not False:
        failures["provenance"] = {"offline_replay": out.get("offline_replay"), "capture_performed": out.get("capture_performed")}
    if out["live77"].get("publisher_equivalence_verified") is not False:
        failures["publisher_equivalence"] = {"must_be": False}
    failures = {k: v for k, v in failures.items() if v}
    facts = ref.window_facts(frames, device_mac, capture_drop=capture_drop)
    return {"passed": not failures, "failures": failures, "frames": len(frames),
            "admitted_frames_native": out["live77"]["admitted_frames"],
            "admitted_frames_reference": None if facts.get("capture_drop") else facts["packets"]["all"],
            "missing_inputs": int(np.isnan(native).sum()), "compared_inputs": len(ref.NAMES)}


def pcap_frames(pcap: Path) -> tuple[list[dict], dict[str, Any]]:
    """The bounded first 10 s window as reference frames; file order is arrival order (``t`` never decreases)."""
    frames, source = read_pcap_window(pcap)
    start, last, clamped, out = source["window_start_ns"], 0.0, 0, []
    for f in frames:
        t = (f["timestamp_ns"] - start) / 1e9
        if t < last:
            clamped, t = clamped + 1, last
        last = t
        out.append({"t": t, "hex": f["hex"], "wire_length": f["original_length"]})
    source["timestamps_clamped_to_file_order"] = clamped
    return out, source


def run(binary: Path, pcap: Path | None) -> dict[str, Any]:
    fx = json.loads(FIXTURES.read_text(encoding="utf-8"))
    if fx["contract_feature_names_sha256"] != ref.CONTRACT["feature_names_sha256"]:
        raise RuntimeError("fixture file does not follow the frozen contract")
    cases = {f["id"]: compare(f["frames"], fx["device"]["mac"], bool(f.get("capture_drop")), binary,
                              expected_vector(f["expected"])) for f in fx["fixtures"]}
    real = None
    if pcap is not None:
        frames, source = pcap_frames(pcap)
        real = compare(frames, PCAP_DEVICE_MAC, False, binary)
        if source["records_read_including_boundary"] != PCAP_RECORDS_ANCHOR:
            real["failures"]["anchor"] = {"records_read_including_boundary": source["records_read_including_boundary"]}
            real["passed"] = False
        source["path"] = str(pcap.relative_to(ROOT)) if pcap.is_relative_to(ROOT) else str(pcap)
        real["source"] = source
        real["device_mac_note"] = "device MAC of the UQIOT2022 benign capture as used by verify_live_feature_replay.py"
    results = [*cases.values(), *([real] if real else [])]
    return {"schema_version": "live77-native-replay-parity-v1",
            "passed": all(r["passed"] for r in results),
            "rule": "native Rust live77 row == Python reference Layer B (and the fixture's hand-computed vector): identical NaN mask, equal float32",
            "contract": "framework/training/csv_v3/live77_contract.json",
            "contract_feature_names_sha256": ref.CONTRACT["feature_names_sha256"],
            "fixture_cases": cases, "real_pcap": real,
            "native_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limits": {"publisher_equivalence_verified": False, "capture_performed": False,
                       "model_inference_performed": False, "dataset_partitions_opened": False,
                       "locked_test_read": False, "frame_bytes_saved": False,
                       "pending_owner_conventions": ref.CONTRACT["pending_owner_signoff"]["conventions"],
                       "pcap_loss_note": "Classic PCAP has no drop telemetry; capture_drop=False is an explicit replay assumption"}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--pcap", type=Path, default=DEFAULT_PCAP)
    parser.add_argument("--fixtures-only", action="store_true")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error(f"Native offline replay binary required; build it explicitly first: {args.binary}")
    report = run(args.binary, None if args.fixtures_only else args.pcap)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "fixture_cases": len(report["fixture_cases"]),
                      "fixture_failures": [k for k, v in report["fixture_cases"].items() if not v["passed"]],
                      "real_pcap_checked": report["real_pcap"] is not None,
                      "real_pcap_passed": None if report["real_pcap"] is None else report["real_pcap"]["passed"],
                      "real_pcap_admitted_frames": None if report["real_pcap"] is None else report["real_pcap"]["admitted_frames_native"],
                      "report": str(args.report)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
