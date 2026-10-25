"""Measure isolated Rust Windows enumeration against real psutil snapshots.

Only OS/Npcap device enumeration and existing synthetic PCAP replay run. There is
no passive capture, transmission, model inference, training or DLL installation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
from pathlib import Path
import platform
import socket
import subprocess
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]


def snapshot() -> dict:
    started = time.perf_counter()
    stats = psutil.net_if_stats()
    counters = psutil.net_io_counters(pernic=True)
    adapters = []
    for name, values in psutil.net_if_addrs().items():
        ips = sorted({str(ipaddress.ip_address(v.address.split("%")[0]))
                      for v in values if v.family in (socket.AF_INET, socket.AF_INET6)})
        adapters.append({"name": name, "addresses": ips,
                         "is_up": bool(stats.get(name) and stats[name].isup),
                         "has_counter": name in counters})
    adapter_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    connections, error = [], None
    try:
        for value in psutil.net_connections(kind="inet"):
            if not value.laddr:
                continue
            connections.append({
                "family": "IPv4" if value.family == socket.AF_INET else "IPv6",
                "protocol": "TCP" if value.type == socket.SOCK_STREAM else "UDP",
                "local_ip": str(ipaddress.ip_address(value.laddr.ip.split("%")[0])),
                "local_port": value.laddr.port,
                "remote_ip": str(ipaddress.ip_address(value.raddr.ip.split("%")[0])) if value.raddr else None,
                "remote_port": value.raddr.port if value.raddr else None,
                "status": value.status, "pid": value.pid or 0,
            })
    except (psutil.AccessDenied, OSError) as exception:
        error = f"{type(exception).__name__}: {exception}"
    return {"adapters": sorted(adapters, key=lambda v: v["name"]),
            "connections": sorted(connections, key=lambda v: json.dumps(v, sort_keys=True)),
            "connection_error": error, "timings_ms": {"adapters": adapter_ms,
                "connections": (time.perf_counter() - started) * 1000}}


def key(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def compare(before: dict, rust: dict, after: dict) -> dict:
    native = {v["name"]: v for v in rust["adapters"]}
    python = {v["name"]: v for v in before["adapters"]}
    details = []
    for name in sorted(native.keys() & python.keys()):
        a, b = native[name], python[name]
        details.append({"name": name, "addresses_equal": a["addresses"] == b["addresses"],
                        "is_up_equal": a["is_up"] == b["is_up"],
                        "native_only_addresses": sorted(set(a["addresses"]) - set(b["addresses"])),
                        "psutil_only_addresses": sorted(set(b["addresses"]) - set(a["addresses"]))})
    stable_adapters = [{k: v[k] for k in ("name", "addresses", "is_up")} for v in before["adapters"]] == [
        {k: v[k] for k in ("name", "addresses", "is_up")} for v in after["adapters"]]
    a = Counter(key(v) for v in before["connections"])
    b = Counter(key(v) for v in rust["connections"])
    c = Counter(key(v) for v in after["connections"])
    stable = a & c
    return {
        "adapters": {"native_count": len(native), "psutil_count": len(python),
            "common_count": len(details), "native_only_names": sorted(native.keys() - python.keys()),
            "psutil_only_names": sorted(python.keys() - native.keys()), "details": details,
            "psutil_before_after_equal": stable_adapters,
            "all_common_addresses_equal": all(v["addresses_equal"] for v in details),
            "all_common_is_up_equal": all(v["is_up_equal"] for v in details)},
        "connections": {"native_count": b.total(), "psutil_before_count": a.total(),
            "psutil_after_count": c.total(), "equal_before": a == b, "equal_after": c == b,
            "common_before_count": (a & b).total(), "common_after_count": (b & c).total(),
            "stable_before_after_count": stable.total(), "stable_missing_native_count": (stable - b).total(),
            "native_only_during_middle_snapshot_count": (b - (a | c)).total(),
            "native_missing_before": list((b - a).elements()), "before_missing_native": list((a - b).elements()),
            "native_missing_after": list((b - c).elements()), "after_missing_native": list((c - b).elements()),
            "psutil_errors": [v for v in (before["connection_error"], after["connection_error"]) if v]},
    }


def run(binary: Path, args: list[str]) -> tuple[dict, float]:
    started = time.perf_counter()
    result = subprocess.run([str(binary), *args], capture_output=True, text=True, check=True, timeout=30)
    if len(result.stdout) > 32 * 1024 * 1024:
        raise ValueError("Spike output exceeds 32 MiB")
    return json.loads(result.stdout), (time.perf_counter() - started) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "rust/spikes/capture/target/release/netsec-capture-spike.exe")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/rust_migration/phase1/capture")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if platform.system() != "Windows":
        raise RuntimeError("Actual native Windows comparison cannot execute on this platform")
    if not 1 <= args.repeats <= 5:
        raise ValueError("Choose 1..5 bounded snapshots")
    binary = args.binary.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for index in range(args.repeats):
        before = snapshot()
        native, process_ms = run(binary, ["probe"])
        after = snapshot()
        cases.append({"id": index, "psutil_before": before, "native": native,
                      "psutil_after": after, "comparison": compare(before, native, after), "process_ms": process_ms})
    missing, missing_ms = run(binary, ["probe", "--simulate-missing-driver"])
    replay, replay_ms = run(binary, ["replay", str(ROOT / "evaluation/fixtures/rust_migration/v1/live/synthetic.pcap"),
                                   str(ROOT / "evaluation/fixtures/rust_migration/v1/live/windows.json")])
    missing_ok = (not missing["npcap"]["packet_capture_available"]
                  and missing["npcap"]["status"] == "not_installed"
                  and not missing["npcap"]["capture_started"] and not missing["npcap"]["devices"])
    enumeration_ok = all(
        c["comparison"]["adapters"]["all_common_addresses_equal"]
        and c["comparison"]["adapters"]["all_common_is_up_equal"]
        and not c["comparison"]["adapters"]["native_only_names"]
        and not c["comparison"]["adapters"]["psutil_only_names"]
        and not c["comparison"]["connections"]["psutil_errors"]
        and c["comparison"]["connections"]["stable_missing_native_count"] == 0
        for c in cases)
    verdict = "go_enumeration_only" if enumeration_ok and missing_ok and replay["all_equal"] else "no_go_unresolved_parity"
    report = {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "psutil_version": psutil.__version__,
        "binary_bytes": binary.stat().st_size, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "cases": cases, "missing_driver_fault": {"result": missing["npcap"], "passed": missing_ok, "process_ms": missing_ms},
        "replay": {"result": replay, "process_ms": replay_ms}, "verdict": verdict,
        "not_executed": ["Opening a live Npcap capture handle / start-time permission check", "Passive live headers",
            "Full live lifecycle/window orchestration", "Model inference or any labelled accuracy evaluation", "Linux runtime", "Native crash containment"],
        "limitations": ["Sequential OS snapshots race changing endpoints; stable-before/after intersection is a diagnostic, not an atomic oracle.",
            "Both psutil and the Rust adapter read Windows IP Helper tables; agreement is compatibility evidence, not independent ground truth.",
            "IPv6 scope suffixes are normalized away as in Python live monitoring. Process names are not collected; PID permission restrictions remain.",
            "Native normalized rows are deduplicated to match the installed psutil Windows set-returning wrapper; raw UDP binding multiplicity is not preserved in psutil's public contract.",
            "Npcap enumeration success does not establish capture permissions, packet delivery or driver crash isolation.",
            "Synthetic replay covers the eight existing header fixtures only; no accuracy or complete parser claim."]}
    destination = args.output / "report.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(destination), "verdict": verdict, "repeat_count": len(cases),
                      "replay_equal": replay["all_equal"], "missing_driver_fault_passed": missing_ok}))


if __name__ == "__main__":
    main()
