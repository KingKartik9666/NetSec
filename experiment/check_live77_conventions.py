"""Read-only dataset checks behind the live77 extraction contract's "dataset_verified" evidence.

Run: .venv/Scripts/python.exe -m experiment.check_live77_conventions
Writes artifacts/app_v0_6_20261102/live77_convention_checks.json, which experiment/build_live77_contract.py reads.

Opens only the files in ``ALLOWED``. The held-out TEST partition is never opened, and any path that names one is
refused. Every convention check uses the 30,000 validation-partition rows of csv_v3_cascade_20261030_v1 (all
validation roles). One fact is read from training rows: the window_duration_seconds column of the fit77 rows the
final model was fit on, because the provenance note makes a claim about the training windows. Nothing is fitted or
scored.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "dataset/experiments/csv_v3_cascade_20261030_v1"
SCHEMA = ROOT / "artifacts/training/live77_20261031_v1/jobs/77_live_final/attempt_0/model/model_schema.json"
OUT = ROOT / "artifacts/app_v0_6_20261102/live77_convention_checks.json"
ALLOWED = {D / n for n in ("manifest.json", "X_validation.npy", "row_ids_validation.npy", "validation_roles.npy",
                           "selected_validation_77class.csv.gz", "X_train.npy", "fit77_train.npy")} | {SCHEMA}
FAMILIES = ("header-length", "ip-flags", "ip-length", "mss", "tcp-flags", "ttl", "window-size")
STATS = ("avg", "max", "min", "std_deviation")
FLAGS = ("ack", "fin", "psh", "rst", "syn", "urg")
opened: list[str] = []


def _path(p: Path) -> Path:
    if re.search(r"(^|[^a-z])test", p.name, re.I) or p not in ALLOWED:
        raise PermissionError(f"refused (not in the read-only allow-list, or names a test partition): {p}")
    opened.append(p.relative_to(ROOT).as_posix())
    return p


def _list(s: str) -> list[str]:
    return json.loads(s.replace("'", '"')) if s else []


def main() -> dict:
    names = json.loads(_path(D / "manifest.json").read_text(encoding="utf-8"))["feature_names"]
    schema = json.loads(_path(SCHEMA).read_text(encoding="utf-8"))["feature_names"]
    col = {n: i for i, n in enumerate(names)}
    X = np.asarray(np.load(_path(D / "X_validation.npy"), mmap_mode="r"))
    ids = np.load(_path(D / "row_ids_validation.npy"))
    roles = np.load(_path(D / "validation_roles.npy"))
    n = len(X)
    assert X.shape == (n, 445) and len(ids) == len(roles) == n
    g = lambda name: X[:, col[name]]  # noqa: E731
    vocab = {d: [int(s.rsplit("_", 1)[1]) for s in schema if s.startswith(f"network_ports_{d}__present_")
                 and not s.endswith("UNSEEN")] for d in ("src", "dst")}

    # Raw list columns, aligned to X_validation by source_row_id.
    pos = {int(r): i for i, r in enumerate(ids)}
    assert len(pos) == n
    raw: dict[str, list] = {k: [None] * n for k in ("ports_src", "ports_dst", "macs_src", "device_mac", "protocols_all",
                                                    "ips_all")}
    cnt = {d: np.zeros(n, dtype=np.int64) for d in ("src", "dst")}
    with gzip.open(_path(D / "selected_validation_77class.csv.gz"), "rt", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        seen = 0
        for row in rd:
            i = pos[int(row["source_row_id"])]
            seen += 1
            for d in ("src", "dst"):
                raw[f"ports_{d}"][i] = [int(p) for p in _list(row[f"network_ports_{d}"])]
                cnt[d][i] = int(float(row[f"network_ports_{d}_count"]))
            raw["macs_src"][i] = _list(row["network_macs_src"])
            raw["device_mac"][i] = row["device_mac"].lower()
            raw["protocols_all"][i] = _list(row["network_protocols_all"])
            raw["ips_all"][i] = _list(row["network_ips_all"])
    assert seen == n, (seen, n)

    out: dict = {"rows": n, "partition": "validation (all roles; held-out test not opened)",
                 "roles": {str(int(k)): int(v) for k, v in zip(*np.unique(roles, return_counts=True))}}

    # Port presence, the 1000 cap and UNSEEN.
    ports = {}
    for d in ("src", "dst"):
        lists, vs = raw[f"ports_{d}"], set(vocab[d])
        listed = np.array([len(set(x)) for x in lists])
        capped = cnt[d] != np.array([len(x) for x in lists])
        P = np.stack([g(f"network_ports_{d}__present_{p}") for p in vocab[d]], axis=1)
        member = np.array([[p in set(x) for p in vocab[d]] for x in lists])
        expect = np.where(member, 1.0, np.where(capped[:, None], np.nan, 0.0))
        unseen = g(f"network_ports_{d}__present_UNSEEN")
        u_expect = np.array([any(p not in vs for p in x) for x in lists], dtype=np.float32)
        ports[d] = {
            "capped_rows": int(capped.sum()), "max_list_length": int(max(len(x) for x in lists)),
            "max_distinct_listed": int(listed.max()),
            "presence_nan_rows": int(np.isnan(P).any(1).sum()),
            "presence_nan_rows_equal_capped_rows": bool((np.isnan(P).any(1) == capped).all()),
            "presence_mismatches": int((~((P == expect) | (np.isnan(P) & np.isnan(expect)))).sum()),
            "unseen_mismatches": int((unseen != u_expect).sum()), "unseen_nan": int(np.isnan(unseen).sum()),
            "unseen_one_in_capped_rows": int((unseen[capped] == 1).sum()),
        }
    out["ports"] = ports

    # Window duration and per_second.
    dur = g("window_duration_seconds")
    per = {}
    for k in ("all", "src", "dst"):
        c, ps = g(f"network_packets_{k}_count"), g(f"network_packets_{k}_count__per_second")
        per[k] = {"exact_float32": int((ps == (c / dur).astype(np.float32)).sum()), "rows": n}
    out["window_duration"] = {
        "values": {str(int(v)): int(c) for v, c in zip(*np.unique(dur, return_counts=True))},
        "all_integer": bool((dur == np.round(dur)).all()), "per_second_equals_count_over_duration": per}

    # Zero-packet rows.
    pk = g("network_packets_all_count")
    z = pk == 0
    stat_cols = [f"network_{f}_{s}" for f in FAMILIES for s in STATS]
    ratio_cols = [n_ for n_ in schema if "__fraction_of_packets" in n_ or n_ in (
        "network_src_packet_fraction", "network_dst_packet_fraction", "network_packets_per_reported_ip")]
    count_cols = [f"network_packets_{k}_count" for k in ("all", "src", "dst")] + [
        f"network_tcp-flags-{f}_count" for f in FLAGS] + ["network_fragmented-packets"] + [
        f"network_ips_{k}_count" for k in ("all", "src", "dst")]
    pres_cols = [n_ for n_ in schema if "__present_" in n_]
    Z = lambda cols: X[np.ix_(z, [col[c] for c in cols])]  # noqa: E731
    out["zero_packet_rows"] = {
        "rows": int(z.sum()), "counts_zero": bool((Z(count_cols) == 0).all()),
        "quality_flag_one": bool((g("quality_no_packets_observed")[z] == 1).all()),
        "quality_flag_zero_elsewhere": bool((g("quality_no_packets_observed")[~z] == 0).all()),
        "stats_nan": bool(np.isnan(Z(stat_cols)).all()), "ratios_nan": bool(np.isnan(Z(ratio_cols)).all()),
        "presence_and_unseen_zero": bool((Z(pres_cols) == 0).all()),
        "per_second_zero": bool((Z([f"network_packets_{k}_count__per_second" for k in ("all", "src", "dst")]) == 0).all())}
    out["packets_all_equals_src_plus_dst_rows"] = int(
        (pk == g("network_packets_src_count") + g("network_packets_dst_count")).sum())

    # ARP: windows whose only protocol is ARP still report IPv4 addresses (ARP sender/target addresses count).
    arp = np.array([p == ["arp"] for p in raw["protocols_all"]])
    sent_only = arp & (g("network_packets_dst_count") == 0) & (g("network_packets_src_count") > 0)
    pair = lambda m: {f"{int(a)}/{int(b)}": int(c) for (a, b), c in zip(*np.unique(np.stack(  # noqa: E731
        [g("network_ips_src_count")[m], g("network_ips_dst_count")[m]], 1), axis=0, return_counts=True))}
    out["arp_only_windows"] = {
        "rows": int(arp.sum()), "rows_ips_all_count_gt0": int((arp & (g("network_ips_all_count") > 0)).sum()),
        "rows_with_0.0.0.0": int(sum("0.0.0.0" in raw["ips_all"][i] for i in np.flatnonzero(arp))),
        "device_sent_only_rows": int(sent_only.sum()), "device_sent_only_ips_src_dst_counts": pair(sent_only)}
    out["rows_with_0.0.0.0_in_ips_all"] = int(sum("0.0.0.0" in x for x in raw["ips_all"]))

    # Empty family gives 0, never NaN, when packets > 0.
    nz = ~z
    fam = {}
    for f in FAMILIES:
        V = X[np.ix_(nz, [col[f"network_{f}_{s}"] for s in STATS])]
        fam[f] = {"nan_rows": int(np.isnan(V).any(1).sum()), "all_zero_rows": int((V == 0).all(1).sum())}
    out["empty_family_when_packets"] = {"rows_with_packets": int(nz.sum()), "families": fam}

    # Population standard deviation: two-packet rows with distinct min/max and the mean at the midpoint.
    two = pk == 2
    std = {}
    for f in ("ttl", "ip-length", "window-size", "tcp-flags", "header-length", "ip-flags", "mss"):
        mx, mn, av, sd = (g(f"network_{f}_{s}") for s in ("max", "min", "avg", "std_deviation"))
        sel = two & (mx > mn) & np.isclose(av, (mx + mn) / 2, rtol=0, atol=1e-4)
        r = sd[sel] / (mx[sel] - mn[sel])
        std[f] = {"rows": int(sel.sum()), "ratio_0_5": int(np.isclose(r, 0.5, atol=1e-5).sum()),
                  "ratio_sample_std_0_7071": int(np.isclose(r, 2 ** -0.5, atol=1e-5).sum())}
    out["population_std"] = std

    # MSS over MSS-bearing packets only; 12-bit TCP flags.
    mss_max, mss_min, syn = g("network_mss_max"), g("network_mss_min"), g("network_tcp-flags-syn_count")
    out["mss"] = {"rows_mss_max_gt0": int((mss_max > 0).sum()),
                  "rows_mss_max_gt0_and_min_0": int(((mss_max > 0) & (mss_min == 0)).sum()),
                  "rows_syn_0_and_mss_max_gt0": int(((syn == 0) & (mss_max > 0)).sum())}
    tf = g("network_tcp-flags_max")
    out["tcp_flags"] = {"rows_max_gt_511": int((tf > 511).sum()), "max": float(np.nanmax(tf))}

    # Header-direction evidence: src MAC sets hold header source MACs, not only the device's own.
    dev = raw["device_mac"]
    other = sum(any(m.lower() != dev[i] for m in raw["macs_src"][i]) for i in range(n))
    out["macs_src_hold_non_device_macs_rows"] = int(other)

    # Training rows (fit77, the exact rows the final model was fit on): five columns, read only, nothing scored.
    fit = np.load(_path(D / "fit77_train.npy"))
    Xt = np.load(_path(D / "X_train.npy"), mmap_mode="r")
    tcols = ["window_duration_seconds", "network_tcp-flags_max", "network_mss_max", "network_mss_min",
             "network_tcp-flags-syn_count"]
    T = np.asarray(Xt[np.flatnonzero(fit)][:, [col[c] for c in tcols]])
    dt, tf_t, mx_t, mn_t, syn_t = T.T
    out["training_fit77"] = {
        "rows": int(fit.sum()),
        "window_duration": {str(int(v)): int(c) for v, c in zip(*np.unique(dt, return_counts=True))},
        "tcp_flags": {"rows_max_gt_511": int((tf_t > 511).sum()), "max": float(np.nanmax(tf_t))},
        "mss": {"rows_mss_max_gt0": int((mx_t > 0).sum()),
                "rows_mss_max_gt0_and_min_0": int(((mx_t > 0) & (mn_t == 0)).sum()),
                "rows_syn_0_and_mss_max_gt0": int(((syn_t == 0) & (mx_t > 0)).sum())}}

    out["opened"] = sorted(set(opened))
    out["generated_by"] = "experiment/check_live77_conventions.py"
    out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return out


if __name__ == "__main__":
    result = main()
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    json.dump(result, sys.stdout, indent=1)
