"""Fix-test data and shared rules (plan: ml_pipeline/notes_20261101/fix77_plan.md; import-safe, no env changes).

``build`` writes a data folder that is the csv_v3 folder plus appended columns (other arrays hard-linked, unchanged);
the drivers read it via ``CASCADE_DATA``:

* CTX, past-window context: per row, the windows of the same ``device_name`` in the same split whose end lies in
  (start - 30 s, start]. Means over those windows (not counts: training is sampled about 3.5x denser than
  validation), ``ctx_has_past`` and the share of past windows with the row's own duration. NaN when no past window.
  Only ``device_name`` and the window timestamps are read from the CSV copies: never a label, scenario or a window
  that ends after the row starts.
* ROB, volume-free features: the 14 ``derived.NAMES`` plus MACs, distinct IPs and distinct dst ports per packet.

``ctx_live`` is the subset of CTX built from exact live columns only (live77 "core"); protocol presence,
packet size and the reported dst-port count are not live-exact, so the live arm does not use them.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from framework.training.csv_v3 import arms, derived, prep

OUT = prep.ROOT / "dataset/experiments/csv_v3_fix77_20261101_v1"
LOOKBACK_NS = 30 * 10**9
META_COLS = ["source_row_id", "device_name", "timestamp_start", "timestamp_end"]
FORBIDDEN_META = ("label", "scenario")             # substrings no context input may carry

# context name -> source v3 column ("pps" = packets_all_count / window_duration_seconds)
CTX_SOURCES = {
    "ctx_syn_fraction": "network_tcp-flags-syn__fraction_of_packets",
    "ctx_rst_fraction": "network_tcp-flags-rst__fraction_of_packets",
    "ctx_packets_per_second": "pps",
    "ctx_packet_size_avg": "network_packet-size_avg",
    "ctx_udp_present": "network_protocols_all__present_udp",
    "ctx_tcp_present": "network_protocols_all__present_tcp",
    "ctx_arp_present": "network_protocols_all__present_arp",
    "ctx_macs": "network_macs_all_count",
    "ctx_dst_ports": "network_ports_dst_count",
}
CTX = list(CTX_SOURCES) + ["ctx_has_past", "ctx_same_duration_share"]
CTX_LIVE = ["ctx_syn_fraction", "ctx_rst_fraction", "ctx_packets_per_second", "ctx_macs", "ctx_has_past",
            "ctx_same_duration_share"]
ROB_EXTRA = ["rob_macs_per_packet", "rob_ips_per_packet", "rob_dst_ports_per_packet"]
ROB = list(derived.NAMES) + ROB_EXTRA
NAMES = CTX + ROB
GROUPS = {"ctx": CTX, "ctx_live": CTX_LIVE, "rob": ROB}
ROB_WEIGHT = {"kind": "scenario", "rho": 1.0}
FOCUS = ("malware_mirai-syn-flood", "malware_mirai-udp-flood", "mitm_arp-spoofing", "mitm_impersonation",
         "recon_ping-sweep")
PING, RECON = "recon_ping-sweep", "recon_scan"


# ------------------------------------------------------------------------------------------- features
def _source_matrix(X: np.ndarray, names: list[str]) -> np.ndarray:
    col = {n: i for i, n in enumerate(names)}
    out = np.empty((len(X), len(CTX_SOURCES)), dtype=np.float64)
    for j, src in enumerate(CTX_SOURCES.values()):
        if src == "pps":
            with np.errstate(divide="ignore", invalid="ignore"):
                dur = X[:, col["window_duration_seconds"]].astype(np.float64)
                out[:, j] = np.where(dur > 0, X[:, col["network_packets_all_count"]] / dur, np.nan)
        else:
            out[:, j] = X[:, col[src]]
    return out


def context_features(device: np.ndarray, start_ns: np.ndarray, end_ns: np.ndarray, duration: np.ndarray,
                     source: np.ndarray, lookback_ns: int = LOOKBACK_NS) -> np.ndarray:
    """CTX columns for every row: means of ``source`` over the same device's windows ending in (start - L, start].

    A row's own window ends after it starts, so it is never its own context. NaN source values are skipped in the
    means; a mean over no finite value is NaN. Returns (n, len(CTX)) float32.
    """
    n, m = source.shape
    out = np.full((n, len(CTX)), np.nan, dtype=np.float64)
    device = np.asarray(device).astype(str)
    start_ns, end_ns = np.asarray(start_ns, np.int64), np.asarray(end_ns, np.int64)
    if np.any(end_ns <= start_ns):
        raise ValueError("every window must end after it starts")
    duration = np.asarray(duration, np.float64)
    for d in np.unique(device):
        rows = np.flatnonzero(device == d)
        order = rows[np.argsort(end_ns[rows], kind="stable")]
        ends = end_ns[order]
        vals = source[order]
        finite = np.isfinite(vals)
        csum = np.vstack([np.zeros(m), np.cumsum(np.where(finite, vals, 0.0), axis=0)])
        ccnt = np.vstack([np.zeros(m), np.cumsum(finite, axis=0)])
        s = start_ns[rows]
        hi = np.searchsorted(ends, s, side="right")                 # ends <= start
        lo = np.searchsorted(ends, s - lookback_ns, side="right")   # ends > start - L
        n_past = hi - lo
        with np.errstate(divide="ignore", invalid="ignore"):
            cnt = ccnt[hi] - ccnt[lo]
            out[rows, :m] = np.where(cnt > 0, (csum[hi] - csum[lo]) / np.maximum(cnt, 1), np.nan)
        out[rows, m] = (n_past > 0).astype(np.float64)
        same = np.zeros(len(rows))
        dur_sorted = duration[order]
        for dv in np.unique(duration[rows]):
            pos = np.flatnonzero(dur_sorted == dv)
            e_d = ends[pos]
            mine = duration[rows] == dv
            same[mine] = (np.searchsorted(e_d, s[mine], side="right")
                          - np.searchsorted(e_d, s[mine] - lookback_ns, side="right"))
        with np.errstate(divide="ignore", invalid="ignore"):
            out[rows, m + 1] = np.where(n_past > 0, same / np.maximum(n_past, 1), np.nan)
    return out.astype(np.float32)


def rob_features(X: np.ndarray, names: list[str]) -> np.ndarray:
    """ROB columns: derived.compute (live conventions) + per-packet ratios (NaN when the window has no packets)."""
    col = {n: i for i, n in enumerate(names)}
    d = derived.compute(X, names)
    pk = X[:, col["network_packets_all_count"]].astype(np.float64)
    extra = np.empty((len(X), len(ROB_EXTRA)), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        for j, src in enumerate(("network_macs_all_count", "network_ips_all_count", "network_ports_dst_count")):
            extra[:, j] = np.where(pk > 0, X[:, col[src]] / pk, np.nan)
    return np.hstack([d, extra]).astype(np.float32)


def meta(split: str, path: Path | None = None) -> dict:
    """Device and window times for one split, read from the CSV copy (only META_COLS), in row_ids order."""
    import pandas as pd
    if any(any(f in c.lower() for f in FORBIDDEN_META) for c in META_COLS):
        raise RuntimeError("META_COLS must not contain label or scenario columns")
    path = path or prep.DEFAULT_OUT / f"selected_{split}_77class.csv.gz"
    frame = pd.read_csv(path, usecols=META_COLS)
    return {"source_row_id": frame["source_row_id"].to_numpy(), "device": frame["device_name"].astype(str).to_numpy(),
            "start_ns": pd.to_datetime(frame["timestamp_start"]).astype("int64").to_numpy(),
            "end_ns": pd.to_datetime(frame["timestamp_end"]).astype("int64").to_numpy()}


def check_order(info: dict, row_ids: np.ndarray) -> None:
    if not np.array_equal(np.asarray(info["source_row_id"]), np.asarray(row_ids)):
        raise RuntimeError("CSV copy row order differs from row_ids: refusing to attach context to the wrong rows")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for piece in iter(lambda: handle.read(1 << 24), b""):
            h.update(piece)
    return h.hexdigest()


def build(out: Path = OUT, chunk: int = 50_000) -> dict:
    src = prep.DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    names = manifest["feature_names"]
    for f in src.glob("*.npy"):
        if f.name not in ("X_train.npy", "X_validation.npy") and not (out / f.name).exists():
            os.link(f, out / f.name)
    shas, nan_share = {}, {}
    dur_col = names.index("window_duration_seconds")
    for split in ("train", "validation"):
        X = np.load(src / f"X_{split}.npy", mmap_mode="r")
        info = meta(split)
        check_order(info, np.load(src / f"row_ids_{split}.npy"))
        if split == "train":
            for key, arr in (("start_ns", "window_start_train"), ("end_ns", "window_end_train")):
                if not np.array_equal(info[key], np.load(src / f"{arr}.npy")):
                    raise RuntimeError(f"CSV {key} differs from {arr}.npy")
        source = np.empty((X.shape[0], len(CTX_SOURCES)), dtype=np.float64)
        dur = np.empty(X.shape[0], dtype=np.float64)
        for a in range(0, X.shape[0], chunk):
            block = np.asarray(X[a:a + chunk])
            source[a:a + chunk] = _source_matrix(block, names)
            dur[a:a + chunk] = block[:, dur_col]
        ctx = context_features(info["device"], info["start_ns"], info["end_ns"], dur, source)
        Y = np.lib.format.open_memmap(out / f"X_{split}.npy", mode="w+", dtype=np.float32,
                                      shape=(X.shape[0], X.shape[1] + len(NAMES)))
        for a in range(0, X.shape[0], chunk):
            block = np.asarray(X[a:a + chunk])
            Y[a:a + chunk, :X.shape[1]] = block
            Y[a:a + chunk, X.shape[1]:X.shape[1] + len(CTX)] = ctx[a:a + chunk]
            Y[a:a + chunk, X.shape[1] + len(CTX):] = rob_features(block, names)
        new = np.asarray(Y[:, X.shape[1]:])
        nan_share[split] = {n: float(np.mean(np.isnan(new[:, j]))) for j, n in enumerate(NAMES)}
        Y.flush()
        del Y, new
        shas[split] = _sha256(out / f"X_{split}.npy")
    manifest = dict(manifest, feature_names=names + NAMES, fix77_features=NAMES, fix77_groups=GROUPS,
                    fix77_source=str(src), fix77_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    fix77_array_sha256=shas, fix77_nan_share=nan_share, fix77_lookback_seconds=LOOKBACK_NS // 10**9)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return {"out": str(out), "features": len(names) + len(NAMES), "nan_share": nan_share}


# -------------------------------------------------------------------------------------- shared rules
def arm_specs(base_names: list[str], extra: dict[str, list[str]]) -> dict:
    """C0 / CTX / ROB / CTX+ROB (CSV) or C0 / CTX (live) feature lists; ROB arms carry scenario weights rho 1.0."""
    out = {"C0": {"names": list(base_names)}}
    if "ctx" in extra:
        out["CTX"] = {"names": list(base_names) + list(extra["ctx"])}
    if "rob" in extra:
        out["ROB"] = {"names": list(base_names) + list(extra["rob"]), "row_weight": dict(ROB_WEIGHT)}
        if "ctx" in extra:
            out["CTXROB"] = {"names": list(base_names) + list(extra["ctx"]) + list(extra["rob"]),
                             "row_weight": dict(ROB_WEIGHT)}
    return out


def choose(results: dict) -> str | None:
    """Plan decision: highest mean J among admitted arms; CTXROB only if its J beats the better of CTX and ROB by
    >= 0.1 pp (otherwise the best admitted single arm). None when nothing is admitted."""
    admitted = [a for a, s in results.items() if s["admitted"]]
    if not admitted:
        return None
    best = max(admitted, key=lambda a: results[a]["J"])
    if best == "CTXROB":
        singles = [results[a]["J"] for a in ("CTX", "ROB") if a in results]
        if singles and results["CTXROB"]["J"] < max(singles) + arms.GAIN:
            rest = [a for a in admitted if a != "CTXROB"]
            return max(rest, key=lambda a: results[a]["J"]) if rest else None
    return best


def ping_merge(proba: np.ndarray, y: np.ndarray, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Fix A: ping-sweep probability added to recon_scan; labels mapped the same way (scoring only)."""
    p, s = classes.index(PING), classes.index(RECON)
    q = np.array(proba, dtype=np.float64, copy=True)
    q[:, s] += q[:, p]
    q[:, p] = 0.0
    return q, np.where(np.asarray(y) == p, s, y)


def focus_metrics(proba: np.ndarray, y: np.ndarray, classes: list[str], benign: int = 0) -> dict:
    """Reported-only numbers: focus-class recall, attack -> benign rate, accuracy after the ping merge."""
    y = np.asarray(y)
    pred = proba.argmax(1)
    out = {}
    for c in FOCUS:
        k = classes.index(c)
        m = y == k
        out[f"recall_{c}"] = float(np.mean(pred[m] == k)) if m.any() else None
    att = y != benign
    out["attack_to_benign"] = float(np.mean(pred[att] == benign)) if att.any() else None
    q, ym = ping_merge(proba, y, classes)
    out["acc_ping_merged"] = float(np.mean(q.argmax(1) == ym))
    out["ping_to_benign"] = int(np.sum((y == classes.index(PING)) & (q.argmax(1) == benign)))
    return out


def fold_metrics(work: Path, y_val: np.ndarray, y_train: np.ndarray, tune, seen, unseen, classes: list[str]) -> dict:
    """Role-0 + fold OOF metrics of one fold job (roles 1/2 never read) plus the reported-only focus numbers."""
    work = Path(work)
    pv = np.load(work / "val_proba.npy")
    idx, po = np.load(work / "oof_index.npy"), np.load(work / "oof_proba.npy")
    n = len(classes)
    out = {**arms.role0_metrics(pv, y_val, tune, seen, unseen, n), **arms.oof_metrics(po, y_train[idx], n)}
    out["focus_role0"] = focus_metrics(pv[tune], y_val[tune], classes)
    out["focus_oof"] = focus_metrics(po, y_train[idx], classes)
    return out


def focus_table_lines(o: dict) -> list[str]:
    """Markdown tables of the reported-only focus numbers (C0 and every arm, OOF and role 0)."""
    cols = [f"recall_{c}" for c in FOCUS] + ["attack_to_benign", "acc_ping_merged", "ping_to_benign"]
    head = ["mirai-syn", "mirai-udp", "ARP", "imperson.", "ping", "attack->benign", "acc ping merged", "ping->benign"]
    fmt = lambda k, v: "-" if v is None else (f"{v:.1f}" if k == "ping_to_benign" else f"{100 * v:.2f}")
    lines = []
    for key, title in (("focus_oof", "fold-1 OOF"), ("focus_role0", "role 0")):
        lines += ["", f"Reported only, {title} (mean over seeds; recall/acc in %, ping->benign = rows):", "",
                  "| arm | " + " | ".join(head) + " |", "|---|" + "---|" * len(head)]
        for name, f in [("C0", o[f"control_{key}"])] + [(a, s[key]) for a, s in o["arms"].items()]:
            lines.append(f"| {name} | " + " | ".join(fmt(k, f[k]) for k in cols) + " |")
    return lines


def mean_focus(runs: list[dict], key: str) -> dict:
    keys = runs[0][key].keys()
    return {k: (None if any(r[key][k] is None for r in runs) else float(np.mean([r[key][k] for r in runs])))
            for k in keys}


# ------------------------------------------------------------------------------------- one GPU at a time
# The two upgrade-round-2 drivers were added after the fix77 stage-1 chain finished (2026-11-02), so a fix77 driver
# started by hand now also refuses to start while an upgrades2 driver is alive (also between its jobs). The upgrades2
# queue (experiment.upgrades2_queue) stays deliberately unlisted: it is a waiting watchdog, not a GPU job.
# Remaining gap: train_live77, train_live77_upgrades and train_cascade_v3 have no cross-driver guard of their own,
# and train_fs77_tabm only looks for fs77/cascade processes in its own study; they are not to be started by hand while
# an upgrades2 driver or the queue is alive (operating rule in experiment/upgrades2_queue.py).
CONTROLLER_MARKERS = ("experiment.train_fix77_tabm", "experiment.train_fix77_live", "experiment.train_fs77_tabm",
                      "experiment.train_live77", "experiment.train_live77_upgrades", "experiment.train_cascade_v3",
                      "experiment.train_upgrades2_live", "experiment.train_fix77_ext")


def other_controllers() -> list[int]:
    """PIDs of any other training controller (fix77, fs77, live77, cascade, upgrades2 drivers) or runner job (runner
    jobs run ``experiment.train_cascade_v3``, a marker); one GPU job at a time."""
    import psutil
    mine = {os.getpid()} | {p.pid for p in psutil.Process().parents()}
    busy = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        if p.info["pid"] in mine:
            continue
        cmd = p.info["cmdline"] or []
        mods = [c for c in cmd if c in CONTROLLER_MARKERS]
        if not mods:
            continue
        # status/stop/start commands are short-lived and never train
        if any(c in ("status", "stop", "start") for c in cmd):
            continue
        busy.append(p.info["pid"])
    return busy
