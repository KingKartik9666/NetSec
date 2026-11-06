"""Derived live features for the upgrade tests (plan: ml_pipeline/notes_20261031/live77_upgrade_plan.md, 6 and 7).

Every formula uses only exact live statistics, so the Rust extractor can compute the same value from the same
window. Conventions follow ``live77.EMULATION``: NaN when the window has no packets; 0 when the family is empty but
packets exist (the dataset stores 0 there). ``build`` writes a data folder that is the v3 folder plus these columns
appended after the 445 v3 columns (other arrays hard-linked, unchanged); ``runner`` reads it via ``CASCADE_DATA``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from framework.training.csv_v3 import prep

INITIAL_TTLS = np.array([32.0, 64.0, 128.0, 255.0])
OUT = prep.ROOT / "dataset/experiments/csv_v3_live77_derived_20261031_v1"

TTL_HOP = ["derived_ttl_hop_min", "derived_ttl_hop_max", "derived_ttl_hop_avg", "derived_ttl_mixed_init"]
RANGES = ["derived_range_ttl", "derived_range_ip-length", "derived_range_header-length", "derived_range_window-size"]
RATIOS = ["derived_ratio_syn_per_ack", "derived_ratio_rst_per_syn", "derived_ratio_fin_per_syn"]
IP_RATES = ["derived_ips_all_per_second", "derived_ips_src_per_second", "derived_ips_dst_per_second"]
NAMES = TTL_HOP + RANGES + RATIOS + IP_RATES
GROUPS = {"ttl_hop": TTL_HOP, "ranges": RANGES, "ratios": RATIOS, "ip_rates": IP_RATES}


def _init_ttl(x: np.ndarray) -> np.ndarray:
    """Smallest common initial TTL >= x (x <= 255); 0 stays 0."""
    idx = np.searchsorted(INITIAL_TTLS, x, side="left").clip(0, len(INITIAL_TTLS) - 1)
    return np.where(x > 0, INITIAL_TTLS[idx], 0.0)


def compute(X: np.ndarray, names: list[str]) -> np.ndarray:
    """(n, len(NAMES)) float32 derived features from v3 rows ``X`` with column ``names``."""
    c = {n: X[:, names.index(n)].astype(np.float64) for n in names if n.startswith(("network_", "window_"))}
    n = c["network_packets_all_count"]
    out = {}
    tmax, tmin, tavg = c["network_ttl_max"], c["network_ttl_min"], c["network_ttl_avg"]
    init = _init_ttl(tmax)
    out["derived_ttl_hop_min"] = np.where(tmax > 0, init - tmax, 0.0)
    out["derived_ttl_hop_max"] = np.where(tmax > 0, np.maximum(init - tmin, 0.0), 0.0)
    out["derived_ttl_hop_avg"] = np.where(tmax > 0, init - tavg, 0.0)
    out["derived_ttl_mixed_init"] = np.where(tmax > 0, (_init_ttl(tmin) != init).astype(np.float64), 0.0)
    for fam in ("ttl", "ip-length", "header-length", "window-size"):
        out[f"derived_range_{fam}"] = c[f"network_{fam}_max"] - c[f"network_{fam}_min"]
    syn, ack = c["network_tcp-flags-syn_count"], c["network_tcp-flags-ack_count"]
    rst, fin = c["network_tcp-flags-rst_count"], c["network_tcp-flags-fin_count"]
    out["derived_ratio_syn_per_ack"] = syn / (ack + 1.0)
    out["derived_ratio_rst_per_syn"] = rst / (syn + 1.0)
    out["derived_ratio_fin_per_syn"] = fin / (syn + 1.0)
    dur = c["window_duration_seconds"]
    for d in ("all", "src", "dst"):
        out[f"derived_ips_{d}_per_second"] = c[f"network_ips_{d}_count"] / dur
    Y = np.stack([out[k] for k in NAMES], axis=1)
    Y[~(n > 0)] = np.nan
    if np.isinf(Y).any():
        raise ValueError("derived feature produced inf")
    return Y.astype(np.float32)


def build(out: Path = OUT, chunk: int = 50_000) -> dict:
    src = prep.DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    names = manifest["feature_names"]
    for f in src.glob("*.npy"):
        if f.name not in ("X_train.npy", "X_validation.npy") and not (out / f.name).exists():
            os.link(f, out / f.name)
    shas = {}
    for split in ("train", "validation"):
        X = np.load(src / f"X_{split}.npy", mmap_mode="r")
        Y = np.lib.format.open_memmap(out / f"X_{split}.npy", mode="w+", dtype=np.float32,
                                      shape=(X.shape[0], X.shape[1] + len(NAMES)))
        for a in range(0, X.shape[0], chunk):
            block = np.asarray(X[a:a + chunk])
            Y[a:a + chunk, :X.shape[1]] = block
            Y[a:a + chunk, X.shape[1]:] = compute(block, names)
        Y.flush()
        del Y
        digest = hashlib.sha256()
        with open(out / f"X_{split}.npy", "rb") as handle:
            for piece in iter(lambda: handle.read(1 << 24), b""):
                digest.update(piece)
        shas[split] = digest.hexdigest()
    manifest = dict(manifest, feature_names=names + NAMES, derived_features=NAMES, derived_groups=GROUPS,
                    derived_source=str(src), derived_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    derived_array_sha256=shas)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return {"out": str(out), "features": len(names) + len(NAMES)}


if __name__ == "__main__":
    print(build())
