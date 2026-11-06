"""Prepare the cascade dataset v3 from verified-v2 without touching validation rows or the test partition.

Changes over v2 are deliberately small and each one is recorded:

* training-only column pruning: constant, all-missing and exact-duplicate columns (checked over the final
  combined matrix, so new features cannot duplicate kept ones);
* presence tokens with weak support on observable training rows (fewer than ``min_scenarios`` scenarios or
  fewer than ``min_devices`` devices, zero support included) are folded into a per-group rare-token count;
  groups with a single weak token feed a shared cross-group count, so no fingerprint column survives renamed;
* four row-local features computed from the exported address/port lists (absent from v2);
* per-target fitting masks that quarantine no-signal rows, identical-vector label conflicts and training
  vectors identical to a validation vector (a validation-conditioned exclusion, labelled as such).

Rows and labels are never deleted or changed. Inputs are hash-checked against verified-v2's
``verification/output_hashes.json``. Only the original training and 30,000 validation selections are read;
held-out test arrays are never opened.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
V2 = ROOT / "dataset/experiments/datasense_verified_clean_20261014_v2"
WRAPUP = ROOT / "artifacts/training/all_models_wrapup_20261023_v1"
DEFAULT_OUT = ROOT / "dataset/experiments/csv_v3_cascade_20261030_v1"
VERSION = "csv_v3_cascade/2"

RECON8 = (
    "recon_host-disc-arp-ping", "recon_host-disc-tcp-ack-ping", "recon_host-disc-tcp-syn-ping",
    "recon_host-disc-tcp-syn-stealth", "recon_host-disc-udp-ping", "recon_os-scan", "recon_port-scan",
    "recon_vuln-scan",
)
MERGED = "recon_scan"
PRESENT = "__present_"
UNSEEN = "UNSEEN"
LIST_COLUMNS = ("network_ips_src", "network_ips_dst", "network_ports_src", "network_ports_dst")
RAW_FEATURES = ("listed_dst_ports_per_dst_ip", "listed_src_ports_per_src_ip",
                "listed_port_overlap_jaccard", "listed_low_dst_port_share")
NON_PREDICTORS = ("source_row_id", "device_name", "device_mac", "label_full", "label1", "label2", "label3",
                  "label4", "label4_original", "timestamp", "timestamp_start", "timestamp_end")
SHARED_RARE = "all_groups"
_TOKEN = re.compile(r"'([^']*)'")


# ---------------------------------------------------------------------------------------------- targets
def targets(classes: list[str]) -> dict:
    """Return 77-class names plus index maps 84->77 and 84->8 (``-1`` for non-recon8 classes).

    The merged class takes the position of the first recon8 class, so benign stays at index 0 and the
    relative order of every other class is unchanged.
    """
    missing = [c for c in RECON8 if c not in classes]
    if missing or len(set(classes)) != len(classes):
        raise ValueError(f"class list incompatible with the recon8 merge: missing={missing}")
    if classes[0] != "benign":
        raise ValueError("benign must be class 0")
    classes77: list[str] = []
    to77 = np.empty(len(classes), dtype=np.int64)
    for i, name in enumerate(classes):
        if name in RECON8:
            if MERGED not in classes77:
                classes77.append(MERGED)
            to77[i] = classes77.index(MERGED)
        else:
            classes77.append(name)
            to77[i] = len(classes77) - 1
    to8 = np.full(len(classes), -1, dtype=np.int64)
    for j, name in enumerate(RECON8):
        to8[classes.index(name)] = j
    return {"classes84": list(classes), "classes77": classes77, "classes8": list(RECON8),
            "map84_to77": to77, "map84_to8": to8}


# ------------------------------------------------------------------------------------------ raw features
def _row_features(ips_src, ips_dst, ports_src, ports_dst) -> tuple[float, float, float, float]:
    """Four descriptors of one row's exported lists; lists are parsed, used and discarded immediately."""
    def parse(cell):
        return set(_TOKEN.findall(cell)) if isinstance(cell, str) else None

    isrc, idst, psrc, pdst = parse(ips_src), parse(ips_dst), parse(ports_src), parse(ports_dst)
    nan = float("nan")
    dst_per_ip = len(pdst) / len(idst) if pdst is not None and idst else nan
    src_per_ip = len(psrc) / len(isrc) if psrc is not None and isrc else nan
    union = (psrc | pdst) if psrc is not None and pdst is not None else None
    jaccard = len(psrc & pdst) / len(union) if union else nan
    low = sum(1 for p in pdst if p.isdigit() and int(p) < 1024) / len(pdst) if pdst else nan
    return dst_per_ip, src_per_ip, jaccard, low


def raw_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Row-local list descriptors; NaN when a list is missing/empty so a real zero never means "unknown".

    Values describe the exported (possibly capped) lists, hence the "listed" prefix.
    """
    rows = [_row_features(*cells) for cells in zip(*(frame[c].to_numpy() for c in LIST_COLUMNS))]
    return pd.DataFrame(rows, columns=list(RAW_FEATURES), index=frame.index, dtype=np.float32)


def raw_features_streamed(split: str, row_ids: np.ndarray, chunk_rows: int = 50_000) -> np.ndarray:
    """Compute :func:`raw_features` chunk by chunk from the v2 raw extract, aligned to ``row_ids``."""
    parts, ids = [], []
    for chunk in pd.read_csv(V2 / f"raw/selected_{split}.csv.gz", usecols=["source_row_id", *LIST_COLUMNS],
                             chunksize=chunk_rows, low_memory=False):
        parts.append(raw_features(chunk).to_numpy(np.float32))
        ids.append(chunk["source_row_id"].to_numpy())
    values, order = np.concatenate(parts), np.concatenate(ids)
    position = pd.Series(np.arange(len(order)), index=order)
    if position.index.has_duplicates or not np.isin(row_ids, order).all():
        raise RuntimeError(f"raw {split} extract does not align with v2 row ids")
    return values[position.loc[row_ids].to_numpy()]


# ------------------------------------------------------------------------------------- column selection
def _column_digest(values: np.ndarray) -> bytes:
    values = np.where(np.isnan(values), np.float32(np.nan), values).astype(np.float32)
    return hashlib.blake2b(np.ascontiguousarray(values).tobytes(), digest_size=16).digest()


def fit_columns(x: np.ndarray, fit: np.ndarray, names: list[str], scenarios: np.ndarray,
                devices: np.ndarray, min_scenarios: int, min_devices: int) -> dict:
    """Decide which v2 columns to keep, using fitting rows only.

    Non-UNSEEN presence tokens are judged on support first: a token present in fewer than
    ``min_scenarios`` scenarios or ``min_devices`` devices (zero included) is folded, not dropped.
    UNSEEN indicators are never folded. Remaining columns are dropped when constant, all-missing or an
    exact duplicate of an earlier kept column.
    """
    rows = np.flatnonzero(fit)
    kept, dropped, rare = [], {}, {}
    seen_digest: dict[bytes, str] = {}
    sub_scen, sub_dev = scenarios[rows], devices[rows]
    for j, name in enumerate(names):
        col = np.asarray(x[rows, j], dtype=np.float32)
        if PRESENT in name and not name.endswith(UNSEEN):
            hit = col == 1
            n_scen, n_dev = np.unique(sub_scen[hit]).size, np.unique(sub_dev[hit]).size
            if n_scen < min_scenarios or n_dev < min_devices:
                rare.setdefault(name.split(PRESENT)[0], []).append(name)
                dropped[name] = f"folded_rare:{n_scen}_scenarios_{n_dev}_devices"
                continue
        finite = col[~np.isnan(col)]
        if finite.size == 0:
            dropped[name] = "all_missing_on_fit_rows"
            continue
        if finite.size == col.size and np.all(finite == finite[0]):
            dropped[name] = "constant_on_fit_rows"
            continue
        digest = _column_digest(col)
        if digest in seen_digest:
            dropped[name] = f"duplicate_of:{seen_digest[digest]}"
            continue
        seen_digest[digest] = name
        kept.append(j)
    # A one-token group would reproduce that token's column under a new name; pool those tokens instead.
    single = [g for g, cols in rare.items() if len(cols) < 2]
    if single:
        pooled = [c for g in single for c in rare.pop(g)]
        rare.setdefault(SHARED_RARE, []).extend(pooled)
    return {"kept": kept, "dropped": dropped, "rare": rare}


def rare_counts(x: np.ndarray, names: list[str], rare: dict[str, list[str]]) -> tuple[np.ndarray, list[str]]:
    """Count present rare tokens per group; NaN when every folded column is missing for that row."""
    index = {n: i for i, n in enumerate(names)}
    cols, out_names = [], []
    for group in sorted(rare):
        block = np.asarray(x[:, [index[n] for n in rare[group]]], dtype=np.float32)
        count = np.nansum(block == 1, axis=1).astype(np.float32)
        count[np.all(np.isnan(block), axis=1)] = np.nan
        cols.append(count)
        out_names.append(f"{group}__rare_token_count")
    if not cols:
        return np.empty((x.shape[0], 0), dtype=np.float32), []
    return np.stack(cols, axis=1), out_names


def duplicate_columns(x: np.ndarray, fit: np.ndarray, names: list[str]) -> dict[str, str]:
    """Exact duplicate columns of the final matrix on fitting rows: {later name: first name}."""
    rows = np.flatnonzero(fit)
    first: dict[bytes, str] = {}
    dupes = {}
    for j, name in enumerate(names):
        digest = _column_digest(np.asarray(x[rows, j], dtype=np.float32))
        if digest in first:
            dupes[name] = first[digest]
        else:
            first[digest] = name
    return dupes


# --------------------------------------------------------------------------------- vectors and conflicts
def row_digests(x: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    """Canonical per-row digest (all NaN payloads equal) used for duplicate/conflict checks."""
    out = []
    for start in range(0, len(x), chunk):
        block = np.asarray(x[start:start + chunk], dtype=np.float32)
        canon = np.ascontiguousarray(np.where(np.isnan(block), np.float32(np.nan), block))
        out.extend(hashlib.blake2b(r.tobytes(), digest_size=16).hexdigest() for r in canon)
    return np.array(out)


def conflict_rows(digests: np.ndarray, y: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Mask of ``rows`` whose identical input vector carries more than one label within ``rows``."""
    frame = pd.DataFrame({"d": digests[rows], "y": y[rows]})
    labels = frame.groupby("d")["y"].nunique()
    bad = set(labels.index[labels > 1])
    mask = np.zeros(len(digests), dtype=bool)
    mask[rows[frame["d"].isin(bad).to_numpy()]] = True
    return mask


# ----------------------------------------------------------------------------------------------- helpers
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


V2_INPUTS = ("model/X_train.npy", "model/X_validation.npy", "model/y_train.npy", "model/y_validation.npy",
             "model/row_ids_train.npy", "model/row_ids_validation.npy", "model/window_start_train.npy",
             "model/window_end_train.npy", "model/manifest.json", "raw/selected_train.csv.gz",
             "raw/selected_validation.csv.gz", "audit/row_facts_train.csv.gz")


def _verify_inputs() -> dict:
    """Hash every v2 input; raise if any differs from verified-v2's recorded output hashes."""
    recorded = json.loads((V2 / "verification/output_hashes.json").read_text(encoding="utf-8"))
    result = {}
    for name in V2_INPUTS:
        digest = _sha256(V2 / name)
        if name in recorded and recorded[name] != digest:
            raise RuntimeError(f"verified-v2 input changed: {name}")
        result[f"v2/{name}"] = {"sha256": digest, "matches_v2_record": name in recorded}
    roles = WRAPUP / "validation_roles.npy"
    result["wrapup/validation_roles.npy"] = {"sha256": _sha256(roles), "matches_v2_record": None}
    return result


def _metadata(split: str, row_ids: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(V2 / f"raw/selected_{split}.csv.gz", usecols=["source_row_id", "label_full", "device_name"])
    frame = frame.set_index("source_row_id")
    if frame.index.has_duplicates or not np.isin(row_ids, frame.index).all():
        raise RuntimeError(f"raw {split} metadata does not align with v2 row ids")
    return frame.loc[row_ids]


# ---------------------------------------------------------------------------------------- 77-class copy
def write_merged_copy(split: str, out: Path, expected_rows: int, expected_merged: int,
                      chunk_rows: int = 50_000) -> dict:
    """Write a 77-class copy of a raw selection: ``label4`` merged, original kept in ``label4_original``.

    ``label3`` and ``label_full`` still name the original subtype and are listed as non-predictors.
    Written to a temporary file and renamed only after the row counts are confirmed.
    """
    target = out / f"selected_{split}_77class.csv.gz"
    temporary = target.with_suffix(".gz.tmp")
    rows = merged = 0
    with gzip.open(temporary, "wt", encoding="utf-8", newline="", compresslevel=6) as handle:
        for i, chunk in enumerate(pd.read_csv(V2 / f"raw/selected_{split}.csv.gz", chunksize=chunk_rows,
                                              dtype=str, keep_default_na=False)):
            original = chunk["label4"]
            chunk.insert(chunk.columns.get_loc("label4") + 1, "label4_original", original)
            is_recon8 = original.isin(RECON8)
            chunk["label4"] = original.where(~is_recon8, MERGED)
            chunk.to_csv(handle, index=False, header=i == 0, lineterminator="\n")
            rows += len(chunk)
            merged += int(is_recon8.sum())
    if (rows, merged) != (expected_rows, expected_merged):
        temporary.unlink()
        raise RuntimeError(f"77-class copy of {split}: rows/merged {rows}/{merged} != {expected_rows}/{expected_merged}")
    os.replace(temporary, target)
    return {"file": target.name, "rows": rows, "merged_rows": merged, "sha256": _sha256(target),
            "non_predictor_columns": list(NON_PREDICTORS),
            "note": "label3 and label_full keep the original subtype names; never use them as predictors"}


# ----------------------------------------------------------------------------------------------- prepare
def prepare(out: Path = DEFAULT_OUT, min_scenarios: int = 3, min_devices: int = 2, progress=print) -> dict:
    """Build v3 arrays, the 77-class CSV copies and the manifest. Refuses to overwrite a finished run."""
    out = Path(out)
    if (out / "manifest.json").exists():
        raise FileExistsError(f"{out} already prepared; create a new version instead of overwriting")
    out.mkdir(parents=True, exist_ok=True)
    progress("verifying verified-v2 inputs")
    inputs = _verify_inputs()
    manifest = json.loads((V2 / "model/manifest.json").read_text(encoding="utf-8"))
    names, classes = manifest["feature_names"], manifest["classes"]
    tmap = targets(classes)

    x_tr = np.load(V2 / "model/X_train.npy", mmap_mode="r")
    x_va = np.load(V2 / "model/X_validation.npy", mmap_mode="r")
    y84_tr, y84_va = np.load(V2 / "model/y_train.npy"), np.load(V2 / "model/y_validation.npy")
    ids_tr, ids_va = np.load(V2 / "model/row_ids_train.npy"), np.load(V2 / "model/row_ids_validation.npy")
    roles = np.load(WRAPUP / "validation_roles.npy")
    if roles.shape != (len(ids_va),):
        raise RuntimeError("validation roles do not align with v2 validation rows")

    progress("loading training/validation metadata")
    meta_tr, meta_va = _metadata("train", ids_tr), _metadata("validation", ids_va)
    facts = pd.read_csv(V2 / "audit/row_facts_train.csv.gz", usecols=["source_row_id", "no_signal", "hard_invalid"])
    facts = facts.set_index("source_row_id").loc[ids_tr]
    no_signal, hard_invalid = facts["no_signal"].to_numpy(bool), facts["hard_invalid"].to_numpy(bool)
    scen_codes, scen_names = pd.factorize(np.concatenate([meta_tr["label_full"].to_numpy(),
                                                          meta_va["label_full"].to_numpy()]))
    scen_tr, scen_va = scen_codes[:len(ids_tr)], scen_codes[len(ids_tr):]
    dev_tr = pd.factorize(meta_tr["device_name"].to_numpy())[0]
    val_seen = np.isin(scen_va, scen_tr)

    observable = ~no_signal & ~hard_invalid
    progress("fitting column selection on observable training rows")
    selection = fit_columns(x_tr, observable, names, scen_tr, dev_tr, min_scenarios, min_devices)
    kept_names = [names[j] for j in selection["kept"]]

    progress("building feature matrices")
    blocks = {}
    for split, x, ids in (("train", x_tr, ids_tr), ("validation", x_va, ids_va)):
        base = np.asarray(x[:, selection["kept"]], dtype=np.float32)
        rare, rare_names = rare_counts(x, names, selection["rare"])
        extra = raw_features_streamed(split, ids)
        blocks[split] = np.concatenate([base, rare, extra], axis=1)
        del base, rare, extra
    feature_names = kept_names + rare_names + list(RAW_FEATURES)
    for split, block in blocks.items():
        if np.isinf(block).any():
            raise RuntimeError(f"infinite values in v3 {split} features")
    dupes = duplicate_columns(blocks["train"], observable, feature_names)
    if dupes:
        keep = [i for i, n in enumerate(feature_names) if n not in dupes]
        for split in blocks:
            blocks[split] = np.ascontiguousarray(blocks[split][:, keep])
        selection["dropped"].update({n: f"duplicate_of:{f}" for n, f in dupes.items()})
        feature_names = [feature_names[i] for i in keep]

    progress("computing duplicate/conflict quarantine")
    d_tr, d_va = row_digests(blocks["train"]), row_digests(blocks["validation"])
    shared_with_validation = np.isin(d_tr, d_va)
    y77_tr, y8_tr = tmap["map84_to77"][y84_tr], tmap["map84_to8"][y84_tr]
    base_fit = observable & ~shared_with_validation
    rows = np.flatnonzero(base_fit)
    fit84 = base_fit & ~conflict_rows(d_tr, y84_tr, rows)
    fit77 = base_fit & ~conflict_rows(d_tr, y77_tr, rows)
    recon_rows = np.flatnonzero(base_fit & (y8_tr >= 0))
    fit8 = base_fit & (y8_tr >= 0) & ~conflict_rows(d_tr, y8_tr, recon_rows)
    for name, mask, y, n in (("84", fit84, y84_tr, 84), ("77", fit77, y77_tr, 77)):
        if np.unique(y[mask]).size != n:
            raise RuntimeError(f"fit mask for {name} classes lost class support")
    if np.unique(y8_tr[fit8]).size != 8:
        raise RuntimeError("fit mask for recon specialist lost class support")

    progress("writing arrays")
    arrays = {
        "X_train": blocks["train"], "X_validation": blocks["validation"],
        "y84_train": y84_tr, "y84_validation": y84_va,
        "y77_train": y77_tr, "y77_validation": tmap["map84_to77"][y84_va],
        "y8_train": y8_tr, "y8_validation": tmap["map84_to8"][y84_va],
        "row_ids_train": ids_tr, "row_ids_validation": ids_va,
        "window_start_train": np.load(V2 / "model/window_start_train.npy"),
        "window_end_train": np.load(V2 / "model/window_end_train.npy"),
        "row_digest_train": d_tr.astype("U32"),
        "scenario_train": scen_tr.astype(np.int64), "scenario_validation": scen_va.astype(np.int64),
        "validation_scenario_seen": val_seen, "validation_roles": roles,
        "fit84_train": fit84, "fit77_train": fit77, "fit8_train": fit8,
        "no_signal_train": no_signal, "shared_with_validation_train": shared_with_validation,
    }
    for key, value in arrays.items():
        np.save(out / f"{key}.npy", value, allow_pickle=False)
    array_hashes = {f"{k}.npy": _sha256(out / f"{k}.npy") for k in arrays}
    n_tr, n_va = len(ids_tr), len(ids_va)
    merged_tr, merged_va = int((y8_tr >= 0).sum()), int((tmap["map84_to8"][y84_va] >= 0).sum())
    group = {**{n: "v2_kept" for n in kept_names}, **{n: "rare_token_count" for n in rare_names},
             **{n: "list_descriptor" for n in RAW_FEATURES}}
    group = {n: group[n] for n in feature_names}
    del blocks, arrays, d_tr, d_va, x_tr, x_va

    progress("writing 77-class CSV copies of the raw selections")
    csv_copies = {
        "train": write_merged_copy("train", out, n_tr, merged_tr),
        "validation": write_merged_copy("validation", out, n_va, merged_va),
    }
    result = {
        "version": VERSION, "min_scenarios": int(min_scenarios), "min_devices": int(min_devices),
        "code_sha256": _sha256(Path(__file__)),
        "library_versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "feature_names": feature_names, "feature_groups": group,
        "kept_v2_column_indexes": selection["kept"],
        "classes84": tmap["classes84"], "classes77": tmap["classes77"], "classes8": tmap["classes8"],
        "map84_to77": tmap["map84_to77"].tolist(), "map84_to8": tmap["map84_to8"].tolist(),
        "benign_index": 0, "v2_feature_count": len(names), "v3_feature_count": len(feature_names),
        "dropped_v2_columns": selection["dropped"], "rare_token_groups": selection["rare"],
        "scenario_names": list(map(str, scen_names)),
        "counts": {
            "train_rows": n_tr, "validation_rows": n_va,
            "no_signal": int(no_signal.sum()), "hard_invalid": int(hard_invalid.sum()),
            "shared_with_validation": int(shared_with_validation.sum()),
            "fit84": int(fit84.sum()), "fit77": int(fit77.sum()), "fit8": int(fit8.sum()),
            "conflict84_excluded": int((base_fit & ~fit84).sum()),
            "conflict77_excluded": int((base_fit & ~fit77).sum()),
            "recon8_train_rows": merged_tr, "recon8_validation_rows": merged_va,
            "validation_seen_scenario_rows": int(val_seen.sum()),
        },
        "inputs": inputs, "array_sha256": array_hashes, "csv_copies_77class": csv_copies,
        "policy": {
            "labels_changed": int(np.sum(np.load(out / "y84_train.npy") != np.load(V2 / "model/y_train.npy"))),
            "validation_rows_changed": int(np.sum(np.load(out / "row_ids_validation.npy") != ids_va)),
            "test_arrays_loaded": False,
            "fitting_statistics_source": "observable original training rows only",
            "validation_conditioned_exclusion": "training rows whose v3 vector equals a validation vector are excluded from fitting",
            "scenario_metadata_use": "training label_full/device_name only to count token support; never predictors",
        },
    }
    (out / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    progress("done")
    return result
