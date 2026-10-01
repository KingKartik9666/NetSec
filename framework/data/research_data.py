"""Chunked inspection and purged within-capture preparation of DataSense CSV."""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from framework.training.tracking import write_json


def temporal_partitions(labels, starts, ends):
    """Class-conditional chronological boundaries shared across devices/windows.

    Windows touching either boundary are purged. This measures within-capture
    temporal generalization only; it is not device- or capture-heldout evaluation.
    """
    result = np.full(len(labels), -1, dtype=np.int8)
    boundaries = {}
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        times = np.unique(starts[ids])
        if len(times) < 10:
            raise ValueError(f"Insufficient distinct times for class {label}")
        first, second = times[int(len(times) * .70)], times[int(len(times) * .85)]
        result[ids[ends[ids] < first]] = 0
        result[ids[(starts[ids] > first) & (ends[ids] < second)]] = 1
        result[ids[starts[ids] > second]] = 2
        boundaries[str(label)] = [int(first), int(second)]
    return result, boundaries


def prevent_duplicate_crossing(partition, hashes):
    """Keep identical float32 feature vectors in their earliest eligible split."""
    result = partition.copy()
    eligible = np.flatnonzero(result >= 0)
    _, inverse = np.unique(hashes[eligible], return_inverse=True)
    owners = np.full(inverse.max() + 1, 3, dtype=np.int8)
    np.minimum.at(owners, inverse, result[eligible])
    remove = eligible[result[eligible] != owners[inverse]]
    result[remove] = -1
    return result, len(remove)


def stratified_cap(ids, labels, maximum, seed):
    """Proportional deterministic sample with at least one row of every class."""
    if len(ids) <= maximum:
        return np.sort(ids)
    classes, counts = np.unique(labels[ids], return_counts=True)
    if maximum < len(classes):
        raise ValueError("Sample cap cannot accommodate all classes")
    quota = np.maximum(1, np.floor(counts / counts.sum() * maximum).astype(int))
    while quota.sum() > maximum:
        quota[np.argmax(quota)] -= 1
    while quota.sum() < maximum:
        available = counts - quota
        quota[np.argmax(available)] += 1
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(ids[labels[ids] == c], int(n), replace=False)
                                   for c, n in zip(classes, quota)]))


def prepare(source: Path, output: Path, artifacts: Path, train_cap=120000, eval_cap=30000, seed=42):
    """Inspect every row but retain compact numeric arrays, not a full text frame."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {output}")
    size = source.stat().st_size
    sample = pd.read_csv(source, nrows=1000)
    required = {"label1", "label4", "timestamp_start", "timestamp_end"}
    if not required.issubset(sample.columns):
        raise ValueError(f"Missing required verified DataSense columns: {required - set(sample.columns)}")
    features = [c for c in sample.select_dtypes(include="number").columns
                if c.startswith(("network_", "log_"))]
    if not features:
        raise ValueError("No verified numeric network/log features")
    chunks, target, starts, ends, hashes = [], [], [], [], []
    missing, infinite = Counter(), Counter()
    binary, class_counts = Counter(), Counter()
    devices, source_types = set(), {}
    total = 0
    for frame in pd.read_csv(source, chunksize=10000):
        if frame[list(required)].isna().any().any():
            raise ValueError("Missing labels or timestamps; explicit resolution required")
        for name, dtype in frame.dtypes.items():
            source_types.setdefault(name, set()).add(str(dtype))
        missing.update({k: int(v) for k, v in frame.isna().sum().items()})
        numeric = frame[features].apply(pd.to_numeric, errors="raise")
        infinite.update({k: int(v) for k, v in np.isinf(numeric).sum().items()})
        values = numeric.to_numpy(dtype=np.float32)
        values[~np.isfinite(values)] = np.nan
        # Canonical NaN and signed-zero representations for stable equality hashes.
        values[values == 0] = 0
        chunks.append(values)
        hashes.append(pd.util.hash_pandas_object(pd.DataFrame(values), index=False).to_numpy())
        target.append(frame.label4.astype(str).to_numpy())
        binary.update(frame.label1.astype(str)); class_counts.update(frame.label4.astype(str))
        if not ((frame.label1 == "benign") == (frame.label4 == "benign")).all():
            raise ValueError("Binary and fine-grained benign labels disagree")
        for name, destination in [("timestamp_start", starts), ("timestamp_end", ends)]:
            time = pd.to_datetime(frame[name], format="mixed", utc=True, errors="raise")
            destination.append(time.astype("int64").to_numpy())
        devices.update(frame.device_name.astype(str).unique())
        total += len(frame)
        if total % 100000 == 0:
            print(f"Inspected {total:,} rows", flush=True)
    X = np.concatenate(chunks); del chunks
    names = np.concatenate(target); del target
    start, end, row_hash = np.concatenate(starts), np.concatenate(ends), np.concatenate(hashes)
    if np.any(end < start):
        raise ValueError("Window end precedes its start")
    classes = sorted(class_counts)
    if "benign" not in classes:
        raise ValueError("Dataset has no verified benign class")
    mapping = {name: i for i, name in enumerate(classes)}
    y = np.array([mapping[v] for v in names], dtype=np.int64)
    split, boundaries = temporal_partitions(y, start, end)
    temporal_purged = int((split < 0).sum())
    split, duplicate_purged = prevent_duplicate_crossing(split, row_hash)
    selected = {}
    counts_before_sample = {}
    unsupported = {}
    for index, name in enumerate(["train", "validation", "test"]):
        ids = np.flatnonzero(split == index)
        counts = np.bincount(y[ids], minlength=len(classes))
        absent = [classes[i] for i in np.flatnonzero(counts == 0)]
        if absent and name == "train":
            raise ValueError(f"Classes absent after purging from {name}: {absent}")
        unsupported[name] = absent
        counts_before_sample[name] = {c: int(n) for c, n in zip(classes, counts)}
        selected[name] = stratified_cap(ids, y, train_cap if name == "train" else eval_cap, seed + index)
    output.mkdir(parents=True)
    sample_counts = {}
    for name, ids in selected.items():
        for prefix, array in [("X", X[ids]), ("y", y[ids]), ("row_ids", ids),
                              ("feature_hash", row_hash[ids]), ("window_start", start[ids]), ("window_end", end[ids])]:
            np.save(output / f"{prefix}_{name}.npy", array)
        sample_counts[name] = {c: int(n) for c, n in zip(classes, np.bincount(y[ids], minlength=len(classes)))}
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    near_constant = {}
    for i, name in enumerate(features):
        _, frequency = np.unique(X[:, i], return_counts=True)
        fraction = float(frequency.max() / total)
        if fraction >= .99:
            near_constant[name] = fraction
    limitations = [
        "Within-capture class-conditional temporal evaluation, not unseen-device/capture generalization.",
        "Multiple aggregation window durations overlap; all boundary-crossing or touching windows purged.",
        "Benign and attacks were collected on different dates; date/capture confounding remains even after dropping timestamps.",
        "Full source inspected; capped proportional subsets used for this initial GPU search, not full-data training.",
        "Test arrays exist for a later frozen comparison; search must never load them.",
        "Duplicate check uses 64-bit hashes of selected float32 feature vectors; near-duplicate traffic is not fully characterized.",
        "Sensor/log numeric features included alongside network features; this is multimodal DataSense classification.",
        "Source provenance supplied by user; publisher checksum unavailable for this combined CSV.",
        f"Classes without independent evaluation rows after purging: {unsupported}; do not claim measured recall for these classes.",
    ]
    manifest = {
        "source": str(source.resolve()), "source_bytes": size, "source_sha256": digest.hexdigest(),
        "source_rows": total, "source_columns": len(sample.columns), "feature_names": features,
        "classes": classes, "benign_index": mapping["benign"], "target_column": "label4",
        "seed": seed, "split_method": "class-conditional chronological unique-start quantiles .70/.85; strict boundary purging; duplicate ownership",
        "boundaries_unix_ns_by_class_index": boundaries,
        "train_only_preprocessing": "not yet fitted; candidate fits on training only",
        "temporal_purged_rows": temporal_purged, "duplicate_cross_partition_purged_rows": duplicate_purged,
        "eligible_class_counts": counts_before_sample, "sample_class_counts": sample_counts,
        "classes_without_support": unsupported,
        "rows": {name: len(ids) for name, ids in selected.items()}, "limitations": limitations,
    }
    report = {**manifest, "binary_counts": dict(binary), "class_counts": dict(class_counts),
              "device_count": len(devices), "dtypes": {k: sorted(v) for k, v in source_types.items()},
              "missing_counts": dict(missing), "infinite_counts": dict(infinite),
              "near_constant_numeric_features_at_least_99_percent": near_constant,
              "repeated_numeric_feature_vectors": total - len(np.unique(row_hash)),
              "excluded_columns": [c for c in sample.columns if c not in features],
              "window_duration_seconds_counts": {str(float(k)): int(v) for k, v in zip(*np.unique((end-start)/1e9, return_counts=True))}}
    write_json(output / "manifest.json", manifest)
    write_json(artifacts / "dataset_report.json", report)
    text = f"# DataSense dataset inspection\n\nSource: `{source.name}`; SHA256 `{digest.hexdigest()}`\n\n{total:,} rows; {len(sample.columns)} columns; {len(features)} numeric inputs; {len(classes)} label4 classes.\n\n"
    text += f"Binary counts: {dict(binary)}. Prepared rows: {manifest['rows']}.\n\n"
    text += "## Split and limitations\n\n" + "\n".join(f"- {v}" for v in limitations)
    text += f"\n\nBoundary-purged rows: {temporal_purged:,}; duplicate cross-partition exclusions: {duplicate_purged:,}.\n\n"
    text += "## Full-source class counts\n\n| Class | Rows |\n|---|---:|\n" + "\n".join(f"| {c} | {class_counts[c]} |" for c in classes)
    (artifacts / "dataset_report.md").write_text(text + "\n", encoding="utf-8")
    print(f"Prepared {manifest['rows']}; {len(classes)} classes. Test remains locked.", flush=True)
    return manifest
