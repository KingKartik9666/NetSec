"""Add train-vocabulary protocol presence to the same fixed research split."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from framework.training.tracking import write_json

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["network_protocols_all", "network_protocols_src", "network_protocols_dst"]


def parse_protocols(value):
    if not isinstance(value, str) or len(value) > 8192:
        raise ValueError("Expected a bounded string containing a protocol list")
    tokens = ast.literal_eval(value)
    if not isinstance(tokens, list) or any(not isinstance(v, str) or len(v) > 80 for v in tokens):
        raise ValueError("Invalid protocol list")
    return set(tokens)


def protocol_matrix(rows, vocabulary):
    result = np.zeros((len(rows), sum(len(vocabulary[f]) + 1 for f in FIELDS)), dtype=np.float32)
    offset = 0
    for field in FIELDS:
        mapping = {token: i for i, token in enumerate(vocabulary[field])}
        for row, text in enumerate(rows[field]):
            for token in parse_protocols(text):
                result[row, offset + mapping.get(token, len(mapping))] = 1
        offset += len(mapping) + 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "dataset/prepared/datasense_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset/prepared/datasense_protocols_v1")
    args = parser.parse_args()
    base, output = args.base, args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest = json.loads((base / "manifest.json").read_text())
    ids = {split: np.load(base / f"row_ids_{split}.npy") for split in ["train", "validation", "test"]}
    pieces = {split: [] for split in ids}
    offset = 0
    # Only feature text is read; the locked test labels are not inspected.
    for chunk in pd.read_csv(manifest["source"], usecols=FIELDS, chunksize=10000):
        for split, indexes in ids.items():
            left, right = np.searchsorted(indexes, [offset, offset + len(chunk)])
            pieces[split].append(chunk.iloc[indexes[left:right] - offset].copy())
        offset += len(chunk)
    frames = {split: pd.concat(parts, ignore_index=True) for split, parts in pieces.items()}
    vocabulary = {}
    for field in FIELDS:
        counts = Counter()
        for value in frames["train"][field]:
            counts.update(parse_protocols(value))
        # Frequency-based selection uses training only; deterministic tie order.
        vocabulary[field] = [name for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:32] if count >= 5]
    feature_names = [f"{field}__present_{token}" for field in FIELDS for token in vocabulary[field] + ["UNSEEN"]]
    output.mkdir(parents=True)
    for split in ids:
        extra = protocol_matrix(frames[split], vocabulary)
        X = np.concatenate([np.load(base / f"X_{split}.npy", mmap_mode="r"), extra], axis=1)
        np.save(output / f"X_{split}.npy", X)
        np.save(output / f"feature_hash_{split}.npy", pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy())
        for prefix in ["y", "row_ids", "window_start", "window_end"]:
            shutil.copy2(base / f"{prefix}_{split}.npy", output / f"{prefix}_{split}.npy")
    manifest["feature_names"] += feature_names
    manifest["protocol_vocabulary"] = vocabulary
    manifest["parent_manifest_sha256"] = hashlib.sha256((base / "manifest.json").read_bytes()).hexdigest()
    manifest["feature_experiment"] = "Original numeric features plus training-only protocol-presence vocabulary; identical row IDs and labels"
    manifest["limitations"].append("Added protocol identities may still encode capture/device differences; external-capture validation remains necessary.")
    write_json(output / "manifest.json", manifest)
    print(f"Protocol features prepared: {len(manifest['feature_names'])} features; same train/validation/test row IDs; test not evaluated.")


if __name__ == "__main__":
    main()
