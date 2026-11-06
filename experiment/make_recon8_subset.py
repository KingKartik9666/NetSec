"""Write a recon-only copy of the cascade v3 dataset: rows of the eight recon-probe subtypes, nothing else.

Owner request 2026-10-31. The source dataset is read only. Every per-row array is filtered with the same mask
(train: ``y8_train >= 0``; validation: ``y8_validation >= 0``), so original row ids, scenario ids and the frozen
validation roles (0 tuning, 1 calibration, 2 assessment) are kept. ``y8`` is the label (0..7, order of
``classes8``); y77/y84 are kept for traceability. The 77-class CSV copies are filtered to the same rows by
``source_row_id`` and their recon label taken from ``label4_original``. No test rows exist in the source.

    .venv/Scripts/python.exe -m experiment.make_recon8_subset
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dataset/experiments/csv_v3_cascade_20261030_v1"
OUT = ROOT / "dataset/experiments/csv_v3_recon8_20261031_v1"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    OUT.mkdir(parents=True)
    masks = {"train": np.load(SOURCE / "y8_train.npy") >= 0, "validation": np.load(SOURCE / "y8_validation.npy") >= 0}
    counts, hashes = {}, {}
    for path in sorted(SOURCE.glob("*.npy")):
        split = "train" if path.stem.endswith("_train") else "validation" if (
            path.stem.endswith("_validation") or path.stem.startswith("validation_")) else None
        if split is None:
            raise SystemExit(f"cannot tell the split of {path.name}")
        arr = np.load(path, mmap_mode="r")
        if len(arr) != len(masks[split]):
            raise SystemExit(f"{path.name}: {len(arr)} rows, mask has {len(masks[split])}")
        out = OUT / path.name
        np.save(out, np.ascontiguousarray(arr[masks[split]]))
        counts[path.stem] = int(masks[split].sum())
        hashes[path.name] = _sha(out)
    y8 = {s: np.load(OUT / f"y8_{s}.npy") for s in masks}
    assert all((v >= 0).all() for v in y8.values())
    ids = {s: set(np.load(OUT / f"row_ids_{s}.npy").tolist()) for s in masks}

    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    classes8 = manifest["classes8"]
    csv_rows = {}
    for split in masks:
        src = SOURCE / f"selected_{split}_77class.csv.gz"
        dst = OUT / f"selected_{split}_recon8.csv.gz"
        n = 0
        with gzip.open(dst, "wt", encoding="utf-8", newline="") as handle:
            for i, chunk in enumerate(pd.read_csv(src, chunksize=200_000, low_memory=False)):
                keep = chunk[chunk["source_row_id"].isin(ids[split])]
                bad = ~keep["label4_original"].isin(classes8)
                if bad.any():
                    raise SystemExit(f"{src.name}: non-recon label {keep.loc[bad, 'label4_original'].iloc[0]!r}")
                keep.to_csv(handle, index=False, header=i == 0)
                n += len(keep)
        if n != len(ids[split]):
            raise SystemExit(f"{dst.name}: {n} CSV rows, {len(ids[split])} array rows")
        csv_rows[split] = n
        hashes[dst.name] = _sha(dst)

    roles = np.load(OUT / "validation_roles.npy")
    manifest.update({
        "version": "csv_v3_recon8_20261031_v1",
        "subset": {"source": str(SOURCE.relative_to(ROOT)), "rule": "rows with y8 >= 0 (eight recon-probe subtypes)",
                   "label": "y8", "classes": classes8, "array_rows": counts, "csv_rows": csv_rows,
                   "per_class_train": np.bincount(y8["train"], minlength=8).tolist(),
                   "per_class_validation": np.bincount(y8["validation"], minlength=8).tolist(),
                   "validation_roles": {str(r): int((roles == r).sum()) for r in (0, 1, 2)},
                   "source_array_sha256": manifest.get("array_sha256")},
        "array_sha256": hashes, "benign_index": None,
    })
    manifest.pop("csv_copies_77class", None)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["subset"], indent=2, default=str))


if __name__ == "__main__":
    main()
