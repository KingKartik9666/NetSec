"""Create bounded LIME reference samples using original TRAINING rows only."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def save(name, X, feature_names, row_ids, source, categorical):
    directory = ROOT / "framework/models" / name
    target = directory / "xai_reference.npz"
    if target.exists():
        raise FileExistsError(f"Preserve the existing reference: {target}")
    values = np.asarray(X, dtype=np.float32)
    with np.errstate(all="ignore"):
        medians = np.nanmedian(values, axis=0)
    medians = np.nan_to_num(medians, nan=0, posinf=0, neginf=0)
    np.savez_compressed(target, training=values, medians=medians)
    metadata = {"kind": "Training-only LIME reference; no validation or test rows",
                "source": source, "rows": len(values), "seed": 42,
                "feature_names": feature_names, "categorical_indices": categorical,
                "selected_row_ids_sha256": hashlib.sha256(np.asarray(row_ids).tobytes()).hexdigest(),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    (directory / "xai_reference.json").write_text(json.dumps(metadata, indent=2))
    print(f"{name}: {len(values)} training reference rows, {values.shape[1]} features")


def main():
    csv_bundle = json.loads((ROOT / "framework/models/xgboost_app/bundle.json").read_text())
    # Use the same prepared protocol schema as the frozen full-data CSV model.
    source = ROOT / "dataset/prepared/datasense_full_protocol"
    if not source.exists():
        candidates = [p for p in (ROOT / "dataset/prepared").iterdir() if (p / "manifest.json").is_file()
                      and json.loads((p / "manifest.json").read_text()).get("feature_names") == csv_bundle["feature_names"]]
        if len(candidates) != 1:
            raise ValueError("Cannot uniquely identify the frozen 170-feature training source")
        source = candidates[0]
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["feature_names"] != csv_bundle["feature_names"]:
        raise ValueError("CSV training reference schema mismatch")
    X = np.load(source / "X_train.npy", mmap_mode="r")
    rows = np.load(source / "row_ids_train.npy", mmap_mode="r")
    selected = np.sort(np.random.default_rng(42).choice(len(X), min(512, len(X)), replace=False))
    save("xgboost_app", X[selected], manifest["feature_names"], rows[selected], source.relative_to(ROOT).as_posix(),
         list(range(len(csv_bundle["numeric_features"]), len(manifest["feature_names"]))))
    from framework.detection.packet_xgboost import FEATURE_NAMES
    source = ROOT / "dataset/prepared/datasense_full_train"
    manifest = json.loads((source / "manifest.json").read_text())
    X = np.load(source / "X_train.npy", mmap_mode="r")
    rows = np.load(source / "row_ids_train.npy", mmap_mode="r")
    start, end = [np.load(source / f"window_{which}_train.npy", mmap_mode="r") for which in ["start", "end"]]
    eligible = np.flatnonzero(end - start == 10_000_000_000)
    selected = np.sort(np.random.default_rng(42).choice(eligible, min(512, len(eligible)), replace=False))
    indices = [manifest["feature_names"].index(name) for name in FEATURE_NAMES]
    save("packet_live", X[np.ix_(selected, indices)], FEATURE_NAMES, rows[selected], source.relative_to(ROOT).as_posix(), [])


if __name__ == "__main__":
    main()
