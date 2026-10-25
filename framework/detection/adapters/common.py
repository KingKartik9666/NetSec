"""Dependency-free manifest checks shared by optional model workers.

Heavy numerical libraries are imported only while executing predictions. A
manifest describes trusted project artifacts; arbitrary uploaded pickle/model
files are never accepted by these adapters.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib.util import find_spec
import json
import math
from pathlib import Path, PureWindowsPath
import re
from typing import Any


def artifact_path(root: Path, relative: str) -> Path:
    """Resolve a portable project-relative path without permitting escape."""
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("An artifact path must be a nonempty project-relative string.")
    portable = relative.replace("\\", "/")
    if (Path(portable).is_absolute() or PureWindowsPath(relative).is_absolute()
            or ":" in portable or ".." in portable.split("/")):
        raise ValueError(f"Artifact path must remain within the project: {relative}")
    base = Path(root).resolve()
    resolved = (base / portable).resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"Artifact path escapes the project: {relative}")
    return resolved


def load_manifest(config: dict[str, Any], root: Path, expected_model_id: str) -> dict[str, Any]:
    """Read model metadata without importing a model dependency or weights."""
    if not isinstance(config, dict):
        raise ValueError("Model configuration must be an object.")
    path = artifact_path(root, config.get("manifest", ""))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("model_id") != expected_model_id:
        raise ValueError(f"Manifest model_id must be {expected_model_id!r}.")
    for key, minimum in (("feature_names", 1), ("classes", 2)):
        values = manifest.get(key)
        if (not isinstance(values, list) or len(values) < minimum
                or any(not isinstance(value, str) or not value.strip() for value in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"Manifest {key} must contain unique nonempty names.")
    benign = manifest.get("benign_index")
    if isinstance(benign, bool) or not isinstance(benign, int) or not 0 <= benign < len(manifest["classes"]):
        raise ValueError("Manifest benign_index must identify a declared class.")
    if not isinstance(manifest.get("version"), str) or not manifest["version"]:
        raise ValueError("Manifest requires a model version.")
    if not isinstance(manifest.get("config", {}), dict):
        raise ValueError("Manifest config must be an object.")
    checkpoint = manifest.get("checkpoint")
    artifact_path(root, checkpoint)
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or checkpoint not in hashes:
        raise ValueError("Manifest must pin its checkpoint in artifact_sha256.")
    for relative, digest in hashes.items():
        artifact_path(root, relative)
        if not isinstance(digest, str) or re.fullmatch(r"[a-fA-F0-9]{64}", digest) is None:
            raise ValueError(f"Invalid SHA256 digest for {relative}.")
    for optional in ("preprocessing", "knn_references"):
        if manifest.get(optional) is not None:
            artifact_path(root, manifest[optional])
            if manifest[optional] not in hashes:
                raise ValueError(f"Manifest must pin its {optional} artifact.")
    manifest.setdefault("metrics", None)
    manifest.setdefault("config", {})
    manifest.setdefault("limitations", [])
    return manifest


def verify_artifacts(manifest: dict[str, Any], root: Path) -> None:
    """Check every recorded artifact before deserializing or importing it."""
    for relative, expected in manifest["artifact_sha256"].items():
        path = artifact_path(root, relative)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected.lower():
            raise ValueError(f"SHA256 mismatch for model artifact: {relative}")


def verify_source_record(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    """Bind ordered labels/features to the hash-pinned completed training record.

Older training records may omit a benign_index field; the checkpoint still
validates that index. Record identity prevents a same-length feature or label
reordering from being accepted as compatible merely because tensor shapes fit.
"""
    relative = manifest.get("source_record")
    if not isinstance(relative, str) or relative not in manifest["artifact_sha256"]:
        raise ValueError("Model manifest requires a hash-pinned source_record.")
    verify_artifacts({"artifact_sha256": {relative: manifest["artifact_sha256"][relative]}}, root)
    record = json.loads(artifact_path(root, relative).read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("candidate", record.get("model_id")) != manifest["model_id"]:
        raise ValueError("Model source_record candidate differs from the selected family.")
    if "model_id" in record and record["model_id"] != manifest["model_id"]:
        raise ValueError("Model source_record model_id conflicts with the selected family.")
    names = record.get("features", record.get("feature_names"))
    if names != manifest["feature_names"]:
        raise ValueError("Model ordered feature schema differs from the source_record.")
    if record.get("classes") != manifest["classes"]:
        raise ValueError("Model ordered class map differs from the source_record.")
    if record.get("config") != manifest["config"]:
        raise ValueError("Model configuration differs from the source_record.")
    if record.get("upstream_commit") != manifest.get("upstream_commit"):
        raise ValueError("Model upstream commit differs from the source_record.")
    if record.get("benign_index") is not None and record["benign_index"] != manifest["benign_index"]:
        raise ValueError("Model benign class index differs from the source_record.")
    return record


def dependency_health(names: tuple[str, ...]) -> dict[str, str]:
    """Inspect import availability without executing expensive imports."""
    for name in names:
        try:
            present = find_spec(name) is not None
        except (ImportError, AttributeError, ValueError):
            present = False
        if not present:
            return {"status": "unavailable", "reason": f"Optional model dependency is missing: {name}"}
    return {"status": "ready", "reason": "Required model dependencies are discoverable."}


def failure_health(error: Exception) -> dict[str, str]:
    """Classify a scoped failure without selecting another model."""
    if isinstance(error, (FileNotFoundError, ImportError)):
        status = "unavailable"
    elif isinstance(error, ValueError) and "SHA256" not in str(error):
        status = "incompatible"
    else:
        status = "failed"
    return {"status": status, "reason": str(error)}


def rows_array(rows: list[list[float | None]], feature_names: list[str], *, allow_missing: bool = False):
    """Validate bounded aligned JSON measurements, then lazily create float32."""
    if not isinstance(rows, list) or not 1 <= len(rows) <= 500:
        raise ValueError("Provide between one and 500 observation rows.")
    cleaned: list[list[float]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(feature_names):
            raise ValueError(f"Each row must contain exactly {len(feature_names)} ordered features.")
        values = []
        for feature, value in zip(feature_names, row):
            if value is None and allow_missing:
                values.append(float("nan"))
                continue
            if value is None:
                raise ValueError(f"Full-only model requires measured feature {feature}; sparse inputs are incompatible.")
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError(f"Feature {feature} must be a numeric measurement, not text or boolean.")
            numeric = float(value)
            if not math.isfinite(numeric) or abs(numeric) > 3.4028234663852886e38:
                raise ValueError(f"Feature {feature} must be finite and representable as float32.")
            values.append(numeric)
        if not any(math.isfinite(value) for value in values):
            raise ValueError("An observation cannot have all features missing.")
        cleaned.append(values)
    import numpy as np
    return np.asarray(cleaned, dtype=np.float32)


def class_results(probabilities, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize real uncalibrated closed-set scores, with no novelty claim."""
    import numpy as np
    values = np.asarray(probabilities)
    if (values.ndim != 2 or values.shape[1] != len(manifest["classes"])
            or not np.isfinite(values).all() or np.any(values < 0)
            or not np.allclose(values.sum(axis=1), 1., atol=1e-4)):
        raise ValueError("Model returned invalid class probabilities.")
    results = []
    for row in values:
        order = np.argsort(-row, kind="stable")[:min(3, len(row))]
        predicted = int(order[0])
        results.append({
            "label": manifest["classes"][predicted],
            "is_attack": predicted != manifest["benign_index"],
            "score": float(row[predicted]), "score_type": "uncalibrated_model_score",
            "novelty": "not_evaluated",
            "top_classes": [{"label": manifest["classes"][int(index)], "score": float(row[index])}
                            for index in order],
        })
    return results


def metadata_copy(manifest: dict[str, Any], capabilities: dict[str, Any]) -> dict[str, Any]:
    """Expose immutable-by-copy metadata and explicitly supported operations."""
    result = deepcopy(manifest)
    result["capabilities"] = deepcopy(capabilities)
    result["inference_device"] = "cpu"
    return result
