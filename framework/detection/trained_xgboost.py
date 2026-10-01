"""CPU inference using a frozen research checkpoint; never a demo fallback."""
from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

MAX_CSV_ROWS = 500
PROTOCOL_FIELDS = ["network_protocols_all", "network_protocols_src", "network_protocols_dst"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_features(values) -> list:
    return [float(v) if np.isfinite(v) else None for v in values]


def default_bundle(root: Path | None = None) -> Path:
    """Select only an explicitly promoted upgrade; never modify the original."""
    root = root or Path(__file__).resolve().parents[2]
    for name in ["xgboost_partial_enhanced", "xgboost_partial"]:
        upgraded = root / "framework/models" / name / "bundle.json"
        if upgraded.is_file():
            bundle = json.loads(upgraded.read_text(encoding="utf-8"))
            if bundle.get("promoted") is True and bundle.get("promotion", {}).get("improved") is True:
                return upgraded
    return root / "framework/models/xgboost_app/bundle.json"


def transform_csv(content: bytes, bundle: dict) -> np.ndarray:
    """Apply frozen training feature order and protocol vocabulary to CSV.

    Accept raw DataSense rows or all explicitly prepared feature columns.
    Labels/identifiers are ignored. No new vocabulary or normalization is fit.
    """
    try:
        text = content.decode("utf-8-sig")
        # Detect duplicate names before pandas silently disambiguates them.
        import csv
        reader = csv.reader(io.StringIO(text), strict=True)
        header = next(reader)
        if len(header) != len(set(header)):
            raise ValueError("CSV contains duplicate column names")
        if not header or any(not name.strip() for name in header):
            raise ValueError("CSV feature columns need nonempty names")
        for row_index, row in enumerate(reader):
            if row and len(row) != len(header):
                raise ValueError(f"CSV row {row_index + 2} has a different number of cells from its header")
            if row_index > MAX_CSV_ROWS:
                break
        frame = pd.read_csv(io.StringIO(text), nrows=MAX_CSV_ROWS + 1)
    except (UnicodeError, csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError, StopIteration) as exc:
        raise ValueError("Upload a valid UTF-8 CSV with a header") from exc
    if not 1 <= len(frame) <= MAX_CSV_ROWS:
        raise ValueError(f"CSV must contain 1 to {MAX_CSV_ROWS} rows; export a smaller batch")
    if bundle.get("schema_version") == 2 and bundle.get("partial_input_supported") is True:
        from framework.data.feature_schema import transform_partial_frame
        return transform_partial_frame(frame, bundle)
    names = bundle["feature_names"]
    numeric = bundle["numeric_features"]
    missing = [name for name in numeric if name not in frame]
    if missing:
        raise ValueError("Missing numeric columns: " + ", ".join(missing[:8]))
    numeric_values = frame[numeric].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    numeric_values[~np.isfinite(numeric_values)] = np.nan  # same research preprocessing
    numeric_values[numeric_values == 0] = 0
    if all(name in frame for name in names):
        flags = frame[names[len(numeric):]].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        if not np.isin(flags, [0, 1]).all():
            raise ValueError("Prepared protocol flags must be explicit zero or one values")
    else:
        if not all(field in frame for field in PROTOCOL_FIELDS):
            raise ValueError("CSV needs all prepared feature columns or the three raw network_protocols fields")
        flags = np.zeros((len(frame), len(names) - len(numeric)), dtype=np.float32)
        offset = 0
        for field in PROTOCOL_FIELDS:
            vocabulary = bundle["protocol_vocabulary"][field]
            mapping = {token: i for i, token in enumerate(vocabulary)}
            for row, text in enumerate(frame[field]):
                if not isinstance(text, str) or len(text) > 8192:
                    raise ValueError(f"{field} requires a bounded Python list string, e.g. ['tcp']")
                try:
                    tokens = ast.literal_eval(text)
                except (ValueError, SyntaxError, TypeError, RecursionError) as exc:
                    raise ValueError(f"Invalid protocol list in {field}, row {row + 1}") from exc
                if not isinstance(tokens, list) or any(not isinstance(t, str) or len(t) > 80 for t in tokens):
                    raise ValueError(f"Invalid protocol list in {field}, row {row + 1}")
                for token in set(tokens):
                    flags[row, offset + mapping.get(token, len(mapping))] = 1
            offset += len(vocabulary) + 1
    if np.isnan(numeric_values).all(axis=1).any():
        raise ValueError("A row has no observed numeric features")
    return np.concatenate([numeric_values, flags], axis=1)


class XGBoostDetector:
    """Saved-model classification with explicit CPU inference and exact TreeSHAP."""

    def __init__(self, bundle_path: str | Path):
        import xgboost as xgb
        self.bundle_path = Path(bundle_path).resolve()
        if not self.bundle_path.is_file():
            raise FileNotFoundError("App bundle missing. Run: python -m experiment.package_app")
        self.bundle = json.loads(self.bundle_path.read_text(encoding="utf-8"))
        if self.bundle.get("schema_version") not in (1, 2):
            raise ValueError("Unsupported app bundle version")
        self.partial_input_supported = self.bundle.get("schema_version") == 2 and self.bundle.get("partial_input_supported") is True
        if self.bundle.get("schema_version") == 2 and not self.partial_input_supported:
            raise ValueError("Schema 2 requires an explicit missing-input training contract")
        self.checkpoint = (self.bundle_path.parent / self.bundle["checkpoint"]).resolve()
        if file_sha256(self.checkpoint) != self.bundle["checkpoint_sha256"]:
            raise ValueError("Trained checkpoint checksum differs from the frozen app bundle")
        self.feature_names = self.bundle["feature_names"]
        self.classes = self.bundle["classes"]
        self.samples = self.bundle["samples"]
        self.booster = xgb.Booster(model_file=str(self.checkpoint))
        self.booster.set_param({"device": "cpu", "nthread": 4})
        config = json.loads(self.booster.save_config())["learner"]
        if (int(config["learner_model_param"]["num_class"]) != len(self.classes)
                or self.booster.num_features() != len(self.feature_names)
                or config["objective"]["name"] != "multi:softprob"
                or config["gradient_booster"]["name"] != "gbtree"
                or config["generic_param"]["device"] != "cpu"):
            raise ValueError("Checkpoint and frozen feature/class schema disagree")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Duplicate feature names")
        numeric = self.bundle["numeric_features"]
        expected = numeric + [f"{field}__present_{token}" for field in PROTOCOL_FIELDS
                              for token in self.bundle["protocol_vocabulary"][field] + ["UNSEEN"]]
        if expected != self.feature_names or any(not name.startswith(("network_", "log_")) for name in numeric):
            raise ValueError("Frozen preprocessing schema is inconsistent")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("Duplicate class names")
        if self.classes[self.bundle["benign_index"]] != "benign":
            raise ValueError("Invalid benign class mapping")

    def status(self) -> dict:
        from framework.data.feature_schema import compatibility
        return {"model": self.bundle["model"], "dataset": "DataSense",
                "partial_input_supported": self.partial_input_supported,
                "feature_compatibility": compatibility(self.feature_names),
                "live_profile_validation": self.bundle.get("live_profile_validation"),
                "execution": {"device": "cpu", "xai": "Exact TreeSHAP (native XGBoost)",
                              "reasoning": "Offline template", "llm_executed": False,
                              "novelty_available": False},
                "max_csv_rows": MAX_CSV_ROWS, "max_upload_bytes": 10 * 1024 * 1024,
                "limitations": ["Validation-selected 84-class DataSense model; test scoring remains locked.",
                                "Within-capture validation does not establish unseen-device/capture accuracy.",
                                "Unknown attacks are not evaluated. Model scores are uncalibrated.",
                                ("Partial and live inputs preserve unavailable features as missing; packet-profile validation is not live-PC accuracy."
                                 if self.partial_input_supported else "This checkpoint accepts full DataSense CSVs; the separate binary packet model is experimental."),
                                "Exact SHAP explains raw class margins, not causal effects or incident confirmation."]}

    def validate_features(self, values) -> np.ndarray:
        if not isinstance(values, list) or len(values) != len(self.feature_names):
            raise ValueError(f"Expected exactly {len(self.feature_names)} ordered feature values")
        if any(v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))) for v in values):
            raise ValueError("Features must be numbers or null for missing numeric values")
        try:
            X = np.array([np.nan if v is None else v for v in values], dtype=np.float32)
        except (ValueError, OverflowError) as exc:
            raise ValueError("Feature value exceeds the supported numeric range") from exc
        observed = X if self.partial_input_supported else X[:len(self.bundle["numeric_features"])]
        if np.isinf(X).any() or not np.isfinite(observed).any():
            raise ValueError("Provide finite numeric observations; null is allowed for individual missing values")
        flags = X[len(self.bundle["numeric_features"]):]
        if self.partial_input_supported:
            flags = flags[np.isfinite(flags)]
        if not np.isin(flags, [0, 1]).all():
            raise ValueError("Protocol presence features must be zero or one")
        return X

    def predict_batch(self, X: np.ndarray) -> list[dict]:
        import xgboost as xgb
        probabilities = self.booster.predict(xgb.DMatrix(X, nthread=4), strict_shape=True)
        results = []
        for row, scores in zip(X, probabilities):
            index = int(scores.argmax())
            top = np.argsort(-scores, kind="stable")[:3]
            results.append({"label": self.classes[index], "is_attack": index != self.bundle["benign_index"],
                            "score": float(scores[index]), "novelty": "not evaluated",
                            "tentative": bool(np.isnan(row).any()),
                            "top_classes": [{"label": self.classes[int(i)], "score": float(scores[i])} for i in top]})
        return results

    def analyze(self, values: list, source: dict) -> dict:
        from framework.data.feature_schema import feature_coverage
        from framework.xai.tree_shap import explain_tree
        from framework.llm.research_template import ResearchTemplateProvider
        X = self.validate_features(values).reshape(1, -1)
        prediction = self.predict_batch(X)[0]
        explanation = explain_tree(self.booster, X, self.feature_names, self.classes, prediction["label"])
        context = {"prediction": prediction, "explanation": explanation, "source": source,
                   "model": self.bundle["model"], "features": json_features(X[0]),
                   "feature_coverage": feature_coverage(X[0], self.bundle)}
        return {**context, "reasoning": ResearchTemplateProvider().generate(context)}
