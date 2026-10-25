"""Independent research API; no model dependency is imported by this module."""
from __future__ import annotations

import csv
import io
import math
from typing import Callable

from flask import Blueprint, Flask, jsonify, request

from framework.app.labels import display_label

MODEL_IDS = {"random_forest", "xgboost", "kan", "hem"}
IGNORED_COLUMNS = {"label", "label1", "label2", "label3", "label4", "source_row", "id", "index", "timestamp", "time"}


def register_model_workspace(app: Flask, get_registry: Callable):
    """Register scoped, explicitly selected model operations and comparison."""
    routes = Blueprint("model_workspace", __name__)

    def call(model_id: str, operation: str, payload: dict):
        try:
            return get_registry().call(model_id, operation, payload)
        except Exception as error:
            if hasattr(error, "model_id") and hasattr(error, "status"):
                raise
            from framework.core.contracts import CapabilityError
            status = "incompatible" if isinstance(error, ValueError) else "unavailable" if isinstance(error, ImportError) else "failed"
            raise CapabilityError(model_id, operation, status, str(error)) from error

    def metadata(model_id: str) -> dict:
        if model_id not in MODEL_IDS:
            raise ValueError("Choose Random Forest, XGBoost, KAN or HEM")
        return call(model_id, "metadata", {})

    def names_for(info: dict) -> list[str]:
        names = info.get("feature_names") or info.get("schema", {}).get("feature_names")
        if not isinstance(names, list) or not names:
            raise ValueError("This model has no verified feature schema")
        return names

    def validate_rows(rows, names: list[str], model_id: str):
        if not isinstance(rows, list) or not 1 <= len(rows) <= 500:
            raise ValueError("Supply 1 to 500 ordered feature rows")
        for row in rows:
            if not isinstance(row, list) or len(row) != len(names):
                raise ValueError(f"Expected exactly {len(names)} ordered values per row for {model_id}")
            if any(value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)) for value in row):
                raise ValueError("Inputs must be finite numbers or null for missing measurements")
            if all(value is None for value in row):
                raise ValueError("A row has no usable measured inputs")
        return rows

    def body() -> dict:
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        return data

    def selected_rows(model_id: str, data: dict):
        info = metadata(model_id)
        names = names_for(info)
        if "sample_id" in data:
            selected = data["sample_id"]
            snapshot = call("xgboost", "snapshot", {})
            samples = snapshot["samples"]
            if type(selected) is not int or not 0 <= selected < len(samples):
                raise ValueError("Choose a valid shared validation sample")
            sample = samples[selected]
            measured = dict(zip(snapshot["feature_names"], sample["features"]))
            absent = [name for name in names if name not in measured]
            if absent:
                raise ValueError("The shared sample cannot supply this model's feature schema: " + ", ".join(absent[:5]))
            rows = [[measured[name] for name in names]]
            source = {"kind": "Shared source-validation observation", "sample_id": selected,
                      "source_row": sample.get("source_row"), "validation_label": sample.get("label")}
        else:
            rows = data.get("rows")
            source = {"kind": "Ordered model feature inputs", "feature_names": names}
        return validate_rows(rows, names, model_id), source, info

    def prediction_result(model_id: str, rows: list, source: dict, info: dict):
        predictions = call(model_id, "predict", {"rows": rows})
        for prediction in predictions:
            prediction["display_label"] = display_label(prediction["label"])
            for category in prediction.get("top_classes", []):
                category["display_label"] = display_label(category["label"])
        return jsonify(model_id=model_id, predictions=predictions, source=source,
                       metadata=info, features=rows)

    @routes.get("/api/models")
    def catalog():
        return jsonify(models=get_registry().describe())

    @routes.get("/api/models/comparison")
    def comparison():
        return jsonify(get_registry().comparison())

    @routes.get("/api/models/<model_id>/status")
    def status(model_id):
        return jsonify(model_id=model_id, metadata=metadata(model_id))

    @routes.post("/api/models/<model_id>/predict")
    def predict(model_id):
        rows, source, info = selected_rows(model_id, body())
        return prediction_result(model_id, rows, source, info)

    @routes.post("/api/models/<model_id>/explain")
    def explain(model_id):
        data = body()
        rows, source, info = selected_rows(model_id, data)
        if len(rows) != 1:
            raise ValueError("Choose one observation for explanation")
        method = data.get("method", "native")
        if not isinstance(method, str) or len(method) > 40:
            raise ValueError("Choose a supported explanation method")
        explanation = call(model_id, "explain", {"rows": rows, "method": method})
        return jsonify(model_id=model_id, explanation=explanation, source=source, metadata=info)

    @routes.post("/api/models/<model_id>/upload")
    def upload(model_id):
        info = metadata(model_id)
        names = names_for(info)
        file = request.files.get("file")
        if file is None or not file.filename or not file.filename.lower().endswith(".csv"):
            raise ValueError("Choose a UTF-8 prepared traffic .csv file")
        try:
            reader = csv.reader(io.StringIO(file.read().decode("utf-8-sig")), strict=True)
            header = next(reader)
            if not header or any(not name.strip() for name in header) or len(header) != len(set(header)):
                raise ValueError("CSV header names must be nonempty and unique; duplicate columns are rejected")
            expected = set(names)
            unknown = [name for name in header if name not in expected and name not in IGNORED_COLUMNS]
            if unknown:
                raise ValueError("Unrecognized prepared feature columns for this model: " + ", ".join(unknown[:5]))
            if not expected.intersection(header):
                raise ValueError("CSV has no recognized measurements for this model")
            allow_missing = info.get("partial_input_supported", False) or info.get("capabilities", {}).get("partial_inputs", False)
            missing = [name for name in names if name not in header]
            if missing and not allow_missing:
                raise ValueError("This full-input model requires every feature column: " + ", ".join(missing[:5]))
            indices = {name: index for index, name in enumerate(header)}
            rows = []
            for index, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise ValueError(f"CSV row {index} has a different number of cells from its header")
                converted = [None if name not in indices or not row[indices[name]].strip()
                             else float(row[indices[name]]) for name in names]
                rows.append(converted)
                if len(rows) > 500:
                    raise ValueError("CSV must contain at most 500 rows")
        except (UnicodeError, csv.Error, StopIteration) as error:
            raise ValueError("Upload a valid UTF-8 prepared CSV with a header and measured rows") from error
        validate_rows(rows, names, model_id)
        source = {"kind": "Prepared traffic CSV", "rows": len(rows), "feature_names": names,
                  "labels_used_as_features": False, "saved_to_disk": False}
        return prediction_result(model_id, rows, source, info)

    app.register_blueprint(routes)
