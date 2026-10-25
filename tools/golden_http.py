"""Freeze Python HTTP behavior with real tiny-model inference and labelled I/O fakes.

No network listener, capture driver, research worker, or language model is started.
The caller owns creation of the synthetic XGBoost bundle and XAI reference.
"""
from __future__ import annotations

import builtins
import base64
import copy
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from framework.app.server import create_app


# Public fixture input, never an actual desktop-session credential.
FIXTURE_TOKEN = "0123456789abcdef" * 4
FIXTURE_TIME = "2000-01-01T00:00:00+00:00"
UNAVAILABLE_MESSAGE = "Synthetic transport fixture: local LLM execution is disabled"


class _RegistryFixture:
    """Only XGBoost executes; absent research families supply no predictions."""

    ids = ("random_forest", "xgboost", "kan", "hem")
    configuration = {"plugins": {"language": True}}

    def __init__(self, detector):
        self.detector = detector
        self.broken = set()

    def describe(self):
        return [{"model_id": model_id, "status": "ready" if model_id == "xgboost" else "unavailable",
                 "provenance": "Synthetic HTTP registry fixture; no research metrics",
                 "capabilities": {"predict": model_id == "xgboost", "explain": model_id == "xgboost"}}
                for model_id in self.ids]

    def comparison(self):
        return {"comparable": False, "reason": "Synthetic HTTP fixture has no research metrics",
                "models": [{"model_id": model_id, "metrics": None, "status": "unavailable"}
                           for model_id in self.ids]}

    def call(self, model_id, operation, payload):
        if model_id in self.broken:
            raise ImportError("Injected synthetic model dependency removal")
        if model_id != "xgboost":
            raise ImportError("No research checkpoint is loaded in this synthetic HTTP fixture")
        detector = self.detector
        if operation == "metadata":
            return {"model_id": model_id, "name": "Real XGBoost on a tiny synthetic fixture",
                    "feature_names": detector.feature_names, "classes": detector.classes,
                    "partial_input_supported": detector.partial_input_supported,
                    "capabilities": {"predict": True, "explain": True}, "metrics": None,
                    "dataset": "Synthetic software contract fixture"}
        if operation == "snapshot":
            return {"feature_names": detector.feature_names, "classes": detector.classes,
                    "samples": copy.deepcopy(detector.samples)}
        if operation == "predict":
            rows = np.vstack([detector.validate_features(row) for row in payload["rows"]])
            return detector.predict_batch(rows)
        if operation == "explain":
            if payload["method"] != "native":
                raise ValueError("Synthetic registry supports native explanation only")
            return detector.analyze(payload["rows"][0], {"kind": "Synthetic HTTP fixture"})["explanation"]
        raise ValueError("Unsupported synthetic registry operation")

    def close(self):
        pass


class _MonitorFixture:
    """In-memory transport state. It never reads OS counters or captures packets."""

    def __init__(self, detector):
        self.detector = detector
        self.running = False
        self.report = None

    def capabilities(self):
        return {"interfaces": [{"name": "synthetic-fixture", "description": "Synthetic HTTP transport fixture"}],
                "recommended_interface": "synthetic-fixture", "connections_available": True,
                "packet_capture_available": False, "model_compatible": True,
                "packet_capture_message": "Synthetic fixture: no OS capture is performed"}

    def snapshot(self):
        return {"running": self.running, "state": "running" if self.running else "stopped",
                "mode": "connections", "interface": "synthetic-fixture", "error": None,
                "fixture_provenance": "Synthetic transport state; not OS observation or live validation",
                "metrics": {"packets_observed": 5 if self.running else 0},
                "assessment": {"status": "benign_looking", "label": "Synthetic rule transport fixture"},
                "model_prediction": None, "multiclass_prediction": copy.deepcopy(self.report)}

    def start(self, interface, mode):
        if interface != "synthetic-fixture" or mode != "connections":
            raise ValueError("Synthetic fixture supports only its explicitly named connection adapter")
        self.running = True
        self.report = self.detector.analyze(self.detector.samples[0]["features"],
                                           {"kind": "Synthetic HTTP observation fixture", "synthetic": True})
        self.report["observation_id"] = "synthetic-observation-0"
        return self.snapshot()

    def stop(self):
        self.running = False
        self.report = None
        return self.snapshot()


class _UnavailableLanguageFixture:
    def status(self):
        return {"available": False, "offline": True, "device": "cpu", "error": UNAVAILABLE_MESSAGE,
                "automatic_generation": False, "llm_executed": False,
                "rag": {"available": False, "error": UNAVAILABLE_MESSAGE}}

    def generate(self, evidence):
        raise RuntimeError(UNAVAILABLE_MESSAGE)


class _AdapterOperations:
    """Transport-only registry substitute calling the actual default adapter.

    Multiprocessing itself is covered by the baseline suite. This lets HTTP
    fixtures freeze LazyXGBoostFacade's current report shape without OS workers.
    """

    def __init__(self, detector):
        from framework.detection.adapters.xgboost import XGBoostModelAdapter
        self.adapter = XGBoostModelAdapter({"bundle_path": str(detector.bundle_path),
                                           "plugins": {"explanation": True, "lime": True}},
                                          Path(__file__).resolve().parents[1])

    def call(self, model_id, operation, payload):
        if model_id != "xgboost":
            raise ValueError("Default-facade fixture supports XGBoost only")
        if operation == "snapshot":
            return self.adapter.snapshot()
        if operation == "legacy_status":
            return self.adapter.legacy_status()
        if operation == "analyze":
            return self.adapter.analyze(payload["values"], payload.get("source", {}))
        if operation == "validate_features":
            return self.adapter.validate_features(payload["values"])
        if operation == "legacy_predict":
            return self.adapter.legacy_predict(payload["rows"])
        if operation == "additional_xai":
            return self.adapter.additional_xai(payload["values"])
        raise ValueError("Unsupported default-facade fixture operation")


class _FixedDatetime:
    @classmethod
    def now(cls, tz=None):
        return datetime(2000, 1, 1, tzinfo=timezone.utc)


def _schema(value):
    """Compact structural observations, retaining null/array alternatives."""
    if value is None:
        return "null"
    if isinstance(value, dict):
        return {key: _schema(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        types = {json.dumps(_schema(item), sort_keys=True) for item in value}
        return {"array_items": [json.loads(item) for item in sorted(types)]}
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def _csv_bytes(detector, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(detector.feature_names)
    writer.writerows([["" if value is None else value for value in row] for row in rows])
    return stream.getvalue().encode("utf-8")


def generate_http_goldens(output_dir: Path, detector) -> dict:
    """Create ``output_dir/http`` once and return its portable evidence summary.

    Responses are captured unchanged. Volatile transport headers are explicitly
    omitted, and the live-report clock is fixed before execution. Fixture labels
    supersede hardcoded runtime DataSense/live descriptions as provenance.
    """
    directory = Path(output_dir) / "http"
    directory.mkdir(parents=True, exist_ok=False)
    responses = directory / "responses"
    responses.mkdir()
    cases = []
    registry = _RegistryFixture(detector)
    monitor = _MonitorFixture(detector)
    language = _UnavailableLanguageFixture()
    app = create_app(detector=detector, monitor=monitor, registry=registry,
                     language_service=language, session_token=FIXTURE_TOKEN)
    client = app.test_client()

    def record(case_id, method, path, expected, *, selected_client=None, json_body=None,
               raw=None, content_type=None, upload=None, headers=None, authenticated=True,
               repeat_body=None, note=None):
        request_headers = {"X-IMDAI-Session": FIXTURE_TOKEN} if authenticated else {}
        request_headers.update(headers or {})
        body = None
        descriptor = None
        if json_body is not None:
            body = json.dumps(json_body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            content_type = "application/json"
        elif upload is not None:
            filename, payload = upload
            boundary = "NETSEC-SYNTHETIC-GOLDEN-BOUNDARY"
            body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                    "Content-Type: text/csv\r\n\r\n").encode() + payload + f"\r\n--{boundary}--\r\n".encode()
            content_type = f"multipart/form-data; boundary={boundary}"
        elif repeat_body is not None:
            pattern, count = repeat_body
            body = pattern.encode() * count
            descriptor = {"encoding": "utf-8-repeat", "text": pattern, "count": count}
        elif raw is not None:
            body = raw if isinstance(raw, bytes) else raw.encode()
        if body is not None and descriptor is None:
            try:
                descriptor = {"encoding": "utf-8", "text": body.decode("utf-8")}
            except UnicodeDecodeError:
                descriptor = {"encoding": "base64", "data": base64.b64encode(body).decode("ascii")}
        if content_type:
            request_headers["Content-Type"] = content_type
        result = (selected_client or client).open(path, method=method, data=body, headers=request_headers)
        if result.status_code != expected:
            raise AssertionError(f"{case_id}: expected HTTP {expected}, got {result.status_code}: {result.get_data(as_text=True)[:500]}")
        response_bytes = result.get_data()
        extension = "json" if result.is_json else "html" if result.mimetype == "text/html" else "txt"
        body_path = f"responses/{case_id}.{extension}"
        (directory / body_path).write_bytes(response_bytes)
        stable_headers = {key: value for key, value in result.headers.items()
                          if key.lower() not in {"date", "last-modified", "etag"}}
        entry = {"id": case_id, "request": {"method": method, "path": path,
                                           "headers": request_headers, "body": descriptor},
                 "response": {"status": result.status_code, "headers": stable_headers,
                              "body_file": body_path, "body_sha256": hashlib.sha256(response_bytes).hexdigest(),
                              "json_schema": _schema(result.get_json()) if result.is_json and response_bytes else None}}
        if note:
            entry["note"] = note
        cases.append(entry)
        return result

    sample = detector.samples[0]["features"]
    csv_payload = _csv_bytes(detector, [sample])
    try:
        for path, name in [("/", "home"), ("/live", "live"), ("/csv", "csv"), ("/samples", "samples"),
                           ("/models", "models"), ("/loading", "loading"), ("/error", "error"), ("/exit", "exit")]:
            record("page_" + name, "GET", path, 200)
            record("auth_page_" + name, "GET", path, 403, authenticated=False)
        for name, path in [("exit_allowed", "/exit?return=/csv"), ("exit_models", "/exit?return=/models"),
                           ("exit_foreign", "/exit?return=https://outside.example"),
                           ("exit_traversal", "/exit?return=//outside.example")]:
            record(name, "GET", path, 200)
        record("static_asset", "GET", "/static/netsec.svg", 200)
        record("static_auth", "GET", "/static/netsec.svg", 403, authenticated=False)
        record("status", "GET", "/api/status", 200)
        record("status_head", "HEAD", "/api/status", 200)
        record("analyze_options", "OPTIONS", "/api/analyze", 200)
        record("options_auth_missing", "OPTIONS", "/api/analyze", 403, authenticated=False)
        record("samples", "GET", "/api/samples", 200)
        record("example_csv", "GET", "/api/example.csv", 200)
        record("analyze", "POST", "/api/analyze", 200, json_body={"sample_id": 0})
        record("upload", "POST", "/api/upload", 200, upload=("fixture.csv", csv_payload))
        record("explain", "POST", "/api/explain", 200, json_body={"features": sample, "row_index": 0})
        record("xai", "POST", "/api/xai", 200, json_body={"kind": "sample", "sample_id": 0})
        record("assistant_status", "GET", "/api/assistant/status", 200)
        record("assistant_unavailable", "POST", "/api/assistant/explain", 500,
               json_body={"kind": "sample", "sample_id": 0}, note="Unavailable language fake; no generation executed")
        # Record the real provider's unavailable contract as well as the above
        # transport double. A private empty directory guarantees no weights load.
        from framework.llm.local_model import LocalLanguageModel
        with TemporaryDirectory(prefix="imdai-missing-llm-") as temporary:
            with patch.dict("os.environ", {"IMDAI_LLM_DIR": temporary}):
                missing_language = LocalLanguageModel()
            missing_language_app = create_app(detector=detector, registry=registry,
                                              language_service=missing_language,
                                              session_token=FIXTURE_TOKEN)
            try:
                missing_language_client = missing_language_app.test_client()
                record("real_assistant_unavailable_status", "GET", "/api/assistant/status", 200,
                       selected_client=missing_language_client,
                       note="Real LocalLanguageModel status with an empty GGUF directory; no inference executed")
                record("real_assistant_unavailable_explain", "POST", "/api/assistant/explain", 400,
                       selected_client=missing_language_client, json_body={"kind": "sample", "sample_id": 0},
                       note="Real provider refusal; no template substitutes for missing GGUF")
            finally:
                missing_language_app.extensions["close_resources"]()
        record("model_catalog", "GET", "/api/models", 200)
        record("model_comparison", "GET", "/api/models/comparison", 200)
        record("model_status", "GET", "/api/models/xgboost/status", 200)
        record("model_predict", "POST", "/api/models/xgboost/predict", 200, json_body={"rows": [sample]})
        record("model_sample", "POST", "/api/models/xgboost/predict", 200, json_body={"sample_id": 0})
        record("model_explain", "POST", "/api/models/xgboost/explain", 200, json_body={"rows": [sample]})
        record("model_upload", "POST", "/api/models/xgboost/upload", 200, upload=("fixture.csv", csv_payload))
        for family in ("random_forest", "kan", "hem"):
            record("unavailable_" + family, "GET", f"/api/models/{family}/status", 503)
        record("model_unknown", "GET", "/api/models/unknown/status", 400)
        record("model_empty_rows", "POST", "/api/models/xgboost/predict", 400, json_body={"rows": []})
        record("model_bad_shape", "POST", "/api/models/xgboost/predict", 400, json_body={"rows": [[]]})
        record("model_null_row", "POST", "/api/models/xgboost/predict", 400,
               json_body={"rows": [[None] * len(sample)]})
        record("model_bool_sample", "POST", "/api/models/xgboost/predict", 400, json_body={"sample_id": True})
        record("model_multi_explain", "POST", "/api/models/xgboost/explain", 400, json_body={"rows": [sample, sample]})
        record("model_unknown_method", "POST", "/api/models/xgboost/explain", 400,
               json_body={"rows": [sample], "method": "unknown"})
        record("model_row_limit", "POST", "/api/models/xgboost/upload", 400,
               upload=("fixture.csv", _csv_bytes(detector, [sample] * 501)))
        record("live_capabilities", "GET", "/api/live/capabilities", 200)
        record("live_stopped", "GET", "/api/live/status", 200)
        record("live_empty_report", "GET", "/api/live/report", 400)
        record("live_empty_csv", "GET", "/api/live/features.csv", 400)
        record("live_bad_start", "POST", "/api/live/start", 400, json_body={})
        record("live_start", "POST", "/api/live/start", 200,
               json_body={"interface": "synthetic-fixture", "mode": "connections"})
        record("live_running", "GET", "/api/live/status", 200)
        with patch("framework.app.server.datetime", _FixedDatetime):
            record("live_report", "GET", "/api/live/report", 200,
                   note="Synthetic transport fixture. Server's unchanged source.synthetic=false is not fixture provenance; clock fixed to " + FIXTURE_TIME)
        record("live_feature_csv", "GET", "/api/live/features.csv?observation_id=synthetic-observation-0", 200)
        record("live_stale_csv", "GET", "/api/live/features.csv?observation_id=old", 400)
        record("live_stale_xai", "POST", "/api/xai", 400, json_body={"kind": "live", "observation_id": "old"})
        record("live_stop", "POST", "/api/live/stop", 200, json_body={})
        record("live_stop_twice", "POST", "/api/live/stop", 200, json_body={})
        record("auth_missing", "GET", "/api/status", 403, authenticated=False)
        record("auth_wrong", "GET", "/api/status", 403, headers={"X-IMDAI-Session": "wrong"})
        record("foreign_origin", "POST", "/api/analyze", 403, json_body={"sample_id": 0},
               headers={"Origin": "https://outside.example"})
        record("same_origin", "POST", "/api/analyze", 200, json_body={"sample_id": 0},
               headers={"Origin": "http://localhost"})
        record("host_ipv4", "GET", "/api/status", 200, headers={"Host": "127.0.0.1"})
        record("host_ipv6", "GET", "/api/status", 400, headers={"Host": "[::1]"},
               note="Observed Flask/Werkzeug baseline rejects bracketed IPv6 despite the configured trust list; preserved discrepancy, not fixed")
        record("foreign_host", "GET", "/api/status", 400, headers={"Host": "outside.example"})
        for name, path in [("page", "/unknown"), ("api", "/api/unknown"), ("static", "/static/missing.js")]:
            record("missing_" + name, "GET", path, 404)
        record("method_not_allowed", "POST", "/api/status", 405, json_body={})
        record("json_array", "POST", "/api/analyze", 400, json_body=[])
        record("json_null", "POST", "/api/analyze", 400, raw="null", content_type="application/json")
        record("json_malformed", "POST", "/api/analyze", 400, raw="{", content_type="application/json")
        record("json_wrong_type", "POST", "/api/analyze", 415, raw="sample_id=0", content_type="text/plain")
        for name, selected in [("negative", -1), ("bool", True), ("string", "0")]:
            record("invalid_sample_" + name, "POST", "/api/analyze", 400, json_body={"sample_id": selected})
        record("invalid_csv_index", "POST", "/api/explain", 400, json_body={"features": sample, "row_index": 500})
        record("invalid_feature_shape", "POST", "/api/explain", 400, json_body={"features": []})
        record("missing_upload", "POST", "/api/upload", 400)
        record("wrong_extension", "POST", "/api/upload", 400, upload=("fixture.txt", csv_payload))
        record("duplicate_columns", "POST", "/api/upload", 400,
               upload=("fixture.csv", b"network_a,network_a\n1,2\n"))
        record("invalid_utf8_upload", "POST", "/api/upload", 500,
               upload=("fixture.csv", b"\xff\xfe\n"),
               note="Existing transform_csv imports csv after UTF-8 decode; invalid encoding triggers UnboundLocalError and generic HTTP 500. Preserved baseline defect.")
        record("csv_row_limit", "POST", "/api/upload", 400, upload=("fixture.csv", _csv_bytes(detector, [sample] * 501)))
        record("body_limit", "POST", "/api/upload", 413, content_type="multipart/form-data; boundary=fixture",
               repeat_body=("x", 10 * 1024 * 1024 + 1), note="Exact 10 MiB limit exercised; repeat descriptor reconstructs the entire request body")
        record("xai_bad_kind", "POST", "/api/xai", 400, json_body={"kind": "unknown"})
        record("assistant_bad_kind", "POST", "/api/assistant/explain", 400, json_body={"kind": "unknown"})

        from framework.detection.lazy_xgboost import LazyXGBoostFacade
        operations = _AdapterOperations(detector)
        facade = LazyXGBoostFacade(operations)
        default_app = create_app(detector=facade, monitor=monitor, registry=registry,
                                 language_service=language, session_token=FIXTURE_TOKEN)
        try:
            default_client = default_app.test_client()
            for name, method, path, arguments in [
                ("default_status", "GET", "/api/status", {}),
                ("default_analyze", "POST", "/api/analyze", {"json_body": {"sample_id": 0}}),
                ("default_explain", "POST", "/api/explain", {"json_body": {"features": sample}}),
                ("default_upload", "POST", "/api/upload", {"upload": ("fixture.csv", csv_payload)}),
                ("default_xai", "POST", "/api/xai", {"json_body": {"kind": "sample", "sample_id": 0}}),
            ]:
                record(name, method, path, 200, selected_client=default_client,
                       note="Actual default LazyXGBoostFacade and XGBoostModelAdapter; registry transport runs in-process for synthetic fixture", **arguments)
            operations.adapter.config["plugins"]["explanation"] = False
            record("default_explanation_disabled", "POST", "/api/analyze", 200,
                   selected_client=default_client, json_body={"sample_id": 0})
            operations.adapter.config["plugins"]["explanation"] = True
            with patch.object(operations.adapter, "_explain", side_effect=RuntimeError("Injected explanation failure")):
                record("default_explanation_failed", "POST", "/api/analyze", 200,
                       selected_client=default_client, json_body={"sample_id": 0})
        finally:
            default_app.extensions["close_resources"]()

        original_import = builtins.__import__
        def missing_optional(name, *args, **kwargs):
            if name in {"framework.data.live_monitor", "framework.llm.local_model"}:
                raise ImportError("Injected optional dependency removal")
            return original_import(name, *args, **kwargs)
        broken_app = create_app(detector=detector, registry=registry, session_token=FIXTURE_TOKEN)
        broken_client = broken_app.test_client()
        with patch("builtins.__import__", side_effect=missing_optional):
            record("import_failed_monitor", "GET", "/api/live/capabilities", 200, selected_client=broken_client)
            record("import_failed_language", "GET", "/api/assistant/status", 200, selected_client=broken_client)
            record("import_failed_monitor_request", "GET", "/api/live/status", 500, selected_client=broken_client)
            record("import_failed_monitor_stop", "POST", "/api/live/stop", 200,
                   json_body={}, selected_client=broken_client)
            record("import_failure_home_survives", "GET", "/", 200, selected_client=broken_client)
            record("import_failure_prediction_survives", "POST", "/api/analyze", 200,
                   json_body={"sample_id": 0}, selected_client=broken_client)
        with patch.object(detector, "status", side_effect=RuntimeError("Injected model status failure")):
            record("runtime_failed_status", "GET", "/api/status", 500)
            record("runtime_failure_home_survives", "GET", "/", 200)
        import framework.app.server as server_module
        original_render = server_module.render_template
        def broken_workspace(template, *args, **kwargs):
            if template == "index.html":
                raise RuntimeError("Injected workspace dependency failure")
            return original_render(template, *args, **kwargs)
        with patch("framework.app.server.render_template", side_effect=broken_workspace):
            record("runtime_failed_html", "GET", "/csv", 500)
            record("runtime_failed_html_status_survives", "GET", "/api/status", 200)
        registry.broken.add("xgboost")
        record("model_import_failure", "POST", "/api/models/xgboost/predict", 503, json_body={"rows": [sample]})
        record("model_failure_catalog_survives", "GET", "/api/models", 200)
        registry.broken.clear()
    finally:
        app.extensions["close_resources"]()

    routes = [{"rule": rule.rule, "methods": sorted(rule.methods)}
              for rule in sorted(app.url_map.iter_rules(), key=lambda rule: rule.rule)]
    summary = {"schema_version": 1, "synthetic": True, "case_count": len(cases),
               "real_components": ["Flask routing, middleware and templates", "Tiny saved XGBoost inference",
                                   "Native TreeSHAP", "LIME package and native model statistics", "CSV parsing"],
               "fakes": {"registry": "XGBoost delegates to real tiny synthetic model; other families unavailable",
                         "default_registry_transport": "Default facade invokes actual XGBoostModelAdapter in-process; no model workers spawned",
                         "monitor": "In-memory synthetic transport only; no OS observation or packet capture",
                         "language": "Explicitly unavailable; no GGUF load or LLM inference"},
               "provenance_notice": "Hardcoded DataSense/live labels in unchanged response bodies are compatibility evidence, not the origin of these synthetic inputs or research results.",
               "session_token": "Public deterministic fixture token only; no real session credentials",
               "normalization": {"omitted_response_headers": ["Date", "Last-Modified", "ETag"],
                                 "live_report_clock": FIXTURE_TIME, "response_bodies_modified": False},
               "detector_paths": {"default_*": "Current default facade/adapter report and optional-capability failures",
                                  "other_cases": "Explicit concrete legacy detector injection; supported compatibility path"},
               "routes": routes, "cases": cases}
    (directory / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"path": "http/manifest.json", "cases": len(cases), "route_patterns": len(routes),
            "synthetic": True, "llm_executed": False, "os_capture_executed": False}
