"""Bounded, local-only HTTP interface; uploads stay in memory."""
import io
import hmac
from pathlib import Path
import sys
import threading
from datetime import datetime, timezone
from urllib.parse import urlsplit

from flask import Flask, jsonify, render_template, request, Response
from werkzeug.exceptions import BadRequest, HTTPException, RequestEntityTooLarge

from framework.app.labels import display_label, sample_names


def create_app(detector=None, monitor=None, session_token=None, registry=None, language_service=None):
    """Build the local shell independently of model and optional dependencies.

    An explicit detector/monitor remains supported for existing integrations.
    The default detector is a lazy, isolated XGBoost facade. Other families are
    selected explicitly through the research endpoints, never as fallbacks.
    """
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=10 * 1024 * 1024,
                      TRUSTED_HOSTS=["localhost", "127.0.0.1", "[::1]"])
    app.extensions["model_registry"] = registry
    runtime_lock = threading.RLock()
    language_model = language_service
    isolate_optional = detector is None and (registry is None or hasattr(registry, "root"))
    monitor_error = None
    language_error = None

    def get_registry():
        nonlocal registry
        with runtime_lock:
            if registry is None:
                from framework.core.registry import ModelRegistry
                root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
                registry = ModelRegistry(root)
                app.extensions["model_registry"] = registry
            return registry

    def get_detector():
        nonlocal detector
        with runtime_lock:
            if detector is None:
                from framework.detection.lazy_xgboost import LazyXGBoostFacade
                detector = LazyXGBoostFacade(get_registry())
            return detector

    def get_monitor():
        nonlocal monitor, monitor_error
        with runtime_lock:
            if monitor is None and monitor_error is None:
                try:
                    if isolate_optional:
                        from framework.core.capabilities import IsolatedMonitorService
                        monitor = IsolatedMonitorService(get_registry())
                    else:
                        # Explicitly injected legacy/test integrations preserve
                        # their in-process service contract.
                        from framework.data.live_monitor import LiveMonitor
                        try:
                            selected_detector = get_detector()
                        except Exception as error:
                            selected_detector = None
                            app.logger.warning("Live multiclass capability unavailable: %s", error)
                        monitor = LiveMonitor(detector=selected_detector)
                except Exception as error:
                    monitor_error = error
                    app.logger.exception("Monitoring capability initialization failed")
            if monitor_error is not None:
                raise RuntimeError("Monitoring capability unavailable: " + str(monitor_error)) from monitor_error
            return monitor

    def get_language_model():
        nonlocal language_model, language_error
        with runtime_lock:
            if registry is not None and not getattr(registry, "configuration", {}).get("plugins", {}).get("language", True):
                raise RuntimeError("Language capability disabled in configuration; prediction remains available")
            if language_model is None and language_error is None:
                try:
                    if isolate_optional:
                        from framework.core.capabilities import IsolatedLanguageService
                        language_model = IsolatedLanguageService(get_registry())
                    else:
                        from framework.llm.local_model import LocalLanguageModel
                        language_model = LocalLanguageModel()
                except Exception as error:
                    language_error = error
                    app.logger.exception("Language capability initialization failed")
            if language_error is not None:
                raise RuntimeError("Language capability unavailable: " + str(language_error)) from language_error
            return language_model

    from framework.app.model_workspace import register_model_workspace
    register_model_workspace(app, get_registry)

    def scoped_error(error):
        status = getattr(error, "status", "failed")
        response = {"model_id": getattr(error, "model_id", None), "capability": getattr(error, "capability", "prediction"),
                    "status": status, "error": getattr(error, "reason", str(error))}
        app.logger.warning("Capability failure: %s", response)
        return jsonify(response), 400 if status == "incompatible" else 503

    def close_resources():
        if monitor is not None:
            try:
                monitor.stop()
            except Exception:
                app.logger.exception("Monitoring shutdown failed")
            if hasattr(monitor, "close"):
                monitor.close()
        if language_model is not None and hasattr(language_model, "close"):
            language_model.close()
        if registry is not None:
            registry.close()
    app.extensions["close_resources"] = close_resources

    @app.before_request
    def reject_foreign_origin():
        if session_token is not None and not hmac.compare_digest(request.headers.get("X-IMDAI-Session", ""), session_token):
            return jsonify(error="This analysis service belongs to the desktop session"), 403
        origin = request.headers.get("Origin")
        if request.method == "POST" and origin and urlsplit(origin).netloc != request.host:
            return jsonify(error="Only requests from this local app are allowed"), 403

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; form-action 'self'")
        return response

    @app.errorhandler(ValueError)
    @app.errorhandler(BadRequest)
    def bad_input(error):
        pieces = request.path.split("/")
        if len(pieces) >= 5 and pieces[1:3] == ["api", "models"]:
            return jsonify(model_id=pieces[3], capability=pieces[4], status="incompatible", error=str(error)), 400
        return jsonify(error=str(error)), 400

    @app.errorhandler(RequestEntityTooLarge)
    def large_upload(error):
        return jsonify(error="Upload exceeds 10 MB. Export a CSV batch of at most 500 rows."), 413

    @app.errorhandler(Exception)
    def failed(error):
        if hasattr(error, "model_id") and hasattr(error, "status"):
            return scoped_error(error)
        if isinstance(error, HTTPException):
            if request.method == "GET" and not request.path.startswith(("/api/", "/static/")):
                return render_template("lifecycle.html", lifecycle_state="error",
                                       page_title="Page unavailable", error_code=error.code), error.code
            return jsonify(error=error.description), error.code
        app.logger.exception("Analysis request failed")
        if request.method == "GET" and not request.path.startswith(("/api/", "/static/")):
            return render_template("lifecycle.html", lifecycle_state="error",
                                   page_title="Workspace unavailable", error_code=500), 500
        return jsonify(error="Analysis failed. Check the local server log; no substitute prediction was returned."), 500

    def body():
        value = request.get_json()
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value

    def readable_report(report):
        prediction = report.get("prediction")
        if not prediction:
            return report
        prediction["display_label"] = display_label(prediction["label"])
        for entry in prediction.get("top_classes", []):
            entry["display_label"] = display_label(entry["label"])
        if "validation_label" in report.get("source", {}):
            report["source"]["validation_display_label"] = display_label(report["source"]["validation_label"])
        return report

    @app.get("/")
    def index():
        return render_template("home.html")

    @app.get("/live")
    def live_workspace():
        return render_template("index.html", workspace_mode="live", page_title="Live screening")

    @app.get("/csv")
    def csv_workspace():
        return render_template("index.html", workspace_mode="upload", page_title="CSV analysis")

    @app.get("/samples")
    def sample_workspace():
        return render_template("index.html", workspace_mode="sample", page_title="Sample analysis")

    @app.get("/models")
    def model_workspace():
        return render_template("models.html", workspace_mode="models", page_title="Model research")

    @app.get("/loading")
    @app.get("/error")
    @app.get("/exit")
    def lifecycle_page():
        state = request.path[1:]
        return_url = request.args.get("return", "/")
        if return_url not in {"/", "/live", "/csv", "/samples", "/models"}:
            return_url = "/"
        titles = {"loading": "Loading workspace", "error": "Service unavailable", "exit": "End workspace"}
        return render_template("lifecycle.html", lifecycle_state=state,
                               page_title=titles[state], return_url=return_url)

    @app.get("/api/status")
    def status():
        return jsonify(get_detector().status())

    @app.get("/api/samples")
    def samples():
        detector = get_detector()
        return jsonify(samples=sample_names([{k: row[k] for k in ["id", "label", "source_row"]} for row in detector.samples]))

    @app.get("/api/example.csv")
    def example_csv():
        detector = get_detector()
        import csv
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow([*detector.feature_names, "label4"])
        # Deterministic first observation of each of the first eight label groups.
        seen = set()
        for sample in detector.samples:
            if sample["label"] in seen:
                continue
            writer.writerow([*sample["features"], sample["label"]])
            seen.add(sample["label"])
            if len(seen) == 8:
                break
        return Response(stream.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=validation_examples.csv"})

    @app.post("/api/analyze")
    def analyze():
        detector = get_detector()
        selected = body().get("sample_id")
        if type(selected) is not int or not 0 <= selected < len(detector.samples):
            raise ValueError("Select a valid validation sample")
        sample = detector.samples[selected]
        return jsonify(readable_report(detector.analyze(sample["features"], {"kind": "DataSense validation observation",
                                                             "validation_label": sample["label"],
                                                             "source_row": sample["source_row"]})))

    @app.post("/api/upload")
    def upload():
        from framework.detection.trained_xgboost import json_features, transform_csv
        detector = get_detector()
        file = request.files.get("file")
        if file is None or not file.filename.lower().endswith(".csv"):
            raise ValueError("Choose a .csv file")
        X = transform_csv(file.read(), detector.bundle)
        predictions = detector.predict_batch(X)
        for prediction in predictions:
            prediction["display_label"] = display_label(prediction["label"])
        return jsonify(total_rows=len(X), rows=[{"row_index": i, "prediction": prediction,
                                                "features": json_features(X[i])}
                                               for i, prediction in enumerate(predictions)])

    @app.post("/api/explain")
    def explain():
        detector = get_detector()
        data = body()
        row = data.get("row_index", 0)
        if type(row) is not int or row < 0 or row >= 500:
            raise ValueError("Invalid CSV row index")
        return jsonify(readable_report(detector.analyze(data.get("features"), {"kind": "Uploaded CSV features", "row_index": row})))

    @app.get("/api/live/capabilities")
    def live_capabilities():
        try:
            return jsonify(get_monitor().capabilities())
        except Exception as error:
            app.logger.warning("Monitoring capability unavailable: %s", error)
            return jsonify(interfaces=[], recommended_interface="", connections_available=False,
                           packet_capture_available=False, model_compatible=False,
                           packet_capture_message=str(error), error=str(error))

    @app.get("/api/live/status")
    def live_status():
        snapshot = get_monitor().snapshot()
        if snapshot.get("multiclass_prediction"):
            readable_report(snapshot["multiclass_prediction"])
        return jsonify(snapshot)

    @app.post("/api/live/start")
    def live_start():
        data = body()
        if not isinstance(data.get("interface"), str) or not isinstance(data.get("mode"), str):
            raise ValueError("Choose an adapter and monitoring mode")
        return jsonify(get_monitor().start(data["interface"], data["mode"]))

    @app.post("/api/live/stop")
    def live_stop():
        body()
        if monitor is None:
            return jsonify(running=False, state="unavailable" if monitor_error else "stopped",
                           error=str(monitor_error) if monitor_error else None)
        return jsonify(get_monitor().stop())

    @app.get("/api/live/report")
    def live_report():
        from framework.llm.live_template import describe_observation
        snapshot = get_monitor().snapshot()
        if not snapshot["metrics"]["packets_observed"]:
            raise ValueError("Start monitoring and collect traffic before exporting a report")
        return jsonify({**snapshot, "generated_at": datetime.now(timezone.utc).isoformat(),
                        "source": {"kind": "Local adapter observation", "synthetic": False,
                                   "packet_headers_captured": snapshot["mode"] == "packets"},
                        "reasoning": describe_observation(snapshot)})

    def current_multiclass(observation_id):
        snapshot = get_monitor().snapshot()
        report = snapshot.get("multiclass_prediction")
        if snapshot.get("error") or not snapshot["running"] or not report:
            raise ValueError("Wait for a current nonempty multiclass packet observation")
        if not observation_id or observation_id != report.get("observation_id"):
            raise ValueError("The packet observation changed. Refresh the evidence and try again.")
        return snapshot, readable_report(report)

    @app.get("/api/live/features.csv")
    def live_features_csv():
        from framework.data.feature_schema import vector_csv
        _, report = current_multiclass(request.args.get("observation_id"))
        detector = get_detector()
        values = detector.validate_features(report["features"])
        return Response(vector_csv(values, detector.feature_names), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=observed_packet_features.csv"})

    def evidence_context(data):
        detector = get_detector()
        kind = data.get("kind")
        if kind == "live":
            snapshot, context = current_multiclass(data.get("observation_id"))
            context["rules"] = snapshot["assessment"]
            values = detector.validate_features(context["features"])
            return context, detector, values
        if kind == "sample":
            selected = data.get("sample_id")
            if type(selected) is not int or not 0 <= selected < len(detector.samples):
                raise ValueError("Choose a valid sample")
            sample = detector.samples[selected]
            features = sample["features"]
            source = {"kind": "DataSense validation observation", "validation_label": sample["label"], "source_row": sample["source_row"]}
        elif kind == "features":
            features, source = data.get("features"), {"kind": "Uploaded feature values"}
        else:
            raise ValueError("Choose sample, features or live evidence")
        values = detector.validate_features(features)
        return readable_report(detector.analyze(features, source)), detector, values

    def additional_evidence(detector, values):
        if hasattr(detector, "additional_xai"):
            return detector.additional_xai(values.tolist())
        # Concrete legacy detectors remain supported without passing native model
        # objects across the new process-isolated facade.
        from framework.xai.model_evidence import explain_additional
        return explain_additional(detector.booster, values, detector.feature_names,
                                  detector.classes, detector.bundle_path.parent / "xai_reference.npz")

    @app.post("/api/xai")
    def additional_xai():
        context, detector, values = evidence_context(body())
        return jsonify({**additional_evidence(detector, values),
                        "source": context["source"], "prediction": context["prediction"],
                        "feature_coverage": context.get("feature_coverage"), "model": context.get("model")})

    @app.get("/api/assistant/status")
    def assistant_status():
        try:
            return jsonify(get_language_model().status())
        except Exception as error:
            app.logger.warning("Language capability unavailable: %s", error)
            return jsonify(available=False, offline=True, device="cpu", error=str(error),
                           automatic_generation=False, llm_executed=False,
                           rag={"available": False, "error": str(error)})

    @app.post("/api/assistant/explain")
    def assistant_explain():
        context, detector, values = evidence_context(body())
        explanation = context.get("explanation")
        if not explanation or not explanation.get("all_features"):
            raise ValueError("A measured model explanation is unavailable for this observation. Prediction remains available.")
        # LIME is an optional second view, never a prerequisite for language
        # generation from independently available signed TreeSHAP evidence.
        extra = None
        try:
            extra = additional_evidence(detector, values)
        except Exception as error:
            app.logger.warning("Optional LIME evidence unavailable for assistant: %s", error)
        prediction = context["prediction"]
        ranked = sorted(explanation["all_features"], key=lambda feature: -abs(feature["shap_value"]))
        measured = [feature for feature in ranked if not feature.get("is_missing")]
        missing = [feature for feature in ranked if feature.get("is_missing")]
        language_features = ([measured[0], missing[0]] if measured and missing else ranked[:2])
        evidence = {"source": context["source"],
                    "prediction": {"label": prediction.get("display_label", prediction["label"]),
                                   "score": round(prediction["score"], 4), "is_attack": prediction["is_attack"],
                                   "top_classes": prediction.get("top_classes", []), "tentative": prediction.get("tentative", False)},
                    "feature_coverage": {k: v for k, v in context.get("feature_coverage", {}).items() if k != "features"},
                    "xai": {"method": explanation["method"], "class_label": explanation["class_label"],
                            "margin": round(explanation["margin"], 4), "bias": round(explanation["bias"], 4),
                            "features": language_features},
                    "rules": {"status": context["rules"]["status"]} if context.get("rules") else None,
                    "lime": ({"class_label": extra["lime"]["class_label"], "local_fidelity_r2": extra["lime"]["local_fidelity_r2"],
                             "surrogate_prediction": extra["lime"]["surrogate_prediction"], "conditions": extra["lime"]["conditions"][:2],
                             "scope": "Approximate explanation from synthetic perturbations, not captured traffic"}
                             if extra and extra.get("lime") else None),
                    "limitations": "Scores are uncalibrated. A prediction is not proof of an attack or safety. Unknown attacks and labelled live accuracy have not been evaluated."}
        return jsonify(get_language_model().generate(evidence))

    return app
