"""Bounded, local-only HTTP interface; uploads stay in memory."""
import io
import hmac
from datetime import datetime, timezone
from urllib.parse import urlsplit

from flask import Flask, jsonify, render_template, request, Response
from werkzeug.exceptions import BadRequest, HTTPException, RequestEntityTooLarge

from framework.detection.trained_xgboost import json_features, transform_csv
from framework.app.labels import display_label, sample_names


def create_app(detector, monitor=None, session_token=None):
    from framework.data.live_monitor import LiveMonitor
    monitor = monitor if monitor is not None else LiveMonitor(detector=detector)
    from framework.llm.local_model import LocalLanguageModel
    language_model = LocalLanguageModel()
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=10 * 1024 * 1024,
                      TRUSTED_HOSTS=["localhost", "127.0.0.1", "[::1]"])

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
        return jsonify(error=str(error)), 400

    @app.errorhandler(RequestEntityTooLarge)
    def large_upload(error):
        return jsonify(error="Upload exceeds 10 MB. Export a CSV batch of at most 500 rows."), 413

    @app.errorhandler(Exception)
    def failed(error):
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
        prediction = report["prediction"]
        prediction["display_label"] = display_label(prediction["label"])
        for entry in prediction["top_classes"]:
            entry["display_label"] = display_label(entry["label"])
        if "validation_label" in report["source"]:
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

    @app.get("/loading")
    @app.get("/error")
    @app.get("/exit")
    def lifecycle_page():
        state = request.path[1:]
        return_url = request.args.get("return", "/")
        if return_url not in {"/", "/live", "/csv", "/samples"}:
            return_url = "/"
        titles = {"loading": "Loading workspace", "error": "Service unavailable", "exit": "End workspace"}
        return render_template("lifecycle.html", lifecycle_state=state,
                               page_title=titles[state], return_url=return_url)

    @app.get("/api/status")
    def status():
        return jsonify(detector.status())

    @app.get("/api/samples")
    def samples():
        return jsonify(samples=sample_names([{k: row[k] for k in ["id", "label", "source_row"]} for row in detector.samples]))

    @app.get("/api/example.csv")
    def example_csv():
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
        selected = body().get("sample_id")
        if type(selected) is not int or not 0 <= selected < len(detector.samples):
            raise ValueError("Select a valid validation sample")
        sample = detector.samples[selected]
        return jsonify(readable_report(detector.analyze(sample["features"], {"kind": "DataSense validation observation",
                                                             "validation_label": sample["label"],
                                                             "source_row": sample["source_row"]})))

    @app.post("/api/upload")
    def upload():
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
        data = body()
        row = data.get("row_index", 0)
        if type(row) is not int or row < 0 or row >= 500:
            raise ValueError("Invalid CSV row index")
        return jsonify(readable_report(detector.analyze(data.get("features"), {"kind": "Uploaded CSV features", "row_index": row})))

    @app.get("/api/live/capabilities")
    def live_capabilities():
        return jsonify(monitor.capabilities())

    @app.get("/api/live/status")
    def live_status():
        snapshot = monitor.snapshot()
        if snapshot.get("multiclass_prediction"):
            readable_report(snapshot["multiclass_prediction"])
        return jsonify(snapshot)

    @app.post("/api/live/start")
    def live_start():
        data = body()
        if not isinstance(data.get("interface"), str) or not isinstance(data.get("mode"), str):
            raise ValueError("Choose an adapter and monitoring mode")
        return jsonify(monitor.start(data["interface"], data["mode"]))

    @app.post("/api/live/stop")
    def live_stop():
        body()
        return jsonify(monitor.stop())

    @app.get("/api/live/report")
    def live_report():
        from framework.llm.live_template import describe_observation
        snapshot = monitor.snapshot()
        if not snapshot["metrics"]["packets_observed"]:
            raise ValueError("Start monitoring and collect traffic before exporting a report")
        return jsonify({**snapshot, "generated_at": datetime.now(timezone.utc).isoformat(),
                        "source": {"kind": "Local adapter observation", "synthetic": False,
                                   "packet_headers_captured": snapshot["mode"] == "packets"},
                        "reasoning": describe_observation(snapshot)})

    def current_multiclass(observation_id):
        snapshot = monitor.snapshot()
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
        values = detector.validate_features(report["features"])
        return Response(vector_csv(values, detector.feature_names), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=observed_packet_features.csv"})

    def evidence_context(data):
        kind = data.get("kind")
        if kind == "live":
            snapshot, context = current_multiclass(data.get("observation_id"))
            context["rules"] = snapshot["assessment"]
            values = detector.validate_features(context["features"])
            return context, detector.booster, values, detector.feature_names, detector.classes, detector.bundle_path.parent, False
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
        return (readable_report(detector.analyze(features, source)), detector.booster, values,
                detector.feature_names, detector.classes, detector.bundle_path.parent, False)

    @app.post("/api/xai")
    def additional_xai():
        from framework.xai.model_evidence import explain_additional
        context, booster, values, names, classes, directory, packet = evidence_context(body())
        return jsonify({**explain_additional(booster, values, names, classes, directory / "xai_reference.npz", packet=packet),
                        "source": context["source"], "prediction": context["prediction"],
                        "feature_coverage": context.get("feature_coverage"), "model": context.get("model")})

    @app.get("/api/assistant/status")
    def assistant_status():
        return jsonify(language_model.status())

    @app.post("/api/assistant/explain")
    def assistant_explain():
        from framework.xai.model_evidence import explain_additional
        context, booster, values, names, classes, directory, packet = evidence_context(body())
        extra = explain_additional(booster, values, names, classes, directory / "xai_reference.npz", packet=packet)
        explanation = context["explanation"]
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
                    "lime": {"class_label": extra["lime"]["class_label"], "local_fidelity_r2": extra["lime"]["local_fidelity_r2"],
                             "surrogate_prediction": extra["lime"]["surrogate_prediction"], "conditions": extra["lime"]["conditions"][:2],
                             "scope": "Approximate explanation from synthetic perturbations, not captured traffic"},
                    "limitations": "Scores are uncalibrated. A prediction is not proof of an attack or safety. Unknown attacks and labelled live accuracy have not been evaluated."}
        return jsonify(language_model.generate(evidence))

    return app
