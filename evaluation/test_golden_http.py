"""Golden transport coverage uses a real tiny model, never research checkpoints."""
import hashlib
import json

import numpy as np
import pytest

from evaluation.test_ids_app import detector
from tools.golden_http import generate_http_goldens


@pytest.fixture
def golden_detector(detector):
    pytest.importorskip("lime")
    random = np.random.default_rng(1042)
    training = random.normal(size=(64, len(detector.feature_names))).astype(np.float32)
    training[:, len(detector.bundle["numeric_features"]):] = (
        training[:, len(detector.bundle["numeric_features"]):] > 0).astype(np.float32)
    path = detector.bundle_path.parent / "xai_reference.npz"
    np.savez(path, training=training, medians=np.median(training, axis=0))
    path.with_suffix(".json").write_text(json.dumps({
        "feature_names": detector.feature_names, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "categorical_indices": list(range(len(detector.bundle["numeric_features"]), len(detector.feature_names))),
        "provenance": "Synthetic golden HTTP test reference; no research observations",
    }), encoding="utf-8")
    return detector


def test_http_goldens_cover_routes_auth_failures_and_real_predictions(tmp_path, golden_detector):
    output = tmp_path / "goldens"
    summary = generate_http_goldens(output, golden_detector)
    manifest = json.loads((output / summary["path"]).read_text(encoding="utf-8"))
    cases = {entry["id"]: entry for entry in manifest["cases"]}
    assert summary["cases"] == len(cases) >= 85
    assert summary["llm_executed"] is summary["os_capture_executed"] is False
    assert manifest["normalization"]["response_bodies_modified"] is False
    rules = {route["rule"] for route in manifest["routes"]}
    assert {"/models", "/api/models/<model_id>/predict", "/api/assistant/explain", "/static/<path:filename>"} <= rules
    for entry in cases.values():
        response = entry["response"]
        body = (output / "http" / response["body_file"]).read_bytes()
        assert hashlib.sha256(body).hexdigest() == response["body_sha256"]
        assert response["headers"]["X-Content-Type-Options"] == "nosniff"
        assert response["headers"]["Cache-Control"] == "no-store"
        assert "script-src 'self'" in response["headers"]["Content-Security-Policy"]
        assert "Date" not in response["headers"]
    def payload(name):
        return json.loads((output / "http" / cases[name]["response"]["body_file"]).read_text())
    actual = golden_detector.predict_batch(golden_detector.validate_features(golden_detector.samples[0]["features"]).reshape(1, -1))[0]
    assert payload("model_predict")["predictions"][0]["score"] == actual["score"]
    assert payload("analyze")["explanation"]["reconstruction_error"] < 1e-4
    assert payload("xai")["lime"]["perturbations"] == 2048
    assert payload("model_comparison")["comparable"] is False
    assert payload("assistant_status")["llm_executed"] is False
    assert payload("real_assistant_unavailable_status")["available"] is False
    assert cases["real_assistant_unavailable_explain"]["response"]["status"] == 400
    assert "Local language model unavailable" in payload("real_assistant_unavailable_explain")["error"]
    assert (output / 'http' / cases['status_head']['response']['body_file']).read_bytes() == b''
    assert cases['analyze_options']['response']['headers']['Allow']
    assert cases['options_auth_missing']['response']['status'] == 403
    default = payload('default_analyze')
    assert default['capability_errors'] == []
    assert default['prediction']['score'] == actual['score']
    assert payload('default_upload')['rows'][0]['prediction']['score'] == actual['score']
    for name in ['default_explanation_disabled', 'default_explanation_failed']:
        result = payload(name)
        assert result['prediction'] == default['prediction']
        assert result['explanation'] is None and result['reasoning'] is None
        assert result['capability_errors'][0]['capability'] == 'explain'
    assert cases["body_limit"]["response"]["status"] == 413
    assert cases["body_limit"]["request"]["body"]["count"] == 10 * 1024 * 1024 + 1
    assert cases["import_failure_prediction_survives"]["response"]["status"] == 200
    assert cases["runtime_failed_html"]["response"]["status"] == 500
    assert "no substitute prediction" in payload("runtime_failed_status")["error"]
    assert payload("model_import_failure")["status"] == "unavailable"
    assert cases["missing_page"]["response"]["headers"]["Content-Type"].startswith("text/html")
    assert cases["missing_api"]["response"]["headers"]["Content-Type"].startswith("application/json")
    assert str(tmp_path) not in (output / summary["path"]).read_text()
    with pytest.raises(FileExistsError):
        generate_http_goldens(output, golden_detector)


def test_http_goldens_are_byte_stable(tmp_path, golden_detector):
    first, second = tmp_path / "first", tmp_path / "second"
    generate_http_goldens(first, golden_detector)
    generate_http_goldens(second, golden_detector)
    first_files = {file.relative_to(first): file.read_bytes() for file in first.rglob("*") if file.is_file()}
    second_files = {file.relative_to(second): file.read_bytes() for file in second.rglob("*") if file.is_file()}
    assert first_files == second_files
