"""Real CPU checkpoint, signed SHAP and app input behavior."""
import hashlib
import io
import json

import numpy as np
import pytest

from framework.detection.trained_xgboost import XGBoostDetector, transform_csv
pytest.importorskip("flask", exc_type=ImportError)
from framework.app.server import create_app


@pytest.fixture
def detector(tmp_path):
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(42)
    X = rng.normal(size=(90, 5)).astype(np.float32)
    X[:, 2:] = (X[:, 2:] > 0).astype(np.float32)
    y = np.where(X[:, 0] > .4, 2, np.where(X[:, 1] < -.4, 1, 0))
    model = xgb.train({"objective": "multi:softprob", "num_class": 3,
                       "device": "cpu", "max_depth": 3, "seed": 42, "nthread": 2},
                      xgb.DMatrix(X, label=y), num_boost_round=8)
    checkpoint = tmp_path / "model.ubj"
    model.save_model(checkpoint)
    bundle = {"schema_version": 1, "checkpoint": "model.ubj",
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "feature_names": ["network_a", "network_b", "network_protocols_all__present_UNSEEN",
                                "network_protocols_src__present_UNSEEN", "network_protocols_dst__present_UNSEEN"],
              "numeric_features": ["network_a", "network_b"],
              "protocol_vocabulary": {f"network_protocols_{v}": [] for v in ["all", "src", "dst"]},
              "classes": ["benign", "probe", "flood"], "benign_index": 0,
              "model": {"name": "XGBoost", "run_id": "cpu_test_fixture", "validation_accuracy": .5,
                        "macro_f1": .4, "training_rows": 90, "features": 5, "classes": 3,
                        "seed": 42, "selection_basis": "test fixture"},
              "samples": [{"id": 0, "label": "benign", "source_row": 17,
                           "features": X[0].tolist()}]}
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    return XGBoostDetector(path)


def test_real_checkpoint_signed_shap_reconstructs_margin_and_score(detector):
    report = detector.analyze(detector.samples[0]["features"], {"kind": "fixture"})
    explanation = report["explanation"]
    assert explanation["method"] == "Exact TreeSHAP (native XGBoost)"
    assert explanation["reconstruction_error"] < 1e-4
    assert explanation["bias"] + explanation["contribution_sum"] == pytest.approx(explanation["margin"], abs=1e-4)
    assert len(explanation["all_features"]) == 5
    assert report["prediction"]["novelty"] == "not evaluated"
    assert report["reasoning"]["llm_executed"] is False
    assert 0 <= report["prediction"]["score"] <= 1
    assert "cpu" in detector.booster.save_config()


def test_app_handles_valid_analysis_and_rejects_bad_input(detector):
    client = create_app(detector).test_client()
    assert client.get("/api/status").json["execution"]["device"] == "cpu"
    result = client.post("/api/analyze", json={"sample_id": 0})
    assert result.status_code == 200
    assert result.json["source"]["validation_label"] == "benign"
    assert client.post("/api/analyze", json={"sample_id": -1}).status_code == 400
    assert client.post("/api/analyze", json={"sample_id": True}).status_code == 400
    assert client.post("/api/explain", json={"features": [1, 2]}).status_code == 400
    assert client.post("/api/explain", json={"features": [None] * 5}).status_code == 400
    assert client.post("/api/explain", data="bad", content_type="application/json").status_code == 400
    assert client.post("/api/explain", json={"features": [1, 2, 1, 0, 0]},
                       headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.get("/api/status", headers={"Host": "evil.example"}).status_code == 400
    assert client.get("/missing").status_code == 404
    assert client.get("/api/example.csv").status_code == 200


def test_csv_column_order_is_safe_and_labels_are_never_features(detector):
    names = detector.feature_names
    header = ["label4", *names[::-1]]
    csv = ",".join(header) + "\n" + ",".join(["malicious arbitrary label", "1", "0", "1", "2", "3"]) + "\n"
    transformed = transform_csv(csv.encode(), detector.bundle)
    assert transformed.tolist() == [[3, 2, 1, 0, 1]]
    client = create_app(detector).test_client()
    result = client.post("/api/upload", data={"file": (io.BytesIO(csv.encode()), "rows.csv")})
    assert result.status_code == 200
    assert result.json["total_rows"] == 1
    assert client.post("/api/upload", data={"file": (io.BytesIO(b"bad\n1"), "bad.csv")}).status_code == 400


def test_checkpoint_tamper_never_silently_falls_back(detector):
    detector.checkpoint.write_bytes(b"not a valid trained model")
    with pytest.raises(ValueError, match="checksum"):
        XGBoostDetector(detector.bundle_path)


def test_raw_protocol_transform_matches_original_preparation(detector):
    import pandas as pd
    from experiment.prepare_protocol_features import protocol_matrix, FIELDS
    frame = pd.DataFrame({"network_a": [1.5, 2], "network_b": [3, np.nan],
                          **{f: ["['tcp', 'tcp']", "[]"] for f in FIELDS}, "label4": ["wrong", "wrong"]})
    X = transform_csv(frame.to_csv(index=False).encode(), detector.bundle)
    np.testing.assert_equal(X[:, 2:], protocol_matrix(frame, detector.bundle["protocol_vocabulary"]))
    assert np.isnan(X[1, 1])


def test_csv_limits_duplicate_headers_and_incomplete_schema(detector):
    names = detector.feature_names
    with pytest.raises(ValueError, match="duplicate"):
        transform_csv(b"network_a,network_a\n1,2", detector.bundle)
    csv = ",".join(names) + "\n" + "1,2,1,0,1\n" * 501
    with pytest.raises(ValueError, match="500"):
        transform_csv(csv.encode(), detector.bundle)
    with pytest.raises(ValueError, match="protocol"):
        transform_csv(b"network_a,network_b\n1,2", detector.bundle)
    client = create_app(detector).test_client()
    client.application.config["MAX_CONTENT_LENGTH"] = 100
    assert client.post("/api/upload", data={"file": (io.BytesIO(b"x" * 200), "large.csv")}).status_code == 413


def test_home_and_workspace_routes_have_distinct_page_contracts(detector):
    client = create_app(detector).test_client()
    home = client.get("/")
    assert home.status_code == 200
    assert "NetSec" in home.get_data(as_text=True)
    assert 'id="analyze-button"' not in home.get_data(as_text=True)
    for route, mode, heading in [("/live", "live", "Live screening"),
                                 ("/csv", "upload", "CSV analysis"),
                                 ("/samples", "sample", "Sample analysis")]:
        response = client.get(route)
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert f'data-workspace-mode="{mode}"' in html
        assert heading in html
        assert 'id="insights-panel"' in html
    assert client.get("/unknown-workspace").status_code == 404


@pytest.mark.parametrize("route", ["/", "/live", "/csv", "/samples", "/loading", "/error", "/exit"])
def test_every_page_keeps_desktop_session_authentication(detector, route):
    token = "c" * 64
    client = create_app(detector, session_token=token).test_client()
    assert client.get(route).status_code == 403
    assert client.get(route, headers={"X-IMDAI-Session": "wrong"}).status_code == 403
    assert client.get(route, headers={"X-IMDAI-Session": token}).status_code == 200


def test_lifecycle_html_keeps_api_errors_json_and_safe_return_routes(detector):
    client = create_app(detector).test_client()
    for state in ["loading", "error", "exit"]:
        response = client.get(f"/{state}")
        assert response.status_code == 200
        assert f'data-lifecycle-state="{state}"' in response.get_data(as_text=True)
        assert "script-src 'self'" in response.headers["Content-Security-Policy"]
    missing = client.get("/unknown-workspace")
    assert missing.status_code == 404
    assert 'data-error-code="404"' in missing.get_data(as_text=True)
    assert client.get("/api/unknown").status_code == 404
    assert client.get("/api/unknown").is_json
    assert 'data-return-url="/"' in client.get("/exit?return=https://evil.example").get_data(as_text=True)
    assert 'data-return-url="/csv"' in client.get("/exit?return=/csv").get_data(as_text=True)
