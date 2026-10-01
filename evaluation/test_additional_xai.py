"""Additional explanations remain linked to the actual model and training reference."""
import hashlib
import json
import numpy as np
import pytest

from evaluation.test_ids_app import detector
from framework.xai.model_evidence import explain_additional
from framework.llm.local_model import LocalLanguageModel


def reference(detector):
    path = detector.bundle_path.parent / "xai_reference.npz"
    rng = np.random.default_rng(42)
    X = rng.normal(size=(90, 5)).astype(np.float32)
    X[:, 2:] = (X[:, 2:] > 0).astype(np.float32)
    np.savez_compressed(path, training=X, medians=np.median(X, axis=0))
    path.with_suffix(".json").write_text(json.dumps({"feature_names": detector.feature_names,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "categorical_indices": [2, 3, 4], "rows": 90,
        "kind": "Synthetic unit-test training fixture"}))
    return path


def test_lime_predicts_the_same_anchor_and_preserves_missing_values(detector):
    pytest.importorskip("lime")
    values = detector.validate_features([1, None, 1, 0, 1])
    result = explain_additional(detector.booster, values, detector.feature_names, detector.classes, reference(detector))
    prediction = detector.predict_batch(values.reshape(1, -1))[0]
    assert result["lime"]["class_label"] == prediction["label"]
    assert result["lime"]["model_score"] == pytest.approx(prediction["score"])
    assert result["lime"]["conditions"] and np.isfinite(result["lime"]["local_fidelity_r2"])
    assert result["statistics"]["features"][1]["value"] is None
    assert result["statistics"]["model_parameters"]["boosting_rounds"] == detector.booster.num_boosted_rounds()
    assert result["llm_executed"] is False


def test_lime_reference_tamper_is_rejected(detector):
    pytest.importorskip("lime")
    path = reference(detector)
    path.write_bytes(b"tampered reference")
    with pytest.raises(ValueError, match="checksum"):
        explain_additional(detector.booster, np.ones(5), detector.feature_names, detector.classes, path)


def test_missing_language_model_never_falls_back_to_a_fake_llm(monkeypatch, tmp_path):
    monkeypatch.setenv("IMDAI_LLM_DIR", str(tmp_path))
    provider = LocalLanguageModel()
    assert provider.status()["available"] is False
    with pytest.raises(ValueError, match="unavailable"):
        provider.generate({"prediction": "fixture"})
    assert provider.status()["busy"] is False
