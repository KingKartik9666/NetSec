"""Synthetic CPU fixtures verify partial-input software contracts, not accuracy."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from framework.llm.local_model import generation_facts
from framework.rag.grounding import check_grounding
from framework.xai.model_evidence import explain_additional
from framework.xai.tree_shap import explain_tree


@pytest.fixture
def partial_model(tmp_path):
    xgb = pytest.importorskip("xgboost")
    # Match the production dimensions while keeping this fixture small, seeded
    # and CPU-only. These synthetic rows are never research/live observations.
    rng = np.random.default_rng(42)
    training = rng.normal(size=(252, 170)).astype(np.float32)
    training[:, 110:] = (training[:, 110:] > 0).astype(np.float32)
    names = [f"network_numeric_{i}" for i in range(110)]
    names += [f"network_protocols_all__present_fixture{i}" for i in range(60)]
    classes = [f"synthetic_class_{i}" for i in range(84)]
    booster = xgb.train({"objective": "multi:softprob", "num_class": 84,
                         "device": "cpu", "max_depth": 2, "seed": 42, "nthread": 2},
                        xgb.DMatrix(training, label=np.arange(len(training)) % 84, nthread=2),
                        num_boost_round=3)
    values = np.full(170, np.nan, dtype=np.float32)
    values[[0, 2, 110]] = [0, 1.5, 0]
    reference = tmp_path / "synthetic_reference.npz"
    np.savez_compressed(reference, training=training, medians=np.median(training, axis=0))
    reference.with_suffix(".json").write_text(json.dumps({
        "feature_names": names, "categorical_indices": list(range(110, 170)),
        "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "kind": "Synthetic unit-test training fixture; not research data", "rows": len(training)}))
    return booster, values, names, classes, reference


def test_native_shap_retains_missing_default_contributions_and_exact_additivity(partial_model):
    import xgboost as xgb
    booster, values, names, classes, _ = partial_model
    scores = booster.predict(xgb.DMatrix(values.reshape(1, -1), nthread=2))[0]
    label = classes[int(scores.argmax())]
    result = explain_tree(booster, values.reshape(1, -1), names, classes, label)
    features = result["all_features"]
    assert len(features) == 170
    assert np.isfinite([result["margin"], result["bias"], *[f["shap_value"] for f in features]]).all()
    assert result["bias"] + sum(f["shap_value"] for f in features) == pytest.approx(result["margin"], abs=1e-4)
    missing = [f for f in features if f["is_missing"]]
    assert len(missing) == 167 and any(f["shap_value"] != 0 for f in missing)
    assert all(f["value"] is None and not f["observed"] and f["attribution_basis"] == "missing-value default routing" for f in missing)
    for index in (0, 110):
        assert features[index]["value"] == 0 and features[index]["observed"] and not features[index]["is_missing"]
    assert result == explain_tree(booster, values.reshape(1, -1), names, classes, label)


def test_lime_preserves_missing_numeric_and_categorical_columns_in_every_prediction(partial_model):
    pytest.importorskip("lime")
    import xgboost as xgb
    booster, values, names, classes, reference = partial_model
    calls = []

    class RecordingBooster:
        def predict(self, matrix, **kwargs):
            sparse = matrix.get_data().tocoo()
            dense = np.full(sparse.shape, np.nan, dtype=np.float32)
            dense[sparse.row, sparse.col] = sparse.data
            calls.append(dense)
            return booster.predict(matrix, **kwargs)

        def __getattr__(self, name):
            return getattr(booster, name)

    result = explain_additional(RecordingBooster(), values, names, classes, reference)
    assert [len(call) for call in calls] == [1, 2048]
    for call in calls:
        assert np.isnan(call[:, np.isnan(values)]).all()
        np.testing.assert_equal(call[0], values)
        assert call[0, 0] == call[0, 110] == 0
    scores = booster.predict(xgb.DMatrix(values.reshape(1, -1), nthread=2))[0]
    assert result["lime"]["class_label"] == classes[int(scores.argmax())]
    assert result["lime"]["model_score"] == pytest.approx(float(scores.max()))
    assert result["lime"]["observed_features"] == 3 and result["lime"]["missing_features"] == 167
    assert np.isfinite(result["lime"]["local_fidelity_r2"])
    assert all(not any(name in condition["condition"] for name in np.array(names)[np.isnan(values)])
               for condition in result["lime"]["conditions"])
    assert result["statistics"]["features"][169]["value"] is None
    assert result["statistics"]["features"][169]["is_missing"]
    assert result["statistics"]["features"][110]["value"] == 0
    assert result["statistics"]["features"][110]["observed"]
    assert result["lime"]["perturbation_constraints"] is None
    assert "unconstrained" in result["lime"]["limitations"]
    assert result == explain_additional(booster, values, names, classes, reference)


def test_canonical_packet_lime_constrains_counts_by_names_and_preserves_missing_anchor(tmp_path):
    pytest.importorskip("lime")
    xgb = pytest.importorskip("xgboost")
    from framework.detection.packet_xgboost import FEATURE_NAMES
    rng = np.random.default_rng(42)
    # Canonical counts are deliberately reordered and surrounded by unavailable
    # numeric/categorical columns; constraints must not assume column zero.
    names = ["log_missing", *reversed(FEATURE_NAMES), "network_protocols_all__present_tcp"]
    training = rng.integers(0, 10, size=(90, len(names))).astype(np.float32)
    training[:, names.index(FEATURE_NAMES[0])] += 20
    training[:, -1] %= 2
    classes = ["fixture_benign", "fixture_other"]
    booster = xgb.train({"objective": "multi:softprob", "num_class": 2, "device": "cpu", "nthread": 2,
                         "max_depth": 2, "seed": 42}, xgb.DMatrix(training, label=np.arange(90) % 2, nthread=2), 3)
    values = training[0].copy()
    values[0] = values[-1] = np.nan
    values[names.index(FEATURE_NAMES[-1])] = 0
    reference = tmp_path / "packet_reference.npz"
    np.savez_compressed(reference, training=training, medians=np.median(training, axis=0))
    reference.with_suffix(".json").write_text(json.dumps({"feature_names": names, "categorical_indices": [8],
        "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(), "kind": "Synthetic unit fixture"}))
    calls = []

    class RecordingBooster:
        def predict(self, matrix, **kwargs):
            coo = matrix.get_data().tocoo()
            dense = np.full(coo.shape, np.nan)
            dense[coo.row, coo.col] = coo.data
            calls.append(dense)
            return booster.predict(matrix, **kwargs)

        def __getattr__(self, name):
            return getattr(booster, name)

    result = explain_additional(RecordingBooster(), values, names, classes, reference, packet=False)
    assert result["lime"]["perturbation_constraints"] == "integer packet and TCP-flag counts"
    assert "rounded and constrained" in result["lime"]["limitations"]
    for call in calls:
        np.testing.assert_equal(call[0], values)
        assert np.isnan(call[:, [0, -1]]).all()
        counts = call[:, [names.index(name) for name in FEATURE_NAMES]]
        assert np.all(counts == np.floor(counts)) and np.all(counts >= 0)
        assert np.all(counts[:, 0] >= 1) and np.all(counts[:, 1:] <= counts[:, :1])
    assert calls[-1].shape == (2048, len(names))
    scores = booster.predict(xgb.DMatrix(values.reshape(1, -1), nthread=2))[0]
    assert result["lime"]["model_score"] == pytest.approx(float(scores.max()))


@pytest.fixture
def partial_evidence():
    return {"source": {"kind": "Actual local packet observation", "mode": "packets", "observation_id": "synthetic-contract-fixture"},
            "prediction": {"label": "SYN flooding (DDoS)", "score": 0.4, "is_attack": True,
                           "top_classes": [{"label": "synthetic_a", "display_label": "SYN flooding (DDoS)", "score": 0.4},
                                           {"label": "synthetic_b", "display_label": "Port scanning (Reconnaissance)", "score": 0.3},
                                           {"label": "synthetic_c", "display_label": "Normal network activity", "score": 0.2}]},
            "feature_coverage": {"observed": 12, "total": 170, "missing": 158,
                                 "profile": "synthetic_partial_contract_fixture", "validated_profile": False},
            "xai": {"method": "Exact TreeSHAP", "features": [
                {"feature": "network_protocols_all__present_tcp", "value": None, "is_missing": True, "shap_value": 0.4},
                {"feature": "network_tcp-flags-syn_count", "value": 0, "is_missing": False, "shap_value": -0.2}]},
            "rules": None}


def test_partial_prompt_supplies_tentative_categories_and_never_turns_missing_into_measurements(partial_evidence):
    text = " ".join(generation_facts(partial_evidence))
    assert check_grounding(text, partial_evidence)["passed"]
    assert "Partial input coverage limits evidence: 12 of 170" in text
    assert "Unavailable protocols all present tcp input: its model default supported" in text
    assert "not measured traffic" in text and "Tentative alternative categories" in text
    assert "Port scanning (Reconnaissance)" in text and "Normal network activity" in text
    assert "not been validated" in text and "human verification" in text


@pytest.mark.parametrize("change,issue", [
    (lambda facts: [f for f in facts if not f.startswith("Partial")], "partial_input_warning_missing"),
    (lambda facts: [f for f in facts if not f.startswith("Tentative")], "tentative_category_missing_or_changed"),
    (lambda facts: [f for f in facts if "not been validated" not in f], "unvalidated_profile_warning_missing"),
    (lambda facts: [f.replace("Unavailable protocols all present tcp input: its model default", "The protocols all present tcp feature") for f in facts],
     "missing_input_presented_as_observed:network_protocols_all__present_tcp"),
    (lambda facts: [*facts, "We measured 999 packets."], "unsupported_number"),
])
def test_partial_grounding_rejects_lost_context_or_fabricated_measurements(partial_evidence, change, issue):
    assert issue in check_grounding(" ".join(change(generation_facts(partial_evidence))), partial_evidence)["issues"]


def test_partial_generation_keeps_two_attempt_limit_and_exact_background(monkeypatch, partial_evidence):
    from evaluation.test_rag_grounding import fixture_provider
    facts = generation_facts(partial_evidence)
    good = " ".join(facts)
    bad = " ".join(f for f in facts if not f.startswith("Partial"))
    provider, model = fixture_provider(monkeypatch, [bad, good])
    result = provider.generate(deepcopy(partial_evidence))
    assert len(model.prompts) == 2 and result["correction"]["retried"]
    assert result["evidence"]["source"]["observation_id"] == "synthetic-contract-fixture"
    for prompt in model.prompts:
        assert result["rag"]["context"] in prompt["messages"][1]["content"]
        assert prompt["max_tokens"] == 200
    assert result["cpu_threads"] == 2 and result["context_tokens"] == 2048
    assert model.closed and provider._model is None and not provider._lock.locked()
