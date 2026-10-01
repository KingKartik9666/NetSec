"""Software checks for the synthetic MVP; these are not research benchmarks."""

from dataclasses import asdict
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from experiment.demo import create_pipeline
from framework.data.loader import FEATURES, PROJECT_ROOT, load_demo_data, select_sample, training_data
from framework.detection.demo_detector import DemoDetector
from framework.detection.hem_adapter import HEMAdapter
from framework.llm.prompt_builder import build_prompt
from framework.llm.provider import TemplateProvider, choose_provider


@pytest.fixture(scope="module")
def traffic() -> pd.DataFrame:
    return load_demo_data()


@pytest.fixture(scope="module")
def pipeline(traffic):
    return create_pipeline(traffic)


def test_demo_dataset_loads_with_declared_features(traffic):
    assert {"sample_id", "split", "label", *FEATURES} <= set(traffic.columns)
    assert len(traffic) == 184
    assert traffic["sample_id"].is_unique
    assert np.isfinite(traffic.loc[:, list(FEATURES)].to_numpy(dtype=float)).all()
    assert set(traffic.loc[traffic["split"] == "demo", "sample_id"]) == {
        "benign", "ddos", "scan", "unknown",
    }


def test_training_excludes_demonstration_rows_labels_and_metadata(traffic, pipeline):
    X, y = training_data(traffic)
    assert tuple(X.columns) == FEATURES
    assert len(X) == len(y) == 180
    assert set(y) == {"BENIGN", "DDoS", "PORT_SCAN"}
    assert set(pipeline.detector.classifier.classes_) == set(y)
    assert set(X.index).isdisjoint(traffic.index[traffic["split"] == "demo"])
    np.testing.assert_allclose(pipeline.detector.scaler.mean_, X.mean(axis=0))

    changed = traffic.copy()
    changed.loc[changed["split"] == "demo", list(FEATURES)] = 1e9
    unchanged_X, unchanged_y = training_data(changed)
    pd.testing.assert_frame_equal(X, unchanged_X)
    pd.testing.assert_series_equal(y, unchanged_y)


def test_unknown_rows_cannot_be_added_to_training(traffic):
    invalid = traffic.copy()
    invalid.loc[invalid["sample_id"] == "unknown", "split"] = "train"
    with pytest.raises(ValueError, match="Training"):
        training_data(invalid)


@pytest.mark.parametrize(
    "sample_name, expected_label, expected_attack, expected_novelty",
    [
        ("benign", "BENIGN", False, "known"),
        ("ddos", "DDoS", True, "known"),
        ("scan", "PORT_SCAN", True, "known"),
        ("unknown", "UNKNOWN", True, "unknown"),
    ],
)
def test_detector_schema_and_obvious_examples(
    traffic, pipeline, sample_name, expected_label, expected_attack, expected_novelty,
):
    sample = select_sample(traffic, sample_name)
    assert set(sample) == set(FEATURES)
    prediction = asdict(pipeline.detector.predict(sample))
    assert {"is_attack", "attack_type", "novelty", "confidence"} <= set(prediction)
    assert prediction["attack_type"] == expected_label
    assert prediction["is_attack"] is expected_attack
    assert prediction["novelty"] == expected_novelty
    assert prediction["detector_name"] == "DemoDetector"
    assert 0 <= prediction["classifier_confidence"] <= 1
    if expected_novelty == "unknown":
        assert prediction["confidence"] is None
        assert prediction["novelty_score"] > prediction["novelty_threshold"]
    else:
        assert 0 <= prediction["confidence"] <= 1
        assert prediction["novelty_score"] <= prediction["novelty_threshold"]


@pytest.mark.parametrize("sample_name", ["benign", "ddos", "scan", "unknown"])
def test_pipeline_reports_real_feature_evidence(traffic, pipeline, sample_name):
    sample = select_sample(traffic, sample_name)
    result = pipeline.analyze(sample)
    assert {"detection", "explanation", "context", "report"} <= set(result)
    assert isinstance(result["report"], str) and result["report"].strip()
    assert result["detection"]["attack_type"] in result["report"]
    json.dumps(result, allow_nan=False)
    important = result["explanation"]["important_features"]
    assert important
    assert len({entry["feature"] for entry in important}) == len(important)
    for entry in important:
        assert entry["feature"] in FEATURES
        assert entry["value"] == pytest.approx(sample[entry["feature"]])
        assert 0 <= entry["importance"] <= 1
        assert entry["feature"] in result["report"]
    scores = [entry["importance"] for entry in important]
    assert scores == sorted(scores, reverse=True)
    assert sum(scores) <= 1 + 1e-12


def test_unknown_explanation_matches_novelty_distance(traffic, pipeline):
    result = pipeline.analyze(select_sample(traffic, "unknown"))
    distance_squared = result["detection"]["novelty_score"] ** 2
    assert distance_squared > 0
    for feature in result["explanation"]["important_features"]:
        expected_share = feature["standardized_deviation"] ** 2 / distance_squared
        assert feature["importance"] == pytest.approx(expected_share)
    assert result["explanation"]["important_features"][0]["feature"] == "avg_packet_size"


def test_template_provider_is_deterministic_and_uses_evidence(traffic, pipeline):
    provider = TemplateProvider()
    known = pipeline.analyze(select_sample(traffic, "ddos"))["context"]
    unknown = pipeline.analyze(select_sample(traffic, "unknown"))["context"]
    first = provider.generate(known)
    assert first == provider.generate(known)
    assert "DDoS" in first
    assert provider.generate(unknown) != first
    assert "UNKNOWN" in provider.generate(unknown)


def test_prompt_contains_evidence_but_excludes_extra_metadata(traffic, pipeline):
    context = pipeline.analyze(select_sample(traffic, "scan"))["context"]
    context = dict(context, arbitrary_note="EXCLUDED_METADATA_SENTINEL")
    prompt = build_prompt(context)
    assert "PORT_SCAN" in prompt
    assert "EXCLUDED_METADATA_SENTINEL" not in prompt
    for feature in context["explanation"]["important_features"]:
        assert feature["feature"] in prompt
        assert str(feature["value"]) in prompt


def test_retraining_has_repeatable_results(traffic, pipeline):
    second = create_pipeline(traffic)
    for sample_name in ("benign", "ddos", "scan", "unknown"):
        sample = select_sample(traffic, sample_name)
        assert pipeline.analyze(sample) == second.analyze(sample)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -1, "invalid"])
def test_detector_rejects_invalid_features(traffic, pipeline, bad_value):
    sample = select_sample(traffic, "ddos")
    sample["packet_rate"] = bad_value
    with pytest.raises(ValueError):
        pipeline.analyze(sample)


def test_detector_rejects_missing_feature(traffic, pipeline):
    sample = select_sample(traffic, "ddos")
    del sample["syn_ratio"]
    with pytest.raises(ValueError, match="Missing"):
        pipeline.analyze(sample)


def test_detector_rejects_invalid_syn_fraction(traffic, pipeline):
    sample = select_sample(traffic, "ddos")
    sample["syn_ratio"] = 1.1
    with pytest.raises(ValueError, match="syn_ratio"):
        pipeline.analyze(sample)


def test_loader_reports_missing_columns(tmp_path, traffic):
    malformed = tmp_path / "missing_feature.csv"
    traffic.drop(columns="byte_rate").to_csv(malformed, index=False)
    with pytest.raises(ValueError, match="Missing"):
        load_demo_data(malformed)


def test_loader_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_demo_data(tmp_path / "absent.csv")


def test_unfitted_detector_rejects_inference(traffic):
    with pytest.raises(ValueError, match="fitted"):
        DemoDetector().predict(select_sample(traffic, "benign"))


def test_hem_adapter_never_silently_substitutes_demo(traffic):
    adapter = HEMAdapter()
    assert adapter.available() is False
    assert isinstance(adapter.status(), dict)
    with pytest.raises(RuntimeError, match="HEM"):
        adapter.predict(select_sample(traffic, "ddos"))
    X, y = training_data(traffic)
    with pytest.raises(RuntimeError, match="HEM"):
        adapter.fit(X, y)


def test_artifact_filenames_do_not_enable_unimplemented_hem(tmp_path, traffic):
    adapter = HEMAdapter(hem_root=tmp_path)
    source = tmp_path / "model_evaluation" / "full_model.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Placeholder inventory fixture, never imported.\n", encoding="utf-8")
    for relative_path in adapter.required_artifacts:
        artifact = tmp_path / relative_path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"Not a model; must never be deserialized.")
    status = adapter.status()
    assert status["source_present"] is True
    assert status["missing_artifacts"] == []
    assert len(status["found_artifacts"]) == len(adapter.required_artifacts)
    assert adapter.available() is False
    with pytest.raises(RuntimeError, match="bridge not implemented"):
        adapter.predict(select_sample(traffic, "ddos"))


def test_default_provider_needs_no_api_key(monkeypatch):
    monkeypatch.delenv("LLM_MODE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert isinstance(choose_provider(), TemplateProvider)


def test_unsupported_provider_has_clear_error():
    with pytest.raises(ValueError, match="supports only.*template"):
        choose_provider("unimplemented-provider")


def test_default_cli_runs_without_network_or_keys_from_another_directory(tmp_path):
    """Block networking before importing the entrypoint or its dependencies."""
    entry = PROJECT_ROOT / "run_demo.py"
    bootstrap = "\n".join([
        "import runpy, socket, sys",
        "def denied(*args, **kwargs):",
        "    raise AssertionError('Network access attempted by offline demo')",
        "class NoNetworkSocket(socket.socket):",
        "    __init__ = denied",
        "    connect = denied",
        "    connect_ex = denied",
        "socket.socket = NoNetworkSocket",
        "socket.create_connection = denied",
        "socket.getaddrinfo = denied",
        "socket.gethostbyname = denied",
        "socket.gethostbyname_ex = denied",
        "socket.gethostbyaddr = denied",
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})",
        f"sys.argv = [{str(entry)!r}]",
        f"runpy.run_path({str(entry)!r}, run_name='__main__')",
    ])
    environment = {
        key: value for key, value in os.environ.items()
        if "API_KEY" not in key.upper() and not key.upper().endswith("_TOKEN")
        and key.upper() != "LLM_MODE"
    }
    # Exercise built-in defaults independently of a developer's optional .env.
    environment["PYTHON_DOTENV_DISABLED"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", bootstrap], cwd=tmp_path, env=environment,
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "DDoS" in completed.stdout
    assert "TemplateProvider" in completed.stdout
    assert "DemoDetector" in completed.stdout
    assert "HEM" in completed.stdout
