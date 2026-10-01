"""Bounded correction contracts; strings here are fixtures, not LLM results."""
from copy import deepcopy

import pytest

from framework.llm.local_model import LocalLanguageModel
from framework.rag.grounding import check_grounding


@pytest.fixture
def evidence():
    return {"source": {"kind": "Actual local packet observation", "mode": "packets"},
            "prediction": {"label": "Suspicious traffic pattern", "is_attack": True},
            "xai": {"method": "Exact TreeSHAP", "features": [
                {"feature": "network_packets_all_count", "shap_value": 0.4},
                {"feature": "network_tcp-flags-syn_count", "shap_value": -0.2}]},
            "rules": None}


GOOD = ("The detector predicted Suspicious traffic pattern. The packets all count feature supported "
        "this prediction. The TCP syn count feature opposed this prediction. "
        "This prediction needs human verification and does not confirm an attack or guarantee safety.")


def test_measured_names_directions_and_uncertainty_with_none_rules_are_accepted(evidence):
    assert check_grounding(GOOD, evidence)["passed"]


@pytest.mark.parametrize("text,issue", [
    (GOOD.replace("Suspicious traffic pattern", "normal traffic"), "predicted_label_missing_or_changed"),
    (GOOD + " It is 99.97% certain.", "unsupported_percentage"),
    (GOOD.replace("count feature supported", "count feature opposed"), "feature_direction_changed:network_packets_all_count"),
    (GOOD.replace("count feature opposed", "count feature supports"), "feature_direction_changed:network_tcp-flags-syn_count"),
    (GOOD.replace("packets all count", "something else"), "feature_name_missing:network_packets_all_count"),
    (GOOD.split("This prediction needs")[0], "uncertainty_missing"),
    ("Confirmed attack. " + GOOD, "unsupported_incident_or_safety_claim"),
    (GOOD + " [invented-evidence]", "invented_source_identifier"),
])
def test_unsupported_rewrites_are_rejected(evidence, text, issue):
    assert issue in check_grounding(text, evidence)["issues"]


def test_rule_disagreement_is_required_and_truncation_is_rejected(evidence):
    evidence["rules"] = {"status": "benign_looking"}
    assert "rule_model_disagreement_missing" in check_grounding(GOOD, evidence)["issues"]
    assert check_grounding(GOOD + " The rules did not flag this pattern and disagree with the model.", evidence)["passed"]
    assert "truncated_response" in check_grounding(GOOD, evidence, finish_reason="length")["issues"]


@pytest.mark.parametrize("prefix", [
    "Rewrite the facts and fix these issues: ",
    "Output only the paragraph. ",
    "predicted_label_missing_or_changed: ",
    "**Rewritten Facts and Fixes:** ",
])
def test_instruction_or_diagnostic_echo_is_rejected_even_when_facts_follow(evidence, prefix):
    assert "instruction_or_diagnostic_echo" in check_grounding(prefix + GOOD, evidence)["issues"]


class FixtureLanguageModel:
    """No model weights or inference are used in these lifecycle tests."""
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []
        self.closed = False

    def tokenize(self, content):
        return content.split()

    def create_chat_completion(self, **kwargs):
        self.prompts.append(deepcopy(kwargs))
        return {"choices": [{"message": {"content": next(self.responses)}, "finish_reason": "stop"}]}

    def close(self):
        self.closed = True


def fixture_provider(monkeypatch, responses):
    provider = LocalLanguageModel()
    model = FixtureLanguageModel(responses)
    provider._model = model
    monkeypatch.setattr(provider, "status", lambda: {"available": True})
    return provider, model


def test_one_retry_uses_exact_retrieved_background_and_releases_resources(monkeypatch, evidence):
    provider, model = fixture_provider(monkeypatch, [GOOD.replace("count feature supported", "count feature opposed"), GOOD])
    result = provider.generate(evidence)
    assert len(model.prompts) == 2 and result["correction"]["retried"]
    assert result["correction"]["generation_attempts"][-1]["passed"]
    first = model.prompts[0]["messages"][1]["content"]
    second = model.prompts[1]["messages"][1]["content"]
    assert result["rag"]["used_in_generation"]
    assert result["rag"]["context"] in first and result["rag"]["context"] in second
    assert "feature_direction_changed" not in second
    assert "previous response did not preserve every fact" in second
    assert "Copy every FACT sentence exactly" in model.prompts[0]["messages"][0]["content"]
    assert model.closed and provider._model is None and not provider._lock.locked()


def test_two_invalid_responses_abstain_without_template_and_release_resources(monkeypatch, evidence):
    provider, model = fixture_provider(monkeypatch, ["Definitely safe.", "Confirmed attack."])
    with pytest.raises(ValueError, match="two attempts.*No substitute"):
        provider.generate(evidence)
    assert len(model.prompts) == 2
    assert model.closed and provider._model is None and not provider._lock.locked()
