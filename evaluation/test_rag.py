"""Local retrieval is scoped, bounded and auditable; it is not web research."""
import hashlib
import json
import pytest

from framework.rag import LocalKnowledgeBase


def packet_evidence():
    return {"source": {"kind": "Actual local packet observation", "mode": "packets"},
            "prediction": {"label": "Suspicious traffic", "is_attack": True, "score": 0.8},
            "xai": {"method": "Exact native TreeSHAP", "features": [
                {"feature": "network_packets_count", "shap_value": 1.2},
                {"feature": "network_tcp-flags-syn", "shap_value": -0.3}]},
            "rules": {"status": "benign_looking"}}


def test_default_corpus_has_cpu_offline_provenance_and_no_generation():
    knowledge = LocalKnowledgeBase()
    status = knowledge.status()
    assert status["available"] and status["chunks"] >= 10
    assert status["device"] == "cpu" and status["offline"] is True
    assert status["llm_executed"] is False
    assert status["corpus_sha256"] == hashlib.sha256(knowledge.corpus_path.read_bytes()).hexdigest()


def test_actual_rule_model_disagreement_is_retrieved_first_and_is_deterministic():
    knowledge = LocalKnowledgeBase()
    result = knowledge.retrieve(packet_evidence())
    assert result == knowledge.retrieve(packet_evidence())
    assert result["matched"] is True
    assert result["passages"][0]["id"] == "live-rule-disagreement"
    assert all("csv" not in passage["scopes"] for passage in result["passages"])


def test_csv_scope_does_not_return_packet_specific_knowledge():
    result = LocalKnowledgeBase().retrieve_query("DataSense CSV sensor log features validation", scope="csv")
    assert result["matched"]
    assert result["passages"][0]["id"] == "datasense-csv-scope"
    assert all("packet" not in passage["scopes"] for passage in result["passages"])


@pytest.mark.parametrize("query", ["", "banana cheesecake lunar agriculture", "the and is to for"])
def test_unrelated_queries_abstain_without_correction(query):
    result = LocalKnowledgeBase().retrieve_query(query, scope="packet")
    assert result["matched"] is False and result["passages"] == [] and result["context"] == ""
    assert result["correction_attempted"] is False


def test_every_injected_passage_has_exact_content_hash_and_source_snapshot():
    knowledge = LocalKnowledgeBase()
    result = knowledge.retrieve(packet_evidence())
    for passage in result["passages"]:
        assert passage["chunk_sha256"] == hashlib.sha256(passage["text"].encode("utf-8")).hexdigest()
        assert passage["corpus_sha256"] == result["corpus_sha256"]
        assert passage["source"]["file"] and passage["source"]["section"]
        assert len(passage["source"]["sha256"]) == 64
        assert passage["text"] in result["context"]


def test_context_and_result_count_stay_bounded_even_when_more_are_requested():
    result = LocalKnowledgeBase().retrieve_query(
        "SHAP LIME fidelity gain cover calibrated scores human verification uncertainty", scope="csv", max_results=50)
    assert result["matched"] and len(result["passages"]) <= 2
    assert len(result["context"].encode("utf-8")) <= 700
    assert all(len(passage["text"].encode("utf-8")) <= 300 for passage in result["passages"])


def test_weak_supported_evidence_gets_one_bounded_query_correction():
    evidence = packet_evidence()
    evidence["prediction"]["label"] = "Quiet activity"
    evidence["prediction"]["is_attack"] = False
    evidence["xai"]["method"] = "arbor explanation"
    evidence["xai"]["features"] = [{"feature": "opaque_a", "shap_value": 0.4}]
    evidence["rules"] = None
    result = LocalKnowledgeBase().retrieve(evidence)
    assert result["matched"] and result["correction_attempted"] is True
    assert len(result["retrieval_attempts"]) == 2
    assert result["retrieval_attempts"][1]["reason"] == "bounded evidence-topic expansion"
    assert "SHAP" in result["query"]


def test_unsupported_source_never_fills_prompt_with_a_default_passage():
    result = LocalKnowledgeBase().retrieve({"source": {"kind": "Unrelated weather record"},
                                           "prediction": {"label": "Sunny"}})
    assert result["matched"] is False and result["passages"] == []
    assert result["correction_attempted"] is False


@pytest.mark.parametrize("contents", [None, "not JSON", json.dumps({"schema_version": 1, "chunks": []})])
def test_missing_or_invalid_corpus_reports_unavailable_without_fabrication(tmp_path, contents):
    corpus = tmp_path / "knowledge.json"
    if contents is not None:
        corpus.write_text(contents, encoding="utf-8")
    knowledge = LocalKnowledgeBase(tmp_path)
    assert knowledge.status()["available"] is False
    result = knowledge.retrieve(packet_evidence())
    assert result["available"] is False and result["matched"] is False
    assert result["passages"] == [] and result["context"] == "" and result["error"]


def test_invalid_content_hash_is_rejected(tmp_path):
    source = LocalKnowledgeBase().corpus_path
    data = json.loads(source.read_text(encoding="utf-8"))
    data["chunks"][0]["text"] = "Changed statement with an unchanged content hash."
    corpus = tmp_path / "knowledge.json"
    corpus.write_text(json.dumps(data), encoding="utf-8")
    assert LocalKnowledgeBase(corpus).status()["available"] is False
