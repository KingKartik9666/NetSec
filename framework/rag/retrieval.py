"""Deterministic CPU TF-IDF retrieval with one bounded evidence-query correction.

This retrieves checked explanatory knowledge, never observations, predictions or
new research results. There is no network client, embedding model or vector DB.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

METHOD = "Local lexical TF-IDF cosine similarity"
MAX_PASSAGE_BYTES = 300
MAX_CONTEXT_BYTES = 700
MIN_SCORE = 0.14
CORRECTION_THRESHOLD = 0.32
STOPWORDS = frozenset("a an and are as at be by for from in is it of on or that the this to with".split())


def _tokens(value):
    return [token for token in re.findall(r"[a-z0-9]+", value.lower())
            if len(token) > 1 and token not in STOPWORDS]


def _vector(tokens, idf):
    counts = Counter(token for token in tokens if token in idf)
    vector = {token: (1 + math.log(count)) * idf[token] for token, count in counts.items()}
    norm = math.sqrt(sum(weight * weight for weight in vector.values()))
    return {token: weight / norm for token, weight in vector.items()} if norm else {}


class LocalKnowledgeBase:
    """Load a small validated corpus once; expose citations without opening sources.

    Source hashes describe the checked curation snapshot, which can differ from
    later documentation revisions. The corpus and returned passage hashes are
    always computed/validated against the exact bytes used for this request.
    """

    def __init__(self, directory=None):
        selected = Path(directory) if directory is not None else Path(__file__).parent
        self.corpus_path = selected if selected.suffix.lower() == ".json" else selected / "knowledge.json"
        self._chunks, self._vectors, self._idf = [], [], {}
        self._sha256 = None
        self._error = None
        try:
            raw = self.corpus_path.read_bytes()
            if len(raw) > 128 * 1024:
                raise ValueError("Knowledge corpus exceeds its bounded size")
            data = json.loads(raw)
            if data.get("schema_version") != 1 or not isinstance(data.get("chunks"), list) or not 1 <= len(data["chunks"]) <= 64:
                raise ValueError("Knowledge corpus schema or chunk count is invalid")
            identifiers = set()
            for chunk in data["chunks"]:
                if not isinstance(chunk, dict):
                    raise ValueError("Knowledge chunk must be an object")
                identifier, text = chunk.get("id"), chunk.get("text")
                if not isinstance(identifier, str) or not re.fullmatch(r"[a-z0-9-]{1,48}", identifier) or identifier in identifiers:
                    raise ValueError("Knowledge chunk identifiers are invalid or duplicated")
                if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > MAX_PASSAGE_BYTES:
                    raise ValueError("Knowledge passage text exceeds its bounded size")
                if chunk.get("chunk_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest():
                    raise ValueError("Knowledge passage checksum does not match")
                if not isinstance(chunk.get("title"), str) or not 1 <= len(chunk["title"]) <= 100:
                    raise ValueError("Knowledge passage title is invalid")
                if not isinstance(chunk.get("scopes"), list) or not chunk["scopes"] or any(scope not in {"common", "packet", "csv"} for scope in chunk["scopes"]):
                    raise ValueError("Knowledge passage scopes are invalid")
                if not isinstance(chunk.get("terms"), list) or not all(isinstance(term, str) and len(term) <= 80 for term in chunk["terms"]):
                    raise ValueError("Knowledge passage terms are invalid")
                source = chunk.get("source", {})
                if not isinstance(source, dict) or not isinstance(source.get("file"), str) or not isinstance(source.get("section"), str) or not re.fullmatch(r"[a-f0-9]{64}", source.get("sha256", "")):
                    raise ValueError("Knowledge source provenance is invalid")
                identifiers.add(identifier)
            chunks = data["chunks"]
            documents = [_tokens(chunk["title"] + " " + " ".join(chunk["terms"]) + " " + chunk["text"]) for chunk in chunks]
            frequencies = Counter(token for document in documents for token in set(document))
            self._idf = {token: math.log((1 + len(documents)) / (1 + count)) + 1 for token, count in frequencies.items()}
            self._vectors = [_vector(document, self._idf) for document in documents]
            self._chunks = chunks
            self._sha256 = hashlib.sha256(raw).hexdigest()
        except (OSError, ValueError, TypeError, AttributeError) as error:
            self._chunks, self._vectors, self._idf = [], [], {}
            self._error = f"Local knowledge unavailable: {type(error).__name__}: {error}"

    def status(self):
        return {"available": bool(self._chunks), "chunks": len(self._chunks), "method": METHOD,
                "retriever": "stdlib TF-IDF", "offline": True, "device": "cpu", "llm_executed": False,
                "corpus_sha256": self._sha256, "error": self._error,
                "max_results": 2, "max_context_bytes": MAX_CONTEXT_BYTES,
                "minimum_similarity": MIN_SCORE, "correction_threshold": CORRECTION_THRESHOLD,
                "max_retrieval_attempts": 2}

    def _search(self, query, scope):
        vector = _vector(_tokens(query), self._idf)
        found = []
        for chunk, weights in zip(self._chunks, self._vectors):
            if "common" not in chunk["scopes"] and scope not in chunk["scopes"]:
                continue
            score = sum(weight * weights.get(token, 0) for token, weight in vector.items())
            if score >= MIN_SCORE:
                found.append((score, chunk))
        return sorted(found, key=lambda item: (-item[0], item[1]["id"]))

    def _result(self, query, scope, results, attempts, max_results, *, correction=False):
        passages, context = [], ""
        for score, chunk in results[:max(0, min(int(max_results), 2))]:
            cited = f"[{chunk['id']}] {chunk['text']}"
            candidate = context + ("\n" if context else "") + cited
            if len(candidate.encode("utf-8")) > MAX_CONTEXT_BYTES:
                break
            context = candidate
            passages.append({"id": chunk["id"], "title": chunk["title"], "text": chunk["text"],
                             "scopes": list(chunk["scopes"]), "score": round(score, 6),
                             "source": dict(chunk["source"]), "chunk_sha256": chunk["chunk_sha256"],
                             "corpus_sha256": self._sha256})
        return {"available": bool(self._chunks), "method": METHOD, "offline": True, "device": "cpu",
                "query": query, "scope": scope, "matched": bool(passages), "passages": passages,
                "context": context, "context_bytes": len(context.encode("utf-8")), "context_chars": len(context),
                "max_context_bytes": MAX_CONTEXT_BYTES, "max_passage_bytes": MAX_PASSAGE_BYTES,
                "corpus_sha256": self._sha256, "correction_attempted": correction,
                "retrieval_attempts": attempts, "error": self._error,
                "limitations": "Small curated project corpus; lexical relevance thresholds are heuristic. Retrieval is background knowledge, not traffic evidence, incident confirmation or measured reasoning accuracy."}

    def retrieve_query(self, query, *, scope="common", max_results=2):
        """Search a literal query once; unrelated text never gets a default match."""
        if scope not in {"common", "packet", "csv"}:
            raise ValueError("Knowledge scope must be common, packet or csv")
        query = str(query)[:480]
        results = self._search(query, scope)
        attempts = [{"query": query, "top_score": round(results[0][0], 6) if results else 0.0,
                     "reason": "literal query", "matched": bool(results)}]
        return self._result(query, scope, results, attempts, max_results)

    def retrieve(self, evidence, max_results=2):
        """Construct a query from checked evidence and correct weak matches once."""
        source = evidence.get("source", {})
        kind = str(source.get("kind", "")).lower()
        if source.get("mode") == "packets" or "packet observation" in kind:
            scope = "packet"
        elif "datasense" in kind or "uploaded feature" in kind or kind in {"sample", "features"}:
            scope = "csv"
        else:
            return self._result("", "common", [], [], max_results)
        prediction, xai = evidence.get("prediction", {}), evidence.get("xai", {})
        features = xai.get("features", [])[:2]
        parts = [str(prediction.get("label", ""))[:80], str(xai.get("method", ""))[:100]]
        parts.extend(str(feature.get("feature", ""))[:80].replace("_", " ").replace("-", " ") for feature in features)
        rules = evidence.get("rules") or {}
        disagreement = (rules.get("status") == "benign_looking" and prediction.get("is_attack") is True) or (rules.get("status") == "suspicious" and prediction.get("is_attack") is False)
        if rules:
            parts.append(str(rules.get("status", ""))[:40].replace("_", " "))
        if disagreement:
            parts.append("rule model disagreement rule model disagreement")
        if evidence.get("lime"):
            parts.append("LIME local fidelity")
        query = " ".join(part for part in parts if part).strip()[:480]
        results = self._search(query, scope)
        attempts = [{"query": query, "top_score": round(results[0][0], 6) if results else 0.0,
                     "reason": "structured evidence query", "matched": bool(results)}]
        correction = False
        # Expand only structurally supported detector/XAI evidence. Arbitrary user
        # text goes through retrieve_query and is never repaired into a fake match.
        has_shap = any(isinstance(feature.get("shap_value"), (float, int)) and math.isfinite(feature["shap_value"]) for feature in features)
        if self._chunks and (not results or results[0][0] < CORRECTION_THRESHOLD) and type(prediction.get("is_attack")) is bool and has_shap:
            correction = True
            topic = "rule model disagreement" if disagreement else "SHAP signed raw margin directions model prediction uncertainty"
            query = (query[:300] + " " + topic).strip()
            results = self._search(query, scope)
            attempts.append({"query": query, "top_score": round(results[0][0], 6) if results else 0.0,
                             "reason": "bounded evidence-topic expansion", "matched": bool(results)})
        return self._result(query, scope, results, attempts, max_results, correction=correction)
