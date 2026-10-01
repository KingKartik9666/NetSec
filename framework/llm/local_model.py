"""Actual offline Qwen generation through llama.cpp; no network, tools or fallback."""
import hashlib
import importlib.util
import os
from pathlib import Path
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "Qwen2.5-0.5B-Instruct · Q4_K_M"
FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
SHA256 = "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"


def generation_facts(evidence):
    """Bounded facts preserve the difference between observations and missingness."""
    from framework.rag.grounding import feature_is_missing, feature_name, hypothesis_labels
    prediction = evidence["prediction"]
    facts = [f"For {evidence['source']['kind']}, the detector predicted {prediction['label']}."]
    alternatives = [label for label in hypothesis_labels(evidence) if label != prediction["label"]]
    if alternatives:
        facts.append("Tentative alternative categories: " + "; ".join(alternatives) + ".")
    coverage = evidence.get("feature_coverage") or {}
    if coverage.get("missing", 0):
        facts.append(f"Partial input coverage limits evidence: {coverage['observed']} of {coverage['total']} inputs observed.")
    if coverage.get("validated_profile") is False:
        facts.append("This input profile has not been validated.")
    for feature in evidence["xai"]["features"][:2]:
        name = feature_name(feature["feature"])
        direction = "supported" if feature["shap_value"] > 0 else "opposed" if feature["shap_value"] < 0 else "did not affect"
        if feature_is_missing(feature):
            facts.append(f"Unavailable {name} input: its model default {direction} this prediction, not measured traffic.")
        else:
            facts.append(f"The {name} feature {direction} this prediction according to SHAP.")
    rules = evidence.get("rules")
    if rules and rules["status"] == "benign_looking" and prediction["is_attack"]:
        facts.append("The separate behavior rules did not flag a suspicious pattern, so the model and rules disagree.")
    facts.append("This model prediction needs human verification and does not confirm an attack or guarantee safety.")
    return facts


class LocalLanguageModel:
    def __init__(self):
        self.directory = Path(os.environ.get("IMDAI_LLM_DIR", ROOT / "artifacts/llm/qwen"))
        self._model = None
        self._lock = threading.Lock()
        self._knowledge = None

    def knowledge(self):
        if self._knowledge is None:
            from framework.rag import LocalKnowledgeBase
            self._knowledge = LocalKnowledgeBase()
        return self._knowledge

    def status(self):
        try:
            rag = self.knowledge().status()
        except (ValueError, OSError) as error:
            rag = {"available": False, "error": str(error), "offline": True}
        return {"available": (self.directory / FILENAME).is_file() and importlib.util.find_spec("llama_cpp") is not None and rag["available"],
                "model": MODEL_NAME, "provider": "Local llama.cpp", "device": "cpu", "offline": True,
                "loaded": self._model is not None, "busy": self._lock.locked(), "automatic_generation": False,
                "cpu_threads": 2, "context_tokens": 2048, "max_output_tokens": 200,
                "release_after_generation": True, "rag": rag, "max_generation_attempts": 2}

    def generate(self, evidence):
        if not self._lock.acquire(blocking=False):
            raise ValueError("A local explanation is already being generated; wait for it to finish")
        try:
            if not self.status()["available"]:
                raise ValueError("Local language model unavailable. Run python -m experiment.setup_local_llm and install desktop requirements.")
            started = time.monotonic()
            if self._model is None:
                model_file = self.directory / FILENAME
                digest = hashlib.sha256()
                with model_file.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != SHA256:
                    raise ValueError("Local Qwen weights differ from the verified official model")
                from llama_cpp import Llama
                self._model = Llama(model_path=str(model_file), n_ctx=2048, n_threads=2, n_threads_batch=2,
                                    n_batch=128, n_gpu_layers=0,
                                    seed=42, chat_format="chatml", verbose=False)
            from framework.rag.grounding import check_grounding
            retrieval = self.knowledge().retrieve(evidence, max_results=2)
            system = (
                "Join every FACT sentence below into one paragraph. Copy every FACT sentence exactly; "
                "do not omit any sentence, category name, feature name or warning. "
                "Reference notes are context only; do not copy them. Output only the joined FACT sentences."
            )
            # A 0.5B model reliably joins bounded English facts, but unconstrained
            # paraphrasing dropped complete category names and missing-input warnings.
            # The model still executes every returned token; no template substitutes
            # for failed inference. Raw SHAP/LIME numbers remain in the XAI layer.
            facts = generation_facts(evidence)
            supplied_data = {"facts": facts, "retrieved_background": retrieval["context"]}
            attempts = []
            for attempt in range(2):
                supplied = ("LOCAL REFERENCE NOTES:\n" + supplied_data["retrieved_background"] +
                            "\n\nFACT SENTENCES TO JOIN:\n" + "\n".join(supplied_data["facts"]))
                if supplied_data.get("correction_required"):
                    supplied += "\n\n" + supplied_data["correction_required"]
                if len(self._model.tokenize((system + supplied).encode("utf-8"))) > 1748:
                    raise ValueError("Evidence exceeds the local model's bounded context")
                result = self._model.create_chat_completion(messages=[{"role": "system", "content": system},
                                                                     {"role": "user", "content": supplied}],
                                                           max_tokens=200, temperature=0, top_p=1)
                text = result["choices"][0]["message"]["content"].strip()
                checked = check_grounding(text, evidence, retrieval["passages"], result["choices"][0].get("finish_reason"))
                attempts.append({"attempt": attempt + 1, **checked})
                if checked["passed"]:
                    break
                # Keep machine-readable failure codes in the audit record, not
                # the prompt: Qwen copied them as instructions and repeated them.
                supplied_data["correction_required"] = (
                    "The previous response did not preserve every fact. "
                    "Start again and copy every FACT sentence exactly, joining them with spaces. "
                    "Keep all category names, feature names, directions and limitations."
                )
            else:
                raise ValueError("Local explanation failed grounding checks after two attempts: " + ", ".join(checked["issues"]) + ". No substitute text was returned.")
            return {"text": text, "mode": "Actual local LLM inference", "llm_executed": True,
                    "model": MODEL_NAME, "runtime": "llama.cpp", "offline": True, "device": "cpu",
                    "elapsed_seconds": round(time.monotonic() - started, 2), "usage": result.get("usage", {}),
                    "cpu_threads": 2, "context_tokens": 2048, "release_after_generation": True,
                    "evidence": evidence,
                    "rag": {**retrieval, "used_in_generation": bool(retrieval["passages"])},
                    "correction": {"generation_attempts": attempts, "retried": len(attempts) > 1,
                                   "retrieval_corrected": retrieval.get("correction_attempted", False),
                                   "method": "Bounded local retrieval assessment and generation checks; not trained Self-RAG"},
                    "generation_facts": facts, "finish_reason": result["choices"][0].get("finish_reason"),
                    "generation_scope": "Actual small-model joining of evidence-derived English facts; not open-ended security reasoning.",
                    "limitations": "Generated interpretation may contain errors. Verify it against the measured evidence; it is not a detector or incident confirmation."}
        finally:
            try:
                if self._model is not None:
                    self._model.close()
                    self._model = None
            finally:
                self._lock.release()
