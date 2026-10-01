"""Keep detector decisions separate from explanation and report generation."""

from dataclasses import asdict
from typing import Any, Mapping

from framework.detection.base import BaseDetector
from framework.llm.prompt_builder import structured_evidence
from framework.llm.provider import LLMProvider
from framework.xai.explainer import Explainer


class IoTSecurityPipeline:
    """Orchestrate interchangeable components without exposing sample metadata."""

    def __init__(self, detector: BaseDetector, explainer: Explainer, llm_provider: LLMProvider) -> None:
        self.detector = detector
        self.explainer = explainer
        self.llm_provider = llm_provider

    def analyze(self, sample: Mapping[str, float]) -> dict[str, Any]:
        """Return JSON-serializable evidence and its generated security report."""
        prediction = self.detector.predict(sample)
        explanation = self.explainer.explain(sample, prediction)
        context = structured_evidence({"detection": asdict(prediction), "explanation": explanation})
        return {
            "detection": context["detection"],
            "explanation": context["explanation"],
            "context": context,
            "report": self.llm_provider.generate(context),
        }
