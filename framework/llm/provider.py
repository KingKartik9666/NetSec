"""Provider boundary and deterministic reasoning over supplied evidence."""

import os
from abc import ABC, abstractmethod
from typing import Any, Mapping

from .prompt_builder import structured_evidence


class LLMProvider(ABC):
    """Translate evidence into a report without changing detector decisions."""

    name: str = "LLMProvider"

    @abstractmethod
    def generate(self, context: Mapping[str, Any]) -> str:
        """Return an analyst-readable report based on structured evidence."""


class TemplateProvider(LLMProvider):
    """Offline formatter; does not execute a language model or network request."""

    name = "TemplateProvider"

    def generate(self, context: Mapping[str, Any]) -> str:
        evidence = structured_evidence(context)
        detection, explanation = evidence["detection"], evidence["explanation"]
        label = detection["attack_type"]
        unknown = detection["novelty"] == "unknown"
        if unknown:
            assessment = (
                "UNKNOWN / anomalous behavior flagged for review. This is outside "
                "the known training distribution, not a confirmed new malicious attack."
            )
            interpretation = (
                "Possible interpretations include unfamiliar legitimate behavior or an "
                "attack absent from training; the supplied evidence cannot distinguish them."
            )
            actions = [
                "Review the anomalous feature values and collect supporting traffic logs.",
                "Compare with the device's expected behavior before assigning an attack label.",
                "Consider containment only if independent evidence confirms malicious activity.",
            ]
        elif detection["is_attack"]:
            assessment = f"The demonstration classifier predicts suspicious behavior: {label}."
            interpretations = {
                "DDoS": "If the DDoS prediction is correct, the pattern may represent traffic flooding.",
                "PORT_SCAN": "If the PORT_SCAN prediction is correct, the pattern may represent probing activity.",
            }
            interpretation = interpretations.get(label, "The predicted attack label requires analyst verification.")
            actions = [
                "Inspect the underlying traffic and preserve relevant logs for verification.",
                "If malicious flooding is confirmed, consider scoped rate limits."
                if label == "DDoS" else "If malicious activity is confirmed, review scoped access restrictions.",
                "Consider device isolation only after confirming the source and operational impact.",
            ]
        else:
            assessment = f"No attack flagged by the demonstration classifier; predicted class: {label}."
            interpretation = "The sample resembles known benign behavior; this does not establish that a device is safe."
            actions = ["Continue monitoring and review changes from the expected traffic baseline."]

        confidence = detection["confidence"]
        confidence_text = (
            "Unavailable for UNKNOWN; no calibrated unknown-attack probability is provided."
            if confidence is None else f"{confidence:.1%} (uncalibrated classifier score)."
        )
        lines = [
            assessment,
            f"Novelty: {detection['novelty'].upper()}; nearest-known distance "
            f"{detection['novelty_score']:.3f}, threshold {detection['novelty_threshold']:.3f}.",
            f"Confidence: {confidence_text}",
            f"Closed-set classifier: {detection['classifier_label']} "
            f"({detection['classifier_confidence']:.1%}; uncalibrated, not confidence in UNKNOWN).",
            "", "Feature evidence (simplified ranking):",
        ]
        for feature in explanation["important_features"]:
            lines.append(
                f"- {feature['feature']} = {feature['value']:.4g}; "
                f"reference {feature['reference_value']:.4g}; "
                f"standardized deviation {feature['standardized_deviation']:+.3f}; "
                f"relative importance {feature['importance']:.3f}."
            )
        lines.extend([
            f"Method: {explanation['method']}. Reference: {explanation['reference']}.",
            "", interpretation, "", "Proposed defensive actions:",
            *[f"{index}. {action}" for index, action in enumerate(actions, start=1)],
            "", "Uncertainty: synthetic demonstration data and a prototype detector; "
            "these outputs do not establish research accuracy or confirm an incident.",
            "The simplified XAI ranking is not SHAP and is not a causal explanation.",
            explanation["limitations"],
            "Reasoning mode: deterministic offline TemplateProvider; no language model is executed.",
        ])
        return "\n".join(lines)


def choose_provider(mode: str | None = None) -> LLMProvider:
    """Select the implemented provider; fail clearly for unsupported modes."""
    selected = (mode if mode is not None else os.getenv("LLM_MODE", "template")).strip().lower()
    if selected != "template":
        raise ValueError(f"Unsupported LLM_MODE {selected!r}. This MVP supports only 'template'.")
    return TemplateProvider()
