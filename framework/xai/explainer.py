"""Heuristic classifier evidence and a geometric novelty explanation, not SHAP."""

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
import pandas as pd

from framework.data.loader import FEATURES, sample_frame, validate_features
from framework.detection.base import Prediction


class Explainer(Protocol):
    """Model-specific explainers share an output boundary with the pipeline."""

    def explain(self, sample: Mapping[str, float], detection: Prediction) -> dict:
        """Return evidence derived from the sample and fitted model."""


class FeatureExplainer:
    """Explain the demo model from train-only statistics and RF importance.

    Known: global RF importance weighted by deviation from a benign centroid.
    Unknown: each feature's share of squared distance to the nearest known row.
    Neither method is a local, causal attribution of the RF prediction.
    """

    def __init__(
        self, training_features: pd.DataFrame, training_labels: Sequence[str],
        feature_importances: Sequence[float],
    ) -> None:
        self.reference = validate_features(training_features).to_numpy()
        labels = np.asarray(training_labels)
        self.weights = np.asarray(feature_importances, dtype=float)
        if len(labels) != len(self.reference) or not np.any(labels == "BENIGN"):
            raise ValueError("Explanation needs matching training labels and benign reference rows.")
        if self.weights.shape != (len(FEATURES),) or not np.isfinite(self.weights).all() or (self.weights < 0).any():
            raise ValueError("Expected one finite, nonnegative model importance per feature.")
        self.benign_reference = self.reference[labels == "BENIGN"].mean(axis=0)
        self.scale = self.reference.std(axis=0)
        self.scale[self.scale == 0] = 1.0

    def explain(self, sample: Mapping[str, float], detection: Prediction) -> dict:
        values = sample_frame(sample).to_numpy()[0]
        if detection.novelty == "unknown":
            residuals = (values - self.reference) / self.scale
            nearest = int(np.argmin(np.sum(residuals**2, axis=1)))
            baseline = self.reference[nearest]
            deviations = residuals[nearest]
            raw = deviations**2
            method = "squared standardized distance shares (novelty)"
            reference = "nearest known training row"
        else:
            baseline = self.benign_reference
            deviations = (values - baseline) / self.scale
            raw = self.weights * np.abs(deviations)
            method = "global RF importance x absolute standardized benign deviation (heuristic)"
            reference = "benign training centroid"
        scores = raw / raw.sum() if raw.sum() > 0 else np.zeros_like(raw)
        ranked = np.argsort(-scores, kind="stable")[:3]
        return {
            "method": method,
            "reference": reference,
            "important_features": [
                {"feature": FEATURES[i], "value": float(values[i]),
                 "importance": float(scores[i]), "reference_value": float(baseline[i]),
                 "standardized_deviation": float(deviations[i])}
                for i in ranked
            ],
            "limitations": (
                "Simplified MVP explanation, not SHAP/LIME or causal evidence. "
                "Known-class rankings are not faithful local RF attributions. "
                "Novelty shares explain distance, not malicious intent. "
                "Only the top three of six feature scores are shown."
            ),
        }
