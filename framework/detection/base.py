"""Normalized detection contract shared by demo and future research adapters."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class Prediction:
    """Confidence is a closed-set model score, never a novelty probability.

    For UNKNOWN, confidence is None and is_attack means 'raise a suspicious
    traffic alert', not proof that the anomaly is malicious.
    """

    is_attack: bool
    attack_type: str
    novelty: Literal["known", "unknown"]
    confidence: float | None
    classifier_label: str
    classifier_confidence: float
    novelty_score: float
    novelty_threshold: float
    detector_name: str = "DemoDetector"


class BaseDetector(ABC):
    """Fit on feature tables and return evidence for one feature-only sample."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Sequence[str]) -> "BaseDetector":
        """Train using known labeled features."""

    @abstractmethod
    def predict(self, sample: Mapping[str, float]) -> Prediction:
        """Classify and assess novelty without consulting ground-truth labels."""
