"""Small synthetic-data classifier plus distance novelty gate; this is NOT HEM."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from framework.data.loader import KNOWN_LABELS, sample_frame, validate_features
from framework.detection.base import BaseDetector, Prediction


class DemoDetector(BaseDetector):
    """CPU-only RF with a nearest-neighbor novelty threshold fitted on train rows."""

    def __init__(self, random_state: int = 42) -> None:
        self.classifier = RandomForestClassifier(
            n_estimators=80, max_depth=6, random_state=random_state, n_jobs=1,
        )
        self.scaler = StandardScaler()
        self.neighbors = NearestNeighbors(n_neighbors=1, n_jobs=1)

    def fit(self, X: pd.DataFrame, y: Sequence[str]) -> "DemoDetector":
        features = validate_features(X)
        labels = np.asarray(y)
        if len(features) != len(labels) or set(labels) != set(KNOWN_LABELS):
            raise ValueError("fit requires matching rows labeled BENIGN, DDoS and PORT_SCAN.")
        self.classifier.fit(features, labels)
        scaled = self.scaler.fit_transform(features)
        self.neighbors.fit(scaled)
        # Query with explicit training rows: neighbor 0 is self, neighbor 1 is
        # the leave-one-out nearest row. No demo data is used for calibration.
        distances = self.neighbors.kneighbors(scaled, n_neighbors=2)[0][:, 1]
        self.novelty_threshold_ = max(0.25, float(np.quantile(distances, 0.95) * 1.5))
        return self

    def predict(self, sample: Mapping[str, float]) -> Prediction:
        if not hasattr(self, "novelty_threshold_"):
            raise ValueError("DemoDetector must be fitted before prediction.")
        features = sample_frame(sample)
        probabilities = self.classifier.predict_proba(features)[0]
        winner = int(np.argmax(probabilities))
        label = str(self.classifier.classes_[winner])
        score = float(probabilities[winner])
        distance = float(self.neighbors.kneighbors(self.scaler.transform(features))[0][0, 0])
        unknown = distance > self.novelty_threshold_
        return Prediction(
            is_attack=unknown or label != "BENIGN",
            attack_type="UNKNOWN" if unknown else label,
            novelty="unknown" if unknown else "known",
            confidence=None if unknown else score,
            classifier_label=label,
            classifier_confidence=score,
            novelty_score=distance,
            novelty_threshold=self.novelty_threshold_,
        )
