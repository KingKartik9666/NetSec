"""JSON-compatible model and optional-capability contracts."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

Row = list[float | None]


class CapabilityError(RuntimeError):
    """One operation failed; callers must not substitute a different model."""

    def __init__(self, model_id: str, capability: str, status: str, reason: str):
        self.model_id, self.capability, self.status, self.reason = model_id, capability, status, reason
        super().__init__(reason)

    def to_dict(self) -> dict:
        return {'model_id': self.model_id, 'capability': self.capability,
                'status': self.status, 'error': self.reason}


@runtime_checkable
class ModelAdapter(Protocol):
    def health_check(self) -> dict: ...
    def load(self) -> None: ...
    def predict(self, rows: list[Row]) -> list[dict]: ...
    def predict_proba(self, rows: list[Row]) -> list[list[float]]: ...
    def metadata(self) -> dict: ...


class ExplanationPlugin(Protocol):
    def explain(self, rows: list[Row], method: str = 'native') -> dict: ...


class FeatureTransform(Protocol):
    def fit(self, rows: list[Row], labels: list[int]) -> Any: ...
    def transform(self, rows: list[Row]) -> list[Row]: ...


class ProbabilityCalibration(Protocol):
    def fit(self, scores: list[list[float]], labels: list[int]) -> Any: ...
    def calibrate(self, scores: list[list[float]]) -> list[list[float]]: ...
