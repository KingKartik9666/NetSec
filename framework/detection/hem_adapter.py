"""Inspect upstream HEM prerequisites without importing or impersonating HEM."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from framework.data.loader import PROJECT_ROOT
from framework.detection.base import BaseDetector, Prediction


class HEMAdapter(BaseDetector):
    """Future inference boundary; artifact presence alone does not enable HEM.

    The inventory follows the upstream all-known PCA training configuration.
    A real integration additionally needs a verified preprocessing schema,
    label map, training-row alignment, alpha/thresholds and inference bridge.
    Never deserialize untrusted .pt/joblib files just to check availability.
    """

    def __init__(self, hem_root: Path | str | None = None, dataset: str = "uq_iot") -> None:
        self.hem_root = Path(hem_root) if hem_root is not None else PROJECT_ROOT / "external" / "HEM"
        if dataset not in {"uq_iot", "ku_iot"}:
            raise ValueError("HEM dataset must be uq_iot or ku_iot.")
        binary, multi = (
            ("1024-256-64-16", "512-256-32-16") if dataset == "uq_iot"
            else ("512-128-64", "512-256-128-64")
        )
        model_dir = Path("model_evaluation") / f"{dataset}_pca"
        preprocessing = Path("feature_extraction") / dataset / "preprocessing_models"
        self.required_artifacts = (
            model_dir / "binary_classifier" / f"layer{binary}_all.pt",
            model_dir / "multi_classifier" / f"layer{multi}_all.pt",
            model_dir / "knn" / "all.joblib",
            preprocessing / "scaler_all.joblib",
            preprocessing / "pca_all.joblib",
        )

    def available(self) -> bool:
        """False until the real inference bridge has been implemented and verified."""
        return False

    def status(self) -> dict:
        """Report the actual source/artifact inventory; never execute external code."""
        found = [str(path) for path in self.required_artifacts if (self.hem_root / path).is_file()]
        missing = [str(path) for path in self.required_artifacts if not (self.hem_root / path).is_file()]
        return {
            "available": self.available(),
            "source_present": (self.hem_root / "model_evaluation" / "full_model.py").is_file(),
            "found_artifacts": found,
            "missing_artifacts": missing,
            "reason": (
                "Real HEM inference is unavailable: inference bridge not implemented. "
                f"{len(missing)} of {len(self.required_artifacts)} expected artifacts missing. "
                "See docs/HEM_INTEGRATION.md for data, preprocessing and validation requirements."
            ),
        }

    def fit(self, X: pd.DataFrame, y: Sequence[str]) -> "HEMAdapter":
        raise RuntimeError(self.status()["reason"])

    def predict(self, sample: Mapping[str, float]) -> Prediction:
        raise RuntimeError(self.status()["reason"])
