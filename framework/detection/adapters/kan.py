"""CPU adapter for the project's authentic upstream efficient-kan checkpoint."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from .common import (artifact_path, class_results, dependency_health, failure_health,
                     load_manifest, metadata_copy, rows_array, verify_artifacts, verify_source_record)


SOURCE = "external/efficient-kan/src/efficient_kan/kan.py"
UPSTREAM_COMMIT = "7b6ce1c87f18c8bc90c208f6b494042344216b11"


class KANAdapter:
    """Load preserved B-spline weights lazily; never use a substitute network."""

    def __init__(self, config: dict[str, Any], root: Path):
        self._root = Path(root).resolve()
        self._manifest = load_manifest(config, self._root, "kan")
        self._model = None
        self._preprocessing = None

    def health_check(self) -> dict[str, str]:
        health = dependency_health(("torch", "joblib", "sklearn", "numpy"))
        if health["status"] != "ready":
            return health
        try:
            if SOURCE not in self._manifest["artifact_sha256"]:
                raise ValueError("KAN manifest must pin the genuine upstream source SHA256.")
            if self._manifest.get("upstream_commit") != UPSTREAM_COMMIT:
                raise ValueError("KAN manifest identifies an unsupported upstream commit.")
            verify_artifacts(self._manifest, self._root)
            verify_source_record(self._manifest, self._root)
        except Exception as error:
            return failure_health(error)
        return {"status": "ready", "reason": "Genuine KAN artifacts verified; CPU loading is lazy."}

    def load(self) -> None:
        if self._model is not None:
            return
        health = self.health_check()
        if health["status"] != "ready":
            if health["status"] == "unavailable":
                raise ImportError(health["reason"])
            raise ValueError(health["reason"])
        import joblib
        import torch
        torch.set_num_threads(2)
        checkpoint = torch.load(artifact_path(self._root, self._manifest["checkpoint"]),
                                map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("candidate") != "kan":
            raise ValueError("Checkpoint is not a genuine KAN candidate.")
        if checkpoint.get("input_size") != len(self._manifest["feature_names"]):
            raise ValueError("KAN checkpoint feature count differs from the manifest.")
        if checkpoint.get("n_classes") != len(self._manifest["classes"]):
            raise ValueError("KAN checkpoint class count differs from the manifest.")
        if checkpoint.get("benign_index") != self._manifest["benign_index"]:
            raise ValueError("KAN checkpoint benign class differs from the manifest.")
        if checkpoint.get("upstream_commit") != self._manifest["upstream_commit"]:
            raise ValueError("KAN checkpoint upstream source differs from the manifest.")
        config = checkpoint.get("config", {})
        if config != self._manifest["config"]:
            raise ValueError("KAN saved architecture configuration differs from the manifest.")
        spec = importlib.util.spec_from_file_location("_imdai_optional_genuine_kan", artifact_path(self._root, SOURCE))
        if spec is None or spec.loader is None:
            raise ImportError("Cannot import the verified upstream efficient-kan source.")
        source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source)
        model = source.KAN(
            [checkpoint["input_size"], *config.get("hidden", [64, 32]), checkpoint["n_classes"]],
            grid_size=int(config.get("grid_size", 5)), spline_order=int(config.get("spline_order", 3)),
        ).to("cpu")
        model.load_state_dict(checkpoint["state_dicts"]["kan"], strict=True)
        model.eval()
        preprocessing_path = self._manifest.get("preprocessing")
        if not preprocessing_path:
            raise ValueError("KAN requires the saved train-only preprocessing artifact.")
        preprocessing = joblib.load(artifact_path(self._root, preprocessing_path))
        if getattr(preprocessing, "n_features_in_", None) != len(self._manifest["feature_names"]):
            raise ValueError("KAN preprocessing feature count differs from the manifest.")
        self._preprocessing, self._model = preprocessing, model

    def predict_proba(self, rows: list[list[float | None]]) -> list[list[float]]:
        values = rows_array(rows, self._manifest["feature_names"])
        self.load()
        import numpy as np
        import torch
        transformed = self._preprocessing.transform(values).astype(np.float32)
        with torch.no_grad():
            probabilities = torch.softmax(self._model(torch.as_tensor(transformed, device="cpu")), dim=1).numpy()
        class_results(probabilities, self._manifest)  # Reject invalid output before returning scores.
        return probabilities.tolist()

    def predict(self, rows: list[list[float | None]]) -> list[dict[str, Any]]:
        return class_results(self.predict_proba(rows), self._manifest)

    def metadata(self) -> dict[str, Any]:
        return metadata_copy(self._manifest, {"predict": True, "predict_proba": True,
                                             "partial_inputs": False, "unknown_detection": False,
                                             "explain": ["native", "gradient", "gradient_input_sensitivity"]})

    def explain(self, rows: list[list[float | None]], method: str = "native") -> dict[str, Any]:
        """Explain one local logit derivative, without claiming SHAP or causality.

Autograd differentiates the selected class's raw logit with respect to the
saved imputed/scaled input, including original spline derivatives. It does not
differentiate the clipping transform. Gradient times scaled input is a signed
heuristic ranking, not additive reconstruction of the class margin.
"""
        if method not in {"native", "gradient", "gradient_input_sensitivity"}:
            raise NotImplementedError("KAN native XAI supports gradient_input_sensitivity; other methods need optional plugins.")
        if not isinstance(rows, list) or len(rows) != 1:
            raise ValueError("KAN gradient explanation accepts exactly one observation.")
        values = rows_array(rows, self._manifest["feature_names"])
        self.load()
        import numpy as np
        import torch
        transformed = self._preprocessing.transform(values).astype(np.float32)
        query = torch.tensor(transformed, device="cpu", requires_grad=True)
        logits = self._model(query)
        if not torch.isfinite(logits).all():
            raise ValueError("KAN explanation received nonfinite model logits.")
        selected = int(logits[0].detach().argmax())
        derivatives = torch.autograd.grad(logits[0, selected], query, create_graph=False)[0].detach().numpy()[0]
        if not np.isfinite(derivatives).all():
            raise ValueError("KAN explanation produced nonfinite gradients.")
        features = [{"feature": name, "value": float(values[0, index]),
                     "scaled_value": float(transformed[0, index]), "gradient": float(derivatives[index]),
                     "importance": float(derivatives[index] * transformed[0, index]),
                     "basis": "raw class-logit derivative times saved scaled input"}
                    for index, name in enumerate(self._manifest["feature_names"])]
        features.sort(key=lambda item: abs(item["importance"]), reverse=True)
        return {
            "method": "gradient_input_sensitivity", "scope": "local", "features": features,
            "predicted_class": self._manifest["classes"][selected],
            "class_logit": float(logits[0, selected].detach()),
            "is_causal": False, "is_feature_attribution": False,
            "units": "raw class-logit change per unit of scaled input",
            "preprocessing": "saved train-only imputation and clipped MinMax scaling; gradients are in scaled coordinates",
            "model_id": "kan", "model_version": self._manifest["version"],
            "limitations": [
                "This is original-network local sensitivity, not SHAP/LIME or causal evidence.",
                "Signed gradient times scaled input is a heuristic, not a complete additive class contribution.",
                "Clipping is not differentiated; a scaled-coordinate derivative does not describe out-of-range raw changes.",
                "Spline derivatives and local sensitivity can vary across observations; scores do not confirm an incident.",
            ],
        }
