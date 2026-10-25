"""Fixed conventional RF loader and explicitly disclosed historical forest loader."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import (artifact_path, class_results, dependency_health, failure_health,
                     load_manifest, metadata_copy, rows_array, verify_artifacts, verify_source_record)


class RandomForestAdapter:
    """Never retrain or alter a fitted baseline while loading it for inference."""

    def __init__(self, config: dict[str, Any], root: Path):
        self._root = Path(root).resolve()
        self._manifest = load_manifest(config, self._root, "random_forest")
        self._model = None
        self._forest = None
        self._format = Path(self._manifest["checkpoint"]).suffix.lower()

    def health_check(self) -> dict[str, str]:
        dependencies = (("xgboost", "numpy") if self._format == ".ubj"
                        else ("sklearn", "joblib", "numpy"))
        health = dependency_health(dependencies)
        if health["status"] != "ready":
            return health
        try:
            if self._format not in {".ubj", ".joblib"}:
                raise ValueError("RF supports trusted conventional joblib or preserved standalone UBJ artifacts.")
            verify_artifacts(self._manifest, self._root)
            if self._manifest.get('source_record'):
                verify_source_record(self._manifest, self._root)
        except Exception as error:
            return failure_health(error)
        return {"status": "ready", "reason": "Fixed RF artifacts verified; CPU loading is lazy."}

    def load(self) -> None:
        if self._model is not None:
            return
        health = self.health_check()
        if health["status"] != "ready":
            if health["status"] == "unavailable":
                raise ImportError(health["reason"])
            raise ValueError(health["reason"])
        checkpoint = artifact_path(self._root, self._manifest["checkpoint"])
        if self._format == ".ubj":
            import xgboost as xgb
            model = xgb.Booster(model_file=str(checkpoint))
            model.set_param({"device": "cpu", "nthread": 2})
            config = json.loads(model.save_config())["learner"]
            parallel = int(config["gradient_booster"]["gbtree_model_param"]["num_parallel_tree"])
            if (model.num_boosted_rounds() != 1 or parallel < 2
                    or config["objective"]["name"] != "multi:softprob"):
                raise ValueError("Historical RF requires a standalone one-round parallel forest, not a boosted substitute.")
            if int(config["learner_model_param"]["num_class"]) != len(self._manifest["classes"]):
                raise ValueError("Historical RF checkpoint class count differs from the manifest.")
            requested_parallel = self._manifest["config"].get("num_parallel_tree")
            if requested_parallel is not None and parallel != int(requested_parallel):
                raise ValueError("Historical RF tree architecture differs from the manifest.")
            forest = model
            feature_count = model.num_features()
        else:
            import joblib
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.pipeline import Pipeline
            model = joblib.load(checkpoint)
            forest = model
            if isinstance(model, Pipeline):
                forest = model.steps[-1][1]
            if not isinstance(forest, RandomForestClassifier):
                raise ValueError("Conventional RF checkpoint must contain RandomForestClassifier.")
            if list(forest.classes_) != list(range(len(self._manifest["classes"]))):
                raise ValueError("RF checkpoint class indices differ from the manifest class map.")
            feature_count = getattr(model, "n_features_in_", None)
        if feature_count != len(self._manifest["feature_names"]):
            raise ValueError("RF checkpoint feature count differs from the manifest.")
        self._model, self._forest = model, forest

    def predict_proba(self, rows: list[list[float | None]]) -> list[list[float]]:
        values = rows_array(rows, self._manifest["feature_names"])
        self.load()
        if self._format == ".ubj":
            import xgboost as xgb
            probabilities = self._model.predict(xgb.DMatrix(values, nthread=2))
        else:
            probabilities = self._model.predict_proba(values)
        class_results(probabilities, self._manifest)
        return probabilities.tolist()

    def predict(self, rows: list[list[float | None]]) -> list[dict[str, Any]]:
        return class_results(self.predict_proba(rows), self._manifest)

    def metadata(self) -> dict[str, Any]:
        return metadata_copy(self._manifest, {"predict": True, "predict_proba": True,
                                             "partial_inputs": False, "unknown_detection": False,
                                             "explain": ["native", "global_importance"]})

    def explain(self, rows: list[list[float | None]], method: str = "global_importance") -> dict[str, Any]:
        rows_array(rows, self._manifest["feature_names"])
        if method not in {"native", "global_importance"}:
            raise NotImplementedError("RF supports global_importance; other XAI methods require independent plugins.")
        self.load()
        names = self._manifest["feature_names"]
        if self._format == ".ubj":
            scores = self._model.get_score(importance_type="gain")
            features = [{"feature": name, "importance": float(scores.get(f"f{index}", scores.get(name, 0.)))}
                        for index, name in enumerate(names)]
            description = "Historical XGBoost forest mean split gain"
        else:
            features = [{"feature": name, "importance": float(score)}
                        for name, score in zip(names, self._forest.feature_importances_)]
            description = "RF impurity-based training feature importance"
        features.sort(key=lambda item: item["importance"], reverse=True)
        return {"method": description, "scope": "global", "local_attribution": False,
                "features": features, "limitations": ["Global importance is not a per-observation SHAP/LIME attribution or causal evidence."]}
