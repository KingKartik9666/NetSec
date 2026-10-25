"""CPU adapter for preserved HEM component checkpoints, never a substitute.

The existing DataSense adaptation uses the original upstream neural classes,
training-fitted preprocessing and a deterministic closed-set KNN vote. Native
Kitsune preprocessing and unknown-attack reproduction remain separate abilities.
Only standard-library modules and the lightweight adapter helpers load at boot.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from .common import artifact_path, dependency_health, load_manifest, rows_array, verify_artifacts, verify_source_record


HEM_COMMIT = "8513ad39c904dbe427727ee6f8470ee8a1ce8e0d"
SOURCE_DIRECTORY = "external/HEM/model_evaluation"
SOURCE_FILES = ("binary_classifier.py", "multiclass_classifier.py", "full_model.py")


class HEMModelAdapter:
    """Load genuine HEM components on CPU with explicit capability limits.

Each adapter belongs to its isolated model worker. It accepts only complete
numeric observations in the persisted feature order. It does not infer native
Kitsune features, probabilities for the ensemble, or unknown-attack thresholds.
"""

    def __init__(self, config: dict[str, Any], root: Path) -> None:
        self.config = dict(config)
        self.root = Path(root).resolve()
        self._manifest: dict[str, Any] | None = None
        self._models: dict[str, Any] = {}
        self._preprocessing: Any = None
        self._references: Any = None
        self._reference_labels: Any = None
        self._training_indices: Any = None
        self._torch: Any = None
        self._k = 0

    def _validated_manifest(self) -> dict[str, Any]:
        manifest = load_manifest(self.config, self.root, "hem")
        if manifest.get("upstream_commit") != HEM_COMMIT:
            raise ValueError("HEM upstream commit differs from the audited original implementation")
        if manifest.get("input_profile") != "full_numeric":
            raise ValueError("HEM requires the evaluated full_numeric profile; partial/live inputs are incompatible")
        pinned = manifest.get("artifact_sha256", {})
        for filename in SOURCE_FILES:
            source = f"{SOURCE_DIRECTORY}/{filename}"
            if source not in pinned:
                raise ValueError(f"HEM original source digest is missing: {source}")
        checkpoint = Path(manifest["checkpoint"])
        preprocessing = manifest.get("preprocessing", (checkpoint.parent / "preprocessing.joblib").as_posix())
        references = manifest.get("knn_references", (checkpoint.parent / "knn_references.npz").as_posix())
        for path in (manifest["checkpoint"], preprocessing, references):
            if path not in pinned:
                raise ValueError(f"HEM required artifact digest is missing: {path}")
        manifest = dict(manifest, preprocessing=preprocessing, knn_references=references)
        verify_artifacts(manifest, self.root)
        if manifest.get("source_record") is not None:
            verify_source_record(manifest, self.root)
        return manifest

    def health_check(self) -> dict[str, str]:
        """Check prerequisites without importing PyTorch or deserializing assets."""
        dependency = dependency_health(("torch", "numpy", "joblib", "sklearn"))
        if dependency["status"] != "ready":
            return dependency
        try:
            self._validated_manifest()
        except (FileNotFoundError, ModuleNotFoundError) as exc:
            return {"status": "unavailable", "reason": str(exc)}
        except Exception as exc:
            return {"status": "failed", "reason": f"HEM artifact validation failed: {exc}"}
        return {"status": "ready", "reason": "Original HEM component assets verified; closed-set numeric adaptation only"}

    def _source_class(self, filename: str, name: str) -> Any:
        path = artifact_path(self.root, f"{SOURCE_DIRECTORY}/{filename}")
        spec = importlib.util.spec_from_file_location(f"_imdai_cpu_hem_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import original HEM component: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, name)

    def load(self) -> None:
        """Validate every owned asset before loading trusted saved components."""
        dependency = dependency_health(("torch", "numpy", "joblib", "sklearn"))
        if dependency["status"] != "ready":
            raise ModuleNotFoundError(dependency["reason"])
        manifest = self._validated_manifest()
        import joblib
        import numpy as np
        import torch

        torch.set_num_threads(2)
        checkpoint = torch.load(artifact_path(self.root, manifest["checkpoint"]), map_location="cpu", weights_only=True)
        if checkpoint.get("candidate") != "hem" or checkpoint.get("upstream_commit") != HEM_COMMIT:
            raise ValueError("Checkpoint is not an audited original-component HEM adaptation")
        if (checkpoint.get("n_classes") != len(manifest["classes"])
                or checkpoint.get("benign_index") != manifest["benign_index"]):
            raise ValueError("HEM checkpoint and manifest class map disagree")
        if checkpoint.get("config") != manifest.get("config"):
            raise ValueError("HEM checkpoint and manifest architecture/configuration disagree")
        dimensions = checkpoint.get("input_size")
        if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions < 1:
            raise ValueError("HEM checkpoint has an invalid component input size")
        preprocessing = joblib.load(artifact_path(self.root, manifest["preprocessing"]))
        if getattr(preprocessing, "n_features_in_", None) != len(manifest["feature_names"]):
            raise ValueError("HEM preprocessing and ordered feature schema disagree")
        with np.load(artifact_path(self.root, manifest["knn_references"]), allow_pickle=False) as saved:
            if set(saved.files) != {"X", "y", "train_indices"}:
                raise ValueError("HEM KNN reference bundle lacks its ordered training provenance")
            references = np.asarray(saved["X"], dtype=np.float32)
            labels = np.asarray(saved["y"])
            indices = np.asarray(saved["train_indices"])
        if (references.ndim != 2 or references.shape[1] != dimensions or not len(references)
                or not np.isfinite(references).all()):
            raise ValueError("HEM KNN reference measurements are invalid or incompatible")
        if (labels.shape != (len(references),) or not np.issubdtype(labels.dtype, np.integer)
                or labels.min() < 0 or labels.max() >= len(manifest["classes"])):
            raise ValueError("HEM KNN reference labels do not match the class map")
        if (indices.shape != labels.shape or not np.issubdtype(indices.dtype, np.integer)
                or indices.min() < 0 or np.any(np.diff(indices) <= 0)):
            raise ValueError("HEM KNN training indices must preserve unique original fit order")
        if "training_rows" in manifest and indices.max() >= manifest["training_rows"]:
            raise ValueError("HEM KNN references extend beyond the training partition")
        k = checkpoint["config"].get("knn_k", 3)
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= len(references):
            raise ValueError("HEM KNN neighbor count is invalid")
        Binary = self._source_class("binary_classifier.py", "BinaryClassifier")
        Multiclass = self._source_class("multiclass_classifier.py", "ExpandableMultiClassClassifier")
        model_config = checkpoint["config"]
        models = {
            "binary": Binary(dimensions, model_config["binary_hidden"]),
            "multiclass": Multiclass(dimensions, model_config["hidden"], len(manifest["classes"])),
        }
        if set(checkpoint.get("state_dicts", {})) != set(models):
            raise ValueError("HEM checkpoint must contain both original neural components")
        for name, model in models.items():
            model.load_state_dict(checkpoint["state_dicts"][name], strict=True)
            model.to("cpu").eval()
            if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise ValueError(f"HEM {name} checkpoint contains nonfinite weights")
        probe = preprocessing.transform(np.zeros((1, len(manifest["feature_names"])), dtype=np.float32))
        if probe.shape != (1, dimensions) or not np.isfinite(probe).all():
            raise ValueError("HEM preprocessing output differs from the saved component input contract")
        self._manifest, self._models, self._preprocessing = manifest, models, preprocessing
        self._references = torch.from_numpy(np.ascontiguousarray(references))
        self._reference_labels, self._training_indices = labels.astype(np.int64), indices.astype(np.int64)
        self._torch, self._k = torch, k

    def metadata(self) -> dict[str, Any]:
        """Return model identity and limits without importing model dependencies."""
        manifest = self._manifest or load_manifest(self.config, self.root, "hem")
        return {
            **manifest,
            "name": "HEM", "device": "cpu", "adaptation": "DataSense closed-set numeric adaptation",
            "capabilities": {"predict": True, "predict_proba": False, "explain": True,
                             "partial_inputs": False, "live": False, "novelty": False,
                             "native_hem_preprocessing": False},
            "probability_semantics": "The voting ensemble has no supported probability; component scores are uncalibrated",
            "native_hem_status": "Unavailable: native Kitsune preprocessing/checkpoints and evaluated novelty thresholds are separate",
        }

    def _input(self, rows: list[list[float | None]]) -> Any:
        if self._manifest is None:
            self.load()
        import numpy as np

        values = rows_array(rows, self._manifest["feature_names"], allow_missing=False)
        names = self._manifest["feature_names"]
        for index, name in enumerate(names):
            if name.endswith("_count"):
                count = values[:, index]
                if np.any(count < 0) or np.any(count > 2 ** 24) or np.any(count != np.floor(count)):
                    raise ValueError(f"HEM measured count {name} must be a nonnegative exactly represented integer")
                if name.startswith("network_tcp-flags-") and "network_packets_all_count" in names:
                    if np.any(count > values[:, names.index("network_packets_all_count")]):
                        raise ValueError(f"HEM measured {name} exceeds the total packet count")
        transformed = self._preprocessing.transform(values).astype(np.float32)
        if not np.isfinite(transformed).all():
            raise ValueError("HEM preprocessing produced nonfinite measurements")
        return values, self._torch.from_numpy(np.ascontiguousarray(transformed))

    def _neighbors(self, queries: Any) -> tuple[Any, Any]:
        """Exact CPU Euclidean neighbors with stable original-reference ties.

Reference chunks bound temporary memory. Sorting retains earlier reference
indices when distances tie, matching the preserved CUDA training helper.
"""
        torch = self._torch
        distances_out, indices_out = [], []
        for start in range(0, len(queries), 16):
            query = queries[start:start + 16]
            best_distances = torch.empty((len(query), 0), dtype=torch.float32)
            best_indices = torch.empty((len(query), 0), dtype=torch.int64)
            for offset in range(0, len(self._references), 8192):
                references = self._references[offset:offset + 8192]
                distances = torch.cdist(query, references, compute_mode="donot_use_mm_for_euclid_dist")
                if self._k == 1:
                    local_distances, local_indices = distances.min(dim=1, keepdim=True)
                    local_indices = local_indices + offset
                    if best_distances.shape[1] == 0:
                        best_distances, best_indices = local_distances, local_indices
                    else:
                        closer = local_distances < best_distances
                        best_distances = torch.where(closer, local_distances, best_distances)
                        best_indices = torch.where(closer, local_indices, best_indices)
                    continue
                indices = torch.arange(offset, offset + len(references))[None, :].expand(len(query), -1)
                combined_distances = torch.cat((best_distances, distances), dim=1)
                combined_indices = torch.cat((best_indices, indices), dim=1)
                order = torch.argsort(combined_distances, dim=1, stable=True)[:, :self._k]
                best_distances = torch.gather(combined_distances, 1, order)
                best_indices = torch.gather(combined_indices, 1, order)
            distances_out.append(best_distances)
            indices_out.append(best_indices)
        return torch.cat(distances_out).numpy(), torch.cat(indices_out).numpy()

    def _infer(self, rows: list[list[float | None]]) -> tuple[list[dict[str, Any]], Any, Any, Any, Any]:
        values, query = self._input(rows)
        import numpy as np

        torch, manifest = self._torch, self._manifest
        classes, benign = manifest["classes"], manifest["benign_index"]
        with torch.no_grad():
            binary_logits = self._models["binary"](query)[:, 0]
            multiclass_logits = self._models["multiclass"](query)
            binary = (binary_logits >= 0).long().numpy()
            multiclass = multiclass_logits.argmax(1).numpy()
            binary_scores = torch.sigmoid(binary_logits).numpy()
            multiclass_scores = torch.softmax(multiclass_logits, dim=1).numpy()
            distances, neighbors = self._neighbors(query)
        if not np.isfinite(binary_scores).all() or not np.isfinite(multiclass_scores).all() or not np.isfinite(distances).all():
            raise ValueError("HEM component inference produced nonfinite outputs")
        neighbor_labels = self._reference_labels[neighbors]
        knn = np.array([np.bincount(row, minlength=len(classes)).argmax() for row in neighbor_labels])
        benign_gate = (binary == 0) & ((multiclass == benign) | (knn == benign))
        final = np.where(benign_gate, benign, np.where(knn != benign, knn, multiclass))
        results = []
        for row, label_index in enumerate(final):
            ordered = np.argsort(-multiclass_scores[row], kind="stable")[:3]
            results.append({
                "label": classes[int(label_index)], "is_attack": int(label_index) != benign,
                "score": None, "novelty": "not evaluated", "top_classes": [],
                "model_id": "hem", "model_version": manifest["version"], "input_profile": "full_numeric",
                "components": {
                    "binary": {"label": "attack" if binary[row] else "benign", "attack_score": float(binary_scores[row]), "calibrated": False},
                    "multiclass": {"label": classes[int(multiclass[row])],
                                   "top_classes": [{"label": classes[int(i)], "score": float(multiclass_scores[row, i])} for i in ordered],
                                   "calibrated": False, "scope": "Multiclass component only; not the ensemble probability"},
                    "knn": {"label": classes[int(knn[row])], "k": self._k,
                            "mean_distance": float(distances[row].mean()),
                            "neighbor_training_indices": self._training_indices[neighbors[row]].tolist()},
                },
            })
        return results, values, query.numpy(), distances, neighbors

    def predict(self, rows: list[list[float | None]]) -> list[dict[str, Any]]:
        """Return genuine closed-set ensemble decisions without requiring XAI."""
        return self._infer(rows)[0]

    def predict_proba(self, rows: list[list[float | None]]) -> list[list[float]]:
        raise NotImplementedError("Original HEM voting has no supported ensemble probability; inspect component scores separately")

    def explain(self, rows: list[list[float | None]], method: str = "native") -> dict[str, Any]:
        """Expose component/PCA/neighbor diagnostics, not fabricated attribution."""
        if method not in {"native", "component_diagnostics"}:
            raise NotImplementedError(f"HEM explanation method {method!r} is unsupported; use component_diagnostics")
        results, values, coordinates, distances, neighbors = self._infer(rows)
        classes = self._manifest["classes"]
        diagnostics = []
        for row, prediction in enumerate(results):
            diagnostics.append({
                "prediction": prediction,
                "feature_values": dict(zip(self._manifest["feature_names"], values[row].astype(float).tolist())),
                "pca_coordinates": coordinates[row].astype(float).tolist(),
                "nearest_neighbors": [{"training_index": int(self._training_indices[index]),
                                       "label": classes[int(self._reference_labels[index])],
                                       "distance": float(distances[row, position])}
                                      for position, index in enumerate(neighbors[row])],
            })
        return {"method": "HEM component, PCA and nearest-training-neighbor diagnostics",
                "is_causal": False, "is_feature_attribution": False, "rows": diagnostics,
                "limitations": ["These diagnostics are not SHAP/LIME or causal feature attribution.",
                                "PCA coordinates combine scaled measurements; they are not raw-feature contributions.",
                                "Neighbor proximity and component votes do not confirm an incident or detect an unknown attack.",
                                "Component scores are uncalibrated and are not ensemble probabilities."],
                "model_id": "hem", "model_version": self._manifest["version"]}
