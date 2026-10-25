"""HEM adapter contracts using real upstream classes and tiny synthetic assets.

These fixtures verify integration, not intrusion-detection research accuracy.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def adapter_class():
    """Fail explicitly when the planned production capability is absent."""
    try:
        found = importlib.util.find_spec("framework.detection.adapters.hem")
    except ModuleNotFoundError:
        found = None
    assert found is not None, "HEM CPU adapter is not implemented"
    return importlib.import_module("framework.detection.adapters.hem").HEMModelAdapter


@pytest.fixture
def hem_assets(tmp_path):
    torch = pytest.importorskip("torch")
    joblib = pytest.importorskip("joblib")
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler

    upstream = tmp_path / "external/HEM/model_evaluation"
    upstream.mkdir(parents=True)
    for name in ("binary_classifier.py", "multiclass_classifier.py", "full_model.py"):
        shutil.copyfile(ROOT / "external/HEM/model_evaluation" / name, upstream / name)
    spec = importlib.util.spec_from_file_location("fixture_hem_binary", upstream / "binary_classifier.py")
    binary_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binary_module)
    spec = importlib.util.spec_from_file_location("fixture_hem_multi", upstream / "multiclass_classifier.py")
    multi_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(multi_module)
    binary = binary_module.BinaryClassifier(1, [1])
    multi = multi_module.ExpandableMultiClassClassifier(1, [1], 2)
    with torch.no_grad():
        for model in (binary, multi):
            for parameter in model.parameters():
                parameter.zero_()
        binary.model[-1].bias.fill_(1)
        multi.model[-1].bias[0] = 1
    preprocessing = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", MinMaxScaler(clip=True)),
        ("pca", PCA(n_components=1, svd_solver="full", random_state=42)),
    ]).fit(np.array([[0], [0], [2]], dtype=np.float32))
    destination = tmp_path / "artifacts/hem"
    destination.mkdir(parents=True)
    checkpoint = destination / "model.pt"
    torch.save({
        "candidate": "hem", "config": {"hidden": [1], "binary_hidden": [1], "knn_k": 1},
        "input_size": 1, "n_classes": 2, "benign_index": 0,
        "upstream_commit": "8513ad39c904dbe427727ee6f8470ee8a1ce8e0d",
        "state_dicts": {"binary": binary.state_dict(), "multiclass": multi.state_dict()},
    }, checkpoint)
    joblib.dump(preprocessing, destination / "preprocessing.joblib")
    np.savez(destination / "knn_references.npz",
             X=preprocessing.transform(np.array([[0], [0], [2]], dtype=np.float32)).astype(np.float32),
             y=np.array([1, 0, 0]), train_indices=np.array([0, 1, 2]))
    paths = [checkpoint, destination / "preprocessing.joblib", destination / "knn_references.npz",
             *(upstream / name for name in ("binary_classifier.py", "multiclass_classifier.py", "full_model.py"))]
    manifest = {
        "schema_version": 1, "model_id": "hem", "version": "synthetic-integration-fixture",
        "checkpoint": "artifacts/hem/model.pt", "feature_names": ["network_packets_all_count"],
        "classes": ["benign", "attack"], "benign_index": 0, "config": {"hidden": [1], "binary_hidden": [1], "knn_k": 1},
        "artifact_sha256": {p.relative_to(tmp_path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "upstream_commit": "8513ad39c904dbe427727ee6f8470ee8a1ce8e0d", "metrics": {},
        "protocol_id": "synthetic-adapter-contract-only", "input_profile": "full_numeric",
        "implementation": "HEM DataSense closed-set adaptation using original component classes",
        "limitations": ["Synthetic fixture; no research accuracy."],
    }
    manifest_path = tmp_path / "artifacts/hem/manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path, {"manifest": "artifacts/hem/manifest.json"}, manifest, manifest_path


def test_original_component_vote_and_earliest_duplicate_neighbor(hem_assets):
    root, config, _, _ = hem_assets
    adapter = adapter_class()(config, root)
    assert adapter.health_check()["status"] == "ready"
    predictions = adapter.predict([[0], [2]])
    assert [item["label"] for item in predictions] == ["attack", "benign"]
    assert [item["is_attack"] for item in predictions] == [True, False]
    assert all(item["score"] is None and item["top_classes"] == [] for item in predictions)
    assert predictions[0]["components"]["knn"]["neighbor_training_indices"] == [0]
    assert predictions[0]["components"]["binary"]["label"] == "attack"
    assert predictions[0]["components"]["multiclass"]["label"] == "benign"
    assert {str(parameter.device) for model in adapter._models.values() for parameter in model.parameters()} == {"cpu"}


@pytest.mark.parametrize("rows", [[[None]], [[float("nan")]], [[float("inf")]], [[-1]], [[1.5]], [[]], [[0, 1]]])
def test_incompatible_measurements_do_not_produce_hem_predictions(hem_assets, rows):
    root, config, _, _ = hem_assets
    with pytest.raises(ValueError):
        adapter_class()(config, root).predict(rows)


def test_component_diagnostics_are_not_ensemble_probability_or_shap(hem_assets):
    root, config, _, _ = hem_assets
    adapter = adapter_class()(config, root)
    explanation = adapter.explain([[0]])
    assert explanation["is_causal"] is False
    assert explanation["is_feature_attribution"] is False
    assert explanation["rows"][0]["nearest_neighbors"][0]["training_index"] == 0
    assert explanation["rows"][0]["nearest_neighbors"][0]["distance"] == pytest.approx(0)
    assert explanation["rows"][0]["feature_values"] == {"network_packets_all_count": 0.0}
    with pytest.raises(NotImplementedError):
        adapter.predict_proba([[0]])
    assert adapter.metadata()["capabilities"]["novelty"] is False


def test_corrupt_checkpoint_is_scoped_health_failure(hem_assets):
    root, config, _, _ = hem_assets
    (root / "artifacts/hem/model.pt").write_bytes(b"corrupt")
    adapter = adapter_class()(config, root)
    assert adapter.health_check()["status"] == "failed"
    with pytest.raises(ValueError):
        adapter.load()


def test_missing_original_source_marks_only_hem_unavailable(hem_assets):
    root, config, _, _ = hem_assets
    (root / "external/HEM/model_evaluation/binary_classifier.py").unlink()
    assert adapter_class()(config, root).health_check()["status"] == "unavailable"


def test_unpinned_source_cannot_be_loaded(hem_assets):
    root, config, manifest, path = hem_assets
    del manifest["artifact_sha256"]["external/HEM/model_evaluation/full_model.py"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source|digest|hash"):
        adapter_class()(config, root).load()


def test_checkpoint_class_map_mismatch_is_rejected(hem_assets):
    root, config, manifest, path = hem_assets
    manifest["classes"].append("unsupported-class")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="class"):
        adapter_class()(config, root).load()


def test_missing_torch_is_explicit_and_does_not_import_it(hem_assets, monkeypatch):
    root, config, _, _ = hem_assets
    module = importlib.import_module("framework.detection.adapters.hem")
    monkeypatch.setattr(module, "dependency_health", lambda names: {"status": "unavailable", "reason": "Missing torch"})
    adapter = module.HEMModelAdapter(config, root)
    assert adapter.health_check() == {"status": "unavailable", "reason": "Missing torch"}
    with pytest.raises(ModuleNotFoundError, match="torch"):
        adapter.load()


def test_multineighbor_ties_preserve_order_across_reference_chunks(hem_assets):
    root, config, manifest, path = hem_assets
    torch = pytest.importorskip("torch")
    import joblib

    checkpoint = root / manifest["checkpoint"]
    saved = torch.load(checkpoint, weights_only=True)
    saved["config"]["knn_k"] = 3
    torch.save(saved, checkpoint)
    manifest["config"]["knn_k"] = 3
    query = joblib.load(checkpoint.parent / "preprocessing.joblib").transform([[0]]).astype(np.float32)[0]
    references = np.full((8194, 1), 10, dtype=np.float32)
    references[[0, 8192, 8193]] = query
    labels = np.zeros(8194, dtype=np.int64)
    labels[[0, 8192]] = 1
    reference_path = checkpoint.parent / "knn_references.npz"
    np.savez(reference_path, X=references, y=labels, train_indices=np.arange(8194))
    for asset in (checkpoint, reference_path):
        manifest["artifact_sha256"][asset.relative_to(root).as_posix()] = hashlib.sha256(asset.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    prediction = adapter_class()(config, root).predict([[0]])[0]
    assert prediction["label"] == "attack"
    assert prediction["components"]["knn"]["neighbor_training_indices"] == [0, 8192, 8193]


def test_bad_training_reference_labels_rejected_even_with_matching_file_hash(hem_assets):
    root, config, manifest, path = hem_assets
    references = root / "artifacts/hem/knn_references.npz"
    with np.load(references) as saved:
        values, indices = saved["X"], saved["train_indices"]
    np.savez(references, X=values, y=np.array([99, 0, 0]), train_indices=indices)
    manifest["artifact_sha256"][references.relative_to(root).as_posix()] = hashlib.sha256(references.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="labels|class"):
        adapter_class()(config, root).load()


@pytest.mark.parametrize("multiclass_bias,observation", [([1.0, 0.0], 0), ([0.0, 1.0], 2)])
def test_binary_benign_with_one_benign_component_activates_original_gate(hem_assets, multiclass_bias, observation):
    root, config, manifest, path = hem_assets
    torch = pytest.importorskip("torch")
    checkpoint = root / manifest["checkpoint"]
    saved = torch.load(checkpoint, weights_only=True)
    saved["state_dicts"]["binary"]["model.2.bias"] = torch.tensor([-1.0])
    saved["state_dicts"]["multiclass"]["model.2.bias"] = torch.tensor(multiclass_bias)
    torch.save(saved, checkpoint)
    manifest["artifact_sha256"][manifest["checkpoint"]] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    prediction = adapter_class()(config, root).predict([[observation]])[0]
    assert prediction["label"] == "benign"
    assert prediction["is_attack"] is False


def pin_source_record(assets, identity_key="model_id"):
    root, config, manifest, path = assets
    source = root / "artifacts/hem/result.json"
    source.write_text(json.dumps({
        identity_key: "hem", "feature_names": ["network_packets_all_count"],
        "classes": ["benign", "attack"], "config": manifest["config"], "benign_index": 0,
        "upstream_commit": "8513ad39c904dbe427727ee6f8470ee8a1ce8e0d",
    }), encoding="utf-8")
    manifest["source_record"] = source.relative_to(root).as_posix()
    manifest["artifact_sha256"][manifest["source_record"]] = hashlib.sha256(source.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return source


def test_source_record_binds_ordered_class_names_beyond_tensor_dimensions(hem_assets):
    root, config, manifest, path = hem_assets
    pin_source_record(hem_assets)
    manifest["classes"] = ["attack", "benign"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="class"):
        adapter_class()(config, root).load()


def test_source_record_modification_is_rejected_before_loading_weights(hem_assets):
    root, config, _, _ = hem_assets
    source = pin_source_record(hem_assets)
    source.write_text(source.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        adapter_class()(config, root).load()


@pytest.mark.parametrize("identity_key", ["candidate", "model_id"])
def test_source_record_accepts_historical_and_controller_identity(hem_assets, identity_key):
    root, config, _, _ = hem_assets
    pin_source_record(hem_assets, identity_key)
    assert adapter_class()(config, root).predict([[0]])[0]["label"] == "attack"


def test_worker_explanation_method_argument_is_supported(hem_assets):
    root, config, _, _ = hem_assets
    adapter = adapter_class()(config, root)
    assert adapter.explain([[0]], "native")["rows"][0]["prediction"]["label"] == "attack"
    assert adapter.explain([[0]], "component_diagnostics")["is_feature_attribution"] is False
    with pytest.raises(NotImplementedError):
        adapter.explain([[0]], "shap")
