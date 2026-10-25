"""Real small checkpoints and injected failures for optional model adapters."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _manifest(root: Path, model_id: str, checkpoint: str, *, artifacts=None, **extra):
    files = [checkpoint, *(artifacts or [])]
    manifest = {
        "model_id": model_id, "version": "fixture-v1", "checkpoint": checkpoint,
        "artifact_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files},
        "feature_names": ["first", "second", "third"],
        "classes": ["benign", "attack-a", "attack-b"], "benign_index": 0,
        "config": {}, "metrics": None, "protocol_id": "fixture-only",
        "input_profile": "full_numeric", "implementation": "fixture", "limitations": [],
        **extra,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {"manifest": "manifest.json"}, manifest


def test_missing_adapter_dependencies_do_not_import_torch_or_xgboost_on_boot():
    # Eager dependency imports would make this fail on a missing optional package.
    script = (
        "import sys; import framework.detection.adapters.kan; "
        "import framework.detection.adapters.random_forest; "
        "assert 'torch' not in sys.modules; assert 'xgboost' not in sys.modules; "
        "assert 'numpy' not in sys.modules; assert 'sklearn' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("bad", [[], [[]], [[1, 2]], [[None, 2, 3]], [[True, 2, 3]],
                                [["1", 2, 3]], [[float("inf"), 2, 3]],
                                [[float("nan"), 2, 3]], [[1e100, 2, 3]],
                                [[1, 2, 3]] * 501])
def test_full_only_adapter_rejects_malformed_or_sparse_measurements(bad):
    from framework.detection.adapters.common import rows_array
    with pytest.raises(ValueError):
        rows_array(bad, ["first", "second", "third"])


def test_manifest_rejects_wrong_family_and_unhashed_checkpoint(tmp_path):
    from framework.detection.adapters.common import load_manifest
    (tmp_path / "model.pt").write_bytes(b"fixture")
    config, manifest = _manifest(tmp_path, "kan", "model.pt")
    with pytest.raises(ValueError, match="model_id"):
        load_manifest(config, tmp_path, "random_forest")
    manifest["artifact_sha256"] = {}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checkpoint"):
        load_manifest(config, tmp_path, "kan")


@pytest.mark.parametrize("relative", ["../outside.pt", "/outside.pt", "C:/outside.pt"])
def test_manifest_paths_cannot_escape_project_root(tmp_path, relative):
    from framework.detection.adapters.common import artifact_path
    with pytest.raises(ValueError):
        artifact_path(tmp_path, relative)


def test_missing_metrics_does_not_prevent_manifest_loading(tmp_path):
    from framework.detection.adapters.common import load_manifest
    (tmp_path / "model.pt").write_bytes(b"fixture")
    config, _ = _manifest(tmp_path, "kan", "model.pt")
    assert load_manifest(config, tmp_path, "kan")["metrics"] is None


@pytest.fixture
def kan_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    joblib = pytest.importorskip("joblib")
    np = pytest.importorskip("numpy")
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler

    source = "external/efficient-kan/src/efficient_kan/kan.py"
    destination = tmp_path / source
    destination.parent.mkdir(parents=True)
    shutil.copyfile(PROJECT_ROOT / source, destination)
    spec = importlib.util.spec_from_file_location("fixture_genuine_kan", destination)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = module.KAN([3, 4, 3], grid_size=3, spline_order=3).eval()
    X = np.array([[0., 1., 2.], [1., 2., 3.], [2., 3., 4.]], dtype=np.float32)
    fitted = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                       ("scaler", MinMaxScaler(feature_range=(-1, 1), clip=True))]).fit(X)
    joblib.dump(fitted, tmp_path / "preprocessing.joblib")
    config = {"hidden": [4], "grid_size": 3, "spline_order": 3}
    commit = "7b6ce1c87f18c8bc90c208f6b494042344216b11"
    torch.save({"candidate": "kan", "config": config, "input_size": 3,
                "n_classes": 3, "benign_index": 0, "upstream_commit": commit,
                "state_dicts": {"kan": model.state_dict()}}, tmp_path / "model.pt")
    record = {"candidate": "kan", "features": ["first", "second", "third"],
              "classes": ["benign", "attack-a", "attack-b"], "config": config,
              "upstream_commit": commit, "benign_index": 0}
    (tmp_path / "result.json").write_text(json.dumps(record))
    adapter_config, manifest = _manifest(
        tmp_path, "kan", "model.pt", artifacts=["preprocessing.joblib", source, "result.json"],
        config=config, upstream_commit=commit, preprocessing="preprocessing.joblib",
        source_record="result.json",
        implementation="efficient-kan B-spline KAN")
    with torch.no_grad():
        expected = torch.softmax(model(torch.as_tensor(fitted.transform(X))), dim=1).numpy()
    return tmp_path, adapter_config, manifest, X.tolist(), expected


def test_genuine_kan_checkpoint_runs_on_cpu_without_training_helper(kan_checkpoint):
    from framework.detection.adapters.kan import KANAdapter
    import numpy as np
    root, config, manifest, rows, expected = kan_checkpoint
    adapter = KANAdapter(config, root)
    assert adapter.health_check()["status"] == "ready"
    actual = np.asarray(adapter.predict_proba(rows))
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    results = adapter.predict(rows)
    assert [result["label"] for result in results] == [manifest["classes"][i] for i in expected.argmax(1)]
    assert all(result["novelty"] == "not_evaluated" for result in results)
    assert all(len(result["top_classes"]) == 3 for result in results)
    assert all(parameter.device.type == "cpu" for parameter in adapter._model.parameters())


def test_deleted_kan_checkout_and_corrupt_checkpoint_are_scoped(kan_checkpoint):
    from framework.detection.adapters.kan import KANAdapter
    root, config, manifest, _, _ = kan_checkpoint
    source = root / "external/efficient-kan/src/efficient_kan/kan.py"
    original = source.read_bytes()
    source.unlink()
    assert KANAdapter(config, root).health_check()["status"] == "unavailable"
    source.write_bytes(original)
    (root / "model.pt").write_bytes(b"corrupt bytes")
    adapter = KANAdapter(config, root)
    assert adapter.health_check()["status"] == "failed"
    with pytest.raises(ValueError, match="SHA256"):
        adapter.load()


def test_missing_torch_dependency_only_changes_kan_health(kan_checkpoint, monkeypatch):
    from framework.detection.adapters.kan import KANAdapter
    from framework.detection.adapters import common
    root, config, _, _, _ = kan_checkpoint
    original = common.find_spec
    monkeypatch.setattr(common, "find_spec", lambda name: None if name == "torch" else original(name))
    health = KANAdapter(config, root).health_check()
    assert health["status"] == "unavailable"
    assert "torch" in health["reason"]


def test_kan_rejects_checkpoint_architecture_mismatch(kan_checkpoint):
    from framework.detection.adapters.kan import KANAdapter
    root, config, manifest, _, _ = kan_checkpoint
    manifest["classes"].append("extra-class")
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="class"):
        KANAdapter(config, root).load()


@pytest.mark.parametrize("field", ["features", "classes", "config", "upstream_commit"])
def test_kan_rejects_manifest_identity_that_disagrees_with_hashed_record(kan_checkpoint, field):
    from framework.detection.adapters.kan import KANAdapter
    root, config, manifest, _, _ = kan_checkpoint
    if field == "features":
        manifest["feature_names"].reverse()
    elif field == "classes":
        manifest["classes"][1:] = manifest["classes"][1:][::-1]
    elif field == "config":
        manifest["config"]["hidden"] = [8]
    else:
        record = json.loads((root / "result.json").read_text())
        record[field] = "a" * 40
        (root / "result.json").write_text(json.dumps(record))
        manifest["artifact_sha256"]["result.json"] = hashlib.sha256((root / "result.json").read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    assert KANAdapter(config, root).health_check()["status"] != "ready"
    with pytest.raises(ValueError):
        KANAdapter(config, root).load()


def test_kan_source_record_accepts_controller_model_id_and_rejects_conflicting_family(kan_checkpoint):
    from framework.detection.adapters.kan import KANAdapter
    root, config, manifest, _, _ = kan_checkpoint
    record = json.loads((root / "result.json").read_text())
    record["model_id"] = record.pop("candidate")
    (root / "result.json").write_text(json.dumps(record))
    manifest["artifact_sha256"]["result.json"] = hashlib.sha256((root / "result.json").read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    assert KANAdapter(config, root).health_check()["status"] == "ready"
    record["candidate"] = "hem"
    (root / "result.json").write_text(json.dumps(record))
    manifest["artifact_sha256"]["result.json"] = hashlib.sha256((root / "result.json").read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    assert KANAdapter(config, root).health_check()["status"] != "ready"


def test_kan_gradient_uses_scaled_inputs_and_matches_hand_calculated_derivative(kan_checkpoint):
    from framework.detection.adapters.kan import KANAdapter
    import torch
    import numpy as np
    root, config, _, rows, _ = kan_checkpoint
    adapter = KANAdapter(config, root)
    adapter.load()
    with torch.no_grad():
        # Genuine SiLU base branches, zero spline branch: at scaled x=0,
        # each SiLU derivative is 1/2. Four hidden paths yield dlogit/dx=1.
        for layer in adapter._model.layers:
            layer.spline_weight.zero_()
            layer.base_weight.fill_(1.)
        adapter._model.layers[-1].base_weight[1].fill_(2.)
        adapter._model.layers[-1].base_weight[2].fill_(-1.)
    evidence = adapter.explain([rows[1]], method="native")
    assert evidence["method"] == "gradient_input_sensitivity"
    assert evidence["predicted_class"] == "benign"
    assert evidence["is_causal"] is False
    assert evidence["is_feature_attribution"] is False
    features = sorted(evidence["features"], key=lambda item: item["feature"])
    np.testing.assert_allclose([feature["scaled_value"] for feature in features], [0., 0., 0.])
    np.testing.assert_allclose([feature["gradient"] for feature in features], [1., 1., 1.], atol=1e-6)
    np.testing.assert_allclose([feature["importance"] for feature in features], [0., 0., 0.])
    assert [feature["value"] for feature in features] == [1., 2., 3.]


def test_kan_explanation_failure_leaves_weights_and_prediction_unchanged(kan_checkpoint, monkeypatch):
    from framework.detection.adapters.kan import KANAdapter
    import torch
    import numpy as np
    root, config, _, rows, _ = kan_checkpoint
    adapter = KANAdapter(config, root)
    before = adapter.predict_proba(rows)
    weights = {name: value.clone() for name, value in adapter._model.state_dict().items()}
    original = torch.autograd.grad
    monkeypatch.setattr(torch.autograd, "grad", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected XAI failure")))
    with pytest.raises(RuntimeError, match="injected XAI failure"):
        adapter.explain(rows[:1], method="gradient")
    monkeypatch.setattr(torch.autograd, "grad", original)
    np.testing.assert_allclose(adapter.predict_proba(rows), before, atol=1e-7)
    assert all(torch.equal(value, weights[name]) for name, value in adapter._model.state_dict().items())
    assert all(parameter.grad is None for parameter in adapter._model.parameters())
    with pytest.raises(ValueError, match="one"):
        adapter.explain(rows[:2])


@pytest.fixture
def forest_checkpoint(tmp_path):
    xgb = pytest.importorskip("xgboost")
    np = pytest.importorskip("numpy")
    X = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
    y = np.asarray([0, 0, 1, 2, 1, 2])
    model = xgb.train({"objective": "multi:softprob", "num_class": 3, "num_parallel_tree": 4,
                       "tree_method": "hist", "device": "cpu", "nthread": 2,
                       "learning_rate": 1, "max_depth": 2, "seed": 42,
                       "subsample": .8, "colsample_bynode": .8}, xgb.DMatrix(X, label=y), num_boost_round=1)
    model.save_model(tmp_path / "model.ubj")
    config, manifest = _manifest(tmp_path, "random_forest", "model.ubj",
                                 config={"num_parallel_tree": 4},
                                 implementation="XGBoost standalone random forest")
    return tmp_path, config, manifest, X.tolist(), model.predict(xgb.DMatrix(X))


def test_preserved_standalone_forest_cpu_scores_match_checkpoint(forest_checkpoint):
    from framework.detection.adapters.random_forest import RandomForestAdapter
    import numpy as np
    root, config, _, rows, expected = forest_checkpoint
    adapter = RandomForestAdapter(config, root)
    np.testing.assert_allclose(adapter.predict_proba(rows), expected, atol=1e-7)
    assert len(adapter.predict(rows)) == len(rows)
    assert "standalone" in adapter.metadata()["implementation"]
    explanation = adapter.explain(rows[:1], method="global_importance")
    assert explanation["scope"] == "global"
    assert explanation["local_attribution"] is False
    assert adapter.explain(rows[:1], method="native")["scope"] == "global"


def test_standalone_forest_refuses_multiround_booster(forest_checkpoint):
    from framework.detection.adapters.random_forest import RandomForestAdapter
    import xgboost as xgb
    root, config, manifest, rows, _ = forest_checkpoint
    boosted = xgb.train({"objective": "multi:softprob", "num_class": 3, "nthread": 2},
                        xgb.DMatrix(rows, label=[0, 0, 1, 2, 1, 2]), num_boost_round=2)
    boosted.save_model(root / "model.ubj")
    manifest["artifact_sha256"]["model.ubj"] = hashlib.sha256((root / "model.ubj").read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="standalone"):
        RandomForestAdapter(config, root).load()


def test_conventional_sklearn_forest_loads_without_changing_fitted_parameters(tmp_path):
    from framework.detection.adapters.random_forest import RandomForestAdapter
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    X = np.asarray([[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 1, 0], [1, 0, 1], [2, 1, 2]], dtype=np.float32)
    model = RandomForestClassifier(n_estimators=4, random_state=42, n_jobs=1).fit(X, [0, 1, 2, 0, 1, 2])
    before = model.get_params()
    joblib.dump(model, tmp_path / "model.joblib")
    config, _ = _manifest(tmp_path, "random_forest", "model.joblib",
                          implementation="sklearn conventional RandomForestClassifier")
    adapter = RandomForestAdapter(config, tmp_path)
    np.testing.assert_allclose(adapter.predict_proba(X.tolist()), model.predict_proba(X))
    assert adapter._model.get_params() == before
    assert adapter.explain(X[:1].tolist(), "global_importance")["scope"] == "global"


@pytest.mark.parametrize("field", ["classes", "feature_names", "config"])
def test_rf_binds_ordered_metadata_to_hash_pinned_training_record(tmp_path, field):
    from framework.detection.adapters.random_forest import RandomForestAdapter
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=3, random_state=42).fit(
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]], [0, 1, 2])
    joblib.dump(model, tmp_path / "model.joblib")
    record = {"model_id": "random_forest", "feature_names": ["first", "second", "third"],
              "classes": ["benign", "attack-a", "attack-b"], "benign_index": 0, "config": {}}
    (tmp_path / "result.json").write_text(json.dumps(record))
    config, manifest = _manifest(tmp_path, "random_forest", "model.joblib",
                                 artifacts=["result.json"], source_record="result.json")
    assert RandomForestAdapter(config, tmp_path).health_check()["status"] == "ready"
    if field == "config":
        manifest[field] = {"n_estimators": 100}
    else:
        manifest[field] = manifest[field][::-1]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    adapter = RandomForestAdapter(config, tmp_path)
    assert adapter.health_check()["status"] == "incompatible"
    with pytest.raises(ValueError, match="source_record"):
        adapter.load()


def test_fixed_rf_training_uses_all_rows_and_only_training_medians(tmp_path):
    from framework.training.fixed_random_forest import train_fixed_random_forest
    import joblib
    import numpy as np
    X = np.asarray([[0., 0., 0.], [1., 1., 1.], [2., np.nan, 2.],
                    [3., 3., 3.], [4., 4., 4.], [5., 5., 5.]], dtype=np.float32)
    y = np.asarray([0, 0, 1, 1, 2, 2])
    result = train_fixed_random_forest(X, y, np.asarray([[100., 100., 100.]]),
                                     np.asarray([2]), 3, {}, 42, tmp_path / "run")
    fitted = joblib.load(result["checkpoint_path"])
    assert result["train_rows"] == 6
    assert result["n_features"] == 3
    assert result["device"] == "cpu"
    assert fitted.named_steps["classifier"].n_estimators == 100
    assert fitted.named_steps["classifier"].class_weight is None
    assert fitted.named_steps["classifier"].bootstrap is True
    np.testing.assert_allclose(fitted.named_steps["imputer"].statistics_, [2.5, 3., 2.5])
    assert result["checkpoint_reload_verified"] is True
    with pytest.raises(FileExistsError):
        train_fixed_random_forest(X, y, X[:1], y[:1], 3, {}, 42, tmp_path / "run")


def test_fixed_rf_training_refuses_tuning(tmp_path):
    from framework.training.fixed_random_forest import train_fixed_random_forest
    with pytest.raises(ValueError, match="fixed"):
        train_fixed_random_forest([[0], [1]], [0, 1], [[0]], [0], 2,
                                  {"max_depth": 8}, 42, tmp_path / "run")
