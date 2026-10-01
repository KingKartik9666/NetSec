"""Research metrics and checkpoint-selection invariants, independent of CUDA."""
import pytest

from framework.training.tracking import classification_metrics, choose_best, completed_trial, prepared_identity, validate_manifest


def test_metrics_measure_benign_false_positives_and_missing_class_recall():
    result = classification_metrics([0, 0, 1, 1, 2], [0, 1, 1, 1, 1],
                                    ["benign", "a", "b"], 0)
    assert result["false_positive_rate"] == 0.5
    assert result["per_class"]["b"]["recall"] == 0
    assert result["accuracy"] == 0.6
    assert result["macro_f1"] < result["weighted_f1"]


def test_worse_trial_cannot_replace_saved_best():
    best = {"run_id": "good", "status": "completed", "metrics": {"macro_f1": .8}}
    worse = {"run_id": "worse", "status": "completed", "metrics": {"macro_f1": .7}}
    failed = {"run_id": "failure", "status": "failed"}
    assert choose_best(best, worse) is best
    assert choose_best(best, failed) is best
    assert choose_best(None, worse) is worse


def test_no_label_columns_are_allowed_as_features():
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest({"feature_names": ["network_count", "label4"],
                           "classes": ["benign", "attack"], "benign_index": 0})


def test_class_predictions_outside_declared_classes_are_rejected():
    with pytest.raises(ValueError):
        classification_metrics([0, 1], [0, 5], ["benign", "attack"], 0)


def test_balanced_accuracy_uses_observed_classes_and_discloses_missing_support():
    result = classification_metrics([0, 1], [0, 1], ["benign", "a", "unobserved"], 0)
    assert result["balanced_accuracy"] == 1
    assert result["macro_recall"] == pytest.approx(2 / 3)
    assert result["classes_without_evaluation_support"] == ["unobserved"]


def test_resume_rejects_different_prepared_data_or_hyperparameters(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    for path in [first, second]:
        path.mkdir()
        (path / "manifest.json").write_text('{"features": ["a"]}')
    checkpoint = tmp_path / "model.ubj"
    checkpoint.write_bytes(b"test checkpoint presence")
    record = {"candidate": "xgboost", "config_id": "same", "seed": 42, "status": "completed",
              "config": {"max_depth": 4}, "checkpoint_path": str(checkpoint), **prepared_identity(first)}
    assert completed_trial([record], "xgboost", "same", 42, record["config"], first) is record
    with pytest.raises(ValueError, match="prepared identity"):
        completed_trial([record], "xgboost", "same", 42, record["config"], second)
    with pytest.raises(ValueError, match="parameters"):
        completed_trial([record], "xgboost", "same", 42, {"max_depth": 8}, first)
    (first / "manifest.json").write_text('{"features": ["b"]}')
    with pytest.raises(ValueError, match="prepared identity"):
        completed_trial([record], "xgboost", "same", 42, record["config"], first)
