"""Training-only augmentation and promotion checks; no research GPU fitting."""
from dataclasses import replace

import numpy as np
import pytest

from experiment import train_partial_enhancements as study
from framework.detection.packet_xgboost import FEATURE_NAMES
from framework.training.partial_augmentation import (
    MAX_EXACT_COUNT, addition_counts, audit_augmentation, augment_packet_minority,
    eligible_packet_parents, interpolate_counts,
)


@pytest.fixture
def parents():
    # Include one zero-traffic observation, retained in research training but
    # excluded from augmentation. A singleton class is not a SMOTE parent pair.
    names = ["log_unavailable", *reversed(FEATURE_NAMES), "network_protocol__present_TCP"]
    counts = np.array([
        [100, 70, 5, 0, 1, 0, 0], [120, 80, 4, 0, 2, 0, 0],
        [140, 90, 3, 0, 3, 0, 0], [160, 95, 2, 0, 4, 0, 0],
        [180, 100, 1, 0, 5, 0, 0], [200, 110, 0, 0, 6, 0, 0],
        [500, 0, 0, 0, 0, 0, 0], [600, 0, 0, 0, 0, 0, 0],
        [10, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float32)
    rows = np.full((len(counts), len(names)), np.nan, dtype=np.float32)
    rows[:, [names.index(name) for name in FEATURE_NAMES]] = counts
    labels = np.array([0]*6+[1]*2+[2, 3], dtype=np.int32)
    ids = np.arange(100, 100+len(rows), dtype=np.int64)
    return rows, labels, ids, names


def generate(parents, method="constrained_smote", **overrides):
    rows, labels, ids, names = parents
    settings = dict(allowed_training_ids=ids, num_classes=4, method=method, seed=42)
    settings.update(overrides)
    return augment_packet_minority(rows, labels, ids, names, **settings)


@pytest.mark.parametrize("method", ["none", "random_oversampling", "constrained_smote"])
def test_augmentation_is_reproducible_and_preserves_missingness(parents, method):
    rows, labels, ids, names = parents
    original, original_labels = rows.copy(), labels.copy()
    batch, repeated = generate(parents, method), generate(parents, method)
    np.testing.assert_array_equal(rows, original)
    np.testing.assert_array_equal(labels, original_labels)
    np.testing.assert_array_equal(batch.values, repeated.values)
    np.testing.assert_array_equal(batch.labels, repeated.labels)
    assert batch.summary == repeated.summary
    assert audit_augmentation(batch, rows, labels, ids, names)
    missing = [i for i, name in enumerate(names) if name not in FEATURE_NAMES]
    assert np.isnan(batch.values[:, missing]).all()
    assert batch.summary["new_independent_observations"] == 0
    assert batch.summary["validation_used"] is False
    assert batch.summary["excluded_augmentation_parent_rows"] == 1
    assert 109 not in batch.provenance["parent_a_row_id"]
    if method == "constrained_smote":
        assert 2 not in batch.labels  # singleton class
        assert batch.summary["synthetic_rows"] == len(batch.labels)
    if method == "random_oversampling":
        assert batch.summary["synthetic_rows"] == 0
        assert batch.summary["duplicate_rows"] == len(batch.labels)


@pytest.mark.parametrize("change", ["source_partition", "allowed_training_ids", "label", "unavailable", "duplicate_id"])
def test_rejects_nontraining_parents_and_fabricated_columns(parents, change):
    rows, labels, ids, names = parents
    kwargs = {}
    if change == "source_partition":
        kwargs["source_partition"] = "validation"
    elif change == "allowed_training_ids":
        kwargs["allowed_training_ids"] = ids[:-1]
    elif change == "label":
        labels[0] = 84
    elif change == "unavailable":
        rows[0, 0] = 0  # missing sensor data cannot be manufactured as zero
    else:
        ids[0] = ids[1]
    with pytest.raises(ValueError):
        generate((rows, labels, ids, names), **kwargs)


def test_only_bounded_positive_integer_counts_are_eligible():
    valid = np.array([[5, 0, 1, 2, 3, 4, 5], [MAX_EXACT_COUNT]*7], dtype=float)
    assert eligible_packet_parents(valid).all()
    invalid = np.repeat(valid[:1], 6, axis=0)
    invalid[0, 0] = 0
    invalid[1, 1] = -1
    invalid[2, 1] = 6
    invalid[3, 1] = 0.5
    invalid[4, 0] = MAX_EXACT_COUNT+1
    invalid[5, 1] = np.nan
    assert not eligible_packet_parents(invalid).any()


def test_log_interpolation_endpoints_and_physical_constraints():
    left = np.array([[1, 0, 1, 0, 1, 0, 0], [MAX_EXACT_COUNT]*7], dtype=float)
    right = np.array([[MAX_EXACT_COUNT]*7, [1, 0, 0, 1, 0, 0, 0]], dtype=float)
    np.testing.assert_array_equal(interpolate_counts(left, right, np.zeros(2)), left)
    np.testing.assert_array_equal(interpolate_counts(left, right, np.ones(2)), right)
    generated = interpolate_counts(left, right, np.array([0.4, 0.9]))
    assert generated.dtype == np.float32
    assert eligible_packet_parents(generated).all()
    with pytest.raises(ValueError):
        interpolate_counts(left, right, np.array([-0.1, 1.1]))


def test_quotas_are_capped_by_training_support_and_global_budget():
    support = np.array([100, 8, 4, 2, 1, 0])
    added, _ = addition_counts(support, minimum_parents=2)
    assert added.sum() <= np.floor(support.sum()*0.5)
    assert np.all(added <= support*2)
    assert added[-2:].tolist() == [0, 0]
    # Exercise proportional allocation when the global budget binds.
    counts = np.array([30, 2, 2, 2, 2])
    capped, _ = addition_counts(counts, target_quantile=1, max_extra_fraction=0.1)
    assert capped.sum() == 3
    np.testing.assert_array_equal(capped, addition_counts(counts, target_quantile=1, max_extra_fraction=0.1)[0])


@pytest.mark.parametrize("target", ["parent", "label", "value", "coefficient", "synthetic"])
def test_provenance_tampering_fails_audit(parents, target):
    batch = generate(parents)
    assert len(batch.labels)
    record = {key: value.copy() for key, value in batch.provenance.items()}
    values, labels = batch.values.copy(), batch.labels.copy()
    if target == "parent":
        record["parent_a_row_id"][0] = 99999
    elif target == "label":
        labels[0] = 0
    elif target == "value":
        values[0, 0] = 0
    elif target == "coefficient":
        record["coefficient"][0] = 1.5
    else:
        record["synthetic"][0] = False
    with pytest.raises(ValueError):
        audit_augmentation(replace(batch, values=values, labels=labels, provenance=record), *parents)


def test_duplicate_parent_vectors_do_not_break_smote(parents):
    rows, labels, ids, names = parents
    rows[7] = rows[6]
    batch = generate((rows, labels, ids, names))
    assert audit_augmentation(batch, rows, labels, ids, names)
    assert batch.summary["added_vectors_already_present_in_training"] > 0


def result(seed, accuracy, f1=0.4):
    return {"request": {"seed": seed}, "profile_metrics": {
        "packet_counts_7": {"accuracy": accuracy, "macro_f1": f1}}}


def test_promotion_requires_three_seed_control_and_incumbent_improvement():
    controls = [result(seed, 0.71) for seed in study.SEEDS]
    treatments = [result(42, 0.73), result(17, 0.72), result(73, 0.72)]
    promoted, best, _ = study.promotion_decision(controls, treatments, 0.701)
    assert promoted and best is treatments[0]
    assert not study.promotion_decision(controls, treatments, 0.74)[0]
    assert not study.promotion_decision([result(seed, 0.75) for seed in study.SEEDS], treatments, 0.701)[0]
    with pytest.raises(ValueError):
        study.promotion_decision(controls[:2], treatments, 0.701)
    with pytest.raises(ValueError):
        study.promotion_decision(controls, [result(42, 0.74)]*3, 0.701)


def test_cuda_parameters_preserve_multiclass_architecture_and_isolate_treatments():
    incumbent = dict(id="fresh", warm_start=False, max_depth=6, learning_rate=0.08,
                     loss="label_smoothed_ce", smoothing=0.05, weight_strategy="sqrt_balanced")
    configs = study.configurations()
    assert len(configs) == 5
    for config in configs:
        params = study.xgboost_parameters(incumbent, config, 17, 84)
        assert params["device"] == "cuda:0" and params["tree_method"] == "hist"
        assert params["objective"] == "multi:softprob" and params["num_class"] == 84
        assert params["seed"] == 17 and params["max_depth"] == 6
        assert not {"id", "warm_start", "smoothing", "loss", "weight_strategy"} & params.keys()
    assert study.RUNTIME_CAP_SECONDS == 600 and study.MAX_ROUNDS == 500
