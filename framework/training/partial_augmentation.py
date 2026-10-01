"""Training-only duplicate and constrained SMOTE views in the existing schema.

Generated labels are inherited training labels, not measured ground truth.
Unavailable features always remain IEEE NaN; this module does not fit a scaler.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from framework.detection.packet_xgboost import FEATURE_NAMES

MAX_EXACT_COUNT = 2**24
METHODS = ("none", "random_oversampling", "constrained_smote")


def array_sha256(values: np.ndarray) -> str:
    """Hash dtype, shape and bytes so incompatible arrays cannot share identity."""
    value = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def eligible_packet_parents(counts: np.ndarray) -> np.ndarray:
    """Only positive, exactly representable physical count vectors can be parents."""
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError("Seven canonical packet counts are required")
    return (np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
            & (values <= MAX_EXACT_COUNT).all(axis=1)
            & (values == np.floor(values)).all(axis=1)
            & (values[:, 0] > 0)
            & (values[:, 1:] <= values[:, :1]).all(axis=1))


def interpolate_counts(left, right, coefficient):
    """Convex log1p interpolation, rounded and projected onto count constraints."""
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    alpha = np.asarray(coefficient, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] != 7 or alpha.shape != (len(a),):
        raise ValueError("Parent pairs and interpolation coefficients must align")
    if not eligible_packet_parents(a).all() or not eligible_packet_parents(b).all():
        raise ValueError("Interpolation requires valid positive-packet training parents")
    if not np.isfinite(alpha).all() or (alpha < 0).any() or (alpha > 1).any():
        raise ValueError("Interpolation coefficients must lie in [0, 1]")
    counts = np.rint(np.expm1((1-alpha[:, None])*np.log1p(a) + alpha[:, None]*np.log1p(b)))
    counts = np.clip(counts, 0, MAX_EXACT_COUNT)
    counts[:, 0] = np.maximum(counts[:, 0], 1)
    counts[:, 1:] = np.minimum(counts[:, 1:], counts[:, :1])
    return counts.astype(np.float32)


def addition_counts(support, *, minimum_parents=1, target_quantile=.75,
                    max_class_multiplier=3., max_extra_fraction=.5):
    """Training-support-only targets with deterministic proportional cap allocation."""
    counts = np.asarray(support)
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer) or (counts < 0).any():
        raise ValueError("Class support must contain nonnegative integer counts")
    if not 0 <= target_quantile <= 1 or max_class_multiplier < 1 or not 0 <= max_extra_fraction <= 1:
        raise ValueError("Invalid bounded augmentation settings")
    if minimum_parents < 1:
        raise ValueError("At least one training parent is required")
    positive = counts[counts > 0]
    if not len(positive):
        return np.zeros_like(counts), 0
    target = int(np.ceil(np.quantile(positive, target_quantile)))
    desired = np.minimum(np.maximum(target-counts, 0),
                         np.floor(counts*(max_class_multiplier-1)).astype(np.int64))
    desired[counts < minimum_parents] = 0
    budget = int(np.floor(counts.sum()*max_extra_fraction))
    if desired.sum() > budget:
        quota = desired.astype(np.float64)*budget/desired.sum()
        allocation = np.floor(quota).astype(np.int64)
        remainder = budget-int(allocation.sum())
        order = np.lexsort((np.arange(len(counts)), -(quota-allocation)))
        allocation[order[:remainder]] += 1
        desired = allocation
    return desired, target


@dataclass(frozen=True)
class AugmentationBatch:
    values: np.ndarray
    labels: np.ndarray
    provenance: dict[str, np.ndarray]
    summary: dict


def augment_packet_minority(rows, labels, row_ids, feature_names, *,
                            allowed_training_ids, num_classes, method, seed,
                            source_partition="train", target_quantile=.75,
                            max_class_multiplier=3., max_extra_fraction=.5,
                            neighbors=5) -> AugmentationBatch:
    """Return additional views only; originals and their labels are never modified.

    Neighbors use Euclidean distance between log1p count vectors from the same
    training class. No validation data, learned normalization or held-out labels
    enter this function. Duplicate count vectors may yield duplicate synthetic
    values; those cases are reported rather than described as novel traffic.
    """
    if source_partition != "train":
        raise ValueError("Augmentation is restricted to the original training partition")
    if method not in METHODS or type(seed) is not int or seed < 0 or neighbors < 1:
        raise ValueError("Invalid augmentation method, seed or neighbor count")
    names = list(feature_names)
    if len(names) != len(set(names)) or not set(FEATURE_NAMES).issubset(names):
        raise ValueError("Use the unchanged canonical feature names")
    x, y, ids = np.asarray(rows), np.asarray(labels), np.asarray(row_ids)
    allowed = np.asarray(allowed_training_ids)
    if x.ndim != 2 or x.shape[1] != len(names) or y.shape != (len(x),) or ids.shape != (len(x),):
        raise ValueError("Training features, labels and source identities must align")
    if not np.issubdtype(y.dtype, np.integer) or (y < 0).any() or (y >= num_classes).any():
        raise ValueError("Training labels must use the fixed class order")
    if not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(allowed.dtype, np.integer):
        raise ValueError("Integer source-row identities are required")
    if len(np.unique(ids)) != len(ids) or not np.isin(ids, allowed).all():
        raise ValueError("Every unique parent must belong to the original training rows")
    columns = [names.index(name) for name in FEATURE_NAMES]
    unavailable = np.ones(len(names), dtype=bool)
    unavailable[columns] = False
    if not np.isnan(x[:, unavailable]).all():
        raise ValueError("Augmentation accepts seven-count views only; other inputs must be missing")
    count_values = np.asarray(x[:, columns], dtype=np.float64)
    eligible = eligible_packet_parents(count_values)
    eligible_indices = np.flatnonzero(eligible)
    support = np.bincount(y[eligible], minlength=num_classes)
    additions, target = addition_counts(support, minimum_parents=2 if method == "constrained_smote" else 1,
                                       target_quantile=target_quantile,
                                       max_class_multiplier=max_class_multiplier,
                                       max_extra_fraction=max_extra_fraction)
    if method == "none":
        additions[:] = 0
    rng = np.random.default_rng(seed)
    parent_a, parent_b, coefficients, generated_labels = [], [], [], []
    for label, size in enumerate(additions):
        if not size:
            continue
        positions = eligible_indices[y[eligible_indices] == label]
        local_a = rng.integers(len(positions), size=int(size))
        selected_a = positions[local_a]
        if method == "random_oversampling":
            selected_b, alpha = selected_a.copy(), np.zeros(size, dtype=np.float64)
        else:
            from sklearn.neighbors import NearestNeighbors
            logged = np.log1p(count_values[positions])
            model = NearestNeighbors(n_neighbors=min(neighbors+1, len(positions)),
                                     algorithm="brute", metric="euclidean", n_jobs=1).fit(logged)
            nearby = model.kneighbors(logged[local_a], return_distance=False)
            selected_b = np.array([positions[rng.choice(candidates[candidates != anchor][:neighbors])]
                                   for anchor, candidates in zip(local_a, nearby)], dtype=np.int64)
            alpha = rng.random(int(size))
        parent_a.extend(selected_a.tolist())
        parent_b.extend(selected_b.tolist())
        coefficients.extend(alpha.tolist())
        generated_labels.extend([label]*int(size))
    a = np.asarray(parent_a, dtype=np.int64)
    b = np.asarray(parent_b, dtype=np.int64)
    alpha = np.asarray(coefficients, dtype=np.float64)
    output_labels = np.asarray(generated_labels, dtype=np.int32)
    generated = np.full((len(a), len(names)), np.nan, dtype=np.float32)
    if len(a):
        generated[:, columns] = (count_values[a].astype(np.float32) if method == "random_oversampling"
                                  else interpolate_counts(count_values[a], count_values[b], alpha))
    provenance = {"parent_a_row_id": ids[a].astype(np.int64), "parent_b_row_id": ids[b].astype(np.int64),
                  "parent_a_label": y[a].astype(np.int32), "parent_b_label": y[b].astype(np.int32),
                  "inherited_label": output_labels.copy(), "coefficient": alpha,
                  "seed": np.full(len(a), seed, dtype=np.int64),
                  "synthetic": np.full(len(a), method == "constrained_smote", dtype=bool),
                  "duplicate_view": np.full(len(a), method == "random_oversampling", dtype=bool),
                  "count_values": generated[:, columns].copy()}
    generated_counts = np.ascontiguousarray(generated[:, columns])
    if len(generated_counts):
        dtype = np.dtype((np.void, generated_counts.shape[1]*generated_counts.dtype.itemsize))
        original_keys = np.ascontiguousarray(count_values[eligible].astype(np.float32)).view(dtype).ravel()
        generated_keys = generated_counts.view(dtype).ravel()
        repeated_values = int(np.isin(generated_keys, np.unique(original_keys)).sum())
    else:
        repeated_values = 0
    summary = {"method": method, "seed": seed, "source_partition": "train",
               "input_training_live_rows": len(x), "eligible_parent_rows": int(eligible.sum()),
               "excluded_augmentation_parent_rows": int((~eligible).sum()),
               "eligibility": "All seven counts finite, nonnegative integers <= 2**24; packets > 0; each flag <= packets",
               "excluded_rows_policy": "Excluded from augmentation only; retained unchanged in original training views",
               "training_parent_source_ids_sha256": array_sha256(ids),
               "training_parent_labels_sha256": array_sha256(y),
               "training_live_input_sha256": array_sha256(x),
               "eligible_support": support.tolist(), "added_support": additions.tolist(),
               "training_support_target": target, "target_quantile": target_quantile,
               "max_class_multiplier": max_class_multiplier, "max_extra_fraction": max_extra_fraction,
               "neighbors": neighbors, "distance": "Euclidean on log1p training-only seven-count vectors; no fitted scaling",
               "added_rows": len(a), "synthetic_rows": len(a) if method == "constrained_smote" else 0,
               "duplicate_rows": len(a) if method == "random_oversampling" else 0,
               "added_vectors_already_present_in_training": repeated_values,
               "generated_values_sha256": array_sha256(generated),
               "generated_labels_sha256": array_sha256(output_labels),
               "provenance_array_sha256": {name: array_sha256(value) for name, value in provenance.items()},
               "new_independent_observations": 0, "validation_used": False, "test_evaluated": False,
               "label_provenance": "Inherited same-class training labels; synthetic labels are not measured truth"}
    return AugmentationBatch(generated, output_labels, provenance, summary)


def audit_augmentation(batch, parent_rows, parent_labels, parent_ids, feature_names):
    """Verify stored parent identities, labels and the generation algebra afresh."""
    x, y, ids = np.asarray(parent_rows), np.asarray(parent_labels), np.asarray(parent_ids)
    names = list(feature_names)
    columns = [names.index(name) for name in FEATURE_NAMES]
    summary, record = batch.summary, batch.provenance
    for key, value in (("training_parent_source_ids_sha256", ids),
                       ("training_parent_labels_sha256", y), ("training_live_input_sha256", x)):
        if summary.get(key) != array_sha256(value):
            raise ValueError("Saved augmentation parent arrays differ from frozen training data")
    if batch.values.shape != (len(batch.labels), len(names)):
        raise ValueError("Saved augmentation rows do not align with the shared schema")
    if not eligible_packet_parents(batch.values[:, columns]).all():
        raise ValueError("Saved augmentation violates packet-count constraints")
    unavailable = np.ones(len(names), dtype=bool)
    unavailable[columns] = False
    if not np.isnan(batch.values[:, unavailable]).all():
        raise ValueError("Saved augmentation fabricated unavailable input values")
    lookup = {int(value): index for index, value in enumerate(ids)}
    if len(lookup) != len(ids):
        raise ValueError("Training parent source identities are not unique")
    try:
        a = np.array([lookup[int(value)] for value in record["parent_a_row_id"]], dtype=np.int64)
        b = np.array([lookup[int(value)] for value in record["parent_b_row_id"]], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("Saved augmentation used a source outside exact-window training parents") from exc
    for label_array in (record["parent_a_label"], record["parent_b_label"], record["inherited_label"], y[a], y[b]):
        if not np.array_equal(batch.labels, label_array):
            raise ValueError("Saved augmentation labels do not match their same-class original training parents")
    if len(a) != len(batch.labels) or not eligible_packet_parents(x[a][:, columns]).all() or not eligible_packet_parents(x[b][:, columns]).all():
        raise ValueError("Saved augmentation includes an ineligible training parent")
    if not np.array_equal(record["seed"], np.full(len(a), summary["seed"])):
        raise ValueError("Saved augmentation seed provenance changed")
    method = summary["method"]
    if method == "random_oversampling":
        if not np.array_equal(a, b) or np.any(record["coefficient"] != 0):
            raise ValueError("Random oversampling must duplicate one existing training parent")
        expected = x[a][:, columns]
    elif method == "constrained_smote":
        expected = interpolate_counts(x[a][:, columns], x[b][:, columns], record["coefficient"])
    else:
        if len(a):
            raise ValueError("Unaugmented control cannot contain generated rows")
        expected = batch.values[:, columns]
    if (not np.array_equal(expected, batch.values[:, columns])
            or not np.array_equal(expected, record["count_values"])):
        raise ValueError("Saved augmentation does not reproduce its documented parent interpolation")
    if (not np.array_equal(record["synthetic"], np.full(len(a), method == "constrained_smote"))
            or not np.array_equal(record["duplicate_view"], np.full(len(a), method == "random_oversampling"))):
        raise ValueError("Synthetic and duplicated training views were mislabelled")
    for name, value in record.items():
        if summary["provenance_array_sha256"].get(name) != array_sha256(value):
            raise ValueError("Saved augmentation provenance array changed")
    if summary["generated_values_sha256"] != array_sha256(batch.values) or summary["generated_labels_sha256"] != array_sha256(batch.labels):
        raise ValueError("Saved augmentation values or labels changed")
    return True
