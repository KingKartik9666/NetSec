"""Offline LIME and native XGBoost statistics, with explicit reference provenance."""
import hashlib
import json
from pathlib import Path
import numpy as np


def explain_additional(booster, values, names, classes, reference_path, *, packet=False):
    from lime.lime_tabular import LimeTabularExplainer
    import xgboost as xgb
    reference_path = Path(reference_path)
    metadata = json.loads(reference_path.with_suffix(".json").read_text())
    if metadata["feature_names"] != names or hashlib.sha256(reference_path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("The frozen training reference checksum or feature order does not match")
    with np.load(reference_path, allow_pickle=False) as reference:
        training, medians = reference["training"], reference["medians"]
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(names),) or np.isinf(values).any():
        raise ValueError("LIME requires one ordered row of finite or missing inputs")
    missing = np.isnan(values)
    observed = np.flatnonzero(~missing)
    if not len(observed):
        raise ValueError("LIME requires at least one observed input")
    from framework.detection.packet_xgboost import FEATURE_NAMES as packet_names
    canonical_packet_profile = {names[i] for i in observed} == set(packet_names)
    count_indices = [names.index(name) for name in packet_names] if canonical_packet_profile else []
    anchor_counts = values[count_indices]
    valid_counts = (canonical_packet_profile and np.all(anchor_counts >= 0)
                    and np.all(anchor_counts <= 2**24) and np.all(anchor_counts == np.floor(anchor_counts))
                    and anchor_counts[0] >= 1 and np.all(anchor_counts[1:] <= anchor_counts[0]))
    if packet and not valid_counts:
        raise ValueError("Packet LIME requires seven valid observed packet-count inputs")
    constrain_packets = bool(valid_counts)
    finite_training = np.where(np.isfinite(training), training, medians)
    # Unavailable numeric AND categorical inputs are excluded from the surrogate
    # space. Imputing them before LIME would change its distances/conditions even
    # when the prediction callback later restores NaN.
    row = values[observed]
    def predict(neighbors):
        neighbors = np.asarray(neighbors, dtype=np.float32)
        matrix = np.full((len(neighbors), len(names)), np.nan, dtype=np.float32)
        matrix[:, observed] = neighbors
        if constrain_packets:
            counts = np.clip(np.rint(matrix[:, count_indices]), 0, 2**24)
            counts[:, 0] = np.maximum(1, counts[:, 0])
            counts[:, 1:] = np.minimum(counts[:, 1:], counts[:, :1])
            matrix[:, count_indices] = counts
        matrix[:, missing] = np.nan
        return booster.predict(xgb.DMatrix(matrix, nthread=2), strict_shape=True)
    scores = predict(row.reshape(1, -1))[0]
    target = int(scores.argmax())
    categorical = set(metadata["categorical_indices"])
    explainer = LimeTabularExplainer(finite_training[:, observed], mode="classification",
                                    feature_names=[names[i] for i in observed], class_names=classes,
                                    categorical_features=[i for i, original in enumerate(observed) if original in categorical],
                                    discretize_continuous=True, random_state=42)
    local = explainer.explain_instance(row, predict, labels=(target,), num_features=min(12, len(observed)), num_samples=2048)
    global_values = {kind: booster.get_score(importance_type=kind)
                     for kind in ["weight", "gain", "cover", "total_gain", "total_cover"]}
    features = []
    for i, name in enumerate(names):
        column = finite_training[:, i]
        features.append({"feature": name, "value": None if missing[i] else float(values[i]),
                         "is_missing": bool(missing[i]), "observed": bool(not missing[i]),
                         **{kind: float(data.get(name, data.get(f"f{i}", 0))) for kind, data in global_values.items()},
                         "reference_median": float(medians[i]), "reference_p25": float(np.percentile(column, 25)),
                         "reference_p75": float(np.percentile(column, 75)),
                         "reference_missing_rows": int(np.isnan(training[:, i]).sum())})
    config = json.loads(booster.save_config())["learner"]
    return {"lime": {"method": "LIME local linear surrogate", "class_label": classes[target], "seed": 42,
                     "perturbations": 2048, "reference_rows": len(training),
                     "observed_features": len(observed), "missing_features": int(missing.sum()),
                     "perturbation_constraints": "integer packet and TCP-flag counts" if constrain_packets else None,
                     "model_score": float(scores[target]), "surrogate_prediction": float(local.local_pred[0]),
                     "local_fidelity_r2": float(local.score),
                     "conditions": [{"condition": name, "weight": float(weight)} for name, weight in local.as_list(label=target)],
                     "limitations": "Approximate local surrogate on synthetic perturbations, not observed traffic or causal evidence. Correlations may be broken; inspect local fidelity. Missing numeric and categorical inputs are excluded from the surrogate and remain NaN in every model prediction. "
                                     + ("Seven-count packet perturbations are rounded and constrained to valid flag counts."
                                        if constrain_packets else "Observed-column perturbations are unconstrained and may not represent physically valid traffic.")},
            "statistics": {"method": "Native XGBoost gain, split count and cover",
                           "scope": "Global tree statistics; not local SHAP or causal effects",
                           "features": features,
                           "model_parameters": {"features": booster.num_features(), "classes": len(classes),
                                                "boosting_rounds": booster.num_boosted_rounds(),
                                                "objective": config["objective"]["name"], "device": "cpu"},
                           "definitions": {"weight": "Number of splits using the feature", "gain": "Mean training loss reduction per split",
                                           "cover": "Mean Hessian-weighted coverage per split", "total_gain": "Sum of split gains",
                                           "total_cover": "Sum of Hessian-weighted split coverage"}},
            "reference": metadata, "llm_executed": False}
