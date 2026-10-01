"""Exact native TreeSHAP on the executed XGBoost checkpoint, in margin units."""
import numpy as np


def explain_tree(booster, X, feature_names, classes, label):
    import xgboost as xgb
    X = np.asarray(X, dtype=np.float32)
    if X.shape != (1, len(feature_names)) or np.isinf(X).any():
        raise ValueError("TreeSHAP requires one ordered row of finite or missing inputs")
    matrix = xgb.DMatrix(X, nthread=2)
    margins = booster.predict(matrix, output_margin=True, strict_shape=True)
    contributions = booster.predict(matrix, pred_contribs=True, approx_contribs=False, strict_shape=True)
    if contributions.shape != (1, len(classes), len(feature_names) + 1):
        raise ValueError("Unexpected TreeSHAP attribution shape")
    totals = contributions.astype(np.float64).sum(axis=2)
    if not np.isfinite(contributions).all() or not np.isfinite(margins).all() or not np.allclose(totals, margins, atol=1e-4, rtol=1e-5):
        raise RuntimeError("SHAP failed to reconstruct model margins; refusing an unverified explanation")
    index = classes.index(label)
    values = contributions[0, index, :-1]
    order = np.argsort(-np.abs(values), kind="stable")
    all_features = [{"feature": name, "value": float(X[0, i]) if np.isfinite(X[0, i]) else None,
                     "is_missing": bool(np.isnan(X[0, i])), "observed": bool(np.isfinite(X[0, i])),
                     "attribution_basis": "missing-value default routing" if np.isnan(X[0, i]) else "observed input",
                     "shap_value": float(values[i])} for i, name in enumerate(feature_names)]
    bias = float(contributions[0, index, -1])
    total = float(values.astype(np.float64).sum())
    margin = float(margins[0, index])
    return {"method": "Exact TreeSHAP (native XGBoost)", "output_space": "raw class margin",
            "class_label": label, "bias": bias, "margin": margin, "contribution_sum": total,
            "reconstruction_error": abs(bias + total - margin),
            "important_features": [all_features[int(i)] for i in order[:10]],
            "all_features": all_features,
            "limitations": "Signed local attributions use the model's training tree-cover reference (tree path dependent). "
                            "Positive contributions increase this class margin; negative ones decrease it. "
                            "They are not probability percentage points, comparisons against the benign class, or causal evidence. "
                            "Missing-input contributions explain the model's default routing, not observed traffic; missing does not mean zero. "
                            "All features and the bias reconstruct the raw margin; the chart displays the top ten."}
