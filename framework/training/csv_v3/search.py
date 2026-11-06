"""Optuna search spaces for cascade v3, centred on the frozen recipes' own parameter names.

Each space returns a complete config: the frozen recipe with sampled overrides, so caps, losses and
selection settings that are not searched stay exactly as recorded.
"""
from __future__ import annotations

KAN_HIDDEN = ([128, 64, 32], [256, 128, 64], [256, 128], [64, 32])
HEM_HIDDEN = ([512, 256, 64, 32], [1024, 512, 128, 32], [256, 128, 64])


def suggest(family: str, task: str, trial, default: dict) -> dict:
    """Sample one configuration for ``family``; ``trial`` may be an Optuna Trial or FixedTrial."""
    c = dict(default)
    if family == "xgboost":
        c.update(
            n_estimators=4000, early_stopping_rounds=60, runtime_seconds=3600,
            max_depth=trial.suggest_int("max_depth", 4, 12),
            lr=trial.suggest_float("lr", 0.02, 0.3, log=True),
            min_child_weight=trial.suggest_float("min_child_weight", 0.5, 50, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
            max_bin=trial.suggest_categorical("max_bin", [128, 256, 512]),
        )
        strategies = ["none", "sqrt_balanced", "balanced"] if task == "8" else ["none", "sqrt_balanced"]
        c["weight_strategy"] = trial.suggest_categorical("weight_strategy", strategies)
        if task == "8":
            c["rows_policy"] = trial.suggest_categorical("rows_policy", ["all", "targeted_only"])
    elif family == "kan":
        c.update(
            lr=trial.suggest_float("lr", 3e-4, 5e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            hidden=list(KAN_HIDDEN[trial.suggest_int("hidden_index", 0, len(KAN_HIDDEN) - 1)]),
            grid_size=trial.suggest_categorical("grid_size", [5, 7, 9, 11]),
            spline_regularization=trial.suggest_float("spline_regularization", 1e-7, 1e-4, log=True),
            weight_strategy=trial.suggest_categorical("weight_strategy", ["none", "sqrt_balanced"]),
            batch_size=trial.suggest_categorical("batch_size", [256, 512]),
        )
    elif family == "hem":
        c.update(
            lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            pca_components=trial.suggest_categorical("pca_components", [32, 64, 128]),
            knn_k=trial.suggest_categorical("knn_k", [1, 3, 5, 9]),
            hidden=list(HEM_HIDDEN[trial.suggest_int("hidden_index", 0, len(HEM_HIDDEN) - 1)]),
            weight_strategy=trial.suggest_categorical("weight_strategy", ["none", "sqrt_balanced"]),
        )
    elif family in ("tabm_ne", "tabm_hier", "tabm_hier_ne"):
        c.update(
            k=trial.suggest_categorical("k", [8, 16, 32]),
            width=trial.suggest_categorical("width", [256, 384, 512]),
            depth=trial.suggest_int("depth", 2, 4),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            lr=trial.suggest_float("lr", 5e-4, 5e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            weight_power=trial.suggest_categorical("weight_power", [0.0, 0.25, 0.5]),
            smoothing=trial.suggest_float("smoothing", 0.0, 0.1),
        )
        if family != "tabm_hier":
            c.update(n_bins=trial.suggest_categorical("n_bins", [16, 32, 48]),
                     d_embedding=trial.suggest_categorical("d_embedding", [8, 12, 16]))
    else:
        raise ValueError(f"no search space for {family}")
    return c
