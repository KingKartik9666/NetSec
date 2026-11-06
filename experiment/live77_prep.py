"""Live 77-class model, steps 4-8: feature groups and window-length evidence. Reads arrays only; fits nothing.

Run: .venv/Scripts/python.exe -m experiment.live77_prep
Writes ml_pipeline/live77_prep.json and figures 04/05 under ml_pipeline/figures.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from framework.training.csv_v3 import live77, prep

sys.path.insert(0, str(prep.ROOT / "ml_pipeline"))
import guard  # noqa: E402

D = prep.DEFAULT_OUT
OUT = prep.ROOT / "ml_pipeline"


def main() -> None:
    m = json.loads((D / "manifest.json").read_text(encoding="utf-8"))
    names = m["feature_names"]
    g = live77.groups()
    assert sum(len(v) for v in g.values()) == len(names) == 445
    X_tr = np.load(D / "X_train.npy", mmap_mode="r")
    X_va = np.load(D / "X_validation.npy", mmap_mode="r")
    fit = np.load(D / "fit77_train.npy")
    y_tr, y_va = np.load(D / "y77_train.npy"), np.load(D / "y77_validation.npy")
    roles = np.load(D / "validation_roles.npy")
    d_col = names.index("window_duration_seconds")
    fit_rows = np.flatnonzero(fit)
    dur_tr = np.asarray(X_tr[fit_rows, d_col])
    dur_va = np.asarray(X_va[:, d_col])

    # Missing share per group on fitting rows (NaN = statistic absent, e.g. no TCP packets).
    sample = np.sort(np.random.default_rng(0).choice(fit_rows, 50_000, replace=False))
    Xs = np.asarray(X_tr[sample])
    group_rows = {}
    for name, cols in g.items():
        idx = [names.index(c) for c in cols]
        group_rows[name] = {"features": len(cols), "rating": live77.RATING[name],
                            "nan_share_fit_sample": float(np.isnan(Xs[:, idx]).mean()) if idx else None}

    durations = list(range(1, 11))
    per_duration = {}
    for d in durations:
        tr = dur_tr == d
        tu = (roles == 0) & (dur_va == d)
        per_duration[d] = {"fit_rows": int(tr.sum()), "fit_classes": int(np.unique(y_tr[fit_rows][tr]).size),
                           "tuning_rows": int(tu.sum()), "tuning_classes": int(np.unique(y_va[tu]).size),
                           "tuning_benign_share": float(np.mean(y_va[tu] == 0)) if tu.any() else None}
    sets = {
        "exact": list(live77.EXACT),
        "exact+approximate": list(live77.EXACT + live77.APPROXIMATE),
        "exact+approximate+mismatch": list(live77.EXACT + live77.APPROXIMATE + live77.MISMATCH),
    }
    summary = {"groups": group_rows, "sets": {k: len(live77.columns(v)) for k, v in sets.items()},
               "per_duration": per_duration, "fit_rows": int(fit.sum())}
    (OUT / "live77_prep.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    plt = guard._plt()
    colors = {"exact": "#2a9d8f", "approximate": "#e9c46a", "mismatch": "#e76f51", "excluded": "#8d99ae"}
    order = sorted(g, key=lambda k: (list(colors).index(live77.RATING[k]), -len(g[k])))
    f, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(order, [len(g[k]) for k in order], color=[colors[live77.RATING[k]] for k in order])
    for i, k in enumerate(order):
        ax.text(len(g[k]) + 1, i, f"{len(g[k])}  ({live77.RATING[k]})", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("v3 features")
    ax.set_title("Live Npcap match of the 445 v3 features")
    exact_n = summary["sets"]["exact"]
    approx_n = summary["sets"]["exact+approximate"]
    guard.fig(4, "live_feature_groups", f,
              f"Each v3 feature is in one group. {exact_n} match the dataset exactly once the Rust extractor copies the "
              f"dataset conventions (core, port presence); approximate groups bring the candidate set to {approx_n}. "
              "Time-delta and protocol names measure something different on one PC, and logs/artifacts are "
              "impossible live, so those groups are not used in the live model.")

    f, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.bar([d - 0.2 for d in durations], [per_duration[d]["fit_rows"] for d in durations], 0.4,
            label="fitting rows", color="#264653")
    ax2 = ax1.twinx()
    ax2.bar([d + 0.2 for d in durations], [per_duration[d]["tuning_rows"] for d in durations], 0.4,
            label="tuning rows", color="#e9c46a")
    for d in durations:
        ax2.text(d + 0.2, per_duration[d]["tuning_rows"], str(per_duration[d]["tuning_classes"]),
                 ha="center", va="bottom", fontsize=7)
    ax1.set_xlabel("window length (s)")
    ax1.set_ylabel("fitting rows")
    ax2.set_ylabel("tuning rows (label = classes covered)")
    ax1.set_xticks(durations)
    ax1.set_title("Rows and class coverage per window length")
    f.legend(loc="upper right", fontsize=8)
    t10 = per_duration[10]
    guard.fig(5, "live_window_lengths", f,
              f"One model is trained on every 1-10 s window with the window length as an input. The live window "
              f"length is then chosen by tuning-role results per length. 10 s windows have {t10['fit_rows']:,} "
              f"fitting rows and {t10['tuning_rows']:,} tuning rows covering {t10['tuning_classes']} classes, so "
              "per-length scores on short windows rest on more rows than long ones.")


if __name__ == "__main__":
    main()
