"""U1 output rule: ``recon_ping-sweep`` merged into ``recon_scan`` (spec, Python reference, golden writer).

Plan: ml_pipeline/notes_20261101/upgrades2_plan.md (U1) and fix77_plan.md (Fix A). Import-safe.

Rule, on one 77-column probability vector p (raw or calibrated; whatever the app displays):

    p'[recon_scan]       = p[recon_scan] + p[recon_ping-sweep]
    p'[recon_ping-sweep] = 0
    p'[c]                = p[c] for every other class
    decision             = argmax p' (lowest index on ties, as numpy and the app's argmax)

Labels are mapped the same way for scoring only (ping-sweep -> recon_scan). p' still sums to 1. The rule is applied
after calibration because it must act on the displayed probabilities.

    python -m framework.training.csv_v3.class_merge golden     # writes artifacts/training/class_merge_20261101/

The golden writer is CPU-only and read-only on its sources: 32 validation rows from the TUNING role (role 0), with
before/after probabilities of the shipped TabM-NE and the default blend (``val_proba_cal.npy``), plus sha256 of
every source. It never reads the calibration or assessment roles' labels, and never the held-out test.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

SOURCE, TARGET = "recon_ping-sweep", "recon_scan"
N_CLASSES = 77
SPEC = {
    "name": "ping_sweep_into_recon_scan", "version": 1, "n_classes": N_CLASSES,
    "source_class": SOURCE, "target_class": TARGET,
    "rule": "p[target] += p[source]; p[source] = 0; other classes unchanged; decision = argmax (lowest index on ties)",
    "applies_to": "the 77-class probability vector the app displays (after calibration; raw if the model has none)",
    "label_mapping_for_scoring": {SOURCE: TARGET},
    "arithmetic": "float64 on the stored float32 probabilities",
    "plan": "ml_pipeline/notes_20261101/upgrades2_plan.md (U1); fix77_plan.md (Fix A)",
}
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/training/class_merge_20261101"
CASCADE = ROOT / "artifacts/training/cascade_v3_20261030_v1"
SOURCES = {"tabm_ne": CASCADE / "jobs/77_tune_tabm_ne_t008/val_proba_cal.npy",
           "default_blend_tabm_xgboost_kan": CASCADE / "jobs/77_hybrid_tuned_tabm_xgboost_kan/val_proba_cal.npy"}
GOLDEN_ROWS = 32


def indices(classes: list[str]) -> tuple[int, int]:
    classes = list(classes)
    if len(classes) != N_CLASSES:
        raise ValueError(f"the merge is defined for the {N_CLASSES}-class catalog, got {len(classes)} classes")
    return classes.index(SOURCE), classes.index(TARGET)


def merge_proba(p: np.ndarray, classes: list[str]) -> np.ndarray:
    """Merged copy of ``p`` (shape (77,) or (n, 77)); the input is never modified."""
    src, dst = indices(classes)
    q = np.array(p, dtype=np.float64, copy=True)
    if q.shape[-1] != N_CLASSES:
        raise ValueError(f"expected {N_CLASSES} probability columns, got {q.shape[-1]}")
    q[..., dst] += q[..., src]
    q[..., src] = 0.0
    return q


def merge_labels(y: np.ndarray, classes: list[str]) -> np.ndarray:
    src, dst = indices(classes)
    y = np.asarray(y)
    return np.where(y == src, dst, y)


def decision(p: np.ndarray, classes: list[str]) -> np.ndarray:
    return merge_proba(p, classes).argmax(-1)


# ------------------------------------------------------------------------------------------ golden writer
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def golden_rows(roles: np.ndarray, y: np.ndarray, probas: list[np.ndarray], classes: list[str],
                n: int = GOLDEN_ROWS, seed: int = 0) -> np.ndarray:
    """Tuning-role rows only: up to n/4 true ping-sweep rows, then the n/4 rows with the largest p(ping) in any
    model, then random tuning rows; all choices deterministic (seed 0); returned sorted."""
    src, _ = indices(classes)
    tune = np.flatnonzero(roles == 0)
    rng = np.random.default_rng(seed)
    pick: list[int] = []
    ping_rows = rng.permutation(tune[y[tune] == src])
    pick += [int(r) for r in ping_rows[: n // 4]]
    p_ping = np.max([p[tune, src] for p in probas], axis=0)
    for r in tune[np.argsort(-p_ping, kind="stable")]:
        if len(pick) >= n // 2:
            break
        if int(r) not in pick:
            pick.append(int(r))
    for r in rng.permutation(tune):
        if len(pick) >= n:
            break
        if int(r) not in pick:
            pick.append(int(r))
    return np.array(sorted(pick), dtype=np.int64)


def write_golden(out: Path = OUT) -> dict:
    from framework.training.csv_v3 import prep
    data = prep.DEFAULT_OUT
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    classes = list(manifest["classes77"])
    roles = np.load(data / "validation_roles.npy")
    y = np.load(data / "y77_validation.npy")
    row_ids = np.load(data / "row_ids_validation.npy", allow_pickle=False)
    probas = {k: np.load(p) for k, p in SOURCES.items()}
    for k, p in probas.items():
        if p.shape != (len(y), N_CLASSES):
            raise RuntimeError(f"{k}: shape {p.shape} does not match the validation rows")
    rows = golden_rows(roles, y, list(probas.values()), classes)
    if not np.all(roles[rows] == 0):
        raise RuntimeError("golden rows must come from the tuning role only")
    src, dst = indices(classes)
    tune = np.flatnonzero(roles == 0)
    effect = {}
    for k, p in probas.items():
        before = float(np.mean(p[tune].argmax(1) == y[tune]))
        after = float(np.mean(decision(p[tune], classes) == merge_labels(y[tune], classes)))
        effect[k] = {"tuning_accuracy_before": before, "tuning_accuracy_after_merge": after,
                     "tuning_rows": int(len(tune))}
    cases = []
    for r in rows:
        case = {"validation_row": int(r), "source_row_id": int(row_ids[r]), "role": "tuning",
                "true_class": classes[int(y[r])], "true_class_merged": classes[int(merge_labels(y[r:r + 1], classes)[0])]}
        for k, p in probas.items():
            q = merge_proba(p[r], classes)
            case[k] = {"before": [float(v) for v in p[r]], "after": [float(v) for v in q],
                       "decision_before": classes[int(np.argmax(p[r]))], "decision_after": classes[int(np.argmax(q))]}
        cases.append(case)
    files = {**{k: v for k, v in SOURCES.items()}, "validation_roles": data / "validation_roles.npy",
             "y77_validation": data / "y77_validation.npy", "row_ids_validation": data / "row_ids_validation.npy",
             "manifest": data / "manifest.json", "class_merge_py": Path(__file__)}
    shas = {k: {"path": p.relative_to(ROOT).as_posix(), "sha256": _sha256(p)} for k, p in files.items()}
    spec = dict(SPEC, source_index=src, target_index=dst, classes=classes,
                tuning_role_effect=effect, sources=shas, written=time.strftime("%Y-%m-%d %H:%M:%S"))
    golden = {"spec_name": SPEC["name"], "version": SPEC["version"], "source_index": src, "target_index": dst,
              "tolerance_abs": 1e-9, "rows": cases, "sources": shas,
              "selection": "tuning role only: true ping-sweep rows, highest p(ping) rows, random rows (seed 0)",
              "test_loaded": False}
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    (out / "golden.json").write_text(json.dumps(golden, indent=1), encoding="utf-8")
    return {"out": str(out), "rows": len(cases), "effect": effect,
            "decision_changes": {k: int(sum(c[k]["decision_before"] != c[k]["decision_after"] for c in cases))
                                 for k in probas}}


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args[:1] != ["golden"]:
        print("usage: python -m framework.training.csv_v3.class_merge golden")
        return 2
    print(json.dumps(write_golden(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
