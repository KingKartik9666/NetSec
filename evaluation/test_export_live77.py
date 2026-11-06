"""Live 77-class SHA-pinned bundle export (live77 integration plan step 3; owner decisions of 2026-11-03).

The exporter runs twice into temporary folders, once under an audit hook that records every file it opens. Only
validation-role rows (roles 0/1/2) are read; the held-out test partition is never opened.
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.native_export import export_live77 as ex

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts/training/live77_20261031_v1/jobs/77_live_final/attempt_0/model"
RESULT = ROOT / "artifacts/training/live77_20261031_v1/jobs/77_live_final/result.json"
SPEC = ROOT / "artifacts/training/class_merge_20261101/spec.json"
RULES = ROOT / "artifacts/training/upgrades2_live_20261101_v1/live_output_rules.json"
CONTRACT = ROOT / "framework/training/csv_v3/live77_contract.json"
DECISIONS = ROOT / "artifacts/app_v0_6_20261102/live77_decisions.json"
SHIPPED = ROOT / "artifacts/application/live_models_v1/live77_xgboost"
CHECKPOINT_SHA = "2ebc7f2da863d79d65b03ed785752b6b48d1bd273f0f446fc9235d059ac272e3"
RULES_VALUE_SHA = "b29e6a19fb6e1dad478358ea92cedd850746467ef72d37b3454ca734de2c4e46"
# Copy of the Live.tsx caveat filter; the label and the provenance note must survive it.
CAVEAT = r"unverif|untested|provisional|limitation|caveat|experimental|accuracy_verified|not[ _-]?verified|unvalidated"
DATA_ALLOWED = {"X_validation.npy", "y77_validation.npy", "row_ids_validation.npy", "validation_roles.npy", "manifest.json"}
# The audit hook sees only opens made through Python. Opens inside native code (xgboost's C++ Booster.load_model) are
# invisible to it, so Booster.load_model is wrapped to record its path as well; the exporter's guard() also covers it.
AUDIT = """import json, os, runpy, sys
import xgboost
opened, loaded = [], []
def hook(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        opened.append(os.fsdecode(args[0]))
load_model = xgboost.Booster.load_model
def recording_load_model(self, fname):
    loaded.append(os.fsdecode(fname) if isinstance(fname, (str, bytes, os.PathLike)) else repr(type(fname)))
    return load_model(self, fname)
xgboost.Booster.load_model = recording_load_model
out, log = sys.argv[1], sys.argv[2]
sys.addaudithook(hook)
sys.argv = ['export_live77', '--output', out]
try:
    runpy.run_module('tools.native_export.export_live77', run_name='__main__')
finally:
    if log:
        with open(log, 'w', encoding='utf-8') as f:
            json.dump(dict(opened=opened, loaded=loaded), f)
"""


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _run(out, log=None):
    return subprocess.run([sys.executable, "-c", AUDIT, str(out), str(log or "")], cwd=ROOT, capture_output=True, text=True)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("live77_export")
    a, b, log = base / "run_a", base / "run_b", base / "opened.json"
    for out, lg in ((a, log), (b, None)):
        proc = _run(out, lg)
        assert proc.returncode == 0, proc.stderr[-4000:]
    log = _json(log)
    return dict(a=a, b=b, opened=log["opened"], loaded=log["loaded"], base=base)


def test_guard_refuses_test_partition_paths():
    for p in ("dataset/experiments/csv_v3_cascade_20261030_v1/X_test.npy", "y77_test.npy", "row_ids_test.npy",
              "selected_test_77class.csv.gz", "test/X.npy"):
        with pytest.raises(ValueError, match="held-out test"):
            ex.guard(ROOT / p)
    assert ex.guard(ex.DATA / "X_validation.npy") == ex.DATA / "X_validation.npy"


def test_export_is_byte_stable_and_refuses_an_existing_folder(runs):
    for name in ("bundle.json", "golden_live77.json", "model.ubj"):
        assert (runs["a"] / name).read_bytes() == (runs["b"] / name).read_bytes(), name
    assert sorted(p.name for p in runs["a"].iterdir()) == ["bundle.json", "golden_live77.json", "model.ubj"]
    again = _run(runs["a"])
    assert again.returncode != 0 and "new folder" in again.stderr


def test_export_opens_only_validation_arrays_and_the_model(runs):
    opened = {Path(p).resolve() for p in runs["opened"]}
    inside = {p for p in opened if ROOT in p.parents and (ROOT / ".venv") not in p.parents}
    data = {p.name for p in inside if ex.DATA in p.parents}
    assert data <= DATA_ALLOWED and {"X_validation.npy", "y77_validation.npy", "validation_roles.npy"} <= data
    assert not any(re.search(r"(^|[^a-z])test", p.name, re.I) for p in inside), sorted(inside)
    assert (MODEL_DIR / "model.ubj").resolve() in opened  # digest() opens it from Python
    assert not any(p.suffix == ".npy" and "train" in p.name for p in opened)
    # native-code load: the only model xgboost loads is the pinned checkpoint
    assert [Path(p).resolve() for p in runs["loaded"]] == [(MODEL_DIR / "model.ubj").resolve()]


def test_export_refuses_output_under_protected_folders(tmp_path):
    assert ROOT / "dataset" in ex.PROTECTED
    for target in (ROOT / "dataset" / "live77_export_probe", ex.DATA / "live77_export_probe",
                   ROOT / "framework/models/live77_export_probe", ROOT / "artifacts/training/live77_export_probe"):
        proc = subprocess.run([sys.executable, "-m", "tools.native_export.export_live77", "--output", str(target)],
                              cwd=ROOT, capture_output=True, text=True)
        assert proc.returncode != 0 and "new folder" in proc.stderr, target
        assert not target.exists(), target


def test_export_refuses_a_contract_whose_provenance_may_not_ship(monkeypatch):
    import tempfile
    c = _json(CONTRACT)
    c["provenance_review"]["ship_allowed"] = False
    folder = tempfile.TemporaryDirectory(prefix="live77_contract_")  # pytest's tmp_path names the test, which guard() refuses
    held = Path(folder.name) / "contract.json"
    held.write_text(json.dumps(c), encoding="utf-8")
    monkeypatch.setattr(ex, "CONTRACT", held)
    with pytest.raises(ValueError, match="provenance"):
        ex.sources()
    c["provenance_review"]["ship_allowed"], c["label"] = True, "Live XGBoost (77-class, 1 s windows)"
    held.write_text(json.dumps(c), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        ex.sources()
    monkeypatch.undo()
    folder.cleanup()


def test_bundle_matches_schema_spec_rules_and_decisions(runs):
    bundle_path = runs["a"] / "bundle.json"
    assert bundle_path.stat().st_size <= 1024 * 1024  # models.rs size bound
    b, schema, spec = _json(bundle_path), _json(MODEL_DIR / "model_schema.json"), _json(SPEC)
    contract, dec, rules = _json(CONTRACT), _json(DECISIONS), _json(RULES)
    d = dec["defaults_chosen_by_app_chat"]
    assert (b["version"], b["model_id"], b["kind"], b["benign_index"]) == (1, d["model_id"], "xgboost", 0)
    assert b["feature_names"] == schema["feature_names"] == contract["feature_names"] and len(b["feature_names"]) == 252
    assert b["classes"] == schema["classes"] == spec["classes"] and len(b["classes"]) == 77
    m = b["class_merge"]
    assert (m["source_class"], m["target_class"], m["source_index"], m["target_index"]) == ("recon_ping-sweep", "recon_scan", 71, 70)
    assert b["classes"][71] == "recon_ping-sweep" and b["classes"][70] == "recon_scan"
    assert m["spec_sha256"] == _sha(SPEC) and (m["name"], m["version"]) == (spec["name"], spec["version"])
    assert {k: rules["value"]["class_merge"][k] for k in ("name", "version", "source_class", "target_class", "rule")} == \
           {k: m[k] for k in ("name", "version", "source_class", "target_class", "rule")}
    cal = rules["value"]["calibration"]
    assert b["calibration"] == {"kind": "temperature", "temperature": cal["temperature"], "alpha": cal["alpha"], "log_floor": 1e-7}
    assert rules["value"]["logit_adjustment"] is None and b["logit_adjustment"] is None
    assert b["output_rules"]["value_sha256"] == rules["sha256"] == RULES_VALUE_SHA and b["output_rules"]["file_sha256"] == _sha(RULES)
    assert (b["checkpoint"], b["checkpoint_sha256"], b["bytes"]) == ("model.ubj", CHECKPOINT_SHA, 5165479)
    assert _sha(runs["a"] / "model.ubj") == CHECKPOINT_SHA == schema["checkpoint_sha256"]
    assert b["window_seconds"] == dec["window"]["length_seconds"] == contract["window"]["length_seconds"] == 10  # owner: 10 s
    assert b["feature_contract"]["sha256"] == _sha(CONTRACT) and b["feature_contract"]["feature_names_sha256"] == contract["feature_names_sha256"]
    assert (b["test_evaluated"], b["live_validation"], b["publisher_equivalence_verified"]) == (False, False, False)
    assert b["output"]["binary_attack_probability"] is None
    assert b["golden"] == {"path": "golden_live77.json", "sha256": _sha(runs["a"] / "golden_live77.json"), "rows": 128,
                           "merged_comparison": "merged is float64; compare within 1e-6, not bit-for-bit"}


def test_bundle_provenance_is_honest_and_survives_the_live_caveat_filter(runs):
    b, contract, dec = _json(runs["a"] / "bundle.json"), _json(CONTRACT), _json(DECISIONS)
    d = dec["defaults_chosen_by_app_chat"]
    assert (b["label"], b["provenance_note"]) == (contract["label"], contract["provenance_note"])
    pr = b["provenance_review"]
    assert pr == contract["provenance_review"] and pr["ship_allowed"] is True  # step-4 loader must also refuse False
    assert b["label"] == "Live XGBoost (77-class, 10 s windows)" and b["window_seconds"] == 10
    assert not re.search(r"\b1[ -]?s(econd)?\b", b["label"] + " " + b["provenance_note"])  # no stale 1 s claim
    assert "1-10 s" in b["provenance_note"] and "scored here on 10 s windows" in b["provenance_note"]
    if pr["status"] == "corrected_under_owner_window_override":  # app-chat default texts kept verbatim, not shipped
        assert (pr["superseded"]["label"], pr["superseded"]["provenance_note"]) == (d["label"], d["provenance_note"])
        assert pr["owner_wording_confirmed"] is False
    for t in (b["label"], b["provenance_note"]):
        assert not re.search(CAVEAT, t, re.I), t
    assert "held-out test was not used" in b["provenance_note"]


def _golden(runs):
    return _json(runs["a"] / "golden_live77.json"), _json(runs["a"] / "bundle.json")


def test_golden_rows_are_validation_only_and_cover_the_edge_cases(runs):
    g, b = _golden(runs)
    rows = g["rows"]
    assert (g["partition"], g["test_loaded"], g["roles_used"], g["xgboost_version"]) == ("validation", False, [0, 1, 2], "3.1.3")
    assert len(rows) == 128 and len({r["validation_row"] for r in rows}) == 128
    assert {r["role"] for r in rows} == {0, 1, 2}
    roles = np.load(ex.DATA / "validation_roles.npy")
    y = np.load(ex.DATA / "y77_validation.npy")
    ids = np.load(ex.DATA / "row_ids_validation.npy")
    for r in rows:
        i = r["validation_row"]
        assert roles[i] == r["role"] and y[i] == r["label_index"] and str(ids[i]) == r["row_id"]
    tags = {t for r in rows for t in r["tags"]}
    assert {"raw_argmax_ping_sweep", "merge_flip", "zero_packets", "src_ports_capped", "dst_ports_capped",
            "most_missing", "window_10s", "stratified"} <= tags
    assert any(np.argmax(r["raw"]) == 71 for r in rows)
    assert any(r["decision"] != r["raw_decision"] for r in rows)
    assert all(len(r["values"]) == 252 for r in rows) and any(None in r["values"] for r in rows)
    name = b["feature_names"].index
    assert any(r["values"][name("window_duration_seconds")] == 10 for r in rows)
    assert any(r["values"][name("quality_no_packets_observed")] == 1 for r in rows)


def test_golden_pipeline_reproduces_in_python(runs):
    import xgboost as xgb
    from framework.training.csv_v3 import calibrate, class_merge
    g, b = _golden(runs)
    rows, classes = g["rows"], b["classes"]
    X = np.array([[np.nan if v is None else v for v in r["values"]] for r in rows], dtype=np.float32)
    booster = xgb.Booster()
    booster.load_model(str(runs["a"] / "model.ubj"))
    booster.set_param({"nthread": 1})
    dm = xgb.DMatrix(X, missing=np.nan)
    raw, margins = booster.predict(dm), booster.predict(dm, output_margin=True)
    contribs = booster.predict(dm, pred_contribs=True)
    assert np.array_equal(raw, np.array([r["raw"] for r in rows], dtype=np.float32))
    assert np.array_equal(margins, np.array([r["margins"] for r in rows], dtype=np.float32))
    cal = calibrate.apply(b["calibration"], raw)
    assert np.array_equal(cal, np.array([r["calibrated"] for r in rows], dtype=np.float32))
    merged = class_merge.merge_proba(cal, classes)
    assert np.array_equal(merged, np.array([r["merged"] for r in rows]))
    # number types: merged holds float64 sums of float32 calibrated values; a float32 replay compares within 1e-6
    golden_merged, golden_cal = np.array([r["merged"] for r in rows]), np.array([r["calibrated"] for r in rows], dtype=np.float32)
    assert merged.dtype == np.float64 and cal.dtype == np.float32
    assert np.array_equal(golden_merged[:, 70], golden_cal[:, 70].astype(np.float64) + golden_cal[:, 71].astype(np.float64))
    f32 = golden_cal.copy()
    f32[:, 70] += f32[:, 71]
    f32[:, 71] = 0
    assert np.abs(golden_merged - f32).max() <= g["tolerance"]["probability"] == 1e-6
    assert g["number_types"]["merged"].startswith("float64") and "not bit-for-bit" in g["number_types"]["merged"]
    assert all(g["number_types"][k] == "float32" for k in ("values", "raw", "margins", "calibrated", "contributions"))
    assert np.abs(np.exp(margins - margins.max(1, keepdims=True)) / np.exp(margins - margins.max(1, keepdims=True)).sum(1, keepdims=True) - raw).max() <= 1e-6
    for i, r in enumerate(rows):
        assert r["calibrated_decision"] == r["raw_decision"] == int(np.argmax(raw[i]))  # temperature keeps the order
        assert r["decision"] == int(np.argmax(merged[i])) and r["decision_class"] == classes[r["decision"]]
        assert r["merge_flip"] == (r["decision"] != r["raw_decision"])
        assert merged[i][71] == 0 and abs(merged[i].sum() - 1) < 1e-5
        want = {str(r["decision"]), "70", "71"}
        assert set(r["contributions"]) == want
        for k, v in r["contributions"].items():
            assert len(v) == 253 and np.array_equal(np.array(v, dtype=np.float32), contribs[i, int(k)])
            assert abs(sum(v) - r["margins"][int(k)]) <= 1e-4
    assert g["shap_reconstruction_max_abs"] <= 1e-4


def test_golden_accuracy_replays_result_json_role_metrics(runs):
    g, _ = _golden(runs)
    report = _json(RESULT)["report"]
    for role, name in ((0, "tuning"), (1, "calibration"), (2, "assessment")):
        rep = g["role_replay"][name]
        assert rep["role"] == role and rep["rows"] == report[name]["rows"]
        assert rep["raw_accuracy"] == report[name]["accuracy"]  # CPU xgboost 3.1.3 replays the study exactly
        assert rep["merged_accuracy"] >= rep["raw_accuracy"]  # the merge moves accuracy in the expected direction
    assert g["study_replay"]["raw_max_abs_difference"] == 0 and g["study_replay"]["calibrated_max_abs_difference"] == 0
    for role in (0, 1, 2):
        sel = [r for r in g["rows"] if r["role"] == role]
        acc = np.mean([r["raw_decision"] == r["label_index"] for r in sel])
        assert g["golden_accuracy"][str(role)]["raw"] == acc


def test_shipped_bundle_is_the_reproducible_export(runs):
    assert (SHIPPED / "bundle.json").exists(), "run python -m tools.native_export.export_live77 first"
    for name in ("bundle.json", "golden_live77.json", "model.ubj"):
        assert (SHIPPED / name).read_bytes() == (runs["a"] / name).read_bytes(), name
