"""Checks for composing the 77-class primary and the 8-class specialist into 84-class decisions (no training)."""
import numpy as np
import pytest

from framework.training.csv_v3 import cascade, prep, runner

needs_data = pytest.mark.skipif(not (prep.DEFAULT_OUT / "manifest.json").exists(), reason="v3 dataset not prepared")


@needs_data
def test_compose_routes_only_recon_scan_rows_and_maps_back_to_84(monkeypatch):
    m = runner.manifest()
    to77, to8 = np.array(m["map84_to77"]), np.array(m["map84_to8"])
    y84 = runner.task_arrays("84")["y_val"]
    n = len(y84)
    p77 = np.eye(77, dtype=np.float32)[to77[y84]]            # perfect stage 1 on the merged task
    routed_truth = to8[y84]
    calls = {}

    def fake_stage2(r, rows):
        calls["rows"] = rows
        k = np.where(routed_truth[rows] >= 0, routed_truth[rows], 0)
        return np.eye(8, dtype=np.float32)[k]                 # perfect specialist on true recon rows

    monkeypatch.setattr(cascade, "_stage1_proba", lambda r: p77)
    monkeypatch.setattr(cascade, "_stage2_proba", fake_stage2)
    monkeypatch.setattr(runner, "selection_value", lambda r: 0.0)
    proba, info = cascade.compose({"job_id": "s1"}, {"job_id": "s2"})
    recon_scan = m["classes77"].index(prep.MERGED)
    assert np.array_equal(calls["rows"], np.flatnonzero(to77[y84] == recon_scan))
    assert info["rows_routed_to_specialist"] == int((to77[y84] == recon_scan).sum())
    assert np.array_equal(proba.argmax(1), y84)               # perfect stages -> perfect 84-class decisions
    assert proba.shape == (n, 84)


@needs_data
def test_compose_never_emits_a_recon_subtype_without_routing(monkeypatch):
    m = runner.manifest()
    to8 = np.array(m["map84_to8"])
    rng = np.random.default_rng(0)
    p77 = rng.dirichlet(np.ones(77), size=30000).astype(np.float32)
    monkeypatch.setattr(cascade, "_stage1_proba", lambda r: p77)
    monkeypatch.setattr(cascade, "_stage2_proba", lambda r, rows: rng.dirichlet(np.ones(8), size=len(rows)).astype(np.float32))
    monkeypatch.setattr(runner, "selection_value", lambda r: 0.0)
    proba, _ = cascade.compose({"job_id": "s1"}, {"job_id": "s2"})
    recon_scan = m["classes77"].index(prep.MERGED)
    subtype = to8[proba.argmax(1)] >= 0
    assert np.array_equal(subtype, p77.argmax(1) == recon_scan)


def test_candidates_skip_results_the_current_controller_never_calibrated(monkeypatch, tmp_path):
    # Hybrids written before duplicate-member dedup were never recalibrated; they must not compete on a raw value.
    import json
    jobs = {"77_hybrid_tuned_combo_a+b": {"tune_value": 0.99},
            "77_hybrid_tuned_ab": {"tune_value": 0.98, "calibration": {"status": "complete", "selection_value": 0.98}},
            "77_hem": {"tune_value": 0.90, "calibration": {"status": "not_calibratable"}}}
    for job, extra in jobs.items():
        (tmp_path / "jobs" / job).mkdir(parents=True)
        (tmp_path / "jobs" / job / "result.json").write_text(
            json.dumps({"job_id": job, "status": "complete", "report": {}, **extra}), encoding="utf-8")
    monkeypatch.setattr(runner, "STUDY", tmp_path)
    assert sorted(r["job_id"] for r in cascade._candidates("77")) == ["77_hem", "77_hybrid_tuned_ab"]
    assert cascade.best("77")["job_id"] == "77_hybrid_tuned_ab"
