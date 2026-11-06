"""Fix-test data (past-window context, robustness columns) and decision rules (plan: fix77_plan.md)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from framework.training.csv_v3 import arms, fix77

ROOT = Path(__file__).resolve().parents[1]
S = 10**9


def _ctx(device, start, end, duration, source):
    return fix77.context_features(np.array(device), np.array(start, np.int64) * S, np.array(end, np.int64) * S,
                                  np.array(duration, float), np.array(source, float))


def test_context_uses_only_past_windows_of_the_same_device():
    m = len(fix77.CTX_SOURCES)
    # rows: A[0,10] A[10,20] A[20,30] future A[25,60] other device B[10,20]; value = row id in every source column
    device = ["A", "A", "A", "A", "B"]
    start, end, dur = [0, 10, 20, 25, 10], [10, 20, 30, 60, 20], [10, 10, 10, 35, 10]
    src = np.repeat(np.arange(5, dtype=float)[:, None], m, axis=1)
    out = _ctx(device, start, end, dur, src)
    has, share = fix77.CTX.index("ctx_has_past"), fix77.CTX.index("ctx_same_duration_share")
    assert np.isnan(out[0, :m]).all() and out[0, has] == 0 and np.isnan(out[0, share])   # no past window
    assert np.allclose(out[1, :m], 0) and out[1, has] == 1 and out[1, share] == 1         # only A[0,10]
    assert np.allclose(out[2, :m], 0.5)                                                  # A[0,10], A[10,20]
    assert np.allclose(out[3, :m], 0.5) and out[3, share] == 0                           # A[20,30] ends after 25
    assert np.isnan(out[4, :m]).all() and out[4, has] == 0                               # device B has no past


def test_context_lookback_and_nan_sources():
    m = len(fix77.CTX_SOURCES)
    src = np.array([[np.nan] * m, [2.0] * m, [4.0] * m])
    out = _ctx(["A"] * 3, [0, 5, 100], [5, 10, 105], [5, 5, 5], src)
    assert np.isnan(out[1, :m]).all() and out[1, fix77.CTX.index("ctx_has_past")] == 1   # past exists, value NaN
    assert np.isnan(out[2, :m]).all() and out[2, fix77.CTX.index("ctx_has_past")] == 0   # beyond 30 s


def test_context_refuses_zero_length_windows():
    with pytest.raises(ValueError):
        _ctx(["A"], [5], [5], [0], np.zeros((1, len(fix77.CTX_SOURCES))))


def test_meta_reads_no_label_columns_and_checks_order(tmp_path):
    import pandas as pd
    path = tmp_path / "x.csv.gz"
    pd.DataFrame({"source_row_id": [7, 8], "device_name": ["d", "d"], "label4": ["a", "b"], "label_full": ["x", "y"],
                  "timestamp_start": ["2025-01-01 00:00:00", "2025-01-01 00:00:05"],
                  "timestamp_end": ["2025-01-01 00:00:05", "2025-01-01 00:00:10"]}).to_csv(path, index=False)
    info = fix77.meta("train", path)
    assert set(info) == {"source_row_id", "device", "start_ns", "end_ns"}
    assert not any(any(f in c for f in fix77.FORBIDDEN_META) for c in fix77.META_COLS)
    fix77.check_order(info, np.array([7, 8]))
    with pytest.raises(RuntimeError):
        fix77.check_order(info, np.array([8, 7]))


def test_arm_specs_and_weights():
    v3 = ["a", "b"]
    s = fix77.arm_specs(v3, {"ctx": ["c"], "rob": ["r"]})
    assert s["C0"] == {"names": v3} and s["CTX"] == {"names": v3 + ["c"]}
    assert s["ROB"]["row_weight"] == s["CTXROB"]["row_weight"] == {"kind": "scenario", "rho": 1.0}
    assert s["CTXROB"]["names"] == v3 + ["c", "r"]
    live = fix77.arm_specs(v3, {"ctx": fix77.CTX_LIVE})
    assert set(live) == {"C0", "CTX"} and "row_weight" not in live["CTX"]


def test_live_context_uses_exact_live_columns_only():
    from framework.training.csv_v3 import live77
    for name in fix77.CTX_LIVE:
        src = fix77.CTX_SOURCES.get(name)
        if src not in (None, "pps"):
            assert live77.RATING[live77.group_of(src)] == "exact", name


def test_choose_rules():
    r = lambda J, ok: {"J": J, "admitted": ok}
    assert fix77.choose({"CTX": r(0.9, False), "ROB": r(0.95, False), "CTXROB": r(0.97, False)}) is None
    assert fix77.choose({"CTX": r(0.9, True), "ROB": r(0.95, False), "CTXROB": r(0.9505, True)}) == "CTX"
    assert fix77.choose({"CTX": r(0.9, True), "ROB": r(0.95, False), "CTXROB": r(0.952, True)}) == "CTXROB"
    assert fix77.choose({"CTX": r(0.9, False), "ROB": r(0.95, False), "CTXROB": r(0.9505, True)}) is None
    assert fix77.choose({"CTX": r(0.96, True), "ROB": r(0.95, True), "CTXROB": r(0.5, False)}) == "CTX"


def test_admission_edge_cases():
    base = {"gains": [0.002] * 3, "gain": 0.002, "gain_sd": 0.0009, "unseen_gain": 0.0, "oof_f1_change": -0.005}
    assert arms.admitted(base)
    assert not arms.admitted({**base, "gain_sd": 0.001})            # gain must exceed 2 x SD
    assert not arms.admitted({**base, "unseen_gain": -1e-9})
    assert not arms.admitted({**base, "oof_f1_change": -0.0051})
    assert not arms.admitted({**base, "gains": [0.002] * 2})        # every seed run


def test_ping_merge_and_focus():
    classes = ["benign", "recon_scan", "recon_ping-sweep", "malware_mirai-syn-flood", "malware_mirai-udp-flood",
               "mitm_arp-spoofing", "mitm_impersonation"]
    p = np.array([[0.5, 0.3, 0.2, 0, 0, 0, 0], [0.1, 0.4, 0.5, 0, 0, 0, 0], [0.6, 0.0, 0.4, 0, 0, 0, 0]])
    y = np.array([2, 2, 2])
    q, ym = fix77.ping_merge(p, y, classes)
    assert (ym == 1).all() and q[:, 2].sum() == 0 and np.allclose(q[0, 1], 0.5)
    f = fix77.focus_metrics(p, y, classes)
    assert f["recall_recon_ping-sweep"] == pytest.approx(1 / 3) and f["ping_to_benign"] == 2
    assert f["acc_ping_merged"] == pytest.approx(1 / 3)       # row 0: tie 0.5/0.5 -> benign (argmax first)


def test_row_weights_leave_the_loss_unchanged_when_absent():
    import torch
    from framework.training.tabm_models import member_loss
    g = torch.Generator().manual_seed(0)
    logits, labels = torch.randn(6, 4, 5, generator=g), torch.tensor([0, 1, 2, 3, 4, 0])
    base = member_loss(logits, labels, False, None, 1., 0.1)
    assert torch.equal(base, member_loss(logits, labels, False, None, 1., 0.1, None))
    assert torch.allclose(base, member_loss(logits, labels, False, None, 1., 0.1, torch.ones(6)))
    w = torch.tensor([2., 0.5, 1., 1., 1., 1.])
    assert not torch.allclose(base, member_loss(logits, labels, False, None, 1., 0.1, w))


@pytest.mark.parametrize("module", ["experiment.train_fix77_tabm", "experiment.train_fix77_live"])
def test_drivers_refuse_other_studies_and_feature_groups(module):
    import os
    env = {**os.environ, "CASCADE_FEATURE_GROUPS": '["core"]', "CASCADE_STUDY": "artifacts/training/elsewhere"}
    proc = subprocess.run([sys.executable, "-c", f"import {module}"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode != 0 and "refusing" in proc.stderr
