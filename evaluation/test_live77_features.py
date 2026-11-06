"""Live 77-class feature groups and the runner's feature-subset guard."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from framework.training.csv_v3 import live77

ROOT = Path(__file__).resolve().parents[1]


def test_every_v3_feature_has_one_group():
    g = live77.groups()
    names = [n for v in g.values() for n in v]
    assert len(names) == len(set(names)) == 445
    assert {k: len(v) for k, v in g.items() if live77.RATING[k] == "exact"} == {"core": 58, "port_presence": 194}
    assert sum(len(g[k]) for k in live77.APPROXIMATE) == 23


def test_excluded_groups_are_refused_and_order_is_kept():
    with pytest.raises(ValueError):
        live77.columns(["core", "log"])
    cols = live77.columns(list(live77.EXACT))
    assert len(cols) == 252 and list(cols) == sorted(cols)
    assert not any(n.startswith("log_") for n in live77.feature_names(list(live77.EXACT)))


def test_feature_groups_need_their_own_study():
    env = {**os.environ, "CASCADE_FEATURE_GROUPS": json.dumps(["core"])}
    env.pop("CASCADE_STUDY", None)
    proc = subprocess.run([sys.executable, "-c", "import framework.training.csv_v3.runner"], cwd=ROOT, env=env,
                          capture_output=True, text=True)
    assert proc.returncode != 0 and "own CASCADE_STUDY" in proc.stderr


def test_scenario_weights_keep_class_totals():
    import numpy as np
    from framework.training.csv_v3 import runner
    a = runner.task_arrays("77")
    fit = np.flatnonzero(a["fit"])[::50]
    w = runner.scenario_weights({"kind": "scenario", "rho": 0.5}, fit, a["y_train"])
    y = a["y_train"][fit]
    assert w.shape == fit.shape and np.isfinite(w).all() and (w > 0).all()
    assert np.allclose(np.bincount(y, weights=w), np.bincount(y), rtol=1e-4)
    assert runner.scenario_weights(None, fit, a["y_train"]) is None


def test_window_bootstrap_f1_skips_absent_classes():
    # Subprocess: importing the driver sets CASCADE_STUDY, which must not leak into other tests.
    code = ("import numpy as np; from experiment.train_live77 import _common_f1 as f; "
            "y = np.array([0, 0, 1, 1]); common = np.array([0, 1, 2]); "
            "assert f(y, y, common, 3) == 1.0; assert f(y[:2], y[:2], common, 3) == 1.0; "
            "assert np.isnan(f(np.array([3]), np.array([3]), common, 4))")
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_derived_features_follow_dataset_conventions():
    import numpy as np
    from framework.training.csv_v3 import derived, live77
    names = list(live77._names())
    X = np.zeros((3, len(names)), dtype=np.float32)
    col = names.index
    X[0, col("network_packets_all_count")] = 0                       # no packets -> NaN
    X[1, col("network_packets_all_count")] = 4                       # packets, empty TTL family -> 0
    X[1, col("window_duration_seconds")] = 2
    X[2, col("network_packets_all_count")] = 5
    X[2, col("window_duration_seconds")] = 1
    X[2, [col("network_ttl_max"), col("network_ttl_min"), col("network_ttl_avg")]] = [126, 60, 100]
    X[2, [col("network_tcp-flags-syn_count"), col("network_tcp-flags-ack_count")]] = [3, 1]
    D = derived.compute(X, names)
    out = dict(zip(derived.NAMES, D[2]))
    assert np.isnan(D[0]).all() and (D[1] == 0).all()
    assert out["derived_ttl_hop_min"] == 2 and out["derived_ttl_hop_max"] == 68 and out["derived_ttl_mixed_init"] == 1
    assert out["derived_ratio_syn_per_ack"] == 1.5 and out["derived_range_ttl"] == 66


# ------------------------------------------------------------------ live77 extraction contract (plan step 1)
MODEL_DIR = ROOT / "artifacts/training/live77_20261031_v1/jobs/77_live_final/attempt_0/model"
DECISIONS = ROOT / "artifacts/app_v0_6_20261102/live77_decisions.json"
CHECKS = ROOT / "artifacts/app_v0_6_20261102/live77_convention_checks.json"
RULES = "artifacts/training/upgrades2_live_20261101_v1/live_output_rules.json"
NAMES_SHA = "d8c7b629b08393ef79db5bb9ea42cae1cff155f3cc81b8a5c211e41fc3c67bfd"
# Copy of the Live.tsx caveat filter; the provenance note must survive it.
CAVEAT = r"unverif|untested|provisional|limitation|caveat|experimental|accuracy_verified|not[ _-]?verified|unvalidated"
STATUSES = ("dataset_verified", "owner_decision", "dataset_verified_and_owner_decision",
            "plan_default_pending_owner", "implementer_default_pending_owner")


def _contract():
    return json.loads(live77.CONTRACT.read_text(encoding="utf-8"))


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from _strings(x)
    elif isinstance(v, list):
        for x in v:
            yield from _strings(x)


def test_contract_names_equal_model_schema():
    import hashlib
    c = _contract()
    schema = json.loads((MODEL_DIR / "model_schema.json").read_text(encoding="utf-8"))
    names = [f["name"] for f in c["features"]]
    assert names == schema["feature_names"] == c["feature_names"] and len(names) == 252
    assert [f["index"] for f in c["features"]] == list(range(252))
    assert hashlib.sha256("\n".join(names).encode()).hexdigest() == NAMES_SHA == c["feature_names_sha256"]
    assert c["model"]["checkpoint_sha256"] == schema["checkpoint_sha256"]
    assert names == live77.feature_names(list(live77.EXACT))  # schema order is the v3 order


def test_contract_groups_and_vocabularies_are_unchanged():
    c = _contract()
    assert {g: sum(f["group"] == g for f in c["features"]) for g in live77.EXACT} == {"core": 58, "port_presence": 194}
    assert all(f["group"] == live77.group_of(f["name"]) for f in c["features"])
    for d in ("src", "dst"):
        stem = f"network_ports_{d}__present_"
        ports = [int(f["name"][len(stem):]) for f in c["features"] if f["name"].startswith(stem) and not f["name"].endswith("UNSEEN")]
        assert ports == c["port_vocabulary"][d] and len(ports) == 96
    rust = json.loads((ROOT / "rust/netsec/crates/runtime/src/live/port_features.json").read_text(encoding="utf-8"))
    assert rust == [f["name"] for f in c["features"] if f["group"] == "port_presence" and not f["name"].endswith("UNSEEN")]


def test_every_contract_feature_has_a_definition_and_nan_rule():
    c = _contract()
    for f in c["features"]:
        assert f["definition"] in c["definitions"], f["name"]
        assert f["nan_rule"] in c["nan_rules"], f["name"]
        assert set(f.get("conventions", [])) <= set(c["conventions"]), f["name"]
    by = {f["name"]: f for f in c["features"]}
    assert by["network_fragmented-packets"]["params"] == {"ipv4_frag_mask": 0x2000}
    assert by["network_tcp-flags_max"]["params"]["bits"] == 12
    assert by["network_mss_min"]["definition"] == "stat.tcp.mss_option" and by["network_mss_min"]["nan_rule"] == "family_stat"
    assert by["network_ports_dst__present_UNSEEN"]["nan_rule"] == "port_unseen"
    assert by["window_duration_seconds"]["params"] == {"seconds": 10.0}
    assert all(c["conventions"][k]["status"] in STATUSES for k in c["conventions"])


def test_emulation_mss_rule_is_corrected():
    text = " ".join(live77.EMULATION.values())
    assert "contributes mss 0" not in text and "mss_absent_zero" not in live77.EMULATION
    assert "MSS-bearing" in live77.EMULATION["mss_bearing_only"]
    assert "arrival order" in live77.EMULATION["port_cap_1000"] and "12-bit" in live77.EMULATION["tcp_flags_12bit"]
    assert set(live77.EMULATION) <= set(_contract()["conventions"])


def test_contract_window_is_the_owner_10_second_window():
    c, dec = _contract(), _json(DECISIONS)
    w = c["window"]
    assert w["length_seconds"] == dec["window"]["length_seconds"] == 10 and w["kind"] == "tumbling"
    assert w["nominal_duration_seconds"] == 10.0 and w["display"] == dec["window"]["display"]
    assert (w["gate"]["nominal_seconds"], w["gate"]["tolerance_seconds"]) == (10.0, 1.0)  # existing packet.rs gate
    assert {f["params"]["seconds"] for f in c["features"] if f["definition"] == "window.nominal_seconds"} == {10.0}
    assert c["conventions"]["nominal_duration"]["rule"].startswith("window_duration_seconds = 10; per_second = count / 10")
    text = json.dumps(c)
    assert "count / 1.0" not in text and "length 1.0" not in text and "1 s tumbling" not in text


def test_contract_statuses_name_who_decided():
    c, dec = _contract(), _json(DECISIONS)
    owner_text = set(_strings(dec))
    for k, v in c["conventions"].items():
        if "owner_decision" in v["status"]:  # an owner decision quotes the decisions file verbatim
            assert v["decision"] in owner_text, k
        if v["status"].startswith("dataset_verified"):  # dataset evidence comes from the kept checks script
            assert v["evidence"] and v["evidence_source"] == CHECKS.relative_to(ROOT).as_posix(), k
    conv = c["conventions"]
    pending = sorted(k for k, v in conv.items() if v["status"].endswith("_pending_owner"))
    assert {"arp_ip_direction", "header_direction", "device_mac_packet_direction", "broadcast_not_admitted",
            "fragments_keep_stats"} <= set(pending)
    assert conv["fragments_keep_stats"]["status"] == "plan_default_pending_owner"
    adm = conv["device_mac_admission"]
    assert adm["status"] == "owner_decision" and not any(w in adm["rule"] for w in ("broadcast", "packets_src", "0.0.0.0"))
    assert c["damaged_window"]["status"]["family_damage"] == "implementer_default_pending_owner"
    signoff = c["pending_owner_signoff"]
    assert signoff["conventions"] == pending and "plan step 5" in signoff["blocks"]
    assert "damaged_window.family_damage" in signoff["other"]
    pr = c["provenance_review"]  # corrected wording no longer blocks the export; only its optional confirmation stays open
    assert ("provenance_review" in signoff["other"]) == (not pr["ship_allowed"])
    if pr["status"] == "corrected_under_owner_window_override":
        assert signoff["non_blocking"] == ["provenance_review.wording_confirmation"]


def test_contract_evidence_is_reproducible_and_validation_only():
    import re
    chk = _json(CHECKS)
    assert chk["script_sha256"] == _sha(ROOT / chk["generated_by"])
    assert not any(re.search(r"(^|[^a-z])test", Path(p).name, re.I) for p in chk["opened"])
    assert chk["partition"].startswith("validation") and chk["rows"] == 30000
    for d in ("src", "dst"):  # UNSEEN is never NaN in the data: a capped list always holds an unseen port
        p = chk["ports"][d]
        assert p["unseen_nan"] == 0 and p["unseen_one_in_capped_rows"] == p["capped_rows"] > 0
    rule = _contract()["nan_rules"]["port_unseen"]
    assert "NaN when ports are damaged" in rule and "NaN when capped" not in rule


def test_contract_is_regenerated_by_its_kept_builder(tmp_path):
    out = tmp_path / "contract.json"
    proc = subprocess.run([sys.executable, "-m", "experiment.build_live77_contract", "--out", str(out)], cwd=ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert out.read_bytes() == live77.CONTRACT.read_bytes()
    src = _contract()["sources"]
    for p in ("experiment/build_live77_contract.py", "experiment/check_live77_conventions.py",
              CHECKS.relative_to(ROOT).as_posix(), DECISIONS.relative_to(ROOT).as_posix()):
        assert src[p] == _sha(ROOT / p), p


def test_live_output_rules_records_file_and_value_hashes():
    from framework.training.csv_v3 import arms
    stored = _json(ROOT / RULES)
    r = _contract()["sources"][RULES]
    assert r["file_sha256"] == _sha(ROOT / RULES)
    assert r["value_sha256"] == arms._sha(stored["value"]) == stored["sha256"]
    assert r["value_sha256"].startswith("b29e6a19") and "sort_keys" in r["value_sha256_rule"]


def test_contract_records_owner_decisions_and_honest_provenance():
    import re
    c, dec, chk = _contract(), _json(DECISIONS), _json(CHECKS)
    assert c["admission"]["device_match"] == "ethernet_mac" and c["admission"]["ipv6"] == "excluded"
    assert c["conventions"]["port_cap_1000"]["params"] == {"limit": 1000, "order": "arrival"}
    assert c["zero_packet_window"]["score"] is True and c["damaged_window"]["capture_drop"] == "all_missing_not_scored"
    d = dec["defaults_chosen_by_app_chat"]
    assert c["model_id"] == d["model_id"]
    assert sorted(int(k) for k in chk["training_fit77"]["window_duration"]) == list(range(1, 11))
    # The app chat's default texts say 1 s; the model was trained on 1-10 s windows and the owner set 10 s windows with
    # "Any step already done with 1 s must be corrected". The contract ships the corrected texts and keeps the
    # superseded ones verbatim; the decisions file itself is not edited.
    stale_re = r"\b1[ -]?s(econd)?\b"
    stale = bool(re.search(stale_re, d["label"] + " " + d["provenance_note"]))
    override = dec["window"]["overrides_plan"]
    pr = c["provenance_review"]
    assert bool(pr["issues"]) == stale
    if stale:
        assert "Any step already done with 1 s must be corrected" in override and dec["window"]["length_seconds"] == 10
        assert (pr["status"], pr["ship_allowed"], pr["owner_wording_confirmed"]) == \
               ("corrected_under_owner_window_override", True, False)
        assert pr["superseded"] == {"label": d["label"], "provenance_note": d["provenance_note"],
                                    "source": DECISIONS.relative_to(ROOT).as_posix() + " defaults_chosen_by_app_chat"}
        assert pr["authority"] == {"window.overrides_plan": override, "constraint": dec["constraints"][2]}
        assert (c["label"], c["provenance_note"]) == (pr["corrected_label"], pr["corrected_provenance_note"])
        assert c["label"] == "Live XGBoost (77-class, 10 s windows)"
        assert "1-10 s" in c["provenance_note"] and "scored here on 10 s windows" in c["provenance_note"]
    else:
        assert (pr["status"], pr["ship_allowed"]) == ("accepted", True)
        assert (c["label"], c["provenance_note"]) == (d["label"], d["provenance_note"])
    assert not re.search(stale_re, c["label"] + " " + c["provenance_note"])  # the shipped texts never say 1 s
    assert "held-out test was not used" in c["provenance_note"]
    for t in (c["label"], c["provenance_note"]):
        assert not re.search(CAVEAT, t, re.I), t
    assert c["held_out_test_used"] is False and c["live_validation"] is False
