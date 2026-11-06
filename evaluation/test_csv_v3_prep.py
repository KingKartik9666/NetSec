"""Unit checks for the cascade dataset v3 preparation (synthetic data only)."""
import numpy as np
import pandas as pd
import pytest

from framework.training.csv_v3 import prep


def _classes():
    return ["benign", "ddos_a", *prep.RECON8[:4], "recon_ping-sweep", *prep.RECON8[4:], "web_x"]


def test_targets_merge_keeps_benign_and_order():
    t = prep.targets(_classes())
    assert t["classes77"] == ["benign", "ddos_a", prep.MERGED, "recon_ping-sweep", "web_x"]
    merged = t["classes77"].index(prep.MERGED)
    for name in prep.RECON8:
        assert t["map84_to77"][_classes().index(name)] == merged
    assert t["classes77"][t["map84_to77"][_classes().index("recon_ping-sweep")]] == "recon_ping-sweep"
    assert sorted(t["map84_to8"][t["map84_to8"] >= 0]) == list(range(8))
    assert t["map84_to8"][_classes().index("benign")] == -1


def test_targets_rejects_missing_recon_or_benign_not_first():
    with pytest.raises(ValueError):
        prep.targets(["benign", "ddos_a"])
    with pytest.raises(ValueError):
        prep.targets(["ddos_a", "benign", *prep.RECON8])


def test_raw_features_nan_semantics():
    frame = pd.DataFrame({
        "network_ips_src": ["['10.0.0.1']", None, "['10.0.0.1']"],
        "network_ips_dst": ["['10.0.0.2', '10.0.0.3']", "[]", "['10.0.0.2']"],
        "network_ports_src": ["['80', '5000']", "[]", "[]"],
        "network_ports_dst": ["['80', '22', '8080']", "[]", "[]"],
    })
    out = prep.raw_features(frame)
    assert list(out.columns) == list(prep.RAW_FEATURES)
    assert out.loc[0, "listed_dst_ports_per_dst_ip"] == pytest.approx(1.5)
    assert out.loc[0, "listed_src_ports_per_src_ip"] == pytest.approx(2.0)
    assert out.loc[0, "listed_port_overlap_jaccard"] == pytest.approx(1 / 4)
    assert out.loc[0, "listed_low_dst_port_share"] == pytest.approx(2 / 3)
    assert out.loc[1].isna().all()                      # missing/empty lists -> unknown, not zero
    assert out.loc[2, "listed_dst_ports_per_dst_ip"] == 0.0   # real zero ports for a listed address


def _support_fixture():
    n = 60
    scen = np.repeat(np.arange(6), 10)
    dev = scen % 3                                       # three devices
    rng = np.random.default_rng(0)
    a = rng.normal(size=n).astype(np.float32)
    one_scen = (scen == 2).astype(np.float32)            # 1 scenario, 1 device -> folded
    one_dev = np.isin(scen, [0, 3]).astype(np.float32)   # 2 scenarios, 1 device (dev 0) -> folded
    common = (scen % 2).astype(np.float32)               # 3 scenarios, 3 devices -> kept
    zero = np.zeros(n, np.float32)                       # zero support -> folded, not dropped
    unseen = np.zeros(n, np.float32)                     # constant UNSEEN -> dropped, never folded
    x = np.stack([a, a.copy(), np.ones(n, np.float32), np.full(n, np.nan, np.float32),
                  one_scen, one_dev, common, zero, unseen,
                  (scen == 4).astype(np.float32)], axis=1)
    names = ["a", "a_dup", "const", "allnan", "p__present_9", "p__present_10", "p__present_80",
             "p__present_99", "p__present_UNSEEN", "q__present_7"]
    return x, names, scen, dev


def test_fit_columns_support_folding_and_pruning():
    x, names, scen, dev = _support_fixture()
    sel = prep.fit_columns(x, np.ones(len(x), bool), names, scen, dev, min_scenarios=3, min_devices=2)
    assert [names[j] for j in sel["kept"]] == ["a", "p__present_80"]
    assert sel["dropped"]["a_dup"] == "duplicate_of:a"
    assert sel["dropped"]["const"] == "constant_on_fit_rows"
    assert sel["dropped"]["allnan"] == "all_missing_on_fit_rows"
    assert sel["dropped"]["p__present_UNSEEN"] == "constant_on_fit_rows"
    assert sel["rare"]["p"] == ["p__present_9", "p__present_10", "p__present_99"]
    # q has a single weak token: pooled into the shared group, never re-emitted as its own column
    assert "q" not in sel["rare"] and sel["rare"][prep.SHARED_RARE] == ["q__present_7"]
    counts, cnames = prep.rare_counts(x, names, sel["rare"])
    assert cnames == [f"{prep.SHARED_RARE}__rare_token_count", "p__rare_token_count"]
    assert counts[scen == 2, 1].tolist() == [1.0] * 10 and counts[scen == 0, 1].tolist() == [1.0] * 10


def test_fit_columns_uses_only_fit_rows():
    x = np.array([[1.0], [1.0], [5.0]], np.float32)
    sel = prep.fit_columns(x, np.array([True, True, False]), ["c"], np.zeros(3, int), np.zeros(3, int), 1, 1)
    assert sel["kept"] == [] and sel["dropped"]["c"] == "constant_on_fit_rows"


def test_duplicate_columns_across_final_matrix():
    x = np.array([[1, 1, 2], [2, 2, 3], [np.nan, np.nan, 4]], np.float32)
    assert prep.duplicate_columns(x, np.ones(3, bool), ["a", "b", "c"]) == {"b": "a"}


def test_conflicts_and_nan_canonical_digests():
    x = np.array([[1, np.nan], [1, np.nan], [2, 0], [2, 0]], np.float32)
    x[1, 1] = np.frombuffer(np.uint32(0x7FC00001).tobytes(), np.float32)[0]  # different NaN payload
    d = prep.row_digests(x, chunk=3)
    assert d[0] == d[1] and d[2] == d[3]
    y = np.array([0, 1, 3, 3])
    assert prep.conflict_rows(d, y, np.arange(4)).tolist() == [True, True, False, False]
    assert prep.conflict_rows(d, y, np.array([0, 2, 3])).tolist() == [False] * 4
