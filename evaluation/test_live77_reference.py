"""Python reference (oracle) for the live77 extraction contract, plan step 2.

Layer A (window facts -> 252 inputs) is checked against the dataset's own encoding on the 30,000
validation-partition rows (all validation roles; the held-out test is never opened). Layer B
(Ethernet frames -> window facts) is checked against the hand-computed vectors in
evaluation/fixtures/live77/frames_v1.json, which the Rust extractor (plan step 5) must also match.
"""
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from framework.training.csv_v3 import live77, live77_reference as ref

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "evaluation/fixtures/live77/frames_v1.json"
CONTRACT = json.loads(live77.CONTRACT.read_text(encoding="utf-8"))
NAMES = CONTRACT["feature_names"]
IDX = {n: i for i, n in enumerate(NAMES)}
# Convention ids a fixture may cite besides the contract's own (reference defaults the contract leaves open).
REFERENCE_DEFAULTS = {"link_vlan", "damage_propagation", "admission_order"}
PLAN_FIXTURES = {"tcp_syn_with_and_without_mss", "arp_only_window", "udp_fragment_train", "ports_over_1000",
                 "reserved_bit_tcp_flags", "lldp_frame", "zero_packet_window", "broadcast_arp_for_local_ip",
                 "ipv6_frame_excluded"}


def _fixtures():
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


def _expected(spec: dict) -> np.ndarray:
    """Expand a sparse expected vector: default, then NaN prefixes, then NaN names, then explicit values."""
    d = spec.get("default", 0.0)
    out = np.full(len(NAMES), np.nan if d is None else d, dtype=np.float32)
    for p in spec.get("nan_prefixes", []):
        hit = [i for n, i in IDX.items() if n.startswith(p)]
        assert hit, p
        out[hit] = np.nan
    for n in spec.get("nan", []):
        out[IDX[n]] = np.nan
    for n, v in spec.get("values", {}).items():
        out[IDX[n]] = np.nan if v is None else np.float32(v)
    return out


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype == np.float32 and bool((np.isnan(a) == np.isnan(b)).all()) and bool(
        (a[~np.isnan(a)] == b[~np.isnan(b)]).all())


def _diff(a, b):
    bad = [n for n, x, y in zip(NAMES, a, b) if not ((np.isnan(x) and np.isnan(y)) or x == y)]
    return {n: (float(a[IDX[n]]), float(b[IDX[n]])) for n in bad}


# ------------------------------------------------------------------------------------------ module surface
def test_reference_names_follow_the_contract():
    assert ref.NAMES == NAMES == live77.feature_names(list(live77.EXACT)) and len(NAMES) == 252
    assert ref.NOMINAL_SECONDS == CONTRACT["window"]["nominal_duration_seconds"] == 10.0
    assert ref.PORT_LIMIT == CONTRACT["conventions"]["port_cap_1000"]["params"]["limit"] == 1000
    assert ref.MF_MASK == CONTRACT["conventions"]["fragment_mf_only"]["params"]["ipv4_frag_mask"] == 0x2000
    assert {d: list(v) for d, v in ref.VOCAB.items()} == CONTRACT["port_vocabulary"]


def test_reference_refuses_test_partition_paths(tmp_path):
    for name in ("X_test.npy", "selected_test_77class.csv.gz", "row_ids_test.npy", "test.csv"):
        with pytest.raises(PermissionError):
            ref.guarded(tmp_path / name)
    with pytest.raises(PermissionError):
        ref.guarded(ROOT / "dataset/experiments/csv_v3_cascade_20261030_v1/X_train.npy")  # not on the allow-list
    assert ref.guarded(ref.DATA / "X_validation.npy") == ref.DATA / "X_validation.npy"


# ----------------------------------------------------------------------------- Layer A: synthetic checks
def _facts(**over):
    f = {"duration_seconds": 10.0, "packets": {"all": 4, "src": 1, "dst": 3}, "fragmented": 0,
         "tcp_flags": {k: 0 for k in ref.FLAGS}, "ips": {"all": 2, "src": 1, "dst": 1},
         "macs": {"all": 2, "src": 1, "dst": 1}, "stats": {f: "empty" for f in ref.FAMILIES},
         "ports": {d: {"kept": [], "capped": False, "damaged": False} for d in ("src", "dst")}}
    f.update(over)
    return f


def test_layer_a_rates_ratios_and_quality_flag():
    v = ref.vector(_facts(tcp_flags={**{k: 0 for k in ref.FLAGS}, "syn": 3}))
    assert v.dtype == np.float32 and v.shape == (252,)
    g = lambda n: v[IDX[n]]  # noqa: E731
    assert g("window_duration_seconds") == 10 and g("quality_no_packets_observed") == 0
    assert g("network_packets_all_count__per_second") == np.float32(0.4)
    assert g("network_packets_dst_count__per_second") == np.float32(0.3)
    assert g("network_tcp-flags-syn__fraction_of_packets") == np.float32(0.75)
    assert g("network_src_packet_fraction") == np.float32(0.25) and g("network_packets_per_reported_ip") == 2
    # Ratio is NaN when the denominator is <= 0 or missing, or the numerator is missing.
    v = ref.vector(_facts(ips={"all": 0, "src": 0, "dst": 0}, tcp_flags=None))
    assert np.isnan(v[IDX["network_packets_per_reported_ip"]])
    assert np.isnan(v[IDX["network_tcp-flags-ack__fraction_of_packets"]]) and np.isnan(v[IDX["network_tcp-flags-ack_count"]])
    v = ref.vector(_facts(duration_seconds=0.0))
    assert np.isnan(v[IDX["network_packets_all_count__per_second"]])


def test_layer_a_empty_family_zero_and_zero_packet_window():
    v = ref.vector(_facts())
    stats = [IDX[f"network_{f}_{s}"] for f in ref.FAMILIES for s in ref.STATS]
    assert (v[stats] == 0).all()  # packets > 0, family empty -> 0
    z = ref.vector(_facts(packets={"all": 0, "src": 0, "dst": 0}, ips={"all": 0, "src": 0, "dst": 0},
                          macs={"all": 0, "src": 0, "dst": 0}))
    assert np.isnan(z[stats]).all() and z[IDX["quality_no_packets_observed"]] == 1
    assert z[IDX["network_packets_all_count__per_second"]] == 0
    assert np.isnan(z[IDX["network_src_packet_fraction"]]) and np.isnan(z[IDX["network_packets_per_reported_ip"]])
    assert z[IDX["network_ports_src__present_UNSEEN"]] == 0 and z[IDX["network_ports_dst__present_80"]] == 0


def test_layer_a_port_presence_cap_and_unseen():
    ports = {"src": {"kept": [80, 51000], "capped": False, "damaged": False},
             "dst": {"kept": list(range(20000, 20999)) + [53], "capped": True, "damaged": False}}
    v = ref.vector(_facts(ports=ports))
    g = lambda n: v[IDX[n]]  # noqa: E731
    assert g("network_ports_src__present_80") == 1 and g("network_ports_src__present_53") == 0
    assert g("network_ports_src__present_UNSEEN") == 1
    assert g("network_ports_dst__present_53") == 1 and np.isnan(g("network_ports_dst__present_80"))
    assert g("network_ports_dst__present_UNSEEN") == 1
    v = ref.vector(_facts(ports={"src": {"kept": [53], "capped": False, "damaged": True},
                                 "dst": {"kept": [], "capped": False, "damaged": False}}))
    assert v[IDX["network_ports_src__present_53"]] == 1 and np.isnan(v[IDX["network_ports_src__present_80"]])
    assert np.isnan(v[IDX["network_ports_src__present_UNSEEN"]]) and v[IDX["network_ports_dst__present_UNSEEN"]] == 0


def test_layer_a_capture_drop_is_all_missing():
    assert np.isnan(ref.vector({"capture_drop": True})).all()


def test_summaries_use_population_std_in_arrival_order():
    assert ref.summarize([]) == "empty"
    s = ref.summarize([64, 128])
    assert s == {"avg": 96.0, "max": 128, "min": 64, "std_deviation": 32.0}
    assert ref.summarize([1, 1, 0])["std_deviation"] == math.sqrt(2 / 9)


# ------------------------------------------------------------------- Layer A: dataset validation rows
def test_layer_a_reproduces_validation_rows_exactly():
    """All 30,000 validation rows: identical NaN mask and max abs diff 0 in float32 (plan step 2a)."""
    got, expected, ids = ref.validation_parity()
    assert got.shape == expected.shape == (30000, 252) and len(set(ids.tolist())) == 30000
    nan_mismatch = np.isnan(got) != np.isnan(expected)
    assert not nan_mismatch.any(), {NAMES[j]: int(nan_mismatch[:, j].sum()) for j in np.flatnonzero(nan_mismatch.any(0))}
    ok = ~np.isnan(expected)
    diff = np.zeros_like(got)
    diff[ok] = np.abs(got[ok] - expected[ok])
    assert float(diff.max()) == 0.0, {NAMES[j]: float(diff[:, j].max()) for j in np.flatnonzero(diff.max(0) > 0)}


# ----------------------------------------------------------------------------------- Layer B fixtures
def test_fixture_file_is_tagged_and_auditable():
    fx = _fixtures()
    assert fx["version"] == 1 and fx["contract_feature_names_sha256"] == CONTRACT["feature_names_sha256"]
    assert fx["nominal_seconds"] == CONTRACT["window"]["nominal_duration_seconds"]
    ids = [f["id"] for f in fx["fixtures"]]
    assert len(ids) == len(set(ids)) and PLAN_FIXTURES <= set(ids)
    conv = CONTRACT["conventions"]
    damaged = {f"damaged_window.{k}": v for k, v in CONTRACT["damaged_window"]["status"].items() if k != "rule"}
    status = {**{k: v["status"] for k, v in conv.items()}, **damaged, **dict.fromkeys(REFERENCE_DEFAULTS, "reference_default")}
    for f in fx["fixtures"]:
        assert f["conventions"] and f["source"] and f["desc"], f["id"]
        assert set(f["conventions"]) <= set(status), f["id"]
        assert f["statuses"] == {c: status[c] for c in f["conventions"]}, f["id"]
        for fr in f["frames"]:
            assert re.fullmatch(r"(?:[0-9a-f]{2})+", fr["hex"]) and fr["desc"], f["id"]
            assert fr.get("wire_length", len(fr["hex"]) // 2) >= len(fr["hex"]) // 2
        ts = [fr["t"] for fr in f["frames"]]
        assert ts == sorted(ts) and all(0 <= t < fx["nominal_seconds"] for t in ts), f["id"]
    # Every packet-level convention is exercised by at least one fixture.
    packet_level = {"empty_family_zero", "mss_bearing_only", "fragment_mf_only", "fragments_keep_stats", "arp_ips",
                    "arp_ip_direction", "port_cap_1000", "port_unseen", "tcp_flags_12bit", "population_std",
                    "header_direction", "device_mac_admission", "device_mac_packet_direction",
                    "broadcast_not_admitted", "all_frames_counted", "ipv6_excluded", "nominal_duration",
                    "zero_packet_window_scored"}
    tagged = {c for f in fx["fixtures"] for c in f["conventions"]}
    assert packet_level <= set(conv) and packet_level | set(damaged) | REFERENCE_DEFAULTS <= tagged


def test_fixture_file_is_rebuilt_by_its_kept_builder():
    import hashlib
    from evaluation.fixtures.live77 import build_frames_v1 as b
    assert (json.dumps(b.build(), indent=1) + "\n").encode() == FIXTURES.read_bytes()
    assert _fixtures()["builder_sha256"] == hashlib.sha256(Path(b.__file__).read_bytes()).hexdigest()
    imports = re.findall(r"^\s*(?:from|import)\s+.*$", Path(b.__file__).read_text(encoding="utf-8"), re.M)
    assert imports and not any("live77" in line for line in imports)  # expected vectors do not come from the oracle


@pytest.mark.parametrize("fixture", _fixtures()["fixtures"], ids=lambda f: f["id"])
def test_layer_b_matches_hand_computed_vector(fixture):
    fx = _fixtures()
    got = ref.frames_vector(fixture["frames"], fx["device"]["mac"], capture_drop=fixture.get("capture_drop", False))
    want = _expected(fixture["expected"])
    assert _same(got, want), _diff(got, want)


def test_layer_b_facts_are_json_and_feed_layer_a():
    fx = _fixtures()
    f = next(x for x in fx["fixtures"] if x["id"] == "ports_over_1000")
    facts = ref.window_facts(f["frames"], fx["device"]["mac"])
    assert json.loads(json.dumps(facts)) == facts
    assert len(facts["ports"]["dst"]["kept"]) == 1000 and facts["ports"]["dst"]["capped"] is True
    assert facts["ports"]["dst"]["kept"][0] == 80 and 443 not in facts["ports"]["dst"]["kept"]
    assert _same(ref.vector(facts), ref.frames_vector(f["frames"], fx["device"]["mac"]))


def test_layer_b_rejects_malformed_fixture_input():
    with pytest.raises(ValueError):
        ref.window_facts([{"t": 0.0, "hex": "00" * 20, "wire_length": 10}], "02:00:00:00:00:50")
    with pytest.raises(ValueError):
        ref.window_facts([{"t": 1.0, "hex": "00" * 60}, {"t": 0.5, "hex": "00" * 60}], "02:00:00:00:00:50")
    with pytest.raises(ValueError):
        ref.window_facts([], "not-a-mac")
