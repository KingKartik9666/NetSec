"""Synthetic migration fixtures preserve observable Python behavior, not accuracy."""
import json

import pytest


def _generate(path):
    from tools.golden_observations import generate_observation_goldens

    return generate_observation_goldens(path)


def _files(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def test_pcap_replay_records_independent_flags_rule_boundaries_and_host_filter(tmp_path, monkeypatch):
    from framework.data.live_monitor import LiveMonitor

    def forbidden(*args, **kwargs):
        raise AssertionError("Golden replay must not inspect the host, capture or infer")

    for method in ("start", "capabilities", "_begin_capture", "_connections", "_predict_window"):
        monkeypatch.setattr(LiveMonitor, method, forbidden)
    summary = _generate(tmp_path)
    fixture = json.loads((tmp_path / "live" / "windows.json").read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in fixture["cases"]}
    mixed = cases["mixed_headers"]["output"]
    assert mixed["stats"] == {
        "packets": 9, "bytes": 430, "syn": 1, "syn_ack": 1, "syn_all": 2,
        "ack": 3, "fin": 1, "rst": 1, "psh": 1, "urg": 1, "seconds": 10.0,
        "unique_ports_per_source": {"192.0.2.10>198.51.100.20": 1},
    }
    assert mixed["assessment"]["status"] == "benign_looking"
    assert len(mixed["recent_packets"]) == 9
    assert [entry["rule"] for entry in cases["syn_burst"]["output"]["assessment"]["evidence"]] == ["syn_burst"]
    assert [entry["rule"] for entry in cases["port_sweep"]["output"]["assessment"]["evidence"]] == ["port_sweep"]
    for case in ("syn_below_count", "syn_below_rate", "port_below_threshold"):
        assert cases[case]["output"]["assessment"]["status"] == "benign_looking"
    for case in ("zero_packets", "unrelated_packets"):
        assert cases[case]["output"]["stats"]["packets"] == 0
        assert cases[case]["output"]["assessment"]["status"] == "idle"
    assert cases["unrelated_packets"]["input"]["packet_count"] == 2
    assert len(cases["syn_burst"]["output"]["recent_packets"]) == 40
    assert summary["live_case_count"] == len(cases)
    assert (tmp_path / "live" / "synthetic.pcap").stat().st_size < 64 * 1024


def test_rag_fixture_preserves_scoped_retrieval_correction_and_grounding_failures(tmp_path):
    _generate(tmp_path)
    retrieval = json.loads((tmp_path / "rag" / "retrieval.json").read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in retrieval["cases"]}
    assert cases["packet_disagreement"]["output"]["passages"][0]["id"] == "live-rule-disagreement"
    assert cases["csv_scope"]["output"]["passages"][0]["id"] == "datasense-csv-scope"
    corrected = cases["weak_supported_evidence"]["output"]
    assert corrected["correction_attempted"] is True and len(corrected["retrieval_attempts"]) == 2
    for name in ("unrelated_query", "empty_query", "unsupported_source"):
        assert cases[name]["output"]["passages"] == []
        assert cases[name]["output"]["correction_attempted"] is False
    assert all(case["output"]["context_bytes"] <= 700 for case in cases.values())
    grounding = json.loads((tmp_path / "rag" / "grounding.json").read_text(encoding="utf-8"))
    checked = {case["id"]: case["output"] for case in grounding["cases"]}
    for name in ("faithful", "valid_citation", "partial_missing_routing", "rule_disagreement"):
        assert checked[name]["passed"] is True
    for name, issue in {
        "unsupported_percentage": "unsupported_percentage",
        "wrong_label": "predicted_label_missing_or_changed",
        "reversed_direction": "feature_direction_changed:network_packets_all_count",
        "invented_citation": "invented_source_identifier",
        "missing_routing_warning": "missing_input_presented_as_observed:network_tcp-flags-syn_count",
        "missing_disagreement": "rule_model_disagreement_missing",
        "truncation": "truncated_response",
    }.items():
        assert issue in checked[name]["issues"]


def test_generated_bytes_are_reproducible_portable_and_clock_patch_is_restored(tmp_path):
    from framework.data import live_monitor

    original_clock = live_monitor.datetime
    first, second = tmp_path / "first", tmp_path / "second"
    assert _generate(first) == _generate(second)
    assert _files(first) == _files(second)
    assert live_monitor.datetime is original_clock
    for name, raw in _files(first).items():
        if name.endswith(".json"):
            text = raw.decode("utf-8")
            assert str(tmp_path) not in text
            json.loads(text, parse_constant=lambda value: pytest.fail(f"Non-JSON numeric constant: {value}"))


def test_existing_fixture_is_never_overwritten(tmp_path):
    _generate(tmp_path)
    before = _files(tmp_path)
    with pytest.raises(FileExistsError):
        _generate(tmp_path)
    assert _files(tmp_path) == before


def test_collision_is_checked_before_writing_other_fixture_subdirectory(tmp_path):
    (tmp_path / "rag").mkdir()
    (tmp_path / "rag" / "keep.txt").write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _generate(tmp_path)
    assert _files(tmp_path) == {"rag/keep.txt": b"existing"}
