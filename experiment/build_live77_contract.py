"""Build the frozen live77 extraction contract (integration plan step 1) from its sources. Deterministic.

Run: .venv/Scripts/python.exe -m experiment.check_live77_conventions   (only when the dataset evidence must be redone)
     .venv/Scripts/python.exe -m experiment.build_live77_contract [--out PATH]
Writes framework/training/csv_v3/live77_contract.json (live77.CONTRACT) unless --out is given.

Inputs: the model schema (252 names in order), the owner's decisions file, the integration plan, the read-only dataset
checks (experiment/check_live77_conventions.py output), Rust port_features.json and the live output rules. An
"owner_decision" quotes the decisions file verbatim. A choice the owner has not made is tagged *_pending_owner and
listed under pending_owner_signoff. Steps 2-6 implement this contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from framework.training.csv_v3 import arms, live77

ROOT = Path(__file__).resolve().parents[1]
APP = "artifacts/app_v0_6_20261102"
MODEL = "artifacts/training/live77_20261031_v1/jobs/77_live_final/attempt_0/model"
P = {
    "schema": f"{MODEL}/model_schema.json", "checkpoint": f"{MODEL}/model.ubj",
    "selection": "artifacts/training/live77_20261031_v1/selection.json",
    "window": "artifacts/training/live77_20261031_v1/window.json",
    "rules": "artifacts/training/upgrades2_live_20261101_v1/live_output_rules.json",
    "decisions": f"{APP}/live77_decisions.json", "plan": f"{APP}/live77_integration_plan.json",
    "checks": f"{APP}/live77_convention_checks.json", "check_script": "experiment/check_live77_conventions.py",
    "builder": "experiment/build_live77_contract.py",
    "rust_ports": "rust/netsec/crates/runtime/src/live/port_features.json",
}
CAVEAT = re.compile(r"unverif|untested|provisional|limitation|caveat|experimental|accuracy_verified|not[ _-]?verified"
                    r"|unvalidated", re.I)  # Live.tsx caveat filter (Live.tsx:12); the provenance text must survive it
STALE_WINDOW = re.compile(r"\b1[ -]?s(econd)?\b")
PROPOSED_LABEL = "Live XGBoost (77-class, 10 s windows)"
PROPOSED_NOTE = ("Trained on DataSense windows of 1-10 s; scored here on 10 s windows; not yet validated on real traffic "
                 "on this PC. Validation results are from the dataset, the held-out test was not used.")
STAT = {"avg": "mean", "max": "max", "min": "min", "std_deviation": "std_population"}
FAMILY = {  # name stem -> (definition, damage family, extra conventions)
    "header-length": ("stat.ipv4.ihl_bytes", "ipv4_header", []), "ip-flags": ("stat.ipv4.flags3", "ipv4_header", []),
    "ip-length": ("stat.ipv4.total_length", "ipv4_header", []), "ttl": ("stat.ipv4.ttl", "ipv4_header", []),
    "mss": ("stat.tcp.mss_option", "tcp_options", ["mss_bearing_only"]),
    "tcp-flags": ("stat.tcp.flags12", "tcp_header", ["tcp_flags_12bit"]),
    "window-size": ("stat.tcp.window_raw", "tcp_header", []),
}
FLAG_MASK = {"fin": 0x01, "syn": 0x02, "rst": 0x04, "psh": 0x08, "ack": 0x10, "urg": 0x20}
PORT_CONV = ["port_cap_1000", "port_unseen", "fragments_keep_stats", "header_direction"]
FRAME_CONV = ["device_mac_admission", "device_mac_packet_direction", "broadcast_not_admitted", "all_frames_counted",
              "ipv6_excluded"]


def _read(key: str):
    return json.loads((ROOT / P[key]).read_text(encoding="utf-8"))


def _sha(key: str) -> str:
    return hashlib.sha256((ROOT / P[key]).read_bytes()).hexdigest()


def _n(v) -> str:
    return f"{v:,}"


def feature(i: int, name: str, seconds: float) -> dict:
    f: dict = {"index": i, "name": name, "group": live77.group_of(name)}

    def put(definition, params, nan_rule, damage, conventions):
        f["definition"] = definition
        if params is not None:
            f["params"] = params
        f["nan_rule"] = nan_rule
        if damage:
            f["damage_family"] = damage
        f["conventions"] = conventions
        return f

    if name == "network_fragmented-packets":
        return put("count.ipv4_mf", {"ipv4_frag_mask": 0x2000}, "family_count", "ipv4_header", ["fragment_mf_only"])
    if m := re.fullmatch(r"network_(header-length|ip-flags|ip-length|ttl|mss|tcp-flags|window-size)_"
                         r"(avg|max|min|std_deviation)", name):
        d, damage, extra = FAMILY[m[1]]
        params = {"stat": STAT[m[2]], **({"bits": 12} if m[1] == "tcp-flags" else {})}
        return put(d, params, "family_stat", damage, ["empty_family_zero", "population_std", "fragments_keep_stats",
                                                      *extra])
    if m := re.fullmatch(r"network_ips_(all|src|dst)_count", name):
        return put("distinct.ipv4_with_arp", {"direction": m[1]}, "family_count", "ipv4_or_arp_header",
                   ["arp_ips", "arp_ip_direction", "ipv6_excluded", "header_direction"])
    if m := re.fullmatch(r"network_macs_(all|src)_count|network_macs_(dst)__observed_distinct_count", name):
        return put("distinct.mac", {"direction": m[1] or m[2]}, "global_only", None,
                   ["device_mac_admission", "broadcast_not_admitted", "header_direction"])
    if m := re.fullmatch(r"network_packets_(all|src|dst)_count", name):
        return put("count.frames", {"direction": m[1]}, "global_only", None, FRAME_CONV)
    if m := re.fullmatch(r"network_tcp-flags-(\w+)_count", name):
        return put("count.tcp_flag", {"flag": m[1], "mask": FLAG_MASK[m[1]]}, "family_count", "tcp_header",
                   ["fragments_keep_stats"])
    if m := re.fullmatch(r"network_ports_(src|dst)__present_UNSEEN", name):
        return put("presence.port_unseen", {"direction": m[1]}, "port_unseen", "ports", PORT_CONV)
    if m := re.fullmatch(r"network_ports_(src|dst)__present_(\d+)", name):
        return put("presence.port", {"direction": m[1], "port": int(m[2])}, "port_presence", "ports", PORT_CONV)
    if name == "window_duration_seconds":
        return put("window.nominal_seconds", {"seconds": seconds}, "global_only", None, ["nominal_duration"])
    if m := re.fullmatch(r"(network_packets_(?:all|src|dst)_count)__per_second", name):
        return put("rate.per_nominal_second", {"numerator": m[1]}, "ratio", None, ["nominal_duration"])
    if name == "quality_no_packets_observed":
        return put("quality.no_packets", None, "global_only", None, ["zero_packet_window_scored"])
    if m := re.fullmatch(r"network_tcp-flags-(\w+)__fraction_of_packets", name):
        return put("ratio.of_packets_all", {"numerator": f"network_tcp-flags-{m[1]}_count"}, "ratio", None, [])
    if m := re.fullmatch(r"network_(src|dst)_packet_fraction", name):
        return put("ratio.of_packets_all", {"numerator": f"network_packets_{m[1]}_count"}, "ratio", None,
                   ["device_mac_packet_direction"])
    if name == "network_packets_per_reported_ip":
        return put("ratio.packets_per_ip", {"numerator": "network_packets_all_count",
                                            "denominator": "network_ips_all_count"}, "ratio", None, ["arp_ips"])
    raise ValueError(f"no contract definition for {name}")


def conventions(dec: dict, chk: dict, seconds: int) -> dict:
    r, a, w = dec["rules"], dec["admission"], dec["window"]
    ev = P["checks"]
    val = f"{_n(chk['rows'])} validation-partition rows of csv_v3_cascade_20261030_v1 (held-out test not opened)"
    tr = chk["training_fit77"]
    fit = f"the {_n(tr['rows'])} fit77 training rows the final model was fit on"
    fam = chk["empty_family_when_packets"]
    ports, std, zero = chk["ports"], chk["population_std"], chk["zero_packet_rows"]
    arp = chk["arp_only_windows"]
    wd = chk["window_duration"]
    return {
        "empty_family_zero": {
            "status": "dataset_verified", "rule": live77.EMULATION["empty_family_zero"], "evidence_source": ev,
            "evidence": f"{val}: with packets_all > 0 ({_n(fam['rows_with_packets'])} rows) none of the 7 summary "
                        f"families is NaN; all-zero rows: " + ", ".join(
                            f"{k} {_n(v['all_zero_rows'])}" for k, v in fam["families"].items()) + "."},
        "mss_bearing_only": {
            "status": "dataset_verified_and_owner_decision", "rule": live77.EMULATION["mss_bearing_only"],
            "replaces": "mss_absent_zero ('a TCP packet without an MSS option contributes mss 0'), contradicted by the "
                        "data",
            "evidence_source": ev,
            "evidence": f"{fit}: 0 of {_n(tr['mss']['rows_mss_max_gt0'])} rows with mss_max > 0 have mss_min == 0 "
                        f"({tr['mss']['rows_mss_max_gt0_and_min_0']} found), and "
                        f"{tr['mss']['rows_syn_0_and_mss_max_gt0']} have syn == 0 with mss_max > 0. {val}: "
                        f"{chk['mss']['rows_mss_max_gt0_and_min_0']} of {_n(chk['mss']['rows_mss_max_gt0'])}; "
                        f"{chk['mss']['rows_syn_0_and_mss_max_gt0']} row has syn == 0 with mss_max > 0 (an MSS option "
                        "outside a SYN; the rule keys on the option, not on SYN).",
            "decision": r["mss"]},
        "fragment_mf_only": {
            "status": "owner_decision", "rule": live77.EMULATION["fragment_mf_only"], "params": {"ipv4_frag_mask": 0x2000},
            "decision": r["fragmented_packets"],
            "unknown": "publisher definition (MF-only vs MF-or-offset) cannot be settled from CSV rows"},
        "fragments_keep_stats": {
            "status": "plan_default_pending_owner",
            "rule": live77.EMULATION["fragments_keep_stats"] + "; a non-initial fragment adds its IPv4 header statistics "
                                                                "and no transport values",
            "source": "live77.EMULATION (2026-10-31 review) and integration plan steps 2 and 5; not in the decisions "
                      "file"},
        "arp_ips": {
            "status": "dataset_verified", "rule": live77.EMULATION["arp_ips"], "evidence_source": ev,
            "evidence": f"{val}: all {_n(arp['rows_ips_all_count_gt0'])} of {_n(arp['rows'])} windows whose only "
                        "protocol is ARP report IPv4 addresses (ips_all_count > 0), so ARP addresses are counted."},
        "arp_ip_direction": {
            "status": "implementer_default_pending_owner",
            "rule": "the ARP sender protocol address counts as src and the target protocol address as dst; "
                    "0.0.0.0 (ARP probe sender) is counted literally",
            "evidence_note": f"{val}: in {_n(arp['device_sent_only_rows'])} ARP-only windows the device only sent, "
                             "ips src/dst counts are " + ", ".join(
                                 f"{k} in {_n(v)}" for k, v in arp["device_sent_only_ips_src_dst_counts"].items())
                             + " windows, consistent with a directional sender/target mapping but not proof of which "
                             f"side is src; 0.0.0.0 appears in {arp['rows_with_0.0.0.0']} ARP-only windows and in "
                             f"{chk['rows_with_0.0.0.0_in_ips_all']} windows overall",
            "unknown": "publisher ARP direction and its handling of 0.0.0.0 probes"},
        "port_cap_1000": {
            "status": "dataset_verified_and_owner_decision", "rule": live77.EMULATION["port_cap_1000"],
            "params": {"limit": 1000, "order": "arrival"}, "evidence_source": ev,
            "evidence": f"{val}: presence is NaN exactly when the reported port count differs from the listed count "
                        f"({_n(ports['src']['capped_rows'])} src / {_n(ports['dst']['capped_rows'])} dst rows), then 1 "
                        f"for listed vocabulary ports and NaN otherwise ({ports['src']['presence_mismatches']} + "
                        f"{ports['dst']['presence_mismatches']} mismatches); lists never exceed "
                        f"{max(ports['src']['max_list_length'], ports['dst']['max_list_length'])} ports.",
            "decision": r["port_cap"], "unknown": "which 1000 ports the publisher exporter keeps"},
        "port_unseen": {
            "status": "dataset_verified",
            "rule": "UNSEEN is 1 when a kept port of that direction is outside the 96-port vocabulary of that direction "
                    "(port_vocabulary below); a capped direction keeps 1000 ports, at most 96 of them in the "
                    "vocabulary, so UNSEEN is 1 there",
            "evidence_source": ev,
            "evidence": f"{val}: UNSEEN recomputed from the raw lists against the 96-port vocabularies: "
                        f"{ports['src']['unseen_mismatches']} + {ports['dst']['unseen_mismatches']} mismatches, "
                        f"{ports['src']['unseen_nan']} + {ports['dst']['unseen_nan']} NaN; UNSEEN is 1 in all "
                        f"{_n(ports['src']['unseen_one_in_capped_rows'])} src and "
                        f"{_n(ports['dst']['unseen_one_in_capped_rows'])} dst capped rows."},
        "tcp_flags_12bit": {
            "status": "dataset_verified_and_owner_decision", "rule": live77.EMULATION["tcp_flags_12bit"],
            "params": {"bits": 12, "expression": "((tcp[12] & 0x0f) << 8) | tcp[13]"}, "evidence_source": ev,
            "evidence": f"{fit}: tcp-flags_max > 511 in {_n(tr['tcp_flags']['rows_max_gt_511'])} rows, max "
                        f"{tr['tcp_flags']['max']:g}. {val}: max {chk['tcp_flags']['max']:g} "
                        f"({chk['tcp_flags']['rows_max_gt_511']} rows above 511).",
            "decision": r["tcp_flags"], "unknown": "semantics of the reserved bits in the publisher exporter"},
        "population_std": {
            "status": "dataset_verified", "rule": "std_deviation is the population standard deviation (ddof=0)",
            "evidence_source": ev,
            "evidence": f"{val}: in rows with 2 packets, distinct min/max and avg at the midpoint, std/(max-min) is "
                        "exactly 0.5 (population), never 0.7071 (sample): " + ", ".join(
                            f"{k} {_n(v['ratio_0_5'])}/{_n(v['rows'])}" for k, v in std.items() if v["rows"]) + "."},
        "header_direction": {
            "status": "implementer_default_pending_owner",
            "rule": "src/dst of ips, macs and ports are the header source/destination fields of each admitted frame, "
                    "independent of the device's packet direction; all is the union of src and dst",
            "evidence_note": f"{val}: macs_src holds a MAC other than the device MAC in "
                             f"{_n(chk['macs_src_hold_non_device_macs_rows'])} rows, so the MAC sets are header "
                             "fields of all admitted frames; ips and ports are assumed to follow the same rule"},
        "device_mac_admission": {
            "status": "owner_decision",
            "rule": "a frame is admitted when its Ethernet source or destination MAC equals the device MAC",
            "decision": a["device_match"]},
        "device_mac_packet_direction": {
            "status": "implementer_default_pending_owner",
            "rule": "packets_src counts admitted frames whose source MAC is the device MAC, packets_dst the rest, so "
                    "packets_all = packets_src + packets_dst",
            "evidence_note": f"{val}: packets_all == packets_src + packets_dst in "
                             f"{_n(chk['packets_all_equals_src_plus_dst_rows'])} rows; which side is the device's own "
                             "frames is inferred, not verified"},
        "broadcast_not_admitted": {
            "status": "implementer_default_pending_owner",
            "rule": "received broadcast/multicast frames that do not carry the device MAC are not admitted (literal "
                    "reading of 'device MAC address only')",
            "unknown": "whether the publisher also counted received broadcast/multicast frames"},
        "all_frames_counted": {
            "status": "owner_decision",
            "rule": "every admitted frame counts in packets and macs, including non-IP frames such as LLDP; the capture "
                    "filter is widened to the device MAC accordingly; payloads are never saved",
            "decision": a["frames"]},
        "ipv6_excluded": {
            "status": "owner_decision",
            "rule": "IPv6 frames are left out of the model window entirely (no packets, macs, ports or statistics)",
            "decision": r["ipv6"], "evidence_note": "the training lists contain no IPv6 addresses (integration plan)"},
        "nominal_duration": {
            "status": "dataset_verified_and_owner_decision",
            "rule": f"window_duration_seconds = {seconds}; per_second = count / {seconds} (the nominal duration); the "
                    f"measured duration only gates the window ({seconds} +/- 1 s)",
            "evidence_source": ev,
            "evidence": f"{val}: window_duration_seconds is always an integer in "
                        f"{min(map(int, wd['values']))}..{max(map(int, wd['values']))} ({_n(wd['values'][str(seconds)])} "
                        f"rows of length {seconds}) and per_second == count / window_duration_seconds exactly in float32 "
                        f"({_n(wd['per_second_equals_count_over_duration']['all']['exact_float32'])} rows).",
            "decision": w["scoring"]},
        "zero_packet_window_scored": {
            "status": "dataset_verified_and_owner_decision",
            "rule": "a window with no admitted frame is scored: counts 0, per_second 0, quality_no_packets_observed 1, "
                    "statistics and ratios NaN, port presence and UNSEEN 0",
            "evidence_source": ev,
            "evidence": f"{val}: all {zero['rows']} zero-packet rows follow exactly this pattern, and the quality flag "
                        "is 0 in every other row.",
            "decision": r["empty_window"]},
    }


OWNER_CORRECTION = "Any step already done with 1 s must be corrected"   # live77_decisions.json window.overrides_plan
HONEST = "honest provenance label must stay visible on the Live page"     # live77_decisions.json constraints


def provenance(dec: dict, train_lengths: list[int], seconds: int) -> dict:
    """Label and note to ship. The app chat's default texts say 1 s; the owner's window override (10 s everywhere, 'Any
    step already done with 1 s must be corrected') supersedes them, so the corrected texts ship and the superseded ones
    are kept verbatim. The decisions file is not edited. Without that override a stale text stays pending (not shipped)."""
    d, w = dec["defaults_chosen_by_app_chat"], dec["window"]
    label, note = d["label"], d["provenance_note"]
    issues = []
    if STALE_WINDOW.search(label):
        issues.append(f"label names a 1 s window; the owner set {seconds} s windows")
    if STALE_WINDOW.search(note):
        issues.append(f"note says the model was trained on 1-second windows; the fit rows have lengths "
                      f"{min(train_lengths)}-{max(train_lengths)} s")
    for t in (label, note, PROPOSED_LABEL, PROPOSED_NOTE):
        assert not CAVEAT.search(t), f"the Live.tsx caveat filter would hide: {t}"
    assert not STALE_WINDOW.search(PROPOSED_LABEL + " " + PROPOSED_NOTE) and f"{seconds} s windows" in PROPOSED_LABEL
    out = {"issues": issues}
    if not issues:
        return {"status": "accepted", "ship_allowed": True, **out, "label": label, "provenance_note": note,
                "rule": "label and provenance_note are copied verbatim from the decisions file"}
    if OWNER_CORRECTION not in w["overrides_plan"] or seconds != 10 or HONEST not in dec["constraints"]:
        return {"status": "pending_owner_correction", "ship_allowed": False, **out, "label": label, "provenance_note": note,
                "proposed_label": PROPOSED_LABEL, "proposed_provenance_note": PROPOSED_NOTE,
                "rule": "the decisions-file texts are stale and no owner override covers them; the export and the Live "
                        "page must not ship them until the owner approves corrected wording (app chat imdai-project-d8)"}
    return {
        "status": "corrected_under_owner_window_override", "ship_allowed": True, **out,
        "label": PROPOSED_LABEL, "provenance_note": PROPOSED_NOTE,
        "corrected_label": PROPOSED_LABEL, "corrected_provenance_note": PROPOSED_NOTE,
        "superseded": {"label": label, "provenance_note": note,
                       "source": f"{P['decisions']} defaults_chosen_by_app_chat"},
        "authority": {"window.overrides_plan": w["overrides_plan"], "constraint": HONEST},
        "owner_wording_confirmed": False,
        "wording_passes_live_caveat_filter": True,
        "rule": "the superseded texts were app-chat defaults written for 1 s windows; the owner's window override orders "
                "every 1 s step corrected and the owner's constraint requires an honest provenance label, so the "
                "corrected texts ship. Only the 1 s window claims changed (to the owner's 10 s scoring window and the "
                "1-10 s training lengths in the fit rows). The decisions file is not edited. The owner has not "
                "separately confirmed this exact wording: to change it, edit PROPOSED_LABEL / PROPOSED_NOTE here (or "
                "the decisions file), rebuild this contract and re-export the bundle into a new folder.",
    }


def build() -> dict:
    schema, dec, chk = _read("schema"), _read("decisions"), _read("checks")
    seconds = int(dec["window"]["length_seconds"])
    names = schema["feature_names"]
    d = dec["defaults_chosen_by_app_chat"]
    rules = _read("rules")
    assert arms._sha(rules["value"]) == rules["sha256"]
    feats = [feature(i, n, float(seconds)) for i, n in enumerate(names)]
    vocab = {k: [f["params"]["port"] for f in feats if f["definition"] == "presence.port"
                 and f["params"]["direction"] == k] for k in ("src", "dst")}
    assert _read("rust_ports") == [f["name"] for f in feats if f["definition"] == "presence.port"]
    conv = conventions(dec, chk, seconds)
    used = {c for f in feats for c in f["conventions"]}
    assert used <= set(conv) and set(live77.EMULATION) <= set(conv), (used - set(conv))
    tr_len = sorted(int(k) for k in chk["training_fit77"]["window_duration"])
    tr = chk["training_fit77"]["window_duration"]
    per = _read("window")["per_length"]
    prov = provenance(dec, tr_len, seconds)
    label, note = prov.pop("label"), prov.pop("provenance_note")
    pending = sorted(k for k, v in conv.items() if v["status"].endswith("_pending_owner"))
    sources = {P[k]: _sha(k) for k in ("selection", "window", "decisions", "plan", "checks", "check_script", "builder")}
    sources[P["rules"]] = {"file_sha256": _sha("rules"), "value_sha256": rules["sha256"],
                           "value_sha256_rule": "sha256(json.dumps(value, sort_keys=True)), the file's own 'sha256' "
                                                "field (arms._sha); the integration plan quotes this one"}
    return {
        "contract": "live77_extraction", "version": 2, "frozen": "2026-11-03",
        "purpose": "Frozen extraction contract for the live 77-class XGBoost model: the 252 inputs in schema order, their "
                   "definitions, NaN rules and the owner decisions. Steps 2-6 (Python reference, export, Rust loader, "
                   "extractor, window) implement it. Parity with this contract is not parity with the publisher.",
        "generated_by": P["builder"],
        "model_id": d["model_id"], "label": label, "provenance_note": note,
        "provenance_review": prov,
        "publisher_equivalence_verified": False, "live_validation": False, "held_out_test_used": False,
        "model": {"checkpoint": P["checkpoint"], "checkpoint_sha256": schema["checkpoint_sha256"], "schema": P["schema"],
                  "schema_sha256": _sha("schema"), "n_features": schema["n_features"],
                  "n_classes": schema["n_classes"], "benign_index": schema["benign_index"]},
        "sources": sources,
        "feature_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "feature_names_sha256_rule": "sha256 of the names joined by '\\n' (same as selection.json)",
        "feature_names": names,
        "group_counts": {g: sum(f["group"] == g for f in feats) for g in live77.EXACT},
        "window": {
            "kind": "tumbling", "length_seconds": seconds, "nominal_duration_seconds": float(seconds),
            "display": dec["window"]["display"],
            "gate": {"nominal_seconds": float(seconds), "tolerance_seconds": 1.0,
                     "rule": f"a window is scored only when its measured duration is within {seconds} +/- 1 s (the "
                             "existing Window::counts gate, packet.rs:470); plan step 6 keeps it"},
            "decision": dec["window"]["decided"],
            "supersedes": "integration plan text for a 1 s window (window_seconds 1, nominal 1.0, 1 s period, 1.0 s gate)",
            "training_lengths": {"fit77_rows_by_length_seconds": tr, "source": P["checks"]},
            "validation_by_length": {k: {"rows": per[str(k)]["rows"], "accuracy": per[str(k)]["accuracy"],
                                         "macro_f1_common": per[str(k)]["macro_f1_common"]} for k in (1, seconds)},
            "note": f"{seconds} s rows are {_n(tr[str(seconds)])} of {_n(sum(tr.values()))} fit rows. validation_by_length "
                    "is copied from window.json: the final model on the validation tuning role (used for tuning, so "
                    "optimistic), restricted to the 40 classes present at every length, before the class merge; the "
                    "held-out test was not used.",
        },
        "admission": {"device_match": "ethernet_mac", "frames": "all_frames_for_device_mac", "ipv6": "excluded",
                      "capture_filter": "ether host <device MAC> (IPv6 removed in software); snaplen 192; no payload "
                                        "saved"},
        "zero_packet_window": {"score": True, "quality_no_packets_observed": 1},
        "damaged_window": {
            "rule": dec["rules"]["damaged_window"], "capture_drop": "all_missing_not_scored",
            "family_damage": {
                "ipv4_header": "an unreadable or truncated IPv4 header, or an incomplete link layer, makes the IPv4 "
                               "statistics and fragmented-packets missing",
                "ipv4_or_arp_header": "as ipv4_header, or an unreadable ARP body, makes the ips counts missing",
                "tcp_header": "a truncated TCP header makes the tcp-flags/window-size statistics and flag counts missing",
                "tcp_options": "a truncated TCP header or malformed TCP options makes the mss statistics missing",
                "ports": "a TCP/UDP header whose ports cannot be read makes absent port presence missing"},
            "not_damage": "non-initial IPv4 fragments, non-IP frames, IPv6 frames (excluded) and options-free TCP",
            "status": {"rule": "owner_decision", "capture_drop": "owner_decision",
                       "family_damage": "implementer_default_pending_owner"},
            "status_note": "capture drop: the current rule nulls every feature and skips the window (features.rs:567-587, "
                           "packet.rs:478). family_damage maps the current per-family voiding onto the 252 inputs; the "
                           "owner has not reviewed that mapping.",
        },
        "shap_after_merge": dec["rules"]["shap_after_merge"],
        "pending_owner_signoff": {
            "conventions": pending,
            "other": ["damaged_window.family_damage", *([] if prov["ship_allowed"] else ["provenance_review"])],
            "blocks": "plan step 5 (extractor) is not frozen until the owner signs these off" + (
                "" if prov["ship_allowed"] else "; the step-3 export does not ship the label until provenance_review "
                                                "is resolved"),
            **({"non_blocking": ["provenance_review.wording_confirmation"]}
               if prov["status"] == "corrected_under_owner_window_override" else {}),
        },
        "port_vocabulary": vocab,
        "definitions": {
            "count.frames": "number of admitted frames (all) / sent by the device MAC (src) / the rest (dst)",
            "count.ipv4_mf": "admitted IPv4 packets with (frag field & 0x2000) != 0",
            "count.tcp_flag": "admitted TCP segments (IPv4, readable TCP header) with (flags & mask) != 0",
            "distinct.ipv4_with_arp": "distinct IPv4 addresses from IPv4 headers plus ARP sender/target protocol "
                                      "addresses",
            "distinct.mac": "distinct Ethernet MAC addresses of admitted frames",
            "stat.ipv4.ihl_bytes": "IPv4 IHL * 4 in bytes (dataset values 0/20/24)",
            "stat.ipv4.flags3": "IPv4 flags, frag field >> 13 (dataset values 0..2)",
            "stat.ipv4.total_length": "IPv4 total length field",
            "stat.ipv4.ttl": "IPv4 TTL",
            "stat.tcp.flags12": "12-bit TCP flags ((tcp[12] & 0x0f) << 8) | tcp[13]",
            "stat.tcp.window_raw": "TCP window field, unscaled",
            "stat.tcp.mss_option": "MSS option value (kind 2, length 4) of TCP segments that carry it",
            "window.nominal_seconds": f"nominal window length ({seconds} s)",
            "rate.per_nominal_second": f"count / nominal window length (count / {seconds})",
            "ratio.of_packets_all": "numerator / packets_all",
            "ratio.packets_per_ip": "packets_all / ips_all",
            "quality.no_packets": "1 when packets_all == 0, else 0",
            "presence.port": "TCP/UDP header port of that direction among the kept ports",
            "presence.port_unseen": "a kept port of that direction outside the 96-port vocabulary",
        },
        "nan_rules": {
            "global_only": "never NaN except in a capture-drop window (all 252 missing, not scored)",
            "family_count": "count (0 in a zero-packet window); NaN when its damage_family is damaged in the window",
            "family_stat": "NaN when packets_all == 0; 0 when packets_all > 0 and the family has no values; NaN when "
                           "its damage_family is damaged; otherwise the statistic",
            "ratio": "NaN when the denominator is 0 or missing, or the numerator is missing",
            "port_presence": "1 when kept; else NaN when that direction is capped (> 1000 distinct) or ports are "
                             "damaged; else 0",
            "port_unseen": "1 when a kept port is outside the vocabulary (always so in a capped direction); else NaN "
                           "when ports are damaged; else 0",
        },
        "conventions": conv,
        "features": feats,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=live77.CONTRACT)
    out = ap.parse_args().out
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{out}  sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
