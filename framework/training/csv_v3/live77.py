"""Live (Npcap) 77-class feature groups over the v3 feature list.

Every one of the 445 v3 features is assigned to exactly one group, with a rating of how well a single Windows
host running Npcap can reproduce the dataset's value:

* ``exact``       same definition from packet headers (port presence needs the ``UNSEEN`` column added live);
* ``approximate`` close but not identical (unpadded sent frames, unknown publisher formula, cardinality counts);
* ``mismatch``    a different quantity live (whole-mirror time gaps, Wireshark highest-layer dissector names);
* ``excluded``    impossible live (MQTT broker logs) or an export artifact that is always 0/False live.

A live model is trained on a list of groups; columns keep their v3 order. Owner rule (2026-10-31): use features
that match the DataSense dataset; among those, keep what yields the best tuning-role results.

"exact" holds only if the live extractor copies the dataset's encoding conventions in ``EMULATION`` (found by the
2026-10-31 review, workflow wf_f8a0a9bd-02f); today's ``features.rs`` does not yet do so. The frozen per-feature
extraction contract for the deployed 252-feature model (owner decisions of 2026-11-03 included) is ``CONTRACT``.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from framework.training.csv_v3 import prep

RATING = {
    "core": "exact", "port_presence": "exact",
    "port_listed": "approximate", "port_reported": "approximate", "packet_size": "approximate",
    "payload": "approximate", "unknown_formula": "approximate",
    "time_delta": "mismatch", "protocol": "mismatch",
    "log": "excluded", "artifact": "excluded",
}
EXACT = ("core", "port_presence")
APPROXIMATE = ("port_listed", "port_reported", "packet_size", "payload", "unknown_formula")
MISMATCH = ("time_delta", "protocol")

# Dataset conventions a live extractor must reproduce for the "exact" rating (evidence: fit77 rows, review 2026-10-31).
EMULATION = {
    "empty_family_zero": "tcp-flags/window-size/mss and ttl/ip-length/header-length/ip-flags statistics are 0 (not "
                         "NaN) when the window has packets but none of that family; NaN only when packets_all == 0",
    # Corrected 2026-11-03 (was mss_absent_zero, "a TCP packet without an MSS option contributes mss 0"): 0 of 17,629
    # sampled fit77 rows with mss_max > 0 have mss_min == 0, so absent options never enter the statistics.
    "mss_bearing_only": "mss statistics use MSS-bearing TCP packets only; a packet without the option adds nothing; "
                        "0 when no packet carries the option and packets_all > 0 (empty_family_zero)",
    "fragment_mf_only": "fragmented-packets counts packets with the IP MF bit set (not offset-only last fragments)",
    "fragments_keep_stats": "non-initial fragments must not void TCP statistics or port presence for the window",
    "arp_ips": "ARP sender/target IPv4 addresses count in ips_* sets and packets_per_reported_ip",
    "port_cap_1000": "per direction, the first 1000 distinct ports in arrival order are kept; once more than 1000 are "
                     "seen, port presence is 1 for kept vocabulary ports and NaN otherwise (dataset lists are capped "
                     "at 1000); later ports set nothing; listed_distinct is clamped at 1000 per direction; "
                     "descriptors on capped lists stay approximate",
    "port_unseen": "emit ports_{src,dst}__present_UNSEEN (an observed port outside the vocabulary)",
    "tcp_flags_12bit": "tcp-flags statistics use the 12-bit field including the reserved bits "
                       "((tcp[12] & 0x0f) << 8 | tcp[13]); training rows reach 2242",
}
# Per-feature definitions for the deployed live model; rebuilt only by experiment/build_live77_contract.py.
CONTRACT = Path(__file__).with_name("live77_contract.json")


def group_of(name: str) -> str:
    if name.startswith("log_") or name == "quality_no_logs_observed":
        return "log"
    if "reported_minus_listed" in name or "membership_incomplete" in name:
        return "artifact"
    if "protocols" in name:
        return "protocol"
    if "time-delta" in name or name == "quality_negative_packet_delta":
        return "time_delta"
    if "payload" in name:
        return "payload"
    if name in ("network_interval-packets", "network_fragmentation-score"):
        return "unknown_formula"
    if "packet-size" in name or name == "network_packet_size_coefficient_of_variation":
        return "packet_size"
    if "ports" in name and "__present" in name:
        return "port_presence"
    if name in ("network_ports_all_count", "network_ports_dst_count", "network_ports_src_count",
                "network_packets_per_reported_port"):
        return "port_reported"
    if "__listed_distinct" in name or name.startswith("listed_"):
        return "port_listed"
    return "core"


@lru_cache(maxsize=1)
def _names() -> tuple[str, ...]:
    m = json.loads((prep.DEFAULT_OUT / "manifest.json").read_text(encoding="utf-8"))
    return tuple(m["feature_names"])


def groups() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {g: [] for g in RATING}
    for n in _names():
        out[group_of(n)].append(n)
    return out


def columns(selected: list[str] | tuple[str, ...]) -> np.ndarray:
    """v3 column indexes of the selected groups, in v3 order. Excluded groups are refused."""
    bad = [g for g in selected if g not in RATING or RATING[g] == "excluded"]
    if bad:
        raise ValueError(f"groups not allowed in a live model: {bad}")
    keep = set(selected)
    return np.array([i for i, n in enumerate(_names()) if group_of(n) in keep], dtype=np.int64)


def feature_names(selected: list[str] | tuple[str, ...]) -> list[str]:
    names = _names()
    return [names[i] for i in columns(selected)]
