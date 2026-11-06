"""Python reference (oracle) for the live77 extraction contract (integration plan step 2).

Two layers, both driven by ``live77_contract.json`` (``live77.CONTRACT``):

* Layer A, :func:`vector`: list-level window facts -> the 252 model inputs in schema order (float32, NaN = missing).
  It reuses the dataset encoding (``csv_redesign.features`` and ``verify_clean_datasense.recompute_derived``):
  ratio NaN when the denominator is <= 0 or either side is missing; per_second = count / duration; port presence
  with the 1000 cap (NaN) and UNSEEN; quality_no_packets_observed. :func:`validation_parity` proves Layer A on the
  30,000 validation-partition rows against ``X_validation.npy``; the held-out test is never opened.
* Layer B, :func:`window_facts`: Ethernet frames of one tumbling window -> window facts, with the contract's packet
  conventions (device-MAC admission, IPv6 excluded, ARP IPs, MF-only fragments, 12-bit TCP flags, MSS-bearing
  packets only, fragments keep statistics, first 1000 distinct ports per direction in arrival order).

Parity with this reference is parity with the contract, not with the publisher's exporter (unpublished). Conventions
the contract marks ``*_pending_owner`` are implemented as written there and stay pending. Three choices the contract
leaves open are reference defaults (fixture tag in brackets): VLAN tags are followed as in packet.rs:118-135
[link_vlan]; damage propagates as in packet.rs:180-187 (an unreadable link layer or IPv4 header also makes the
transport families of that packet unmeasurable) [damage_propagation]; a frame is admitted before IPv6 is removed,
and a frame shorter than an Ethernet header is not admitted [admission_order].

Float rules (Rust step 5 must copy them): statistics are computed in float64 over the window's samples in arrival
order (mean = exact integer sum / n; population std = sqrt(sum((x - mean)^2) / n), two passes) and cast to float32;
a ratio is float32(float64(float32(numerator)) / float64(float32(denominator))).
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import re
from pathlib import Path

import numpy as np

from framework.training.csv_v3 import live77, prep

CONTRACT = json.loads(live77.CONTRACT.read_text(encoding="utf-8"))
NAMES: list[str] = list(CONTRACT["feature_names"])
FEATURES: list[dict] = CONTRACT["features"]
VOCAB: dict[str, tuple[int, ...]] = {d: tuple(CONTRACT["port_vocabulary"][d]) for d in ("src", "dst")}
NOMINAL_SECONDS = float(CONTRACT["window"]["nominal_duration_seconds"])
PORT_LIMIT = int(CONTRACT["conventions"]["port_cap_1000"]["params"]["limit"])
MF_MASK = int(CONTRACT["conventions"]["fragment_mf_only"]["params"]["ipv4_frag_mask"])
FLAG_MASK = {f["params"]["flag"]: f["params"]["mask"] for f in FEATURES if f["definition"] == "count.tcp_flag"}
FLAGS = tuple(FLAG_MASK)
STATS = ("avg", "max", "min", "std_deviation")
_STAT = {"mean": "avg", "max": "max", "min": "min", "std_population": "std_deviation"}
FAMILY_OF = {"stat.ipv4.ihl_bytes": "header-length", "stat.ipv4.flags3": "ip-flags",
             "stat.ipv4.total_length": "ip-length", "stat.ipv4.ttl": "ttl", "stat.tcp.flags12": "tcp-flags",
             "stat.tcp.window_raw": "window-size", "stat.tcp.mss_option": "mss"}
FAMILIES = tuple(FAMILY_OF.values())
DAMAGE_OF = {"header-length": "ipv4_header", "ip-flags": "ipv4_header", "ip-length": "ipv4_header",
             "ttl": "ipv4_header", "tcp-flags": "tcp_header", "window-size": "tcp_header", "mss": "tcp_options"}
DAMAGE = ("ipv4_header", "ipv4_or_arp_header", "tcp_header", "tcp_options", "ports")

DATA = prep.DEFAULT_OUT
ALLOWED = {DATA / n for n in ("manifest.json", "X_validation.npy", "row_ids_validation.npy",
                              "selected_validation_77class.csv.gz")}


def guarded(p: Path) -> Path:
    """Refuse any path naming a test partition or outside the read-only validation allow-list."""
    p = Path(p)
    if re.search(r"(^|[^a-z])test", p.name, re.I) or p not in ALLOWED:
        raise PermissionError(f"refused (held-out test, or not on the validation allow-list): {p}")
    return p


# ---------------------------------------------------------------------------------------------- Layer A
def _num(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x


def _ratio(num, den) -> float:
    num, den = _num(num), _num(den)
    if num is None or den is None or den <= 0:
        return math.nan
    return float(np.float32(float(np.float32(num)) / float(np.float32(den))))


def _get(d, k):
    return None if d is None else d.get(k)


def vector(facts: dict) -> np.ndarray:
    """252 float32 inputs in schema order from window facts (see :func:`window_facts` for the shape)."""
    if facts.get("capture_drop"):
        return np.full(len(NAMES), np.nan, dtype=np.float32)
    pk = facts["packets"]
    packets = _num(pk.get("all"))
    dur = facts["duration_seconds"]
    out: dict[str, float] = {}
    for f in FEATURES:
        name, d, p = f["name"], f["definition"], f.get("params", {})
        if d == "count.frames":
            v = _num(pk.get(p["direction"]))
        elif d == "count.ipv4_mf":
            v = _num(facts.get("fragmented"))
        elif d == "count.tcp_flag":
            v = _num(_get(facts.get("tcp_flags"), p["flag"]))
        elif d == "distinct.ipv4_with_arp":
            v = _num(_get(facts.get("ips"), p["direction"]))
        elif d == "distinct.mac":
            v = _num(_get(facts.get("macs"), p["direction"]))
        elif d in FAMILY_OF:
            s = facts["stats"].get(FAMILY_OF[d])
            # NaN when packets_all == 0; 0 when the family is empty but packets exist; NaN when damaged.
            v = None if packets == 0 or s is None else 0 if s == "empty" else _num(s.get(_STAT[p["stat"]]))
        elif d == "presence.port":
            q = facts["ports"][p.get("direction") or name.split("_")[2]]
            port = int(name.rsplit("_", 1)[1])
            v = 1 if port in q["kept"] else None if q["capped"] or q["damaged"] else 0
        elif d == "presence.port_unseen":
            q = facts["ports"][p["direction"]]
            v = 1 if any(x not in VOCAB[p["direction"]] for x in q["kept"]) else None if q["damaged"] else 0
        elif d == "window.nominal_seconds":
            v = _num(dur)
        elif d == "quality.no_packets":
            v = None if packets is None else float(packets == 0)
        elif d == "rate.per_nominal_second":
            v = _ratio(out_raw(facts, p["numerator"]), dur)
        elif d == "ratio.of_packets_all":
            v = _ratio(out_raw(facts, p["numerator"]), packets)
        elif d == "ratio.packets_per_ip":
            v = _ratio(packets, out_raw(facts, p["denominator"]))
        else:
            raise ValueError(f"unknown contract definition {d} for {name}")
        out[name] = math.nan if v is None else float(v)
    return np.array([out[n] for n in NAMES], dtype=np.float32)


def out_raw(facts: dict, name: str):
    """A count input by its feature name, before the float32 cast (numerator/denominator of a ratio)."""
    m = re.fullmatch(r"network_(packets|ips)_(all|src|dst)_count", name)
    if m:
        return _get(facts.get("packets" if m[1] == "packets" else "ips"), m[2])
    m = re.fullmatch(r"network_tcp-flags-(\w+)_count", name)
    if m:
        return _get(facts.get("tcp_flags"), m[1])
    raise ValueError(f"not a count input: {name}")


def summarize(samples: list[int]):
    """Window statistics over samples in arrival order; ``"empty"`` when there are none."""
    n = len(samples)
    if n == 0:
        return "empty"
    mean = sum(samples) / n  # exact integer sum, one correctly rounded division
    acc = 0.0
    for x in samples:
        dev = x - mean
        acc += dev * dev
    return {"avg": mean, "max": max(samples), "min": min(samples), "std_deviation": math.sqrt(acc / n)}


# ------------------------------------------------------------------------------- Layer A: dataset rows
def dataset_facts(row: dict) -> dict:
    """Window facts of one dataset CSV row, after the dataset's own row cleaning (blank cell = missing)."""
    from framework.training.csv_redesign.features import _clean, _number

    clean, rf, observed, _ = _clean(row)
    incomplete = set(rf["incomplete_fields"])

    def num(k):
        v = _number(clean[k])
        return None if math.isnan(v) else v

    dst_macs = observed["network_macs_dst"]
    return {
        "duration_seconds": rf["duration"],
        "packets": {d: num(f"network_packets_{d}_count") for d in ("all", "src", "dst")},
        "fragmented": num("network_fragmented-packets"),
        "tcp_flags": {f: num(f"network_tcp-flags-{f}_count") for f in FLAGS},
        "ips": {d: num(f"network_ips_{d}_count") for d in ("all", "src", "dst")},
        "macs": {"all": num("network_macs_all_count"), "src": num("network_macs_src_count"),
                 "dst": None if dst_macs is None else len(dst_macs)},
        "stats": {fam: {s: num(f"network_{fam}_{s}") for s in STATS} for fam in FAMILIES},
        "ports": {d: {"kept": [int(x) for x in observed[f"network_ports_{d}"] or ()],
                      "capped": f"network_ports_{d}" in incomplete,
                      "damaged": observed[f"network_ports_{d}"] is None} for d in ("src", "dst")},
    }


def validation_parity() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Layer A on the raw validation CSV, X_validation[:, EXACT columns], row ids), aligned by row id."""
    names = json.loads(guarded(DATA / "manifest.json").read_text(encoding="utf-8"))["feature_names"]
    cols = live77.columns(list(live77.EXACT))
    if [names[i] for i in cols] != NAMES:
        raise RuntimeError("X_validation columns do not follow the contract order")
    X = np.asarray(np.load(guarded(DATA / "X_validation.npy"), mmap_mode="r")[:, cols], dtype=np.float32)
    ids = np.load(guarded(DATA / "row_ids_validation.npy"))
    pos = {int(r): i for i, r in enumerate(ids)}
    got = np.full(X.shape, np.nan, dtype=np.float32)
    seen = np.zeros(len(ids), dtype=bool)
    with gzip.open(guarded(DATA / "selected_validation_77class.csv.gz"), "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            i = pos[int(row["source_row_id"])]
            if seen[i]:
                raise RuntimeError(f"duplicate validation row {row['source_row_id']}")
            seen[i] = True
            got[i] = vector(dataset_facts(row))
    if not seen.all():
        raise RuntimeError(f"{int((~seen).sum())} validation rows missing from the raw CSV")
    return got, X, ids


# ---------------------------------------------------------------------------------------------- Layer B
def _mac(text: str) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{2}(:[0-9a-f]{2}){5}", text.lower()):
        raise ValueError(f"invalid MAC address: {text!r}")
    return bytes.fromhex(text.replace(":", ""))


def _r16(b: bytes, o: int):
    return int.from_bytes(b[o:o + 2], "big") if len(b) >= o + 2 else None


class _Window:
    def __init__(self):
        self.packets = {"all": 0, "src": 0, "dst": 0}
        self.macs = {"src": set(), "dst": set()}
        self.ips = {"src": set(), "dst": set()}
        self.fragmented = 0
        self.flags = {f: 0 for f in FLAGS}
        self.samples = {fam: [] for fam in FAMILIES}
        self.ports = {d: {"kept": [], "seen": set(), "capped": False} for d in ("src", "dst")}
        self.damage: set[str] = set()

    def damage_transport(self, protocol):
        """An unmeasurable transport header: TCP families when it is (or may be) TCP, ports for TCP/UDP."""
        if protocol in (None, 6):
            self.damage |= {"tcp_header", "tcp_options"}
        if protocol in (None, 6, 17):
            self.damage.add("ports")

    def port(self, direction: str, port: int):
        q = self.ports[direction]
        if port in q["seen"]:
            return
        if len(q["kept"]) >= PORT_LIMIT:  # first 1000 distinct in arrival order; later ports set nothing
            q["capped"] = True
            return
        q["seen"].add(port)
        q["kept"].append(port)


def _ipv4(w: _Window, b: bytes, o: int, wire: int):
    ip = b[o:]
    if len(ip) < 20 or ip[0] >> 4 != 4:
        w.damage |= {"ipv4_header", "ipv4_or_arp_header"}
        w.damage_transport(None)
        return
    protocol, ihl, total = ip[9], (ip[0] & 15) * 4, _r16(ip, 2)
    if ihl < 20 or ihl > len(ip) or total < ihl or total > wire - o:
        w.damage |= {"ipv4_header", "ipv4_or_arp_header"}
        w.damage_transport(protocol)
        return
    frag = _r16(ip, 6)
    w.ips["src"].add(".".join(map(str, ip[12:16])))
    w.ips["dst"].add(".".join(map(str, ip[16:20])))
    for fam, v in (("header-length", ihl), ("ip-length", total), ("ip-flags", frag >> 13), ("ttl", ip[8])):
        w.samples[fam].append(v)
    w.fragmented += bool(frag & MF_MASK)
    if frag & 0x1FFF:  # non-initial fragment: IPv4 statistics only, nothing voided
        return
    start, available = o + ihl, total - ihl
    if protocol == 6:
        tcp = b[start:start + 20]
        if len(tcp) < 20 or available < 20:
            w.damage_transport(6)
            return
        size = (tcp[12] >> 4) * 4
        if size < 20 or size > available or len(b) < start + size:
            w.damage_transport(6)
            return
        w.samples["tcp-flags"].append(((tcp[12] & 0x0F) << 8) | tcp[13])
        for f, m in FLAG_MASK.items():
            w.flags[f] += bool(tcp[13] & m)
        w.samples["window-size"].append(_r16(tcp, 14))
        w.port("src", _r16(tcp, 0))
        w.port("dst", _r16(tcp, 2))
        mss, malformed, i = None, False, start + 20
        while i < start + size:  # TCP options as in packet.rs:358-382
            kind = b[i]
            if kind == 0:
                break
            if kind == 1:
                i += 1
                continue
            length = b[i + 1] if i + 1 < len(b) else 0
            if length < 2 or i + length > start + size:
                malformed = True
                break
            if kind == 2:
                if length != 4 or mss is not None:
                    malformed = True
                    break
                mss = _r16(b, i + 2)
            i += length
        if malformed:
            w.damage.add("tcp_options")
        elif mss is not None:
            w.samples["mss"].append(mss)
    elif protocol == 17:
        udp = b[start:start + 8]
        if len(udp) < 8 or available < 8:
            w.damage_transport(17)
            return
        length = _r16(udp, 4)
        if length < 8 or (length > available and not frag & MF_MASK):
            w.damage_transport(17)
            return
        w.port("src", _r16(udp, 0))
        w.port("dst", _r16(udp, 2))


def window_facts(frames: list[dict], device_mac: str, *, capture_drop: bool = False,
                 nominal_seconds: float = NOMINAL_SECONDS) -> dict:
    """Window facts of one tumbling window from its frames ``[{"t": s, "hex": captured bytes, "wire_length"?}]``.

    ``t`` is seconds from the window start and must be non-decreasing (arrival order sets the kept ports).
    """
    dev = _mac(device_mac)
    w = _Window()
    last = -math.inf
    for fr in frames:
        b = bytes.fromhex(fr["hex"])
        wire = int(fr.get("wire_length", len(b)))
        if wire < len(b) or not fr["t"] >= last:
            raise ValueError("frame captured beyond its wire length, or frames out of arrival order")
        last = fr["t"]
        if len(b) < 14:
            continue  # no readable Ethernet header: the device MAC cannot match [admission_order]
        dst_mac, src_mac = b[0:6], b[6:12]
        if dev not in (src_mac, dst_mac):  # device MAC only; broadcasts to others are not admitted
            continue
        o, kind, link_ok = 14, _r16(b, 12), True
        for _ in range(2):  # 802.1Q / 802.1ad tags as packet.rs:118-135 [link_vlan]
            if kind not in (0x8100, 0x88A8):
                break
            nxt = _r16(b, o + 2)
            if nxt is None:
                link_ok = False
                break
            kind, o = nxt, o + 4
        if kind in (0x8100, 0x88A8):
            link_ok = False
        if link_ok and kind == 0x86DD:
            continue  # IPv6 frames are left out of the model window entirely
        direction = "src" if src_mac == dev else "dst"
        w.packets["all"] += 1
        w.packets[direction] += 1
        w.macs["src"].add(src_mac.hex(":"))
        w.macs["dst"].add(dst_mac.hex(":"))
        if not link_ok:
            w.damage |= {"ipv4_header", "ipv4_or_arp_header"}
            w.damage_transport(None)
        elif kind == 0x0800:
            _ipv4(w, b, o, wire)
        elif kind == 0x0806:
            arp = b[o:o + 28]
            if len(arp) < 28:
                w.damage.add("ipv4_or_arp_header")
            elif _r16(arp, 2) == 0x0800 and arp[4] == 6 and arp[5] == 4:
                w.ips["src"].add(".".join(map(str, arp[14:18])))  # sender protocol address
                w.ips["dst"].add(".".join(map(str, arp[24:28])))  # target protocol address
    if capture_drop:
        return {"capture_drop": True}
    dmg = w.damage
    ips_ok = "ipv4_or_arp_header" not in dmg
    return {
        "capture_drop": False,
        "duration_seconds": float(nominal_seconds),
        "packets": dict(w.packets),
        "fragmented": None if "ipv4_header" in dmg else w.fragmented,
        "tcp_flags": None if "tcp_header" in dmg else dict(w.flags),
        "ips": {"all": len(w.ips["src"] | w.ips["dst"]) if ips_ok else None,
                "src": len(w.ips["src"]) if ips_ok else None, "dst": len(w.ips["dst"]) if ips_ok else None},
        "macs": {"all": len(w.macs["src"] | w.macs["dst"]), "src": len(w.macs["src"]), "dst": len(w.macs["dst"])},
        "stats": {fam: None if DAMAGE_OF[fam] in dmg else summarize(w.samples[fam]) for fam in FAMILIES},
        "ports": {d: {"kept": list(q["kept"]), "capped": q["capped"], "damaged": "ports" in dmg}
                  for d, q in w.ports.items()},
        "damage": sorted(dmg),
    }


def frames_vector(frames: list[dict], device_mac: str, *, capture_drop: bool = False) -> np.ndarray:
    return vector(window_facts(frames, device_mac, capture_drop=capture_drop))
