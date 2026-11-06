"""Write frames_v1.json: shared live77 frame fixtures with hand-computed expected vectors (integration plan step 2).

Run: .venv/Scripts/python.exe -m evaluation.fixtures.live77.build_frames_v1

Frames are packed field by field below, so every byte can be audited against its ``desc``. Expected vectors are
literals worked out by hand from the contract (framework/training/csv_v3/live77_contract.json); this script never
imports the Python reference, so the reference and the Rust extractor (plan step 5) are both tested against them.
Expected vectors are sparse: every input is ``default`` (0 unless given), then names under ``nan_prefixes`` and
``nan`` are NaN, then ``values`` override. ``window_duration_seconds`` is always the nominal 10 s.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name("frames_v1.json")
CONTRACT = json.loads((ROOT / "framework/training/csv_v3/live77_contract.json").read_text(encoding="utf-8"))
REFERENCE_DEFAULTS = {"link_vlan", "damage_propagation", "admission_order"}

DEV, GW, OTHER, BCAST = "02:00:00:00:00:50", "02:00:00:00:00:01", "02:00:00:00:00:07", "ff:ff:ff:ff:ff:ff"
LLDP_MC, V6_MC = "01:80:c2:00:00:0e", "33:33:00:00:00:01"
DEV_IP, GW_IP, LAN_GW_IP, OTHER_IP, REMOTE = "192.168.1.50", "10.0.0.1", "192.168.1.1", "192.168.1.7", "10.0.0.1"
SYN, ACK, RST = 0x02, 0x10, 0x04


# ----------------------------------------------------------------------------------------- frame packing
def mac(t: str) -> bytes:
    return bytes.fromhex(t.replace(":", ""))


def ip4(t: str) -> bytes:
    return bytes(int(x) for x in t.split("."))


def checksum(b: bytes) -> int:
    s = sum(struct.unpack(f"!{len(b) // 2}H", b + b"\0" * (len(b) % 2)))
    while s >> 16:
        s = (s & 0xFFFF) + (s >> 16)
    return ~s & 0xFFFF


def eth(dst: str, src: str, ethertype: int, payload: bytes, vlan: int | None = None, pad: bool = False) -> bytes:
    tag = struct.pack("!HH", 0x8100, vlan) if vlan is not None else b""
    f = mac(dst) + mac(src) + tag + struct.pack("!H", ethertype) + payload
    return f + b"\0" * (60 - len(f)) if pad and len(f) < 60 else f


def ipv4(src: str, dst: str, proto: int, payload: bytes, ttl: int = 64, frag: int = 0, ident: int = 1) -> bytes:
    h = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + len(payload), ident, frag, ttl, proto, 0, ip4(src), ip4(dst))
    return h[:10] + struct.pack("!H", checksum(h)) + h[12:] + payload


def tcp(sport: int, dport: int, flags: int, window: int, options: bytes = b"", byte12: int | None = None) -> bytes:
    assert len(options) % 4 == 0
    off = byte12 if byte12 is not None else (5 + len(options) // 4) << 4
    return struct.pack("!HHIIBBHHH", sport, dport, 1000, 0, off, flags, window, 0, 0) + options


def udp(sport: int, dport: int, payload: bytes = b"", length: int | None = None) -> bytes:
    return struct.pack("!HHHH", sport, dport, length if length is not None else 8 + len(payload), 0) + payload


def arp(op: int, sha: str, spa: str, tha: str, tpa: str) -> bytes:
    return struct.pack("!HHBBH", 1, 0x0800, 6, 4, op) + mac(sha) + ip4(spa) + mac(tha) + ip4(tpa)


def lldp() -> bytes:
    tlv = lambda t, v: struct.pack("!H", (t << 9) | len(v)) + v  # noqa: E731
    return tlv(1, b"\x04" + mac(DEV)) + tlv(2, b"\x03" + mac(DEV)) + tlv(3, struct.pack("!H", 120)) + tlv(0, b"")


def ipv6_tcp_syn() -> bytes:
    seg = tcp(51000, 80, SYN, 64800, options=bytes([2, 4, 5, 0xA0]))
    src = bytes.fromhex("fe800000000000000000000000000050")
    dst = bytes.fromhex("ff020000000000000000000000000001")
    return struct.pack("!IHBB", 0x60000000, len(seg), 6, 64) + src + dst + seg


def frame(t: float, b: bytes, desc: str, wire_length: int | None = None) -> dict:
    out = {"t": t, "desc": desc, "hex": b.hex()}
    if wire_length is not None:
        out["wire_length"] = wire_length
    return out


MSS1460 = bytes([2, 4, 0x05, 0xB4])
MSS536 = bytes([2, 4, 0x02, 0x18])


# ----------------------------------------------------------------------------------------- expected helpers
def stats(fam: str, avg, mx, mn, std) -> dict:
    return {f"network_{fam}_avg": avg, f"network_{fam}_max": mx, f"network_{fam}_min": mn,
            f"network_{fam}_std_deviation": std}


def counts(group: str, all_, src, dst) -> dict:
    name = lambda d: "network_macs_dst__observed_distinct_count" if (group, d) == ("macs", "dst") else f"network_{group}_{d}_count"  # noqa: E731,E501
    return {name("all"): all_, name("src"): src, name("dst"): dst}


def rates(all_, src, dst) -> dict:
    return {"network_packets_all_count__per_second": all_, "network_packets_src_count__per_second": src,
            "network_packets_dst_count__per_second": dst}


BASE = {"window_duration_seconds": 10.0}
IPV4_FAMS = ("header-length", "ip-flags", "ip-length", "ttl")
TCP_PREFIXES = ["network_tcp-flags", "network_window-size_"]


def fixtures() -> list[dict]:
    fx = []

    def add(id_, conventions, source, desc, frames, expected, capture_drop=False):
        fx.append({"id": id_, "conventions": conventions, "source": source, "desc": desc,
                   "capture_drop": capture_drop, "frames": frames, "expected": expected})

    # 1. One SYN carrying MSS 1460 (sent) and one RST/ACK without options (received).
    add("tcp_syn_with_and_without_mss",
        ["mss_bearing_only", "population_std", "header_direction", "device_mac_packet_direction",
         "device_mac_admission", "port_unseen", "nominal_duration"],
        "contract conventions.mss_bearing_only (replaces EMULATION mss_absent_zero); population_std; Rust "
        "features.rs already keeps MSS-bearing packets only",
        "a sent SYN with MSS 1460 and a received RST/ACK without options: MSS statistics use the one MSS-bearing "
        "packet; header src/dst give both directions in every set; per_second = count / 10",
        [frame(0.10, eth(GW, DEV, 0x0800, ipv4(DEV_IP, REMOTE, 6, tcp(51000, 80, SYN, 64240, MSS1460), frag=0x4000)),
               "sent: 192.168.1.50:51000 -> 10.0.0.1:80 SYN, MSS 1460, DF, ttl 64, window 64240, total 44"),
         frame(0.20, eth(DEV, GW, 0x0800, ipv4(REMOTE, DEV_IP, 6, tcp(80, 51000, RST | ACK, 0), ttl=128), pad=True),
               "received (padded to 60): 10.0.0.1:80 -> 192.168.1.50:51000 RST/ACK, no options, ttl 128, window 0, "
               "total 40")],
        {"values": {**BASE, **stats("header-length", 20, 20, 20, 0), **stats("ip-flags", 1, 2, 0, 1),
                    **stats("ip-length", 42, 44, 40, 2), **stats("ttl", 96, 128, 64, 32),
                    **stats("tcp-flags", 11, 20, 2, 9), **stats("window-size", 32120, 64240, 0, 32120),
                    **stats("mss", 1460, 1460, 1460, 0), **counts("packets", 2, 1, 1), **counts("ips", 2, 2, 2),
                    **counts("macs", 2, 2, 2), "network_tcp-flags-ack_count": 1, "network_tcp-flags-rst_count": 1,
                    "network_tcp-flags-syn_count": 1, "network_ports_src__present_80": 1,
                    "network_ports_src__present_UNSEEN": 1, "network_ports_dst__present_80": 1,
                    "network_ports_dst__present_UNSEEN": 1, **rates(0.2, 0.1, 0.1),
                    "network_tcp-flags-ack__fraction_of_packets": 0.5,
                    "network_tcp-flags-rst__fraction_of_packets": 0.5,
                    "network_tcp-flags-syn__fraction_of_packets": 0.5, "network_src_packet_fraction": 0.5,
                    "network_dst_packet_fraction": 0.5, "network_packets_per_reported_ip": 1}})

    # 2. TCP without any MSS option: the MSS family is empty, so 0.
    add("tcp_ack_without_mss", ["mss_bearing_only", "empty_family_zero"],
        "contract conventions.mss_bearing_only + empty_family_zero",
        "one sent ACK without options: MSS statistics are 0 (packets > 0, no MSS-bearing packet)",
        [frame(0.5, eth(GW, DEV, 0x0800, ipv4(DEV_IP, REMOTE, 6, tcp(51000, 443, ACK, 502), frag=0x4000)),
               "sent: 192.168.1.50:51000 -> 10.0.0.1:443 ACK, no options, DF, ttl 64, window 502, total 40")],
        {"values": {**BASE, **stats("header-length", 20, 20, 20, 0), **stats("ip-flags", 2, 2, 2, 0),
                    **stats("ip-length", 40, 40, 40, 0), **stats("ttl", 64, 64, 64, 0),
                    **stats("tcp-flags", 16, 16, 16, 0), **stats("window-size", 502, 502, 502, 0),
                    **counts("packets", 1, 1, 0), **counts("ips", 2, 1, 1), **counts("macs", 2, 1, 1),
                    "network_tcp-flags-ack_count": 1, "network_ports_src__present_UNSEEN": 1,
                    "network_ports_dst__present_443": 1, **rates(0.1, 0.1, 0),
                    "network_tcp-flags-ack__fraction_of_packets": 1, "network_src_packet_fraction": 1,
                    "network_packets_per_reported_ip": 0.5}})

    # 3. ARP-only window: request (sent, broadcast) and reply (received).
    add("arp_only_window", ["arp_ips", "arp_ip_direction", "empty_family_zero", "all_frames_counted", "header_direction"],
        "contract conventions.arp_ips (dataset-verified) and arp_ip_direction (pending owner); current Rust "
        "features.rs:258-268 ignores ARP addresses",
        "ARP who-has 192.168.1.1 from the device to broadcast, and the gateway's reply: ARP sender/target IPv4 "
        "count as src/dst; IPv4 and TCP families are empty, so 0",
        [frame(1.0, eth(BCAST, DEV, 0x0806, arp(1, DEV, DEV_IP, "00:00:00:00:00:00", LAN_GW_IP)),
               "sent: ARP request, sender 02:..:50/192.168.1.50, target 192.168.1.1, to ff:ff:ff:ff:ff:ff"),
         frame(1.001, eth(DEV, GW, 0x0806, arp(2, GW, LAN_GW_IP, DEV, DEV_IP), pad=True),
               "received (padded to 60): ARP reply, sender 02:..:01/192.168.1.1, target 02:..:50/192.168.1.50")],
        {"values": {**BASE, **counts("packets", 2, 1, 1), **counts("ips", 2, 2, 2), **counts("macs", 3, 2, 2),
                    **rates(0.2, 0.1, 0.1), "network_src_packet_fraction": 0.5, "network_dst_packet_fraction": 0.5,
                    "network_packets_per_reported_ip": 1}})

    # 4. ARP probe: sender 0.0.0.0 is counted literally.
    add("arp_probe_zero_sender", ["arp_ip_direction", "arp_ips"],
        "contract conventions.arp_ip_direction (implementer default pending owner: 0.0.0.0 counted literally)",
        "an ARP probe from the device (sender 0.0.0.0, target 192.168.1.50): 0.0.0.0 is one src IP",
        [frame(1.5, eth(BCAST, DEV, 0x0806, arp(1, DEV, "0.0.0.0", "00:00:00:00:00:00", DEV_IP)),
               "sent: ARP probe, sender 0.0.0.0, target 192.168.1.50, to broadcast")],
        {"values": {**BASE, **counts("packets", 1, 1, 0), **counts("ips", 2, 1, 1), **counts("macs", 2, 1, 1),
                    **rates(0.1, 0.1, 0), "network_src_packet_fraction": 1, "network_packets_per_reported_ip": 0.5}})

    # 5. UDP datagram in three fragments.
    f1 = udp(5000, 53, b"\x11" * 16, length=48)
    add("udp_fragment_train", ["fragment_mf_only", "fragments_keep_stats", "population_std"],
        "contract conventions.fragment_mf_only (owner decision; current Rust packet.rs:225 counts MF-or-offset) and "
        "fragments_keep_stats (plan default pending owner; current Rust voids ports at features.rs:359/376-378)",
        "one 48-byte UDP datagram (5000 -> 53) in 3 fragments (MF/offset 0, MF/offset 3, offset 5): 2 MF packets; "
        "every fragment adds IPv4 statistics; ports come from the first fragment and nothing is voided",
        [frame(2.0, eth(GW, DEV, 0x0800, ipv4(DEV_IP, REMOTE, 17, f1, frag=0x2000, ident=0x1234)),
               "sent: fragment 1, MF, offset 0, UDP header (length 48) + 16 bytes, total 44"),
         frame(2.001, eth(GW, DEV, 0x0800, ipv4(DEV_IP, REMOTE, 17, b"\x22" * 16, frag=0x2003, ident=0x1234)),
               "sent: fragment 2, MF, offset 3 (24 bytes), 16 bytes, total 36"),
         frame(2.002, eth(GW, DEV, 0x0800, ipv4(DEV_IP, REMOTE, 17, b"\x33" * 8, frag=0x0005, ident=0x1234)),
               "sent: fragment 3, last, offset 5 (40 bytes), 8 bytes, total 28")],
        {"values": {**BASE, "network_fragmented-packets": 2, **stats("header-length", 20, 20, 20, 0),
                    **stats("ip-flags", 2 / 3, 1, 0, (2 / 9) ** 0.5),
                    **stats("ip-length", 36, 44, 28, (128 / 3) ** 0.5), **stats("ttl", 64, 64, 64, 0),
                    **counts("packets", 3, 3, 0), **counts("ips", 2, 1, 1), **counts("macs", 2, 1, 1),
                    "network_ports_src__present_UNSEEN": 1, "network_ports_dst__present_53": 1,
                    **rates(0.3, 0.3, 0), "network_src_packet_fraction": 1, "network_packets_per_reported_ip": 1.5}})

    # 6. More than 1000 distinct destination ports.
    dports = [80] + list(range(20000, 20999)) + [443, 80]
    frames = [frame(round(0.001 * i, 3), eth(GW, DEV, 0x0800, ipv4(DEV_IP, REMOTE, 17, udp(40000, p), ident=i + 1)),
                    f"sent: UDP 192.168.1.50:40000 -> 10.0.0.1:{p}, no payload, total 28") for i, p in enumerate(dports)]
    add("ports_over_1000", ["port_cap_1000", "port_unseen"],
        "contract conventions.port_cap_1000 (owner decision: first 1000 distinct in arrival order) and port_unseen; "
        "current Rust SET_LIMIT is 65,536 (features.rs:9)",
        "1002 sent UDP datagrams to dst ports 80, 20000..20998 (1000 distinct so far), then 443 (1001st distinct, "
        "not kept) and 80 again: dst presence is 1 for kept vocabulary port 80 and NaN for every other dst "
        "vocabulary port (443 included); dst UNSEEN 1",
        frames,
        {"nan_prefixes": ["network_ports_dst__present_"],
         "values": {**BASE, **stats("header-length", 20, 20, 20, 0), **stats("ip-length", 28, 28, 28, 0),
                    **stats("ttl", 64, 64, 64, 0), **counts("packets", 1002, 1002, 0), **counts("ips", 2, 1, 1),
                    **counts("macs", 2, 1, 1), "network_ports_src__present_UNSEEN": 1,
                    "network_ports_dst__present_80": 1, "network_ports_dst__present_UNSEEN": 1,
                    **rates(100.2, 100.2, 0), "network_src_packet_fraction": 1, "network_dst_packet_fraction": 0,
                    "network_packets_per_reported_ip": 501}})

    # 7. Reserved TCP header bits.
    add("reserved_bit_tcp_flags", ["tcp_flags_12bit"],
        "contract conventions.tcp_flags_12bit (owner decision; current Rust packet.rs:355 keeps 9 bits)",
        "tcp[12]=0x58 (offset 5, reserved bit 0x8) with CWR|ECE|SYN = 0x8C2 = 2242 (the training maximum), and "
        "tcp[12]=0x51 (NS bit) with ACK = 0x110 = 272",
        [frame(3.0, eth(GW, DEV, 0x0800, ipv4(DEV_IP, REMOTE, 6, tcp(51000, 80, 0xC2, 1024, byte12=0x58))),
               "sent: 51000 -> 80, byte 12 = 0x58, flags byte 0xC2, window 1024, total 40"),
         frame(3.1, eth(GW, DEV, 0x0800, ipv4(DEV_IP, REMOTE, 6, tcp(51001, 80, ACK, 1024, byte12=0x51), ident=2)),
               "sent: 51001 -> 80, byte 12 = 0x51, flags byte 0x10, window 1024, total 40")],
        {"values": {**BASE, **stats("header-length", 20, 20, 20, 0), **stats("ip-length", 40, 40, 40, 0),
                    **stats("ttl", 64, 64, 64, 0), **stats("tcp-flags", 1257, 2242, 272, 985),
                    **stats("window-size", 1024, 1024, 1024, 0), **counts("packets", 2, 2, 0),
                    **counts("ips", 2, 1, 1), **counts("macs", 2, 1, 1), "network_tcp-flags-syn_count": 1,
                    "network_tcp-flags-ack_count": 1, "network_ports_src__present_UNSEEN": 1,
                    "network_ports_dst__present_80": 1, **rates(0.2, 0.2, 0),
                    "network_tcp-flags-syn__fraction_of_packets": 0.5,
                    "network_tcp-flags-ack__fraction_of_packets": 0.5, "network_src_packet_fraction": 1,
                    "network_packets_per_reported_ip": 1}})

    # 8. LLDP: a non-IP frame counts in packets and MACs only.
    add("lldp_frame", ["all_frames_counted", "empty_family_zero"],
        "contract conventions.all_frames_counted (owner decision; current BPF filter '(ip or ip6 or arp)' drops it, "
        "native.rs:249-291)",
        "one LLDP frame from the device to 01:80:c2:00:00:0e: counted in packets and MACs; no IPs, so "
        "packets_per_reported_ip is NaN (denominator 0)",
        [frame(4.0, eth(LLDP_MC, DEV, 0x88CC, lldp()), "sent: LLDP chassis/port/TTL/end TLVs")],
        {"nan": ["network_packets_per_reported_ip"],
         "values": {**BASE, **counts("packets", 1, 1, 0), **counts("macs", 2, 1, 1), **rates(0.1, 0.1, 0),
                    "network_src_packet_fraction": 1}})

    # 9. Zero-packet window: only a frame between two other hosts.
    stat_names = [f"network_{f}_{s}" for f in ("header-length", "ip-flags", "ip-length", "mss", "tcp-flags", "ttl",
                                              "window-size") for s in ("avg", "max", "min", "std_deviation")]
    ratio_names = [f"network_tcp-flags-{f}__fraction_of_packets" for f in ("ack", "fin", "psh", "rst", "syn", "urg")] + [
        "network_src_packet_fraction", "network_dst_packet_fraction", "network_packets_per_reported_ip"]
    add("zero_packet_window", ["zero_packet_window_scored", "device_mac_admission"],
        "contract conventions.zero_packet_window_scored (dataset-verified + owner decision); current Rust nulls "
        "every feature at packets == 0 (features.rs:567-587)",
        "a unicast UDP frame between two other hosts is not admitted, so the window has no packets: counts and "
        "per_second 0, quality flag 1, statistics and ratios NaN, port presence and UNSEEN 0",
        [frame(5.0, eth(GW, OTHER, 0x0800, ipv4(OTHER_IP, REMOTE, 17, udp(40000, 53))),
               "not admitted: 02:..:07/192.168.1.7 -> 02:..:01 UDP 40000 -> 53")],
        {"nan": stat_names + ratio_names, "values": {**BASE, "quality_no_packets_observed": 1}})

    # 10. Broadcast ARP asking for the device's IP, sent by another host.
    add("broadcast_arp_for_local_ip", ["broadcast_not_admitted", "device_mac_admission", "arp_ips"],
        "contract conventions.broadcast_not_admitted (implementer default pending owner) and device_mac_admission "
        "(owner decision); current Rust admits by IP or MAC (packet.rs:155-170)",
        "another host's broadcast ARP request for 192.168.1.50 is not admitted (device MAC only), the device's "
        "unicast reply is",
        [frame(6.0, eth(BCAST, OTHER, 0x0806, arp(1, OTHER, OTHER_IP, "00:00:00:00:00:00", DEV_IP), pad=True),
               "not admitted: ARP who-has 192.168.1.50 from 02:..:07/192.168.1.7 to broadcast"),
         frame(6.001, eth(OTHER, DEV, 0x0806, arp(2, DEV, DEV_IP, OTHER, OTHER_IP)),
               "sent: ARP reply 192.168.1.50 is-at 02:..:50, to 02:..:07/192.168.1.7")],
        {"values": {**BASE, **counts("packets", 1, 1, 0), **counts("ips", 2, 1, 1), **counts("macs", 2, 1, 1),
                    **rates(0.1, 0.1, 0), "network_src_packet_fraction": 1, "network_packets_per_reported_ip": 0.5}})

    # 11. IPv6 left out entirely.
    dns = eth(GW, DEV, 0x0800, ipv4(DEV_IP, LAN_GW_IP, 17, udp(40000, 53, b"\x44" * 12)))
    add("ipv6_frame_excluded", ["ipv6_excluded"],
        "contract conventions.ipv6_excluded (owner decision; current Rust counts IPv6 in packets/macs/ips)",
        "an IPv6 TCP SYN to 33:33:00:00:00:01 is left out (no packet, MAC, port or statistic); one IPv4 UDP "
        "datagram 40000 -> 53 remains",
        [frame(7.0, eth(V6_MC, DEV, 0x86DD, ipv6_tcp_syn()), "sent, excluded: IPv6 fe80::50 -> ff02::1 TCP SYN 51000 -> 80"),
         frame(7.1, dns, "sent: UDP 192.168.1.50:40000 -> 192.168.1.1:53, 12-byte payload, total 40")],
        {"values": {**BASE, **stats("header-length", 20, 20, 20, 0), **stats("ip-length", 40, 40, 40, 0),
                    **stats("ttl", 64, 64, 64, 0), **counts("packets", 1, 1, 0), **counts("ips", 2, 1, 1),
                    **counts("macs", 2, 1, 1), "network_ports_src__present_UNSEEN": 1,
                    "network_ports_dst__present_53": 1, **rates(0.1, 0.1, 0), "network_src_packet_fraction": 1,
                    "network_packets_per_reported_ip": 0.5}})

    # 12. Capture drop: every input missing, window not scored.
    add("capture_drop_window", ["damaged_window.capture_drop"],
        "contract damaged_window.capture_drop = all_missing_not_scored (owner decision: keep the current rule, "
        "features.rs:567-587 / packet.rs:478)",
        "a window with capture drops: all 252 inputs are missing whatever the frames say",
        [frame(8.0, eth(GW, DEV, 0x0800, ipv4(DEV_IP, REMOTE, 6, tcp(51000, 443, ACK, 502))),
               "sent: 51000 -> 443 ACK, total 40 (ignored: the window had capture drops)")],
        {"default": None}, capture_drop=True)

    # 13. Truncated TCP header (snaplen cut).
    syn = eth(GW, DEV, 0x0800, ipv4(DEV_IP, REMOTE, 6, tcp(51000, 80, SYN, 64240)))
    add("truncated_tcp_header", ["damaged_window.family_damage"],
        "contract damaged_window.family_damage.tcp_header and .ports (implementer default pending owner)",
        "a sent TCP SYN captured to 44 of 54 bytes (TCP header cut at 10 bytes) and a complete UDP datagram "
        "40000 -> 53: TCP statistics, flag counts and MSS are missing; IPv4 statistics use both packets; kept ports "
        "stay 1, absent vocabulary ports and UNSEEN without an unseen kept port are missing",
        [frame(9.0, syn[:44], "sent: 51000 -> 80 SYN, total 40, captured 44 of 54 bytes", wire_length=len(syn)),
         frame(9.1, eth(GW, DEV, 0x0800, ipv4(DEV_IP, REMOTE, 17, udp(40000, 53), ident=2)),
               "sent: UDP 40000 -> 53, no payload, total 28")],
        {"nan_prefixes": TCP_PREFIXES + ["network_mss_", "network_ports_src__present_", "network_ports_dst__present_"],
         "values": {**BASE, **stats("header-length", 20, 20, 20, 0), **stats("ip-flags", 0, 0, 0, 0),
                    **stats("ip-length", 34, 40, 28, 6), **stats("ttl", 64, 64, 64, 0),
                    **counts("packets", 2, 2, 0), **counts("ips", 2, 1, 1), **counts("macs", 2, 1, 1),
                    "network_ports_src__present_UNSEEN": 1, "network_ports_dst__present_53": 1,
                    **rates(0.2, 0.2, 0), "network_src_packet_fraction": 1, "network_dst_packet_fraction": 0,
                    "network_packets_per_reported_ip": 1}})

    # 14. Malformed TCP options: only MSS is missing.
    add("malformed_tcp_options", ["damaged_window.family_damage", "mss_bearing_only"],
        "contract damaged_window.family_damage.tcp_options (implementer default pending owner); options parsed as "
        "packet.rs:358-382",
        "a sent SYN whose only option is kind 2 with length 6 (overruns the 4 option bytes): MSS statistics are "
        "missing, every other TCP statistic stays",
        [frame(0.3, eth(GW, DEV, 0x0800, ipv4(DEV_IP, REMOTE, 6, tcp(51000, 80, SYN, 64240, bytes([2, 6, 5, 0xB4])),
                                         frag=0x4000)),
               "sent: 51000 -> 80 SYN, option bytes 02 06 05 b4, DF, ttl 64, window 64240, total 44")],
        {"nan_prefixes": ["network_mss_"],
         "values": {**BASE, **stats("header-length", 20, 20, 20, 0), **stats("ip-flags", 2, 2, 2, 0),
                    **stats("ip-length", 44, 44, 44, 0), **stats("ttl", 64, 64, 64, 0),
                    **stats("tcp-flags", 2, 2, 2, 0), **stats("window-size", 64240, 64240, 64240, 0),
                    **counts("packets", 1, 1, 0), **counts("ips", 2, 1, 1), **counts("macs", 2, 1, 1),
                    "network_tcp-flags-syn_count": 1, "network_ports_src__present_UNSEEN": 1,
                    "network_ports_dst__present_80": 1, **rates(0.1, 0.1, 0),
                    "network_tcp-flags-syn__fraction_of_packets": 1, "network_src_packet_fraction": 1,
                    "network_packets_per_reported_ip": 0.5}})

    # 15. Invalid IPv4 header (IHL 4) carrying protocol 6, plus an ARP request.
    bad = bytearray(ipv4(DEV_IP, REMOTE, 6, tcp(51000, 80, SYN, 64240)))
    bad[0] = 0x44
    add("invalid_ipv4_header", ["damaged_window.family_damage", "damage_propagation"],
        "contract damaged_window.family_damage.ipv4_header / ipv4_or_arp_header; reference default "
        "damage_propagation (packet.rs:180-187, 205-215: the TCP segment inside cannot be located)",
        "an IPv4 packet with IHL 4 (16 bytes, invalid) and protocol 6, and a valid ARP request: IPv4 statistics, "
        "fragmented-packets, IP counts, TCP families, MSS and every port input are missing; packets and MACs stay",
        [frame(0.4, eth(GW, DEV, 0x0800, bytes(bad)), "sent: IPv4 version 4, IHL 4, protocol 6, total 40"),
         frame(0.5, eth(BCAST, DEV, 0x0806, arp(1, DEV, DEV_IP, "00:00:00:00:00:00", LAN_GW_IP)),
               "sent: ARP who-has 192.168.1.1")],
        {"nan_prefixes": [f"network_{f}_" for f in IPV4_FAMS] + TCP_PREFIXES + ["network_mss_", "network_ips_",
                                                                                "network_ports_"],
         "nan": ["network_fragmented-packets", "network_packets_per_reported_ip"],
         "values": {**BASE, **counts("packets", 2, 2, 0), **counts("macs", 3, 1, 2), **rates(0.2, 0.2, 0),
                    "network_src_packet_fraction": 1, "network_dst_packet_fraction": 0}})

    # 16. 802.1Q-tagged IPv4.
    add("vlan_tagged_ipv4", ["link_vlan"],
        "reference default link_vlan (packet.rs:118-135 follows up to two 802.1Q/802.1ad tags); not in the contract",
        "a sent UDP datagram inside an 802.1Q tag (VLAN 100) is measured like an untagged one",
        [frame(0.6, eth(GW, DEV, 0x0800, ipv4(DEV_IP, LAN_GW_IP, 17, udp(40000, 53, b"\x44" * 12)), vlan=100),
               "sent: 802.1Q VLAN 100, UDP 192.168.1.50:40000 -> 192.168.1.1:53, 12-byte payload, total 40")],
        {"values": {**BASE, **stats("header-length", 20, 20, 20, 0), **stats("ip-length", 40, 40, 40, 0),
                    **stats("ttl", 64, 64, 64, 0), **counts("packets", 1, 1, 0), **counts("ips", 2, 1, 1),
                    **counts("macs", 2, 1, 1), "network_ports_src__present_UNSEEN": 1,
                    "network_ports_dst__present_53": 1, **rates(0.1, 0.1, 0), "network_src_packet_fraction": 1,
                    "network_packets_per_reported_ip": 0.5}})

    # 17. MSS option outside a SYN counts (the rule keys on the option).
    add("mss_option_outside_syn", ["mss_bearing_only", "population_std"],
        "contract conventions.mss_bearing_only evidence: 1 validation row has syn == 0 with mss_max > 0",
        "a sent ACK carrying MSS 536 and a sent SYN carrying MSS 1460: both enter the MSS statistics",
        [frame(0.7, eth(GW, DEV, 0x0800, ipv4(DEV_IP, REMOTE, 6, tcp(51000, 80, ACK, 64240, MSS536), frag=0x4000)),
               "sent: 51000 -> 80 ACK with MSS 536, DF, total 44"),
         frame(0.8, eth(GW, DEV, 0x0800, ipv4(DEV_IP, REMOTE, 6, tcp(51000, 80, SYN, 64240, MSS1460), frag=0x4000,
                                         ident=2)),
               "sent: 51000 -> 80 SYN with MSS 1460, DF, total 44")],
        {"values": {**BASE, **stats("header-length", 20, 20, 20, 0), **stats("ip-flags", 2, 2, 2, 0),
                    **stats("ip-length", 44, 44, 44, 0), **stats("ttl", 64, 64, 64, 0),
                    **stats("tcp-flags", 9, 16, 2, 7), **stats("window-size", 64240, 64240, 64240, 0),
                    **stats("mss", 998, 1460, 536, 462), **counts("packets", 2, 2, 0), **counts("ips", 2, 1, 1),
                    **counts("macs", 2, 1, 1), "network_tcp-flags-ack_count": 1, "network_tcp-flags-syn_count": 1,
                    "network_ports_src__present_UNSEEN": 1, "network_ports_dst__present_80": 1,
                    **rates(0.2, 0.2, 0), "network_tcp-flags-ack__fraction_of_packets": 0.5,
                    "network_tcp-flags-syn__fraction_of_packets": 0.5, "network_src_packet_fraction": 1,
                    "network_packets_per_reported_ip": 1}})

    # 18. Runt frame (no Ethernet header) is not admitted.
    add("runt_frame_not_admitted", ["admission_order", "device_mac_admission"],
        "reference default admission_order: a frame shorter than 14 bytes has no readable MAC (packet.rs:91-93 "
        "drops it too)",
        "a 10-byte runt is not admitted; the device's one ARP request is",
        [frame(0.9, bytes.fromhex("0200000000500200000000"[:20]), "runt: 10 bytes, no complete Ethernet header"),
         frame(1.0, eth(BCAST, DEV, 0x0806, arp(1, DEV, DEV_IP, "00:00:00:00:00:00", LAN_GW_IP)),
               "sent: ARP who-has 192.168.1.1")],
        {"values": {**BASE, **counts("packets", 1, 1, 0), **counts("ips", 2, 1, 1), **counts("macs", 2, 1, 1),
                    **rates(0.1, 0.1, 0), "network_src_packet_fraction": 1, "network_packets_per_reported_ip": 0.5}})
    return fx


def build() -> dict:
    conv = CONTRACT["conventions"]
    status = {**{k: v["status"] for k, v in conv.items()},
              **{f"damaged_window.{k}": v for k, v in CONTRACT["damaged_window"]["status"].items() if k != "rule"},
              **dict.fromkeys(REFERENCE_DEFAULTS, "reference_default")}
    fx = fixtures()
    for f in fx:
        f["statuses"] = {c: status[c] for c in f["conventions"]}
        names = set(CONTRACT["feature_names"])
        assert set(f["expected"].get("values", {})) <= names and set(f["expected"].get("nan", [])) <= names, f["id"]
    return {
        "version": 1,
        "purpose": "Shared live77 frame fixtures (integration plan step 2). The Python reference "
                   "(framework/training/csv_v3/live77_reference.py, Layer B) and the Rust extractor (plan step 5) must "
                   "both reproduce every expected vector exactly: identical NaN positions, equal float32 values.",
        "built_by": "evaluation/fixtures/live77/build_frames_v1.py",
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "contract": "framework/training/csv_v3/live77_contract.json",
        "contract_feature_names_sha256": CONTRACT["feature_names_sha256"],
        "nominal_seconds": CONTRACT["window"]["nominal_duration_seconds"],
        "device": {"mac": DEV, "ipv4": DEV_IP, "note": "the IPv4 address is informational: admission is by MAC only"},
        "expected_format": "sparse: every input starts at 'default' (0; null = NaN), then names starting with a "
                           "'nan_prefixes' entry and names in 'nan' become NaN, then 'values' override (null = NaN); "
                           "compare as float32",
        "frame_format": "t = seconds from the window start (non-decreasing, arrival order); hex = captured bytes "
                        "from the Ethernet header; wire_length = original length when the capture was cut "
                        "(default: captured length)",
        "honesty": "expected vectors encode the contract, including conventions still pending owner sign-off "
                   "(see statuses); they are not evidence about the dataset publisher's exporter",
        "fixtures": fx,
    }


if __name__ == "__main__":
    OUT.write_bytes((json.dumps(build(), indent=1) + "\n").encode("utf-8"))  # LF on every platform
    print(OUT)
