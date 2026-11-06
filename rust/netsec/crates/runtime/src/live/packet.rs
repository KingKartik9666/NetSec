//! Pure header parsing and bounded counters. No packet payload is retained.
use super::features::{Live77Features, WindowFeatures, LIVE77_NOMINAL_SECONDS};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

pub const COUNT_NAMES: [&str; 7] = [
    "network_packets_all_count",
    "network_tcp-flags-ack_count",
    "network_tcp-flags-fin_count",
    "network_tcp-flags-psh_count",
    "network_tcp-flags-rst_count",
    "network_tcp-flags-syn_count",
    "network_tcp-flags-urg_count",
];

pub struct LocalAddresses {
    pub ips: BTreeSet<IpAddr>,
    pub mac: Option<[u8; 6]>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Header {
    pub protocol: &'static str,
    pub source: String,
    pub destination: String,
    pub length: u32,
    pub flags: u8,
    pub destination_port: Option<u16>,
    pub complete_tcp: bool,
    #[serde(skip)]
    pub facts: PacketFacts,
}

#[derive(Debug, Clone, Default)]
pub struct PacketFacts {
    pub timestamp_ns: Option<i64>,
    pub timestamp_resolution_ns: Option<u64>,
    pub source_mac: Option<[u8; 6]>,
    pub destination_mac: Option<[u8; 6]>,
    pub source_ip: Option<IpAddr>,
    pub destination_ip: Option<IpAddr>,
    pub source_local: bool,
    pub destination_local: bool,
    pub direction_known: bool,
    pub ip_version: Option<u8>,
    pub ip_length: Option<u32>,
    pub ip_header_length: Option<u16>,
    pub ip_flags: Option<u8>,
    pub ttl: Option<u8>,
    pub fragmented: Option<bool>,
    pub source_port: Option<u16>,
    pub destination_port: Option<u16>,
    pub tcp_flags: Option<u16>,
    pub tcp_window: Option<u16>,
    pub tcp_mss: Option<u16>,
    pub transport_payload_length: Option<u32>,
    pub invalid_ip: bool,
    pub incomplete_link: bool,
    pub incomplete_transport: bool,
    pub incomplete_tcp: bool,
    pub incomplete_ports: bool,
    pub malformed_tcp_options: bool,
}

fn read16(bytes: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_be_bytes([
        *bytes.get(offset)?,
        *bytes.get(offset + 1)?,
    ]))
}
fn mac(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect::<Vec<_>>()
        .join(":")
}

pub fn parse(bytes: &[u8], length: u32, own: &LocalAddresses) -> Option<Header> {
    parse_at(bytes, length, None, own)
}

/// Parse only available header bytes; the original wire length is a separate fact.
pub fn parse_at(
    bytes: &[u8],
    length: u32,
    timestamp_ns: Option<i64>,
    own: &LocalAddresses,
) -> Option<Header> {
    if bytes.len() < 14 || bytes.len() > length as usize {
        return None;
    }
    let source_mac = <[u8; 6]>::try_from(&bytes[6..12]).ok()?;
    let destination_mac = <[u8; 6]>::try_from(&bytes[..6]).ok()?;
    let source_local = own.mac == Some(source_mac);
    let destination_local = own.mac == Some(destination_mac);
    let mut header = Header {
        protocol: "Ethernet",
        source: mac(&source_mac),
        destination: mac(&destination_mac),
        length,
        flags: 0,
        destination_port: None,
        complete_tcp: true,
        facts: PacketFacts {
            timestamp_ns,
            source_mac: Some(source_mac),
            destination_mac: Some(destination_mac),
            source_local,
            destination_local,
            direction_known: source_local || destination_local,
            ..PacketFacts::default()
        },
    };
    let mut offset = 14;
    let mut kind = read16(bytes, 12)?;
    for _ in 0..2 {
        if ![0x8100, 0x88a8].contains(&kind) {
            break;
        }
        if let Some(next) = read16(bytes, offset + 2) {
            kind = next;
            offset += 4;
        } else {
            header.facts.incomplete_link = true;
            break;
        }
    }
    if [0x8100, 0x88a8].contains(&kind) {
        header.facts.incomplete_link = true;
    }
    if header.facts.incomplete_link {
        incomplete_transport(&mut header, None);
    } else {
        match kind {
            0x0800 => parse_ipv4(bytes, offset, &mut header),
            0x86dd => parse_ipv6(bytes, offset, &mut header),
            0x0806 => {
                header.protocol = "ARP";
                if let Some(arp) = bytes.get(offset..offset + 28) {
                    if read16(arp, 2) == Some(0x0800) && arp[4] == 6 && arp[5] == 4 {
                        header.facts.source_ip = Some(IpAddr::V4(Ipv4Addr::new(
                            arp[14], arp[15], arp[16], arp[17],
                        )));
                        header.facts.destination_ip = Some(IpAddr::V4(Ipv4Addr::new(
                            arp[24], arp[25], arp[26], arp[27],
                        )));
                    }
                }
            }
            _ => {}
        }
    }
    let source_ip_local = header
        .facts
        .source_ip
        .is_some_and(|ip| own.ips.contains(&ip));
    let destination_ip_local = header
        .facts
        .destination_ip
        .is_some_and(|ip| own.ips.contains(&ip));
    if source_ip_local || destination_ip_local {
        header.facts.source_local = source_ip_local;
        header.facts.destination_local = destination_ip_local;
        header.facts.direction_known = true;
    }
    if !header.facts.direction_known {
        return None;
    }
    if let Some(ip) = header.facts.source_ip {
        header.source = ip.to_string();
    }
    if let Some(ip) = header.facts.destination_ip {
        header.destination = ip.to_string();
    }
    Some(header)
}

fn incomplete_transport(header: &mut Header, protocol: Option<u8>) {
    header.facts.incomplete_transport = true;
    if protocol.is_none_or(|p| p == 6) {
        header.facts.incomplete_tcp = true;
        header.complete_tcp = false;
    }
    header.facts.incomplete_ports = protocol.is_none_or(|p| matches!(p, 6 | 17));
}

fn parse_ipv4(bytes: &[u8], offset: usize, header: &mut Header) {
    header.facts.ip_version = Some(4);
    header.protocol = "IPv4";
    let Some(ip) = bytes
        .get(offset..)
        .filter(|ip| ip.len() >= 20 && ip[0] >> 4 == 4)
    else {
        header.facts.invalid_ip = true;
        incomplete_transport(header, None);
        return;
    };
    header.facts.source_ip = Some(IpAddr::V4(Ipv4Addr::new(ip[12], ip[13], ip[14], ip[15])));
    header.facts.destination_ip = Some(IpAddr::V4(Ipv4Addr::new(ip[16], ip[17], ip[18], ip[19])));
    let protocol = ip[9];
    header.protocol = match protocol {
        6 => "TCP",
        17 => "UDP",
        1 => "ICMP",
        _ => "IPv4",
    };
    let ihl = usize::from(ip[0] & 15) * 4;
    let total = usize::from(read16(ip, 2).expect("checked IPv4 base header"));
    if ihl < 20
        || ihl > ip.len()
        || total < ihl
        || total > (header.length as usize).saturating_sub(offset)
    {
        header.facts.invalid_ip = true;
        incomplete_transport(header, Some(protocol));
        return;
    }
    let fragment = read16(ip, 6).expect("checked IPv4 base header");
    header.facts.ip_header_length = Some(ihl as u16);
    header.facts.ip_length = Some(total as u32);
    header.facts.ip_flags = Some((fragment >> 13) as u8);
    header.facts.ttl = Some(ip[8]);
    header.facts.fragmented = Some(fragment & 0x3fff != 0);
    if fragment & 0x1fff != 0 {
        // Noninitial fragments never start with a transport header.
        if matches!(protocol, 6 | 17 | 1) {
            incomplete_transport(header, Some(protocol));
        }
        return;
    }
    parse_transport(bytes, offset + ihl, total - ihl, protocol, header);
    if fragment & 0x2000 != 0 {
        // Initial-fragment header fields can be valid; payload length cannot.
        header.facts.transport_payload_length = None;
        header.facts.incomplete_transport = true;
    }
}

fn parse_ipv6(bytes: &[u8], offset: usize, header: &mut Header) {
    header.facts.ip_version = Some(6);
    header.protocol = "IPv6";
    let Some(ip) = bytes
        .get(offset..)
        .filter(|ip| ip.len() >= 40 && ip[0] >> 4 == 6)
    else {
        header.facts.invalid_ip = true;
        incomplete_transport(header, None);
        return;
    };
    header.facts.source_ip = Some(IpAddr::V6(Ipv6Addr::from(
        <[u8; 16]>::try_from(&ip[8..24]).expect("checked IPv6 base header"),
    )));
    header.facts.destination_ip = Some(IpAddr::V6(Ipv6Addr::from(
        <[u8; 16]>::try_from(&ip[24..40]).expect("checked IPv6 base header"),
    )));
    let payload = usize::from(read16(ip, 4).expect("checked IPv6 base header"));
    let declared = payload + 40;
    let mut next = ip[6];
    if declared > (header.length as usize).saturating_sub(offset) || (payload == 0 && next != 59) {
        // Empty No Next Header is valid; unimplemented jumbograms remain unknown.
        header.facts.invalid_ip = true;
        incomplete_transport(header, None);
        return;
    }
    header.facts.ip_length = Some(declared as u32);
    header.facts.ttl = Some(ip[7]);
    header.facts.fragmented = Some(false);
    let end = offset + declared;
    let mut cursor = offset + 40;
    for _ in 0..8 {
        if !matches!(next, 0 | 43 | 44 | 51 | 60) {
            break;
        }
        let Some(prefix) = bytes.get(cursor..cursor + 2).filter(|_| cursor + 2 <= end) else {
            header.facts.invalid_ip = true;
            incomplete_transport(header, None);
            return;
        };
        let size = match next {
            44 => 8,
            51 => (usize::from(prefix[1]) + 2) * 4,
            _ => (usize::from(prefix[1]) + 1) * 8,
        };
        if next == 51 && size < 12 {
            header.facts.invalid_ip = true;
            incomplete_transport(header, None);
            return;
        }
        let Some(extension) = bytes
            .get(cursor..cursor + size)
            .filter(|_| cursor + size <= end)
        else {
            header.facts.invalid_ip = true;
            incomplete_transport(header, None);
            return;
        };
        let extension_type = next;
        next = extension[0];
        cursor += size;
        if extension_type == 44 {
            let fragment = u16::from_be_bytes([extension[2], extension[3]]);
            header.facts.fragmented = Some(fragment & 0xfff9 != 0);
            if fragment & 0xfff8 != 0 {
                incomplete_transport(
                    header,
                    (!matches!(next, 0 | 43 | 44 | 51 | 60)).then_some(next),
                );
                return;
            }
        }
    }
    if matches!(next, 0 | 43 | 44 | 51 | 60) {
        header.facts.invalid_ip = true;
        incomplete_transport(header, None);
        return;
    }
    header.facts.ip_header_length = Some((cursor - offset) as u16);
    header.protocol = match next {
        6 => "TCP",
        17 => "UDP",
        58 => "ICMPv6",
        _ => "IPv6",
    };
    parse_transport(bytes, cursor, end - cursor, next, header);
    if header.facts.fragmented == Some(true) {
        header.facts.transport_payload_length = None;
        header.facts.incomplete_transport = true;
    }
}

fn parse_transport(
    bytes: &[u8],
    start: usize,
    available: usize,
    protocol: u8,
    header: &mut Header,
) {
    match protocol {
        6 => {
            let Some(tcp) = bytes.get(start..start + 20).filter(|_| available >= 20) else {
                incomplete_transport(header, Some(protocol));
                return;
            };
            let size = usize::from(tcp[12] >> 4) * 4;
            if size < 20 || size > available || bytes.get(start..start + size).is_none() {
                incomplete_transport(header, Some(protocol));
                return;
            }
            header.flags = tcp[13];
            header.destination_port = read16(tcp, 2);
            header.facts.source_port = read16(tcp, 0);
            header.facts.destination_port = header.destination_port;
            header.facts.tcp_flags = Some(u16::from(tcp[13]) | (u16::from(tcp[12] & 1) << 8));
            header.facts.tcp_window = read16(tcp, 14);
            header.facts.transport_payload_length = Some((available - size) as u32);
            let mut option = start + 20;
            while option < start + size {
                match bytes[option] {
                    0 => break,
                    1 => option += 1,
                    kind => {
                        let option_size = bytes.get(option + 1).copied().unwrap_or(0) as usize;
                        if option_size < 2 || option + option_size > start + size {
                            header.facts.malformed_tcp_options = true;
                            break;
                        }
                        if kind == 2 {
                            if option_size != 4 || header.facts.tcp_mss.is_some() {
                                header.facts.malformed_tcp_options = true;
                                break;
                            }
                            header.facts.tcp_mss = read16(bytes, option + 2);
                        }
                        option += option_size;
                    }
                }
            }
            if header.facts.malformed_tcp_options {
                header.facts.tcp_mss = None;
            }
        }
        17 => {
            let Some(udp) = bytes.get(start..start + 8).filter(|_| available >= 8) else {
                incomplete_transport(header, Some(protocol));
                return;
            };
            let length = usize::from(read16(udp, 4).expect("checked UDP header"));
            // Initial-fragment UDP length can exceed this fragment.
            if length < 8 || (length > available && header.facts.fragmented != Some(true)) {
                incomplete_transport(header, Some(protocol));
                return;
            }
            header.facts.source_port = read16(udp, 0);
            header.facts.destination_port = read16(udp, 2);
            if length <= available {
                header.facts.transport_payload_length = Some((length - 8) as u32);
            }
        }
        1 | 58 => {
            if available >= 8 && bytes.get(start..start + 8).is_some() {
                header.facts.transport_payload_length = Some((available - 8) as u32);
            } else {
                incomplete_transport(header, Some(protocol));
            }
        }
        _ => {}
    }
}

// ------------------------------------------------------------------ live77 model-row frame facts (plan step 5)
// A separate projection of one Ethernet frame for the live77 model row, following
// framework/training/csv_v3/live77_contract.json and its Python reference
// (live77_reference.py Layer B). It never changes `parse_at` or the
// npcap-network-v1 observation view. Admission is by the device MAC only; IPv6 is
// left out; ARP sender/target IPv4 addresses count; MF-only fragments; 12-bit
// TCP flags; non-initial fragments keep their IPv4 statistics and void nothing.
pub const DAMAGE_IPV4_HEADER: u8 = 1;
pub const DAMAGE_IPV4_OR_ARP_HEADER: u8 = 2;
pub const DAMAGE_TCP_HEADER: u8 = 4;
pub const DAMAGE_TCP_OPTIONS: u8 = 8;
pub const DAMAGE_PORTS: u8 = 16;
const IPV4_MORE_FRAGMENTS: u16 = 0x2000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Live77Ipv4 {
    pub header_length: u16,
    pub total_length: u16,
    pub flags: u16,
    pub ttl: u8,
    pub more_fragments: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Live77Tcp {
    /// ((tcp[12] & 0x0f) << 8) | tcp[13], including the reserved bits.
    pub flags12: u16,
    pub window: u16,
}

/// Header facts of one admitted frame. Holds no payload bytes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Live77Frame {
    /// The Ethernet source is the device MAC (packets_src); otherwise packets_dst.
    pub sent: bool,
    pub source_mac: [u8; 6],
    pub destination_mac: [u8; 6],
    /// IPv4 header addresses, or ARP sender/target protocol addresses.
    pub source_ip: Option<[u8; 4]>,
    pub destination_ip: Option<[u8; 4]>,
    pub ipv4: Option<Live77Ipv4>,
    pub tcp: Option<Live77Tcp>,
    pub mss: Option<u16>,
    pub ports: Option<(u16, u16)>,
    /// DAMAGE_* bits: families this frame makes unmeasurable for the window.
    pub damage: u8,
}

fn live77_transport_damage(protocol: Option<u8>) -> u8 {
    let mut damage = 0;
    if protocol.is_none_or(|p| p == 6) {
        damage |= DAMAGE_TCP_HEADER | DAMAGE_TCP_OPTIONS;
    }
    if protocol.is_none_or(|p| matches!(p, 6 | 17)) {
        damage |= DAMAGE_PORTS;
    }
    damage
}

/// `None` when the frame is not admitted: shorter than an Ethernet header,
/// captured beyond its wire length, not to or from the device MAC, or IPv6.
pub fn live77_frame(bytes: &[u8], wire_length: u32, device: [u8; 6]) -> Option<Live77Frame> {
    if bytes.len() < 14 || bytes.len() > wire_length as usize {
        return None;
    }
    let destination_mac = <[u8; 6]>::try_from(&bytes[..6]).ok()?;
    let source_mac = <[u8; 6]>::try_from(&bytes[6..12]).ok()?;
    if source_mac != device && destination_mac != device {
        return None;
    }
    let (mut offset, mut kind, mut link_ok) = (14, read16(bytes, 12)?, true);
    for _ in 0..2 {
        if ![0x8100, 0x88a8].contains(&kind) {
            break;
        }
        if let Some(next) = read16(bytes, offset + 2) {
            kind = next;
            offset += 4;
        } else {
            link_ok = false;
            break;
        }
    }
    if [0x8100, 0x88a8].contains(&kind) {
        link_ok = false;
    }
    if link_ok && kind == 0x86dd {
        return None;
    }
    let mut frame = Live77Frame {
        sent: source_mac == device,
        source_mac,
        destination_mac,
        ..Live77Frame::default()
    };
    if !link_ok {
        frame.damage |=
            DAMAGE_IPV4_HEADER | DAMAGE_IPV4_OR_ARP_HEADER | live77_transport_damage(None);
    } else if kind == 0x0800 {
        live77_ipv4(bytes, offset, wire_length as usize, &mut frame);
    } else if kind == 0x0806 {
        match bytes.get(offset..offset + 28) {
            None => frame.damage |= DAMAGE_IPV4_OR_ARP_HEADER,
            Some(arp) if read16(arp, 2) == Some(0x0800) && arp[4] == 6 && arp[5] == 4 => {
                frame.source_ip = <[u8; 4]>::try_from(&arp[14..18]).ok();
                frame.destination_ip = <[u8; 4]>::try_from(&arp[24..28]).ok();
            }
            Some(_) => {}
        }
    }
    Some(frame)
}

fn live77_ipv4(bytes: &[u8], offset: usize, wire_length: usize, frame: &mut Live77Frame) {
    let header_damage = DAMAGE_IPV4_HEADER | DAMAGE_IPV4_OR_ARP_HEADER;
    let Some(ip) = bytes
        .get(offset..)
        .filter(|ip| ip.len() >= 20 && ip[0] >> 4 == 4)
    else {
        frame.damage |= header_damage | live77_transport_damage(None);
        return;
    };
    let protocol = ip[9];
    let ihl = usize::from(ip[0] & 15) * 4;
    let total = usize::from(read16(ip, 2).expect("checked IPv4 base header"));
    if ihl < 20 || ihl > ip.len() || total < ihl || total > wire_length.saturating_sub(offset) {
        frame.damage |= header_damage | live77_transport_damage(Some(protocol));
        return;
    }
    let fragment = read16(ip, 6).expect("checked IPv4 base header");
    frame.source_ip = <[u8; 4]>::try_from(&ip[12..16]).ok();
    frame.destination_ip = <[u8; 4]>::try_from(&ip[16..20]).ok();
    frame.ipv4 = Some(Live77Ipv4 {
        header_length: ihl as u16,
        total_length: total as u16,
        flags: fragment >> 13,
        ttl: ip[8],
        more_fragments: fragment & IPV4_MORE_FRAGMENTS != 0,
    });
    if fragment & 0x1fff != 0 {
        // Non-initial fragment: IPv4 statistics only; nothing is voided.
        return;
    }
    let (start, available) = (offset + ihl, total - ihl);
    match protocol {
        6 => {
            let Some(tcp) = bytes.get(start..start + 20).filter(|_| available >= 20) else {
                frame.damage |= live77_transport_damage(Some(6));
                return;
            };
            let size = usize::from(tcp[12] >> 4) * 4;
            if size < 20 || size > available || bytes.len() < start + size {
                frame.damage |= live77_transport_damage(Some(6));
                return;
            }
            frame.tcp = Some(Live77Tcp {
                flags12: (u16::from(tcp[12] & 0x0f) << 8) | u16::from(tcp[13]),
                window: read16(tcp, 14).expect("checked TCP header"),
            });
            frame.ports = read16(tcp, 0).zip(read16(tcp, 2));
            let (mut mss, mut malformed, mut option) = (None, false, start + 20);
            while option < start + size {
                match bytes[option] {
                    0 => break,
                    1 => option += 1,
                    kind => {
                        let length = bytes.get(option + 1).copied().map_or(0, usize::from);
                        if length < 2 || option + length > start + size {
                            malformed = true;
                            break;
                        }
                        if kind == 2 {
                            if length != 4 || mss.is_some() {
                                malformed = true;
                                break;
                            }
                            mss = read16(bytes, option + 2);
                        }
                        option += length;
                    }
                }
            }
            if malformed {
                frame.damage |= DAMAGE_TCP_OPTIONS;
            } else {
                frame.mss = mss;
            }
        }
        17 => {
            let Some(udp) = bytes.get(start..start + 8).filter(|_| available >= 8) else {
                frame.damage |= live77_transport_damage(Some(17));
                return;
            };
            let length = usize::from(read16(udp, 4).expect("checked UDP header"));
            if length < 8 || (length > available && fragment & IPV4_MORE_FRAGMENTS == 0) {
                frame.damage |= live77_transport_damage(Some(17));
                return;
            }
            frame.ports = read16(udp, 0).zip(read16(udp, 2));
        }
        _ => {}
    }
}

#[derive(Debug, Default, Serialize)]
pub struct Window {
    pub packets: u64,
    pub bytes: u64,
    pub syn: u64,
    pub syn_ack: u64,
    pub syn_all: u64,
    pub ack: u64,
    pub fin: u64,
    pub rst: u64,
    pub psh: u64,
    pub urg: u64,
    pub incomplete_headers: u64,
    pub capture_drops: u64,
    #[serde(skip)]
    pub ports: BTreeMap<String, BTreeSet<u16>>,
    pub recent_packets: VecDeque<Header>,
    #[serde(skip)]
    pub features: WindowFeatures,
    /// Separate live77 model-row projection; it never feeds `features` or the counts above.
    #[serde(skip)]
    pub live77: Live77Features,
}
impl Window {
    /// Offer one captured frame (header prefix + original wire length) to the live77 projection.
    pub fn add_live77_frame(&mut self, bytes: &[u8], wire_length: u32) {
        self.live77.add_frame(bytes, wire_length);
    }
    /// The live77 model row in the order of `names` (the bundle's feature names); all missing after capture loss.
    pub fn live77_row(&self, names: &[String]) -> Result<Vec<Option<f32>>, String> {
        self.live77.row(names, self.capture_drops > 0)
    }
    pub fn add(&mut self, header: Header) {
        self.features.add(&header);
        self.packets += 1;
        self.bytes += u64::from(header.length);
        if !header.complete_tcp {
            self.incomplete_headers += 1;
        }
        let flags = header.flags;
        let syn_only = flags & 2 != 0 && flags & 16 == 0;
        self.syn += u64::from(syn_only);
        self.syn_ack += u64::from(flags & 2 != 0 && flags & 16 != 0);
        self.syn_all += u64::from(flags & 2 != 0);
        self.ack += u64::from(flags & 16 != 0);
        self.fin += u64::from(flags & 1 != 0);
        self.rst += u64::from(flags & 4 != 0);
        self.psh += u64::from(flags & 8 != 0);
        self.urg += u64::from(flags & 32 != 0);
        if syn_only {
            if let Some(port) = header.destination_port {
                let key = format!("{}>{}", header.source, header.destination);
                if self.ports.len() < 1000 || self.ports.contains_key(&key) {
                    let ports = self.ports.entry(key).or_default();
                    if ports.len() < 100 {
                        ports.insert(port);
                    }
                }
            }
        }
        self.recent_packets.push_back(header);
        if self.recent_packets.len() > 40 {
            self.recent_packets.pop_front();
        }
    }
    pub fn feature_observation(&self, seconds: f64) -> serde_json::Value {
        self.features.observation(seconds, self.capture_drops)
    }
    /// The live77 window gate (plan step 6; owner decision 3 Nov 2026). It replaces the seven-count gate
    /// for packets-mode scoring: the measured tumbling window must be 10 ± 1 s, and capture loss is not scored.
    /// Zero-packet windows pass (scored with quality_no_packets_observed = 1, as in training), and so do
    /// unreadable headers, whose live77 inputs are missing instead of blocking the whole window.
    pub fn live77_gate(&self, seconds: f64) -> Result<(), String> {
        if !seconds.is_finite() || (seconds - LIVE77_NOMINAL_SECONDS).abs() > 1.0 {
            return Err(format!("Model skipped delayed window ({seconds:.2} s): live77 scoring requires ten seconds ± one second"));
        }
        if self.capture_drops > 0 {
            return Err(format!("Capture loss ({} dropped frames): this window's live77 inputs are missing, so it is not scored", self.capture_drops));
        }
        Ok(())
    }
    /// The retired seven-count input view, kept for the observation tests and the replay example;
    /// it no longer gates any model (packets mode uses `live77_gate`).
    pub fn counts(&self, seconds: f64) -> Result<[f32; 7], String> {
        if !seconds.is_finite() || (seconds - 10.0).abs() > 1.0 {
            return Err(
                "Model skipped delayed window: inference requires ten seconds ± one second".into(),
            );
        }
        if self.packets == 0 {
            return Err("No captured packets: there is no traffic window to classify".into());
        }
        if self.incomplete_headers > 0 || self.capture_drops > 0 {
            return Err("Model skipped incomplete packet window: header parsing or capture loss prevents a complete seven-count vector".into());
        }
        let counts = [
            self.packets,
            self.ack,
            self.fin,
            self.psh,
            self.rst,
            self.syn_all,
            self.urg,
        ];
        if counts.iter().any(|v| *v > 1 << 24) || counts[1..].iter().any(|v| *v > self.packets) {
            return Err("Packet counts exceed the verified float32/count constraints".into());
        }
        Ok(counts.map(|v| v as f32))
    }
    pub fn assessment(&self, seconds: f64) -> serde_json::Value {
        let mut evidence = Vec::new();
        let duration = seconds.max(0.1);
        if self.packets >= 100
            && self.syn as f64 / self.packets as f64 >= 0.8
            && self.syn as f64 / duration >= 50.0
        {
            evidence.push(serde_json::json!({"rule":"syn_burst","observed":self.syn as f64/duration,"threshold":50,"message":"Frequent SYN-only connection attempts; verify the endpoints and intent"}));
        }
        let ports = self.ports.values().map(BTreeSet::len).max().unwrap_or(0);
        if ports >= 25 {
            evidence.push(serde_json::json!({"rule":"port_sweep","observed":ports,"threshold":25,"message":"One source attempted many distinct destination ports; verify intent"}));
        }
        serde_json::json!({"status":if !evidence.is_empty(){"suspicious"}else if self.packets>0{"benign_looking"}else{"idle"},
            "method":"Live behavior rules","evidence":evidence,"limitations":"Experimental header rules; neither incident confirmation nor proof of safety. No payload inspection or automatic mitigation."})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn own() -> LocalAddresses {
        LocalAddresses {
            ips: ["192.0.2.10".parse().unwrap()].into(),
            mac: None,
        }
    }
    fn tcp(flags: u8) -> Vec<u8> {
        let mut bytes = vec![0; 54];
        bytes[12..14].copy_from_slice(&0x0800u16.to_be_bytes());
        bytes[14] = 0x45;
        bytes[16..18].copy_from_slice(&40u16.to_be_bytes());
        bytes[23] = 6;
        bytes[26..30].copy_from_slice(&[192, 0, 2, 10]);
        bytes[30..34].copy_from_slice(&[198, 51, 100, 20]);
        bytes[36..38].copy_from_slice(&443u16.to_be_bytes());
        bytes[47] = flags;
        bytes[46] = 0x50;
        bytes
    }
    #[test]
    fn independent_syn_ack_bits_and_ownership() {
        let mut window = Window::default();
        window.add(parse(&tcp(0x12), 54, &own()).unwrap());
        assert_eq!(window.counts(10.0).unwrap(), [1., 1., 0., 0., 0., 1., 0.]);
        assert_eq!(window.syn, 0);
        assert_eq!(window.syn_ack, 1);
        let elsewhere = LocalAddresses {
            ips: ["203.0.113.3".parse().unwrap()].into(),
            mac: None,
        };
        assert!(parse(&tcp(2), 54, &elsewhere).is_none());
    }
    #[test]
    fn truncated_and_fragmented_tcp_never_become_complete_model_inputs() {
        let mut frame = tcp(2);
        frame.truncate(42);
        let mut window = Window::default();
        window.add(parse(&frame, 54, &own()).unwrap());
        assert!(window.counts(10.0).is_err());
        let mut frame = tcp(2);
        frame[20..22].copy_from_slice(&1u16.to_be_bytes());
        assert!(!parse(&frame, 54, &own()).unwrap().complete_tcp);
    }
    #[test]
    fn exact_duration_no_traffic_and_loss_gates() {
        let mut window = Window::default();
        assert!(window.counts(10.0).is_err());
        window.add(parse(&tcp(2), 54, &own()).unwrap());
        assert!(window.counts(11.01).is_err());
        assert!(window.counts(9.0).is_ok());
        window.capture_drops = 1;
        assert!(window.counts(10.0).is_err());
    }
    #[test]
    fn live77_gate_keeps_ten_seconds_scores_empty_and_damaged_windows_and_skips_loss() {
        // Owner decision 3 Nov 2026: 10 s tumbling windows, measured 10 ± 1 s.
        let mut window = Window::default();
        for seconds in [9.0, 9.5, 10.0, 10.75, 11.0] {
            assert_eq!(window.live77_gate(seconds), Ok(()), "{seconds} s");
        }
        for seconds in [0.0, 1.0, 8.99, 11.01, 20.0, -10.0, f64::NAN, f64::INFINITY] {
            let error = window.live77_gate(seconds).unwrap_err();
            assert!(error.contains("ten seconds ± one second"), "{seconds} s: {error}");
        }
        // A zero-packet window passes (scored with quality_no_packets_observed = 1) and so do
        // unreadable headers (those live77 inputs are missing); the seven-count gate refuses both.
        assert!(window.counts(10.0).is_err());
        let mut frame = tcp(2);
        frame.truncate(42);
        window.add(parse(&frame, 54, &own()).unwrap());
        assert!(window.incomplete_headers > 0 && window.counts(10.0).is_err());
        assert_eq!(window.live77_gate(10.0), Ok(()));
        // Capture loss is never scored.
        window.capture_drops = 1;
        assert!(window.live77_gate(10.0).unwrap_err().contains("Capture loss"));
        assert!(window.live77_gate(12.0).is_err());
    }
    #[test]
    fn history_and_scan_state_are_bounded() {
        let mut window = Window::default();
        for port in 0..200 {
            let mut header = parse(&tcp(2), 54, &own()).unwrap();
            header.destination_port = Some(port);
            window.add(header);
        }
        assert_eq!(window.recent_packets.len(), 40);
        assert_eq!(window.ports.values().next().unwrap().len(), 100);
        assert_eq!(window.assessment(10.0)["status"], "suspicious");
    }

    fn measured_tcp(payload: usize, flags: u8) -> Vec<u8> {
        let mut bytes = tcp(flags);
        bytes.resize(54 + payload, 0);
        bytes[16..18].copy_from_slice(&((40 + payload) as u16).to_be_bytes());
        bytes[20..22].copy_from_slice(&0x4000u16.to_be_bytes());
        bytes[22] = 64;
        bytes[34..36].copy_from_slice(&12345u16.to_be_bytes());
        bytes[46] = 0x50;
        bytes[48..50].copy_from_slice(&1024u16.to_be_bytes());
        bytes
    }

    #[test]
    fn broad_statistics_use_packet_timestamps_and_population_deviation() {
        let first = measured_tcp(10, 0x12);
        let mut second = measured_tcp(42, 0x10);
        second[22] = 32;
        let mut window = Window::default();
        window.add(parse_at(&first, first.len() as u32, Some(2_000_000_000), &own()).unwrap());
        window.add(parse_at(&second, second.len() as u32, Some(1_000_000_000), &own()).unwrap());
        let row = window.feature_observation(10.0);
        assert_eq!(row["features"]["network_packet-size_avg"], 80.0);
        assert_eq!(row["features"]["network_packet-size_std_deviation"], 16.0);
        assert_eq!(row["features"]["network_header-length_avg"], 20.0);
        assert_eq!(row["features"]["network_time-delta_avg"], -1.0);
        assert_eq!(row["sample_counts"]["network_time-delta"], 1);
        assert_eq!(row["features"]["network_src_packet_fraction"], 1.0);
        assert!(row["features"]["network_mss_avg"].is_null());
        assert_eq!(window.counts(10.0).unwrap(), [2., 2., 0., 0., 0., 1., 0.]);
    }

    #[test]
    fn tcp_options_and_missing_transport_measurements_are_not_zero_filled() {
        let mut frame = measured_tcp(4, 2);
        frame[46] = 0x60;
        frame[54..58].copy_from_slice(&[2, 4, 0x05, 0xb4]);
        let mut window = Window::default();
        window.add(parse_at(&frame, frame.len() as u32, Some(0), &own()).unwrap());
        let row = window.feature_observation(10.0);
        assert_eq!(row["features"]["network_mss_avg"], 1460.0);
        assert_eq!(row["features"]["network_payload-length_avg"], 0.0);
        assert_eq!(row["sample_counts"]["network_mss"], 1);
        let mut fragment = measured_tcp(0, 2);
        fragment[20..22].copy_from_slice(&1u16.to_be_bytes());
        window.add(parse_at(&fragment, fragment.len() as u32, Some(1), &own()).unwrap());
        let incomplete = window.feature_observation(10.0);
        assert!(incomplete["features"]["network_tcp-flags-syn_count"].is_null());
        assert!(incomplete["features"]["network_ports_all_count"].is_null());
        assert_eq!(incomplete["features"]["network_fragmented-packets"], 1.0);
        assert!(window.counts(10.0).is_err());
    }

    #[test]
    fn non_tcp_and_capture_loss_have_explicit_availability() {
        let mut udp = measured_tcp(0, 0);
        udp.truncate(42);
        udp[16..18].copy_from_slice(&28u16.to_be_bytes());
        udp[23] = 17;
        udp[38..40].copy_from_slice(&8u16.to_be_bytes());
        let mut window = Window::default();
        window.add(parse_at(&udp, 42, Some(1000), &own()).unwrap());
        let row = window.feature_observation(10.0);
        assert!(row["features"]["network_tcp-flags_avg"].is_null());
        assert!(row["features"]["network_time-delta_avg"].is_null());
        assert_eq!(row["features"]["network_tcp-flags-syn_count"], 0.0);
        assert_eq!(row["features"]["network_ports_all_count"], 2.0);
        assert!(row["features"]["network_macs_dst_count"].is_null());
        assert!(row["features"]["network_protocols_all_count"].is_null());
        window.capture_drops = 1;
        let loss = window.feature_observation(10.0);
        assert!(loss["features"]
            .as_object()
            .unwrap()
            .values()
            .all(|v| v.is_null()));
    }

    fn row_for(bytes: &[u8], length: u32, local: &LocalAddresses) -> serde_json::Value {
        let mut window = Window::default();
        window.add(parse_at(bytes, length, Some(0), local).unwrap());
        window.feature_observation(10.0)
    }

    fn mac_owned() -> LocalAddresses {
        LocalAddresses {
            ips: BTreeSet::new(),
            mac: Some([0; 6]),
        }
    }

    #[test]
    fn payload_prefix_and_original_wire_length_have_separate_meanings() {
        let frame = measured_tcp(1_000, 2);
        let row = row_for(&frame[..54], 1_054, &own());
        assert_eq!(row["features"]["network_packet-size_avg"], 1_054.0);
        assert_eq!(row["features"]["network_ip-length_avg"], 1_040.0);
        assert_eq!(row["features"]["network_payload-length_avg"], 1_000.0);
        assert_eq!(row["features"].as_object().unwrap().len(), 79);
        assert!(parse_at(&frame[..54], 42, Some(0), &own()).is_none());

        // Captured Ethernet padding is not a TCP header beyond the IP boundary.
        let mut frame = measured_tcp(0, 2);
        frame[16..18].copy_from_slice(&20u16.to_be_bytes());
        let row = row_for(&frame, 54, &own());
        assert!(row["features"]["network_tcp-flags-syn_count"].is_null());
        assert!(row["features"]["network_ports_all_count"].is_null());
    }

    #[test]
    fn truncated_ip_and_vlan_headers_preserve_unknown_measurements() {
        let frame = measured_tcp(0, 2);
        let row = row_for(&frame[..24], 54, &mac_owned());
        assert_eq!(row["quality"]["invalid_ipv4_headers"], 1);
        assert!(row["features"]["network_ips_all_count"].is_null());
        assert!(row["features"]["network_fragmented-packets"].is_null());
        assert!(row["features"]["network_tcp-flags-ack_count"].is_null());
        let mut tagged = frame.clone();
        tagged[12..14].copy_from_slice(&0x8100u16.to_be_bytes());
        let row = row_for(&tagged[..14], 58, &mac_owned());
        assert_eq!(row["quality"]["incomplete_link_headers"], 1);
        assert!(row["features"]["network_ips_all_count"].is_null());
        assert!(row["features"]["network_tcp-flags-ack_count"].is_null());
        assert_eq!(row["features"]["network_packet-size_avg"], 58.0);
    }

    #[test]
    fn double_vlan_tags_are_checked_and_do_not_change_ip_lengths() {
        let frame = measured_tcp(0, 2);
        for tags in 1..=3 {
            let mut tagged = frame[..12].to_vec();
            for _ in 0..tags {
                tagged.extend_from_slice(&[0x81, 0, 0, 1]);
            }
            tagged.extend_from_slice(&frame[12..]);
            let row = row_for(&tagged, tagged.len() as u32, &mac_owned());
            if tags <= 2 {
                assert_eq!(row["features"]["network_ip-length_avg"], 40.0);
                assert_eq!(row["features"]["network_tcp-flags-syn_count"], 1.0);
            } else {
                assert!(row["features"]["network_tcp-flags-syn_count"].is_null());
                assert_eq!(row["quality"]["incomplete_link_headers"], 1);
            }
        }
    }

    #[test]
    fn incoming_endpoints_are_not_relabelled_as_outgoing_sets() {
        let outgoing = measured_tcp(0, 2);
        let mut incoming = measured_tcp(0, 0x10);
        incoming[26..30].copy_from_slice(&[198, 51, 100, 20]);
        incoming[30..34].copy_from_slice(&[192, 0, 2, 10]);
        incoming[34..36].copy_from_slice(&443u16.to_be_bytes());
        incoming[36..38].copy_from_slice(&12345u16.to_be_bytes());
        let mut window = Window::default();
        window.add(parse_at(&outgoing, 54, Some(0), &own()).unwrap());
        window.add(parse_at(&incoming, 54, Some(1_000_000_000), &own()).unwrap());
        let row = window.feature_observation(10.0);
        assert_eq!(row["features"]["network_packets_src_count"], 1.0);
        assert_eq!(row["features"]["network_packets_dst_count"], 1.0);
        assert_eq!(row["features"]["network_ips_src_count"], 2.0);
        assert_eq!(row["features"]["network_ips_dst_count"], 2.0);
        assert_eq!(row["features"]["network_ports_src_count"], 2.0);
        assert_eq!(row["features"]["network_ports_dst_count"], 2.0);
        assert_eq!(row["features"]["network_dst_packet_fraction"], 0.5);
    }

    #[test]
    fn malformed_mss_and_tcp_offsets_cannot_create_measurements() {
        for options in [[2, 1, 0, 0], [2, 3, 0, 0], [2, 5, 0, 0], [3, 0, 0, 0]] {
            let mut frame = measured_tcp(4, 2);
            frame[46] = 0x60;
            frame[54..58].copy_from_slice(&options);
            let row = row_for(&frame, 58, &own());
            assert!(row["features"]["network_mss_avg"].is_null());
            assert_eq!(row["quality"]["malformed_tcp_options"], 1);
            assert_eq!(row["features"]["network_window-size_avg"], 1_024.0);
        }
        let mut duplicate = measured_tcp(8, 2);
        duplicate[46] = 0x70;
        duplicate[54..62].copy_from_slice(&[2, 4, 5, 180, 2, 4, 5, 180]);
        let row = row_for(&duplicate, 62, &own());
        assert!(row["features"]["network_mss_avg"].is_null());
        assert_eq!(row["quality"]["malformed_tcp_options"], 1);
        for offset in [0x40, 0x60, 0xf0] {
            let mut frame = measured_tcp(0, 2);
            frame[46] = offset;
            let header = parse_at(&frame, 54, Some(0), &own()).unwrap();
            assert!(!header.complete_tcp);
            let row = row_for(&frame, 54, &own());
            assert!(row["features"]["network_tcp-flags-syn_count"].is_null());
        }
    }

    fn ipv6_udp(fragment: Option<u16>) -> Vec<u8> {
        let mut frame = vec![0; if fragment.is_some() { 70 } else { 62 }];
        frame[12..14].copy_from_slice(&0x86ddu16.to_be_bytes());
        frame[14] = 0x60;
        frame[18..20]
            .copy_from_slice(&(if fragment.is_some() { 16u16 } else { 8u16 }).to_be_bytes());
        frame[20] = if fragment.is_some() { 44 } else { 17 };
        frame[21] = 37;
        frame[22..38].copy_from_slice(&"2001:db8::1".parse::<Ipv6Addr>().unwrap().octets());
        frame[38..54].copy_from_slice(&"2001:db8::2".parse::<Ipv6Addr>().unwrap().octets());
        let start = if let Some(fragment) = fragment {
            frame[54] = 17;
            frame[56..58].copy_from_slice(&fragment.to_be_bytes());
            62
        } else {
            54
        };
        frame[start..start + 2].copy_from_slice(&123u16.to_be_bytes());
        frame[start + 2..start + 4].copy_from_slice(&456u16.to_be_bytes());
        frame[start + 4..start + 6].copy_from_slice(&8u16.to_be_bytes());
        frame
    }

    #[test]
    fn ipv6_measurements_and_fragments_do_not_fill_ipv4_dataset_fields() {
        for fragment in [None, Some(0), Some(1), Some(8)] {
            let frame = ipv6_udp(fragment);
            let row = row_for(&frame, frame.len() as u32, &mac_owned());
            assert!(row["features"]["network_ttl_avg"].is_null());
            assert!(row["features"]["network_ip-length_avg"].is_null());
            assert_eq!(row["features"]["network_ips_all_count"], 0.0);
            assert_eq!(row["features"]["network_fragmented-packets"], 0.0);
            assert_eq!(row["live_only_features"]["live_ipv6_hop-limit_avg"], 37.0);
            assert_eq!(row["live_only_features"]["live_ipv6_ips_all_count"], 2.0);
            assert_eq!(
                row["live_only_features"]["live_ipv6_fragmented_packets"],
                if fragment.is_some_and(|v| v != 0) {
                    1.0
                } else {
                    0.0
                }
            );
            if fragment == Some(8) {
                assert!(row["features"]["network_ports_all_count"].is_null());
            } else {
                assert_eq!(row["features"]["network_ports_all_count"], 2.0);
            }
            if fragment.is_some_and(|v| v != 0) {
                assert!(row["features"]["network_payload-length_avg"].is_null());
            }
        }
    }

    #[test]
    fn ipv6_extension_lengths_cannot_read_padding_past_declared_packet() {
        let mut frame = ipv6_udp(None);
        frame[20] = 0; // Hop-by-hop extension claims 16 bytes in an 8-byte payload.
        frame[54..56].copy_from_slice(&[6, 1]);
        frame.resize(100, 0);
        let row = row_for(&frame, 100, &mac_owned());
        assert_eq!(row["quality"]["invalid_ipv6_headers"], 1);
        assert!(row["features"]["network_tcp-flags-syn_count"].is_null());
        assert!(row["live_only_features"]["live_ipv6_fragmented_packets"].is_null());
    }

    #[test]
    fn ipv6_authentication_headers_reject_impossibly_short_lengths() {
        let mut frame = ipv6_udp(None);
        frame[20] = 51;
        frame[54..56].copy_from_slice(&[59, 0]); // AH needs at least twelve bytes.
        let row = row_for(&frame, frame.len() as u32, &mac_owned());
        assert_eq!(row["quality"]["invalid_ipv6_headers"], 1);
        assert!(row["features"]["network_tcp-flags-syn_count"].is_null());
    }

    #[test]
    fn extension_chain_limit_and_192_byte_snaplen_are_explicit() {
        for extensions in [8, 9] {
            let mut frame = ipv6_udp(None)[..54].to_vec();
            frame[20] = 0;
            frame[18..20].copy_from_slice(&((extensions * 8 + 8) as u16).to_be_bytes());
            for index in 0..extensions {
                frame.extend_from_slice(&[
                    if index + 1 == extensions { 17 } else { 60 },
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]);
            }
            frame.extend_from_slice(&[0, 53, 0, 80, 0, 8, 0, 0]);
            let row = row_for(&frame, frame.len() as u32, &mac_owned());
            if extensions == 8 {
                assert_eq!(row["features"]["network_ports_all_count"], 2.);
                assert_eq!(
                    row["live_only_features"]["live_ipv6_header-length_avg"],
                    104.
                );
            } else {
                assert_eq!(row["quality"]["invalid_ipv6_headers"], 1);
                assert!(row["features"]["network_ports_all_count"].is_null());
            }
        }
        let frame = measured_tcp(1000, 2);
        let row = row_for(&frame[..192], 1054, &own());
        assert_eq!(row["features"]["network_payload-length_avg"], 1000.);
        assert_eq!(row["features"]["network_packet-size_avg"], 1054.);
        let mut frame = ipv6_udp(None);
        frame[18..20].copy_from_slice(&168u16.to_be_bytes());
        frame[20] = 0;
        frame[54..56].copy_from_slice(&[17, 19]); // 160-byte extension exceeds snaplen.
        frame.resize(192, 0);
        let row = row_for(&frame, 222, &mac_owned());
        assert_eq!(row["quality"]["invalid_ipv6_headers"], 1);
        assert!(row["features"]["network_ports_all_count"].is_null());
        assert_eq!(row["features"]["network_packet-size_avg"], 222.);
    }

    #[test]
    fn initial_fragments_keep_complete_header_evidence_without_payload_lengths() {
        let mut frame = measured_tcp(8, 0x12);
        frame[20..22].copy_from_slice(&0x2000u16.to_be_bytes());
        let row = row_for(&frame, frame.len() as u32, &own());
        assert_eq!(row["features"]["network_tcp-flags-syn_count"], 1.);
        assert_eq!(row["features"]["network_ports_all_count"], 2.);
        assert_eq!(row["features"]["network_fragmented-packets"], 1.);
        assert!(row["features"]["network_payload-length_avg"].is_null());
        frame.truncate(42);
        frame[23] = 17;
        frame[16..18].copy_from_slice(&28u16.to_be_bytes());
        frame[38..40].copy_from_slice(&1008u16.to_be_bytes());
        let row = row_for(&frame, 42, &own());
        assert_eq!(row["features"]["network_ports_all_count"], 2.);
        assert!(row["features"]["network_payload-length_avg"].is_null());
    }

    #[test]
    fn arp_addresses_do_not_fill_ipv4_header_address_sets() {
        let mut frame = vec![0; 42];
        frame[12..14].copy_from_slice(&0x0806u16.to_be_bytes());
        frame[16..18].copy_from_slice(&0x0800u16.to_be_bytes());
        frame[18] = 6;
        frame[19] = 4;
        frame[28..32].copy_from_slice(&[192, 0, 2, 10]);
        frame[38..42].copy_from_slice(&[192, 0, 2, 11]);
        let row = row_for(&frame, 42, &own());
        assert_eq!(row["features"]["network_packets_src_count"], 1.0);
        assert_eq!(row["features"]["network_ips_all_count"], 0.0);
        assert!(row["features"]["network_packets_per_reported_ip"].is_null());
        assert!(row["features"]["network_header-length_avg"].is_null());
        assert!(row["features"]["network_mss_avg"].is_null());
    }

    #[test]
    fn live77_projection_is_separate_from_the_observation_view() {
        let device = [2, 0, 0, 0, 0, 0x10];
        let mut frame = measured_tcp(0, 2);
        frame[6..12].copy_from_slice(&device);
        let mut plain = Window::default();
        plain.add(parse_at(&frame, 54, Some(0), &own()).unwrap());
        let mut both = Window {
            live77: Live77Features::new(Some(device)),
            ..Window::default()
        };
        both.add(parse_at(&frame, 54, Some(0), &own()).unwrap());
        both.add_live77_frame(&frame, 54);
        // An LLDP frame from the device counts in the model row only.
        let mut lldp = vec![0; 60];
        lldp[..6].copy_from_slice(&[0x01, 0x80, 0xc2, 0, 0, 0x0e]);
        lldp[6..12].copy_from_slice(&device);
        lldp[12..14].copy_from_slice(&0x88ccu16.to_be_bytes());
        both.add_live77_frame(&lldp, 60);
        assert_eq!(
            plain.feature_observation(10.0),
            both.feature_observation(10.0)
        );
        assert_eq!((both.packets, both.counts(10.0)), (1, plain.counts(10.0)));
        let names = [
            "network_packets_all_count",
            "network_tcp-flags-syn_count",
            "network_macs_dst__observed_distinct_count",
            "quality_no_packets_observed",
        ]
        .map(String::from);
        assert_eq!(
            both.live77_row(&names).unwrap(),
            [Some(2.), Some(1.), Some(2.), Some(0.)]
        );
        // A frame that matches neither device MAC is not admitted (IP ownership does not admit it).
        let mut foreign = frame.clone();
        foreign[6..12].copy_from_slice(&[2, 0, 0, 0, 0, 0x99]);
        both.add_live77_frame(&foreign, 54);
        assert_eq!(both.live77_row(&names).unwrap()[0], Some(2.));
        both.capture_drops = 1;
        assert!(both.live77_row(&names).unwrap().iter().all(Option::is_none));
        assert!(Window::default().live77_row(&names).is_err());
    }

    #[test]
    fn missing_and_extreme_timestamps_do_not_overflow_or_bridge_gaps() {
        let frame = measured_tcp(0, 0);
        let mut window = Window::default();
        window.add(parse_at(&frame, 54, Some(i64::MAX), &own()).unwrap());
        window.add(parse_at(&frame, 54, Some(i64::MIN), &own()).unwrap());
        let row = window.feature_observation(10.0);
        assert!(row["features"]["network_time-delta_avg"].as_f64().unwrap() < 0.0);
        assert_eq!(row["quality"]["negative_time_deltas"], 1);
        window.add(parse_at(&frame, 54, None, &own()).unwrap());
        window.add(parse_at(&frame, 54, Some(0), &own()).unwrap());
        let row = window.feature_observation(10.0);
        assert!(row["features"]["network_time-delta_avg"].is_null());
        assert_eq!(row["sample_counts"]["network_time-delta"], 1);
        assert_eq!(row["quality"]["missing_timestamps"], 1);
        assert!(row["timestamp_precision"]["source_resolution_ns"].is_null());
    }
}
