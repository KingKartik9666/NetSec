//! Pure header parsing and bounded counters. No packet payload is retained.
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
    if bytes.len() < 14 || length < 14 {
        return None;
    }
    let own_mac = own
        .mac
        .is_some_and(|m| bytes[..6] == m || bytes[6..12] == m);
    let mut offset = 14;
    let mut kind = read16(bytes, 12)?;
    for _ in 0..2 {
        if ![0x8100, 0x88a8].contains(&kind) {
            break;
        }
        kind = read16(bytes, offset + 2)?;
        offset += 4;
    }
    let mut header = Header {
        protocol: "Ethernet",
        source: mac(&bytes[6..12]),
        destination: mac(&bytes[..6]),
        length,
        flags: 0,
        destination_port: None,
        complete_tcp: true,
    };
    if [0x8100, 0x88a8].contains(&kind) {
        header.complete_tcp = false;
    }
    let mut own_ip = false;
    let mut tcp = None;
    if kind == 0x0800 {
        let ip = bytes.get(offset..)?;
        if ip.len() < 20 || ip[0] >> 4 != 4 {
            return own_mac.then_some(Header {
                complete_tcp: false,
                ..header
            });
        }
        let source = IpAddr::V4(Ipv4Addr::new(ip[12], ip[13], ip[14], ip[15]));
        let destination = IpAddr::V4(Ipv4Addr::new(ip[16], ip[17], ip[18], ip[19]));
        own_ip = own.ips.contains(&source) || own.ips.contains(&destination);
        header.source = source.to_string();
        header.destination = destination.to_string();
        header.protocol = match ip[9] {
            6 => "TCP",
            17 => "UDP",
            1 => "ICMP",
            _ => "IPv4",
        };
        if ip[9] == 6 {
            let ihl = usize::from(ip[0] & 15) * 4;
            // Noninitial fragments do not contain a TCP header. Flags cannot be
            // reconstructed, so an affected window is excluded from ML inference.
            if ihl < 20 || read16(ip, 6)? & 0x1fff != 0 {
                header.complete_tcp = false;
            } else {
                tcp = Some(offset + ihl);
            }
        }
    } else if kind == 0x86dd {
        let ip = bytes.get(offset..)?;
        if ip.len() < 40 || ip[0] >> 4 != 6 {
            return own_mac.then_some(Header {
                complete_tcp: false,
                ..header
            });
        }
        let source = IpAddr::V6(Ipv6Addr::from(<[u8; 16]>::try_from(&ip[8..24]).ok()?));
        let destination = IpAddr::V6(Ipv6Addr::from(<[u8; 16]>::try_from(&ip[24..40]).ok()?));
        own_ip = own.ips.contains(&source) || own.ips.contains(&destination);
        header.source = source.to_string();
        header.destination = destination.to_string();
        header.protocol = "IPv6";
        let mut next = ip[6];
        let mut cursor = offset + 40;
        for _ in 0..8 {
            match next {
                0 | 43 | 60 => {
                    let Some(extension) = bytes.get(cursor..cursor + 2) else {
                        header.complete_tcp = false;
                        break;
                    };
                    next = extension[0];
                    let size = (usize::from(extension[1]) + 1) * 8;
                    if cursor + size > bytes.len() {
                        header.complete_tcp = false;
                        break;
                    }
                    cursor += size;
                }
                44 => {
                    let Some(extension) = bytes.get(cursor..cursor + 8) else {
                        header.complete_tcp = false;
                        break;
                    };
                    next = extension[0];
                    if u16::from_be_bytes([extension[2], extension[3]]) & 0xfff8 != 0 {
                        header.complete_tcp = false;
                        break;
                    }
                    cursor += 8;
                }
                51 => {
                    let Some(extension) = bytes.get(cursor..cursor + 2) else {
                        header.complete_tcp = false;
                        break;
                    };
                    next = extension[0];
                    let size = (usize::from(extension[1]) + 2) * 4;
                    if cursor + size > bytes.len() {
                        header.complete_tcp = false;
                        break;
                    }
                    cursor += size;
                }
                _ => break,
            }
        }
        match next {
            6 => {
                header.protocol = "TCP";
                if header.complete_tcp {
                    tcp = Some(cursor);
                }
            }
            17 => header.protocol = "UDP",
            58 => header.protocol = "ICMPv6",
            0 | 43 | 44 | 51 | 60 => header.complete_tcp = false,
            _ => {}
        }
    } else if kind == 0x0806 {
        header.protocol = "ARP";
        let arp = bytes.get(offset..)?;
        if arp.len() >= 28 && read16(arp, 2) == Some(0x0800) && arp[4] == 6 && arp[5] == 4 {
            let source = IpAddr::V4(Ipv4Addr::new(arp[14], arp[15], arp[16], arp[17]));
            let destination = IpAddr::V4(Ipv4Addr::new(arp[24], arp[25], arp[26], arp[27]));
            own_ip = own.ips.contains(&source) || own.ips.contains(&destination);
            header.source = source.to_string();
            header.destination = destination.to_string();
        }
    }
    if !own_mac && !own_ip {
        return None;
    }
    if let Some(tcp) = tcp {
        if bytes.get(tcp..tcp + 20).is_some() {
            header.flags = bytes[tcp + 13];
            header.destination_port = read16(bytes, tcp + 2);
        } else {
            header.complete_tcp = false;
        }
    }
    Some(header)
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
}
impl Window {
    pub fn add(&mut self, header: Header) {
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
        bytes[23] = 6;
        bytes[26..30].copy_from_slice(&[192, 0, 2, 10]);
        bytes[30..34].copy_from_slice(&[198, 51, 100, 20]);
        bytes[36..38].copy_from_slice(&443u16.to_be_bytes());
        bytes[47] = flags;
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
}
