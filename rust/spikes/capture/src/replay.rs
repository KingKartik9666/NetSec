//! Pure bounded Ethernet/IP/ARP/TCP header parser for existing synthetic goldens.
//! This is offline replay, never an Npcap or model substitute.
use crate::Error;
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
    net::{Ipv4Addr, Ipv6Addr},
    path::Path,
};

const MAX_FILE: usize = 1024 * 1024;

fn read_small(path: impl AsRef<Path>) -> Result<Vec<u8>, Error> {
    let mut result = Vec::new();
    std::fs::File::open(path)?
        .take((MAX_FILE + 1) as u64)
        .read_to_end(&mut result)?;
    if result.len() > MAX_FILE {
        return Err(Error::Invalid("Replay input exceeds 1 MiB".into()));
    }
    Ok(result)
}

fn u32le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}
fn be16(bytes: &[u8]) -> u16 {
    u16::from_be_bytes([bytes[0], bytes[1]])
}
pub fn frames(data: &[u8]) -> Result<Vec<&[u8]>, Error> {
    if data.len() < 24 || data[..4] != [0xd4, 0xc3, 0xb2, 0xa1] || u32le(&data[20..24]) != 1 {
        return Err(Error::Invalid(
            "Replay supports little-endian microsecond Ethernet PCAP only".into(),
        ));
    }
    if data[4..8] != [2, 0, 4, 0] {
        return Err(Error::Invalid("Unsupported PCAP version".into()));
    }
    let mut cursor = 24;
    let mut result = Vec::new();
    while cursor < data.len() {
        if data.len() - cursor < 16 {
            return Err(Error::Invalid("Truncated PCAP record header".into()));
        }
        let size = u32le(&data[cursor + 8..cursor + 12]) as usize;
        if size > 65535 || size != u32le(&data[cursor + 12..cursor + 16]) as usize {
            return Err(Error::Invalid(
                "Truncated or oversized captured frame".into(),
            ));
        }
        cursor += 16;
        if size > data.len() - cursor {
            return Err(Error::Invalid("Truncated PCAP frame".into()));
        }
        result.push(&data[cursor..cursor + size]);
        cursor += size;
        if result.len() > 10000 {
            return Err(Error::Invalid("Replay frame count exceeds 10000".into()));
        }
    }
    Ok(result)
}

fn mac(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect::<Vec<_>>()
        .join(":")
}

struct Header {
    source: String,
    destination: String,
    own: bool,
    tcp_flags: u8,
    tcp_port: u16,
}
fn header(frame: &[u8], ips: &BTreeSet<String>, macs: &BTreeSet<String>) -> Result<Header, Error> {
    if frame.len() < 14 {
        return Err(Error::Invalid("Truncated Ethernet header".into()));
    }
    let source_mac = mac(&frame[6..12]);
    let destination_mac = mac(&frame[..6]);
    let mut value = Header {
        source: source_mac.clone(),
        destination: destination_mac.clone(),
        own: macs.contains(&source_mac) || macs.contains(&destination_mac),
        tcp_flags: 0,
        tcp_port: 0,
    };
    let mut protocol = be16(&frame[12..14]);
    let mut offset = 14;
    for _ in 0..2 {
        if protocol != 0x8100 && protocol != 0x88a8 {
            break;
        }
        if frame.len() < offset + 4 {
            return Err(Error::Invalid("Truncated VLAN header".into()));
        }
        protocol = be16(&frame[offset + 2..offset + 4]);
        offset += 4;
    }
    let mut tcp = None;
    match protocol {
        0x0800 => {
            if frame.len() < offset + 20 || frame[offset] >> 4 != 4 {
                return Err(Error::Invalid("Truncated/invalid IPv4 header".into()));
            }
            let length = (frame[offset] & 15) as usize * 4;
            if length < 20 || frame.len() < offset + length {
                return Err(Error::Invalid("Invalid IPv4 header length".into()));
            }
            value.source = Ipv4Addr::from([
                frame[offset + 12],
                frame[offset + 13],
                frame[offset + 14],
                frame[offset + 15],
            ])
            .to_string();
            value.destination = Ipv4Addr::from([
                frame[offset + 16],
                frame[offset + 17],
                frame[offset + 18],
                frame[offset + 19],
            ])
            .to_string();
            // Noninitial fragments lack the transport header; do not guess flags.
            if frame[offset + 9] == 6 && be16(&frame[offset + 6..offset + 8]) & 0x1fff == 0 {
                tcp = Some(offset + length);
            }
        }
        0x86dd => {
            if frame.len() < offset + 40 || frame[offset] >> 4 != 6 {
                return Err(Error::Invalid("Truncated/invalid IPv6 header".into()));
            }
            let mut source = [0_u8; 16];
            source.copy_from_slice(&frame[offset + 8..offset + 24]);
            let mut destination = [0_u8; 16];
            destination.copy_from_slice(&frame[offset + 24..offset + 40]);
            value.source = Ipv6Addr::from(source).to_string();
            value.destination = Ipv6Addr::from(destination).to_string();
            match frame[offset + 6] {
                6 => tcp = Some(offset + 40),
                0 | 43 | 44 | 50 | 51 | 60 => {
                    return Err(Error::Invalid(
                        "IPv6 extension headers not implemented in this bounded spike".into(),
                    ));
                }
                _ => (),
            }
        }
        0x0806 => {
            if frame.len() < offset + 28 || frame[offset + 4..offset + 6] != [6, 4] {
                return Err(Error::Invalid("Unsupported/truncated ARP header".into()));
            }
            value.source = Ipv4Addr::from([
                frame[offset + 14],
                frame[offset + 15],
                frame[offset + 16],
                frame[offset + 17],
            ])
            .to_string();
            value.destination = Ipv4Addr::from([
                frame[offset + 24],
                frame[offset + 25],
                frame[offset + 26],
                frame[offset + 27],
            ])
            .to_string();
        }
        _ => (),
    }
    value.own |= ips.contains(&value.source) || ips.contains(&value.destination);
    if let Some(offset) = tcp {
        if frame.len() < offset + 20 {
            return Err(Error::Invalid("Truncated TCP header".into()));
        }
        value.tcp_flags = frame[offset + 13];
        value.tcp_port = be16(&frame[offset + 2..offset + 4]);
    }
    Ok(value)
}

#[derive(Default, Serialize)]
struct Counts {
    packets: u64,
    bytes: u64,
    syn: u64,
    syn_ack: u64,
    syn_all: u64,
    ack: u64,
    fin: u64,
    rst: u64,
    psh: u64,
    urg: u64,
    seconds: f64,
    unique_ports_per_source: BTreeMap<String, usize>,
}
fn count(
    frames: &[&[u8]],
    ips: &BTreeSet<String>,
    macs: &BTreeSet<String>,
    seconds: f64,
) -> Result<Counts, Error> {
    let mut counts = Counts {
        seconds,
        ..Counts::default()
    };
    let mut ports: BTreeMap<String, BTreeSet<u16>> = BTreeMap::new();
    for frame in frames {
        let value = header(frame, ips, macs)?;
        if !value.own {
            continue;
        }
        counts.packets += 1;
        counts.bytes += frame.len() as u64;
        let flags = value.tcp_flags;
        let syn = flags & 2 != 0;
        let ack = flags & 16 != 0;
        counts.syn += u64::from(syn && !ack);
        counts.syn_ack += u64::from(syn && ack);
        counts.syn_all += u64::from(syn);
        counts.ack += u64::from(ack);
        counts.fin += u64::from(flags & 1 != 0);
        counts.rst += u64::from(flags & 4 != 0);
        counts.psh += u64::from(flags & 8 != 0);
        counts.urg += u64::from(flags & 32 != 0);
        if syn && !ack {
            let key = format!("{}>{}", value.source, value.destination);
            if ports.len() < 1000 || ports.contains_key(&key) {
                let set = ports.entry(key).or_default();
                if set.len() < 100 {
                    set.insert(value.tcp_port);
                }
            }
        }
    }
    counts.unique_ports_per_source = ports.into_iter().map(|(k, v)| (k, v.len())).collect();
    Ok(counts)
}
fn status(value: &Counts) -> &'static str {
    let burst = value.packets >= 100
        && value.syn as f64 / value.packets as f64 >= 0.8
        && value.syn as f64 / value.seconds.max(0.1) >= 50.0;
    if burst || value.unique_ports_per_source.values().any(|v| *v >= 25) {
        "suspicious"
    } else if value.packets > 0 {
        "benign_looking"
    } else {
        "idle"
    }
}
fn strings(value: &Value) -> Result<BTreeSet<String>, Error> {
    value
        .as_array()
        .ok_or_else(|| Error::Invalid("Expected address array".into()))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(String::from)
                .ok_or_else(|| Error::Invalid("Expected string address".into()))
        })
        .collect()
}
fn index(value: &Value) -> Result<usize, Error> {
    value
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| Error::Invalid("Expected bounded integer".into()))
}

pub fn golden(pcap: &str, expected: &str) -> Result<Value, Error> {
    let start = std::time::Instant::now();
    let data = read_small(pcap)?;
    let frames = frames(&data)?;
    let expected: Value = serde_json::from_slice(&read_small(expected)?)?;
    if expected["synthetic"] != true {
        return Err(Error::Invalid(
            "Replay harness only accepts labelled synthetic goldens".into(),
        ));
    }
    let mut cases = Vec::new();
    let mut all = true;
    for case in expected["cases"]
        .as_array()
        .ok_or_else(|| Error::Invalid("Missing cases".into()))?
    {
        let input = &case["input"];
        let offset = index(&input["packet_start"])?;
        let length = index(&input["packet_count"])?;
        let end = offset
            .checked_add(length)
            .ok_or_else(|| Error::Invalid("Packet range overflow".into()))?;
        let selected = frames
            .get(offset..end)
            .ok_or_else(|| Error::Invalid("Packet range outside PCAP".into()))?;
        let seconds = input["window_seconds"]
            .as_f64()
            .filter(|v| v.is_finite() && *v >= 0.0)
            .ok_or_else(|| Error::Invalid("Invalid duration".into()))?;
        let actual = count(
            selected,
            &strings(&input["local_addresses"])?,
            &strings(&input["local_macs"])?,
            seconds,
        )?;
        let stats = serde_json::to_value(&actual)?;
        let stats_equal = stats == case["output"]["stats"];
        let status_equal = status(&actual) == case["output"]["assessment"]["status"];
        all &= stats_equal && status_equal;
        cases.push(json!({"id":case["id"],"stats":stats,"stats_equal":stats_equal,"assessment_status":status(&actual),"assessment_status_equal":status_equal}));
    }
    Ok(
        json!({"schema_version":1,"synthetic":true,"all_equal":all,"case_count":cases.len(),"pcap_packet_count":frames.len(),"cases":cases,"elapsed_ms":start.elapsed().as_secs_f64()*1000.0,"limitations":"Existing invented header-only fixtures; not live capture, complete parser coverage, rule prose parity or detection accuracy. IPv6 extension headers and non-Ethernet link types are not implemented."}),
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    #[test]
    fn malformed_and_unsupported_inputs_fail() {
        assert!(frames(&[]).is_err());
        assert!(header(&[0; 13], &BTreeSet::new(), &BTreeSet::new()).is_err());
    }
    #[test]
    fn synthetic_fixture_parity() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../evaluation/fixtures/rust_migration/v1/live");
        let value = golden(
            root.join("synthetic.pcap").to_str().unwrap(),
            root.join("windows.json").to_str().unwrap(),
        )
        .unwrap();
        assert_eq!(value["all_equal"], true);
        assert_eq!(value["case_count"], 8);
        assert_eq!(value["pcap_packet_count"], 359);
    }
    #[test]
    fn ownership_and_independent_tcp_bits() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../evaluation/fixtures/rust_migration/v1/live/synthetic.pcap");
        let data = read_small(root).unwrap();
        let packets = frames(&data).unwrap();
        let own = ["192.0.2.10".into()].into();
        let empty = BTreeSet::new();
        let counts = count(&packets[1..2], &own, &empty, 10.0).unwrap();
        assert_eq!(counts.syn_all, 1);
        assert_eq!(counts.ack, 1);
        assert_eq!(counts.syn, 0);
        assert_eq!(
            count(&packets[..1], &empty, &empty, 10.0).unwrap().packets,
            0
        );
    }
    #[test]
    fn record_bounds_fail() {
        let mut bytes = vec![0xd4, 0xc3, 0xb2, 0xa1, 2, 0, 4, 0];
        bytes.resize(24, 0);
        bytes[20] = 1;
        bytes.push(0);
        assert!(frames(&bytes).is_err());
    }
}
