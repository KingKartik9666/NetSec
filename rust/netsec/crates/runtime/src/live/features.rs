//! Bounded header measurements; candidate CSV names are not publisher parity.
//! Summary deviations use population variance (M2 / n), never sample variance.
use super::packet::{
    live77_frame, Header, Live77Frame, DAMAGE_IPV4_HEADER, DAMAGE_IPV4_OR_ARP_HEADER, DAMAGE_PORTS,
    DAMAGE_TCP_HEADER, DAMAGE_TCP_OPTIONS,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::net::IpAddr;
use std::sync::OnceLock;

const SET_LIMIT: usize = 65_536;

struct PortFeature {
    name: String,
    side: usize,
    port: u16,
}

fn port_vocabulary() -> &'static [PortFeature] {
    static VOCABULARY: OnceLock<Vec<PortFeature>> = OnceLock::new();
    VOCABULARY.get_or_init(|| {
        let names: Vec<String> = serde_json::from_str(include_str!("port_features.json"))
            .expect("Embedded numeric port vocabulary is valid JSON");
        names
            .into_iter()
            .map(|name| {
                let (prefix, token) = name
                    .split_once("__present_")
                    .expect("Embedded port feature has its membership separator");
                let side = match prefix {
                    "network_ports_all" => 0,
                    "network_ports_src" => 1,
                    "network_ports_dst" => 2,
                    _ => panic!("Embedded port feature has an invalid endpoint direction"),
                };
                let port = token.parse::<u16>().expect("Embedded port token fits u16");
                PortFeature { name, side, port }
            })
            .collect()
    })
}

fn port_feature_indices() -> &'static BTreeMap<(usize, u16), usize> {
    static INDICES: OnceLock<BTreeMap<(usize, u16), usize>> = OnceLock::new();
    INDICES.get_or_init(|| {
        port_vocabulary()
            .iter()
            .enumerate()
            .map(|(index, feature)| ((feature.side, feature.port), index))
            .collect()
    })
}
const GROUPS: [&str; 10] = [
    "network_header-length",
    "network_ip-flags",
    "network_ip-length",
    "network_mss",
    "network_packet-size",
    "network_payload-length",
    "network_tcp-flags",
    "network_time-delta",
    "network_ttl",
    "network_window-size",
];
const EXCLUDED: [(&str, &str); 6] = [
    (
        "network_fragmentation-score",
        "Publisher fragmentation-score formula is unverified",
    ),
    (
        "network_interval-packets",
        "Publisher interval formula is unverified; measured duration is separate",
    ),
    (
        "network_protocols_all_count",
        "TShark decoder/application protocol vocabulary is not a transport-protocol count",
    ),
    (
        "network_protocols_src_count",
        "TShark decoder/application protocol vocabulary is not a transport-protocol count",
    ),
    (
        "network_protocols_dst_count",
        "TShark decoder/application protocol vocabulary is not a transport-protocol count",
    ),
    (
        "network_macs_dst_count",
        "Dataset destination-MAC count has a copied-field defect; native measurement is separate",
    ),
];

#[derive(Debug, Default)]
struct Stats {
    n: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}
impl Stats {
    fn add(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.n += 1;
        if self.n == 1 {
            self.min = value;
            self.max = value;
        }
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        let delta = value - self.mean;
        self.mean += delta / self.n as f64;
        self.m2 += delta * (value - self.mean);
    }
    fn values(&self) -> [Option<f64>; 4] {
        if self.n == 0 {
            return [None; 4];
        }
        [
            Some(self.mean),
            Some(self.max),
            Some(self.min),
            Some((self.m2.max(0.) / self.n as f64).sqrt()),
        ]
    }
}

#[derive(Debug)]
struct ExactSet<T> {
    values: BTreeSet<T>,
    limit: usize,
    overflow: bool,
}
impl<T: Ord> ExactSet<T> {
    fn new(limit: usize) -> Self {
        Self {
            values: BTreeSet::new(),
            limit,
            overflow: false,
        }
    }
    fn insert(&mut self, value: T) {
        if self.values.contains(&value) {
            return;
        }
        if self.values.len() == self.limit {
            self.overflow = true;
        } else {
            self.values.insert(value);
        }
    }
    fn count(&self) -> Option<f64> {
        (!self.overflow).then_some(self.values.len() as f64)
    }
}

#[derive(Debug)]
pub struct WindowFeatures {
    packets: u64,
    outgoing: u64,
    incoming: u64,
    unknown_direction: u64,
    fragmented: u64,
    ipv4_packets: u64,
    ipv6_packets: u64,
    ipv6_fragmented: u64,
    flags: [u64; 6],
    groups: BTreeMap<&'static str, Stats>,
    ipv6_groups: BTreeMap<&'static str, Stats>,
    ips: [ExactSet<IpAddr>; 3],
    ipv6_ips: [ExactSet<IpAddr>; 3],
    macs: [ExactSet<[u8; 6]>; 3],
    ports: [ExactSet<u16>; 3],
    // Fixed vocabulary membership survives a cardinality-set cap: a known
    // positive remains known even when a different count is unavailable.
    observed_port_features: [bool; 192],
    protocols: BTreeSet<&'static str>,
    previous_timestamp: Option<i64>,
    missing_timestamps: u64,
    negative_deltas: u64,
    invalid_ipv4: u64,
    invalid_ipv6: u64,
    incomplete_links: u64,
    incomplete_transport: u64,
    incomplete_tcp: u64,
    incomplete_ports: u64,
    malformed_options: u64,
    tcp_without_mss: u64,
    timestamp_resolution_ns: Option<u64>,
    unknown_timestamp_resolution: bool,
}
impl Default for WindowFeatures {
    fn default() -> Self {
        Self::with_limit(SET_LIMIT)
    }
}
impl WindowFeatures {
    fn with_limit(limit: usize) -> Self {
        Self {
            packets: 0,
            outgoing: 0,
            incoming: 0,
            unknown_direction: 0,
            fragmented: 0,
            ipv4_packets: 0,
            ipv6_packets: 0,
            ipv6_fragmented: 0,
            flags: [0; 6],
            groups: BTreeMap::new(),
            ipv6_groups: BTreeMap::new(),
            ips: std::array::from_fn(|_| ExactSet::new(limit)),
            ipv6_ips: std::array::from_fn(|_| ExactSet::new(limit)),
            macs: std::array::from_fn(|_| ExactSet::new(limit)),
            ports: std::array::from_fn(|_| ExactSet::new(limit)),
            observed_port_features: [false; 192],
            protocols: BTreeSet::new(),
            previous_timestamp: None,
            missing_timestamps: 0,
            negative_deltas: 0,
            invalid_ipv4: 0,
            invalid_ipv6: 0,
            incomplete_links: 0,
            incomplete_transport: 0,
            incomplete_tcp: 0,
            incomplete_ports: 0,
            malformed_options: 0,
            tcp_without_mss: 0,
            timestamp_resolution_ns: None,
            unknown_timestamp_resolution: false,
        }
    }
    fn stat(&mut self, name: &'static str, value: Option<f64>) {
        if let Some(value) = value {
            self.groups.entry(name).or_default().add(value);
        }
    }
    pub fn add(&mut self, header: &Header) {
        let facts = &header.facts;
        self.packets += 1;
        self.outgoing += u64::from(facts.source_local);
        self.incoming += u64::from(facts.destination_local);
        self.unknown_direction += u64::from(!facts.direction_known);
        self.ipv4_packets += u64::from(facts.ip_version == Some(4));
        self.ipv6_packets += u64::from(facts.ip_version == Some(6));
        self.fragmented += u64::from(facts.ip_version == Some(4) && facts.fragmented == Some(true));
        self.ipv6_fragmented +=
            u64::from(facts.ip_version == Some(6) && facts.fragmented == Some(true));
        self.invalid_ipv4 += u64::from(facts.invalid_ip && facts.ip_version == Some(4));
        self.invalid_ipv6 += u64::from(facts.invalid_ip && facts.ip_version == Some(6));
        self.incomplete_links += u64::from(facts.incomplete_link);
        self.incomplete_transport += u64::from(facts.incomplete_transport);
        self.incomplete_tcp += u64::from(facts.incomplete_tcp);
        self.incomplete_ports += u64::from(facts.incomplete_ports);
        self.malformed_options += u64::from(facts.malformed_tcp_options);
        self.tcp_without_mss += u64::from(
            facts.tcp_flags.is_some() && facts.tcp_mss.is_none() && !facts.malformed_tcp_options,
        );
        self.protocols.insert(header.protocol);
        for (side, ip) in [(1, facts.source_ip), (2, facts.destination_ip)] {
            if let Some(ip) = ip {
                if facts.ip_version == Some(4) {
                    self.ips[0].insert(ip);
                    self.ips[side].insert(ip);
                } else if facts.ip_version == Some(6) {
                    self.ipv6_ips[0].insert(ip);
                    self.ipv6_ips[side].insert(ip);
                }
            }
        }
        for (side, mac) in [(1, facts.source_mac), (2, facts.destination_mac)] {
            if let Some(mac) = mac {
                self.macs[0].insert(mac);
                self.macs[side].insert(mac);
            }
        }
        for (side, port) in [(1, facts.source_port), (2, facts.destination_port)] {
            if let Some(port) = port {
                self.ports[0].insert(port);
                self.ports[side].insert(port);
                for direction in [0, side] {
                    if let Some(index) = port_feature_indices().get(&(direction, port)) {
                        self.observed_port_features[*index] = true;
                    }
                }
            }
        }
        if let Some(flags) = facts.tcp_flags {
            for (index, mask) in [16, 1, 8, 4, 2, 32].iter().enumerate() {
                self.flags[index] += u64::from(flags & mask != 0);
            }
        }
        self.stat("network_packet-size", Some(f64::from(header.length)));
        self.stat("network_mss", facts.tcp_mss.map(f64::from));
        self.stat(
            "network_payload-length",
            facts.transport_payload_length.map(f64::from),
        );
        self.stat("network_tcp-flags", facts.tcp_flags.map(f64::from));
        self.stat("network_window-size", facts.tcp_window.map(f64::from));
        if facts.ip_version == Some(4) && !facts.invalid_ip {
            self.stat(
                "network_header-length",
                facts.ip_header_length.map(f64::from),
            );
            self.stat("network_ip-length", facts.ip_length.map(f64::from));
            self.stat("network_ip-flags", facts.ip_flags.map(f64::from));
            self.stat("network_ttl", facts.ttl.map(f64::from));
        }
        if facts.ip_version == Some(6) && !facts.invalid_ip {
            for (name, value) in [
                (
                    "live_ipv6_header-length",
                    facts.ip_header_length.map(f64::from),
                ),
                ("live_ipv6_ip-length", facts.ip_length.map(f64::from)),
                ("live_ipv6_hop-limit", facts.ttl.map(f64::from)),
            ] {
                if let Some(value) = value {
                    self.ipv6_groups.entry(name).or_default().add(value);
                }
            }
        }
        if let Some(now) = facts.timestamp_ns {
            if let Some(resolution) = facts.timestamp_resolution_ns.filter(|v| *v > 0) {
                if self
                    .timestamp_resolution_ns
                    .is_some_and(|v| v != resolution)
                {
                    self.unknown_timestamp_resolution = true;
                }
                self.timestamp_resolution_ns = Some(resolution);
            } else {
                self.unknown_timestamp_resolution = true;
            }
            if let Some(previous) = self.previous_timestamp {
                // i128 subtraction avoids overflow on malformed extreme input.
                let delta = (i128::from(now) - i128::from(previous)) as f64 / 1e9;
                self.negative_deltas += u64::from(delta < 0.);
                self.stat("network_time-delta", Some(delta));
            }
            self.previous_timestamp = Some(now);
        } else {
            self.missing_timestamps += 1;
            self.previous_timestamp = None;
        }
    }

    pub fn observation(&self, seconds: f64, capture_drops: u64) -> Value {
        let mut features: BTreeMap<String, Option<f64>> = BTreeMap::new();
        let mut unavailable: BTreeMap<String, String> = BTreeMap::new();
        let mut samples: BTreeMap<String, u64> = BTreeMap::new();
        let mut live_only: BTreeMap<String, Option<f64>> = BTreeMap::new();
        let mut port_presence: BTreeMap<String, Option<f64>> = port_vocabulary()
            .iter()
            .enumerate()
            .map(|(index, feature)| {
                let set = &self.ports[feature.side];
                let value = if self.observed_port_features[index] {
                    Some(1.)
                } else if set.overflow || self.incomplete_ports > 0 {
                    None
                } else {
                    Some(0.)
                };
                (feature.name.clone(), value)
            })
            .collect();
        for group in GROUPS {
            let empty = Stats::default();
            let stats = self.groups.get(group).unwrap_or(&empty);
            samples.insert(group.into(), stats.n);
            let invalid = match group {
                "network_header-length"
                | "network_ip-length"
                | "network_ip-flags"
                | "network_ttl" => self.invalid_ipv4 > 0 || self.incomplete_links > 0,
                "network_mss" => self.incomplete_tcp > 0 || self.malformed_options > 0,
                "network_payload-length" => self.incomplete_transport > 0,
                "network_tcp-flags" | "network_window-size" => self.incomplete_tcp > 0,
                "network_time-delta" => self.missing_timestamps > 0,
                _ => false,
            };
            for (suffix, measured) in ["avg", "max", "min", "std_deviation"]
                .into_iter()
                .zip(stats.values())
            {
                let name = format!("{group}_{suffix}");
                let value = if invalid { None } else { measured };
                if value.is_none() {
                    unavailable.insert(
                        name.clone(),
                        if invalid {
                            "Incomplete measurement family"
                        } else {
                            "No applicable observations"
                        }
                        .into(),
                    );
                }
                features.insert(name, value);
            }
        }
        for (name, value) in [
            ("network_packets_all_count", Some(self.packets as f64)),
            (
                "network_packets_src_count",
                (self.unknown_direction == 0).then_some(self.outgoing as f64),
            ),
            (
                "network_packets_dst_count",
                (self.unknown_direction == 0).then_some(self.incoming as f64),
            ),
            (
                "network_fragmented-packets",
                (self.invalid_ipv4 == 0 && self.incomplete_links == 0)
                    .then_some(self.fragmented as f64),
            ),
        ] {
            features.insert(name.into(), value);
            if value.is_none() {
                unavailable.insert(name.into(), "Incomplete direction or IP measurement".into());
            }
        }
        for (index, direction) in ["all", "src", "dst"].into_iter().enumerate() {
            for (family, count) in [
                (
                    "ips",
                    if self.invalid_ipv4 == 0 && self.incomplete_links == 0 {
                        self.ips[index].count()
                    } else {
                        None
                    },
                ),
                ("macs", self.macs[index].count()),
                (
                    "ports",
                    if self.incomplete_ports == 0 {
                        self.ports[index].count()
                    } else {
                        None
                    },
                ),
            ] {
                let name = format!("network_{family}_{direction}_count");
                features.insert(name.clone(), count);
                if count.is_none() {
                    unavailable.insert(
                        name,
                        "Exact cardinality limit or incomplete transport header".into(),
                    );
                }
            }
            live_only.insert(
                format!("live_ipv6_ips_{direction}_count"),
                if self.invalid_ipv6 == 0 && self.incomplete_links == 0 {
                    self.ipv6_ips[index].count()
                } else {
                    None
                },
            );
        }
        live_only.insert("live_macs_dst_count".into(), self.macs[2].count());
        live_only.insert(
            "live_parsed_protocol_count".into(),
            Some(self.protocols.len() as f64),
        );
        live_only.insert(
            "live_ipv4_packets_count".into(),
            Some(self.ipv4_packets as f64),
        );
        live_only.insert(
            "live_ipv6_packets_count".into(),
            Some(self.ipv6_packets as f64),
        );
        live_only.insert(
            "live_ipv6_fragmented_packets".into(),
            (self.invalid_ipv6 == 0 && self.incomplete_links == 0)
                .then_some(self.ipv6_fragmented as f64),
        );
        for (name, stats) in &self.ipv6_groups {
            samples.insert((*name).into(), stats.n);
            for (suffix, value) in ["avg", "max", "min", "std_deviation"]
                .into_iter()
                .zip(stats.values())
            {
                live_only.insert(
                    format!("{name}_{suffix}"),
                    if self.invalid_ipv6 == 0 { value } else { None },
                );
            }
        }
        for (index, flag) in ["ack", "fin", "psh", "rst", "syn", "urg"]
            .into_iter()
            .enumerate()
        {
            let name = format!("network_tcp-flags-{flag}_count");
            features.insert(
                name.clone(),
                (self.incomplete_tcp == 0).then_some(self.flags[index] as f64),
            );
            if self.incomplete_tcp > 0 {
                unavailable.insert(name, "Incomplete TCP header".into());
            }
        }
        for (name, reason) in EXCLUDED {
            features.insert(name.into(), None);
            unavailable.insert(name.into(), reason.into());
        }
        let mut derivatives = Vec::new();
        for direction in ["all", "src", "dst"] {
            derivatives.push((
                format!("network_packets_{direction}_count__per_second"),
                features[&format!("network_packets_{direction}_count")],
                Some(seconds),
            ));
        }
        for flag in ["ack", "fin", "psh", "rst", "syn", "urg"] {
            derivatives.push((
                format!("network_tcp-flags-{flag}__fraction_of_packets"),
                features[&format!("network_tcp-flags-{flag}_count")],
                features["network_packets_all_count"],
            ));
        }
        for direction in ["src", "dst"] {
            derivatives.push((
                format!("network_{direction}_packet_fraction"),
                features[&format!("network_packets_{direction}_count")],
                features["network_packets_all_count"],
            ));
        }
        for (name, numerator, denominator) in [
            (
                "network_packet_size_coefficient_of_variation",
                "network_packet-size_std_deviation",
                "network_packet-size_avg",
            ),
            (
                "network_payload_fraction",
                "network_payload-length_avg",
                "network_packet-size_avg",
            ),
            (
                "network_packets_per_reported_ip",
                "network_packets_all_count",
                "network_ips_all_count",
            ),
            (
                "network_packets_per_reported_port",
                "network_packets_all_count",
                "network_ports_all_count",
            ),
        ] {
            derivatives.push((name.into(), features[numerator], features[denominator]));
        }
        for (name, numerator, denominator) in derivatives {
            let value = numerator
                .zip(denominator)
                .and_then(|(a, b)| (b.is_finite() && b > 0.).then_some(a / b))
                .filter(|v| v.is_finite());
            if value.is_none() {
                unavailable.insert(
                    name.clone(),
                    "Unavailable numerator or nonpositive denominator".into(),
                );
            }
            features.insert(name, value);
        }
        let global_reason = if capture_drops > 0 {
            Some("Capture loss makes the window incomplete")
        } else if !seconds.is_finite() || seconds <= 0. {
            Some("Invalid measured window duration")
        } else if self.packets == 0 {
            Some("No captured packets; no observation for inference")
        } else {
            None
        };
        if let Some(reason) = global_reason {
            for (name, value) in &mut features {
                *value = None;
                unavailable.insert(name.clone(), reason.into());
            }
            for value in live_only.values_mut() {
                *value = None;
            }
            for value in port_presence.values_mut() {
                *value = None;
            }
        }
        json!({"schema_version":"npcap-network-v1", "publisher_equivalence_verified":false,
            "window_seconds":if seconds.is_finite(){Some(seconds)}else{None}, "features":features,
            "feature_names":features.keys().collect::<Vec<_>>(), "sample_counts":samples,
            "unavailable":unavailable, "live_only_features":live_only,
            "port_presence":port_presence,
            "port_presence_contract":{"candidate_fields":port_vocabulary().len(),
                "semantics":"Observed numeric TCP/UDP endpoint token is 1; complete valid absence is 0; unknown absence is null",
                "direction":"Source and destination are header endpoints, independent of local packet direction",
                "invalidation":"Overflow or incomplete transport invalidates unknown absence; capture loss, empty windows or invalid duration invalidate every value",
                "application_identity_inferred":false},
            "statistical_convention":"population standard deviation ddof=0; first packet has no time delta; signed capture deltas in seconds",
            "unverified_fields":{"network_payload-length_*":"Provisional TCP/UDP/ICMP transport payload length; not uniformly IP payload",
                "network_mss_*":"Present valid MSS options only; absence is not zero",
                "network_ips_*":"Observed IPv4 header addresses only; IPv6 and ARP protocol addresses are excluded",
                "IPv6":"IPv6 summaries have live_ names; IPv4 candidate summaries are not filled with IPv6 analogues"},
            "timestamp_precision":{"storage_unit":"nanoseconds", "source_resolution_ns":if self.unknown_timestamp_resolution { None } else { self.timestamp_resolution_ns }, "accuracy_verified":false},
            "provenance":{"packet_size":"Original capture-reported frame length; includes supplied padding, no FCS reconstruction",
                "window_membership":"Capture-order admission involving the selected local IP or MAC; first admitted packet has no prior delta",
                "direction":"Packet src/dst counts mean local outgoing/incoming; endpoint sets use source/destination fields across all packets; self-traffic counts in both directions",
                "fragmented_packets":"IPv4 MF bit or nonzero fragment offset; IPv6 fragment measurements have separate live_ names",
                "transport":"TCP/UDP ports; raw unscaled TCP window; TCP-only flags and explicitly present MSS options",
                "cardinality":"Bounded exact endpoint sets; overflow makes the affected count unavailable"},
            "profile_contract":{"raw_dataset_fields":64,"derived_fields":15,"core_raw_fields":54,"core_derived_fields":14,
                "extended_raw_fields":58,"extended_derived_fields":15,"withheld_fields":EXCLUDED.iter().map(|(name,_)| *name).collect::<Vec<_>>(),
                "extended_only_fields":["network_payload-length_avg","network_payload-length_max","network_payload-length_min","network_payload-length_std_deviation","network_payload_fraction"]},
            "quality":{"capture_drops":capture_drops,"packets_observed":self.packets,"missing_timestamps":self.missing_timestamps,
                "negative_time_deltas":self.negative_deltas,"invalid_ipv4_headers":self.invalid_ipv4,"invalid_ipv6_headers":self.invalid_ipv6,
                "incomplete_transport_headers":self.incomplete_transport,"incomplete_tcp_headers":self.incomplete_tcp,
                "incomplete_port_headers":self.incomplete_ports,"incomplete_link_headers":self.incomplete_links,
                "malformed_tcp_options":self.malformed_options,"tcp_packets_without_mss_option":self.tcp_without_mss,
                "unknown_direction_packets":self.unknown_direction,"exact_cardinality_limit":self.ips[0].limit,
                "set_overflow":self.ips.iter().any(|s|s.overflow)
                    || self.ipv6_ips.iter().any(|s|s.overflow) || self.macs.iter().any(|s|s.overflow) || self.ports.iter().any(|s|s.overflow)},
            "privacy":{"payload_retained":false,"raw_address_identities_in_features":false},
            "model_compatibility":"Candidate measurement profile only; no existing checkpoint is silently given additional inputs"})
    }
}

// ------------------------------------------------------------------ live77 model-row projection (plan step 5)
// The 252 live77 inputs, by the bundle's names, as defined by
// framework/training/csv_v3/live77_contract.json and reproduced exactly by its
// Python reference (live77_reference.py; shared fixtures
// evaluation/fixtures/live77/frames_v1.json). Parity is with that contract, not
// with the unpublished DataSense exporter. Owner decisions (3 Nov 2026): 10 s
// tumbling windows with window_duration_seconds = 10 and per_second = count / 10;
// device-MAC admission of all frames (IPv6 left out); 12-bit TCP flags;
// MSS-bearing packets only; MF-only fragments; first 1000 distinct ports per
// direction in arrival order; zero-packet windows scored with
// quality_no_packets_observed = 1; capture loss or unreadable headers leave the
// affected inputs missing. Float rules copy the reference: float64 statistics
// over arrival-order samples (mean = exact integer sum / n, two-pass population
// deviation) cast to float32; ratio = f32(f64(f32(num)) / f64(f32(den))).
pub const LIVE77_NOMINAL_SECONDS: f64 = 10.0;
pub const LIVE77_PORT_LIMIT: usize = 1000;
/// Arrival-order samples kept per family for the two-pass deviation; past it, exact integer moments are used.
pub const LIVE77_SAMPLE_LIMIT: usize = 1 << 21;
/// Distinct MAC/IPv4 values per set; past it the count is missing (never clipped).
pub const LIVE77_SET_LIMIT: usize = 1 << 20;
const LIVE77_FLAGS: [(&str, u16); 6] = [
    ("ack", 16),
    ("fin", 1),
    ("psh", 8),
    ("rst", 4),
    ("syn", 2),
    ("urg", 32),
];
const LIVE77_FAMILIES: [(&str, u8); 7] = [
    ("header-length", DAMAGE_IPV4_HEADER),
    ("ip-flags", DAMAGE_IPV4_HEADER),
    ("ip-length", DAMAGE_IPV4_HEADER),
    ("ttl", DAMAGE_IPV4_HEADER),
    ("tcp-flags", DAMAGE_TCP_HEADER),
    ("window-size", DAMAGE_TCP_HEADER),
    ("mss", DAMAGE_TCP_OPTIONS),
];
const LIVE77_STATS: [&str; 4] = ["avg", "max", "min", "std_deviation"];

#[derive(Debug, Default)]
struct Samples {
    values: Vec<u16>,
    bounded: bool,
    n: u64,
    sum: u128,
    squares: u128,
    min: u16,
    max: u16,
}
impl Samples {
    fn add(&mut self, value: u16, limit: usize) {
        if self.n == 0 || value < self.min {
            self.min = value;
        }
        if self.n == 0 || value > self.max {
            self.max = value;
        }
        self.n += 1;
        self.sum += u128::from(value);
        self.squares += u128::from(value) * u128::from(value);
        if !self.bounded && self.values.len() < limit {
            self.values.push(value);
        } else if !self.bounded {
            self.bounded = true;
            self.values = Vec::new();
        }
    }
    /// [mean, max, min, population std]; `None` when the family has no values.
    fn summary(&self) -> Option<[f64; 4]> {
        if self.n == 0 {
            return None;
        }
        let n = self.n as f64;
        let mean = self.sum as f64 / n;
        let deviation = if self.bounded {
            let numerator = u128::from(self.n) * self.squares - self.sum * self.sum;
            (numerator as f64 / (n * n)).sqrt()
        } else {
            let mut acc = 0.0;
            for value in &self.values {
                let delta = f64::from(*value) - mean;
                acc += delta * delta;
            }
            (acc / n).sqrt()
        };
        Some([mean, f64::from(self.max), f64::from(self.min), deviation])
    }
}

#[derive(Debug, Default)]
struct KeptPorts {
    kept: Vec<u16>,
    seen: BTreeSet<u16>,
    capped: bool,
}
impl KeptPorts {
    fn add(&mut self, port: u16) {
        if self.seen.contains(&port) {
            return;
        }
        if self.kept.len() >= LIVE77_PORT_LIMIT {
            // First 1000 distinct in arrival order; later ports set nothing.
            self.capped = true;
            return;
        }
        self.seen.insert(port);
        self.kept.push(port);
    }
}

/// One 10 s window of device-MAC frames, projected to the live77 model row by name.
#[derive(Debug)]
pub struct Live77Features {
    device: Option<[u8; 6]>,
    sample_limit: usize,
    packets: [u64; 3],
    macs: [ExactSet<[u8; 6]>; 2],
    ips: [ExactSet<[u8; 4]>; 2],
    fragmented: u64,
    flags: [u64; 6],
    samples: [Samples; 7],
    ports: [KeptPorts; 2],
    damage: u8,
}
impl Default for Live77Features {
    fn default() -> Self {
        Self::new(None)
    }
}
impl Live77Features {
    /// `device` is the selected adapter's MAC; without it no frame is admitted and no row exists.
    pub fn new(device: Option<[u8; 6]>) -> Self {
        Self::with_limits(device, LIVE77_SAMPLE_LIMIT, LIVE77_SET_LIMIT)
    }
    pub fn with_limits(device: Option<[u8; 6]>, sample_limit: usize, set_limit: usize) -> Self {
        Self {
            device,
            sample_limit,
            packets: [0; 3],
            macs: std::array::from_fn(|_| ExactSet::new(set_limit)),
            ips: std::array::from_fn(|_| ExactSet::new(set_limit)),
            fragmented: 0,
            flags: [0; 6],
            samples: Default::default(),
            ports: Default::default(),
            damage: 0,
        }
    }
    pub fn device(&self) -> Option<[u8; 6]> {
        self.device
    }
    pub fn packets(&self) -> u64 {
        self.packets[0]
    }
    pub fn add_frame(&mut self, bytes: &[u8], wire_length: u32) {
        if let Some(frame) = self
            .device
            .and_then(|device| live77_frame(bytes, wire_length, device))
        {
            self.add(&frame);
        }
    }
    fn sample(&mut self, family: usize, value: u16) {
        self.samples[family].add(value, self.sample_limit);
    }
    fn add(&mut self, frame: &Live77Frame) {
        self.packets[0] += 1;
        self.packets[if frame.sent { 1 } else { 2 }] += 1;
        self.macs[0].insert(frame.source_mac);
        self.macs[1].insert(frame.destination_mac);
        if let Some(ip) = frame.source_ip {
            self.ips[0].insert(ip);
        }
        if let Some(ip) = frame.destination_ip {
            self.ips[1].insert(ip);
        }
        if let Some(ip) = frame.ipv4 {
            self.sample(0, ip.header_length);
            self.sample(1, ip.flags);
            self.sample(2, ip.total_length);
            self.sample(3, u16::from(ip.ttl));
            self.fragmented += u64::from(ip.more_fragments);
        }
        if let Some(tcp) = frame.tcp {
            self.sample(4, tcp.flags12);
            self.sample(5, tcp.window);
            for (index, (_, mask)) in LIVE77_FLAGS.iter().enumerate() {
                self.flags[index] += u64::from(tcp.flags12 & mask != 0);
            }
        }
        if let Some(mss) = frame.mss {
            self.sample(6, mss);
        }
        if let Some((source, destination)) = frame.ports {
            self.ports[0].add(source);
            self.ports[1].add(destination);
        }
        self.damage |= frame.damage;
    }
    fn damaged(&self, bit: u8) -> bool {
        self.damage & bit != 0
    }
    fn distinct<T: Ord>(sets: &[ExactSet<T>; 2], index: usize) -> Option<f64> {
        match index {
            0 => (!sets[0].overflow && !sets[1].overflow)
                .then(|| sets[0].values.union(&sets[1].values).count() as f64),
            side => sets[side - 1].count(),
        }
    }
    fn ips_count(&self, index: usize) -> Option<f64> {
        if self.damaged(DAMAGE_IPV4_OR_ARP_HEADER) {
            None
        } else {
            Self::distinct(&self.ips, index)
        }
    }
    fn flag_count(&self, index: usize) -> Option<f64> {
        (!self.damaged(DAMAGE_TCP_HEADER)).then_some(self.flags[index] as f64)
    }
    fn value(&self, name: &str) -> Result<Option<f64>, String> {
        let unknown = || format!("Unknown live77 model input: {name}");
        let packets = self.packets[0];
        let all = Some(packets as f64);
        let side = |d: &str| ["all", "src", "dst"].iter().position(|v| *v == d);
        let ratio = |numerator: Option<f64>, denominator: Option<f64>| {
            numerator.zip(denominator).and_then(|(a, b)| {
                ((b as f32) > 0.).then(|| (f64::from(a as f32) / f64::from(b as f32)) as f32 as f64)
            })
        };
        match name {
            "network_fragmented-packets" => {
                return Ok((!self.damaged(DAMAGE_IPV4_HEADER)).then_some(self.fragmented as f64))
            }
            "window_duration_seconds" => return Ok(Some(LIVE77_NOMINAL_SECONDS)),
            "quality_no_packets_observed" => return Ok(Some(f64::from(u8::from(packets == 0)))),
            "network_packets_per_reported_ip" => return Ok(ratio(all, self.ips_count(0))),
            "network_macs_all_count" => return Ok(Self::distinct(&self.macs, 0)),
            "network_macs_src_count" => return Ok(Self::distinct(&self.macs, 1)),
            "network_macs_dst__observed_distinct_count" => {
                return Ok(Self::distinct(&self.macs, 2))
            }
            "network_src_packet_fraction" => return Ok(ratio(Some(self.packets[1] as f64), all)),
            "network_dst_packet_fraction" => return Ok(ratio(Some(self.packets[2] as f64), all)),
            _ => {}
        }
        for (family, (prefix, bit)) in LIVE77_FAMILIES.iter().enumerate() {
            for (stat, suffix) in LIVE77_STATS.iter().enumerate() {
                if name
                    .strip_prefix("network_")
                    .and_then(|n| n.strip_prefix(prefix))
                    .is_some_and(|rest| rest.strip_prefix('_') == Some(suffix))
                {
                    // NaN when packets_all == 0 or damaged; 0 when the family is empty but packets exist.
                    return Ok((packets > 0 && !self.damaged(*bit))
                        .then(|| self.samples[family].summary().map_or(0., |s| s[stat])));
                }
            }
        }
        if let Some(rest) = name.strip_prefix("network_packets_") {
            let (direction, rate) = match rest.strip_suffix("_count__per_second") {
                Some(direction) => (direction, true),
                None => (rest.strip_suffix("_count").ok_or_else(unknown)?, false),
            };
            let count = Some(self.packets[side(direction).ok_or_else(unknown)?] as f64);
            return Ok(if rate {
                ratio(count, Some(LIVE77_NOMINAL_SECONDS))
            } else {
                count
            });
        }
        if let Some(direction) = name
            .strip_prefix("network_ips_")
            .and_then(|n| n.strip_suffix("_count"))
        {
            return Ok(self.ips_count(side(direction).ok_or_else(unknown)?));
        }
        if let Some(rest) = name.strip_prefix("network_tcp-flags-") {
            let (flag, fraction) = match rest.strip_suffix("__fraction_of_packets") {
                Some(flag) => (flag, true),
                None => (rest.strip_suffix("_count").ok_or_else(unknown)?, false),
            };
            let index = LIVE77_FLAGS
                .iter()
                .position(|(f, _)| *f == flag)
                .ok_or_else(unknown)?;
            let count = self.flag_count(index);
            return Ok(if fraction { ratio(count, all) } else { count });
        }
        if let Some((direction, token)) = live77_port_name(name) {
            let ports = &self.ports[direction - 1];
            if token == "UNSEEN" {
                let unseen = ports
                    .kept
                    .iter()
                    .any(|p| !port_feature_indices().contains_key(&(direction, *p)));
                return Ok(if unseen {
                    Some(1.)
                } else if self.damaged(DAMAGE_PORTS) {
                    None
                } else {
                    Some(0.)
                });
            }
            let port = token
                .parse::<u16>()
                .ok()
                .filter(|p| p.to_string() == token)
                .filter(|p| port_feature_indices().contains_key(&(direction, *p)))
                .ok_or_else(|| {
                    format!("Live77 port input outside the embedded vocabulary: {name}")
                })?;
            return Ok(if ports.seen.contains(&port) {
                Some(1.)
            } else if ports.capped || self.damaged(DAMAGE_PORTS) {
                None
            } else {
                Some(0.)
            });
        }
        Err(unknown())
    }
    /// The model row in the order of `names` (the bundle's feature names). Unknown names, a port outside
    /// the embedded vocabulary, an incomplete vocabulary beside UNSEEN or a missing device MAC are errors.
    /// A capture-drop window is all missing (it is not scored).
    pub fn row(&self, names: &[String], capture_drop: bool) -> Result<Vec<Option<f32>>, String> {
        if self.device.is_none() {
            return Err("Live77 needs the selected adapter's MAC address: device-MAC admission cannot run without it".into());
        }
        for direction in [1, 2] {
            let mut ports = BTreeSet::new();
            let mut unseen = false;
            for (side, token) in names.iter().filter_map(|n| live77_port_name(n)) {
                if side == direction {
                    unseen |= token == "UNSEEN";
                    ports.insert(token);
                }
            }
            let vocabulary = port_vocabulary()
                .iter()
                .filter(|f| f.side == direction)
                .count();
            if unseen && ports.len() != vocabulary + 1 {
                return Err(
                    "Live77 UNSEEN needs the complete embedded port vocabulary of its direction"
                        .into(),
                );
            }
        }
        let row = names
            .iter()
            .map(|name| self.value(name).map(|v| v.map(|v| v as f32)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(if capture_drop {
            vec![None; row.len()]
        } else {
            row
        })
    }
}

/// (1 = src, 2 = dst, token) for `network_ports_{src,dst}__present_{token}`.
fn live77_port_name(name: &str) -> Option<(usize, &str)> {
    let (prefix, token) = name.split_once("__present_")?;
    let side = match prefix {
        "network_ports_src" => 1,
        "network_ports_dst" => 2,
        _ => return None,
    };
    Some((side, token))
}

#[cfg(test)]
mod tests {
    use super::super::packet::PacketFacts;
    use super::*;

    fn port_header(source: u16, destination: u16) -> Header {
        Header {
            protocol: "UDP",
            source: String::new(),
            destination: String::new(),
            length: 42,
            flags: 0,
            destination_port: None,
            complete_tcp: true,
            facts: PacketFacts {
                source_port: Some(source),
                destination_port: Some(destination),
                source_local: false,
                destination_local: true,
                direction_known: true,
                ..PacketFacts::default()
            },
        }
    }

    #[test]
    fn port_membership_preserves_endpoint_direction_and_unknown_absence() {
        let mut window = WindowFeatures::with_limit(2);
        window.add(&port_header(1883, 12345));
        let row = window.observation(10., 0);
        assert_eq!(row["port_presence"].as_object().unwrap().len(), 192);
        assert_eq!(row["port_presence"]["network_ports_src__present_1883"], 1.);
        assert_eq!(row["port_presence"]["network_ports_dst__present_1883"], 0.);
        // Incoming traffic still has a remote source header port.
        assert_eq!(row["features"]["network_packets_dst_count"], 1.);
        window.add(&port_header(53, 80));
        window.add(&port_header(23, 22));
        let overflow = window.observation(10., 0);
        assert_eq!(
            overflow["port_presence"]["network_ports_src__present_1883"],
            1.
        );
        assert_eq!(
            overflow["port_presence"]["network_ports_src__present_23"],
            1.
        );
        assert!(overflow["port_presence"]["network_ports_src__present_9200"].is_null());
        assert!(overflow["features"]["network_ports_all_count"].is_null());
        let mut incomplete = port_header(80, 53);
        incomplete.facts.incomplete_ports = true;
        window.add(&incomplete);
        let row = window.observation(10., 0);
        assert_eq!(row["port_presence"]["network_ports_src__present_1883"], 1.);
        assert!(row["port_presence"]["network_ports_dst__present_1883"].is_null());
    }

    #[test]
    fn invalid_windows_invalidate_all_port_projections() {
        let mut window = WindowFeatures::default();
        for row in [window.observation(10., 0), window.observation(0., 0)] {
            assert_eq!(row["port_presence"].as_object().unwrap().len(), 192);
            assert!(row["port_presence"]
                .as_object()
                .unwrap()
                .values()
                .all(Value::is_null));
        }
        window.add(&port_header(1883, 12345));
        for row in [
            window.observation(10., 1),
            window.observation(-1., 0),
            window.observation(f64::NAN, 0),
        ] {
            assert!(row["port_presence"]
                .as_object()
                .unwrap()
                .values()
                .all(Value::is_null));
        }
    }

    #[test]
    fn port_vocabulary_has_exactly_192_unique_numeric_endpoint_features() {
        assert_eq!(port_vocabulary().len(), 192);
        assert_eq!(port_feature_indices().len(), 192);
        let directions: BTreeSet<_> = port_vocabulary()
            .iter()
            .map(|feature| feature.side)
            .collect();
        assert_eq!(directions, [1, 2].into());
    }

    // ---------------------------------------------------------------- live77 model-row projection (plan step 5)
    const CONTRACT: &str = "framework/training/csv_v3/live77_contract.json";
    const FIXTURES: &str = "evaluation/fixtures/live77/frames_v1.json";

    fn workspace_json(relative: &str) -> Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(4)
            .unwrap()
            .join(relative);
        serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{path:?}: {e}")))
            .unwrap()
    }
    fn contract_names() -> Vec<String> {
        serde_json::from_value(workspace_json(CONTRACT)["feature_names"].clone()).unwrap()
    }
    fn hex(text: &str) -> Vec<u8> {
        (0..text.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
            .collect()
    }
    fn mac_of(text: &str) -> [u8; 6] {
        hex(&text.replace(':', "")).try_into().unwrap()
    }
    /// Sparse fixture format: default (0; null = NaN), then nan_prefixes, then nan names, then values.
    fn expected_row(spec: &Value, names: &[String]) -> Vec<Option<f32>> {
        let default = match spec.get("default") {
            Some(Value::Null) => None,
            Some(v) => Some(v.as_f64().unwrap() as f32),
            None => Some(0.),
        };
        let mut row = vec![default; names.len()];
        for prefix in spec["nan_prefixes"].as_array().into_iter().flatten() {
            let prefix = prefix.as_str().unwrap();
            assert!(names.iter().any(|n| n.starts_with(prefix)), "{prefix}");
            for (i, n) in names.iter().enumerate() {
                if n.starts_with(prefix) {
                    row[i] = None;
                }
            }
        }
        for name in spec["nan"].as_array().into_iter().flatten() {
            row[names.iter().position(|n| n == name).unwrap()] = None;
        }
        for (name, value) in spec["values"].as_object().into_iter().flatten() {
            row[names.iter().position(|n| n == name).unwrap()] = value.as_f64().map(|v| v as f32);
        }
        row
    }
    fn fixture_row(features: &mut Live77Features, fixture: &Value) {
        for frame in fixture["frames"].as_array().unwrap() {
            let bytes = hex(frame["hex"].as_str().unwrap());
            let wire = frame["wire_length"]
                .as_u64()
                .map_or(bytes.len() as u32, |v| v as u32);
            features.add_frame(&bytes, wire);
        }
    }
    fn differences(got: &[Option<f32>], want: &[Option<f32>], names: &[String]) -> Vec<String> {
        names
            .iter()
            .zip(got.iter().zip(want))
            .filter(|(_, (g, w))| g.map(f32::to_bits) != w.map(f32::to_bits))
            .map(|(n, (g, w))| format!("{n}: got {g:?}, want {w:?}"))
            .collect()
    }

    #[test]
    fn live77_rows_match_every_shared_frame_fixture_exactly() {
        let names = contract_names();
        let fixtures = workspace_json(FIXTURES);
        assert_eq!(fixtures["nominal_seconds"], LIVE77_NOMINAL_SECONDS);
        let device = mac_of(fixtures["device"]["mac"].as_str().unwrap());
        let cases = fixtures["fixtures"].as_array().unwrap();
        assert_eq!(cases.len(), 18);
        for fixture in cases {
            let mut features = Live77Features::new(Some(device));
            fixture_row(&mut features, fixture);
            let drop = fixture["capture_drop"] == true;
            let got = features.row(&names, drop).unwrap();
            let want = expected_row(&fixture["expected"], &names);
            let bad = differences(&got, &want, &names);
            assert!(bad.is_empty(), "{}: {bad:#?}", fixture["id"]);
        }
    }

    #[test]
    fn live77_vocabulary_window_and_names_follow_the_frozen_contract() {
        let contract = workspace_json(CONTRACT);
        assert_eq!(
            contract["window"]["nominal_duration_seconds"],
            LIVE77_NOMINAL_SECONDS
        );
        assert_eq!(
            contract["conventions"]["port_cap_1000"]["params"]["limit"],
            LIVE77_PORT_LIMIT
        );
        for (side, direction) in [(1, "src"), (2, "dst")] {
            let embedded = port_vocabulary()
                .iter()
                .filter(|f| f.side == side)
                .map(|f| f.port)
                .collect::<Vec<_>>();
            assert_eq!(
                contract["port_vocabulary"][direction],
                json!(embedded),
                "{direction}"
            );
        }
        let names = contract_names();
        assert_eq!(names.len(), 252);
        // Every contract input is projected by name; the observation view's withheld name is not.
        let row = Live77Features::new(Some([2, 0, 0, 0, 0, 0x50]))
            .row(&names, false)
            .unwrap();
        assert_eq!(row.len(), 252);
        assert!(!names.iter().any(|n| n == "network_macs_dst_count"));
    }

    #[test]
    fn live77_row_refuses_unknown_inputs_unknown_device_and_foreign_ports() {
        let names = contract_names();
        let device = Some([2, 0, 0, 0, 0, 0x50]);
        let features = Live77Features::new(device);
        for bad in [
            "network_macs_dst_count",
            "network_ports_all_count",
            "network_ports_src__present_8080",
            "network_packet-size_avg",
            "network_ports_src__present_",
        ] {
            let mut changed = names.clone();
            changed.push(bad.into());
            assert!(features.row(&changed, false).is_err(), "{bad}");
        }
        // UNSEEN needs the full vocabulary of its direction, or it would count a modelled port as unseen.
        let partial = names
            .iter()
            .filter(|n| n.as_str() != "network_ports_dst__present_80")
            .cloned()
            .collect::<Vec<_>>();
        assert!(features.row(&partial, false).is_err());
        // Without the device MAC nothing can be admitted, so even an empty window is not a zero-packet row.
        let error = Live77Features::new(None).row(&names, false).unwrap_err();
        assert!(error.contains("MAC"), "{error}");
        // A capture-drop window is all missing.
        let mut busy = Live77Features::new(device);
        busy.add_frame(&hex("ffffffffffff020000000050080600010800060400010200000000500a00000100000000000a000002"), 42);
        assert!(busy.row(&names, true).unwrap().iter().all(Option::is_none));
    }

    #[test]
    fn live77_bounded_statistics_and_sets_stay_honest() {
        let names = contract_names();
        let fixtures = workspace_json(FIXTURES);
        let device = mac_of(fixtures["device"]["mac"].as_str().unwrap());
        // A family past its sample bound switches to exact integer moments; values stay float32 equal here.
        for fixture in fixtures["fixtures"].as_array().unwrap() {
            let mut exact = Live77Features::new(Some(device));
            let mut bounded = Live77Features::with_limits(Some(device), 1, LIVE77_SET_LIMIT);
            fixture_row(&mut exact, fixture);
            fixture_row(&mut bounded, fixture);
            let (a, b) = (
                exact.row(&names, false).unwrap(),
                bounded.row(&names, false).unwrap(),
            );
            for ((name, x), y) in names.iter().zip(&a).zip(&b) {
                match (x, y) {
                    (Some(x), Some(y)) => {
                        assert!((x - y).abs() <= 1e-6 * x.abs().max(1.), "{name}: {x} {y}")
                    }
                    _ => assert_eq!(x.is_none(), y.is_none(), "{name}"),
                }
            }
        }
        // Distinct-address sets past their bound make the count missing, never a clipped number.
        let fixture = fixtures["fixtures"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["id"] == "tcp_syn_with_and_without_mss")
            .unwrap();
        let mut tight = Live77Features::with_limits(Some(device), LIVE77_SAMPLE_LIMIT, 1);
        fixture_row(&mut tight, fixture);
        let row = tight.row(&names, false).unwrap();
        let get = |n: &str| row[names.iter().position(|v| v == n).unwrap()];
        assert_eq!(get("network_packets_all_count"), Some(2.));
        assert_eq!(get("network_macs_src_count"), None);
        assert_eq!(get("network_ips_all_count"), None);
        assert_eq!(get("network_packets_per_reported_ip"), None);
    }

    #[test]
    fn population_stats_and_exact_set_limits_are_explicit() {
        let mut stats = Stats::default();
        stats.add(1.);
        stats.add(3.);
        assert_eq!(stats.values(), [Some(2.), Some(3.), Some(1.), Some(1.)]);
        assert_eq!(Stats::default().values(), [None; 4]);
        let mut set = ExactSet::new(2);
        set.insert(1);
        set.insert(2);
        set.insert(2);
        assert_eq!(set.count(), Some(2.));
        set.insert(3);
        assert_eq!(set.count(), None);
    }
}
