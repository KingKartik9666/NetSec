# Npcap and DataSense feature contract audit

Date: 27 October 2026. Scope: feature definitions and training-only source
inspection. This audit does not train, access a test split, capture traffic or
promote a model. Its only authored artifact is this document.

## Evidence and scope

The supplied raw CSV has **76 network columns: 64 numeric fields and 12 list
fields**. The 71-column prepared numeric manifest adds seven log/sensor fields.
The list fields are all/source/destination sets of IPs, MACs, ports and protocol
labels. Additional features in the 655-column cleaned model representation are
derived measurements, list membership encodings and quality information; they
are not 655 independent Npcap measurements.

Primary DataSense sources:

- [Official UNB dataset description](https://www.unb.ca/cic/datasets/iiot-dataset-2025.html).
- [Publisher paper](https://doi.org/10.3390/electronics14204095), section 3.4 and
  Table 5, pages 22–23. Its definitions distinguish IP header length, IP packet
  length, frame size, TCP flags, timing, address diversity and directional counts.
- Local archived publisher PDF:
  `dataset/experiments/xgboost_data_audit_20261012_v1/deep_clean_v2/research/datasense_original_paper.pdf`.
  SHA-256: `02a111377506904bd9b8cd2c7d9df5ecc284f06ec150c8ec66e9212dc05149ca`.
  Extracted text is `datasense_original_paper.txt` beside it.

Local evidence:

- `dataset/prepared/datasense_full_train/manifest.json` supplies the ordered
  numeric schema. Only its training arrays and training IDs were inspected.
- Original CSV samples were whitelisted against `row_ids_train.npy` before
  their fields were analyzed. Source row IDs below are zero-based, excluding
  the CSV header.
- `docs/DATASET_VERIFICATION_20261014.md` records the copied destination-MAC
  count defect and count/list inconsistencies. `docs/DATA_CLEANING_20261013.md`
  records the prior bounded primary-source search and aggregation uncertainty.
- `experiment/verify_clean_datasense.py::recompute_derived` and
  `experiment/prepare_datasense_quality.py` contain existing documented ratios.

No original publisher extractor was located in the examined primary paper,
official dataset page, or bounded source-code search. Third-party IDS repositories
and reported accuracies do not establish its extraction semantics. Publisher
equivalence remains **unverified for the entire proposed profile**, including
the original seven-count profile. Paired source PCAP/window reconstruction is
needed to establish it.

## Eligibility decision

The broad default candidate has **54 raw numeric columns**. Four additional
payload summaries remain available only as explicitly provisional extensions.
Six raw fields are excluded. Fourteen compatible derivatives produce a **68-field
default candidate before training-only redundancy/usefulness selection**.

"Core" below means eligible for that broad candidate under its stated capture
contract. It does not mean packet-level equivalence to the publisher has passed.
Each summary uses valid observations of its own header field; packets where the
field is absent must not silently contribute a measured zero.

| Numeric CSV field | Default | Proposed capture contract / reason |
|---|---|---|
| `network_fragmentation-score` | Exclude | Publisher does not specify the score formula. Do not invent one. |
| `network_fragmented-packets` | Core | Count IPv4 packets with MF set or nonzero fragment offset; record this chosen definition. |
| `network_header-length_avg` | Core | Mean IPv4 IHL in bytes; exclude non-IPv4 packets. |
| `network_header-length_max` | Core | Maximum observed IPv4 IHL in bytes. |
| `network_header-length_min` | Core | Minimum observed IPv4 IHL in bytes. |
| `network_header-length_std_deviation` | Core | Population deviation of observed IPv4 IHL values. |
| `network_interval-packets` | Exclude | Formula and units are unresolved; not safely interchangeable with inverse packet rate. |
| `network_ip-flags_avg` | Core | Mean numeric three-bit IPv4 flags mask, excluding offset bits. |
| `network_ip-flags_max` | Core | Maximum observed IPv4 flags mask. |
| `network_ip-flags_min` | Core | Minimum observed IPv4 flags mask. |
| `network_ip-flags_std_deviation` | Core | Population deviation of observed IPv4 flags masks. |
| `network_ip-length_avg` | Core | Mean IPv4 total-length field, excluding Ethernet headers/padding. |
| `network_ip-length_max` | Core | Maximum observed IPv4 total length. |
| `network_ip-length_min` | Core | Minimum observed IPv4 total length. |
| `network_ip-length_std_deviation` | Core | Population deviation of observed IPv4 total lengths. |
| `network_ips_all_count` | Core | Distinct union of admitted source/destination IPv4 header addresses; cardinality-estimator equivalence is unresolved. |
| `network_ips_dst_count` | Core | Distinct destination IPv4 header addresses across admitted packets. |
| `network_ips_src_count` | Core | Distinct source IPv4 header addresses across admitted packets. |
| `network_macs_all_count` | Core | Distinct union of admitted Ethernet source/destination addresses. |
| `network_macs_dst_count` | Exclude | Existing audit finds it exactly copies source-MAC count throughout the prepared training data. |
| `network_macs_src_count` | Core | Distinct Ethernet source addresses across admitted packets. |
| `network_mss_avg` | Core | Mean actually observed advertised TCP MSS options; absence is not an MSS measurement. |
| `network_mss_max` | Core | Maximum observed TCP MSS option value. |
| `network_mss_min` | Core | Minimum observed TCP MSS option value. |
| `network_mss_std_deviation` | Core | Population deviation of observed TCP MSS values. |
| `network_packet-size_avg` | Core | Mean original frame length, not retained/captured prefix length; FCS/padding contract must be explicit. |
| `network_packet-size_max` | Core | Maximum original frame length. |
| `network_packet-size_min` | Core | Minimum original frame length. |
| `network_packet-size_std_deviation` | Core | Population deviation of original frame lengths. |
| `network_packets_all_count` | Core | All admitted packets in the defined device/window scope. |
| `network_packets_dst_count` | Core | Device-relative inbound packet count; local identity and self-traffic policy must be explicit. |
| `network_packets_src_count` | Core | Device-relative outbound packet count; local identity and self-traffic policy must be explicit. |
| `network_payload-length_avg` | Hold | Source protocol/denominator semantics are unresolved; no default IP-payload substitution. |
| `network_payload-length_max` | Hold | Same payload-definition uncertainty. |
| `network_payload-length_min` | Hold | Same payload-definition uncertainty. |
| `network_payload-length_std_deviation` | Hold | Same payload-definition uncertainty, plus valid-observation support. |
| `network_ports_all_count` | Core | Distinct union of observed valid TCP/UDP source/destination port values; reported-cardinality equivalence is unresolved. |
| `network_ports_dst_count` | Core | Distinct destination transport-header ports across admitted packets. |
| `network_ports_src_count` | Core | Distinct source transport-header ports across admitted packets. |
| `network_protocols_all_count` | Exclude | Raw lists include application/dissector labels; header protocol counts cannot reproduce them. |
| `network_protocols_dst_count` | Exclude | Same decoder/direction uncertainty. |
| `network_protocols_src_count` | Exclude | Same decoder/direction uncertainty. |
| `network_tcp-flags-ack_count` | Core | Count observed ACK bits, including SYN+ACK. |
| `network_tcp-flags-fin_count` | Core | Count observed FIN bits. |
| `network_tcp-flags-psh_count` | Core | Count observed PSH bits. |
| `network_tcp-flags-rst_count` | Core | Count observed RST bits. |
| `network_tcp-flags-syn_count` | Core | Count observed SYN bits, including SYN+ACK. |
| `network_tcp-flags-urg_count` | Core | Count observed URG bits. |
| `network_tcp-flags_avg` | Core | Mean TCP header flags over valid TCP observations, not all frames. |
| `network_tcp-flags_max` | Core | Maximum observed TCP flags value. |
| `network_tcp-flags_min` | Core | Minimum observed TCP flags value. |
| `network_tcp-flags_std_deviation` | Core | Population deviation of observed TCP flags values. |
| `network_time-delta_avg` | Core | Mean signed capture-order packet timestamp difference in seconds; boundary/precision equivalence remains unresolved. |
| `network_time-delta_max` | Core | Maximum signed observed inter-packet difference. |
| `network_time-delta_min` | Core | Minimum signed observed inter-packet difference. |
| `network_time-delta_std_deviation` | Core | Population deviation of observed signed differences. |
| `network_ttl_avg` | Core | Mean IPv4 TTL; IPv6 hop limit is a separate measurement. |
| `network_ttl_max` | Core | Maximum observed IPv4 TTL. |
| `network_ttl_min` | Core | Minimum observed IPv4 TTL. |
| `network_ttl_std_deviation` | Core | Population deviation of observed IPv4 TTL values. |
| `network_window-size_avg` | Core | Mean raw unscaled 16-bit TCP receive-window values, valid TCP observations only. |
| `network_window-size_max` | Core | Maximum observed raw TCP receive-window value. |
| `network_window-size_min` | Core | Minimum observed raw TCP receive-window value; a measured zero is valid. |
| `network_window-size_std_deviation` | Core | Population deviation of observed raw TCP receive-window values. |

The native extractor may measure additional fields and the four held payload
summaries for inspection. The default selected-model profile must not silently
admit held/excluded fields, endpoint identities, guessed application protocols,
or flow statistics that cannot be reconstructed from the existing CSV.

## Statistical and protocol evidence

### TCP-only support and population deviation

Training source row 0 has 161,243 total packets and reported TCP flag mean
6.848718648440795. Its six flag counts imply a numeric flag sum of 1,104,260,
which divided by the mean gives exactly **161,236 TCP observations**. Thus the
seven remaining frames do not contribute zero-valued TCP observations in this
mixed TCP/ARP example. Training row 22 similarly implies 174,312 TCP observations
against 174,314 total packets.

For row 0, the compatible reconstructed flag frequencies are:

| Flags value | Count |
|---:|---:|
| 2 | 75,150 |
| 4 | 42,542 |
| 18 | 43,544 |

These produce mean 6.848718648440795 and population standard deviation
6.832396829752492, matching the reported 6.832396829752491 to floating-point
rounding. Sample standard deviation is 6.832418017417192 and does not match.
This supplies direct population-deviation consistency for one real source row;
it is not a proof of every aggregation or the original exporter implementation.
The paper does not specify ddof, missing-field denominators or first-delta policy.

### MSS, receive window, payload and absent fields

The original training array's largest reported TCP window is **65,535**. That is
consistent with the raw 16-bit field rather than a scaled calculated window.
The selected capture contract therefore uses raw values. A measured zero window
is valid and must remain zero. See the distinct raw/calculated fields in
[Wireshark's TCP reference](https://www.wireshark.org/docs/dfref/t/tcp.html).

The largest reported training MSS is 1,460; 395,910 of 480,405 training rows have
`network_mss_max == 0`. Row 15 is a TCP-only example with MSS max zero and nonzero
receive-window/flag statistics. An absent MSS option cannot be replaced with a
guessed normal MSS. The paper does not say whether source zero means no observed
option, actual zero or an aggregation default. Preserve raw source values and
state the missing-option conversion policy explicitly before model fitting.

Payload definition remains held because direct source examples rule out a simple
universal interpretation. Row 0 has IPv4 header mean 20, IPv4 packet mean about
41.08, frame mean 60 and payload mean zero. TCP-only row 15 has IPv4 length 40,
header 20 and payload zero. In contrast, row 66 has protocol labels `data, icmp`,
IPv4 length about 36.005, header 20 and payload mean 16. These observations do not
identify whether the exporter uses a protocol-specific payload, dissector data
length, reassembly result or another expression. They do show that substituting
`ip_length - ip_header_length` everywhere is unjustified.

No valid TCP observations, no MSS options and no IPv4 packets are different
missing-support states. They cannot all be encoded as measured zero summaries.
Captured TCP zero flags/window values, by contrast, are genuine possible values.
Noninitial IP fragments cannot be interpreted as fresh TCP/UDP headers.

### Direction, endpoint sets and packet admission

The paper describes packet source/destination counts as outbound/inbound relative
to the device. Address and port source/destination fields describe header
endpoints. They are different dimensions: an incoming packet has a remote source
address and source port. Do not treat all `*_src` values as outbound-only values.

The original device/window admission rule is unavailable. A desktop-only capture
filter, entire-adapter capture, bridged device capture and publisher device view
can yield different values even if every packet parser is correct. The window
duration, boundary inclusion rule and capture loss need separate evidence.

Raw IP/MAC identities can encode a testbed or scenario and should remain evidence,
not default model predictors. Unique-address counts retain diversity information
without introducing specific endpoint identities. Capped lists and approximate
reported cardinalities must not be silently replaced with exact live counts and
called identical. [Elasticsearch documents approximate cardinality](https://www.elastic.co/docs/reference/aggregations/search-aggregations-metrics-cardinality-aggregation);
that is a plausible source of differences, not identification of the publisher's
exporter. The observed destination-list cardinality in the cleaned 655 schema is
not a verified repair of the true destination-MAC cardinality.

### IPv4, IPv6, protocol labels and time precision

The default compatible header/length/flags/TTL/IP-set contract uses IPv4.
IPv6 hop limit and extension-header lengths should be measured under separate
names. Twenty early training rows with reported `dhcpv6` labels were inspected;
their IP sets contained IPv4 addresses only. Rows 267 and 268 explicitly also
have `icmpv6` labels but still contain only IPv4 entries in the reported IP sets.
This supports the IPv4-only choice; dissector labels alone do not prove genuine
IPv6 traffic, and a sample cannot prove every exported set is IPv4-only.

Reported protocol sets contain labels such as `mqtt`, `json`, `tls`, `data`,
`dhcpv6` and decoder-specific names. A port-number heuristic is not a verified
application-protocol observation. Pure Ethernet/IP/TCP/UDP parsing therefore
cannot reproduce the three protocol cardinalities or their membership encodings.

Original training delta minima include approximately -1 microsecond (1,130
negative-minimum rows). Row 0 has a reported minimum difference of 26 nanoseconds.
Keep signed capture-order differences; sorting, absolute values or zero-clamping
would change the measurement. Record capture timestamp precision explicitly.
[Npcap precision documentation](https://npcap.com/guide/wpcap/pcap_set_tstamp_precision.html)
describes a microsecond default and device-dependent nanosecond support; precision
does not itself prove timestamp accuracy. The first packet's prior-frame context
and whether the source delta was computed before device filtering are unresolved.

## Existing derived candidates

The default 68-column candidate adds fourteen existing measured derivatives to
the 54 raw candidates:

- Three packet rates: all, outbound and inbound counts divided by window duration.
- Six TCP-flag fractions: each individual flag count divided by all packet count.
- Two direction fractions: outbound/all and inbound/all packets.
- Packet-size coefficient of variation: packet-size standard deviation / mean.
- Packets per reported distinct IP and packets per reported distinct port.

The fifteenth existing network ratio, `network_payload_fraction`, depends on the
held payload mean and remains held too. It is a ratio of means, not a measured
wire-byte fraction. Use the actual window duration for rates; a constant ten-second
duration is metadata/normalization input rather than an informative predictor.
Undefined ratios require an explicit missing value. At fixed duration, rates
and counts can be redundant and training-only selection should account for that.

Keep quality and support metadata (valid observation counts, timestamp precision,
missing options, truncation, dropped packets and cardinality overflow) available
for admission and review. Such metadata is not automatically a model predictor.
Select useful columns using fitting data only; do not use final validation scores
or traffic predictions to decide measurement definitions or labels.

## Remaining proof

Required before claiming publisher-equivalent live deployment: aligned original
source PCAPs or per-packet exporter output, known device identity and admission
rule, exact window boundaries, a reference extractor, and independently labelled
capture evaluation. Synthetic header tests and a replay from another dataset can
prove software arithmetic while leaving these research questions open.
