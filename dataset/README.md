# Dataset boundary

`demo/sample_traffic.csv` is **synthetic demonstration data**. It exists only to exercise the software pipeline. It is not captured IoT traffic, a public research dataset, or evidence of intrusion-detection accuracy. Do not report performance on it as research results.

## Schema

| Column | Meaning |
| --- | --- |
| `sample_id` | Unique synthetic record identifier; demonstration IDs select CLI examples |
| `split` | `train` or `demo`; only `train` rows are fitted |
| `label` | BENIGN, DDoS, PORT_SCAN, or the held-out UNKNOWN example |
| `packet_rate` | Illustrative packets per second |
| `byte_rate` | Illustrative bytes per second |
| `syn_ratio` | Illustrative fraction of packets carrying SYN, between 0 and 1 |
| `connection_rate` | Illustrative connections per second |
| `unique_destinations` | Illustrative destination count in an aggregation window |
| `avg_packet_size` | Illustrative average packet size in bytes |

These aggregate features are invented for an understandable demo and are not a packet-extraction specification. The generator derives `byte_rate` from `packet_rate * avg_packet_size`; other features come from illustrative class profiles. No real IP addresses, device identities, or packet payloads are included.

## Construction and separation

The supplied CSV was generated with random seed 42. It contains 60 training records for each of BENIGN, DDoS, and PORT_SCAN (180 total), plus four separate demonstration records selected by `benign`, `ddos`, `scan`, and `unknown`. Regenerate the same demonstration fixture from the project root with:

```powershell
python -m experiment.generate_data
```

This overwrites only the synthetic demo CSV. The UNKNOWN example includes an unusually large average packet size (8,500 bytes), creating a clear distribution mismatch. Such a value alone does not establish an attack; unfamiliar legitimate traffic or measurement differences are possible explanations.

The three training classes form intentionally distinct clusters. The UNKNOWN example is deliberately far outside those clusters. UNKNOWN is not a classifier training label. Metadata (`sample_id`, `split`, and `label`) never enters the feature matrix. Both scaling and distance-threshold estimation use only training records; demonstration rows are not used to fit the detector. The demo split is a set of presentation examples, not an independently collected research test set.

## Future real datasets

Real UQ-IoT PCAPs are now present in the user-added dataset directory under
`external/HEM/`. They remain in place. `python -m experiment.hem_features` reads
a bounded prefix and saves 100 original HEM/Kitsune features per accepted packet
under `dataset/processed/`, plus provenance JSON. These are real capture-derived
features, distinct from `dataset/demo/`. They are not labeled training splits,
fitted PCA inputs or detection results. A capture filename is not a packet label.

Scapy decoding is our wrapper; feature computation executes original HEM source.
Timestamp regressions and unsupported packets are explicitly counted and skipped;
each capture starts fresh feature history. Raw packet data and generated CSVs stay
ignored by Git. The dataset's supplied `FurtherInformation.txt` and `bag-info.txt`
identify its provenance and license location.

CICIoT2023 and datasets supported by the HEM implementation are candidates, subject to their access conditions and licenses. [HEM integration notes](../docs/HEM_INTEGRATION.md) describe the inspected source requirements.

A real integration needs a documented feature extractor, label mapping, units and aggregation windows, missing-value policy, class balance, device/time-aware splits, and explicitly held-out attack families for unknown detection. Fit preprocessing and models on training data, select thresholds using validation data, and evaluate once on a separate test set. HEM's research features cannot be replaced with these six synthetic columns by renaming them.

Keep large raw captures, prepared datasets, and generated models outside normal source control. Version compact metadata and reproducible preparation procedures instead.
