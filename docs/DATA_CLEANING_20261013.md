# DataSense copy: evidence-based cleaning and redundancy study

Date: 13 October 2026. Scope: the isolated copy under
`dataset/experiments/xgboost_data_audit_20261012_v1/deep_clean_v2/`.

## User objective and boundaries

Preserve the maximum recoverable data and useful features, explain zeros and
redundancy, reduce noise with justified transformations, and measure model
performance after training. No method can promise every metric will improve.
The original copy, prior experiments, active application and upstream code are
preserved. Existing train/validation membership and original labels are fixed.
The held-out test remains sealed. Source traversal locates the selected IDs;
unselected records are not included in profiling, vocabulary fitting or training.

Two requested research agents initially stopped because of a service usage
limit, then resumed. The searcher completed a bounded primary-source review;
the justifier reviewed preparation, training, calibration and saved verification
artifacts. Neither independently reran the full CSV scan. The primary executed
the full permitted audit, transformations and model comparisons.

## Primary-source research

The [official UNB dataset page](https://www.unb.ca/cic/datasets/iiot-dataset-2025.html)
identifies DataSense as CIC IIoT Dataset 2025. This establishes the dataset name;
it does not authenticate the bytes of the user's combined CSV.

The [original paper](https://doi.org/10.3390/electronics14204095), section 3.4,
describes synchronized TShark network capture and Filebeat MQTT logs stored in
Elasticsearch. Its Table 5, pages 22–23, describes 84 measurement fields spanning
logs, packet counts, address/port/protocol sets, flags, timing and size summaries.
The paper's benign evaluation uses ten-second nonoverlapping slices. The local
export contains multiple durations and overlapping windows; its evaluation
protocol must not be equated with the paper's. The publisher PDF is archived
locally with extracted text in the experiment's `research/` directory.

[Elasticsearch cardinality](https://www.elastic.co/docs/reference/aggregations/search-aggregations-metrics-cardinality-aggregation)
is approximate; [terms aggregation](https://www.elastic.co/docs/reference/aggregations/search-aggregations-bucket-terms-aggregation)
returns a bounded list. These are plausible explanations for count/list
disagreements, not verified identification of the original exporter. No public
extractor was located in the examined official paper, download page and source
searches. The download page requires a form; no personal data was submitted.

[TCP RFC 9293](https://www.rfc-editor.org/rfc/rfc9293.html#section-3.8.6.1)
explicitly permits a zero receive window. Zero flags, payloads and variability
also need not indicate corrupt data. [Wireshark's time API](https://www.wireshark.org/docs/wsar_html/nstime_8h.html)
allows signed differences; a negative delta alone does not reveal its cause.
Capture order cannot be reconstructed from window aggregates.

## Decisions frozen before model evaluation

1. Preserve every selected raw record and original label. No prediction-based
   relabelling, synthetic observations, quantile clipping or deletion of rare
   attacks. Device identities, IP/MAC identities, absolute dates and labels are
   provenance, not predictor inputs.
2. Record exact zero locations, per-field distributions, list cardinalities,
   source record IDs, count/list discrepancies and summary constraints. All
   comparisons use the full selected partitions, not a sampled glance.
3. Preserve signed sensor minima and packet deltas. Preserve high traffic
   counts, port zero, zero TCP windows and rare protocols. A small number or an
   outlier is not sufficient evidence of an error.
4. When an event count is zero, a zero summary statistic has no supporting
   event. In a separate model representation mark that statistic unavailable
   and record its location; keep its original value in the raw evidence.
   This is an explicit interpretation, not a claim that a sensor measured NaN.
5. When a list contains at least 1,000 distinct entries and its reported count
   exceeds that number, treat unlisted membership as unknown. Preserve observed
   membership, original counts, listed counts, differences and a censoring flag.
   Do not replace an aggregate count with the length of a potentially capped list.
6. Add previously omitted observed log types and protocol tokens, fitted on
   training only. Preserve the existing port vocabulary. Add documented ratios
   of measured quantities; use missing values for undefined denominators.
7. Remove only exact training constants or identical columns from the compact
   model view. Keep every full column plus its mapping. High correlation alone
   does not justify deleting a useful feature. Validation does not select columns.
8. Compress repeated exact `(model inputs, original label)` pairs with a
   multiplicity and reversible source-row mapping. Different labels remain
   separate. Equal measurements at different timestamps are not mislabeled as
   duplicate physical records. Sample weights preserve all represented counts.
9. Keep signal-free windows in the master. Compare training on all retained
   information with an explicitly labeled observed-signal view. Validation is
   never filtered. No-signal filtering changes the training distribution and
   is an ablation, not a universal declaration that the records are invalid.

## Execution stages and acceptance criteria

- `01_raw_audit`: validate copy SHA256; traverse raw CSV; audit the train/validation
  whitelist; preserve original selected cells and issue ledger.
- `02_profile`: quantify every raw column, record zero bitmaps, check numeric
  constraints and exact record duplication, retain no-signal masks.
- `03_candidate`: build explicit missingness/censoring, measured features,
  complete and compact arrays, reversible grouping and readable CSV exports.
- `runs`: fixed-configuration CPU training comparisons with unchanged 30,000-row
  validation, saved probabilities, checkpoints and CPU reload verification.
- `verification`: independent metric recomputation, source/input integrity,
  reconstruction checks, default and all four demos, full evaluation suite.

The previously completed 79-round seed-42 CPU XGBoost arms are preserved controls:
original accuracy 79.2233%, cleaned 80.0000%, cleaned/enriched 80.6167%.
New data rules are frozen before their model metrics are observed. Fitting both
retention alternatives is exploratory validation, not a held-out test result.
Any additional model or tuning comparison must be labeled separately from a
matched data-only comparison. No automatic promotion or installation occurs.

## Final representation and review qualifications

The permitted audit covers 480,405 training and 30,000 validation observations,
all 94 original fields. The remaining source records and locked test are outside
this audit. The source copy has SHA256
`4db33c7b73a6e715d1b4484f4d4c77747a5e9355abf70c394c043cd27579a9f9`.

Stage 03 adds measurement quality and missingness to create 527 full features
and a 431-column compact view. Stage 04 retains those and adds 128 directional
port indicators: 64 new ports per direction ranked by training-only mutual
information of occurrence in the exported lists, with at least ten occurrences.
This ranking describes exported membership; a capped list does not establish
true absence. Its frozen selection policy and all 131,072 port scores are saved.
The final master has 655 features; the compact training view has 559.

The 96 removed compact columns comprise 77 exact training duplicates and 19
training constants. This is training-derived selection, not universal lossless
column compression: `network_protocols_dst__present_pktc.mtafqdn` differs from its
retained all-direction counterpart on five validation rows. Its independent
full column remains in the master. Training input/label grouping is exactly
reversible: 383,630 groups and multiplicities represent all 480,405 observations.
There are no exact duplicate raw CSV observations including their provenance.

The 1,000-entry list rule is a conservative suspected-censoring hypothesis; the
examined author material does not publish the extractor or that threshold.
Elasticsearch's default terms size is ten, not evidence of a 1,000-entry exporter
setting. Reported counts, raw lists and their differences remain available.
`network_payload_fraction` is a ratio of reported payload/packet-size means,
not a verified fraction of all transmitted wire bytes. Zero-event masking
cannot identify feature-specific unobserved values in a nonempty packet window.

XGBoost multiplicities preserve represented counts and additive weighted loss.
They change subsampling units when `subsample=0.8`; histogram sketching and
feature sampling can also change fitting. Comparisons measure the combined
representation and compression experiment, not cleaning in isolation. RF uses
expanded original observations and its unchanged 100-tree settings. Neural
models receive all original training rows and train-only fitted preprocessing.

## Four-model fitting and calibration assessment

The user's subsequent instruction extends fitting to fixed conventional RF,
XGBoost, genuine efficient-KAN, and HEM-derived closed-set voting. Neural models
use signed log compression, train-fitted median imputation and MinMax scaling;
HEM additionally uses 64-component train-fitted randomized PCA and all authentic training
rows as exact nearest-neighbor references. These preprocessing/capacity changes
are separate configured model experiments, not unchanged HEM reproduction.
Neural fits use fixed final epochs, with no validation-based stopping. GPU
research fits are explicitly requested by their command; defaults and deployed
inference remain CPU, offline and credential-free.

Calibration is a separate exploratory assessment in this isolated study. The
existing validation is divided chronologically within each class: 14,545 early
rows fit calibrators, 14,754 later rows assess them, and 701 touching-window or
repeated-input rows are excluded from this comparison only. All 30,000 remain
in the master and in the separately reported full-validation model metrics.
There are 65 calibration-supported classes and 83 assessment-supported classes;
no per-class calibration claim applies to the other eighteen. This split is
within-class later validation, not an independent unseen capture. Prior model
and representation experiments already used the original validation, so results
remain exploratory. The prior `csv_research_v3` training-only OOF study is unchanged.

RF, XGBoost and KAN use scalar temperature calibration. The saved scores and
their class decisions are checked after reloading. Calibration changes score
reliability and does not improve accuracy by changing the predicted class.
HEM hard voting has no multiclass probability API: separately calibrated neural
components plus a regularized correctness estimator provide an explicitly
labeled probability that its unchanged vote is correct. Component probabilities
must not be presented as calibrated HEM multiclass probabilities.

Method references: [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html),
[temperature-scaling paper](https://arxiv.org/abs/1706.04599),
[XGBoost subsampling and feature-sampling parameters](https://xgboost.readthedocs.io/en/stable/parameter.html).
Calibrator fitting and assessment data are disjoint. No locked-test access,
prediction-based relabeling or automatic app-model replacement occurs.

An initial port-refinement build used the wrong parser kind and lost integer
membership matches. It was caught before fitting, preserved under the clearly
marked invalid directory, corrected, and rebuilt with selected-port positive
count reconstruction checks. One new-data RF attempt was interrupted when
concurrent fits approached Windows commit exhaustion and preserved as an
interrupted attempt; the final RF fit is run separately. These attempts are not
completed model results.

The first HEM setup using a full SVD was also interrupted at low available
commit. The final HEM implementation uses exact train-only imputation/scaling
statistics processed in chunks, followed by randomized PCA fitted on all training
rows. A small probe caught that sklearn's `copy=False` PCA also squares its private
input cache during variance calculation. Training projections now come from fresh
scaled chunks through the saved PCA, with direct training/inference equality
checks; prepared data are read-only. Sixteen focused quality tests and the full
508-test final evaluation suite pass. Both final CSV exports were reloaded in chunks:
every feature cell, original label and source ID matches the arrays exactly.

## Executed outcome

Eight completed fits are preserved; four families have completed checkpoints and
calibration assessments. On the original 30,000 validation observations: RF
79.1467% accuracy / 73.4811% macro F1; XGBoost 80.5267% / 77.6984%; KAN
78.5100% / 72.9696%; HEM-derived 75.7267% / 62.8095%. All increase over their
listed original seed-42 accuracy references. The earlier enriched XGBoost still
has slightly higher accuracy (80.6167%) and lower macro F1 (76.8126%) than the new
selected XGBoost recipe. The new HEM false-alarm rate and RF attack-miss rate
regress. Complete comparisons, calibration scores and figures are in the study's
`RESULTS.md` and `report/` directory.

Multiclass temperature calibration worsens ECE for RF and KAN. A second fixed,
regularized Platt correctness recipe was fitted only on the pinned early
calibration portion, with no later-assessment tuning. Decision-confidence ECE
changes are XGBoost 0.06986 -> 0.01321, KAN 0.03121 -> 0.03046, RF
0.06479 -> 0.02898, HEM 0.11546 -> 0.06619. KAN correctness Brier rises slightly;
the other three improve. Correctness confidence is explicitly distinct from a
multiclass probability distribution. All class predictions remain unchanged.

The deeper CUDA XGBoost fit completed all 160 rounds and saved its checkpoint.
The initial CPU/GPU score check failed a 1e-7 threshold by one float32 epsilon.
Recovery loaded separate GPU/CPU boosters from that untouched checkpoint,
verified every one of the 30,000 class decisions identical, measured maximum
probability difference 1.1920928955078125e-7, and passed a 2e-7 absolute score
tolerance. No model was refitted or checkpoint rewritten during recovery.

Reproduction commands from project root, in the project Python environment:

```powershell
python -B -m experiment.audit_datasense_raw scan
python -B -m experiment.audit_datasense_raw profile
python -B -m experiment.prepare_datasense_quality build
python -B -m experiment.refine_datasense_ports
python -B -m experiment.prepare_datasense_quality export --candidate 04_port_refinement
python -B -m experiment.train_datasense_quality --candidate 04_port_refinement --arm all_records
python -B -m experiment.train_datasense_quality --candidate 04_port_refinement --arm all_records --device cuda --depth 8 --rounds 160
python -B -m experiment.train_datasense_quality --family random_forest --candidate 04_port_refinement --arm all_records
python -B -m experiment.train_datasense_neural --family kan --device cuda
python -B -m experiment.train_datasense_neural --family hem --device cuda
python -B -m experiment.calibrate_datasense_models
python -B -m experiment.calibrate_datasense_confidence
python -B -m experiment.verify_datasense_exports
python -B -m experiment.verify_datasense_quality --candidate 04_port_refinement
python -B -m experiment.report_datasense_quality
```

Preparation/fitting refuse overwriting completed stage/run directories. These
commands describe a fresh study, not a request to destroy or rerun existing
results. The optional `--directory` arguments route stages to another isolated
study; source and existing prepared train/validation inputs remain frozen.
