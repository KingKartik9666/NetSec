# Improving live packet screening: evidence and bounded experiments

Date: 27 October 2026. This note separates current local evidence, proposed
experiments, and methods requiring new packet-level ground truth. Research used
primary papers and official documentation. Reading sources used the network;
the proposed execution remains offline, CPU-only and credential-free. No packet
capture, attack execution, held-out test access, model promotion or installed-app
replacement was performed by this research task.

## Decision

Keep the broad measured-header representation and assess its stability before
spending another large search budget. The first comparison should freeze the
60-field profile, fit a small fixed XGBoost recipe on the existing fit role, and
evaluate seven-count, broad, and no-MSS variants on the same tune rows. Add
binary/family summaries from the same probability outputs and stress the
source-zero/live-missing MSS mismatch. These experiments can answer useful
questions with existing training data. They cannot establish Windows live
accuracy or publisher-equivalent feature extraction.

The largest unresolved risks are measurement mismatch, source/device/time shift,
and insufficient information for some of the 84 labels. Model capacity alone
does not resolve these. This prioritization is our inference from the local
audit and the operational evaluation concerns raised by Sommer and Paxson,
not a promised accuracy improvement. [Outside the Closed World, IEEE S&P 2010](https://gangw.cs.illinois.edu/class/cs562/papers/Closed_World-IDS-sp10.pdf)

## Verified local starting point

Read `docs/PROJECT_CONTEXT.md` completely. Current source and artifacts inspected
for this note include `framework/training/live_study/data.py`, the native
`live/packet.rs` and `live/features.rs` sources, the companion
`docs/NPCAP_DATASET_FEATURE_AUDIT.md`, and
`artifacts/training/live_features_20261027_v1/{RESULTS.md,selection.json,selected_profile.json}`.
Historical context is not a substitute for these current records.
The later cross-chat review also read
`artifacts/feature_audit_20261027/REPORT.md` completely and inspected its
655-feature review schema and the cleaned membership derivation source.

- The audit identifies 64 raw numeric network fields. Its broad default proposes
  54 raw fields plus 14 derivatives. Undefined fragmentation score, interval-packets,
  three application-protocol counts and the defective destination-MAC count stay
  excluded. Four payload summaries and their ratio remain optional, provisional
  extensions; the default is 68 candidates, not the extended 73.
- Screening retained 63 nonredundant candidates and selected 60 using three
  class-conditional, chronological, purged folds inside the fit role. There was
  no arbitrary target feature count.
- The current fit/tune/blend/calibration roles have 11,716 / 954 / 945 / 1,637
  observations. Sparse classes remain in fit; evaluation support is incomplete.
  The original dataset test remains locked.
- The broad selector did not use original validation or held-out test arrays.
  The tune role was not used for this broad feature selection, but it was reused
  by the prior seven-count training study. A new assessment on it is exploratory,
  not an untouched final evaluation.
- No aligned publisher PCAP/extractor/window reference has been established.
  A matching CSV field name and a correctly implemented parser do not prove
  equivalence of the measurement or aggregation population.

The current screen reports the following **feature-selection estimates**:

| Within-fit fold | Assessment rows | Supported labels | Seven counts accuracy | Selected 60 accuracy |
|---|---:|---:|---:|---:|
| 1 | 536 | 43 | 74.63% | 90.67% |
| 2 | 340 | 43 | 84.71% | 93.82% |
| 3 | 864 | 52 | 63.54% | 65.16% |
| Unweighted fold mean | — | — | 74.2915% | 83.2191% |

Mean macro F1 over the fixed 84-label schema is 38.2935% versus 44.1661%.
Unsupported labels and fold composition affect that statistic. Preserve it for
comparability and additionally report supported-label macro F1 with the support
counts; never silently replace one definition with the other. These folds were
used to select features, so their improvements are selection-biased estimates.
The third-fold result warrants investigation rather than being hidden by the
mean. Current records do not establish its cause.

## Immediate experiment: fixed profiles, reused training roles

Freeze the schema order, input hashes, role IDs, seed list, family map and recipe
before evaluation. Preserve the original screening artifacts. Use CPU XGBoost,
100 rounds and seeds 42, 43 and 44, with all other settings fixed. No new Optuna
search and no early stopping against assessment rows are needed for this first
diagnostic. The parent implementation owns execution and result artifacts; the
following table is a protocol, not a claim that these fits have completed.

| Variant | Fit input | Assessment input | Question |
|---|---|---|---|
| Seven counts | Existing seven measured counts | Same seven | Matched baseline under the new fixed recipe |
| Selected 60 | Frozen profile, raw source values | Same profile | Does broad improvement survive outside its screening folds? |
| No MSS | Selected profile minus all four MSS summaries | Same reduced profile | How much gain depends on a known semantic uncertainty? |
| MSS missingness stress | Reuse each selected-60 fit | Convert only all-zero MSS summary blocks to NaN | How sensitive is the fitted source model to a plausible live support pattern? |

There are nine model fits for the first three rows; the stress row only requires
inference. Restrict all fitting to fit-role rows and assess the unchanged tune
role. Do not recalculate feature selection after seeing tune outcomes. Record
per-seed outputs and paired differences, not only the best seed.

The stress transformation is a hypothesis, not a repair of the dataset. All-zero
MSS summaries could represent no observed option or another exporter default;
the source lacks the option-support count needed to decide. Preserve raw data,
perform the transformation in an explicitly named copy, and count affected rows.
Do not replace isolated zero statistics indiscriminately: zero standard deviation
can be valid when all observed MSS options agree. Native absence remains null.
XGBoost learns missing-value branch directions during fitting; being able to
accept NaN does not mean a model trained on zeros is robust to new missingness.
[XGBoost missing-value documentation](https://xgboost.readthedocs.io/en/stable/faq.html#how-to-deal-with-missing-values)

For every variant, emit multiclass accuracy, both macro-F1 definitions, class
support, confusion counts, attack-score average precision, attack recall/FPR at
a preregistered threshold, multiclass log loss and binary Brier score. Add
decision-confidence reliability bins with bin counts. These are calibration
diagnostics before any new calibrator is fitted. Report MSS-affected and
unaffected subsets separately. Three seeds assess fit variability, not variability
across independent captures or devices.

Interpret results conservatively:

- Broad improvement across seeds supports keeping the broad representation for
  further research; it does not establish a deployed score.
- A large no-MSS or missingness-stress change identifies dependence worth resolving.
  It does not prove which source encoding is physically correct.
- A high binary score alongside weak subtype scores supports coarser reporting.
  It does not establish that a separately trained hierarchical model is better.
- If broad improvement collapses outside screening folds, inspect class support,
  time/device provenance and feature distributions before increasing model size.

General tabular benchmarks justify an inexpensive tree baseline at this data
scale, while not proving that XGBoost is the best model for this particular task.
After the contract is stronger, compare fixed RF, a regularized linear baseline
with fit-only preprocessing, and the existing neural families under comparable
budgets. Do not run expensive nested tuning for every family before resolving
measurement uncertainty. [Grinsztajn et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html)

## Binary, family and subtype outputs

The immediate diagnostic can aggregate a normalized 84-class probability vector:

```text
attack_score = 1 - p(benign)
p(family) = sum of p(subtype) for the fixed subtypes in that family
```

Validate the map against every label. Include benign as its own group, then
bruteforce, ddos, dos, malware, mitm, recon and web from the current label schema.
The official source describes seven attack categories; the local 84 labels
include more detailed variants and must retain their own provenance.
[Official DataSense description](https://www.unb.ca/cic/datasets/iiot-dataset-2025.html)

Summing probabilities is an algebraic aggregation of one model. It is **not** a
trained binary detector or a trained hierarchy. Binary thresholding at 0.5 is
also different from mapping the winning multiclass label to benign/attack:
several attack labels can collectively outweigh benign without any one beating
it. Report the rule used for every FPR/recall value.

If aggregation is useful, a later bounded comparison can train a binary head and
conditional family/subtype heads using the same frozen features and splits.
Compare end-to-end results, because a missed attack at the first gate cannot be
recovered by a later classifier. Permit an attack indication with an unresolved
family, and a family indication with an unresolved subtype. A single-host view
may not reveal distributed coordination, and similar header patterns can occur
across attacks with different application-layer intent. Treat such distinctions
as hypotheses unless the available measurement supports them.

## Make capture semantics part of the model contract

The following requirements are measurement checks, not predictors of an attack:

| Contract field | Required behavior / diagnostic |
|---|---|
| Scope and direction | Record selected adapter, device identity rule, local-address changes, admission filter and self-traffic policy. Source/destination header endpoints are distinct from inbound/outbound relative to a device. |
| Length | Retain capture length and original reported length separately; derive each statistic from its declared field. Never use a truncated prefix as the original frame size. State FCS, padding and link-type assumptions. |
| Timestamp | Use packet capture timestamps for packet timing, not callback arrival times. Record returned precision and source if known, signed out-of-order deltas, and first-packet/window-boundary policy. Multiplying microseconds by 1,000 does not add nanosecond accuracy. |
| Header support | Count valid IPv4, TCP, UDP and MSS observations independently. No IPv4/TCP/MSS support is not a measured zero mean; noninitial fragments do not expose new transport headers. |
| Capture quality | Preserve drop counters, truncation, parse failures, unsupported link types, capped cardinalities, malformed options and unsupported timing. Unavailable quality telemetry stays unknown. |
| Window | Keep exact start/end, elapsed coverage, reset policy and shutdown partial-window status. Do not normalize an incomplete interval as a complete ten-second observation. |

Libpcap explicitly distinguishes bytes available after capture slicing (`caplen`)
from the length before slicing (`len`). That latter value is still a capture-layer
measurement, not proof of the physical wire representation.
[libpcap packet-header source](https://github.com/the-tcpdump-group/libpcap/blob/master/pcap/pcap.h)

Npcap documents that checksum and segmentation/coalescing offloads can make the
captured packet differ from the physical-link packet, including apparently large
packets and invalid checksums. Consequently, an oversized observation or checksum
anomaly alone must not establish maliciousness. Log capture conditions; do not
change the user's adapter configuration as part of this research.
[Npcap Users' Guide](https://npcap.com/guide/npcap-users-guide.html#npcap-issues)

Host timestamps can reflect batching, queuing and unsynchronized ordering.
Returned numerical precision is not timestamp accuracy. Signed deltas and
support metadata are necessary for honest timing features; a future
timestamp-quantization stress must be labelled a simulation if per-packet times
are unavailable. Rounding an aggregate mean cannot reproduce re-extraction of
rounded packet timestamps. [Npcap timestamp documentation](https://npcap.com/guide/wpcap/pcap-tstamp.html)

Do not silently replace device windows with all-adapter windows. A host capture
also cannot observe every peer-to-peer exchange on a switched network merely
because it is connected to that network. Per-device aggregation, gateway
aggregation and per-flow aggregation are separate model contracts. Test them
using corresponding labelled packet windows when available. Packet parsers may
share code while observation schemas and trained models remain explicit.

## Capture quality, uncertainty and abstention

Use separate fields for `measurement_status`, `distribution_support` and
`model_result`. Unsupported link types, schema mismatch or untrustworthy required
measurements should produce an unavailable/insufficient-evidence state. Ordinary
protocol absence such as no MSS option can be allowed only by an explicit,
tested model policy. A capture-quality failure is not an attack label, and
abstention is not a benign decision.

For a later research gate, compute simple fit-reference diagnostics: missingness
pattern, per-feature range violations, robustly scaled nearest-neighbor distance,
and disagreement among already fitted seeds. Use bounded reference samples and
fit-only transforms. Evaluate their ability to identify errors on the existing
assessment role; do not equate an outlier score with a zero-day attack. A normal
Windows application or a changed adapter can be out of distribution too.

Report coverage, error among accepted observations, attack recall over **all**
attack observations, benign abstention rate and attack abstention rate. Otherwise,
a system can appear more accurate simply by declining difficult attacks. The
selective-classification literature supplies the risk-versus-coverage framework;
using that reporting framework does not mean implementing SelectiveNet or
inheriting its guarantees. [El-Yaniv and Wiener, JMLR 2010](https://jmlr.org/papers/v11/el-yaniv10a.html)

Calibration is a separate task. Fit a low-parameter calibrator only after the
representation/model choice is frozen, using a designated role, then assess it
on another available role or a later independent capture. Prior reuse must be
disclosed. Temperature scaling is a useful small-sample candidate, not an assured
fix; it preserves class ordering within an observation but changes probabilities.
[Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)

Calibration learned on the source distribution can fail after deployment shift.
Evaluate Brier/log loss, reliability and threshold behavior again for each new
capture environment. Do not calibrate to the model's own live predictions or
describe unlabelled traffic as confirmed benign. [Ovadia et al., NeurIPS 2019](https://papers.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html)

Conformal prediction is an optional later uncertainty experiment. Standard
finite-sample coverage needs exchangeability; adjacent network windows and
changing devices weaken that premise. Weighted/nonexchangeable methods still
have assumptions and coverage-loss terms. Neither ordinary nor weighted
conformal prediction provides a blanket guarantee for arbitrary unseen attacks.
Measure set size, abstention and empirical coverage by capture/device/class;
do not label prediction-set membership a verified attack finding.
[Barber et al., Conformal Prediction Beyond Exchangeability](https://arxiv.org/abs/2202.13415)

## Feature selection that keeps useful information

Use the 68-field pool as a declared candidate contract. Keep physically distinct
measurements unless data and task evidence justify removal. Exact mathematical
redundancies at a fixed ten-second duration can be documented and excluded from
model input without discarding the measurements from export. A constant in one
capture is not necessarily constant on another device.

Group raw summaries and their derived ratios when permuting, and use a shared
row permutation within a group. Permuting a numerator while leaving its ratio
unchanged can create impossible vectors. Drop-column/group ablations test whether
the model can recover the information from other variables; they answer a
different question from shuffling a fitted model's inputs. Correlated predictors
can mask one another in ordinary single-feature permutation importance.
[scikit-learn correlated-feature example](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html)

Prefer foldwise importance/ablation signs, class-specific effects and worst-fold
behavior to a single global rank. Do not select a fixed top-k or delete a weak
standalone feature that supports a useful interaction. If more screening becomes
necessary, Boruta is an all-relevant alternative using randomized shadow features;
its selection is model/data dependent, not proof of causality or live transportability.
Keep ambiguous features marked tentative until independently assessed.
[Kursa and Rudnicki, JSS 2010](https://www.jstatsoft.org/article/view/v036i11/0)

Resampling can reveal selection instability. Here, preserve capture/device/time
groups instead of randomly mixing highly related rows. Report selection
frequencies as empirical stability. Do not claim formal false-discovery control
from an arbitrary time-block modification of stability selection.
[Meinshausen and Bühlmann, 2010](https://stat.ethz.ch/Manuscripts/buhlmann/stabilityselection-epub.pdf)

The selected-60 experiment should finish before adding another selector. A
large feature-search loop can overfit the same small assessment folds even when
no training row crosses a split boundary.

## Cross-chat dataset audit: narrow follow-up, preserving the frozen profile

The user's latest dataset-feature chat supplies a useful, distinct evidence
source: `artifacts/feature_audit_20261027/REPORT.md`. Its frozen full-schema
XGBoost diagnostics used previously reused validation observations. The
2,048-row screening sample overlaps its 7,225-row follow-up assessment. These
are exploratory perturbation results from a 655-field model, not broad-profile
retraining results or new independent validation.

| Prior audit intervention | Observed change | Appropriate next question |
|---|---|---|
| Shuffle TCP-flags standard deviation | Accuracy +0.443 to +0.692 percentage points; macro F1 +0.246 to +0.337 | Retrain one matched broad-profile variant without this field. |
| Shuffle minimum TTL | Accuracy +0.554 to +0.775 pp; macro F1 -0.422 to -0.333 | Test a separate removal and disclose class-balance tradeoffs. |
| Shuffle all eight IP/TCP flag-mask summaries together | Macro F1 -4.08 to -5.35 pp | Preserve the group until an actual retrained ablation justifies removal. |
| Shuffle 194 directional port-presence indicators together | Macro F1 -18.37 to -18.79 pp; accuracy -2.99 to -3.31 pp | Test whether a compatible port-presence extension helps the live research profile. |

The audit's older matched port-extension retraining comparison also had mixed
results: accuracy -0.13 pp and macro F1 +0.7542 pp. Thus port relevance is a
reason to test the group, not a guarantee of better accuracy. Do not choose ports
from validation importance ranks or discard all sparse indicators: a rare port
may distinguish a rare class. The broad 60-field profile currently has port
cardinalities but no individual-port presence, so this is a real representational
gap worth measuring.

Use a separately versioned, immutable extension with the following contract:

1. Join cleaned source membership to the existing eligible ten-second training
   observations by original row ID, checking label and window alignment. Do not
   join by array position or borrow validation rows. Preserve every original
   role assignment and the locked test boundary.
2. Freeze the port vocabulary and ordered columns using observed fit-role
   memberships only. The existing 194-column training-derived schema is an
   admissible explicitly labelled existing-schema control, but its vocabulary
   is not a newly fit-only selection. If extending vocabulary beyond its slots,
   obtain it from fit source lists without inspecting tune/validation labels.
   If later running inner CV for selection, learn vocabulary within each inner
   training fold. No arbitrary top-k is required.
3. Encode **1** when a valid source/destination port token is observed;
   **0** only when the relevant membership measurement is complete and the token
   is absent; **NaN** when absence cannot be determined because of source
   censorship/incomplete recovery or relevant live capture/parser uncertainty.
   Preserve the cleaned dataset's incomplete-membership policy. Observation of
   one port is positive evidence even when other memberships remain unknown.
4. Map `src` and `dst` to the TCP/UDP header endpoints across admitted packets.
   They do not mean local outbound and inbound packets. Treat the all-port union
   separately. Do not add a TCP/UDP-specific distinction if the matching CSV
   field cannot provide it. Valid zero ports are values, not generic missingness.
5. Keep unseen-token and incompleteness indicators available for diagnostics;
   their being constant in fit means a fitted tree cannot learn their future
   discrimination. Cap native state explicitly and preserve overflow status.
6. Compare `selected60 + compatible port presence` with selected60 using the same
   fit/tune IDs, fixed recipe and seeds. Test the two single-field removals
   separately; do not bundle every hypothesis into one unidentifiable treatment.
   Preserve v1's selected profile and hashes; any accepted extension gets a new
   schema fingerprint and its own results. Native export remains a later step.

Port presence is observed transport information, not verified application
protocol identity. IANA explicitly warns that traffic on a registered port need
not correspond to its assigned service. Source ephemeral ports and exact service
ports can also identify scenario/device configuration, so cross-device/time
assessment is necessary even when within-source macro F1 improves.
[IANA service-name and port registry](https://www.iana.org/assignments/service-names-port-numbers)

These narrow follow-ups add bounded fits to the nine-fit baseline protocol;
they are not a reopening of a large all-family tuning search. Their final
executed counts and outcomes belong in the experiment's results artifact.

## Evaluation must match the deployment question

Existing class-conditional chronology is useful for within-class later-window
assessment, but it is not one global past-to-future boundary. A later sample from
one attack scenario can coexist with an earlier assessment sample from another.
Report that design honestly. Keep all preprocessing, feature selection, weighting
and augmentation inside fitting data; reserve assessment rows for a frozen
comparison. General security-ML pitfalls include leakage and unrealistic
evaluation conditions. [Arp et al., USENIX Security 2022](https://www.usenix.org/conference/usenixsecurity22/presentation/arp)

Once trustworthy provenance exists, assess in stages: later times from the same
devices; held-out devices/captures; then a distinct Windows capture environment.
Keep overlapping windows and all windows from the same event together, purge
shared packet intervals, and embargo the longest feature lookback. When a class
does not exist before the global cut, report it as unseen rather than moving
future examples backward. Spatial/temporal evaluation constraints are motivated
by TESSERACT; its malware-specific performance findings are not network-IDS
results. [Pendlebury et al., USENIX Security 2019](https://www.usenix.org/conference/usenixsecurity19/presentation/pendlebury)

Use metrics that expose the operational tradeoff:

- Report attack average precision and the full precision-recall curve with
  attack prevalence. State whether a number is average precision or trapezoidal
  PR-AUC. ROC-AUC remains supplementary; a large benign population can make false
  alarms operationally dominant despite a visually strong ROC curve.
  [Saito and Rehmsmeier, PLOS ONE 2015](https://pmc.ncbi.nlm.nih.gov/articles/4349800/)
- Report recall at fixed low FPR targets, false alerts per observed device-hour,
  and benign/attack denominators. Select thresholds on a designated tuning role;
  never optimize and claim final threshold performance on the same rows.
- On labelled continuous captures, add event recall, detection delay from event
  start, duplicate alerts per incident and benign episodes triggering any alert.
  Freeze any consecutive-window rule and cooldown before assessment; smoothing
  can lower alerts while increasing delay or hiding short attacks.
- Give class/family and device/capture breakdowns. Bootstrap independent capture
  or event blocks for uncertainty where possible, not adjacent windows as if they
  were independent. Seed spread is not a replacement for sampling uncertainty.

For scale, one device produces 8,640 nonoverlapping ten-second windows per full
day. At a 1% benign-window FPR, the arithmetic expectation is about 86 flagged
windows per benign day; actual alert count depends on coverage and grouping.
Also, with prevalence pi, precision equals
`TPR*pi / (TPR*pi + FPR*(1-pi))`. At pi=0.001, TPR=0.9 and FPR=0.01,
precision is about 8.3%. These are illustrative calculations, not measured
NetSec operating rates.

Zero observed false positives does not prove zero risk. Under independent
Bernoulli sampling, the one-sided 95% upper bound after zero false positives in
n benign samples is `1 - 0.05**(1/n)` (approximately `3/n`). Correlated windows
and reused model-selection data do not satisfy that simple interpretation;
report those limits. [NIST exact binomial confidence limits](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm)

## Next feature generation requires packet-level evidence

After the fixed-profile experiment, prioritize stateful features rather than
inventing more CSV equivalents. Zeek's connection schema demonstrates concrete
examples: originator/responder packets and bytes, connection state/history,
content gaps and partial-connection status. These have explicit direction and
support semantics. [Official Zeek connection schema](https://docs.zeek.org/en/current/scripts/base/protocols/conn/main.zeek.html)

Useful candidates for a separately trained packet/flow profile include handshake
completion and pending-SYN counts, reset rate, distinct destination-port growth,
peer fan-out, per-direction packet/byte rates, flow lifetime, interarrival
dispersion, repeated connection failures and bounded retransmission estimates.
They require flow identity, timeout, bidirectional tracking, midstream policy,
sequence-wrap handling and memory limits. Missing packets and NAT can change
their meaning. Do not synthesize these features from aggregate CSV columns that
do not contain the necessary event order or joint information.

Compare the existing ten-second window with causal 1/10/60-second summaries or
damped incremental statistics only when raw events or compatible sufficient
statistics exist. Kitsune demonstrates compact online statistics over multiple
time scales and network channels; borrowing the feature idea does not make the
current application a KitNET detector. Keep warm-up state and the longest lookback
inside split rules, and measure memory/CPU/latency alongside detection quality.
[Mirsky et al., Kitsune, NDSS 2018](https://arxiv.org/pdf/1802.09089)

Finally, acquire a paired reference with packet timestamps, device identity,
exact windows, extractor version and independently established labels before
claiming live accuracy. Synthetic parser fixtures prove arithmetic and handling
of malformed/unsupported packets; unrelated public PCAP replay proves replay
execution; neither establishes publisher equivalence or local attack accuracy.
Keep reference extractors isolated under `external/` if later needed.

Do not assume unlabelled deployment shift is only a changed attack prevalence.
Label-shift correction such as BBSE assumes stable class-conditional feature
distributions and other identifiability conditions. New OS/device traffic,
offloading and zero/null mismatches can violate that assumption.
[Lipton et al., ICML 2018](https://proceedings.mlr.press/v80/lipton18a.html)

## Delivery boundary

This note recommends a finite, reviewable next step: nine baseline CPU fits,
shared probability diagnostics, three inference-only MSS stresses, explicit
measurement-quality states, and the bounded cross-chat ablations above.
Execute the repository's required demos and
evaluation suite with the implementation changes. Keep actual result artifacts
separate from this research protocol, retain every limitation above, and require
independent capture evidence before deployment-performance claims. Real HEM,
HEM-derived experiments, DemoDetector, simplified XAI, templates and real LLM
inference retain their existing distinct identities.

## Executed follow-up, 28 October 2026

The matched experiment now has 21 completed XGBoost fits across seven feature
arms and three seeds. Fitting used explicit CUDA and each saved model was reloaded
on CPU. The frozen candidate is 60 numeric features plus 192 port-presence fields;
mean tuning accuracy/macro F1 is 90.43%/54.09%, versus 84.91%/45.34% for numeric60.
See `artifacts/training/live_features_20261027_v1/matched_assessment/RESULTS.md` for
per-arm results and MSS stresses. No additional untested combined deletion was
made. This result is exploratory due to role/vocabulary reuse; it is not measured
Windows live-screening accuracy.

The native feature implementation and independent CSV/PCAP checks are complete
within the stated contract. Broad model adaptation now has a separate resumable
Optuna queue; `docs/BROAD_LIVE_TRAINING.md` distinguishes implemented techniques,
completed smoke evidence and remaining deployment research. The proposed future
stateful flow features and alternate windows above are still proposals.
