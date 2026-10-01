# HEM integration findings

**The default MVP does not execute HEM inference.** It uses `DemoDetector`.
`HEMAdapter` is a future integration boundary. The separate GPU research workflow
now trains and runs a documented DataSense closed-set HEM adaptation, described
below; it does not reproduce the original feature pipeline or unknown-attack
experiments.

Inspected the existing [HEM checkout](../external/HEM/README.md) at commit
`8513ad39c904dbe427727ee6f8470ee8a1ce8e0d`. Inspection covered its README,
requirements, Python modules, notebook source cells, labeling rules and artifact
paths. The initial review was static. Subsequently the user supplied UQ-IoT PCAPs
and Scapy/Cython became available; the optional feature command below now executes
the original feature engine. HEM source remains unchanged.

## Separate DataSense training adaptation

`framework/training/neural.py` imports the original binary and multiclass neural
classes without modifying HEM. Our training code fits them on DataSense inputs,
fits imputation/scaling/PCA on training rows only, and evaluates a CUDA KNN
reference set with voting checked against the original closed-set decision rule.
The reference subset is bounded to 20,000 or 40,000 training observations as
recorded per trial. Class ordering, preprocessing and both selected neural states
are saved with each checkpoint. These are newly trained DataSense artifacts,
not upstream pretrained models.

Use `python -m experiment.predict_trained --record artifacts/training/selected/hem.json`
to reload the selected adaptation and classify a real validation row. No novelty
threshold, unknown-attack benchmark, original HEM performance reproduction, XAI
or LLM inference is claimed by this command. See [GPU training](GPU_TRAINING.md)
for the split, actual backend and experiment limitations. The original HEM
preprocessing/novelty integration requirements below still apply to that separate
integration.

## Verified executable feature component

`python -m experiment.hem_features --sample syn --limit 1000` imports original
`after_image.net_stat.netStat` and calls `updateGetStats` on actual PCAP packets.
`after_image.py` computes 100 stateful Kitsune statistics unchanged. Our wrapper
uses Scapy packet iteration and matches the upstream `FE` host/session caps.
It does not use upstream `FE.get_next_vector_scapy`, whose source indexes
`PcapReader.read_packet()` as a `(packet, metadata)` tuple; the installed Scapy
iterator supplies a Packet. No upstream patch is necessary.

Actual runs on locally supplied UQ-IoT files, each limited to 1,000 packets:

| Capture | Feature rows | Skipped |
| --- | ---: | ---: |
| `SYN_Flooding_Smart_Clock_1.pcap` | 999 | 1 timestamp regression |
| `Port_Scanning_Smart_Clock_1.pcap` | 1,000 | 0 |
| `whole_week.pcap` | 1,000 | 0 |

Each CSV has `timestamp`, `length`, `protocol`, and `kit_fe_0..99`. Companion JSON
records input path/size, source path/hash, original feature-name mapping, protocol
counts and filtering. Unsupported packets and timestamp regressions are skipped
and counted; each capture starts fresh feature state. These are bounded functional
extraction checks, not reproduction of complete preprocessing or published metrics.

Captures currently reside in a user-added nested `69897e94e24170c0_UQIOT2022_A7369`
directory under HEM; the command discovers them without moving them. Capture names
are not per-packet labels: the SYN-named prefix contains UDP background traffic.
Raw features still need a verified protocol vocabulary, fitted scaler/PCA, label
mapping and trained models before real ensemble inference can run.

## What is available

| Component | Local implementation | Reuse requirements |
| --- | --- | --- |
| Packet features | `feature_extraction/after_image/` and extraction notebooks | Ordered PCAP traffic, Scapy or TShark, stateful Kitsune statistics; preserve MIT notices |
| Binary and multiclass classifiers | `model_evaluation/binary_classifier.py`, `multiclass_classifier.py` | PyTorch, matching architecture, trained weights and preprocessing |
| Ensemble and novelty logic | `model_evaluation/full_model.py`, `DecisionModel` | Both classifiers, fitted KNN, matching training labels, thresholds and class mapping |
| Training and evaluation | `3-1_train_novel_scenarios.py`, `3-2_train_all_known_scenario.py`, `4-1_check_alpha.py`, evaluation notebooks | Prepared datasets, separate research environment and path/correctness fixes below |
| Dataset metadata | `ku_iot_dataset/labeling_info.csv` | Labels original IoT-NID captures; it is not traffic data or a trained model |

Source structure and algorithms are reusable as an integration reference now.
The upstream repository supplies **no pretrained `.pt`/`.joblib` models or fitted
preprocessors**. Its README states that benchmark data is not distributed in Git.
The user has now added real PCAPs locally; our wrapper generates raw feature CSVs
separately. Prepared training splits, fitted preprocessing and models remain absent.
Configuration lives in script/notebook variables, not a standalone configuration
file. The upstream requirements include PyTorch, Jupyter, Scapy, Cython and
research plotting/baseline packages; they are intentionally separate from MVP
requirements. HEM has a CPU fallback; its feature engine has been runtime-tested
here. The separate DataSense workflow also executes its neural classes with our
adapted training and closed-set ensemble; the original ensemble/training scripts
have not been reproduced end to end.
Its pinned research stack includes `torch==2.6.0`, `scikit-learn==1.6.1`,
`numpy==2.2.5` and `pandas==2.2.3`. No installed-environment compatibility test
was performed; adding its heavy dependencies to the lightweight MVP is unnecessary.

HEM's [root license](../external/HEM/LICENSE) is MIT, copyright 2026 Janet Hyunjae
Kang. The adapted Kitsune/AfterImage extractor retains its separate
[MIT notice](../external/HEM/feature_extraction/after_image/LICENSE). Preserve both
applicable attributions when reusing code. No upstream code was copied into the MVP.

## Data and exact feature contract

HEM's implemented dataset branches are **UQ-IoT-IDS** (`uq_iot`) and **IoT-NID**
(`ku_iot`). CICIoT2023 would require a separate ingestion/label mapping study.
UQ extraction expects `UQIOT2022/data/` PCAPs; its sampling notebook also expects
packet CSV metadata with `Protocol` and `Length`. IoT-NID expects PCAPs under
`ku_iot_dataset/data/` and the supplied labeling rules. Its labeling and metadata
notebooks explicitly require TShark, even though feature extraction can use Scapy.
The source sequence is `uqiot-01-feature_extraction.ipynb` then
`uqiot-02-sampling.ipynb`, or `kuiot-01-feature_extraction.ipynb`,
`kuiot-02-labeling.ipynb` and `kuiot-03-sampling.ipynb`, followed by the shared
preprocessing notebook below.

The preprocessing contract is not the six understandable features in the demo:

1. Extract **100 stateful Kitsune features** per packet: MAC/IP (15), host/host
   (35), host/host jitter (15), and host/port relationships (35), over decay
   factors `[5, 3, 1, 0.1, 0.01]`. Inputs include MAC/IP addresses, ports/protocol,
   packet timestamp and frame length. Packet ordering and extractor history
   therefore matter. See `after_image/net_stat.py:289-309,377-385`.
2. Sampling notebooks construct `length`, `kit_fe_0` through `kit_fe_99`,
   dataset-specific `protocol_*` one-hot columns, and `label`. Preserve protocol
   vocabulary and exact column order; UQ initially builds the vocabulary as a set.
3. Fit `MinMaxScaler` on **training data only**, scaling `length` and the 100
   Kitsune columns. Protocol columns remain one-hot. Fit PCA on every feature
   except `label`. The notebook concatenates unscaled columns before scaled
   columns, so PCA sees protocol columns followed by length/Kitsune columns.
   Persist the fitted transformers and their ordered schema.
4. Produce `PC1` through `PC12` for UQ, or `PC1` through `PC26` for IoT-NID,
   plus `label`. For leave-one-attack-out scenarios, exclude the held-out class
   from training **and validation**, refit preprocessing, remap known labels
   contiguously, and map the held-out test class to `-1`. Benign remains `0`.

These operations are in
[`uqiot-03-kuiot-04-new_class_preprocessing.ipynb`](../external/HEM/feature_extraction/uqiot-03-kuiot-04-new_class_preprocessing.ipynb),
source cells 3-4 (one-based notebook cell numbering). Sampling balances/subsamples
classes; this is a research preparation pipeline, not a raw-flow inference API.

## Models, decisions and artifacts

The configured UQ binary MLP has hidden layers `[1024, 256, 64, 16]`, multiclass
MLP `[512, 256, 32, 16]`, and KNN `k=3`. IoT-NID uses binary
`[512, 128, 64]`, multiclass `[512, 256, 128, 64]`, and `k=7`
(`3-1_train_novel_scenarios.py:8-42`). MLPs use ReLU hidden layers; binary outputs
use sigmoid at inference and multiclass outputs use softmax.

`DecisionModel.get_kNN_thresholds` computes each class threshold as the mean of
all KNN distances over at most 2,000 sampled training rows of that class
(`full_model.py:318-330`); those queries include self-neighbors. After its benign
agreement gate, `voting_ensemble` can mark a row novel when the row's mean KNN
distance exceeds `alpha * threshold[predicted_knn_class]`
(`full_model.py:477-526`). This is more specific than a global nearest-neighbor
distance cutoff. Alpha is swept from 1 through 8; the final-results notebook
selects 5 for UQ and 3 for IoT-NID. Those are upstream configuration choices,
not values validated by this project.

Precisely, let `b`, `m` and `k` be binary, multiclass and KNN predicted labels.
If `b == 0` and either `m == 0` or `k == 0`, return benign immediately. Otherwise,
when novelty is enabled, return `-1` if the distance test passes. If it does not
pass (or novelty is disabled), return `0` when `m == k == 0`, else return `k`
when `k > 0`, otherwise return `m`. This follows the active branches of
`voting_ensemble`; it is not generic majority voting.

The ensemble returns integer labels: `0` benign, positive scenario-specific
known class, and `-1` novel. Submodel methods expose probabilities and distances,
but `full_predict` returns labels only, **not a calibrated ensemble confidence**.
Research evaluation writes prediction CSVs (`y_true`, `y_pred`) and per-class
precision/recall/F1 CSVs. **No published HEM metric is currently reproducible
from source and the available capture prefixes alone**, because trained/preprocessed
artifacts are absent and source issues remain. No HEM evaluation result has been reproduced
here; notebook output saved upstream is not local experimental evidence.

For one consistent dataset/scenario, future inference needs:

- Two PyTorch **state dictionaries**: `model_evaluation/<dataset>_pca/`
  `binary_classifier/layer<layers>_<scenario>.pt` and
  `multi_classifier/layer<layers>_<scenario>.pt`.
- Fitted KNN: `model_evaluation/<dataset>_pca/knn/<scenario>.joblib`.
- Fitted scaling/PCA: `feature_extraction/<dataset>/preprocessing_models/`
  `scaler_<scenario>.joblib` and `pca_<scenario>.joblib`.
- Architecture parameters, ordered feature/protocol schema, dataset/scenario
  class map, selected alpha, and threshold calibration data or persisted
  thresholds. Current HEM also needs training rows/labels in **KNN fit order**.

Here `<scenario>` is `all` or `no_<attack_name>`. Save routines are in
`full_model.py:338-345`; preprocessing persistence is in notebook cell 4.
Thresholds and a complete inference manifest are not saved by those routines.

## Issues to resolve before real execution

Static source inspection identified integration blockers; these are not runtime
validation results and no upstream fixes were made:

- The preprocessing notebook calls `PCA` without importing it. It writes
  `processed_data/`, while HEM trainers read `processed_data_pca/`. Sampling and
  final-results notebook paths also need reconciliation with their producers.
- Training shuffles rows with seed 777 before fitting KNN. The alpha evaluator
  reloads the unshuffled CSV, while `get_all_kNN_metrics` indexes
  `self.train_set['label']` with fitted KNN neighbor indices. Preserve fit order
  or correct this label lookup before trusting predictions
  (`3-1_train_novel_scenarios.py:55-59`, `4-1_check_alpha.py:85-101`,
  `full_model.py:435-450`).
- `full_predict(..., include_novel=False)` does not forward that flag to
  `voting_ensemble`; the ensemble defaults to novelty enabled and may use unset
  thresholds (`full_model.py:529-531`).
- Same-directory imports, batch loaders that expect a `label` column, output
  directory creation and automatic device/worker selection need an explicit
  inference wrapper. Do not import training scripts for prediction.

The main training entry points are the two `3-*` scripts above. Inference is
implemented in `DecisionModel.submodels_predict`, `voting_ensemble` and
`full_predict` in `model_evaluation/full_model.py`. It is Python class code, but
is **not a ready installable inference package**: its absolute sibling imports
assume `model_evaluation/` is on the import path. A controlled module-loading
boundary or packaging refactor plus the fixes above is required for integration;
simply importing it from the project root does not establish working inference.

## Adapter boundary

The MVP adapter should report missing artifacts and raise an actionable
unavailability error. Discovering artifact filenames alone cannot establish
that a working HEM backend exists; it must never silently return demo predictions.

A later adapter should validate a versioned artifact bundle, apply the exact
packet-feature/scaler/PCA contract, call the repaired and verified HEM inference
methods, and normalize output to the framework's detection schema. Keep original
submodel scores, distance, threshold and class-map provenance available as
evidence. Define confidence semantics explicitly rather than inventing an
unknown-attack probability.

The pipeline and reasoning provider can retain their interfaces. A HEM-capable
explainer must separately explain its actual MLP/KNN/PCA evidence; the demo's
Random Forest explanation cannot be relabeled as HEM XAI. Train and validate on
real known/held-out attacks before enabling real HEM mode or making accuracy
claims. The current synthetic demonstration supplies none of that validation.
