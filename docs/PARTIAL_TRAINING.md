# Partial-input multiclass XGBoost study

The optional study keeps the original 170-column, 84-label classifier contract
and trains a new checkpoint to handle unavailable features as native NaNs.
It does not replace missing packet measurements with synthetic values, estimate
sensor logs from network counters, or merge the separate binary packet model
with multiclass results. Default demo and application inference remain CPU,
offline and credential-free; this explicitly invoked research command uses CUDA.

Run from the project environment:

```powershell
python -m experiment.train_partial_xgboost
```

The command resumes completed trials only when their frozen request, source-code
hashes and checkpoint checksum match. An in-progress worker writes to
`artifacts/training/partial_xgboost_v1/runs/<configuration>_s<seed>/console.log`.
Do not start a second coordinator while one is running. `progress.json` records
completed fits, and `RESULTS.json` is written only after all planned fits finish.

## Data and masking

Only the existing `datasense_full_protocols` training and validation arrays are
opened. Test arrays stay locked. All 480,405 original training rows retain their
original 170 feature values and labels. The exact ten-second subset contains
20,966 training rows and 1,571 validation rows; training supports all 84 labels,
while this validation subset supports only 60 labels.

Four additional views of each exact-ten-second training row are added: one
retains the seven directly counted packet/TCP-flag fields, and three independently
mask fields with nominal observed fractions 15%, 40% and 70%. A fixed mask seed
61429 is shared across configurations and training-seed repeats, so repeats
isolate stochastic model fitting. Expanded flags from a protocol field are
masked together. Masks only replace existing values with NaN; they never fill
existing missing values. At least one raw field remains available per random
mask. The effective matrix has 564,269 rows, including 83,864 masked views.

The seven retained columns are `network_packets_all_count` and TCP flag counts
for ACK, FIN, PSH, RST, SYN and URG. SYN+ACK contributes to both corresponding
counts. The remaining 163 columns, including protocol-presence flags, are NaN
for the live profile. Sensor/log data and absent protocol observations are not
fabricated. No exact publisher-extractor equivalence is claimed.

Original full rows have profile weight 1, seven-count views weight 8, and random
partial views weight 2. These fixed weights give explicit training emphasis to
the intended packet profile while retaining the full-row task. Masked views are
not independent observations. Class weights are computed only from original
training labels, using none, square-root inverse frequency, or inverse frequency;
the existing normalization and cap of 20 are retained. Per-trial records contain
the resulting weight sum for every profile and the exact class-weight vector.

Source-row identifiers are disjoint across training and validation. Projection
to seven counts introduces repeated feature vectors: 764 validation vectors
also appear in training. This is disclosed in `plan.json`; neither the count
projection nor the original within-capture split proves cross-capture or Windows
live generalization. Original prepared files and checkpoints are preserved.

The separate descriptive `INPUT_AMBIGUITY.json` audit found 923 unique seven-count
vectors among 1,571 validation rows. Eighteen vectors have conflicting multiclass
labels, affecting 381 rows, with up to eight labels sharing one vector. Any
deterministic function of only these seven counts must therefore make at least
45 errors on this exact partition: its in-partition label-consistency ceiling is
97.135582%. This is an input-information diagnostic calculated from validation
labels, not model performance, an attainable generalization estimate, or a rule
used to predict labels. The 99.97% target is impossible for this fixed profile
and partition without adding information or changing the task. Training has
257 conflicting vectors, 10,176 affected rows and a 92.044262% analogous
diagnostic ceiling. Validation labels were never used to construct predictors.

## Frozen comparisons and selection

Twelve seed-42 configurations compare continuation from the original 76-round
checkpoint against fresh fitting. Depths 4, 6 and 8, learning rates, row/column
sampling, regularization, three class-weight strategies and two loss functions
are specified completely in `plan.json` before research fits. Every fit permits
at most 500 additional rounds and stops after 40 rounds without lexicographic
improvement in seven-count validation accuracy, followed by all-84-class macro
F1. A 720-second runtime cap is checked after every complete multiclass boosting
iteration; recorded elapsed time can exceed the cap by one iteration and setup.

The two strongest seed-42 configurations are repeated at seeds 17 and 73,
giving 16 fits. Selection first ranks configurations by their three-seed mean
seven-count accuracy, with mean macro F1 breaking ties, then chooses the best
checkpoint within that configuration. Mean and sample standard deviation are
reported. These are repeated validation results, not held-out test estimates.

The original checkpoint is evaluated on all three identical validation profiles
before training. Its measured full-input accuracy is 80.053333%, seven-count
accuracy 0.190961%, and random-partial accuracy 10.142160%. These different input
profiles must never be pooled into one accuracy claim. The new bundle is marked
promoted only if its seven-count accuracy exceeds the original masked baseline.

Selection predicts raw validation `DMatrix` inputs, matching application
inference. Training still uses `QuantileDMatrix`. A code review caught the risk
that re-quantizing validation against augmented training cuts could route old
warm-start splits differently. The initial incomplete attempt was stopped and
preserved under `attempts/pre_raw_validation_20261001`; its configuration and
source snapshot remain available. No completed trial or published score was
reused from that attempt. The twelve configurations and masking protocol were
unchanged; the corrected source hash is frozen in the active plan. Selected
per-row GPU predictions and probabilities are checked against CPU reload.

Each checkpoint is evaluated independently on all 30,000 full validation rows,
the 1,571 seven-count rows, and 4,713 deterministic random-partial views. Records
include top-1/top-3 accuracy, macro/weighted F1, 84-class confusion matrix,
per-class support, benign false-positive rate, and attack-as-benign missed rate.
Macro F1 uses all 84 declared classes, assigning zero to unsupported classes;
the 24 unsupported live-validation classes are listed explicitly. FPR collapses
all non-benign multiclass outputs to attack for this particular binary diagnostic.

## Custom objective and numerical proof

Native multiclass cross entropy is compared with label-smoothed cross entropy,
whose target is `(1 - epsilon) * one_hot + epsilon / number_of_classes`.
The gradient is `weight * (softmax - target)`. Its true Hessian includes
cross-class terms; the XGBoost solver receives the positive diagonal upper bound
`max(2 * weight * p * (1-p), 1e-6)`, following the native multiclass approach.
This upper bound is not claimed to be the exact Hessian.

Before any research fit, ten tests check finite-difference gradients and true
Hessian diagonals at smoothing 0, .02 and .05, extreme-margin stability,
invalid inputs, masking integrity and the frozen comparison scope. A separate
preflight compares CUDA Torch derivatives to their NumPy formulas and executes
two CUDA boosting rounds with native CE and zero-smoothed custom CE; their
probability outputs were exactly equal in the recorded preflight. Numerical
test matrices are generated solely for correctness checks, not research metrics.

Torch CUDA performs custom softmax/gradient/Hessian arithmetic. XGBoost 3.1.3
accepts NumPy or CuPy gradient arrays, so explicit host transfers bridge the
Torch objective and XGBoost API. There is no silent CPU arithmetic fallback for
this objective or CPU tree-training fallback. Both preflight and every training
callback inspect the actual XGBoost device setting. Runtime device configuration,
peak process RAM, device-wide VRAM, stopping history, source hashes and checkpoint
reload equality are preserved. Device-wide VRAM includes other applications.

The derivative contract follows the official
[XGBoost multiclass custom objective example](https://xgboost.readthedocs.io/en/stable/python/examples/custom_softmax.html)
and [custom-objective limitations](https://xgboost.readthedocs.io/en/stable/tutorials/advanced_custom_obj.html).

## Packaging and scope

A successful promotion writes schema version 2 to
`framework/models/xgboost_partial/bundle.json`. It retains the original 170-feature
order, 84-class order, protocol vocabulary and sample observations unchanged.
The full-input metrics remain in the existing model fields; seven-count metrics,
profile metrics, parent checkpoint hash, selected run and promotion comparison
are separate explicit fields. The original 512-row train-only XAI reference NPZ
and JSON are copied byte-for-byte, preserving its source/sample lineage.

This study is actual multiclass XGBoost training on supplied DataSense data. It
does not execute original HEM, discover unknown attacks, validate Windows live
accuracy, generate real LLM inference, or establish the 99.97% target. Scores are
uncalibrated. Existing synthetic DemoDetector, simplified XAI, TemplateProvider,
real model inference and real LLM paths keep their separate meanings.

## Completed base-study results — 1 October 2026

All 16 planned CUDA fits completed. The promoted base checkpoint is
`partial-v1-fresh_d6_smooth_s42`, selected from the fresh depth-6, label-smoothed
configuration. It contains 85 saved boosting rounds; fitting executed 125 rounds
including the 40-round validation stopping interval. Its checkpoint SHA-256 is
`6d8e3afa3a1b92cac0f40485f42908474be3a03e84963416356f9ce647768b21`.
`artifacts/training/partial_xgboost_v1/RESULTS.json` and the selected run's
`result.json` contain the complete per-class metrics and 84-by-84 confusion
matrices. The original checkpoint remains unchanged.

| Validation profile | Rows | Original accuracy | Promoted accuracy | Promoted top-three accuracy | Promoted macro F1 | Promoted benign false-positive rate |
|---|---:|---:|---:|---:|---:|---:|
| Full CSV | 30,000 | 80.0533% | 79.3233% | 86.2933% | 0.661233 | 1.2400% |
| Seven packet counts | 1,571 | 0.1910% | 70.1464% | 82.1133% | 0.387033 | 2.8150% |
| Fixed random partial views | 4,713 | 10.1422% | 70.9527% | 81.4555% | 0.386185 | 2.0554% |

The live-profile priority costs 0.7300 percentage points of full-CSV accuracy
and 0.035230 macro F1 relative to the original full-input model. Full-CSV benign
false-positive rate increased from 0.0193% to 1.2400%. These regressions are
retained explicitly; the upgrade is not an improvement on every metric.
Seven-count attack-as-benign missed rate is 6.3030%. Macro F1 includes all 84
declared classes; the seven-count and random-partial partitions support only 60
classes, while full validation supports 83. Top-three accuracy measures inclusion
of the true label among three hypotheses, not three simultaneous detections.

The selected configuration uses learning rate 0.08, maximum depth 6, `max_bin`
128, row sampling 0.8, tree-column sampling 1.0, node-column sampling 0.9,
minimum child weight 3, L2 regularization 3, L1 regularization 0.05 and gamma 0.
Class weighting is square-root balanced and label smoothing is 0.05. The app
executes the saved `multi:softprob` trees on CPU; it does not execute the custom
training objective during inference.

| Repeated configuration | Seeds | Mean seven-count accuracy | Sample standard deviation | Mean macro F1 |
|---|---|---:|---:|---:|
| `fresh_d6_smooth` | 42, 17, 73 | 70.10397% | 0.03675 percentage points | 0.387882 |
| `warm_d8_ce` | 42, 17, 73 | 69.55230% | 0.38893 percentage points | 0.365862 |

An independent execution of
`python artifacts/training/partial_xgboost_v1/audit_study.py` passed. The resulting
`AUDIT.json` verifies the actual CUDA setting for every fit, all 16 checkpoint
hashes, recomputed accuracy/macro F1 on every saved profile, normalized finite
probabilities, frozen source/data hashes, the two three-seed repeat groups and
preserved original bundle fields. It reconstructed all 512 XAI reference rows
from the original training partition. Reloaded selected-checkpoint CPU
probabilities were bit-for-bit identical to saved probabilities on all three
validation profiles. Test arrays were not opened or evaluated.

The completed base study is frozen. Optional imbalance/optimization extensions
are recorded separately in `docs/PARTIAL_ENHANCEMENTS.md`; they must not rewrite
these results or imply that a promoted extension is the original base checkpoint.

## Selected-base packet execution evidence

The selected base model, rather than an unselected development checkpoint, was
executed through the production packet callback and canonical vector pipeline:

```powershell
python -m experiment.verify_partial_capture --mode both --interface Wi-Fi
```

The check passed and preserved the original replay proof. Its new record is
`artifacts/partial_verification/20261001T032332469316Z_117ea9eb/verification.json`.
Offline replay counted 108 headers from the existing benign-source UQ-IoT
recording. The multiclass hypothesis was benign with a 48.7014% uncalibrated
score; command injection and ARP spoofing followed. Actual Windows Npcap capture
counted 1,482 host-relevant headers in a completed ten-second Wi-Fi window. Its
leading hypothesis was Mirai SYN flooding at 20.9201%, followed by Mirai UDP
flooding at 15.5461% and Slowloris port 80 at 14.1483%. This was unlabelled PC
traffic: these low-score hypotheses do not confirm an attack or establish live
accuracy. The separate binary model remains a labelled comparison.

Both executions independently checked packet/TCP-bit counts against a header
oracle, retained exactly seven measured inputs and 163 nulls, reproduced the
vector and top-three prediction through CSV export/reimport, and reproduced
predictions after CPU checkpoint reload. Exact TreeSHAP reconstructed raw class
margins. Stop cleared both multiclass and binary verdicts. No payload was saved,
attack traffic generated, or network mitigation performed.
