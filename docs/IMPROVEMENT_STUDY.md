# Detector improvement study

This is a separate, bounded research experiment. It leaves the installed app's
model, original checkpoints, prepared splits and source CSV unchanged. RAG and
the local language model explain detector evidence; they do not increase the
classifier's measured accuracy by themselves.

## Protocol

The input is the same supplied DataSense dataset and the same within-capture,
chronologically purged partitions used by the full-data study. Every detector
fit includes all **480,405 eligible real training rows**, with **30,000 fixed real
validation rows**, **170 inputs** and **84 classes**. Test arrays are never loaded.
No label, device identifier or timestamp is added to the detector input.
Training uses the actual RTX 4050 Laptop GPU, which has 6 GiB of VRAM on this PC;
XGBoost and PyTorch reject CPU training fallback. App inference remains CPU-only.

The saved app baseline is replayed before fitting. Its initial measured
validation accuracy is **80.053333%**, with macro F1 **0.696463**. A fresh full-data
CUDA refit has reproduced these scores exactly. Validation macro F1 still chooses
the stopping iteration, preserving comparability with the original study;
accuracy, benign false-positive rate, missed-as-benign rate and per-class metrics
are reported separately. This does not silently switch the research selection
criterion to whichever metric looks best.

The following contrasts were frozen before their scores were observed:

| Fit | Change from joint_1 baseline |
|---|---|
| Baseline refit | Same full-data inputs, parameters and seed 42 |
| Gradient sampling | CUDA gradient-based row sampling; subsample 0.6 |
| Leaf-wise growth | `grow_policy=lossguide`, `max_depth=0`, `max_leaves=64` |
| Class weighting | Capped square-root inverse training-frequency weights |
| GAN augmentation | Same baseline detector with additional synthetic minority rows |

An equal-weight probability ensemble of the saved baseline, gradient-sampling
and leaf-wise checkpoints is also specified in advance. Its weights are not
optimized against validation labels. An ensemble is one derived inference
method, not another fitted trial. An unavailable constituent prevents ensemble
reporting rather than silently substituting a different model.

New fits have a 12-minute runtime budget, checked between boosting rounds, in
addition to 500 requested rounds and 40 rounds of validation patience. A
runtime-limited selected checkpoint is explicitly identified in its metadata.
Completed initial fits are preserved when adding this runtime guard.

## What the GAN actually does

`framework/training/augmentation.py` implements a genuine conditional
adversarial generator and discriminator in PyTorch on CUDA. The generator takes
32 noise values, a learned class embedding and fixed donor context. Both networks
have two 128-unit hidden layers. Binary real/fake losses train the two networks;
a small mean-matching term supplements the generator loss. This is a compact
experimental conditional GAN, not CTGAN, SMOTE or random noise renamed as GAN.

Training classes with frequencies in `[100, 1000)` are selected from training
labels alone, giving 7,945 supported minority training rows on this dataset.
Selection and preprocessing do not inspect validation or test values. Signed-log
scaling uses training-only bounds. Class-balanced mini-batches train the networks
for 1,200 steps; each selected class receives at most the lesser of its original
training frequency and 256 synthetic rows.

Protocol-presence bits, protocol counts, window intervals and missing masks are
copied from a real training donor of the conditioning class. Generated numeric
channels are projected into training class ranges, counts are rounded and kept
nonnegative, min/average/max groups are ordered, and TCP flag/packet subset counts
cannot exceed total packets. A schema failure or severe exact-row collapse
rejects the output. Saved synthetic rows, donor indices and GAN checkpoint have
checksums, and resume validates donor class identity and schema constraints.

These checks do **not** establish that a generated row represents a physically
realistic attack. Conditioning labels are synthetic labels, not research ground
truth. The real, unchanged validation partition provides an A/B test of whether
these synthetic training rows help the detector; a negative result remains a
valid result and does not justify app deployment.

## Execution and evidence

From the project root:

```powershell
.\.venv\Scripts\python.exe -m experiment.improve_detector
.\.venv\Scripts\python.exe -m experiment.improve_detector --audit-only
.\.venv\Scripts\python.exe -m pytest evaluation/test_augmentation.py -v
```

The study identity and measured results are under
`artifacts/training/improvement_v1/`. Each fit keeps its checkpoint, native CUDA
configuration, timing, complete validation predictions/probabilities and metrics.
`gan/` keeps the actual trained adversarial networks and synthetic/donor audit.
Resume rejects altered checkpoints, arrays or changed study identity. The audit
command replays every completed saved checkpoint across all validation rows.

The experiment is still a development study on reused validation data. A single
winning seed is not a decisive generalization result. The 99.97% target must be
reported as unmet unless an executed experiment actually reaches it; different
tasks such as the separate binary packet model cannot substitute for this
84-class task. No unseen-capture, unknown-attack or live-PC accuracy is claimed.

Primary references: [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html),
[XGBoost GPU execution](https://xgboost.readthedocs.io/en/stable/gpu/index.html),
and [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784).

## Measured results

Final completed and runtime-limited results are recorded in the study's
`RESULTS.json` and `RESULTS.md`. This section is updated after the bounded batch
and saved-checkpoint audits complete; proposed settings above are not claimed
as successful improvements.
