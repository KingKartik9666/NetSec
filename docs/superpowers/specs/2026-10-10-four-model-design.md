# Four-model extension design

The user requests an incremental, working extension of NetSec with exactly four
model families, retained original architectures, fresh training on the same
CIC-IIoT-2025 dataset, and failures contained to the affected capability.
The attached execution specification authorizes incremental implementation.

## Verified starting point

The existing default is a promoted partial-input XGBoost with CPU inference,
native TreeSHAP, optional LIME, bounded local Qwen/RAG, and passive live capture.
Original datasets, checkpoints, unsuccessful studies and releases must survive.
The root is not a Git repository: use additive modules, a source snapshot and
hash preservation records rather than claiming a branch or commits exist.

The genuine efficient-kan checkout is commit
`7b6ce1c87f18c8bc90c208f6b494042344216b11`; genuine HEM checkout is
`8513ad39c904dbe427727ee6f8470ee8a1ce8e0d`.
HEM's working DataSense adaptation retains the original binary/multiclass neural
classes, KNN and closed-set vote. Native Kitsune preprocessing/checkpoints and
novelty thresholds remain unavailable. Do not enable unknown detection from the
adaptation's k=1 self-neighbor distances. Its ensemble has no supported probability.

Historical RF is an XGBoost standalone forest, not conventional sklearn RF.
The base-paper preprint does not specify RF hyperparameters. Preserve the old
forest; the user is being asked which fixed implementation to train. No RF tuning.

## Dataset and protocol

Official UNB now names DataSense as CIC IIoT Dataset 2025:
https://www.unb.ca/cic/datasets/iiot-dataset-2025.html . Local download provenance
is not proven by a matching title. Preserve source checksum and the inspected
local 84-label hierarchy rather than claiming the publisher has 84 attack types.
Use all 480,405 eligible original training observations and the same 30,000
validation row IDs. Compare manifest/class/row identities of the 71-feature and
170-feature representations. No final test loads or tuning. Report within-capture
limits, zero-support classes, changed feature representations and live limitations.

## Architecture

Keep existing framework/data, detection, training, xai, llm, app and experiment
directories. Add framework/core contracts, configuration and a registry, plus
independent adapters in framework/detection/adapters. Model dependencies are loaded
in spawned, bounded worker processes only on selection. A timeout, exception or
native process exit marks that model/capability unavailable or failed, never changes
the chosen family and never returns substituted predictions. Serialize requests per
worker and terminate/reap workers on shutdown. Metadata, Home and comparison boot
without any model dependency/checkpoint. Prediction never requires explanation,
calibration, retrieval, LIME or a language model.

Contracts: ModelAdapter.health_check/load/predict/predict_proba/metadata; explicit
optional explain support. Registry.call(model_id, operation, payload) returns JSON
compatible values or a scoped CapabilityError. Config controls four model entries
and separate plugin flags. Exactly random_forest,xgboost,kan,hem are accepted.

Preserve existing routes through a lazy XGBoost facade. Add /models and scoped
/api/models endpoints for selector, per-model metadata/health/results/explanations
and comparison. Live mode remains clearly tied to the promoted partial XGBoost;
full-only KAN/HEM/RF inputs are refused for sparse packet measurements. Model names,
versions, measurement coverage and evaluated task remain visible.

## Research and enhancement selection

Fresh bounded control fits retain saved architecture/configuration. XGBoost:
independent L2 regularization ablation. KAN: independent supported spline-weight
regularization ablation. HEM adaptation: independent AdamW weight-decay ablation.
No untested technique combinations or RF optimization. GPU required for these
three and historical forest; conventional RF, if selected, has an explicit CPU
exception because sklearn RF has no CUDA backend. Every fit records CUDA evidence,
training rows, seeds, stopping, runtime/resources, hashes, predictions and full
metrics; saved CPU reload must match. Selection by validation macro F1, keeping
control on ties; upgrade adoption only on measurable improvement. Additional
seeds and justified combinations follow the measured gate, not an assumed gain.

## Verification and release boundary

Tests cover missing dependencies/checkouts/checkpoints, corrupt bytes, disabled
plugins, prediction/explanation exceptions, incompatible rows, missing metrics,
worker timeout/native exit, recovery, auth and shutdown. Distinguish real execution
from injected test doubles. Run all five synthetic demos and complete evaluation.
Run browser four-family selection and real saved-model predictions. Source changes
do not update existing installer bytes; packaging requires a separate complete
freeze/build/installed smoke pass before a new release is claimed.
