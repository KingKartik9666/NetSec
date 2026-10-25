# Preserved model implementations and fixed Random Forest control

Audit date: 10 October 2026. This records source and saved-checkpoint verification;
fresh extension-study results belong in their separate experiment manifests.
Original upstream repositories, previous experiments, datasets and checkpoints
are preserved. No historical metrics are renamed as newly measured results.

## Random Forest

The [base paper preprint](https://arxiv.org/html/2409.13177v1), corresponding to
[IEEE DOI 10.1109/WF-IoT62078.2024.10811456](https://doi.org/10.1109/WF-IoT62078.2024.10811456),
uses Random Forest with CICIoT2023. It does not publish sufficient RF settings
to reproduce the classifier exactly. Its reported accuracy cannot be transferred
to this project's different dataset, feature representation or label hierarchy.

Following the user's explicit selection, the extension uses a **fixed
conventional sklearn RandomForestClassifier**, with 100 trees, Gini splits,
bootstrap sampling, square-root feature sampling, unrestricted depth and default
split/leaf/regularization settings. Seed 42 and two CPU worker threads are explicit
reproducibility/platform choices. No class weighting, augmentation, feature
selection, parameter search or validation-driven stopping is added to this RF.
See [`framework/training/fixed_random_forest.py`](../framework/training/fixed_random_forest.py)
and [the pinned-version RF documentation](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

The trainer uses every supplied eligible original training row and fits median
imputation on training only. Dataset-compatible preprocessing and the encoded
class map are recorded with the run. The model is saved as a trusted local joblib
Pipeline. Loading does not change any fitted RF parameter. sklearn RF training
uses CPU because that implementation has no native CUDA backend; it is an explicit
exception to the CUDA research policy, not an undisclosed GPU replacement.

This is a fixed algorithm-family baseline using documented chosen defaults,
**not exact reproduction of the paper's unpublished parameters**. Newly executed
scores, training time, tree-node count and artifact size must come from its run.

Historical `framework/training/trees.py` uses **XGBoost's standalone GPU forest**:
one boosting round, parallel trees, sampling without replacement and a second-order
objective. It is not the paper's conventional RF. That implementation, its tuned
historical results and its 170-feature checkpoint remain preserved and inactive
in the new default selection. The RF adapter can explicitly load its UBJ format
for historical audit, with disclosed metadata; it never substitutes it for a
missing conventional RF checkpoint.

## KAN

The project's original KAN implementation is the genuine
[Blealtan/efficient-kan repository](https://github.com/Blealtan/efficient-kan),
MIT, package version `0.1.0`, commit
`7b6ce1c87f18c8bc90c208f6b494042344216b11`. The audited checkout has no tracked
modifications. The package is loaded from its source checkout rather than an
installed Python distribution. Its research foundation is
[KAN: Kolmogorov-Arnold Networks (2024)](https://arxiv.org/abs/2404.19756).

`external/efficient-kan/src/efficient_kan/kan.py` combines a SiLU base branch with
learned B-spline edge functions. It is not an MLP substitute. The upstream package
supports grid updates and a weight-L1/entropy regularizer, which differs from
pykan's sample-based sparsity regularization. Existing training uses fixed grids;
unsupported pruning or symbolic extraction is not claimed.

The preserved full-data seed-42 checkpoint has architecture 71 → 96 → 48 → 84,
grid size 5 and spline order 3. Each edge has eight spline coefficients. Its
fitted preprocessing is training-only median imputation followed by clipped
MinMax scaling to [-1, 1]. The selected epoch is 9. Historical validation accuracy
is 75.7367% and macro F1 is 0.622731 on the existing 30,000-row validation partition.
These are source-data validation results, not test or live-PC accuracy.

Checkpoint:
`artifacts/training/datasense_all_models_full_v1/runs/kan_full_previous_best_s42_20260929T180844161663Z/model.pt`

SHA256: `4bcf55323dfe7c8b0b731a6ac4862321d0a2525a5e4ba65e6283cecc24f782d2`.
The checkpoint is 632,987 bytes; preprocessing is a separate `preprocessing.joblib`.

`framework/detection/adapters/kan.py` imports only the hash-pinned original source
inside the selected worker, uses `torch.load(weights_only=True, map_location="cpu")`,
restores all saved spline/grid weights, and applies the saved preprocessing.
No CUDA-only training helper or alternate network is used for app inference.
Missing Torch/source/preprocessing or corrupted bytes affect only this adapter.
Raw softmax scores remain uncalibrated; unknown-attack detection is not enabled.

The adapter also exposes optional `gradient_input_sensitivity` through its
`native`/`gradient` explanation operation. Autograd differentiates the selected
class's raw logit against the saved scaled input, including the genuine spline
derivatives. Raw measured values, scaled values and signed gradients are retained.
The signed gradient-times-scaled-input ranking is a heuristic, not a complete
additive contribution, SHAP/LIME, or causal evidence. Clipping is not differentiated;
scaled-coordinate sensitivity does not describe arbitrary out-of-range raw changes.
Explanation failures do not mutate model weights or prevent subsequent prediction.

The hash-pinned `source_record` additionally binds feature order, class order,
configuration and upstream commit to the completed training record. Equal tensor
dimensions alone are insufficient evidence of schema compatibility.

## XGBoost and HEM boundaries

The existing promoted partial-input XGBoost remains its genuine preserved
checkpoint. Its canonical missing-value schema and its separate binary packet
comparison are documented in `PARTIAL_TRAINING.md` and `PARTIAL_ENHANCEMENTS.md`.
One family can contain these explicitly different evaluated tasks; their scores
must not be combined into a single ranking.

The HEM checkout and the previously trained closed-set adaptation retain original
binary/multiclass neural classes, KNN and source-equivalent ensemble voting.
See `HEM_INTEGRATION.md` and `FOUR_MODEL_HEM.md` for the source/native feature and
novelty boundaries. Tabular adaptation is not native Kitsune preprocessing or
original unknown-attack reproduction. Ensemble labels have no fabricated
probability. Removing HEM does not disable RF, XGBoost or KAN.

## Protocol and executed adapter verification

The existing prepared representations retain the same 480,405 eligible training
row IDs and 30,000 validation row IDs with 84 local fine-grained category names.
KAN/HEM and the conventional RF use the 71 numeric inputs. The historical forest
and promoted XGBoost use 170 numeric/protocol inputs. Such representation differences
must remain visible in comparisons. The test set stays locked.

On 10 October, actual CPU reloads of the preserved full-data KAN and historical
forest processed 128 existing validation observations. Their finite `(128, 84)`
probabilities produced exactly the saved class predictions. Loading plus prediction
was 0.1672 seconds for KAN and 0.7644 seconds for the historical forest; these are
execution checks, not steady-state latency or new accuracy benchmarks.

`evaluation/test_modular_neural_adapters.py` uses real small genuine-KAN,
standalone-forest and conventional-RF fixture checkpoints. Tests cover missing
dependencies/checkouts, corrupt bytes, class/architecture mismatch, lazy imports,
path containment, ordered training-record identity, missing metrics and malformed/sparse inputs.
KAN gradient tests verify a hand-calculated local derivative and injected explanation
failure without weight or prediction changes. The fixtures are
software-test data, not research experiments. RF trainer tests verify train-only
medians, all rows retained, fixed settings, saved CPU reload and overwrite refusal.

Full-only models reject sparse packet vectors. Missing values are never silently
converted to measured zeros. RF's supported global importance is explicitly marked
as global, not per-observation SHAP/LIME or causal evidence. Prediction remains
independent of optional explanation plugins.
