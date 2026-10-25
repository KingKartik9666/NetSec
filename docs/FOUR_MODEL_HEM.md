# HEM in the four-model workspace

The HEM family uses the original binary and multiclass neural classes from
`external/HEM`, commit `8513ad39c904dbe427727ee6f8470ee8a1ce8e0d`.
The adapter is additive; upstream tracked source and historical checkpoints are
preserved. The working mode is explicitly a **DataSense closed-set adaptation**.
It is not reproduction of the native UQ-IoT/IoT-NID benchmark or unknown-attack
evaluation.

## Original architecture and current mode

Upstream `model_evaluation/binary_classifier.py` defines a ReLU MLP and binary
logit. `multiclass_classifier.py` defines the original expandable ReLU MLP and
multiclass logits. `full_model.py` combines these with KNN. Its benign gate
requires a benign binary vote and a benign multiclass or KNN vote. Otherwise,
the closed-set rule prefers a nonbenign KNN class, then the multiclass class.
The existing adaptation retains these component roles and that decision rule.

The preserved full-data checkpoint uses 71 numeric inputs, training-fitted median
imputation and clipped MinMax scaling, 16 PCA dimensions, both original neural
architectures, and one nearest neighbor over 480,405 training references. Its
label map contains 84 categories. These values describe that checkpoint; fresh
research runs must record their own manifests and results. Stored metrics do not
become newly executed benchmarks merely because an adapter reloads a model.

The separate native pipeline requires 100 stateful Kitsune/AfterImage statistics,
packet length, the fitted protocol vocabulary, scaler/PCA and native model
artifacts. UQ uses 12 PCA dimensions and k=3; IoT-NID uses 26 dimensions and k=7.
DataSense columns and seven Npcap counts are not equivalent to those stateful
statistics. The original extractor can run independently, while native trained
ensemble inference remains unavailable. See `HEM_INTEGRATION.md` for verified
source contracts and outstanding native integration issues.

## Adapter contract and isolation

`framework/detection/adapters/hem.py:HEMModelAdapter` accepts a configuration
containing a project-relative manifest path. The manifest binds ordered feature
names, classes, benign index, architecture, checkpoint version, protocol identity
and SHA256 hashes. It must pin the checkpoint, fitted preprocessing, ordered KNN
references and all three original model source files. Explicit preprocessing and
reference paths are supported; otherwise they are siblings of the checkpoint.
When `source_record` is supplied, its relative path must also be hash-pinned.
The adapter verifies the record before deserialization and checks family identity,
ordered feature names, ordered classes, configuration, upstream commit and its
recorded benign index. Both historical `candidate`/`features` names and new
controller `model_id`/`feature_names` names are supported. Release packaging pins
this record; accepting legacy manifests without it does not independently bind
same-length reordered labels/features that old checkpoints never stored.

Dependencies (`torch`, NumPy, joblib and scikit-learn) load only in the selected
HEM worker. Health checks inspect availability and hashes without deserializing
weights. Loading verifies checkpoint architecture/class identity, preprocessing
dimensions, finite weights/references and training-reference order. The adapter
uses CPU only and imports the original neural classes directly; it does not call
the research helper that requires CUDA or upstream automatic device selection.

Prediction accepts complete finite numeric rows in manifest order. Missing
columns/cells, malformed shapes, negative/fractional count measurements and TCP
flag counts exceeding total packets are rejected. Sparse packet observations
are incompatible until evaluated separately. No detector is substituted.

CPU KNN computes exact Euclidean distances in bounded reference/query chunks.
Distance ties retain earlier stored reference indices; class-vote ties select
the lowest class index. Labels and original training indices remain aligned.
The registry owns the process timeout/native-exit boundary: HEM failure affects
its worker instead of taking down unrelated application capabilities.

## Probabilities, novelty and explanation

Final predictions return `score: null`, `top_classes: []` and
`novelty: "not evaluated"`. The original voting ensemble has no supported
probability. Binary sigmoid and multiclass softmax values are available as
explicitly uncalibrated **component** scores, never ensemble confidence.
`predict_proba` raises an explicit unsupported-capability error.

The optional explanation exposes actual component votes, PCA coordinates,
measured feature values, nearest training indices, labels and distances. It is
labelled **component/PCA/nearest-neighbor diagnostics**, not SHAP, LIME, causal
attribution, or a calibrated incident assessment. PCA coordinates combine
scaled measurements; proximity to a training row does not confirm an intrusion.
Prediction does not depend on this diagnostic capability.
The worker may request `native` or `component_diagnostics`; unsupported methods
return an explicit capability error instead of mislabeled explanations.

Unknown detection stays disabled. In particular, the adaptation's k=1 produces
zero training self-neighbor distances; reusing those as native novelty thresholds
would be invalid. Native threshold evaluation also requires fixing fit-order
handling and the upstream `include_novel` forwarding issue through an explicit,
verified bridge without modifying original sources silently.

## Verification scope

Run the adapter tests with the project environment:

```powershell
.venv/Scripts/python.exe -B -m pytest evaluation/test_modular_hem_adapter.py -v
```

The tests use original upstream neural classes with tiny, explicitly synthetic
integration checkpoints. They verify component voting, deterministic duplicate
neighbor handling, CPU execution, incompatible measurements, corrupt/missing
assets, missing dependencies and unsupported ensemble probabilities. They do
not establish research classification accuracy. Fresh checkpoint reload and
research comparisons require separately recorded real-data runs.

HEM is MIT licensed; preserve its root copyright notice and the separate Kitsune
MIT notice when packaging upstream assets. Original sources:
[HEM repository](https://github.com/trifle19/HEM),
[HEM paper DOI](https://doi.org/10.1016/j.future.2026.108447).
