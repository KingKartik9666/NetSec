# Bounded enhancements to the partial-input XGBoost model

This study extends the completed 16-fit partial-input study. It preserves the
original DataSense observations, labels, train/validation separation, 170-column
order and 84 categories. It does not evaluate the held-out test or claim accuracy
on live Windows traffic. It is separate from real HEM execution, the synthetic
DemoDetector, XAI and language-model explanation.

## Frozen comparison

The immutable incumbent is `partial-v1-fresh_d6_smooth_s42`, selected by the
original study. Its seven-count source validation accuracy is 70.1464%, top-three
accuracy is 82.1133%, and macro F1 is 0.387033. Full-CSV validation accuracy is
79.3233%. The validation set contains 30,000 full observations and 1,571 exact
ten-second observations; the latter supports 60 of 84 categories. All 84 remain
supported in the 20,966 exact-window training observations.

All treatments continue from that same saved checkpoint and include its
unmodified prediction as round zero. Tree depth 6, learning rate 0.08,
square-root balanced class weights and label-smoothed cross-entropy (0.05) remain
fixed. A matched control measures the effect of additional training alone.

| Treatment | Change from matched control |
|---|---|
| Continuation control | No augmentation or parameter override |
| Random oversampling | Duplicate 2,981 minority-class seven-count training views |
| Constrained SMOTE-style interpolation | Add 2,981 same-class synthetic seven-count training views |
| Conservative leaf update | Set `max_delta_step=1` |
| Gradient-based CUDA sampling | Set `sampling_method=gradient_based` |

Each fit retains all 480,405 original full observations, 20,966 seven-count views
and 62,898 deterministic partial views. Class weights are computed from original
full training labels, with the original profile weights (full 1, seven-count 8,
random partial 2). Augmentation views receive profile weight 8. The control has
564,269 effective rows; augmented treatments have 567,250. Repeated views are
not additional independently collected observations.

## Training-only augmentation and provenance

Parents are drawn only from original exact-ten-second training rows. Each must
have seven finite, nonnegative integer counts no larger than `2**24`, a positive
packet count and each TCP-flag count no greater than the packet count. The 278
zero-traffic rows are excluded as augmentation parents while remaining unchanged
in the original training matrices. There are 20,688 eligible parents.

Each class target is the 75th percentile of eligible training class support.
Additional views cannot exceed twice its original support (three times total
support after augmentation), and total additions cannot exceed half the eligible
parent count. Singleton classes may be duplicated but cannot form a SMOTE pair.

The interpolation is an explicitly constrained SMOTE-style variant: select a
same-class training neighbor using Euclidean distance on `log1p` of the seven
counts, interpolate in that transformed space, then round/project to physical
count bounds. This is not the unmodified imbalanced-learn SMOTE estimator. No
scaler is fitted, protocol flags are not interpolated, and all 163 unavailable
features remain `NaN`. Inherited labels are not independently measured truth;
physical bounds do not prove that generated traffic has the inherited attack
semantics.

Saved provenance includes original parent IDs, both parent labels, the inherited
label, interpolation coefficient, fixed augmentation seed 73219, synthetic versus
duplicate status, generated counts and hashes of every provenance/input array.
The audit reconstructs the generated counts and checks each label against the
original training parent arrays. All three XGBoost seeds use the identical saved
augmentation views. The initial constrained batch contains 260 count vectors
already present in training; these are disclosed rather than claimed as new
traffic. Random oversampling duplicates all 2,981 vectors.

## Budget, selection and release gate

The initial comparison has five CUDA fits with seed 42. Each is bounded to 500
additional boosting rounds, validation patience 40, a 580-second soft stop and a
600-second hard child-process timeout including setup. CPU validation occurs
after the GPU process exits. Selection uses raw `DMatrix` seven-count accuracy,
with macro F1 as tie breaker. Full-input and deterministic partial-input metrics,
top-three accuracy, confusion matrices and benign false-positive rates are
reported separately. Saved GPU probabilities must agree with CPU checkpoint
reload predictions before a run is accepted.

If the best treatment's seed-42 seven-count accuracy exceeds both control and
incumbent, that treatment and control repeat at seeds 17 and 73, for at most nine
fits. Promotion requires the treatment's three-seed mean accuracy to exceed the
matched control's mean and the selected treatment checkpoint to exceed the
immutable incumbent. Full-input regression is disclosed. Otherwise the base
checkpoint stays selected. This is a validation-driven bounded comparison, not
statistical proof of an improvement or a promised 99.97% result.

The source/profile/checkpoint hashes are frozen in
`artifacts/training/partial_enhancements_v1/plan.json`. Reviewed plan SHA256:
`4ac16fc1ee94c0995818096bddcd7158d25845f3a4066640808e0084a70e94b1`.
Original models, datasets and study records are preserved. The application
selects an enhanced bundle only when explicit promotion metadata passes its
release gate; no runtime augmentation is performed.

## Execution

From the project environment:

```powershell
.venv/Scripts/python.exe -B -m pytest evaluation/test_partial_augmentation.py evaluation/test_partial_training.py -v -p no:cacheprovider
.venv/Scripts/python.exe -B -m experiment.train_partial_enhancements
.venv/Scripts/python.exe -B -u -m experiment.train_partial_enhancements --run --reviewed-plan-sha256 4ac16fc1ee94c0995818096bddcd7158d25845f3a4066640808e0084a70e94b1
```

The first experiment invocation only prepares/verifies a CPU plan; it does not
fit a model. The explicit `--run` invocation requires CUDA and refuses CPU
substitution. Existing completed runs resume only with matching request and
checkpoint hashes. An interrupted/failed trial is retained and requires review
before retrying. Research code and generated artifacts stay outside runtime UI
and default CPU inference.

## Verified checks and measured results

The 29 augmentation/objective tests passed. They cover deterministic generation,
missing-feature preservation, physical constraints, zero/singleton and duplicate
parents, training identity allowlists, bounded quotas, provenance tampering,
three-seed promotion gates, CUDA parameter isolation and objective derivatives.
Five actual CUDA fits completed. No technique exceeded the matched control's
seven-count validation accuracy, so the predeclared repeat gate did not open.
No additional seeds were run in this extension, no enhanced bundle was promoted,
and the application's final model remains `partial-v1-fresh_d6_smooth_s42` from
the independently repeated base study. Continuing the control improved two of
1,571 seven-count validation predictions, but that single-seed result does not
satisfy the extension's repeated comparison/release gate.

| Selected checkpoint | Seven-count accuracy | Macro F1 | Top three | Full CSV accuracy | Saved rounds |
|---|---:|---:|---:|---:|---:|
| Immutable incumbent | 70.1464% | 0.387033 | 82.1133% | 79.3233% | 85 |
| Continuation control | 70.2737% | 0.385856 | 82.5589% | 79.2500% | 94 |
| Random oversampling | 70.1464% | 0.387033 | 82.1133% | 79.3233% | 85 |
| Constrained SMOTE | 70.1464% | 0.387033 | 82.1133% | 79.3233% | 85 |
| Conservative leaf update | 70.2737% | 0.387427 | 82.4952% | 79.2867% | 93 |
| Gradient sampling | 70.2101% | 0.387654 | 82.1770% | 79.3600% | 92 |

Random oversampling and constrained SMOTE selected round zero, preserving the
unchanged parent model. Their best checkpoints after at least one added round
were worse: 69.8918% and 70.0827% seven-count accuracy, respectively. Conservative
leaf updates tied the control on the primary accuracy measure while increasing
macro F1 slightly; the required strict accuracy improvement was absent. Gradient
sampling improved full-input accuracy slightly but ranked below the control on
the priority seven-count metric. These negative findings are preserved rather
than replaced with a more favorable input profile.

| Checkpoint | Random-partial accuracy | Full-input benign FPR | Seven-count benign FPR | Random-partial benign FPR |
|---|---:|---:|---:|---:|
| Immutable incumbent / unchanged augmentation selections | 70.9527% | 1.2400% | 2.8150% | 2.0554% |
| Continuation control | 71.1649% | 1.2335% | 2.6810% | 2.2341% |
| Conservative leaf update | 71.2073% | 1.1629% | 2.6810% | 2.0554% |
| Gradient sampling | 71.1649% | 1.1821% | 2.6810% | 2.1001% |

Random-partial evaluation uses the unchanged 4,713 deterministic masked views.
These source validation scores do not establish live-PC accuracy. Confusion
matrices, all 84 per-class supports, accuracy, top-three accuracy, macro F1,
benign false-positive rates and saved probability matrices for all three input
profiles are recorded per run and in
`artifacts/training/partial_enhancements_v1/RESULTS.json`.

Independent audit passed all five saved checkpoints. It reconstructed the full
training matrices, weights and augmentation provenance; verified frozen source
and checkpoint hashes; checked actual `grow_gpu_hist`/CUDA configuration and
parameter overrides; and reproduced all three stored CPU probability matrices
bit-for-bit. Selected CUDA seven-count predictions agree with CPU reloads within
the documented floating-point tolerance. Actual booster rounds equal both saved
metadata and 85 parent rounds plus selected additional rounds. Round-zero
checkpoints reproduce every incumbent profile metric. Each GPU fit completed in
136–187 seconds, with 40–49 additional rounds executed, below the 600-second and
500-round limits. The immutable incumbent and all original study assets remain
unchanged. See `artifacts/training/partial_enhancements_v1/audit.json` and its
read-only reconstruction script.

## Primary references and prior negative findings

[imbalanced-learn oversampling](https://imbalanced-learn.org/stable/over_sampling.html)
describes duplicate sampling and same-class neighbor interpolation; mixed
categorical data requires different handling. Our constrained implementation
only interpolates the seven numeric counts. [Its leakage guidance](https://imbalanced-learn.org/stable/common_pitfalls.html)
motivates preserving partitions and never resampling validation observations.
[XGBoost parameters](https://xgboost.readthedocs.io/en/release_3.1.0/parameter.html)
document bounded leaf updates and CUDA histogram gradient-based sampling. These
are feasible candidates, not assurances that validation scores will improve.

The preserved earlier full-feature GAN experiment achieved 79.6533% validation
accuracy compared with its unchanged 80.0533% baseline. Its macro F1 increased to
0.702074 from 0.696463, but it did not improve accuracy. Those are different
full-input experiments, not the seven-count results here. Repeating GAN is not
part of this bounded extension, and no synthetic GAN evidence is presented as
real captured traffic.
