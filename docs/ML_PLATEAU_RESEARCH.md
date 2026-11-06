# Existing-model performance and dataset intelligence

Research started on 29 October 2026, following the owner's optimization brief.
This is a separate study of the existing 655-feature models. The active
252-feature live adaptation queue and installed application are preserved.

## Evidence and acceptance criteria

The starting evidence is the independent NetSec audit, checked against source
and artifacts in `artifacts/audit_verification_20261029/REPORT.md`, together with
the exhaustive catalogue in `artifacts/feature_audit_20261027/`. The external
reports were read earlier in this chat; their original `K:/NetSec_Audit` paths
were unavailable when this implementation resumed. The preserved verification
report and its historical report hashes remain available. Do not claim a fresh
hash check of those now-unavailable external documents.

The verified leading hybrid scores 81.5033% on all historically reused validation
rows and 79.9170% on its reserved assessment role. Shared errors do not establish
an information ceiling. Merging recon classes changes the task. Declared-84 and
supported-class macro F1 are distinct metrics; both should be reported.

The immediate acceptance criteria are an isolated reproducible pilot, equal
row/seed/round budgets across representations, fitting-only duplicate selection,
explicit projected-vector purging, actual CPU checkpoint replay, resumable saved
results, and fresh project checks. A higher score is an empirical outcome, not a
prerequisite for honestly completing an experiment. Original models, labels,
data, application defaults and locked test arrays must remain unchanged.

The local data are a DataSense-format copy. The authors identify DataSense with
the CIC IIoT 2025 release, but publisher byte identity for this local CSV remains
unverified. See the [authors' paper](https://www.mdpi.com/2079-9292/14/20/4095)
and [CIC dataset catalogue](https://www.unb.ca/cic/datasets/).

## First controlled experiment

Implementation: `framework/training/plateau/` and
`experiment/investigate_plateau.py`. Study:
`artifacts/research/plateau_20261029_v2/`. Revision 1 preserves preflight evidence
and prepared arrays but ran no real-data fits. Independent review found a saved
result binding weakness; revision 2 adds protocol checksums and result/request
receipts before real-data execution. Both versions remain available.

One deterministic proportional sample contains 24,000 rows from frozen fold 0's
original fitting role and at most 8,000 from its later `early_stop` role. The
pilot retains the original fitting population, including no-signal observations;
it does not apply the historical global strict mask. Original validation and
OOF arrays are not loaded. The pilot does not perform early stopping or select
hyperparameters using its reporting set.

Four independent representations are compared:

| Arm | Intervention | Reason to test |
| --- | --- | --- |
| all_features | All 655 fields | Matched control |
| drop_tcp_flag_std | Remove only TCP flag standard deviation | Prior permutation suggested possible harm; removal requires retraining |
| drop_bitmask_summaries | Remove eight IP/TCP numeric bitmask summaries; keep flag counts/fractions | Tests the group, whose prior perturbation actually hurt macro F1 |
| deduplicate_nonconstant | Keep one of each exact nonconstant fitting-sample duplicate group | Tests redundancy; constants and missing states remain |

Every arm uses the same fitting and reporting rows, seeds 42/43/44, native
multiclass cross entropy, CPU histogram XGBoost, depth 5, max-bin 64, learning
rate 0.1, min-child-weight 5, lambda 1, row subsampling 0.9, and all-column
sampling. Full column sampling avoids changing a random feature-subsampling
budget when columns are removed. All fits run exactly 96 rounds, with metrics at
32 and 96 rounds; there are 12 fits. One CPU thread and below-normal process
priority limit competition with the existing GPU study. Per-fit safety budget:
30 minutes. Overall wall budget: six hours from the first run, including stopped
time. Both budgets are cooperative: they are checked between boosting rounds and
scoring/publication steps, and a stuck native operation can overrun. There is no
external watchdog. A safety-interrupted fit is incomplete, never a shorter
comparable result.

Removing columns can make previously different rows identical. The preparation
therefore computes fit-to-reporting vector collisions separately under all four
representations and removes their union once from the shared reporting sample.
It preserves the removed source IDs, counts and class support before/after. This
changes the reporting population to rows novel under all tested projections;
the exclusion is not hidden. Remaining within-reporting duplicate counts are
reported. Equality preserves NaN as distinct from zero and canonicalizes signed
zero. Duplicate selectors see only fitting values, never reporting values or labels.

Metrics include accuracy, declared-84 and supported-class macro F1, per-class
precision/recall, multiclass argmax benign false alarms and attack misses, exact
recon/non-recon accuracy, NLL, multiclass Brier score and uncalibrated ECE.
Calibration is measured here, not fitted. CPU reload must reproduce saved
probabilities. The inference schema and prediction/checkpoint hashes are saved.

This is exploratory: the frozen 655 vocabulary includes supervised port selection
using the complete original training partition, including the later internal
folds. Historical fold roles have also been reused. The historical global strict
mask considered label conflicts across all training rows; this pilot avoids that
mask, but cannot retroactively remove the vocabulary reuse. These results are
neither an independent holdout nor comparable absolute scores to the frozen
full-scale 80-82% models. The pilot does not test whether extending the selected
depth-9, 215-round checkpoint would improve it.

## Commands and recovery

From the project environment:

```powershell
python -m experiment.investigate_plateau prepare
python -m experiment.investigate_plateau run
python -m experiment.investigate_plateau status
python -m experiment.investigate_plateau stop
```

The controller has an exclusive OS lock. Each retry uses a new attempt folder.
Completed results are bound to the protocol, row-array hashes and request, then
hash checked and reused. To resume after an intentional
stop, remove only this study's `STOP` file and invoke `run` within the original
wall budget. Source/package/input changes fail closed and need a new versioned
study. A process launched independently of the chat can continue after token
exhaustion while the PC remains on. This pilot adds no reboot/startup recovery.
`status` checks process identity; `state.json` alone is not proof of liveness.

## Subsequent experiments, not yet executed by this pilot

1. Build train-only scenario/device summaries and blocked evaluation panels.
   Report unsupported classes explicitly, including single-scenario classes.
   Fit supervised port vocabulary, conflict decisions and selectors within each
   fitting fold for a stronger nested protocol. Preserve the original source
   validation as a later frozen comparison and leave the held-out test locked.
2. Quantify family-specific representation loss: neural clipping/saturation,
   quantile-tail collapse, missingness masks, rare categorical support and
   missing-versus-zero behavior. Test signed log/robust and quantile transforms
   using matched TabM and KAN recipes, keeping the current architectures.
3. Run full-scale XGBoost budget controls with the same selected depth-9 recipe
   and train-only stopping roles. Compare fixed completed round budgets, learning
   curves and class tradeoffs. The recorded runtime truncation is a hypothesis,
   not evidence that extra rounds must improve generalization.
4. Test independent feature groups and justified interactions: rate/timing
   combinations, directional asymmetry, missingness, list-censoring indicators,
   exact ports versus service groups and log/network interactions. Keep signed
   and out-of-range source values until extractor semantics justify changes.
   Refit selectors inside folds. Do not combine all proposed removals at once.
5. Use bounded Optuna TPE conditional searches for the existing XGBoost, KAN,
   TabM variants and HEM-derived adaptation after promising representations
   survive matched comparisons. Include class-weight strategies, capacity,
   regularization and seed repeats; preserve fixed RF as a comparison. This is
   planned work, not a newly running all-model search.
6. Reassess error complementarity before OOF hybrid weighting. Fit calibrators
   on a separate development role, evaluate false-alarm tradeoffs and uncertainty,
   and measure CPU latency/memory for frozen candidates. HEM-derived remains a
   closed-set adaptation, not original HEM reproduction.

Improvement requires consistent matched gains plus macro-F1, class-recall and
false-alarm checks; a single higher accuracy is insufficient. Window dependence
requires block/group-aware uncertainty, not naive independent-row significance.
Live compatibility, publisher-extractor parity and labelled Npcap accuracy are
separate future evidence requirements.

## Method references

- [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html):
  CPU histogram fitting, thread/seed controls, tree depth and feature sampling.
- [scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance):
  fitted-model sensitivity is conditional on model quality; it is not a removal proof.
- [Correlated-feature example](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html):
  correlated predictors complicate individual permutation interpretation.
- [Cross-validation guidance](https://sklearn.org/stable/modules/cross_validation.html):
  group and time dependence require evaluation that respects those structures.
