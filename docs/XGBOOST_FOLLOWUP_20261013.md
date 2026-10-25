# XGBoost replication, feature ablation and calibration

Authorized by the user's "do it" after the recommendation to replicate the
cleaned/enriched experiment, separate feature contributions, inspect errors,
tune the strongest representation and calibrate on training predictions.

## Frozen protocol

- Input: the preserved `xgboost_data_audit_20261012_v1/prepared` training and
  validation arrays. Every validation observation and label stays unchanged.
  The test remains locked. The newer `deep_clean_v2` study is a separate
  representation/recipe comparison, not a matched control for this experiment.
- CPU histogram XGBoost, four threads, seeds 42/43/44. Retain the original
  79-round, depth-six, 0.05 label-smoothed recipe for representation comparisons.
- Five arms: original 170 inputs/all training rows; filtered 170 inputs;
  filtered plus ports; filtered plus measured duration/rates; filtered plus
  both. Filtering means the earlier explicit all-zero-input ablation, not a
  claim that quiet source windows or their labels are corrupt.
- Reuse the three completed seed-42 original/filtered/combined fits only after
  checking their requests, checkpoints and saved predictions. Twelve new fits
  complete the 15-cell comparison. Report each seed, mean, standard deviation
  and paired changes. Seeds share one validation set; they are not independent
  data replications or confidence intervals for unseen captures.
- Select the highest mean validation accuracy among arms whose per-seed macro
  recall does not decline by more than 0.01 and benign false-positive rate does
  not rise by more than 0.005 against the original control. Break ties by macro
  F1, then fewer features. Report all tradeoffs.
- Tune that arm with three seed-42, single-factor trials: depth 8; 160 rounds;
  zero label smoothing. All other settings remain fixed. If the best eligible
  trial improves seed-42 accuracy, repeat it at seeds 43 and 44. Retain it only
  if mean accuracy improves and the same guards hold against the base recipe
  in every seed. No additional adaptive search is authorized by this protocol.
- Cross-fit the selected seed-42 recipe in three training-only temporal blocks
  within each class. Purge fitting windows that touch the held block and fitting
  rows with an original input hash in that block. This is within-capture,
  class-conditional cross-fitting, not a future/external-capture evaluation.
  The label-independent feature vocabulary is frozen from the original
  training representation; it is not refitted per fold. Fit one positive scalar
  temperature on the held-fold predictions. Validation labels never fit the
  calibrator. Compare NLL, multiclass Brier, confidence ECE and unchanged class
  metrics on all validation rows. Keep worsening calibration results visible.
- Export the selected seed-42 checkpoint, feature schema/transform, recipe,
  temperature and hashes as a frozen research candidate. Do not promote it to
  the app or score the locked test. Training-only calibration after validation
  recipe selection remains exploratory, not an untouched test assessment.

## Execution and evidence

Implementation: `experiment/xgboost_followup.py` (data, fitting, calibration
boundaries) and `experiment/run_xgboost_followup.py` (resumable bounded queue,
comparison, errors and frozen candidate). Results and atomic `state.json` live
under `dataset/experiments/xgboost_followup_20261013_v1/`.

Only one follow-up fit runs at a time. The runner checks available Windows
commit before each fit, records waiting/resource/failure states, and preserves
interrupted attempts. Existing GPU work is left under its own controller.
Completed results and source/input hashes are verified on resume.

Focused tests cover feature isolation, overlap/duplicate purging, seed handling,
saved-model replay, calibration argmax invariance and input/result integrity.
After changes run the default demo, all four sample options and the complete
`python -m pytest evaluation -v` suite in the project environment.
