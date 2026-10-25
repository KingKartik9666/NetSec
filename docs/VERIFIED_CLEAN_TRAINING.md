# Verified cleaned DataSense four-model study — 16 October 2026

User authorization: start all four models on verified cleaned v2, retune and
recalibrate, targeting 90% accuracy. This separately authorizes RF tuning for this
study; the earlier fixed RF baseline and all previous studies stay intact.

Data: `dataset/experiments/datasense_verified_clean_20261014_v2/model/`.
Full training has 480,405 rows, strict training 384,012, validation 30,000;
all use the ordered 655 features and original labels. No test array is loaded.
The dataset is DataSense-format, with unverified publisher byte identity.

The frozen initial search has 24 fits: four families, two training arms, three
recipes each, seed 42. Families are sklearn RF (CPU), XGBoost (CUDA), genuine
efficient-KAN (CUDA), and upstream-derived closed-set HEM (CUDA). The selected
recipe per family repeats with seeds 43/44: 32 planned fits total. Architecture,
weights and scaling vary in named recipes; no synthetic observations are added.
This is an initial bounded tuning campaign, not a guarantee of 90%. Its maximum
per-fit watchdog is two hours; neural/boosted fitting has a 55-minute internal
budget plus inference/reload overhead. Failures are explicit and preserved.

Validation roles are frozen before fitting: chronological earlier half within
class tunes/checkpoints; the later half is divided again for calibration and
later assessment. Touching/crossing windows and duplicate vectors crossing roles
are purged. Extremely short classes may be assessment-only. All 30,000 rows
remain available for a separately labelled full-validation report. No
calibration/assessment result chooses a model. Prior reuse of this validation
set means even the separate assessment is exploratory, not a fresh final test.

Selection maximizes tuning-role accuracy, with macro F1 as tie-break. Report
full-validation accuracy, per-class precision/recall/F1, false-positive and
missed-attack rates, later assessment accuracy and calibration quality. The 90%
target is evaluated for each family across seeds 42/43/44, without changing
labels or hiding unsupported classes. A family that misses 90% remains a miss.

RF/XGBoost/KAN class probabilities receive scalar temperature calibration.
All four receive regularized correctness-confidence calibration. HEM's hard
vote has no native multiclass ensemble probability distribution: its confidence
means estimated correctness of the unchanged vote, derived from component
scores/agreement. Calibration never changes category decisions. Report both
improvements and regressions; do not confuse confidence with accuracy.

Consumed data and source files are hash-pinned, dataset verification hashes are
checked, checkpoint CPU predictions are replayed (all validation for RF/XGB/KAN,
128 rows for HEM), and artifacts are hashed. HEM nearest neighbors use authentic
training observations only. The strict subset changes training distribution;
its exclusions are not proof of incorrect source labels.

Output: `artifacts/training/verified_clean_four_models_20261016_v1/`.
Only new study files, this document, `experiment/train_verified_clean.py`, and
`evaluation/test_verified_clean_training.py` are changed. There is no Git
repository in this workspace; additive filenames provide isolation. No app,
upstream, original source, old controller, dataset, or installed package changes.

The existing study may keep training. New CUDA work acquires the same legacy
controller locks and waits without interrupting active work. CPU RF can start
when at least 4 GiB physical and 6 GiB commit headroom are available. Only one
worker runs within the new study. No overall deadline or automatic deployment.

```powershell
.venv/Scripts/python.exe -B -m experiment.train_verified_clean start
.venv/Scripts/python.exe -B -m experiment.train_verified_clean status
.venv/Scripts/python.exe -B -m experiment.train_verified_clean stop
.venv/Scripts/python.exe -B -m experiment.train_verified_clean resume
```

Stop interrupts the owned worker and preserves outputs. Interrupted/orphaned
attempts without a failure record resume in a new directory (maximum three
attempts); completed/failed attempts are never overwritten. The detached runner
survives closing the chat, but reboot requires the explicit resume command.

Verification: initial safety tests were observed failing because this runner
did not exist; subsequent tests exercise temporal separation, duplicate
ownership, tuning-only selection, four-family/arm coverage, prohibited test
arms, and an actual small CPU RF fit/checkpoint reload/calibration. The small
fixture establishes software behavior only, never research performance.
