# Durable live-model training implementation

Goal: train live-screening adaptations of all ten current outputs while the chat
is offline, with persistent Optuna search, bounded retries and actual calibration.
The owner's 25 October request explicitly authorizes this new live study and
supersedes the historical 95-percent/all-model prerequisite. The completed CSV
study, held-out test and original checkpoints stay sealed.

Architecture: independent data/protocol, search, calibration, worker, controller
and supervisor modules under framework/training/live_study. The CLI is
experiment/train_live_models.py. One controller owns Optuna SQLite; one worker
uses the GPU at a time. The watchdog and controller use exclusive OS locks,
process identity records, heartbeat, atomic reports, bounded retry and owned
Windows process jobs. No model output becomes a training label.

Scope: Random Forest, XGBoost, efficient-KAN, HEM-derived, TabM-NE, TabM-Hier,
TabM-Hier-NE; auxiliary TabPack; three existing hybrid combinations. New live
models train on seven directly counted header features for nonempty exact
ten-second windows. No sensor or protocol value is fabricated. The baseline
fixed RF remains preserved; tuning here creates a separate live RF adaptation.

Protocol: within original training data, class-conditional chronological roles
are fitting/tuning/blending/calibration. Overlapping boundary windows and exact
projected duplicates are purged across roles. Preprocessors fit on fitting rows
only. Optuna TPE selects on tuning labels; hybrids fit nonnegative probability
weights on independent blending rows. Temperature/Platt or HEM correctness
calibration fit on calibration rows. Final original-validation scoring follows
frozen selection only, with separate results for unseen seven-count vectors.
It remains reused within-capture source validation, not live accuracy.

Initial budget: 24 wall-clock hours; 16 trials per eight trainable branches,
24 combiner trials per hybrid, winner repeats at seeds43/44, CPU reload/parity,
full provenance and final assessment/export. Round-robin scheduling ensures
all branches start before any branch receives a second trial. Median pruning
uses actual epoch/round metrics for XGBoost, KAN and HEM. TabM/TabPack use early
stopping; RF uses a tree cap and an external timeout. A budget expiry is
reported as incomplete, never completion. No endless failed-trial loop.

Each evaluated trial records raw/calibrated tuning metrics, calibration parameters,
artifact hashes and measured CPU latency. Calibration data is never supplied to
the neural or tree fitting function. HEM retains hard-vote semantics and a
separate correctness estimator, not invented multiclass probabilities.

Manual comparison: generate a reviewable table of genuine CSV windows and
captured Npcap windows, exact feature checks, nearest reference distances,
ambiguous equal-vector labels and model disagreements. This checks compatibility
and plausibility; unlabelled PC traffic cannot establish detection accuracy.

- [x] Prepare and audit train-only partitions with isolation and leakage tests.
- [x] Implement persisted proposals, worker fit/calibrate/reload and retry logic.
- [x] Implement hybrid optimization, seed repeats and sealed final assessment.
- [x] Verify crash/resume, process ownership, STOP, deadline and corrupt results.
- [x] Run one bounded real trial for every branch before unattended launch.
- [x] Launch hidden supervisor, install recoverable logon restart, verify heartbeat.
- [x] Provide exact status/stop/resume paths and progress counts.

Executed evidence: live_loop_smoke_20261025_v2 covers controller crash recovery
and STOP/resume; live_loop_smoke_20261026_v3 completes all eight branch trials,
six hybrid trials, eleven repeated fits/replays and final assessment with the
final process ownership code. The original smoke failure is preserved in v1.
All five demos and 621 evaluation tests pass. See docs/LIVE_TRAINING_LOOP.md.
The full 24-hour study is live_models_20261025_v1; its name retains the original
preparation date. Starting it does not mean its training is complete.

App follow-up: current source records Electron/React/Rust as the owner's corrected
stack; do not reintroduce retired Tauri. Native live export/integration, app
installation and GitHub delivery must be verified against current source after
the study. The training loop does not silently replace the installed app.
