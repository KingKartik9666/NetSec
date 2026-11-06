# Broad live-screening adaptation study

28 October 2026. This is a new research study using 252 ordered features:
60 numeric packet/window features and 192 directional port-presence indicators.
It preserves the original full-schema models, the earlier seven-count study,
installed application and locked test. Models remain Python research checkpoints;
native Rust deployment and labelled live accuracy are separate work.

## Selection evidence

`artifacts/training/live_features_20261027_v1/training_profile.json` freezes the
feature order and provenance. The candidate is selected from seven matched
XGBoost ablation arms, each fitted with seeds 42, 43 and 44 on 11,716 fitting rows
and compared on the same 954 tuning rows. All 21 fits completed with CPU reload
verification. Mean accuracy / macro F1 over all 84 classes:

| Representation | Features | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Previous seven counts | 7 | 69.95% | 35.41% |
| Broad numeric profile | 60 | 84.91% | 45.34% |
| Numeric plus port presence | 252 | 90.43% | 54.09% |

The broad candidate beat numeric-only accuracy and macro F1 at every seed.
Zero false positives at the fixed 0.5 binary threshold occurred among only 108
benign tuning windows; this is not a live false-alarm guarantee. Individual
removals of TCP-flags standard deviation, TTL minimum, direction fraction and
MSS gave mixed tradeoffs, so their untested combination is not adopted.
`matched_assessment/RESULTS.md` records every arm and inference-only MSS stress.

Selection is exploratory: the tuning role has historical reuse, and port
vocabulary comes from broader original training. Numeric selection used three
purged folds within fitting rows. No original validation or test was used for
this broad-profile comparison. Original validation is historically reused and
is opened only after each adaptation study's selections and repeats are frozen.

## Architecture and search

The CLI is `experiment/train_broad_live.py`. Small modules under
`framework/training/broad_live` own the new data projection, search settings,
queue, watchdog and assessment. They reuse the established `live_study` workers,
calibration, CPU loaders, atomic writes and Windows process ownership. Existing
training sources were not rewritten to introduce a second feature schema.

The queue targets 400 Optuna proposals, followed by 22 repeat fits/replays:

| Models | Proposals each | Parameters and weights |
| --- | ---: | --- |
| Random Forest | 32 | Tree count/depth, leaf size, feature sampling, class weights |
| XGBoost | 32 | Depth, learning rate, row/column sampling, child weight, L1/L2, class weights |
| KAN | 32 | Width/depth, spline grid/order/regularization, optimizer, scaling, class weights |
| HEM-derived | 32 | MLP widths, optimizer, scaling, PCA dimensions 16–128, KNN k, class weights |
| TabM-NE, TabM-Hier, TabM-Hier-NE | 32 each | Ensemble size, width/depth, dropout, embeddings/bins, optimizer, class-weight power, smoothing, subtype loss where applicable |
| Auxiliary TabPack | 32 | Eight member architectures, optimizer settings, dropout, bins/scaling; fitted member ensemble weights |
| Three hybrids | 48 each | One global blend, then regularized classwise blends with TPE-selected penalty |

Hybrids are TabM + XGBoost, TabM + XGBoost + KAN, and TabPack + XGBoost.
TabPack is an auxiliary branch, leaving ten visible model choices. HEM-derived
remains a closed-set adaptation; this is not a real HEM reproduction.

Optuna uses multivariate/group TPE with eight startup proposals and persisted
SQLite ask/tell trials. Its sampler is deliberately reseeded by proposal count;
SQLite does not serialize sampler RNG. A pending proposal resumes exactly from
its saved request. This is a finite Bayesian search, not enumeration of every
possible parameter/weight combination. Installed versions are sealed in study.json.

Each branch proposal trains a new model from fitting rows. Median pruning applies
to the existing intermediate macro-F1 telemetry for XGBoost, KAN and HEM-derived;
the other neural trainers retain early stopping and explicit runtime limits.
Optuna chooses winners by tuning macro F1; checkpoint early stopping uses tuning
accuracy. TabPack member-weight fitting is distinct from loss class weighting.

Blends fit on the independent blend role using SLSQP / L-BFGS-B, not OOF stacking.
Temperature/sigmoid confidence calibration and alert-threshold selection use the
existing separate calibration/tuning roles. Each evaluated fit writes calibration,
per-class/family scores, false alarms, confidence error, checkpoint hashes, CPU
reload/replay checks and cold single-window inference latency. Calibration may
fail to improve a metric; actual outputs are retained without claiming otherwise.

After search, selected branch configurations are refitted at seeds 43 and 44;
frozen hybrid weights are replayed on those independently refitted branches.
Final assessment records full and conservative seven-count-novel CSV subsets,
all classes, seed results, floating-point feature values and missing cells. The
novel subset is explicitly the historical seven-count mask, not new 252-feature
novelty. It is not an unknown-attack test.

## Data and execution boundaries

Preserved original-training roles are fit 11,716 / tune 954 / blend 945 /
calibration 1,637, with 5,436 purged windows. All 84 classes have fitting support.
Joins require unique row IDs and identical labels and ten-second window bounds.
Unknown values remain NaN; measured zero and signed/fractional values survive.
No rare-port frequency threshold or arbitrary top-k cut removes features.
Source hashes are verified during preparation; each worker verifies frozen arrays,
small provenance pins, code and library versions. No test-array loading path exists
in this study. Predictions never become training labels.

Default application/demo behavior remains offline, CPU-only and credential-free.
This research queue requires explicit `--cuda` for fitting; RF, blend optimization,
calibration and inference checks use CPU. Only one owned fitting worker runs at a
time. It needs no network or paid service and does not start packet capture.

## Start, inspect, stop and resume

Run from the project root in PowerShell. The main study is initialized once:

```powershell
.venv/Scripts/python.exe experiment/train_broad_live.py init --cuda --hours 24
.venv/Scripts/python.exe experiment/train_broad_live.py install-logon
& experiment/start_broad_training.ps1
.venv/Scripts/python.exe experiment/train_broad_live.py status
```

The hidden supervisor is independent of the chat and token budget. It writes
`state.json`, `supervisor.json`, `controller.log`, `optuna.sqlite3` and per-attempt
worker logs below `artifacts/training/broad_live_models_20261028_v2`. State includes
process identities, heartbeat, current trial, counts and current best tuning scores.
Use process creation times and recent heartbeats, not merely an old PID, to check
liveness. `completion.json` plus verified assessment/export hashes establish
completion; an initialized directory does not.

```powershell
.venv/Scripts/python.exe experiment/train_broad_live.py stop
# After status reports stopped and the owned processes have exited:
& experiment/start_broad_training.ps1 -Resume
# Remove only this study's logon recovery entry if it is no longer wanted:
.venv/Scripts/python.exe experiment/train_broad_live.py remove-logon
```

There is a 24-hour wall-clock cap from the first controller start. Each fit has
bounded runtime, at most two failed attempts, and retained logs. The watchdog has
bounded restart attempts; source/version mismatch and hard failures require
attention instead of silent substitution. STOP terminates owned work; resume
refits an interrupted proposal with the same configuration without resetting the
deadline. Completed trials are reused. Research results remain on disk at a stop
or budget expiry, even if the complete 400-proposal plan has not finished.

The supervisor requests Windows stay awake while working. The PC still needs power;
training cannot run while it is off. Current-user logon recovery uses a scoped
hidden Startup entry, obeys STOP/deadline, and removes itself on completion.
No setting extends an expired budget or changes a sealed study automatically.

## Verification and outstanding research

`broad_live_smoke_20261028_v1` completed eight branch trials, six hybrid trials,
eleven repeats/replays and final CPU CSV assessment, with no failed trials.
Its stop/resume proof verifies exact pending proposals, unchanged deadline and
successful completion. Short smoke fits verify the software; they are not the
final tuned results. `broad_live_verification_20261028` contains fresh full-suite,
demo and native check logs.

Rust capture now owns both window rotation and the closing Npcap drop-statistics
sample. Its regression test checks late loss and counter reset. Independent
offline replay passes fifteen synthetic cases and one bounded real-PCAP window,
covering 79 numeric/derived fields and 192 port indicators. Fifteen original CSV
examples supply 3,900 compared cells. These checks do not establish publisher
extractor equivalence, hardware timestamp accuracy or labelled live detection.
Capture windows are receipt-based and driver telemetry can have limitations.
Paired original captures, a reference exporter and independent labels remain
necessary before claiming deployment accuracy. See the feature audit and live
accuracy research notes for those boundaries and future flow-feature work.
