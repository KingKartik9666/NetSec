# Live model training loop

The 25 October 2026 study is separate from the completed full-CSV study. Its
supervisor, controller and worker are ordinary local Python processes. They do
not need this chat, an API key, internet access or more Codex tokens to continue.
GPU fitting is explicitly authorized for this study; application inference and
the CLI's default status operation stay offline and CPU-only.

## Run and inspect

Use the project PowerShell environment from `K:\IMDAI Project`:

```powershell
# Initialization seals the data, source files, library versions and budget.
.venv/Scripts/python.exe experiment/train_live_models.py init --hours 24
.venv/Scripts/python.exe experiment/train_live_models.py install-logon
./experiment/start_live_training.ps1

.venv/Scripts/python.exe experiment/train_live_models.py status
.venv/Scripts/python.exe experiment/train_live_models.py stop
./experiment/start_live_training.ps1 -Resume
```

Initialization is idempotent only when the sealed settings still match. Resume
retains the original deadline. A completed or exhausted study is preserved;
another budget requires an explicitly named new revision. Pass `--study` to the
Python CLI, or `-Study` to the PowerShell launcher, for an alternate revision.
An intentional source or library change requires a new revision as well.

The current-user Startup entry resumes at the next Windows logon. It obeys STOP
and the saved deadline, needs no administrator access and removes itself after
successful completion. `remove-logon` removes only this study's matching entry.
The watchdog requests that Windows stay awake while training runs; it does not
change power settings or keep the display on. Explicit sleep, shutdown and loss
of power still interrupt computation. The original wall-clock budget continues.

The study directory contains:

| Artifact | Meaning |
|---|---|
| `study.json` | Immutable scope, budget, schema, code hashes and package versions |
| `data/manifest.json` | Accepted windows, purges, class support, source/array hashes |
| `data/csv_window_review.csv` | Real source-window examples, features and conflicting labels |
| `optuna.sqlite3` | Persisted TPE proposals, parameters, scores and trial states |
| `state.json` | Atomic progress, current owned worker and controller heartbeat |
| `supervisor.json` | Watchdog liveness and recovery count |
| `runs/<family>/trialNNNN/attemptNN/` | Request, checkpoint, calibration, predictions, metrics and log |
| `selected.json` | Frozen selected recipes, before final source validation |
| `repeats/` | Actual seeded branch refits and frozen-weight hybrid replays |
| `assessment/` | Final original-validation metrics and row-by-row CSV predictions |
| `exports/manifest.json` | CPU checkpoint inventory; native export still required |
| `completion.json` | Present only after all required phases succeed |

`state.json` alone is not proof of a live process. The status command checks PID
creation times and reports heartbeat age. `needs_attention`, `budget_exhausted`
and `insufficient_disk_space` are explicit incomplete outcomes.

## Training and selection

Eight independently trainable branches feed ten visible outputs:

| Group | Models |
|---|---|
| Original families, new live adaptations | Random Forest, XGBoost, efficient-KAN, HEM-derived |
| TabM | TabM-NE, TabM-Hier, TabM-Hier-NE |
| Auxiliary branch | TabPack, eight packed members |
| Hybrids | TabM + XGBoost; TabM + XGBoost + KAN; TabPack + XGBoost |

The planned search is 16 proposals per branch and 24 per hybrid: 200 Optuna
trials, followed by seeds 43 and 44 for all eight branches and three hybrid
replays per seed. One worker runs at a time. Round-robin scheduling gives every
branch its first trial before starting second trials. The 24-hour cap can end
the queue before all proposals finish; counts distinguish completed, pruned and
failed trials. Automatic retries use new attempt directories and are bounded.

Techniques actually implemented:

- Seeded Optuna TPE Bayesian search with conditional model-specific spaces and
  SQLite persistence. Proposals survive controller restarts.
- Train-only weighting, learning-rate/regularization search, tree depth/leaf
  search, KAN grid/width/depth search, TabM ensemble/embedding search and TabPack
  member configuration search.
- Early stopping and checkpoint selection on the tuning role for iterative
  trainers. Median pruning uses real XGBoost rounds and KAN/HEM epochs. TabM and
  TabPack currently use early stopping without Optuna pruning. RF has fixed tree
  caps and an external timeout.
- Global nonnegative probability weights fitted by constrained SLSQP on a
  separate blending role. Class-specific weights use L-BFGS-B with shrinkage;
  classes with fewer than ten blending observations retain global weights.
- Temperature scaling and monotonic Platt calibration after each evaluated
  trial, using a separate calibration role. HEM keeps its actual hard vote and
  fits a separate vote-correctness estimator. It has no invented ensemble
  probability distribution.
- A raw-probability alert threshold selected on tuning data for a 2% false-alarm
  cap. This is a tuning constraint, not a guarantee on later traffic. Classification
  argmax/vote and the binary screening policy are reported separately.
- Two independent CPU checkpoint reloads, deterministic replay, per-trial cold
  inference latency, preserved artifact hashes and final repeated-seed assessment.

TPE ranks tuning macro F1 across all 84 declared classes. Iterative checkpoint
early stopping ranks tuning accuracy with a macro-F1 tie-break. Calibration
does not change class predictions. All trial metrics label their role; final
source-validation metrics never feed TPE, calibration or repeat configuration.

## Data and deployment boundary

Inputs are exactly seven packet-header counts: total packets and TCP ACK, FIN,
PSH, RST, SYN and URG counts in a ten-second window. A SYN+ACK increments both
counts. Flags cannot exceed packet count; negative, fractional, nonfinite and
zero-traffic windows are rejected. No absent sensor reading is invented.

The source contains 20,966 exact ten-second training windows; 20,688 pass the
count checks. Within each source class, chronological fitting/tuning/blending/
calibration roles use 60/15/10/15 percent boundaries. Overlapping boundary windows
and exact seven-count vectors crossing roles are purged. Sparse classes with
fewer than 30 start times remain in fitting only. The resulting roles contain
11,716 / 954 / 945 / 1,637 rows. All 84 classes occur in fitting; 32 / 29 / 27
classes lack tuning / blending / calibration support and are explicitly listed.

There are 256 training vectors shared by conflicting labels. Their empirical
training-sample exact-label ceiling is about 92.04%; that statistic is neither a
model score nor a future accuracy bound. It demonstrates why seven counts cannot
reliably distinguish every subtype. The final 1,571 source-validation windows
are evaluated only after selection, including a separate analysis of vectors
absent from all accepted original training rows. This remains reused,
within-capture validation. The original held-out test is never loaded.

The output bundles are live-feature research candidates. Matching packet counts
and CSV rows, CPU reload, and calibrated source scores do not establish live
detection accuracy. The original CSV extractor's window/device semantics still
need comparison with Npcap, together with real captured windows, disagreements
and distribution shifts. Unlabelled desktop traffic supplies no attack labels.
Native Rust export, current-app integration and promotion require that evidence;
the loop never replaces installed models automatically.

## Recovery contracts

The watchdog and controller each hold an OS lock. The watchdog assigns its
controller to an owned Windows Job Object before allowing it to start workers;
closing that job kills its descendants. Worker cancellation also stops Python
redirector children. Completed atomic results are reconciled with the pending
Optuna trial after a crash. Interrupted fits restart the same proposal in a new
attempt, rather than claiming mid-epoch optimizer-state resumption. Previously
completed checkpoints are reused, not retrained because the chat restarted.

The controller owns Optuna ask/tell and liveness. Optuna's built-in heartbeat
does not protect ask/tell trials; the watchdog does that explicitly. It attempts
at most three controller recoveries and two failed attempts per proposal. Code,
schema or checkpoint corruption fails closed. Logs and failed attempts remain
available for diagnosis.

Primary references: [Optuna RDBStorage](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.RDBStorage.html)
and [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html).
