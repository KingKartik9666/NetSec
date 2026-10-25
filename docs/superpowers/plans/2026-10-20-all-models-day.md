# All-model 24-hour execution ledger

User authorization: execute the agreed all-model optimization plan and keep it
running independently of chat context/token limits.

Scope: additive modules under experiment/ and framework/training/, focused
evaluation tests, this plan and docs/ALL_MODELS_DAY.md, new study artifacts.
Protected: datasets, upstream repositories, installed app, held-out test,
historical checkpoints/results and sealed source files.

Isolation: git root/branch/worktree/status all report no Git repository. This
workspace already uses additive modules and source snapshots. No existing source
is edited; original source pins stay intact. New paths are checked before use.

Tasks:
1. Reproduce Windows replacement sharing failure; implement bounded atomic JSON
   retries in an additive module and a versioned recovery entry point.
2. Implement persistent multivariate TPE policy, elapsed-time budgets, seven
   tunable branches, fixed RF, role separation and bounded recovery.
3. Implement training, pruning, selected repeats, OOF hybrids, calibration,
   CPU inference verification and final reports.
4. Run focused tests, genuine data smoke fits, all evaluation tests/five demos,
   independent review; launch a detached supervisor and verify live progress.

Ruling: a detached Python supervisor, rather than a token-consuming chat loop,
owns the full run. It cannot run while Windows is off/asleep; it requests awake
execution only while active and does not change persistent power settings.

Budget: deadline is fixed at first start, never extended by restart. Search ends
early enough to reserve repeats, OOF fits and final reporting. Recovery of the
previous search is capped; any incompletion is explicitly reported.

Progress:
- Real Windows sharing-lock reproduction failed in the original writer with
  WinError 5. New bounded atomic writer passed the same reproduction.
- The additive recovery adapter resumed the final TabPack trial in attempt_001;
  source/recipe pins remain unchanged.
- Fifteen focused tests pass, including persistent TPE requests, actual Optuna
  pruning, strict OOF disjointness, parameter-tamper rejection, per-tag HEM
  calibration, explicit CPU neighbors, failure reporting and deadline guards.
- Four genuine-row CPU smoke fits (333 real fitting rows, 655 columns, original
  84-class vocabulary) passed two epochs and checkpoint replay. These are not
  full training or research performance results.
- Earlier full verification: 592 passed, 13 warnings; five demos exit 0. A fresh
  final suite is required after final review fixes.

Independent review: one read-only reviewer. Important findings addressed:
missing OOF run paths; strict-arm mapping; branch exports needing explicit full
validation artifact locations; HEM KNN device default; calibration tag/hash
collisions; unpinned HEM/efficient-KAN sources/packages; omitted export failures
and initial hybrid baselines. Relevant focused tests passed after fixes.

Ruling: plain TabM-Hier's embedding dimensions remain inert compatibility fields
in its configuration. Its model explicitly bypasses embeddings. Cost: a small
amount of redundant search; no change to model semantics or research labels.

Next / upcoming task: final integrated checks, initialize/pin, then launch and
verify the detached supervisor and live recovery progress.

## Pause and explicit resume — 21 October 2026

All project workers/controllers/watchers were stopped at the user's request;
pause_request.json and pause_verified.json preserve the active job and evidence.
On "resume everything", the old deadline had elapsed while paused. Added
experiment/resume_all_models_day.py and evaluation/test_day_resume.py, preserving
the original sealed source/protocol and restoring only unused active time in a
new continuation directory. Frozen selections and source/data/role pins carry
forward. No historical/cancelled studies are restarted.

Proof: 18 focused tests pass; full evaluation reports 602 passed, 17 warnings;
all five demos exit 0. Independent read-only reviewer day_study_review found no
blocker for the actual paused state. Its limitation (prior successful repeat/OOF
imports not consumed) does not apply: no completed repeats or OOF registry existed.

Checkpoint: implementation -> change-review completed; no outstanding blocker.
Continuation action: detached research execution, with live process/state checks.
Source and original checkpoints preserved; no UI/deployment/installed-app change.
Next / upcoming task: background seed repeats, OOF hybrids, calibration and CPU
verification under the continuation deadline; report completion or explicit gaps.
