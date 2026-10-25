# TabM Study Implementation Plan

> **For agentic workers:** Execute natively using the accepted three-model design
> and the user's instruction to tune and calibrate. No additional approval gate
> is needed for this already-authorized local work.

**Goal:** Implement, verify and run a reproducible dataset-specific three-TabM study.
**Architecture:** Shared preprocessing and official TabM modules; flat or
hierarchical probability heads; isolated study runner and post-selection calibration.
**Tech Stack:** Python, NumPy, PyTorch, tabm0.0.3, rtdl-num-embeddings0.0.12, sklearn.
**Spec:** `docs/TABM_STUDY.md`.

## Global constraints

- Existing project directories; no upstream/application changes.
- Train-only transforms; frozen validation roles; locked test remains unopened.
- CPU default and optional serialized CUDA; authentic labels and deterministic seeds.
- Preserve failures and metrics; do not claim tuning/calibration before execution.

## Review focus

- Unknown indicators versus absence: assert different mask outputs.
- Constants/repeated bins/all-missing inputs: assert finite transforms and reload.
- Benign-only batches: assert finite hierarchical loss and zero subtype gradients.
- Head/member composition: assert exact hierarchical NLL and probability normalization.
- Selection contamination and process interruption: test tuning-only ranking and retries.

## Tasks

- [x] Add `framework/training/tabm_data.py`: fitted schema-aware transforms,
  numerical bins, explicit masks and train-only profile; test edge cases.
- [x] Add `framework/training/tabm_models.py`: official ensemble backbone,
  numerical input module and flat/hierarchical losses/probabilities; test gradients
  and memberwise composition.
- [x] Add `framework/training/tabm_fit.py`: actual fit, early stopping, weights,
  atomic checkpoint and CPU inference replay; exercise all three models on fixtures.
- [x] Add `experiment/train_tabm.py`: versioned search, role/data pins,
  serialized worker, resume/status/stop and post-selection calibration. Tests
  must show assessment cannot affect selection and calibration stays disjoint.
- [x] Run all five demos, full evaluation and dependency consistency.
- [x] Start explicit CUDA study, verify live state/logs and report measured progress.

No Git repository exists here; additive files and source snapshots preserve evidence.

## Execution ledger

17 October: all549 evaluation tests, five demos and pip dependency check passed.
All three actual architectures performed a CPU optimization step on168 original
training rows with655 inputs; this is software evidence, not model accuracy.
The CUDA controller was launched and its first full480,405-row fit observed.
Tuning/calibration remain ongoing until completed artifacts exist.

Ruling: additive files/source snapshots replace Git worktrees and commits because
this workspace has no Git repository. Existing upstream/application files stay
separate. Cost if wrong: isolation is file/version based rather than branch based.

Ruling: sparse numerical bin edges use nonzero training quantiles, with zero
reserved. Robust scaling has a training-extent floor to retain distinct observed
values. The new regression test failed before the scale fix and passes afterward.
Cost if wrong: sensitivity changes are measured through matched recipe scores.

Final independent review: no Critical/Important findings in the new TabM files.
The reviewer confirmed the older four-family controller misidentifies its own
Windows launcher as competing GPU work. The user then explicitly requested all
seven models train. A recorded controller adapter fixes resource detection
without changing the frozen original study source/model recipes. Two adapter
regression tests were observed failing before implementation and passing after.
Cost of the additive adapter: resume the original four via its documented
compatibility entry point, which records source adoption on subsequent requests.

Declined-to-judge rulings: final model quality and30-fit/calibration completion
remain ongoing and require measured artifacts; the reviewer did not rerun the
whole suite, whose observed execution is recorded in this session.
