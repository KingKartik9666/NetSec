# Hybrid study implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan inline; one final independent review.

**Goal:** Build and execute the three requested TabM/XGBoost/KAN/TabPack hybrids.
**Architecture:** Share leakage-controlled OOF branch fits, learn probability
combiners, then calibrate and export CPU-reloadable bundles. Run serial workers
in an additive study with frozen inputs and preserved earlier trials.
**Tech stack:** Existing NumPy/SciPy/PyTorch/XGBoost/efficient-KAN/TabM, isolated
upstream TabPack sources; no default-app dependency changes.
**Spec:** `docs/HYBRID_STUDY.md`.

## Global constraints

- Exact verified v2 schema, 655 inputs and 84 classes; test remains locked.
- Preprocessing and OOF combiner fitting use training rows only.
- Calibration and assessment roles do not select models or combiners.
- Original studies and checkpoints remain unchanged.
- Default application execution stays offline, CPU-only and credential-free.

## Review focus

- Duplicate windows and strict-arm filtering must not introduce OOF leakage.
- Invalid/misaligned probabilities and schemas must fail explicitly.
- Failed workers and resumes must preserve attempts and release GPU locks.
- TabPack compatibility must execute real upstream arithmetic.
- Exported bundle replay must match the probabilities used for reported scores.

### Task 1: Fold and combiner contracts

Create `framework/training/hybrid_data.py`, `hybrid_blend.py` and
`evaluation/test_hybrid_study.py`.
- [ ] Test purging/temporal separation, duplicate ownership, invalid inputs,
  zero-weight rejection of a bad branch, class-specific blending and replay.
- [ ] Implement `forward_folds(labels, starts, ends, hashes)` and
  `fit_blend(probabilities, labels, names, penalty=None)` / `blend(...)`.
- [ ] Execute the focused tests.

### Task 2: Neural and upstream TabPack adapter

Create `framework/training/tabpack_backend.py` and `hybrid_neural.py`.
- [ ] Test actual upstream packed forward/backward, independent members,
  checkpoint replay, invalid source pins and TabM replay diagnostics.
- [ ] Implement pinned isolated loading, fixed heterogeneous pack config,
  independent member loss, online greedy selection and bounded training.
- [ ] Execute actual 655-input/84-class CPU and CUDA smoke checks.

### Task 3: Study orchestration and deployable research bundles

Create `experiment/train_hybrids.py`, `framework/training/hybrid_bundle.py`
and focused integration tests.
- [ ] Freeze datasets, recipes, packages, folds and source snapshots.
- [ ] Implement worker/controller/status/stop/resume, common GPU locking,
  CPU validation inference and learned/calibrated composite export.
- [ ] Verify input/source changes fail and incomplete branches cannot claim
  a complete hybrid; verify bundle inference with authentic validation rows.

### Task 4: Verify and execute

- [ ] Run `python -B -m pytest evaluation -v` and all five demo commands.
- [ ] Obtain one fresh-context implementation review; fix material findings.
- [ ] Start the authorized serial CUDA experiment and verify actual progress.
- [ ] Record command outcomes, actual counts, limitations and current status.

## Execution ledger

Context and current source inspected. No project Git repository exists; use
additive files and source snapshots instead of commits/worktrees. User's explicit
"make this" authorizes implementation of the discussed design. Continue without
repeating design approval. This costs no changes to the historical studies.

18 October 2026 execution record:
- Tasks 1–3 implemented; 22 focused tests pass, including seven regression
  failures observed before fixing final-review findings.
- Actual 655-input/84-class TabM and upstream TabPack CPU/CUDA smoke training
  and same-device checkpoint checks passed. These are software checks only.
- Final review found three important issues: early completion publication,
  incomplete ordered-schema binding, and launcher/controller lock racing.
  Fixed all three with failure-path and permutation/startup regression tests.
- No minor findings deferred. Failed-fit policy now states that resume skips
  failures for diagnosis; it does not claim an unimplemented retry command.
- Frozen study initialized; final full suite, demos and raw CPU composite
  replays are recorded in the study's `verification/` directory.

Memory-constrained execution revision (18 October):
- The initial controller waited for 4 GiB free RAM and fitted zero jobs; stopped
  it and preserved its immutable protocol and source snapshots.
- New active version: `hybrids_verified_v2_20261018_v2`. Neural row/column access
  now indexes the original memory map without copying a full fitting fold.
  Controller metrics avoid importing PyTorch. Same observations and recipes.
- Two new tests first failed for the absent indexed-row adapter, then passed
  with exact dense/indexed preprocessor state and batch parity. All 24 focused
  tests pass; actual 655-input/84-class indexed CUDA smoke fits passed for both
  neural architectures. Five demos pass for this revision.
- Neural physical headroom is 2.5 GiB, other families 4 GiB, commit 6 GiB.
  Eligible neural jobs may run while larger jobs wait for memory. The frozen
  v2 protocol records this policy; the initial v1 protocol was not rewritten.
