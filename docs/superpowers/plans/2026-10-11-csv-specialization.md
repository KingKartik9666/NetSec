# CSV specialization implementation plan

Goal: train and improve XGBoost, genuine KAN and GitHub-derived HEM exclusively
on complete CIC IIoT2025/DataSense CSV observations. The user cancelled mixed
CSV/live training and raised the three-model target to 96% validation accuracy.

Design: retain the existing row partitions, labels, protocol vocabulary and
original artifacts. A separately versioned CSV controller reuses bounded process
execution, integrity checks and Windows recovery. All three candidates receive
the existing 170-column representation; fixed conventional RF remains unchanged
on its original 71-column representation. Neither masked views nor live-profile
selection participates in this study. Naturally missing dataset values remain
missing until each model's training-fitted preprocessing handles them.

Select every candidate by full-input accuracy, then macro F1. Independent trials
compare native/smoothed XGBoost loss, weighting and capacity; neural trials compare
scaling, learning-rate schedules, capacity and HEM PCA. Follow independently
improving changes with combinations and bounded, reproducible local search.
Repeat leaders and matched CSV controls with seeds 43/44 before eligibility for
promotion. Record per-class support, false alarms, confusion matrices, and
supported top-three metrics. Do not claim any score before measured execution.

Constraints: original wall-clock deadline 2026-10-12 08:24:23 UTC persists;
maximum 20 minutes per fit, sequential CUDA, test locked, CPU offline reload.
96% is a target, not a guarantee or permission to extend the runtime indefinitely.
Live deployment remains a later separately trained capability; the previous
all-four 95% gate and independent-label requirement were not revoked. No model
promotion or installed-app replacement occurs automatically.

- [ ] Preserve the stopped mixed study and source, then test the CSV policy,
      complete-row loading, training-only transformations and accuracy selection.
- [ ] Implement dedicated full-input XGBoost training and configurable neural
      preprocessing/selection while preserving historical defaults.
- [ ] Implement a versioned CSV controller, resume/deadline/stop behavior,
      immutable requests and CPU checkpoint verification. Reuse lifecycle code.
- [ ] Test the actual CUDA fits, reloads, isolation and full suite; run default
      demo and all four samples. Obtain one independent final code review.
- [ ] Launch detached CSV training after validation; verify actual progress,
      update project context and exact resume commands.

Implementation decisions: work inline in the existing non-Git workspace;
preserve source snapshots instead of commits/worktrees. The user's explicit
implementation instruction and existing study authorization cover this bounded
change; no additional design approval is needed.


### Adaptive extension verification

Implemented per-round/epoch metric snapshots, deterministic plateau proposals and immutable decision logging. Independent review identified combination-request instability on resume and incomplete repeats incorrectly marked completed; both were reproduced as failing regression tests, then fixed. Targeted checks: 41 passed. Complete evaluation: 365 passed, one existing joblib warning. Source snapshots and the completed v1 XGBoost control are preserved; adaptive v2 adopts verified controls and retains the original deadline.
