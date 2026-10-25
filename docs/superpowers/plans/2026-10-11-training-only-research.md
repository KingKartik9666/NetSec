# Training-only research implementation plan

Accepted specification: the user's Training-only study toward repeat-validated 95% accuracy plan (11 October 2026).

Goal: at least one full-CSV 84-category family reaches >=95% validation accuracy in seeds 42/43/44 with verified CPU reload. XGBoost, efficient-KAN and upstream-derived HEM are optimized; conventional RF is fixed. No overall deadline; isolated per-fit watchdogs remain. Data, splits, originals and the held-out test are preserved.

- [x] Train-only transformation and constrained augmentation pipeline; synthetic provenance and raw-schema reload.
- [x] Solo/pair recipes, duplicate-safe fingerprinting, Optuna tuning, capacity escalation and repeat gate.
- [x] Neural loss/scheduler/pruning controls with unchanged defaults and numerical tests.
- [x] Detached open-ended controller, recovery, study transition, calibration/ensemble evaluation.
- [x] CUDA smoke, complete evaluation, default/four-sample demos and independent final review.
- [x] Launch verified study and write concrete resume instructions.

All code stays in existing project directories; optional research dependencies never enter default app imports. No app/UI/installer modifications. Progress ledger: artifacts/training/csv_research_v3/implementation-ledger.md.

Implementation verified:459 evaluation tests and five demos passed; one fresh review's
three important findings fixed with regression tests. Actual CUDA CTGAN/ensemble/raw
pipeline checks passed. Study launched and GPU progress observed on original real rows.
The open-ended training objective remains ongoing:95% repeat-validated success is not
achieved by implementing or launching the controller. See docs/RESUME_INSTRUCTIONS.md.
