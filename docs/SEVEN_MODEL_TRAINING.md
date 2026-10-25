# Seven-model training on verified cleaned DataSense v2

Execution requested17 October2026: train/tune/calibrate all seven families on
the latest verified cleaned dataset. The original study resumes its preserved
progress; the new TabM study adds three architectures.

| Study | Families | Planned fits |
|---|---|---:|
| verified_clean_four_models_20261016_v1 | RF, XGBoost, genuine efficient-KAN, HEM-derived closed-set adaptation | 32 |
| tabm_verified_v2_20261016_v1 | Flat TabM with numerical embeddings; hierarchical TabM; hierarchical TabM with numerical embeddings | 30 |

Both consume the exact same corrected v2 manifest and ordered655 columns,
84 original classes,480,405 full-training rows,384,012 optional strict-training
rows, and30,000 validation rows. Model selection uses the same frozen tuning
role. Calibration and later assessment are disjoint. The held-out test remains
locked. Previously reused validation makes these exploratory comparisons.
Architecture/representation/recipe changes are labelled comparisons, not proof
that any single component caused a gain or that95% is achievable.

All six GPU families share the three existing controller locks. They take
turns on the RTX4050. CPU RF may coexist with a GPU fit only with at least4GiB
physical and6GiB commit headroom. Completed attempts remain preserved.

## Windows controller compatibility

`experiment/train_original_four_compat.py` resumes the original four-family
controller with an explicit process-detection adapter. The prior detector
mistook its own `.venv` launcher for a competing CUDA trainer. The adapter
recognizes actual `--fit` workers; model recipes and the original controller
source remain unchanged. Adoption hashes and source snapshots are recorded in
the original study's `controller_adapters/` and `source_snapshot/` directories.
New requests record the adapter identity. No historical result is rewritten.

```powershell
# Start/resume the original four; uses the recorded Windows adapter.
.venv/Scripts/python.exe -B -m experiment.train_original_four_compat resume

# Start/resume the three new models, using the pinned explicit CUDA choice.
.venv/Scripts/python.exe -B -m experiment.train_tabm resume

# Current combined status (read-only).
.venv/Scripts/python.exe -B -m experiment.seven_model_status

# Stop only the requested study, retaining completed evidence.
.venv/Scripts/python.exe -B -m experiment.train_original_four_compat stop
.venv/Scripts/python.exe -B -m experiment.train_tabm stop
```

The two controllers survive closing the chat; after a reboot use the resume
commands above. Calibration runs automatically after recipe selection.
See `docs/VERIFIED_CLEAN_TRAINING.md` and `docs/TABM_STUDY.md` for their frozen
search budgets, safeguards, confidence meanings and limitations. Default
application/demo execution stays offline, CPU-only and credential-free.
