# Current model handoff: completed (24 October 2026)

The ten-model frozen study is complete. Do not resume an old controller as part
of this wrap-up. Current evidence:
`artifacts/training/all_models_wrapup_20261023_v1/CLOSEOUT.md`,
`FINAL_REPORT.md`, `FINAL_METRICS.json`, `completion.json`, and
`verification/final_closeout/manifest.json`. All eight OOF branches, fourteen
branch-repeat assessments and ten CPU bundles completed. No worker is active.
Highest full reused-validation accuracy is 81.5033% for TabM + XGBoost + KAN;
assessment accuracy is 79.9170%. Held-out test remains locked, and no app model
was replaced. Read CLOSEOUT.md for mixed calibration, unequal tuning budgets,
TabPack repeat-seed limitations, alert-policy FPR and CPU timing caveats.

The following section is preserved historical context for a different stopped
controller; its old goal and next steps are not active wrap-up instructions.

# Historical open-ended full-CSV research handoff

Optimize XGBoost, efficient-KAN and HEM-derived closed-set models. Conventional RF fixed.
Target: one family >=95% full 84-category validation accuracy in seeds42/43/44.
No overall deadline. Per-fit watchdogs:20min/1h/4h. CUDA fits sequential; CPU reload verified.
Real training/validation partitions unchanged; no locked-test loading, live training or app promotion.
Status:stopped; phase:solo_and_pairs; PID:10116
Completed fits:{'xgboost': 62, 'hem': 86, 'kan': 82}
Active:null

Commands from the project root:
```powershell
.venv/Scripts/python.exe -B -m experiment.optimize_csv_research status
.venv/Scripts/python.exe -B -m experiment.optimize_csv_research stop
.venv/Scripts/python.exe -B -m experiment.optimize_csv_research resume
```

Do not resume retired optimize_models or optimize_csv_models controllers.
Recipe requests/results/logs and Optuna SQLite:artifacts/training/csv_research_v3/.
Synthetic additions are labelled training augmentations, not independently captured traffic.
Next:finish solo/pairs, supported combinations, Optuna and capacity trials, seed repeats.
Separate future live stage remains blocked; reaching this one-family goal does not open its prior all-four gate.
- xgboost: best provisional accuracy 0.800167, macro recall 0.677986, F1 0.700498; pair_weight_none__native_ce
- kan: best provisional accuracy 0.781833, macro recall 0.633067, F1 0.633114; pair_smotenc__weight_none
- hem: best provisional accuracy 0.767367, macro recall 0.627301, F1 0.632924; capacity_0011
