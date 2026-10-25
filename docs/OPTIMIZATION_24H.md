# Detached four-family research training

Run from the project root in PowerShell:

```powershell
.venv/Scripts/python.exe -B -m experiment.optimize_models start
.venv/Scripts/python.exe -B -m experiment.optimize_models status
.venv/Scripts/python.exe -B -m experiment.optimize_models stop
.venv/Scripts/python.exe -B -m experiment.optimize_models resume
```

`start` launches a hidden, detached Python controller. It completes missing fits
from the existing four-model study, then benchmarks XGBoost, genuine efficient-KAN
and the original GitHub HEM neural classes with the existing closed-set adaptation.
The conventional sklearn Random Forest runs once with fixed settings on CPU.
The other three research trainers require CUDA, without silent CPU fallback.

On 11 October the user authorized architecture changes for the three optimized
families, while RF stays fixed. The corrected `architecture_v3` policy adds independently
benchmarked XGBoost depths 4/8/10/12 and 800/1200-round ceilings, KAN layer widths,
depth and grid density, plus HEM-derived neural widths, PCA dimensions and k=3/5.
Original classes and controls are retained; modified HEM variants are explicitly
distinguished from the unchanged GitHub architecture. These options are experiments,
not assured improvements. Larger XGBoost round limits remain constrained by
validation stopping, its 720-second internal runtime cap, and the external fit budget.
The first v2 policy and source are preserved; v3 fixes capacity-option retention
for checkpoint resume/repeat fits and waits for worker exit before controller refresh.

`.venv/Scripts/python.exe -B -m experiment.optimize_models refresh` activates a
new controller at the current completed-fit boundary without resetting the deadline.
The transition and immutable versioned policy are recorded beside the study.

The persisted deadline is 24 hours after the first start. Resume does not extend
it. Search fits are bounded by the first 18 hours; leader/control repeats use
seeds 43 and 44 within the remaining budget. Search may exhaust its finite,
bounded candidates early and move to repeats. Each fit has a
maximum 20-minute external budget; the existing XGBoost implementation also has
its shorter internal budget and validation stopping; capacity variants may raise
the original 500-round ceiling up to 1200 rounds.

An OS lock prevents concurrent controllers. A Windows Job Object terminates trial
descendants when the controller dies. Interrupted trials get a new attempt
directory on resume; partial weights are preserved as evidence, not reused as a
completed fit. Complete results require checkpoint integrity and matching request,
seed/config/protocol. Auxiliary inference artifacts and input/source hashes are
pinned. Corrupt or changed evidence stops recovery rather than mixing studies.

Recovery is registered for the current user's logon using a hidden script in the
Windows Startup folder named `IMDAI-Training-Resume.vbs`. It exits after expiry or
completion. Remove that exact script to disable logon recovery. A temporary
execution-state request inhibits automatic sleep while the controller is running;
shutdown, power loss, user-requested sleep and logout still interrupt execution.
No permanent power settings or administrator credentials are changed.

Results, per-attempt logs, state, integrity records and the final report live in
`artifacts/training/optimization_24h_v1/`. `docs/RESUME_CONTEXT.md` is updated at
fit boundaries. No installed application or primary model is automatically replaced.

Selection uses seven-feature accuracy then macro F1 for XGBoost, and full-input
macro F1 then accuracy for KAN/HEM. Full-input regressions are reported separately.
Promotion eligibility requires beating the matched seed-42 control and both
additional matched-seed controls. The final test partition is never loaded.
Repeat agreement is a conservative selection gate, not statistical proof of
generalization; these are source-validation measurements, not live-PC accuracy.

95%+ accuracy is a target, not a guarantee. A later stage would create four live
deployment versions alongside the four CSV versions, retaining four families.
It must remain blocked until all four CSV models individually meet 95% validated
accuracy, including unchanged RF, and independent labels are available. Npcap
traffic resemblance to CIC IIoT2025 and model predictions cannot establish labels.
Feature definitions, units, windows and label mappings must be checked before
using other datasets or controlled captures. No attack traffic is generated here.
