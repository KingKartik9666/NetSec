# Resume the training-only research study

Paste this into a new chat:

> Continue the csv_research_v3 training-only study in K:\IMDAI Project.
> Read docs/PROJECT_CONTEXT.md completely, then docs/RESUME_CONTEXT.md,
> docs/CSV_MODEL_TRAINING.md and artifacts/training/csv_research_v3/implementation-ledger.md.
> Inspect current controller processes, state.json, report.json, active progress/fit logs
> and saved results before taking action. Do not duplicate a running controller.
> If it is stopped and I have explicitly requested resuming training, use the new
> experiment.optimize_csv_research resume command. Never restart retired controllers.
> Optimize only full-CSV XGBoost, genuine efficient-KAN and HEM-derived closed-set models;
> conventional RF stays fixed. Preserve the real training/validation split and locked test.
> Success needs one identical recipe at >=95% validation accuracy in seeds42/43/44,
> verified CPU reload/CUDA execution and the recall/false-positive regression guards.
> Keep synthetic additions labelled, preserve checkpoints/data/evidence, and defer live
> models, UI, packaging and app replacement. There is no overall deadline; per-fit
> watchdogs, failure isolation and explicit Stop remain. Do not claim toy checks or
> intermediate validation scores establish research success.

The user cancelled the proposed reversed-split cycle on 12 October 2026.
Its watcher is stopped, recovery entry removed and STOP marker preserved.
Do not queue or resume experiment.reverse_csv_study. The original test remains
locked, and only the existing forward training study should continue.
See docs/REVERSED_RESUME_CONTEXT.md for the cancellation record.

Commands from the project root:

```powershell
.venv/Scripts/python.exe -B -m experiment.optimize_csv_research status
.venv/Scripts/python.exe -B -m experiment.optimize_csv_research stop
.venv/Scripts/python.exe -B -m experiment.optimize_csv_research resume
```

The detached controller continues while the PC stays awake, independently of chat.
Stop terminates its owned worker; wait until status is stopped and active_trial is null
before gaming. Logon recovery respects STOP. Completed fits are reused after hash
verification; the interrupted fit restarts from scratch in a preserved new attempt.
Shutdown/power loss interrupt computation, not completed evidence. Recovery resumes
an unstopped study after logon. CPU checkpoint reload is all validation rows for
XGBoost/KAN and a documented 128-row check for expensive HEM nearest-neighbor inference.

Verified on 11 October 2026:459 evaluation tests passed, dependency check passed,
default/four-sample demos passed. Actual CUDA toy fits, same-family ensembles, CTGAN
generation and training-only OOF calibration executed. Those toy checks are not
dataset results.26 completed original CSV fits were hash-verified/adopted; the new
study launched and its first XGBoost CUDA fit was observed progressing on all real
training observations. Live status in RESUME_CONTEXT supersedes this launch snapshot.
