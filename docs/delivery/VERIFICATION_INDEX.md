# Verification index

20 October 2026. Evidence root: `artifacts/delivery/20261020/`. All commands use the project `.venv` unless stated otherwise. Existing historic Rust evidence is linked separately; it was not regenerated.

| Record | Meaning |
| --- | --- |
| `baseline.py`, `commands.json` | Reproducible baseline driver and individual command exits/wall times. Refuses to replace recovery snapshot |
| `source-before.zip`, `source-before.json`, `source-after.json`, `preservation.json` | Recoverable selected source/config/doc snapshot and 474-file hash comparison; no Git branch/commit exists |
| `environment.json` | Interpreter/platform/library versions, child CPU environment, RAM snapshot and research-process inventory without command-line secrets |
| `demo-default.log`, `demo-benign.log`, `demo-ddos.log`, `demo-scan.log`, `demo-unknown.log` | Successful separate synthetic CLI flows |
| `pytest.log`, `pytest.xml` | 564 passed, 20 skipped, 11 warnings; 54.10 s; XML retains per-test outcomes |
| `golden-verify.log` | Sealed v1 migration fixtures verified without expected-value regeneration |
| `desktop-baseline.cjs`, `.log`, `.json` | Failed actual source Electron analysis run; Home/isolation/auth checks and startup/RSS measurements retained |
| `desktop-home.png`, `desktop-failure.png` | Actual fixed-window Electron screenshots; failure visibly shows unavailable samples and timed-out requests |
| `lifecycle.log`, `lifecycle/desktop-lifecycle-checks.json` | Existing source lifecycle harness; all 12 checks passed, with startup-failure screenshot |
| `runtime-probe.py`, `.log`, `.json`, `runtime-probe-backend.log` | Same current registry works standalone; actual desktop service status/samples fail after 60 s |
| `diagnose-http.py`, `diagnostics/sitecustomize.py`, `http-stacks.log`, `diagnose-http.log` | Isolated child-only trace hook reproduces timeout and records parent waiting in registry response polling; root cause not asserted |
| `standalone-analysis.py`, `.json`, `.log` | Actual included validation-sample inference/SHAP/template result, 1.235 s first and .047 s warm repeat; not desktop success |
| `inventory.py`, `dependency-inventory.json` | Manifest/catalog/package/native/notice inventory without loading research arrays |
| `installers.json` | Fresh exact size/SHA256/NotSigned status of two different old 0.3.0 installers |
| `final-integrity.json` | Final protected-source comparison, owned-process check and evidence-file index |

## Commands

In PowerShell at the project root, set CPU controls only in the command's child environment:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
.\.venv\Scripts\python.exe -B run_demo.py
.\.venv\Scripts\python.exe -B run_demo.py --sample benign
.\.venv\Scripts\python.exe -B run_demo.py --sample ddos
.\.venv\Scripts\python.exe -B run_demo.py --sample scan
.\.venv\Scripts\python.exe -B run_demo.py --sample unknown
.\.venv\Scripts\python.exe -B -m pytest evaluation -v
.\.venv\Scripts\python.exe -B tools/make_golden.py --verify
```

Desktop capture uses Playwright's Electron launcher, the existing Electron binary and an isolated user-data directory. Set `IMDAI_PLAYWRIGHT` to an already installed Playwright Node package and run `node artifacts/delivery/20261020/desktop-baseline.cjs`. The local verification package path is recorded in the chat/tool evidence; it is not a product dependency. For the existing lifecycle harness, set `IMDAI_VERIFICATION_DIR` to a new evidence directory, then run `node experiment/verify_desktop_lifecycle.cjs`. Do not pass `--full` to older capture scripts as an incidental check: those may start actual capture/generation.

The preserved failure artifacts should not be overwritten. Use a new run directory/harness output location for further checks. The diagnosis hook is only activated by the explicit child `PYTHONPATH` in `diagnose-http.py`; it is not installed into `.venv` or the app.

## What remains unexecuted

Successful source Electron sample/CSV/SHAP, fresh actual Qwen/live capture, renderer crash with active native work, full descendant cleanup under forced parent exit, installer probe and clean-machine matrix, Store certification, accessibility/performance qualification, remote CI, full vulnerability/license/secret scans, rebuilt package and signatures. Missing evidence is not a pass. No accuracy tuning, research retraining, locked-test reading or model promotion was performed.

## Historic migration pointers

- `artifacts/rust_migration/phase1/xgboost-expanded/summary.json`
- `artifacts/rust_migration/phase1/lime/parity.json`
- `artifacts/rust_migration/phase1/llm/measurements/measurements.json`, `grounding-parity.json`, `refusal-checks.json`
- `artifacts/rust_migration/phase1/capture/report.json`, `tests.log`

Their scope and caveats are reconciled in `ARCHITECTURE_BASELINE.md`. The existing source snapshot and sealed fixtures remain the baseline for later review.
