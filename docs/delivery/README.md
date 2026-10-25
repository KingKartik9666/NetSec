# NetSec Stage 0 delivery baseline

20 October 2026. This is the requested baseline, distribution feasibility, risk-ranked backlog and proposed first migration slice. It is not a production release or an authorization to install/publish. Current app/research source, checkpoints and sealed fixtures were not changed.

## Decision

Preserve Electron and the existing Python rollback path. Repair the reproduced desktop worker timeout before using the current application as a migration oracle. Then reconcile the already implemented Rust spikes and request the existing Phase 2 migration approval for a supervised sample-analysis slice. Keep React migration behind stable API contracts. Do not restart the completed XGBoost spike or silently replace Python LIME with the incompatible Rust approximation.

Stage 0 evidence collection is delivered; the successful desktop analysis acceptance remains **failed**. Stage 1 feasibility is **provisional**, with the installed-location probe pending. Direct release and Store submission are both **not ready**. No clean-machine claim is made.

## Fresh checks

| Check | Result | Evidence under `artifacts/delivery/20261020/` |
| --- | --- | --- |
| Default demo and benign/ddos/scan/unknown | 5/5 exited 0 | `commands.json`, `demo-*.log` |
| Full evaluation suite | 564 passed, 20 skipped, 11 warnings, 54.10 s pytest time | `pytest.log`, `pytest.xml` |
| Sealed migration fixtures | Verification exited 0; expected values not regenerated | `golden-verify.log` |
| Source Electron Home | Opens; sandbox/context isolation and unauthenticated-client rejection pass | `desktop-baseline.json`, `desktop-home.png` |
| Source Electron Samples | **Failed**: initialization displays `signal timed out`; never reaches ready | `desktop-baseline.json`, `desktop-failure.png` |
| Independent source lifecycle harness | 12/12: missing runtime, retry, owned backend crash, recovery, exit, stopped monitoring | `lifecycle/desktop-lifecycle-checks.json`, `lifecycle.log` |
| Standalone current XGBoost worker | Actual included validation sample prediction and exact native TreeSHAP succeeded twice | `standalone-analysis.json` |
| Actual desktop HTTP service | `/api/status` and `/api/samples` each return 503 after approximately 60 s | `runtime-probe.json`, `runtime-probe-backend.log` |
| Diagnostic repeat | Same HTTP failure; server waits at `registry.py:142` for worker response | `http-stacks.log`, `diagnose-http.log` |
| Recovery snapshot | 474 protected source/config/document files; baseline comparison reports zero changes | `source-before.zip`, `source-before.json`, `preservation.json` |

CUDA was hidden only in verification children; CPU libraries were limited to two threads through environment variables. The suite includes tiny temporary synthetic fitting tests, not a research training campaign. The 20 skips and XGBoost CPU-fallback warnings are visible and are not GPU validation. The separate hybrid optimization study was detected and left running. No study-control command, model promotion, dataset cleaning or held-out-test access occurred.

## Measurements and limits

Development machine: Windows 11 Home Single Language x64, build 26300; Python 3.11.9; Electron 44.5.0, embedded Node 24.21.0 and Chromium 152.0.7977.130. Six research/controller processes were present when the baseline was recorded. These single-run measurements include concurrent machine activity and are not hardware-independent performance claims.

| Measurement | Observed | Interpretation |
| --- | --- | --- |
| Electron launch to Home | 1.890 s | Navigation shell ready, not model ready |
| Home process-tree RSS | 445,964,288 bytes (425.3 MiB) | Sum of working sets including launch helpers; shared pages may be counted more than once |
| Standalone first sample analysis | 1.235 s | Existing worker already created by snapshot call; includes lazy model load, prediction, SHAP and template report |
| Standalone second sample analysis | 0.047 s | Warm repeat of the same included sample |
| TreeSHAP reconstruction error | 9.095492714550346e-7 | Selected class raw margin; no accuracy inference |
| Desktop analysis latency | Unavailable | Startup failure prevented successful sample and CSV flows |
| Normal exit/backend recovery | Executed successfully | 12-check lifecycle run verifies port shutdown; it does not establish forced-termination cleanup of all grandchildren |

Proposed initial development budgets: Home <=5 s, idle Home tree RSS <=600 MiB, healthy sample/CSV initialization <=15 s, warm single-row analysis <=2 s, graceful close <=10 s, and zero owned descendants after forced-parent termination. Only Home time/RSS are currently demonstrated against these proposed budgets. Peak analysis memory, cancellation, installer disk-headroom thresholds and low-end hardware performance remain to be measured.

## Report map

- [Architecture and capability baseline](ARCHITECTURE_BASELINE.md): current routes, schemas, six XAI methods, Rust evidence and dependency graph.
- [Security baseline](SECURITY_BASELINE.md): concrete boundaries, reproduced failure, lifecycle risks and verification obligations.
- [Distribution and dependency matrix](DISTRIBUTION_MATRIX.md): package sizes/hashes/signatures, runtime gaps, licensing and channel conditions.
- [Requirements and risk-ranked backlog](REQUIREMENTS_BACKLOG.md): stable IDs, owners, acceptance and channel impact.
- [First-slice decisions](FIRST_SLICE.md): precise task packets, dependencies, rollback and existing approval boundary.
- [Verification index](VERIFICATION_INDEX.md): evidence, reproduction and exclusions.
- [Release checklist and runbook](RELEASE_RUNBOOK.md): candidate-to-release controls, owner-supplied blockers.
- [Task ledger](TASK_LEDGER.md): current states and next work.

## Source quality and review

Current source and fresh executed checks override older prose. `ARCHITECTURE.md` and `APP.md` omit parts of the current Models workspace; `RUST_MIGRATION.md` still presents Phase 0 as current despite Phase 1 artifacts. Earlier Rust measurements are identified as historical, not freshly rerun. Three read-only specialists returned useful partial findings but hit the service usage limit before handoff documents. The lead verified the findings and wrote the reports. No independent final review is claimed.

The requested reconnaissance boundary is complete enough for an informed next-stage decision; release-critical work remains. Next / upcoming task: diagnose and repair S0-FIX-01 desktop worker startup, then rerun the real desktop sample/CSV path and close the Phase 1 migration decision record before Phase 2 authorization.
