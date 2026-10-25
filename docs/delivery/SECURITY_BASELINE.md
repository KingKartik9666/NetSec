# Security and reliability baseline

20 October 2026. Bounded source review plus the fresh checks listed below. This is not a penetration test, comprehensive dependency scan or security certification. No external scanner uploads were made. Severity describes release impact; potential risks are distinguished from reproduced failures.

## Trust boundaries and existing controls

Untrusted inputs include CSV content, labels, model/LLM text and local websites. The renderer is sandboxed and isolated, without Node integration; fixed preload actions validate sender/main frame; navigation and remote requests are restricted (`desktop/main.cjs:25,29,112,133,158`; `preload.cjs`). Fresh Electron execution verified isolation and rejection of an unauthenticated client. Per-launch token injection occurs in the main process, with constant-time server comparison (`server.py:121`). This does not protect a compromised Windows user account.

The backend binds `127.0.0.1` on an ephemeral port. Host/Origin checks, CSP, no-store and nosniff, a 10 MiB body limit and 500-row limits already exist. Model worker operations are allowlisted; request payloads cannot name Python imports. Local trusted configuration can select adapters, so it is executable configuration, not an untrusted document format. Model hashes establish integrity relative to the local manifest; they do not authenticate a maliciously replaced manifest/checkpoint pair.

Monitor and language services are separate processes. Monitoring starts explicitly; no traffic payload archive or telemetry is implemented in the inspected flows. Native crashes need process containment beyond Rust panic handling. RAG text remains background with heuristic checks, not authoritative instructions or proof of semantic grounding.

## Findings and backlog

| ID / priority | Finding and evidence | Status / owner / acceptance |
| --- | --- | --- |
| REL-01 / P1 | Real desktop Samples never ready. Independent authenticated HTTP `/api/status` and `/api/samples` return 503 after ~60 s. Standalone same-registry calls and saved inference pass. Trace waits at `registry.py:142`; worker-startup/response cause not yet proven | **Reproduced**, runtime engineer; repair with no model change, then real Electron sample/CSV/SHAP and timeout/cancel regression |
| SEC-01 / P1 | No `render-process-gone` or `unresponsive` handler. Native close/before-quit request renderer confirmation (`main.cjs:130,180`), so a dead renderer may prevent reliable close while work continues | Static lifecycle risk, desktop engineer; crash/hang owned renderer while a bounded fixture worker is active, ensure independent close and descendant cleanup |
| SEC-02 / P1 | Electron force timeout kills only immediate backend (`main.cjs:146`). Optional-service close waits for a lock held by a call that may last 15/60 s (`capabilities.py:57,101`); monitor can own model children | Static ownership risk, runtime engineer; Windows Job Object or equivalent owned-tree proof, bounded shutdown, native failure/parent kill tests. Normal exit pass does not close this risk |
| SEC-03 / P2 | Generic `/api/live/report` accepts any retained nonzero packet count (`server.py:308`); Stop clears verdicts but retains metrics (`live_monitor.py:131`). Unlike feature CSV, the route is not bound to observation ID/running state | Static API consistency risk; evidence engineer. Either clearly version/export a stopped historical snapshot or require current immutable ID. No claim that stale model predictions currently survive Stop |
| SEC-04 / P2 | Token created once per Electron launch (`main.cjs:13`), reused on service restart; master request requires rotation on restart as well | Confirmed source policy gap; desktop security owner. New backend credential and rejection of old token after restart; do not expose tokens to renderer/logs |
| SEC-05 / P2 | Invalid UTF-8 hits local `csv` import after decoding but exception tuple references it (`trained_xgboost.py:41`), producing historical generic 500 instead of scoped invalid input | Frozen baseline defect in migration goldens; runtime owner. Review intentional compatibility correction and reject malformed UTF-8 without internal exception leakage |
| SEC-06 / P2 | IPC/service errors may include exception text and local paths; bounded diagnostics are not a complete redaction/export policy (`main.cjs:48,74`; `core/worker.py:61`) | Static privacy gap; security/product owner. Redact tokens/paths/evidence and preview optional diagnostics |
| SEC-07 / P2 | Synchronous HTTP operations and per-model locks do not supply explicit job identities, bounded global queues, cancellation and terminal states | Architecture gap; runtime owner. Stress bounded work, cancel/detach safely, stale results cannot become current |
| SEC-08 / P1 release gate | No executed dependency vulnerability/secret/license scan or complete PE-signature inventory; current installers unsigned | Release/security owner. Offline/local scans and explicit exceptions with owner/expiry before candidate promotion |

No critical remote exploit was established by this review. Lack of such a finding is not a security pass.

## Fresh verification and exclusions

The 564-pass suite covers injected model failures, auth/limits, stale observation IDs and other contracts; many use small fixtures or injected services. Fresh Electron lifecycle verification passed 12 checks: absent runtime, retry, owned backend crash, restart, stopped monitor, exit and backend port closure. It did not kill a live capture worker or a hung renderer. Screenshots are in `artifacts/delivery/20261020/` and `lifecycle/`.

Unexecuted: actual packet capture, admin-only/missing-driver hardware variants, renderer crash during capture, forced-parent descendant cleanup, malformed-input fuzzing, signing/update-chain tampering, full dependency scans, accessibility certification and clean-machine install/uninstall. These remain explicit acceptance tasks, not skipped passes.

Official baseline checked 20 October 2026: [Electron security guidance](https://www.electronjs.org/docs/latest/tutorial/security/) supports the isolation, narrow IPC, sender validation, navigation and CSP review. [Electron support policy](https://www.electronjs.org/docs/latest/tutorial/electron-timelines) supports only the latest three stable majors and the latest minor within each. The installed 44.5.0 pin needs a release-time supported-version/patch review; no dependency upgrade was performed here.
