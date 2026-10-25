# Requirements, acceptance and prioritized backlog

20 October 2026. D = direct EXE, S = Store. Status refers to this delivery program. A passed software check does not imply a research result or release qualification. Related concrete findings are in `SECURITY_BASELINE.md`.

| Requirement | Owner / task | Acceptance evidence | Channel | Current state |
| --- | --- | --- | --- | --- |
| NS-001 Research isolation | Lead / S0-BASE | Source snapshot/hash comparison, read-only process inventory, CPU child environment | D,S | Baseline passed; no study actions |
| NS-002 Reproducible baseline | QA / S0-BASE | Five demos, full evaluation, immutable fixture verification | D,S | 5 demos; 564 passed/20 skipped; fixture check passed |
| NS-003 Real desktop analysis | Runtime / S0-FIX-01 | Fresh Electron sample, CSV and SHAP with actual selected artifacts | D,S | **Failed** worker response timeout |
| NS-004 Lifecycle isolation | Desktop / S6-LIFE | Normal/EOF/renderer crash/backend crash/parent kill; zero owned orphans | D,S | 12 normal/recovery checks passed; forced-tree cases pending |
| NS-005 Four-model availability | Runtime / S3-MODELS | Every catalog state honest; package can execute each declared model with correct schema | D,S | RF manifest missing; package omits four-model assets/torch |
| NS-006 Versioned contracts | Architecture / S2-CONTRACT | DTO/error/job/capability/observation schemas sealed; model order/preprocessing explicit | D,S | Legacy golden corpus exists; new contracts pending |
| NS-007 Feature boundaries | Architecture / S2-ARCH | No cross-feature implementation imports; disable each optional feature independently | D,S | Python isolation exists; target Rust enforcement absent |
| NS-008 Rust sample slice | Runtime / S3-SAMPLE | Native prediction/SHAP within 1e-5/1e-4, exact ordering, real Electron success | D,S | Spike exists; integration gated |
| NS-009 Model provenance | Evidence / S3-PROVENANCE | Artifact/schema/class/input hashes travel with immutable results and exports | D,S | Partial metadata exists; complete protocol pending |
| NS-010 Supervised Python | Runtime / S3-WORKER | Versioned allowlist, bounded bytes/time/queues, packaged dependencies, crash cleanup | D,S | Allowlist/isolation exists; complete ownership/protocol pending |
| NS-011 React product shell | Frontend / S4-UI | Representative screen approved by runtime evidence; existing navigation/actions preserved | D,S | Plain JS/Jinja; React absent |
| NS-012 Job states/cancellation | Runtime/frontend / S2-JOBS | No duplicates; current immutable selection; terminal cancelled jobs cannot republish | D,S | Renderer generations partly protect replies; full jobs absent |
| NS-013 Accessibility | Frontend/QA / S4-A11Y | Keyboard, names, focus, small window, high DPI, reduced motion; WCAG 2.2 AA coverage report | D,S | Some semantic controls/reduced motion; full audit unexecuted |
| NS-014 Model comparison | Runtime/frontend / S5-COMPARE | Snapshot input, run compatible models boundedly, progressive isolated errors/cancel, honest scores | D,S | Recorded metric table exists; multi-run comparison mode absent |
| NS-015 SHAP | XAI / S5-SHAP | Selected final output, class-margin units, reconstruction, reference provenance | D,S | XGBoost implemented/standalone executed; others missing |
| NS-016 LIME | XAI / S5-LIME | Actual package or distinct approximation label; missingness/constraints/fidelity | D,S | Python XGBoost implemented; Rust drop-in no-go |
| NS-017 Anchors | XAI / S5-ANCHOR | Valid selected-model capability, settings, coverage/precision and limitations | D,S | Missing |
| NS-018 Counterfactuals | XAI / S5-CF | Measured/derived constraints; valid probe; never causal/mitigation claim | D,S | Missing |
| NS-019 Grouped permutation | XAI / S5-PERM | Approved training reference hash, groups/settings/metric/caching; no test access | D,S | Missing |
| NS-020 ALE | XAI / S5-ALE | Compatible reference, intervals/support, bounded compute and cache identity | D,S | Missing |
| NS-021 Local language/RAG | Runtime / S3-LLM | Actual CPU inference, no template fallback, explicit requested use and heuristic limits | D,S | Implemented/historical spike proof; fresh integrated candidate pending |
| NS-022 Live capture | Capture / S3-LIVE | Explicit Start/Stop, missing-driver/permission errors, stale ID rejection, no auto-resume | D,S | Python implemented; Rust only enumeration/replay; fresh hardware deferred |
| NS-023 Auth/IPC/CSP | Security / S6-BOUNDARY | Invalid credentials/senders/origins rejected; token rotation on restart | D,S | Existing controls; restart rotation gap |
| NS-024 Input/export safety | Security / S6-INPUT | Malformed Unicode/rows/limits, path/HTML injection, formula policy and scoped errors | D,S | Many tests pass; invalid UTF-8 baseline defect and export policy unresolved |
| NS-025 Privacy/storage | Product/runtime / S6-DATA | Per-user paths, atomic versioned data, recovery, retention, exact deletion/diagnostics scope | D,S | In-memory uploads; product data-management controls pending |
| NS-026 Product identity/help | Product/owner / S7-PRODUCT | Consistent icons/version/About/help; verified legal and support fields/policies | D,S | Partial branding; owner fields unresolved |
| NS-027 Lawful self-contained package | Release/owner / S1-RIGHTS | Exact runtime/assets/SBOM/rights and offline package probe | D,S | Old package exists; rights/coverage unresolved |
| NS-028 Installer lifecycle | Release/QA / S8-INSTALL | Clean install/reinstall/upgrade/downgrade policy/uninstall; standard user, paths/disk failures | D,S | Fresh execution pending |
| NS-029 Updates/signing | Release/security / S8-TRUST | Channel-specific consent/trust/rollback; sign exact tested bytes, verify timestamp/signatures | D,S | Unsigned; protected infrastructure absent/unverified |
| NS-030 Automated gates | Lead / S2-CI,S9-PIPELINE | Local + hosted Python/Rust/frontend/architecture/security/package checks, failure retention | D,S | Local baseline executed; no root Git/CI/repository binding |
| NS-031 Store qualification | Release/owner / S10-STORE | Current policy, silent/offline/immutable URL, all PE signatures, driver decision, listing/certification | S | Provisional route only |
| NS-032 Independent qualification | QA / S10-QUALIFY | Exact installed candidate, faults/performance/accessibility/clean machine; independent review | D,S | Pending; reconnaissance workers hit usage limit |
| NS-033 Release handoff | Lead/owner / S11-HANDOFF | Hashes/signature/support scope/runbook and separate publish/install authorizations | D,S | Runbook draft; no release candidate |

## Work order

1. **P1 S0-FIX-01**: diagnose desktop HTTP worker startup/response and restore real Samples/CSV execution. Standalone prediction/SHAP success narrows the failure; do not edit model bytes or inflate timeouts as an unexplained workaround.
2. **P1 S6-LIFE**: close renderer-death and owned-tree shutdown risks. A healthy shutdown test alone cannot qualify capture/native failures.
3. **P1 S1-RIGHTS + S3-MODELS**: decide lawful, exact distributable model set and runtime closure. RF unavailable must remain honest until an existing compatible artifact is explicitly selected; no training/promotion during app work.
4. **P1 S1-PROBE**: isolated installed-location feasibility probe and package coverage gate. Keep existing installed app unchanged.
5. **P1 S2-CONTRACT/CI**: reconcile Phase 1 outcomes, preserve LIME/capture Python paths, obtain required Phase 2 approval, freeze contracts and implement local quality runner. Remote CI requires a real repository/hosting decision.
6. **P1 S3-SAMPLE**: Rust supervised sample slice through unchanged Electron/UI, with Python rollback; independent review before CSV/XAI/live expansion.
7. **P2 S4/S5**: React representative screen, bounded jobs/comparison and six-method capability implementation; no empty decorative screens or universal support claims.
8. **P1 release gates S6-S11**: integrated security, identities/licenses, installer matrix, signing, clean-machine qualification and channel-specific handoff. These cannot be deferred past publication.

Owners are roles, not invented named people. Assign actual maintainers before remote CI exceptions and public release. Owner-only legal/signing/account decisions do not block preparing local engineering evidence.
