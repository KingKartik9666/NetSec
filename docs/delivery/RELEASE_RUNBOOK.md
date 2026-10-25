# Release checklist and maintenance runbook draft

20 October 2026. This checklist prepares future local candidates. It does not authorize signing, uploading, Store submission or replacement of the owner's installed application.

## Current gate state

| Gate | State |
| --- | --- |
| Context, source inventory, baseline logs and recovery snapshot | Recorded |
| Successful current desktop analysis | **Failed**; S0-FIX-01 required |
| Complete Rust/React application | Pending; existing migration approval gate applies |
| Lawful complete runtime/model package | Pending; old package lacks four-model assets; rights unresolved |
| Independent integrated review | Pending; specialist quota interruption is not approval |
| Clean-machine installer lifecycle | Not executed |
| Public identity/policies/support | Owner fields unresolved |
| Signing and post-sign verification | Not configured/verified; current installers unsigned |
| Direct publication / installed replacement | Not authorized by this baseline |
| Store submission/certification | Not ready; route provisional |

## Candidate procedure

1. Preserve a recoverable source snapshot or actual Git revision and existing user changes. Keep upstream repositories and active research outside the edit/build closure. Record exact model/schema/class/reference identities; no automatic selection of newer research assets.
2. Resolve baseline failures, complete the selected stage's tests and independent review. Freeze compatibility contracts before parallel implementation. Keep Python rollback explicit until native parity and retained-runtime packaging pass.
3. Execute local gates: five demos/full evaluation; Rust locked fmt/clippy/tests/architecture checks; frontend typecheck/lint/tests/build when introduced; contract and golden checks without regeneration; secret/dependency/license review with owner/expiry for exceptions. Retain skipped hardware checks visibly.
4. Build a new uniquely versioned unsigned candidate from pinned dependencies. Produce a hash-complete package manifest/SBOM, notices, source/build identity and dependency/runtime closure. Compare current source/model/UI hashes with packaged bytes. Never mark `--refresh-assets` as a full backend rebuild.
5. Test a disposable installation on a supported clean standard-user Windows VM without Python/Node/Rust/Git/Docker. Verify offline sample/CSV/XAI, honest unavailable optional features, DLL/asset lookup, per-user writable paths, EOF/exit/crash cleanup. Explicitly test driver missing/installed permissions in an isolated hardware-capable environment.
6. Test same-version/reinstall, upgrade from exact previous hash, refused incompatible downgrade, uninstall/reinstall and data-retention behavior. Include spaced/Unicode path, low disk, files in use and interrupted install. Preserve shared Npcap and unrelated processes. Do not touch the owner's current install to obtain this evidence.
7. Close security/performance/accessibility findings or obtain a specific owner-approved release exception. Obtain verified legal publisher, copyright, support/security contacts, policy URLs and asset redistribution decisions. Keep current limitations inside offline Help/About and channel metadata.
8. Present the exact candidate path/hash, tests, notices, blockers and rollback for the applicable release authorization. Protected signing uses owner-approved credentials/infrastructure, timestamping and verification; retest signed artifact. Publication promotes tested bytes, never a different rebuild.
9. Direct channel: publish only with explicit authorization and verified update/trust/rollback policy. Store channel: recheck effective policy, silent switches/return codes, all PE signatures, immutable versioned HTTPS URL and driver disclosure; prepare accurate listing/reviewer notes. Certification remains Microsoft's decision.
10. Installed-app replacement is a separate action. Inspect existing version/hash/data, make a scoped recovery plan, and use only the applicable owner authorization.

## Recovery and maintenance

Before migration, record current per-user data schema and retention. After migration, use transactional/recoverable data upgrades and refuse unsafe downgrades. Preserve the previous tested candidate and user-data backup under a documented retention policy. Rollback must never overwrite research source/checkpoints or delete shared drivers. Verify process ownership before any stop/cleanup.

Assign named release and vulnerability-response owners before publication. Proposed cadence: review dependency advisories weekly and before every release; prioritize critical/high findings immediately, with documented mitigation and patch deadlines. Public security/support reporting path remains unresolved; do not invent an email address or promise an SLA.

Post-release smoke should check launch, sample/CSV, explanation provenance, offline operation, optional-feature refusal and shutdown. A meaningful failure triggers triage, exact version/hash identification and a reviewed rollback/update decision. Diagnostics must be redacted and user-previewed; no unsolicited telemetry. Channel updates follow their separately verified servicing rules.
