# Proposed first migration slice and architecture decisions

20 October 2026. Proposed work, not implemented. The existing Python source remains unchanged; Rust parity spikes are reusable evidence, not an installed backend.

## Decisions

**ADR-001: Preserve the shell and use a modular monolith.** Keep Electron main/preload separate from React renderer. Keep transport outside domain operations. Feature implementations depend on shared contracts/pure logic, not each other. The composition root owns registrations. Native/Python workers remain supervised processes where required. Source: master prompt and `RUST_MIGRATION.md` target architecture.

**ADR-002: Repair the baseline before replacing it.** Current desktop HTTP model calls fail while standalone registry prediction/SHAP succeeds. The first task restores that boundary and records an authentic end-to-end oracle. A larger Rust rewrite must not hide this failure or silently narrow functionality.

**ADR-003: Reuse native XGBoost, retain Python LIME and unsupported models.** The historical numeric-tree/TreeSHAP spike passes its tested tolerance; the LIME spike explicitly rejects drop-in parity. RF/KAN/HEM formats are not covered by the XGBoost parser. Keep optional Python workers with declared dependencies until separately qualified native alternatives exist. No automatic model selection/promotion.

**ADR-004: Stage source changes and packages separately.** Use new immutable candidate versions/hashes; old installations remain untouched. Direct signed NSIS first is provisional. Store route and optional Npcap remain explicit separate gates. No user needs Docker or build tools for a qualified package.

## Bounded task S0-FIX-01 / runtime and QA

- User outcome/scope: restore saved sample and compatible CSV analysis through the current Electron shell; identify why HTTP-spawned model workers time out while standalone workers respond.
- Owned files: a narrowly reviewed set under `framework/app/desktop_backend.py`, `framework/core/registry.py`/`worker.py` if proven necessary; focused evaluation test and delivery evidence. Do not assume all need edits.
- Exclusions: research code, datasets, splits, model bytes, current installed app, large UI rewrite, new features or dependencies without a demonstrated need.
- Prerequisites/frozen contracts: current sealed migration v1, exact current app artifact/schema/class order and the Stage 0 source snapshot. Preserve existing route shapes unless a documented defect correction requires an explicit fixture review.
- Inputs/interfaces: `runtime-probe.json`, `http-stacks.log`, standalone report and source desktop handshake/pipe protocol; output is the same authenticated HTTP contract with working owned model execution.
- Acceptance including failure/cancellation: sample/CSV/SHAP success in Electron; missing/corrupt model scoped error; request timeout does not wedge later work; close/EOF cleans owned children; no capture/generation starts; no fallback or checkpoint rewrite.
- Required commands/evidence: five demos, full evaluation, sealed fixture verification, current desktop baseline/lifecycle harness, selected worker fault test; save screenshot/report and compare source/artifact hashes.
- Performance/security: child-only CPU limits, bounded model calls, no tokens/evidence in logs, no research-process manipulation. Preserve measured scopes when comparing latency.
- Migration/rollback: restore only changed owned sources from the snapshot if acceptance fails; never restore research directories or mass-copy over concurrent work.
- Handoff: changed files/diff, root cause, fresh command exits/counts, normal/failure cleanup, remaining limitations and the Phase 1 decision prerequisite.

## Bounded task S3-SAMPLE / Rust runtime, desktop and independent QA

- User outcome/scope: one actual selected sample analyzed by Rust through the existing Electron shell, including startup, auth, errors and shutdown.
- Owned directories: future `rust/crates/{core-api,core,shared}`, `rust/app`, HTTP/model adapters and `features/sample-analysis`; one lead integrates manifests/lockfiles, Electron selector and CI. Freeze DTOs before frontend concurrency.
- Exclusions: React screen expansion, CSV/live migration, research/training, model promotion, silent Python fallback, installation/publishing.
- Prerequisites/frozen contracts: S0-FIX-01 accepted; Phase 1 consolidated decision including capture limitations; **owner approval for existing migration Phase 2 skeleton**; versioned request/result/error/capability/lifecycle contracts and sealed fixtures.
- Inputs/interfaces: checksum-pinned UBJ, ordered schema/classes, immutable included sample, existing numeric-tree spike. Output includes actual model/input IDs, probabilities/margins, stable top-three, exact SHAP and provenance.
- Acceptance: actual Electron sample; margins/probabilities <=1e-5 and SHAP <=1e-4 versus Python, identical class/top-three and reconstruction; NaN routes, wrong hash, unsupported format, disabled feature, bad token, timeout, native failure and cancellation. Failed explanation preserves prediction when its contract permits.
- Commands/evidence: locked Rust fmt/clippy/tests/architecture tests, five demos/full Python evaluation, sealed fixtures, same-input parity, Electron screenshots and owned-tree shutdown tests. Independent reviewer checks the integrated slice, not their own implementation.
- Constraints: offline CPU credential-free, explicit resource budgets/queue limits, narrow worker protocol and no renderer Node access.
- Rollback: `NETSEC_BACKEND=python|rust`, Python default initially; backend choice recorded. Failure never silently executes the other backend.
- Handoff: exact source snapshot/build hash, contract version, executed cases/differences, supported formats, retained runtimes and next CSV/report dependency.

## Approval reconciliation

`docs/RUST_MIGRATION.md:270` explicitly says **“Later gates — approval required”**; its Phase 2 **“Skeleton”** begins at line 281. Earlier context records approved Phase 1 work, and the current spike artifacts confirm that work exists. No later skeleton approval was found in the inspected project evidence. The new master request expressly says to honor these gates and return this baseline before large implementation. Do not interpret this reconnaissance or its `continue` message as automatic waiver of that explicit migration gate.

This is a concrete proposal for the next work, not an approval request for publishing or replacing the installed app. Finish baseline repair and the Phase 1 go/no-go decision before seeking approval for the specific reviewed Phase 2 implementation.
