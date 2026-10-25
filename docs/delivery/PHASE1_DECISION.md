# Rust Phase 1 decision and bounded Phase 2 proposal

22 October 2026. This reconciles existing spike evidence against the current files. The Rust spikes, Qwen inference and packet observations were not rerun during the desktop repair. The source Electron sample/CSV baseline is now working; its new evidence is separate from the historical spike results.

## Component decisions

| Component | Evidence and scope | Decision |
| --- | --- | --- |
| XGBoost and native TreeSHAP | Six executed expanded cases pass: synthetic multiclass/packet/ties, saved packet model on synthetic inputs, and saved app/partial models on included validation samples plus missing masks. Identical tested argmax/top-three; maximum probability difference 5.96e-8; tested margin and SHAP differences zero. Enhanced bundle was absent, so not tested. | Go for the tested numeric-tree formats, subject to fresh integrated parity. No detector-accuracy claim. Reject categorical splits, DART, vector leaves and unsupported layouts explicitly. |
| LIME | Three synthetic cases and technical checks pass, but RNG/numerical differences, constant-target R2 behavior and ranking/fidelity differences remain. Existing report explicitly says drop-in replacement no-go. | Retain Python package LIME. Do not present the Rust approximation as equivalent or promote it silently. |
| Local language and grounding | Historical actual Qwen inference on three synthetic evidence cases, repeated across Python, supervised child and native binding modes. Output text matched by case; 13/13 grounding cases and refusal/deadline checks passed. | Supervised llama.cpp child is a feasible integration candidate. Integration, runtime resource/cleanup checks and package qualification remain required; text equality does not establish semantic faithfulness. |
| Capture | Windows adapter/endpoint enumeration comparison, missing-driver handling and eight synthetic replay cases passed. Report verdict is `go_enumeration_only`. | Go only for enumeration/replay. No-go for replacing live capture until actual handle permissions, delivery, start/stop/window behavior and native failure containment are verified. |
| RF, KAN and HEM | XGBoost parser/spikes do not cover these model formats. Current RF manifest is missing; KAN/HEM remain existing Python adapters with their declared schemas and artifact dependencies. | Retain explicit Python adapters and truthful unavailable states. No substitute, artifact selection or promotion is authorized by this decision. |

The aggregate decision is **no-go for wholesale native replacement**. A bounded skeleton is a reasonable next proposal because it can establish contracts and supervision while Python remains the default. This is not a blanket Phase 1 parity pass and does not waive the existing owner gate.

Evidence: [consolidated records and source hashes](../../artifacts/delivery/20261022/phase1-decision-evidence.json), [expanded XGBoost report](../../artifacts/rust_migration/phase1/xgboost-expanded/summary.json), [LIME decision](../../artifacts/rust_migration/phase1/lime/parity.json), [Qwen measurements](../../artifacts/rust_migration/phase1/llm/measurements/measurements.json), [grounding](../../artifacts/rust_migration/phase1/llm/measurements/grounding-parity.json), [refusal checks](../../artifacts/rust_migration/phase1/llm/measurements/refusal-checks.json), [capture decision](../../artifacts/rust_migration/phase1/capture/report.json).

## Reviewable task S2-SKELETON

- **Stage/owners/outcome:** existing Rust migration Phase 2; architecture/runtime lead plus independent desktop QA. Add an explicitly selectable Rust service skeleton that starts through Electron, authenticates requests and reports honest capability/status states. Keep Python the default.
- **Owned paths:** new `rust/app`, `rust/crates/core-api`, `rust/crates/core`, `rust/crates/shared`, minimal HTTP/static adapters and feature template; root workspace manifest/lock; narrowly scoped Electron launcher selector; architecture/contract checks and delivery docs. One integrator owns manifests and shared contracts. Existing `rust/spikes` stay isolated.
- **Exclusions:** feature ports, React rewrite, model conversion/promotion, research jobs/data, packet capture, LLM generation, installer replacement, publishing and Python retirement. Separate feature gates remain in force.
- **Prerequisites:** accepted desktop repair, this component decision, and owner approval of the existing Phase 2 gate with these retained/deferred capabilities. Freeze DTO/error/capability/lifecycle contracts before parallel implementation.
- **Interfaces:** retain loopback-only random-port readiness, session authentication, existing Electron bridge/security settings and explicit shutdown/EOF. Add `NETSEC_BACKEND=python|rust`; record selected backend, reject invalid selection and report unsupported Rust capabilities explicitly. No automatic fallback.
- **Acceptance:** authenticated status through real Electron; wrong token/foreign origin rejected; init/start/stop and repeated stop; failed start; bounded worker error/panic; disabled-feature startup; copied feature-template build with one composition-root registration; no cross-feature implementation dependencies; shutdown removes owned children. Existing Python sample/CSV path still passes.
- **Proof:** locked Cargo format/clippy/tests; dependency enforcement from `cargo metadata`; dependency-policy check; five demos/full Python evaluation; sealed fixture verification; Electron launch/auth/status/shutdown screenshots and failure checks. Hosted CI remains contingent on a real repository/hosting choice.
- **Constraints:** offline, CPU-only, credential-free; bounded requests, queues and diagnostics; no renderer Node access; no changes to existing checkpoint bytes, schemas/class order or sealed fixture expectations.
- **Rollback:** select Python; restore only owned source changes from a new recovery snapshot. No installed package is changed.
- **Handoff:** exact source hashes, contract version, executed success/failure cases, independent findings, limitations and the separate feature-port approval prerequisite. A native sample-analysis port remains the next gated task, not part of skeleton completion.

## Approval source

[`docs/RUST_MIGRATION.md`](../RUST_MIGRATION.md) states **“Later gates — approval required”**, followed by Phase 2 **“Skeleton”**. The [master prompt](../NETSEC_PRODUCTION_DELIVERY_MASTER_PROMPT.md) rule 9 says **“Honor existing explicit owner approval gates.”** Earlier Phase 1 authorization is preserved; no Phase 2 approval has been recorded. The next decision is approval of S2-SKELETON with the component limitations above, not release authorization.
