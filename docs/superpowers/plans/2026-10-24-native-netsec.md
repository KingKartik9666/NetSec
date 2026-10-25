# Native NetSec implementation plan

> **For agentic workers:** Use executing-plans for the integration and bounded
> delegated implementation slices; verify each result before acceptance.

**Goal:** Retain NetSec's theme while delivering a Tauri desktop shell, Rust
application/inference runtime, ten-model selection/comparison, six honest XAI
methods and actual local Qwen English conversion.

**Architecture:** Three independent libraries for model inference, explanations
and application operations; one composition root owns HTTP/jobs/native ports.
Tauri embeds the existing themed web assets. Python remains an offline research,
export and oracle tool only. No Python child executes app predictions.

**Spec:** Current user request and approved Tauri choice, plus applicable behavior
in docs/NETSEC_PRODUCTION_DELIVERY_MASTER_PROMPT.md and docs/RUST_MIGRATION.md.
Current explicit migration authorization supersedes their historical phase
approval stops. No release publication or installed-app replacement is inferred.

**Tech stack:** Rust, serde/JSON, axum/tokio, csv, checked existing native Rust
XGBoost/llama.cpp/Windows adapters, Tauri/WebView2; plain HTML/CSS/ES modules.
Avoid a new frontend framework, model framework or Python production worker.

## Isolation and constraints

No Git repository exists. Recoverable source hashes/copies are in
artifacts/app_migration_20261024/baseline. Existing app/research sources are
preserved; primary changes are additive under rust/netsec, tools/native_export
and new evaluation tests. Any later launcher edit is backed by the snapshot.
Dataset/test/model studies, upstream repositories and completed evidence remain
untouched. Native model exports have separate artifact paths/provenance.

Defaults remain offline, CPU-only and credential-free. Preserve missing vs zero,
feature/class order, actual calibration semantics, HEM-derived hard votes and
model identity. Models with 655 prepared inputs are not live-compatible merely
because seven packet counts exist. No silent template, model or capture fallback.

## Contracts and ownership

- Model worker: tools/native_export, rust/netsec/crates/models, focused exporter
  tests. Native-export-v1 catalog/typed tensor blobs, ModelCatalog/LoadedModel,
  nullable-input conversion and real-artifact parity; auxiliary TabPack hidden.
- UI worker: rust/netsec/web and dependency-free UI state tests. Theme assets,
  catalog, multi-select, extensive jobs, cancellation, row inspection, six XAI
  methods, actual English generation and explicit live compatibility.
- Explanation worker: rust/netsec/crates/explain. Model-independent bounded
  callbacks, grouped reference perturbations, additivity/fidelity/precision and
  feasibility evidence, deterministic seeds and cancellation.
- Root: workspace/lockfile, runtime contracts, immutable inputs, jobs, CSV,
  loopback transport/security, language/native capture ports, Tauri shell,
  integration, parity, browser/desktop verification and independent review.

Shared HTTP contract: /api/v1/catalog, /samples, /jobs, /jobs/{id},
/jobs/{id}/cancel, /explain, /english, /live/{status,adapters,start,stop},
/settings. Per-model errors preserve other results. Immutable job/input/model
identities bind explanation and generation to the actual analyzed observation.

## Review focus

- Wrong schema/order, nullable data and model mismatch must fail explicitly.
- Hybrid probabilities use raw branches; final calibration occurs afterwards.
- Cancellation and late responses cannot replace newer observations/results.
- Explanation perturbations preserve grouped semantics and measured masks.
- Loopback cross-origin requests, arbitrary paths, hostile CSV/model/LLM text,
  parent/window exit and native resource cleanup require executed checks.

## Tasks and proof

- [ ] Export and port all eight branches and three hybrids; test scalar math,
  corrupted tensors/paths/schema; compare diverse real inputs to frozen Python
  CPU oracle, with exact classes/top ranks and measured probability tolerances.
- [ ] Add bounded immutable jobs, selected/extensive mode, per-model progress,
  failure isolation/cancellation; test mixed success/failure and stale IDs.
- [ ] Add strict CSV/schema parsing with 10 MiB/500-row caps, no guessed fields,
  duplicate headers/invalid flags and nonfinite-value rejection.
- [ ] Implement and label all six XAI methods with meaningful red/green tests;
  use exact native TreeSHAP only for supported tree margins, never as hybrid
  final-output SHAP. Verify actual model explanations, limits and cancellation.
- [ ] Connect real CPU Qwen/llama.cpp to method-neutral measured evidence,
  bounded curated retrieval and grounding/refusal checks. No template fallback.
- [ ] Port live observations/capture lifecycle with explicit intent and scoped
  compatibility; never present enumeration or replay as live capture.
- [ ] Deliver Tauri shell and themed UI; verify actual multi-model job,
  explanation, English result, CSV, errors, keyboard/responsive layout and exit.
- [ ] Run cargo fmt/clippy/tests, native parity/integration, dependency boundary
  checks, all five Python demos and python -m pytest evaluation -v.
- [ ] Independent review of integrated changes; correct material findings,
  record app/runtime packaging evidence and remaining release-only limitations.

Next / upcoming task: native runtime contracts/jobs and integration with the
parallel model, explanation and web modules.
