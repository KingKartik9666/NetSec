# NetSec 0.4 desktop architecture

The active application is **Electron + React/TypeScript + Rust**, following the
owner's 24 October 2026 correction. Tauri has been retired from the active source
and build. The cream/coral theme, typography and existing layout are retained.

## Boundaries

- `framework/app/desktop`: sandboxed Electron window, fixed-action preload,
  authenticated startup, native process supervision and installer configuration.
- `rust/netsec/web/src`: React screens, typed API contracts, analysis/evidence
  state, live observations and window controls. Vite emits `web/dist` only.
- `rust/netsec/crates/runtime`: loopback HTTP, input snapshots, cancellable jobs,
  native workers, explanations, explicit monitoring and local language execution.
- `rust/netsec/crates/models` and `explain`: existing native model and explanation
  implementations. These remain unchanged by the shell/renderer migration.
- Python remains for research, approved model export and development verification.
  No Python interpreter or Python prediction process is packaged.

The ten visible full-schema models are Random Forest, XGBoost, KAN, HEM-derived,
TabM-NE, TabM-Hier, TabM-Hier-NE, TabM + XGBoost, TabM + XGBoost + KAN and
TabPack + XGBoost. TabPack's auxiliary branch is internal, not an eleventh choice.
Separate partial-input/binary live models do not make full-schema models live
compatible. HEM-derived remains explicitly distinct from real HEM.

### Model bundle: 77 classes (since 31 October 2026)

`artifacts/application/native_models_v1` now holds the cascade v3 models
(`artifacts/training/cascade_v3_20261030_v1`, exported by
`tools/native_export/export_v3.py`). They use 445 CSV v3 features and 77 classes,
and the eight nmap recon subtypes form one class, `recon_scan`. CSV input must use
the 445 v3 columns. An old 655-column file is rejected with an error that names
the first missing v3 column, and extra columns are ignored. The default selection is TabM + XGBoost + KAN, the tuning-role pick at
94.82% assessment-role accuracy. That is validation evidence only; the held-out
test was never scored.

Calibration was fitted on the calibration role and adopted only if tuning-role
Brier dropped:
- class probabilities use temperature, vector or identity calibration with
  smoothing α;
- the binary attack score uses a new Platt map. TabPack and TabPack + XGBoost
  keep the raw `1 − p(benign)`;
- HEM has a new vote-correctness map.

Native parity matched the Python oracle on 128 validation rows: 0 changed
decisions, maximum difference 3.1e-6. The former 84-class bundle is archived
under `artifacts/application/archive/native_models_v1_84class_wrapup_20261023/`.

All six explanation methods retain their specific output units and limitations.
Native TreeSHAP is exact for supported trees; neural/hybrid SHAP is approximate,
and the native LIME-style grouped surrogate is not the Python LIME package.
The local language model (Qwen3-4B-Instruct-2507 Q4_K_M, pinned by
`artifacts/llm/model.json`) writes plain English through one loopback llama.cpp
b11564 `llama-server` child, GPU first with the CPU build as fallback; detectors
stay on the CPU. Every sentence is verified against the server-owned evidence
before it is shown and labelled **Model**; failed or missing output is replaced by
labelled NetSec template sentences. No template or synthetic detector replaces
real inference. See [XAI chat](XAI_CHAT.md). Analysis and live discussions pin
immutable evidence; the assistant has no general-purpose tools or authority to
change results.

## Build and run

Build prerequisites: Windows x64, Node.js, the Rust MSVC toolchain pinned in
`rust/netsec/rust-toolchain.toml`, Microsoft C++ build tools/redistributable files,
the project Python environment, and the already-approved native model/LLM assets.
Source checkout alone deliberately does not contain model weights or datasets.

Install build dependencies once (network is needed only for this development step):

```powershell
npm.cmd --prefix rust/netsec/web ci
pnpm.cmd --dir framework/app/desktop install --frozen-lockfile
```

Build and seal the native development/staging resources:

```powershell
.\.venv\Scripts\python.exe tools/build_native_desktop.py
.\.venv\Scripts\python.exe run_desktop.py
```

Build the local Windows installer after verification:

```powershell
.\.venv\Scripts\python.exe tools/build_native_desktop.py --installer
```

Output: `artifacts/desktop/releases/NetSec-Setup-0.4.1-x64.exe`.
The installed application does not require Node, Python, Rust or credentials.
`experiment.build_desktop` is the legacy Python freeze path, not this release's
build command. The installer retains `edu.imdai.security` and per-user upgrade
identity. No Npcap redistributable is bundled; packet capture remains an optional
explicit driver-dependent capability. The ordinary app starts offline; detectors run on the CPU and only the language server uses a free GPU.

## Verification

```powershell
npm.cmd --prefix rust/netsec/web test
npm.cmd --prefix rust/netsec/web run build
node --test framework/app/desktop/tests/native-runtime.test.cjs
.\.venv\Scripts\python.exe -m tools.verify_native_app
.\.venv\Scripts\python.exe -m tools.verify_native_chat --browser --output artifacts/locked_explainer_20261031/verification/chat
.\.venv\Scripts\python.exe tools/eval_explainer.py
.\.venv\Scripts\python.exe tools/verify_native_desktop.py
.\.venv\Scripts\python.exe -m tools.verify_native_ownership --output artifacts/native-ownership-check
.\.venv\Scripts\python.exe run_demo.py
.\.venv\Scripts\python.exe run_demo.py --sample benign
.\.venv\Scripts\python.exe run_demo.py --sample ddos
.\.venv\Scripts\python.exe run_demo.py --sample scan
.\.venv\Scripts\python.exe run_demo.py --sample unknown
.\.venv\Scripts\python.exe -m pytest evaluation -v
```

Run `cargo test --workspace` inside `rust/netsec`. Runtime checks never train,
load locked held-out tests or start packet capture. Synthetic browser race tests
are state regression evidence, not detector accuracy evidence.

## Ownership and packaging

Electron generates an ephemeral 256-bit token and bootstraps an HttpOnly,
SameSite cookie on its own loopback service. The renderer has no Node or raw IPC.
Requests, navigation and new windows are constrained to the owned workspace.
Windows job ownership retires native descendants even on forced runtime exit;
parent-stdin EOF and explicit shutdown handle normal shell termination.

Staging seals compiled React assets, approved native model exports, the language
model and its license, license notices and the native executable. For both llama
folders (`native-b11564` CPU and `native-b11564-cuda`) it stages exactly the files
the runtime pins: `llama-server.exe`, every DLL except the other llama.cpp tools'
entry DLLs, and the folder's license files (`tools/stage_native_app.py`
`closure_names`, mirrored from `language/server.rs` `closure()`). The VC runtime
DLLs are copied beside the native executable and both llama servers. Build
validation rejects changed or unsealed files. The runtime verifies shared assets
at startup and model bytes on model loading; the 2.5 GB GGUF and the CUDA files are
size-checked at startup and hashed by a background pass before the language
server first starts.

Migration evidence and recoverable source snapshot:
`artifacts/app_electron_react_20261024/`. The retired Tauri tree is retained there
for recovery and excluded from source delivery. Existing research artifacts are
unchanged. Direct installation verification on this PC is distinct from clean-VM
qualification, signing and Microsoft Store certification; those are separate gates.

Dependency decisions checked against official documentation on 24 October 2026:
[React build tooling](https://react.dev/learn/build-a-react-app-from-scratch) and
[Vite setup and Node requirements](https://vite.dev/guide/). Versions are pinned in
the renderer and desktop lockfiles; Vite runs only during development/build.
