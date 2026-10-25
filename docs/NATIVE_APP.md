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

All six explanation methods retain their specific output units and limitations.
Native TreeSHAP is exact for supported trees; neural/hybrid SHAP is approximate,
and the native LIME-style grouped surrogate is not the Python LIME package.
Local Qwen uses llama.cpp on CPU and returns verified execution/grounding evidence
or an explicit failure. No template or synthetic detector replaces real inference.

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

Output: `artifacts/desktop/releases/NetSec-Setup-0.4.0-x64.exe`.
The installed application does not require Node, Python, Rust or credentials.
`experiment.build_desktop` is the legacy Python freeze path, not this release's
build command. The installer retains `edu.imdai.security` and per-user upgrade
identity. No Npcap redistributable is bundled; packet capture remains an optional
explicit driver-dependent capability. The ordinary app starts offline on CPU.

## Verification

```powershell
npm.cmd --prefix rust/netsec/web test
npm.cmd --prefix rust/netsec/web run build
node --test framework/app/desktop/tests/native-runtime.test.cjs
.\.venv\Scripts\python.exe -m tools.verify_native_app
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

Staging seals compiled React assets, approved native model exports, CPU llama.cpp,
Qwen, license notices and the native executable. Its VC runtime DLLs are copied
beside both native executables. Build validation rejects changed or unsealed files.
The runtime verifies shared assets at startup and model bytes on model loading.

Migration evidence and recoverable source snapshot:
`artifacts/app_electron_react_20261024/`. The retired Tauri tree is retained there
for recovery and excluded from source delivery. Existing research artifacts are
unchanged. Direct installation verification on this PC is distinct from clean-VM
qualification, signing and Microsoft Store certification; those are separate gates.

Dependency decisions checked against official documentation on 24 October 2026:
[React build tooling](https://react.dev/learn/build-a-react-app-from-scratch) and
[Vite setup and Node requirements](https://vite.dev/guide/). Versions are pinned in
the renderer and desktop lockfiles; Vite runs only during development/build.
