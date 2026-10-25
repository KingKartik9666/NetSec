# NetSec

Current desktop: **Electron + React/TypeScript + Rust native inference**, with ten visible models and the existing theme. See [the native application guide](docs/NATIVE_APP.md) for architecture, build and verification. Python sections below describe retained research and legacy workflows.

A local desktop research app for **PC traffic observation → model prediction → explainable evidence → optional language explanation**.

The desktop offers ten full-schema models, separately identified live-compatible models, OS connection rules and a synthetic architecture demo. Native model inference, explanations and optional Qwen generation run locally on CPU. Model choices and feature compatibility remain explicit; monitoring and language generation start only when requested.

## Open the desktop app

From the existing project environment in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe run_desktop.py
```

The Electron window hosts the React/TypeScript interface and starts its own authenticated native Rust service on a random local port. The service and its native workers close with the window. No browser address, API key or GPU is needed.

For a source checkout, follow [the native application guide](docs/NATIVE_APP.md) to install the frontend and Electron dependencies, build Rust and stage the approved native model assets. Python remains available for research, build helpers and the synthetic demo. The packaged desktop app includes its runtime and local model assets.

Download the verified Windows installer or full app-folder ZIP from [NetSec 0.4.0](https://github.com/KingKartik9666/NetSec/releases/tag/v0.4.0-windows). The installer updates an existing installation; the package runs without a Python installation. See [downloads and checksums](downloads/README.md) and [the update notes](docs/UPDATES_0.4.0.md). This Windows x64 build is unsigned; Linux runtime execution has not been verified here.

## Use the app

- **Live screening:** select an adapter, choose **Connections & traffic** or **Packet inspection**, then press **Start monitoring**. Stop when finished. Connections mode uses OS counters and transparent rules. Packet mode requires Npcap on Windows and aligns seven measured counts into the main model's 170-feature schema, displaying three tentative categories. The binary packet classifier remains a labelled comparison.
- **Samples:** analyze a real DataSense validation observation. Its supplied label remains visible for comparison with the prediction.
- **CSV analysis:** analyze up to 500 compatible rows / 10 MB; download an example first. Recognized partial columns are accepted, blanks remain missing and genuine zeros stay zero. Custom feature combinations are marked unvalidated. Unrelated schemas and arbitrary PCAPs are not compatible CSV inputs.
- **Explanation:** inspect signed TreeSHAP, request LIME and global model statistics, or explicitly ask the local Qwen model to describe the supplied evidence. Generated wording can be wrong; verify it against the actual evidence.

Npcap is a separately licensed driver. The installer integration offers an optional verified download and opens the vendor's interactive wizard; it does not embed or silently install Npcap. See [live monitoring](docs/LIVE_MONITORING.md).

## Research status

| Component | Current scope |
| --- | --- |
| Promoted model | Actual saved 84-class / 170-feature XGBoost; base upgrade **79.3233%** full-input and **70.1464%** seven-count DataSense validation accuracy; seven-count top-three accuracy **82.1133%** |
| Preserved original | Unchanged full-input checkpoint, **80.0533%** validation accuracy; its seven-count missing-input baseline is **0.1910%** |
| Packet model | Separate experimental binary / seven-count XGBoost; **95.4169%** DataSense window validation accuracy |
| Live rules | Explicit OS/packet thresholds; no measured detection accuracy |
| XAI | Exact native TreeSHAP; real LIME approximation; native global XGBoost statistics |
| Language | Optional actual Qwen2.5-0.5B-Instruct Q4_K_M via llama.cpp with bounded local RAG and grounding checks; deterministic reports remain separately labelled templates |
| HEM | Original feature extraction and a separate research adaptation; original trained HEM is not the desktop runtime |
| Unknown attacks / test | Not validated in the desktop models; held-out test scoring remains locked |

These validation figures describe different tasks and are **not measured live-PC accuracy**. The seven-count validation has 1,571 windows and support for 60 of 84 categories. The source dataset is DataSense, not independently verified CIC-IIoT2025. Saved-model reload, real packet replay and Windows Npcap execution have passed; unlabelled PC traffic establishes execution only. Model predictions can disagree with rules. Neither confirms an incident or guarantees safety. The 99.97% target has not been reached. The base upgrade trades a 0.73 percentage-point full-input regression for improved limited-input validation; enhancement experiments remain separately recorded.

## Offline architecture demo

The original synthetic demonstration remains CPU-only, offline and credential-free after dependencies are installed:

```powershell
.\.venv\Scripts\python.exe run_demo.py
.\.venv\Scripts\python.exe run_demo.py --sample benign
.\.venv\Scripts\python.exe run_demo.py --sample ddos
.\.venv\Scripts\python.exe run_demo.py --sample scan
.\.venv\Scripts\python.exe run_demo.py --sample unknown
.\.venv\Scripts\python.exe -m pytest evaluation -v
```

It runs **DemoDetector**, prototype distance-based novelty, simplified feature rankings and **TemplateProvider** on synthetic data. It does not execute HEM, TreeSHAP, LIME or a language model, and supplies no research performance claim.

## Project guide

[Project structure](docs/PROJECT_STRUCTURE.md) · [Desktop and CSV guide](docs/APP.md) · [Live monitoring](docs/LIVE_MONITORING.md) · [Architecture](docs/ARCHITECTURE.md)

Research and provenance: [persistent project context](docs/PROJECT_CONTEXT.md), [HEM integration](docs/HEM_INTEGRATION.md), [partial-input contract](docs/PARTIAL_INPUTS.md), [partial GPU study](docs/PARTIAL_TRAINING.md), [enhancement experiments](docs/PARTIAL_ENHANCEMENTS.md), [full-data study](docs/FULL_DATA_TRAINING.md), [GPU workflow](docs/GPU_TRAINING.md), [training verification](docs/TRAINING_VERIFICATION.md), [cleanup and recovery](docs/CLEANUP.md).

Large data, model weights, environments and experiment records remain local and separate from upstream repositories under `external/`. The app observes passively and recommends human checks; it does not transmit attacks or apply network mitigations.

