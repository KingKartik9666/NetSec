# NetSec desktop and saved-model analysis

> Current desktop architecture (24 October 2026): Electron, React/TypeScript and native Rust inference across ten visible models. [NATIVE_APP.md](NATIVE_APP.md) supersedes historical Python-service, Tauri and four-model delivery directions below. Research workflows remain separate.

The Electron app provides local PC observation, saved-model CSV analysis, exact TreeSHAP, optional LIME/model statistics and explicitly requested local Qwen generation. Model inference and language generation run on CPU. Launching the app starts its service, but starts neither monitoring, training nor generation.

## Launch from source

From the project root in the existing Windows environment:

```powershell
.\.venv\Scripts\python.exe run_desktop.py
```

The desktop launcher selects that interpreter and starts Electron in `framework/app/desktop/`. Electron starts `framework.app.desktop_backend` on an ephemeral 127.0.0.1 port, supplies a fresh per-launch session token and injects it into local requests. The renderer uses context isolation, sandboxing and no Node integration. Closing the app asks the service to stop observation and exit; a timeout terminates the child.

The source desktop uses a frameless window with a themed draggable title bar,
minimize, maximize/restore and close controls, without the File/View menu.
Close opens the integrated exit screen; cancel returns to the workspace and
confirm shuts down the owned service. The preload bridge exposes only fixed
minimize, toggleMaximize, requestClose, confirmClose, cancelClose, retry, getState and
onState methods, with no arbitrary IPC interface.

For a fresh source environment, install `requirements-desktop.txt` and `npm install --prefix framework/app/desktop`. Preparing Qwen with `python -m experiment.setup_local_llm` downloads the pinned official 0.5B GGUF and checks its SHA256. Setup needs internet access; subsequent runtime analysis is local. These desktop dependencies are separate from the small demo dependencies and GPU training environment. Platform-specific llama.cpp wheels/build prerequisites may apply to source setup.

The optional browser/developer path remains:

```powershell
.\.venv\Scripts\python.exe run_app.py
```

Open `http://127.0.0.1:8501`; `--port 8502` selects another port. This launcher uses the same local UI/API, without Electron's per-session authentication. Stop it with Ctrl+C. `requirements-app.txt` supports CSV analysis; `requirements-live.txt` adds OS/packet observation. Neither installs Npcap or local Qwen.

## Desktop packaging status

The Windows NSIS configuration bundles a native frozen Python service, CPU inference libraries, trained CSV/packet checkpoints, XAI references and the actual Qwen2.5-0.5B-Instruct Q4_K_M model. Its purpose is installation without Python, pip, Node.js, GPU runtimes or credentials on the destination PC. Current output names use NetSec. Existing installers and installed copies remain older builds until a full Python backend freeze and Electron rebuild are performed and verified. `python -m experiment.build_desktop --refresh-assets` copies UI data only and cannot update the frozen Python routes added for this redesign.

Npcap is separately licensed and is not embedded. If absent, the interactive setup offers an optional official download, verifies the pinned SHA256 and Nmap Software LLC signature, then opens the vendor's normal wizard. Internet access, license acceptance and Windows driver approval are required for that step. Skipping it leaves Samples, CSV and connection monitoring available. Uninstalling NetSec does not remove Npcap.

Linux AppImage packaging is configured. A backend must be built natively for its target OS; the packaging guard rejects a Windows backend for Linux. Linux runtime and AppImage execution have not been tested in this Windows session.

## Presentation flow

Home (`/`) opens first. Its three links and shared navigation open **Live
screening** (`/live`), **CSV analysis** (`/csv`) and **Samples** (`/samples`).
Each analysis page has its own input section plus XAI and local LLM sections;
the NetSec title links back Home. The layout keeps source and result panels
aligned and uses equal outer XAI/LLM cards.

The local `/loading`, `/error` and `/exit` screens use the same theme. Loading
represents application initialization; service errors provide a local recovery
screen. The desktop startup shell can display loading/errors before the Python
service is available. Browser mode retains the browser's native window controls.
Analysis validation failures still report their specific error in the workspace;
they do not substitute synthetic observations or template text for failed model
or LLM execution.

1. Open **Live screening**, select the active adapter and press **Start monitoring**. Connections mode uses actual OS counters and rules. Packet mode uses passive headers and the promoted multiclass model, with the binary packet model labelled as a comparison; read [LIVE_MONITORING.md](LIVE_MONITORING.md) and [PARTIAL_INPUTS.md](PARTIAL_INPUTS.md).
2. In **Samples**, choose a readable validation example and analyze it. Compare the model prediction with the supplied validation label. Raw labels and source-row identifiers remain in JSON/provenance.
3. In **CSV analysis**, download the example, upload a compatible small file and select a row. The server retains uploads in memory.
4. Inspect signed TreeSHAP. Request additional LIME/statistics or the local assistant explicitly. Distinguish exact attribution, approximate surrogate, global statistics and generated wording.
5. Export JSON if needed, press **Stop**, then use the themed close control and confirm exit.

## Preserved original CSV model and schema-1 contract

The preserved original bundle contains `xgboost_joint_1_s42_20260929T182104981183Z`: **80.0533% validation accuracy**, **0.6964633 macro F1**, 480,405 training rows, 84 classes, 170 inputs and 76 boosting rounds. This is the highest single-run accuracy in the original full-data study. The study's three-seed macro-F1-selected configuration remains `joint_5`; selection by validation accuracy is not independent final testing. The source app now selects the explicitly promoted partial-input upgrade, whose results and full-input regression are recorded in [PARTIAL_TRAINING.md](PARTIAL_TRAINING.md).

The source is DataSense, not independently verified CIC-IIoT2025. Validation is within-capture and held-out test scoring remains locked. Included observations are the first two validation rows per supported class, without filtering for correct predictions. They participated in model selection and are not independent test or live traffic. Wrong predictions are displayed.

Upload a UTF-8 CSV of at most **500 rows / 10 MB**, containing either:

- All 71 raw numeric DataSense columns plus `network_protocols_all`, `network_protocols_src`, `network_protocols_dst` as Python list strings, such as `['tcp', 'ip']`.
- All 170 prepared columns with the exact frozen feature names; protocol flags must be 0 or 1.

Column order is normalized. Extra labels, timestamps and identifiers are ignored. Individual missing numeric values remain missing; nonfinite values become NaN as in training. Missing schemas, malformed protocol lists and rows without numeric evidence are rejected. The fixed vocabulary's `UNSEEN` flag is a protocol indicator, not unknown-attack detection.

Arbitrary PCAPs, original 100-feature HEM exports, the synthetic demo's six features and unrelated intrusion datasets are incompatible. The app does not invent missing sensor/log inputs for live packets.

The separately versioned schema-2 upgrade changes this contract: full or partial
recognized columns are accepted, NaNs include missing protocol flags, and seven
live count measurements align into the original 170-column order. Infinite
values, malformed row widths, invalid flags and all-empty rows are rejected.
The active model version and coverage are visible in each report. An upgrade
is selected only after an explicit successful validation promotion; otherwise
the original schema-1 checkpoint stays active. The separate binary packet model
is always labelled as a comparison, not substituted for multiclass inference.
See [PARTIAL_INPUTS.md](PARTIAL_INPUTS.md) and
[PARTIAL_TRAINING.md](PARTIAL_TRAINING.md) for the shared schema, validation scope,
60-class live-validation support and unchanged held-out test lock.

`python -m experiment.package_app` recreates a new small CSV bundle from retained study assets; it refuses to overwrite an existing bundle, verifies manifest/checkpoint identity and checks all 30,000 validation observations without opening held-out test arrays. This research bundle operation is separate from the native desktop installer build.

## What each explanation means

**Native TreeSHAP.** The checkpoint runs with `pred_contribs=True`, `approx_contribs=False`, `strict_shape=True`. For every class the implementation checks:

```text
bias + sum(all feature SHAP contributions) = raw class margin
```

The chart shows the largest absolute signed contributions for the predicted class. Positive values increase its raw margin; negative values decrease it. Exports contain all features and reconstruction error. Values are not probability percentage points, causal effects, incident confirmation or a comparison against benign. Their reference uses the model's tree-cover/path-dependent statistics.

**LIME.** `framework/xai/model_evidence.py` fits a local linear surrogate with 2,048 deterministic perturbations, seed 42, and a checksum-verified 512-row training-only reference for each model. It shows feature conditions, weights, surrogate prediction and local fidelity R². These perturbations are synthetic, not captured traffic. Correlations can break and fidelity can be poor; read R² and compare the surrogate with the model score. Missing CSV columns remain missing in model calls; packet perturbations are rounded and constrained to valid counts.

**Global statistics.** Native XGBoost split count, gain, cover, total gain and total cover describe the trained trees. Cover is Hessian-weighted coverage. These are global statistics, not local attribution or causation. The panel also reports model parameters and training-reference summaries.

**Local assistant.** The optional Qwen2.5-0.5B-Instruct Q4_K_M weights execute through llama.cpp on CPU, two threads, a 2,048-token context and up to 200 output tokens. The model loads only for a requested explanation, checks its pinned hash, and closes after generation. It rewrites compact source/classification facts and the two strongest SHAP directions, including rule/model disagreement where relevant. Raw SHAP and LIME stay available separately as audit evidence; the tiny model does not interpret their mixed numerical units. No template substitutes for failed generation. Generated wording still needs verification; quality and factual faithfulness have not been benchmarked. The recorded request completed normally and proves real inference, not validated security reasoning.

**Offline RAG and bounded correction.** The assistant also retrieves from twelve
project-curated method/limitation passages. Standard-library TF-IDF on CPU selects
at most two scoped passages within a 700-byte background context. One evidence-
topic expansion may correct a weak query. The actual Qwen prompt includes that
background separately from measured facts. Lexical label/name/direction and
uncertainty checks allow one corrective generation; persistent failure returns
an error. Source cards and JSON retain identifiers, hashes and attempts. These
engineering checks are not trained Self-RAG or proof of claim-level citation
support. See [RAG.md](RAG.md).

Deterministic research/live reports remain explicitly labelled templates. The original `TemplateProvider` in `run_demo.py` remains separate.

## Evidence and limitations

The separate packet classifier measured **95.4169% binary validation accuracy** on DataSense windows; this is a different target from the CSV model and not live-PC accuracy. Original HEM inference, unknown/zero-day evaluation, labelled live generalization, held-out test scoring, retrieval/LLM quality evaluation and automatic response remain pending.

Actual Npcap header capture on this PC is recorded in `artifacts/ui_polish_verification/npcap_live_proof.json`: 640 packets, benign-looking rules and a suspicious packet-model prediction. The older UQ-IoT replay also produced a suspicious prediction on a benign-source window. Preserve both observations and their validation limits; neither is a labelled live accuracy experiment.

Actual local Qwen generation before RAG was added is recorded in `artifacts/desktop/verification/local_llm.json`, including CPU settings and unloaded status afterward. That development-PC request took 2.72 seconds overall, with 2.42 seconds in the generation call; the Python process peaked at 802.6 MiB RSS (513.3 MiB above its baseline) and returned to 310.9 MiB afterward. This includes the Python/XAI/request process and is not dedicated model RAM, a RAG timing measurement or a hardware-independent benchmark. `rag-generation.json` records actual retrieval-conditioned Qwen execution for an attack-class sample and a benign-class sample, with checked wording and retained source audit. These two cases are not a reasoning-quality benchmark. Earlier CSV/live browser tests under `artifacts/app_verification/` and `artifacts/live_verification/` describe their recorded source revisions, including the earlier missing-driver state. Current desktop packaging and UI checks must be read from their own verification records.

Run the default demo, all four `--sample` options and `python -m pytest evaluation -v` after changes. Software tests validate execution/contracts; they do not establish research detection accuracy. See [ARCHITECTURE.md](ARCHITECTURE.md), [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) and [CLEANUP.md](CLEANUP.md).
