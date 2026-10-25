# Project structure

Use the root launchers and this guide for navigation. Research files retain their original locations so checkpoint references, prepared-data manifests and provenance remain reproducible.

```text
IMDAI Project/
├── run_desktop.py                 Electron source launcher
├── run_app.py                     Optional browser/developer launcher
├── run_demo.py                    Separate offline synthetic CLI
├── requirements*.txt             Separate demo, app, desktop and research dependencies
├── framework/                    Our runtime and reusable research code
│   ├── app/
│   │   ├── desktop/               Electron window, build config and Npcap setup helper
│   │   ├── desktop_backend.py     Owned authenticated local service
│   │   ├── server.py              Bounded API, sample/CSV/live/XAI/assistant routes
│   │   ├── labels.py              Readable names with original-label provenance
│   │   ├── templates/             App page
│   │   └── static/                Local styles and UI behavior
│   ├── data/                     CSV contracts, OS/packet monitor, HEM features
│   ├── detection/                CSV XGBoost, packet XGBoost, rules, demo, HEM boundary
│   ├── xai/                      Native TreeSHAP, LIME/statistics, simplified demo XAI
│   ├── llm/                      Actual local Qwen and separately labelled templates
│   ├── models/                   Small frozen model bundles and training-only XAI references
│   ├── pipeline/                 Original synthetic end-to-end composition
│   └── training/                 Research model/tracking utilities
├── experiment/                   Dataset preparation, training, packaging and CLI experiments
├── evaluation/                   Executable software and research-contract checks
├── dataset/
│   ├── demo/                     Synthetic demonstration only
│   ├── processed/                Real offline-PCAP HEM feature exports/provenance
│   └── prepared/                 Real DataSense partitions and derived representations
├── artifacts/
│   ├── training/                 Checkpoints, validation results and source snapshots
│   ├── llm/qwen/                 Actual GGUF weights and model provenance
│   ├── desktop/                  Build environment, native packages and verification
│   ├── app_verification/         Earlier CSV app checks and cleanup records
│   ├── live_verification/        OS checks, real-PCAP replay and UI fixture evidence
│   └── ui_polish_verification/   Desktop refresh baseline and actual Npcap capture proof
├── docs/                         Guides, persistent context and research limitations
└── external/                     Isolated upstream HEM, IoT-Agent and efficient-kan repositories
```

The source DataSense CSV, prepared arrays, GGUF weights and training checkpoints are large local assets. They are not disposable source clutter and should not be committed casually. The configured desktop package stages its runtime resources separately; it does not require relocating the research originals.

## Where to start

| Task | Entry point / guide |
| --- | --- |
| Run the desktop | `run_desktop.py` · [APP.md](APP.md) |
| Observe this PC | `framework/data/live_monitor.py` · [LIVE_MONITORING.md](LIVE_MONITORING.md) |
| Explain a saved prediction | `framework/xai/tree_shap.py`, `model_evidence.py` |
| Run actual language generation | `framework/llm/local_model.py`; optional explicit UI action |
| Demonstrate the portable pipeline | `run_demo.py` · [ARCHITECTURE.md](ARCHITECTURE.md) |
| Reproduce research training | `experiment/` · [FULL_DATA_TRAINING.md](FULL_DATA_TRAINING.md), [GPU_TRAINING.md](GPU_TRAINING.md) |
| Understand original HEM scope | [HEM_INTEGRATION.md](HEM_INTEGRATION.md) |
| Understand earlier decisions | Read [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) completely; its earlier snapshots are historical |
| Review safe cleanup / restore | [CLEANUP.md](CLEANUP.md) and recorded per-operation JSON |

## Component boundaries

- The 170-feature CSV detector, seven-count packet detector, live rules and synthetic DemoDetector have distinct inputs and claims.
- Native TreeSHAP is local signed model evidence. LIME is an approximate local surrogate. Gain/cover are global tree statistics. The synthetic demo's rankings remain simplified XAI.
- Local Qwen performs actual generative inference. TemplateProvider and the research/live template reports are deterministic formatters.
- Original HEM feature extraction, the DataSense closed-set HEM adaptation and original trained HEM inference are different activities. The last is not the desktop runtime.
- Software verification, within-capture validation and unknown/held-out/live generalization studies remain separate evidence.

Keep external code isolated. Preserve checkpoints, prepared data, provenance, negative observations, source snapshots, environments and original research records. Generated caches may be recreated; only exact audited cleanup targets should be removed. This documentation update performs no deletion.

## Runtime migration preparation (12 October 2026)

See [RUST_MIGRATION.md](RUST_MIGRATION.md) for the current-source dependency map,
HTTP compatibility contract, proposed boundaries and approval gates. Phase 0
adds `tools/make_golden.py` with focused helpers `tools/golden_http.py` and
`tools/golden_observations.py`; small synthetic contract fixtures belong under
`evaluation/fixtures/rust_migration/v1/`, while baseline logs/source hashes belong
under `artifacts/rust_migration/phase0/`. Fixtures are software evidence, not
research results. The original structure document is preserved there as
`project-structure-before.md`.

The owner has sanctioned one future layout addition: `rust/`, containing
`app/` and `crates/{core-api,core,shared,adapters,features}`. It is **planned, not
implemented in Phase 0**. Python training/experiments and the separate synthetic
CLI retain their locations. The current Python `framework/core/` model registry,
`framework/detection/adapters/` and Models workspace also form part of the
compatibility baseline; the older tree above does not list every current file.
