# IMDAI Project --- Codex Research & Engineering Context

> **Purpose:** This file is the persistent context for Codex/AI coding
> agents working on the IMDAI university research project. Read this
> file before planning, modifying, or evaluating the repository.
>
> **Status date:** 24 September 2026
>
> **Important:** This document consolidates the decisions and research
> direction from the ChatGPT project conversations. It distinguishes
> established project decisions from items that still require
> verification or implementation. Do not turn planned work into claims
> of completed work.

### Current UI implementation (1 October 2026)

The visible application is now named **NetSec**. Home (`/`), Live screening
(`/live`), CSV analysis (`/csv`) and Samples (`/samples`) are separate local
pages. Every analysis workspace retains XAI and explicitly requested local LLM
sections; the NetSec title returns Home. The editorial cream/coral presentation,
local geometry and pill navigation were informed by Gapsy Studio and andhuman.
This is a reference-inspired application redesign, not a certified pixel clone.

Loading, error and exit surfaces share the main app theme. Initialization is
bounded; failed model status produces a recovery screen without substituted
predictions. Missing HTML pages retain HTTP 404; API errors remain JSON. Exit
cancellation preserves current evidence. Confirmed browser exit stops monitoring
and offers Home; confirmed desktop exit shuts down the owned backend.

Electron uses a frameless themed title bar with drag, minimize, maximize/restore
and close controls, and no File/View menu. The sandboxed, isolated renderer has
a narrow preload bridge for fixed window actions and state notifications, with
sender/main-frame validation. The self-contained local shell can recover from a
missing or stopped runtime. These UI changes do not alter the distinctions among
real HEM, DemoDetector, simplified XAI, TemplateProvider, real LLM execution,
synthetic observations and research evaluations.

UI verification records are separate from research results under
`artifacts/ui_redesign/`: browser checks cover responsive alignment, real CSV and
TreeSHAP behavior, loading/error recovery and exit cancellation; desktop checks
cover controls, renderer isolation, service failure/restart and owned shutdown.
See `docs/APP.md`, `docs/DESKTOP.md` and `docs/ARCHITECTURE.md` for the current
routes and launch/build instructions. Editing source does not update existing
installer bytes; a checked full backend freeze and Electron package are required.

------------------------------------------------------------------------

## 1. Executive Summary

The project is an **end-to-end adaptive IoT cybersecurity research
framework**, not a standalone chatbot and not merely an
intrusion-detection classifier.

The intended system combines:

``` text
IoT devices / network traffic
        ↓
Data ingestion
        ↓
Feature extraction and preprocessing
        ↓
ML intrusion detection
        ↓
Known vs unknown / novel attack detection
        ↓
Explainable AI (XAI)
        ↓
Structured security evidence
        ↓
LLM-assisted security reasoning
        ↓
Human-readable explanation
        ↓
Mitigation recommendations
        ↓
Future automated response / policy layer
```

The central architectural rule is:

-   **Detection is performed by ML/HEM.**
-   **Novelty/unknown-attack detection is performed by ML/distance/HEM
    logic.**
-   **Explanation is performed by XAI.**
-   **Reasoning and communication are performed by the LLM.**

The LLM must not be presented as magically detecting attacks from
arbitrary raw traffic. It should consume structured evidence generated
by the security pipeline.

The main research foundation currently selected for the detector is
**HEM --- Hybrid Ensemble Framework for Unknown Attack Detection in IoT
Networks**. Its repository has already been cloned into `external/HEM/`.

The immediate objective is to create a small, credible, runnable
vertical slice that demonstrates the final architecture while preserving
a clean migration path to real HEM inference, real IoT datasets,
stronger XAI, RAG, LLM evaluation, and eventually real-time operation.

------------------------------------------------------------------------

## 2. Current Repository State

At the latest confirmed point, the local project is:

``` text
IMDAI Project/
├── dataset/          # empty
├── docs/             # empty
├── evaluation/       # empty
├── experiment/       # empty
├── external/
│   └── HEM/          # cloned research repository; contains existing code
└── framework/        # empty
```

The local Windows path used by the student is currently:

``` text
K:\IMDAI Project
```

Do **not** hard-code this path in application code. Use `pathlib` and
paths relative to the repository root.

### Directory responsibilities

Use the existing directories rather than creating an unnecessary second
project hierarchy:

-   `dataset/` --- datasets, small demo data, dataset documentation and
    later preprocessing outputs.
-   `docs/` --- architecture, research context, integration notes,
    references, methodology.
-   `evaluation/` --- tests, metrics, benchmark/evaluation scripts.
-   `experiment/` --- runnable experiments and research demos.
-   `external/` --- upstream third-party/research repositories. Keep
    external code isolated here.
-   `framework/` --- our actual reusable framework implementation.

Do not implement our application directly inside `external/HEM/` or
other upstream repositories.

------------------------------------------------------------------------

## 3. Research Question / Project Direction

The project evolved from research into:

-   LLMs for cybersecurity;
-   zero-day / unknown vulnerability and attack detection;
-   cybersecurity benchmark datasets;
-   IoT intrusion detection;
-   explainability;
-   end-to-end IoT security frameworks;
-   LLM reasoning over structured sensor/security evidence.

The chosen direction is broader than "benchmark an LLM for
cybersecurity." The project should become an **end-to-end IoT security
framework** where conventional/ML security components perform detection
and the LLM adds interpretation/reasoning rather than replacing the
detector.

A useful working title is:

> **Adaptive End-to-End IoT Intrusion Detection, Unknown Attack
> Detection, Explainability and LLM Security Reasoning Framework**

Alternative concise title:

> **Adaptive IoT Security Framework with Unknown Attack Detection,
> Explainable AI and LLM-Assisted Analysis**

These are working descriptions, not necessarily the final paper title.

------------------------------------------------------------------------

## 4. Key Research Foundations Discussed

### 4.1 HEM --- primary detector foundation

Repository:

https://github.com/trifle19/HEM

HEM has been selected as the primary research implementation to study
and integrate for the intrusion-detection / unknown-attack side.

The project conversations characterize HEM as a hybrid ensemble approach
involving concepts such as:

-   binary classification;
-   multiclass classification;
-   KNN or distance/similarity information;
-   ensemble decision logic;
-   known attack classification;
-   unknown/novel attack detection.

### Critical integration rule

Do **not** claim "HEM is implemented" or "HEM is running" unless the
actual HEM preprocessing, trained model artifacts and inference logic
are executing.

If pretrained artifacts are unavailable or HEM requires dataset
preparation/training, build an adapter around it and use a clearly
labelled prototype detector for the first vertical slice.

Preferred terminology:

``` text
PrototypeDetector / DemoDetector
```

Do not call a substitute classifier `HEMDetector`.

The framework should make replacement easy:

``` text
DemoDetector
     ↓ later replace
HEMAdapter / real HEM inference
```

without requiring a rewrite of the XAI, LLM or orchestration layers.

### Codex must inspect HEM itself

Before relying on any exact implementation detail, inspect:

-   `external/HEM/README*`
-   requirements/environment files
-   feature extraction
-   preprocessing
-   training
-   evaluation
-   KNN/distance logic
-   ensemble logic
-   model artifact expectations
-   dataset expectations

Write verified integration findings into `docs/HEM_INTEGRATION.md`.

Do not rely on this context document as a substitute for reading the
actual upstream repository.

------------------------------------------------------------------------

### 4.2 IoT-LLM / IoT-Agent

Repository discussed:

https://github.com/NTUMARS/IoT-Agent

This repository/paper is relevant primarily for the **LLM reasoning
architecture**, especially reasoning over real-world IoT/sensor
information.

Concepts of interest include:

-   data loading;
-   domain knowledge;
-   retrieval;
-   prompt construction;
-   structured IoT context;
-   LLM abstraction;
-   evaluation.

If it is useful and not already present, clone it under:

``` text
external/IoT-Agent/
```

Do not merge or blindly copy its entire source tree into our framework.
Treat it as an upstream/reference implementation and adapt only
ideas/components whose licenses and technical role are understood.

------------------------------------------------------------------------

### 4.3 Adaptive end-to-end IoT security using XAI + LLM

Research link discussed:

https://arxiv.org/abs/2409.13177

IEEE link discussed:

https://ieeexplore.ieee.org/document/10811456/

This work is important to the selected project direction because the
architecture connects conventional ML intrusion detection to XAI and
then to an LLM for human-readable interpretation/mitigation.

Conceptual pattern:

``` text
ML intrusion detector
        ↓
XAI (e.g. SHAP/LIME)
        ↓
structured explanation/evidence
        ↓
LLM
        ↓
human-readable analysis
        ↓
mitigation recommendation
```

Our intended contribution/direction is to combine this general
architecture with stronger explicit handling of known vs unknown/novel
IoT attacks, using HEM as the main research foundation on the detection
side.

------------------------------------------------------------------------

## 5. Other Research Threads Already Considered

Earlier project conversations also investigated or mentioned research
areas/papers including:

-   benchmarking LLMs for zero-day vulnerabilities;
-   SECURE-style cybersecurity LLM benchmarking;
-   CyberMetric --- a RAG-oriented benchmark dataset for evaluating LLM
    cybersecurity knowledge;
-   "Exploiting Deep Neural Networks as Covert Channels";
-   recent 2024--2026 work that improves upon older IoT/security/LLM
    frameworks;
-   code availability and whether newer work introduces meaningful
    architectural improvements rather than simply repackaging prior
    systems;
-   applications/SaaS/working systems using techniques from 2024--2026;
-   verification of research claims to reduce hallucinated citations.

These threads are background research. They do **not** override the
currently selected core implementation direction:

``` text
IoT traffic
→ detector / HEM
→ known vs unknown
→ XAI
→ structured evidence
→ LLM reasoning
→ report / mitigation
```

When a paper or implementation from those earlier threads becomes
relevant, verify the exact paper title, publication venue, year,
repository and license before adding it to code or documentation.

------------------------------------------------------------------------

## 6. Research Integrity Requirements

This project is academic work. Accuracy about what is implemented
matters more than making the demo look sophisticated.

### Never make these false equivalences

Do not say:

``` text
RandomForest + NearestNeighbors == HEM
```

Do not say:

``` text
feature_importances_ == complete SHAP implementation
```

Do not say:

``` text
template-generated prose == an actual LLM inference
```

Do not say:

``` text
synthetic demo dataset == experimental benchmark dataset
```

Instead explicitly label each component.

Example:

``` text
Detector mode: PROTOTYPE ML DETECTOR
Novelty mode: PROTOTYPE DISTANCE-BASED DETECTION
XAI mode: SIMPLIFIED FEATURE CONTRIBUTION
Reasoning mode: OFFLINE TEMPLATE PROVIDER
HEM integration: ADAPTER / PENDING TRAINED INFERENCE
Dataset: SYNTHETIC DEMONSTRATION DATA
```

### Experimental results

Never fabricate:

-   accuracy;
-   precision;
-   recall;
-   F1;
-   AUROC;
-   zero-day detection rate;
-   false positive rate;
-   latency;
-   LLM evaluation score;
-   comparisons with research papers.

Only report metrics produced by an actual reproducible experiment.

------------------------------------------------------------------------

## 7. Target Architecture

The desired architecture is modular.

``` mermaid
flowchart TD
    A[IoT Devices / Network Traffic] --> B[Data Ingestion]
    B --> C[Feature Extraction / Preprocessing]
    C --> D[ML / HEM Intrusion Detection]
    D --> E{Attack?}
    E -->|No| F[Benign]
    E -->|Yes| G[Known / Unknown Detection]
    G --> H[XAI Explanation]
    H --> I[Structured Security Evidence]
    I --> J[LLM Security Reasoning]
    J --> K[Human-Readable Explanation]
    J --> L[Mitigation Recommendations]
    L --> M[Future Automated Response / Policy]
```

### Architectural separation

#### Data layer

Responsible for:

-   reading IoT/network datasets;
-   validating schema;
-   preprocessing;
-   feature extraction;
-   future real-time capture adapters.

It should not contain LLM logic.

#### Detection layer

Responsible for:

-   benign vs attack decision;
-   attack family/type classification;
-   confidence where meaningful;
-   known vs unknown/novel behavior.

Long-term implementation: HEM or an improved derivative/integration.

#### XAI layer

Responsible for explaining detector output.

Possible methods:

-   SHAP;
-   LIME;
-   feature importance;
-   per-sample feature deviation/contribution for the MVP.

The XAI layer should explain evidence; it should not invent a second
unrelated classification.

#### Structured security context

This is the contract between detection/XAI and the LLM.

Example:

``` json
{
  "detection": {
    "is_attack": true,
    "attack_type": "DDoS",
    "novelty": "known",
    "confidence": 0.94
  },
  "important_features": [
    {
      "feature": "syn_ratio",
      "value": 0.91,
      "importance": 0.42
    },
    {
      "feature": "packet_rate",
      "value": 8500,
      "importance": 0.31
    }
  ]
}
```

#### LLM layer

Responsible for:

-   interpreting detector evidence;
-   translating XAI into readable cybersecurity reasoning;
-   explaining uncertainty;
-   suggesting defensive mitigations;
-   future retrieval over a cybersecurity knowledge base.

The LLM should be explicitly instructed not to invent packet/network
evidence absent from the structured context.

#### Response/policy layer --- future

Potential future responsibilities:

-   alerting;
-   isolation recommendations;
-   firewall/rate-limit policy suggestions;
-   human approval;
-   automated mitigation.

Do not implement dangerous or irreversible automated response merely for
demo value.

------------------------------------------------------------------------

## 8. Immediate MVP / Teacher Demonstration

The immediate deliverable is a small end-to-end prototype that can be
run with:

``` bash
python run_demo.py
```

The prototype should prioritize reliability and architecture over
research-scale model performance.

A good demo flow:

``` text
[1/5] Loading IoT traffic...
[2/5] Running intrusion detector...
[3/5] Evaluating known/unknown behavior...
[4/5] Generating XAI explanation...
[5/5] Generating security reasoning report...
```

Example output:

``` text
============================================================
ADAPTIVE IoT SECURITY FRAMEWORK — PROTOTYPE
============================================================

Execution mode:
Detector: DEMONSTRATION ML MODEL
Reasoning: OFFLINE TEMPLATE PROVIDER

DETECTION RESULT

Status:
ATTACK DETECTED

Classification:
DDoS

Novelty:
KNOWN

Confidence:
94%

XAI EVIDENCE

Most influential features:
1. syn_ratio
2. packet_rate
3. connection_rate

SECURITY ANALYSIS

The detector identified traffic characteristics consistent with
denial-of-service behavior. The strongest evidence includes...

Recommended defensive actions:
1. Inspect the source traffic.
2. Apply appropriate rate limiting.
3. Isolate suspicious IoT devices if necessary.
4. Preserve logs for investigation.

RESEARCH STATUS

Detector:
Prototype ML detector.

HEM:
Integration boundary prepared; real trained HEM inference pending
required model/data preparation.

Reasoning:
Offline template provider.
============================================================
```

The exact values do not matter for the architecture demo. They must,
however, be generated consistently by the actual running code rather
than hard-coded to impersonate model results.

Useful CLI targets, if easy:

``` bash
python run_demo.py --sample benign
python run_demo.py --sample ddos
python run_demo.py --sample scan
python run_demo.py --sample unknown
```

------------------------------------------------------------------------

## 9. Recommended Repository Architecture

Build toward:

``` text
IMDAI Project/
├── dataset/
│   ├── demo/
│   │   └── sample_traffic.csv
│   └── README.md
│
├── docs/
│   ├── PROJECT_CONTEXT.md
│   ├── ARCHITECTURE.md
│   └── HEM_INTEGRATION.md
│
├── evaluation/
│   ├── __init__.py
│   └── test_pipeline.py
│
├── experiment/
│   ├── __init__.py
│   └── demo.py
│
├── external/
│   ├── HEM/
│   └── IoT-Agent/                 # optional/reference if cloned
│
├── framework/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── demo_detector.py
│   │   └── hem_adapter.py
│   ├── xai/
│   │   ├── __init__.py
│   │   └── explainer.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── provider.py
│   │   └── prompt_builder.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── security_pipeline.py
│   └── models/
│       └── .gitkeep
│
├── .env.example
├── .gitignore
├── AGENTS.md
├── requirements.txt
├── README.md
└── run_demo.py
```

Small technically justified changes are acceptable. Avoid unnecessary
abstraction.

------------------------------------------------------------------------

## 10. MVP Detector Design

If real HEM inference cannot yet run, create a lightweight prototype
detector.

Suggested dependencies:

-   NumPy
-   pandas
-   scikit-learn

Possible synthetic/demo features:

-   `packet_rate`
-   `byte_rate`
-   `syn_ratio`
-   `connection_rate`
-   `unique_destinations`
-   `avg_packet_size`

Possible demonstration classes:

-   BENIGN
-   DDoS
-   PORT_SCAN

Use a deterministic random seed.

A simple MVP can use:

``` text
RandomForestClassifier
        +
NearestNeighbors / normalized distance
```

The classifier demonstrates attack-family classification. The distance
mechanism demonstrates the architectural idea of known vs unknown
behavior.

Again: this is a **prototype**, not HEM.

### Detector output contract

Normalize detector-specific results into a stable structure such as:

``` python
{
    "is_attack": True,
    "attack_type": "DDoS",
    "novelty": "known",
    "confidence": 0.94,
    "detector": "PrototypeDetector"
}
```

This stable contract is important because the XAI/LLM layers should not
care whether the backend is the demo detector or HEM.

------------------------------------------------------------------------

## 11. HEM Adapter Strategy

Create an adapter rather than tightly coupling the whole application to
HEM internals.

Conceptually:

``` python
class BaseDetector:
    def fit(self, X, y):
        ...
    def predict(self, sample):
        ...


class PrototypeDetector(BaseDetector):
    ...


class HEMAdapter(BaseDetector):
    ...
```

`HEMAdapter` should:

1.  locate the external HEM repository safely;
2.  inspect/validate required model artifacts;
3.  expose whether real HEM inference is available;
4.  translate HEM-specific output into the framework's normalized
    detection schema;
5.  fail clearly if required HEM artifacts are unavailable.

Never silently fall back inside `HEMAdapter` and then claim HEM ran.
Fallback selection should be explicit at the application/configuration
level.

------------------------------------------------------------------------

## 12. XAI Strategy

Long-term target:

-   SHAP;
-   LIME;
-   comparison of explanation quality/stability;
-   explanation evaluation.

MVP priority:

-   reliability;
-   low dependency burden;
-   per-sample evidence understandable during a teacher demo.

If SHAP installs and works reliably, it can be used. Otherwise, use a
clearly labelled simplified explanation such as feature importance
combined with per-sample deviation.

Example output:

``` json
{
  "method": "prototype_feature_contribution",
  "important_features": [
    {
      "feature": "syn_ratio",
      "value": 0.91,
      "importance": 0.42
    }
  ]
}
```

Do not let optional SHAP issues prevent the full demo from running.

------------------------------------------------------------------------

## 13. LLM / Reasoning Strategy

The project must be able to run without an API key for the immediate
prototype.

Recommended provider abstraction:

``` python
class LLMProvider:
    def generate(self, context):
        ...


class TemplateProvider(LLMProvider):
    ...


# Later:
class OpenAIProvider(LLMProvider):
    ...

class LocalHuggingFaceProvider(LLMProvider):
    ...

class GeminiProvider(LLMProvider):
    ...
```

### Default MVP mode

``` text
LLM_MODE=template
```

`TemplateProvider` is deterministic and offline. It is not an LLM and
must not be described as one in experimental claims.

Its purpose is to validate the interface and full pipeline before a real
model is introduced.

### Prompt builder

Even if the template provider is used initially, build a real prompt
builder for future LLM providers.

Prompt principles:

-   cybersecurity analyst role;
-   consume only structured detector/XAI evidence;
-   explain why the detector result is plausible;
-   explain known/unknown status;
-   state confidence/uncertainty;
-   interpret important features;
-   provide concise defensive mitigations;
-   do not invent packet evidence;
-   distinguish model output from verified ground truth.

------------------------------------------------------------------------

## 14. Data Strategy

### Immediate demo

Create a small synthetic dataset solely to demonstrate software flow.

Clearly document:

> The demo dataset is synthetic and is not used to claim research
> performance.

### Later research data

The project conversations have considered real IoT intrusion datasets
and HEM's expected datasets. Before selecting final benchmark datasets,
inspect HEM's actual preprocessing/data requirements and verify dataset
licensing, labels and availability.

Likely future work includes integrating established IoT intrusion
datasets such as those used by HEM and recent IoT IDS research. Do not
assume exact compatibility without checking schemas and HEM source code.

### Dataset separation

Keep:

``` text
dataset/demo/
```

separate from real benchmark datasets.

Large datasets should generally not be committed to Git.

------------------------------------------------------------------------

## 15. Evaluation Strategy

The final project needs evaluation at several levels, not just
classifier accuracy.

### Detection evaluation

Eventually measure:

-   accuracy where appropriate;
-   precision;
-   recall;
-   F1;
-   confusion matrix;
-   false-positive rate;
-   per-class performance.

### Unknown-attack evaluation

Especially important:

-   unknown/novel attack detection rate;
-   known-vs-unknown confusion;
-   false unknown rate;
-   threshold sensitivity;
-   behavior when an attack family is withheld from training.

A strong research experiment may use leave-one-attack-family-out or
similar protocols, depending on the chosen dataset and HEM methodology.

### XAI evaluation

Possible later questions:

-   Are explanations stable?
-   Do important features correspond to detector behavior?
-   Does SHAP/LIME agree with simplified methods?
-   Does the explanation help downstream LLM reasoning?

### LLM evaluation

Do not evaluate the LLM solely by "the output sounds good."

Potential later dimensions:

-   faithfulness to supplied evidence;
-   hallucination rate;
-   mitigation relevance;
-   consistency;
-   cybersecurity knowledge correctness;
-   known vs unknown uncertainty handling;
-   performance with/without retrieval;
-   local vs API model comparison.

Earlier research into CyberMetric and other cybersecurity LLM benchmarks
may be relevant here.

### End-to-end evaluation

Eventually evaluate:

``` text
traffic → detection → explanation → reasoning
```

rather than treating each module as unrelated.

------------------------------------------------------------------------

## 16. Planned RAG / Knowledge Layer

RAG is a later-stage component, not required for the immediate demo.

Potential architecture:

``` text
Detector + XAI evidence
        ↓
structured security query
        ↓
retrieval
        ↓
curated cybersecurity knowledge
        ↓
LLM
        ↓
grounded explanation / mitigation
```

Potential knowledge sources should be curated and cited appropriately.
Do not ingest random internet text merely to claim RAG exists.

The earlier IoT-Agent/CyberMetric research is relevant to designing
retrieval and evaluation.

------------------------------------------------------------------------

## 17. Real-Time IoT Integration --- Later Stage

The long-term framework may support:

-   packet capture;
-   flow generation;
-   IoT telemetry;
-   streaming inference;
-   alerting;
-   device isolation recommendations;
-   response policies.

Do not prioritize this before the offline research pipeline works
reproducibly.

A staged progression is preferred:

``` text
CSV/offline benchmark
→ reproducible detector
→ unknown attack experiment
→ XAI
→ LLM
→ RAG
→ end-to-end evaluation
→ streaming / live traffic
→ response layer
```

------------------------------------------------------------------------

## 18. Code Reuse Policy

A major project goal from earlier conversations was to identify code
from the actual research papers/repositories rather than randomly
assembling unrelated GitHub code.

Therefore:

1.  Prefer official repositories associated with the research being
    implemented.
2.  Keep upstream repositories under `external/`.
3.  Record repository URL, paper association and license.
4.  Do not blindly copy code into `framework/`.
5.  Reuse/adapt only components that are understood.
6.  Preserve attribution and licenses.
7.  Document what is upstream and what is our implementation.
8.  Never imply authorship of copied research code.
9.  When newer 2024--2026 papers are considered, identify what is
    genuinely new versus a repackaging of older architecture.

------------------------------------------------------------------------

## 19. Engineering Principles

Use:

-   Python;
-   `pathlib`;
-   type hints;
-   docstrings;
-   small focused modules;
-   deterministic seeds;
-   graceful errors;
-   explicit configuration;
-   CPU-compatible default execution;
-   tests.

Avoid:

-   hard-coded `K:\...` paths;
-   API keys in source;
-   giant monolithic scripts;
-   unnecessary frameworks;
-   premature web UI;
-   premature Docker/Kubernetes;
-   CUDA dependency for the basic demo;
-   notebooks as the only way to run the system;
-   hidden fallback behavior;
-   fake model metrics.

The first implementation should be intentionally small.

A few hundred clean lines that execute end-to-end are more valuable than
thousands of incomplete lines.

------------------------------------------------------------------------

## 20. Immediate Dependencies

Keep the first environment lightweight.

Preferred initial requirements:

``` text
numpy
pandas
scikit-learn
python-dotenv
pytest
```

SHAP is optional for the first demo.

Do not install PyTorch, TensorFlow, Transformers or GPU stacks merely to
make the prototype appear more advanced.

Real model dependencies can be added when a specific research component
requires them.

------------------------------------------------------------------------

## 21. Tests Required for MVP

At minimum verify:

1.  demo dataset loads;
2.  detector returns normalized schema;
3.  obvious synthetic benign/attack examples are handled;
4.  novelty field is present;
5.  XAI returns important features;
6.  reasoning/template provider returns non-empty text;
7.  full pipeline executes;
8.  default demo requires no API key;
9.  default demo requires no network access.

Preferred command:

``` bash
python -m pytest evaluation -v
```

The coding agent should actually execute tests and fix errors rather
than merely generating test files.

------------------------------------------------------------------------

## 22. Configuration / Secrets

Create `.env.example`, not `.env` with real credentials.

Possible future configuration:

``` text
DETECTOR_MODE=prototype
LLM_MODE=template
OPENAI_API_KEY=
```

`.gitignore` should include:

``` text
.env
.venv/
venv/
__pycache__/
*.pyc
.pytest_cache/
```

Do not ignore `.env.example`.

------------------------------------------------------------------------

## 23. Documentation Required

The project should eventually contain:

### `README.md`

Short public-facing overview:

-   problem;
-   architecture;
-   current implementation;
-   how to run;
-   research status;
-   limitations;
-   roadmap.

### `docs/PROJECT_CONTEXT.md`

This document or an updated version.

### `docs/ARCHITECTURE.md`

Detailed component boundaries and data flow.

### `docs/HEM_INTEGRATION.md`

Verified findings from reading actual HEM source.

It should identify:

-   useful modules;
-   dataset requirements;
-   training requirements;
-   model artifacts;
-   inference entry points;
-   feature schema;
-   what can be reused now;
-   what remains to integrate.

### Later documentation

Potentially:

``` text
docs/RESEARCH_REFERENCES.md
docs/DATASETS.md
docs/EVALUATION_PLAN.md
docs/LLM_EVALUATION.md
docs/ROADMAP.md
```

------------------------------------------------------------------------

## 24. What to Tell the Teacher at the MVP Stage

Use technically precise language.

Good explanation:

> The project is an adaptive IoT security framework. HEM is the research
> foundation for the detector and unknown-attack component. The
> framework separates detection, explainability and language-model
> reasoning. The current vertical slice uses a lightweight detector
> through the same interface because full HEM execution requires its
> data/training artifacts. XAI converts the detector decision into
> feature evidence, and the reasoning layer converts that structured
> evidence into an analyst-readable report. The next stage replaces the
> prototype detector with trained HEM inference and evaluates it on real
> IoT intrusion datasets.

Avoid:

> "We implemented HEM"

unless actual HEM inference is running.

Avoid:

> "The LLM detects the cyberattack"

if the detector produced the attack decision.

------------------------------------------------------------------------

## 25. Development Roadmap

### Phase 0 --- current

-   HEM cloned under `external/HEM/`.
-   Other project folders established but initially empty.
-   Research direction selected.
-   Architecture decided.

### Phase 1 --- runnable vertical slice

Implement:

-   synthetic demo dataset;
-   data loader;
-   detector interface;
-   lightweight prototype detector;
-   simple novelty logic;
-   HEM adapter boundary;
-   XAI output;
-   structured security context;
-   offline template reasoning provider;
-   prompt builder;
-   end-to-end pipeline;
-   CLI demo;
-   tests;
-   architecture documentation.

Success criterion:

``` bash
python run_demo.py
```

works reliably.

### Phase 2 --- understand and reproduce HEM

-   inspect exact HEM code;
-   reproduce its environment;
-   obtain required dataset(s);
-   run preprocessing;
-   reproduce training/evaluation;
-   record baseline results;
-   understand unknown-attack protocol.

Do not modify HEM before obtaining a reproducible baseline unless
required for environment compatibility.

### Phase 3 --- integrate real HEM

-   connect HEM inference through `HEMAdapter`;
-   normalize HEM outputs;
-   preserve detector abstraction;
-   compare prototype vs real HEM only as a software sanity check, not
    as a scientific claim unless protocols match.

### Phase 4 --- stronger XAI

-   implement SHAP/LIME where appropriate;
-   generate per-sample explanations;
-   measure explanation stability/faithfulness;
-   pass XAI output into reasoning layer.

### Phase 5 --- real LLM

-   choose one or more actual LLM providers;
-   preserve provider abstraction;
-   benchmark evidence-grounded reasoning;
-   prevent unsupported claims;
-   compare template vs LLM only for communication/reasoning, not
    intrusion-detection accuracy.

### Phase 6 --- retrieval / domain knowledge

-   build curated cybersecurity knowledge base;
-   implement RAG;
-   evaluate whether retrieval improves correctness and mitigation
    quality;
-   use insights from IoT-Agent/CyberMetric-style research.

### Phase 7 --- research experiments

-   establish benchmark datasets;
-   define known/unknown protocols;
-   reproduce HEM baseline;
-   implement proposed improvements;
-   compare under identical splits and metrics;
-   run ablations.

### Phase 8 --- real-time framework

-   flow/packet ingestion;
-   streaming inference;
-   alerting;
-   optional human-approved response/policy layer.

------------------------------------------------------------------------

## 26. Potential Research Contribution Areas

The final novelty has not been frozen. Do not prematurely claim a novel
contribution.

Promising areas to investigate after reproducing the baseline include:

1.  Better integration of unknown-attack detection with explanation.
2.  Evidence-grounded LLM reasoning that explicitly handles
    novel/uncertain attacks.
3.  RAG grounded in cybersecurity knowledge for mitigation
    recommendations.
4.  Evaluation of LLM faithfulness to XAI/detector evidence.
5.  Improved unknown-attack decision logic over the HEM baseline.
6.  Cross-dataset generalization.
7.  More recent 2024--2026 detection or explanation techniques
    integrated into the end-to-end framework.
8.  A reproducible modular framework allowing detector/XAI/LLM
    components to be swapped and benchmarked independently.

Any claimed contribution must be supported by experiments.

------------------------------------------------------------------------

## 27. Questions Codex Should Answer by Inspecting the Repository

Before serious HEM integration, answer these from source rather than
guessing:

1.  What exact HEM repository version/commit is cloned?
2.  What datasets does it expect?
3.  What feature columns are required?
4.  What preprocessing scripts produce those features?
5.  Does the repository contain pretrained weights/models?
6.  What classifiers are actually used?
7.  How exactly is KNN/distance used?
8.  How is the known/unknown threshold selected?
9.  What is the exact ensemble decision logic?
10. What files are the main training and inference entry points?
11. Can inference be imported as a Python module, or does it require
    refactoring?
12. What licenses apply?
13. What dependencies conflict with our lightweight framework?
14. What published metrics can be reproduced locally?
15. What minimal adapter is required to expose normalized framework
    output?

Put verified answers in `docs/HEM_INTEGRATION.md`.

------------------------------------------------------------------------

## 28. Instructions for AI Coding Agents

When using this file as Codex context:

### Before coding

1.  Inspect the repository.
2.  Read this document.
3.  Read `external/HEM/README*`.
4.  Inspect relevant HEM source.
5.  Check Git status.
6.  Do not delete existing work.
7.  Determine whether `external/IoT-Agent/` is already present before
    cloning it.

### While coding

1.  Work inside existing top-level directories.
2.  Keep external code isolated.
3.  Prefer a working minimal implementation.
4.  Clearly label prototype components.
5.  Use normalized interfaces.
6.  Keep default execution offline.
7.  Add tests as functionality is added.
8.  Preserve research provenance.

### Before claiming completion

Actually run:

``` bash
python run_demo.py
python -m pytest evaluation -v
```

If CLI sample selection exists, also run:

``` bash
python run_demo.py --sample benign
python run_demo.py --sample ddos
python run_demo.py --sample scan
python run_demo.py --sample unknown
```

Report:

-   final directory tree;
-   files created;
-   files modified;
-   external repositories cloned;
-   demo result;
-   tests passed/failed;
-   exact detector backend;
-   whether real HEM executed;
-   exact reasoning backend;
-   limitations;
-   next development steps.

Do not claim a command passed unless it was executed successfully.

------------------------------------------------------------------------

## 29. Source / Reference Index from Project Discussions

The following links were explicitly central to the project discussions
and should be verified before formal academic citation:

### HEM

Repository:

https://github.com/trifle19/HEM

Purpose in project:

Primary research/reference implementation for IoT intrusion detection
and unknown/novel attack detection.

### IoT-Agent / IoT-LLM

Repository:

https://github.com/NTUMARS/IoT-Agent

Purpose in project:

Reference architecture for LLM reasoning over IoT/sensor context, prompt
construction, retrieval and evaluation.

### Adaptive IoT Security + XAI + LLM

Preprint:

https://arxiv.org/abs/2409.13177

IEEE page discussed:

https://ieeexplore.ieee.org/document/10811456/

Purpose in project:

Architectural reference for connecting intrusion detection, XAI and
LLM-based human-readable analysis/mitigation.

### Additional research topics already investigated

The project chats also discussed:

-   "Benchmarking LLM for Zero-day Vulnerabilities";
-   SECURE cybersecurity LLM benchmarking;
-   CyberMetric: a benchmark dataset based on retrieval-augmented
    generation for evaluating LLM cybersecurity knowledge;
-   "Exploiting Deep Neural Networks as Covert Channels";
-   2024--2026 papers extending or improving IoT/LLM/cybersecurity
    frameworks.

Before adding formal bibliography entries for these, verify title,
authors, venue, DOI and repository from authoritative sources.

------------------------------------------------------------------------

## 30. Final North Star

Every major implementation decision should support this progression:

``` text
REAL IoT SECURITY DATA
        ↓
REPRODUCIBLE DETECTION
        ↓
UNKNOWN ATTACK RECOGNITION
        ↓
EXPLAINABLE EVIDENCE
        ↓
GROUNDED LLM REASONING
        ↓
USEFUL SECURITY REPORT
        ↓
MEASURABLE END-TO-END EVALUATION
```

The project should not become an LLM wrapper around a CSV.

Likewise, it should not become a standalone IDS that adds an unrelated
chatbot at the end.

The research value is in creating and evaluating a coherent pipeline
where each component has a clear responsibility and where the language
model's output remains grounded in evidence produced by the security
system.

------------------------------------------------------------------------

## 31. Immediate Instruction to Codex

If this is the first Codex session on the repository, proceed in this
order:

``` text
1. Inspect repository tree.
2. Read this context file.
3. Inspect external/HEM thoroughly.
4. Create docs/HEM_INTEGRATION.md from verified source findings.
5. Clone external/IoT-Agent only if useful and missing.
6. Build the minimal framework architecture.
7. Implement a clearly labelled prototype detector if real HEM cannot run.
8. Implement novelty output.
9. Implement lightweight XAI.
10. Implement structured security context.
11. Implement offline TemplateProvider + future prompt builder.
12. Implement end-to-end pipeline.
13. Create synthetic demo data.
14. Create run_demo.py.
15. Add tests.
16. Run the demo.
17. Run tests.
18. Fix failures.
19. Report exactly what is real, prototype, external and pending.
```

**Priority:** a truthful, runnable vertical slice first; full research
reproduction second; improvements and novel experiments only after the
baseline is reproducible.

## 32. Verified implementation update — 30 September 2026

The earlier repository snapshot and HEM-first roadmap above are historical.
Following the user's model-comparison and app requests, the separate DataSense
full-data GPU study trained six families on 480,405 eligible training rows:
GPU standalone forest, XGBoost, genuine KAN, HEM closed-set adaptation, CatBoost
and LightGBM. There were 24 completed trials and one failed additional LightGBM
setting. Highest single-run validation accuracy was 80.0533% for XGBoost joint_1,
seed 42; the three-seed macro-F1-selected XGBoost configuration was joint_5.
Neither 96% nor 99.97% was reached. Test scoring remains locked.

The app now freezes the highest-accuracy XGBoost checkpoint, rather than claiming
original HEM reproduction. `run_app.py` runs actual saved-model inference on CPU
with compatible DataSense CSVs or included validation observations, exact native
TreeSHAP with class-margin additivity checks, and deterministic offline research
template reports. Unknown-attack detection, real LLM inference, live capture and
automated response remain pending. The supplied CSV's provenance is DataSense;
it is not independently verified CIC-IIoT2025. See `docs/APP.md`, the full-data
study's RESULTS files, and `docs/CLEANUP.md` for current execution and preservation
details. The synthetic DemoDetector and original TemplateProvider remain separate
and CPU/offline defaults remain intact.

## 33. Verified live-monitor implementation update — 30 September 2026

The app now defaults to a **Live PC traffic** tab with explicit Start/Stop,
plus the existing Samples and Upload CSV paths. Sample and prediction names
are readable descriptions, such as Normal network activity — Example 1; raw
class labels and source-row identifiers remain in provenance/JSON.

`framework/data/live_monitor.py` reads actual psutil OS adapter counters and
connection states in Connections & traffic mode. This mode was executed on the
development PC's active Wi-Fi adapter. Its separate Live behavior rules flag
observable patterns and use benign-looking/suspicious terminology; no saved
model is executed in connections mode and rule outcomes do not confirm safety
or an incident.

Packet inspection is implemented as opt-in passive Scapy header capture,
nonpromiscuous and limited by the callback to traffic involving the selected
adapter's own IP/MAC addresses. On Windows it requires Npcap. Npcap was absent
on this PC, so actual live packet capture was not verified and the UI refuses
packet Start until that dependency is present. No fixture or OS counter mode is
silently substituted as successful packet capture.

The original 84-class, 170-feature CSV checkpoint remains separate. It cannot
truthfully consume arbitrary PC packets because those packets do not supply all
DataSense sensor/log features. A new **experimental binary packet-only XGBoost**
instead uses seven directly counted packet/TCP flags per ten-second window.
Three actual CUDA trials completed on all 20,966 eligible exact ten-second
training windows from the existing source partition, with 1,571 validation
windows. The selected depth-7, seed-42 run achieved 95.4169% binary validation
accuracy, macro F1 0.954158, false-positive rate 1.2064% and missed-attack rate
7.6364%. These measured DataSense results are a different task from the original
80.0533% 84-class result, not live-PC accuracy or an independently reproduced
research benchmark. The 99.97% target has not been reached.

764 of the 1,571 validation rows have a seven-count feature vector present in
training after projection; source partitions and their original purge are
retained. The publisher's exact extractor was unavailable, so the live mapping
follows published count descriptions without a verified source-equivalence
claim. Zero-traffic live windows receive no model prediction; delayed windows
outside the ten-second tolerance skip the model. Both app models infer on CPU
and use exact native TreeSHAP with raw-class-margin reconstruction checks.
Rules, SHAP and deterministic offline template reports remain distinct. Real
HEM inference, unknown-attack evaluation, actual LLM inference, held-out test
scoring and labelled live Windows evaluation remain pending.

Monitoring starts only by explicit user action and is bounded to one hour,
with bounded in-memory history/header/connection records. No payload/PCAP is
saved, observation is sent to an external service, attack traffic is transmitted,
or mitigation performed. Default synthetic CLI execution remains offline,
CPU-only and credential-free. See `docs/LIVE_MONITORING.md`, `docs/APP.md`,
the packet study under `artifacts/training/packet_live_v1/`, and the separate
live verification artifacts for reproducible execution and limitations.

Recorded checks passed all 95 evaluation tests, the default synthetic demo and
all four sample options, dependency consistency and 16 actual Edge browser
checks. OS-mode Start/Stop and missing-driver blocking were verified, alongside
retained CSV/TreeSHAP/upload/JSON behavior. An offline real UQ-IoT PCAP replay
exercised the live header counter, independent seven-count oracle, saved packet
model and TreeSHAP. However, the selected 108-packet ten-second window from a
benign-source capture was predicted suspicious with an uncalibrated 73.3656%
score. Preserve this cross-dataset false-alarm observation; it is not a
successful Windows capture, independently verified per-packet annotation,
accuracy estimate or evidence of live generalization. See
`artifacts/live_verification/pcap_replay_confirmed.json` and the browser/checks JSON
records for the distinction between executed software and research validation.
The original replay proof remains preserved. A separate five-check browser API
fixture verified seven-feature SHAP rendering, separate rule/model outcomes and
failure-state clearing using that recorded replay result; its backend monitor
stayed stopped and no actual packet capture was started. Lifecycle tests also
verify that session expiry and Stop timeouts clear previous verdicts.

## 34. Verified partial-input and desktop update — 1 October 2026

Sections 32–33 describe earlier snapshots. The source app now selects the
explicitly promoted `partial-v1-fresh_d6_smooth_s42` bundle. The original
84-category/170-feature checkpoint, binary packet comparison, datasets and
experiment evidence remain preserved. Real HEM reproduction and unknown-attack
validation remain separate unfinished research work; test scoring is still locked.

The completed partial-input study contains twelve CUDA configurations and four
additional repeat fits on the existing training partition. The selected fresh
depth-six checkpoint has 85 rounds, learning rate 0.08, square-root class weights
and 0.05 label smoothing. All 480,405 original training observations were retained,
with 83,864 deterministic masked views added. Exact ten-second training has 20,966
rows covering all 84 categories; the seven-count validation has 1,571 rows and
support for only 60 categories. The selected checkpoint scored 79.3233% full-input
accuracy, 70.1464% seven-count accuracy and 82.1133% seven-count top-three accuracy.
Full-input macro F1 is 0.661233 and seven-count macro F1 is 0.387033. Full accuracy
regressed 0.73 percentage points versus the preserved original. These are
DataSense source-validation results, not labelled Windows-PC accuracy.

The separate five-fit CUDA enhancement study compared matched continuation,
capped random oversampling, constrained seven-count SMOTE, conservative leaf
updates and gradient-based sampling. No treatment beat matched-control seven-count
accuracy, so the conditional repeat gate did not open and no enhancement was
promoted. Independent audits verified saved bytes, rounds, CUDA configuration,
training-only augmentation provenance and CPU reload predictions. Do not describe
synthetic or duplicated training views as new independently collected observations.

Full/partial CSVs and completed Npcap windows share the canonical schema. Live
classification directly supplies seven measured counts and 163 NaNs; CSV export
uses the identical vector. Protocol/sensor measurements are never guessed.
Custom combinations and physically invalid seven-count inputs are marked
unvalidated. The UI shows three tentative category hypotheses, model version,
feature coverage and separate binary/rule evidence. Stop and error states clear
model verdicts. Signed native TreeSHAP reconstructs class margins; LIME preserves
unavailable inputs and reports its approximation/fidelity limitations.

Actual Qwen2.5-0.5B-Instruct Q4_K_M executes offline through CPU llama.cpp when
requested. It conservatively joins evidence-derived English facts rather than
performing independent attack detection. A curated local RAG corpus supplies
background, with at most two generation attempts and lexical grounding checks.
No template substitutes for failed inference. The runtime uses two CPU threads,
2048 context tokens, a 200-token output cap and releases model memory afterwards.
These checks are heuristic and do not prove semantic correctness or trained Self-RAG.

Selected-checkpoint reload, real UQ-IoT PCAP replay and actual Windows Npcap
capture passed. A fresh development Electron run passed 61 checks, including
actual Wi-Fi capture, CSV round trips, SHAP/LIME, real Qwen/RAG and shutdown.
Responsive routes and startup/error/recovery were verified separately. The final
Python suite passed 225 tests; the synthetic demo retains DemoDetector, simplified
XAI and TemplateProvider. Default inference remains offline, CPU and credential-free.

NetSec 0.3.0 was rebuilt with an embedded CPU Python service, original/partial/
binary checkpoints, training-only XAI references and the small Qwen weights.
The standalone backend passed actual authenticated inference, LIME, Qwen/RAG and
parent-EOF cleanup without Python on PATH. Packaging gates compare current runtime,
UI, checkpoint, bundle and reference files. Windows installation verification is
recorded separately after the install completes; merely editing source never updates
an older installation. The user explicitly requested upgrading the existing
`%LOCALAPPDATA%/Programs/IMDAI Security` folder. Its old installation was backed up.
