# Four-model research workspace

> Current desktop architecture (24 October 2026): Electron, React/TypeScript and native Rust inference across ten visible models. [NATIVE_APP.md](NATIVE_APP.md) supersedes historical Python-service, Tauri and four-model delivery directions below. Research workflows remain separate.

The source application adds a **Models** workspace at `/models`, preserving
Home, live screening, CSV analysis, Samples and the separate synthetic demo.
The selector contains exactly Random Forest, XGBoost, KAN and HEM. Choosing a
research model never silently changes the promoted partial-input XGBoost used
by the existing live workspace.

## Run the source application

From the project root in the existing environment:

```powershell
.\.venv\Scripts\python.exe run_desktop.py
```

Open **Models** in the navigation. The optional browser launcher is:

```powershell
.\.venv\Scripts\python.exe run_app.py --port 8501
```

Open `http://127.0.0.1:8501/models`. Default inference is CPU-only, offline and
credential-free. Launching the application starts neither capture, research
training nor language generation. Research CUDA training remains a separate
command and environment.

## Use a selected model

1. Choose a family. Inspect its recorded version, architecture, capabilities,
   input schema and limitations. Availability discovery reads configuration and
   manifests; checkpoints load for their requested model operations.
2. Select a shared validation example and press **Run selected model**. Feature
   names align the shared observation to the selected model's order. Unsupported
   schemas or unavailable full-input measurements produce a precise error.
3. Alternatively, expand **Use your prepared feature values** and upload a
   prepared CSV or enter one ordered JSON row. CSV headers match the displayed
   feature names; labels are ignored. Omitted columns are allowed only when that
   model explicitly supports partial input. Missing values are `null`/blank;
   measured zeros remain zero. CSV limits are 500 rows and 10 MB.
4. For a CSV batch, choose a row from its prediction list. Request its supported
   explanation independently. The source, selected row, model version and raw
   explanation remain inspectable. An unavailable explanation never discards
   an existing prediction or substitutes another model.
5. Read the comparison's protocol and input conditions. Missing metrics remain
   unavailable. Different tasks or undocumented protocols do not justify a
   performance ranking. Validation examples used during selection are not
   independent test observations or labelled live traffic.

The new upload endpoint accepts **prepared numerical columns**. The preserved
CSV workspace additionally understands the existing raw DataSense protocol-list
representation. These input contracts are explicit; the app does not infer
application protocols from ports or manufacture sensor logs.

Random Forest is a fixed control, with its precise implementation recorded in
the manifest. The authentic KAN implementation is preserved; no MLP substitute
is accepted. The HEM dashboard distinguishes the source-compatible closed-set
adaptation from original native Kitsune/AfterImage reproduction. Its ensemble
class vote does not provide a supported probability or evaluated unknown-attack
detector. Versioned manifests are authoritative for the current trained assets;
this interface does not invent metrics or upgrades.

## Runtime boundaries and configuration

`framework/core/registry.py` supervises independent spawned model workers.
`framework/app/model_workspace.py` contains the research HTTP blueprint and
model-specific input boundaries. `framework/app/server.py` retains existing
workspaces and lazily initializes optional services. The default production
path isolates monitoring and local-language execution in their own services.
Explicitly injected concrete detectors/services remain supported for existing
integrations and lightweight tests; those integrations deliberately retain
their original in-process contract.

Configuration lives at `framework/models/four_models/config.json`; the browser
launcher accepts `--model-config path/to/config.json`. Each of the four families
has its own enabled flag, adapter and manifest. Explanation, LIME, language,
calibration and conformal capabilities have separate flags. Disabling a model
does not select another. Disabled uncertainty plugins leave raw predictions
available; their absence must not be described as calibrated probabilities.

The existing explicit `--bundle path/to/bundle.json` flag overrides only the
primary legacy XGBoost bundle. It does not rewrite artifacts or accept uploaded
model binaries. No endpoint allows a request to choose an arbitrary Python
adapter import or executable path.

Worker exceptions, missing dependencies/checkpoints, invalid inputs, timeouts
and native exits return scoped diagnostics. A failed model or explanation leaves
the shell, navigation and unrelated model operations available. Monitor-model
failures retain independent packet counters and rules; an actual failed capture
has no current verdict. Stop, freshness checks and observation IDs prevent old
live evidence from being reused. Optional local Qwen/RAG inference remains
explicitly requested and returns no fake template substitute.

## HTTP contracts

| Endpoint | Purpose |
| --- | --- |
| `GET /api/models` | Four-family catalog; per-model availability and metadata |
| `GET /api/models/comparison` | Recorded metrics and comparable-protocol status |
| `GET /api/models/<id>/status` | Selected model's declared metadata/capabilities |
| `POST /api/models/<id>/predict` | `{"sample_id": 0}` or `{"rows": [[...]]}` |
| `POST /api/models/<id>/upload` | Multipart prepared CSV field `file` |
| `POST /api/models/<id>/explain` | One ordered row plus optional declared `method` |

Identifiers are `random_forest`, `xgboost`, `kan`, `hem`. Prediction responses
include `model_id`, `predictions`, `source`, `metadata` and aligned `features`.
Explanation responses carry the same model/source boundary. Scoped errors
include `model_id`, `capability`, `status` and `error`; incompatible requests
return HTTP 400, unavailable/failed worker capabilities HTTP 503. Invalid uploads
over the application limit return HTTP 413. Existing routes, same-origin checks,
trusted hosts, CSP and desktop per-launch authentication remain in force.

## Verification evidence and release scope

`evaluation/test_model_workspace.py` deliberately injects unavailable model
workers, explanation errors, missing metrics, optional-module removal and
capture-model failures. It exercises real Flask/input/lifecycle behavior around
small transport doubles. Tiny real CPU XGBoost fixtures verify that lost
TreeSHAP does not remove packet predictions, and real native TreeSHAP can still
feed the assistant when LIME is absent. A simulated generation response proves
that HTTP boundary only; it is not claimed as real language inference.

Browser checks and screenshots are recorded separately under
`artifacts/four_model_app_verification/`. The initial actual source check covered
27 checks: all four selector entries, scoped unavailable states, five pages at
1440/740/390 pixels, no JavaScript errors and preserved real primary XGBoost and
TreeSHAP. That first record ran before new research manifests were available;
it establishes graceful unavailable behavior, not four trained-model success.
Later saved-model checks must be reported with their own actual records.

After source changes, run the default demo, benign/ddos/scan/unknown options and
the complete evaluation suite. Software checks establish execution and
contracts, not detection accuracy or operational safety.

This source extension does not update an existing installed app or installer.
Shipping it requires a complete backend freeze, Electron package and independent
installed checks. Existing models, prepared data, releases and evidence remain
preserved. Native Linux execution, labelled live generalization, held-out test
evaluation and original unknown-attack HEM reproduction require their own proof.
