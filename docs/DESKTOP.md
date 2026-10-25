# NetSec desktop application

> Current desktop architecture (24 October 2026): Electron, React/TypeScript and native Rust inference across ten visible models. [NATIVE_APP.md](NATIVE_APP.md) supersedes historical Python-service, Tauri and four-model delivery directions below. Research workflows remain separate.

The Windows packaging configuration installs NetSec, its embedded Python interpreter,
CPU inference libraries, the original and promoted XGBoost checkpoints, training-only XAI
reference rows, local fonts and a small quantized language model. The receiving
PC does **not** need Python, pip, Node.js, CUDA or an API key. The app runs
locally; Electron manages a private authenticated loopback analysis service and
stops it when the desktop app closes. The current source UI has separate Home,
Live screening, CSV analysis and Samples pages, with XAI and local LLM controls
in every analysis workspace. No monitoring or language generation starts on launch.

## Installing on Windows

The checked UI release is
`artifacts/desktop/ui-releases/NetSec-Setup-0.3.0-x64.exe`. Its packaged executable
passed all 30 desktop checks, including multipage navigation, themed controls,
lifecycle pages, renderer isolation and owned-service shutdown. This release uses
the verified full source snapshot under `artifacts/ui_redesign/package-source`
to remain consistent while other backend work continues. Check records are under
`artifacts/ui_redesign/packaged` and `artifacts/ui_redesign/snapshot-verification.json`.
Existing installers and installed copies retain their previous bytes until the
new setup is run. Future releases require a full checked freeze and Electron
rebuild; changing source or refreshing assets alone cannot update Python routes.

Open a verified rebuilt installer and choose an installation directory. Setup creates desktop and
Start menu shortcuts and registers an uninstaller. It installs per user rather
than changing the system Python. The current development build is unsigned;
signing a public release requires the publisher's own code-signing identity.

Live packet inspection requires Npcap. If missing, the setup wizard offers to
download Npcap 1.89 directly from its publisher over HTTPS, verify its pinned
SHA256 and valid Nmap Software LLC Authenticode signature, then open its normal
interactive wizard. Complete that wizard's license and driver approval prompts.
If skipped or offline, sample analysis, CSV analysis and OS connection monitoring
remain usable. Npcap can be installed later from [the official download
page](https://npcap.com/#download); restart NetSec afterwards.

Npcap is not inside the NetSec installer. Its publisher reserves redistribution
and silent installation for [Npcap OEM licensing](https://npcap.com/oem/).
No OEM license has been supplied for this project. The optional setup step
respects the vendor's interactive installer. Uninstalling NetSec leaves the
separate driver in place because other applications may use it.

## Using the app

1. Launch **NetSec** from source, or from a verified rebuilt shortcut. Home (`/`)
   links to **Live screening** (`/live`), **CSV analysis** (`/csv`) and
   **Samples** (`/samples`). Clicking the NetSec title returns Home.
2. In Live screening, choose your active adapter, select packet inspection and
   Start. Use your computer normally and allow at least ten seconds for the
   first nonempty packet window. Stop when finished.
3. Inspect the separately labelled rule outcome, main model's top-three tentative
   categories, binary packet comparison and exact TreeSHAP contributions. Open the additional evidence
   controls for LIME and native XGBoost statistics.
4. Request the local language explanation to translate that evidence into
   English. Generation runs only on request; the model is released afterwards.
5. Export the observation report, or open Samples / CSV analysis. Read the
   displayed model/input coverage; packet rules, the binary packet checkpoint
   and any explicitly promoted partial-input multiclass checkpoint remain
   separately labelled. See [PARTIAL_INPUTS.md](PARTIAL_INPUTS.md).

The desktop source uses a frameless window with a themed draggable title bar and
minimize, maximize/restore and close buttons. The File/View menu is removed.
Close opens the app's exit screen; cancel returns to the workspace, while confirm
asks the owned service to stop before exiting. The isolated renderer receives
only fixed window/lifecycle methods through a preload bridge: minimize,
toggleMaximize, requestClose, confirmClose, cancelClose, retry, getState and onState.
It does not receive arbitrary IPC access.

Local `/loading`, `/error` and `/exit` pages share the NetSec theme. The desktop
also has a self-contained startup/error shell, so a missing or stopped Python
service can show an error inside the app. In the browser launcher, the same
visual pages cannot replace the browser's own title bar or close an arbitrary
browser tab.

The model is Qwen2.5-0.5B-Instruct Q4_K_M, 491,400,032 bytes, running through
llama.cpp on two CPU threads. It translates structured detector/XAI evidence;
it does not inspect packets itself. It needs no network connection after setup.
Its Apache 2.0 license accompanies the model. Python and library notices are
included under `resources/backend/_internal/THIRD_PARTY_LICENSES`; Electron's
notices accompany the runtime.

These are research models, not a guarantee of safety or a confirmed incident.
The packet model's DataSense validation is not labelled Windows-PC accuracy.
Native SHAP explains class margins; LIME is an approximate local surrogate with
reported fidelity; XGBoost gain/cover are global tree statistics. The generated
English can make mistakes. HEM reproduction, unknown-attack validation and
automated mitigation are separate unfinished research work.

## Rebuilding on Windows

From the repository root with Python 3.11 x64 and Node.js available:

```powershell
python -m experiment.setup_local_llm
python -m experiment.build_desktop --install
npm install --prefix framework/app/desktop
npm run dist:win --prefix framework/app/desktop
```

The repository already contains the frozen training-reference packages; preserve
them. Only when initially creating an unpopulated research workspace, run
`python -m experiment.package_xai_reference` with its matching prepared training
partitions available. That command deliberately refuses to overwrite references.
`setup_local_llm` checks the model digest.
`build_desktop` creates an isolated CPU build environment under
`artifacts/desktop/build-env` and stages copies of the original checkpoints.
Only the staged bundle paths change to `model.ubj`; the original training
bundles and checkpoint bytes remain unchanged. It runs actual inference, LIME,
language generation and authenticated service lifecycle checks against the
frozen executable from an independent working directory before marking the
build ready. Proof is written to `artifacts/desktop/backend-smoke.json`.

For these multipage/lifecycle changes, run a full `build_desktop` freeze and
then rebuild Electron. `--refresh-assets` copies templates and static files
only; it cannot update frozen `server.py` route code. It is therefore insufficient
to bring an older installer up to the current source application.

For source development, use `python run_desktop.py` after installing the desktop
Node dependencies. A portable Windows build is available through
`npm run dist:portable --prefix framework/app/desktop`. The setup EXE is the
preferred deliverable because it provides an uninstaller and shortcuts.

## Linux target

Linux builds must be produced on Linux x64 using its native Python interpreter,
shared libraries and Electron runtime. The packaging hook rejects a Windows
backend in a Linux AppImage. This Windows development machine has not validated
Linux runtime or capture behavior; do not describe the AppImage as tested here.

On Linux with Python 3.11, Node.js and the desktop runtime prerequisites:

```bash
python3 -m experiment.setup_local_llm
python3 -m experiment.build_desktop --install
npm install --prefix framework/app/desktop
npm run dist:linux --prefix framework/app/desktop
```

Npcap is Windows-only. Linux uses the OS packet-capture facilities and appropriate
packet socket permissions. The user/admin must configure those permissions for
the chosen deployment; do not run the Electron browser as root or disable its
sandbox. Until native Linux capture is verified, use CSV/sample analysis and OS
connection monitoring for demonstrations on that platform.

## Release checks

Run the repository's default demo, all four sample options and full evaluation
suite before publishing a rebuilt app. Then test the installed executable with
the development Python environment unavailable, including sample/CSV analysis,
SHAP/LIME, the actual local language model, packet capture and clean shutdown.
Verify setup on a clean supported Windows machine without preinstalled Python;
testing the frozen build on the development PC is not that clean-machine check.
Retain setup SHA256, model hashes, platform manifest and executed test records.

Technical references: [PyInstaller packaging](https://pyinstaller.org/en/stable/usage.html),
[electron-builder NSIS](https://www.electron.build/docs/nsis/),
[Qwen model repository](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF).
