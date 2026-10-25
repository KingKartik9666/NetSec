# NetSec files and leftovers audit

Audit date: 25 October 2026. Scope: application source, build outputs, migration
evidence and delivery folders under `K:\IMDAI Project`. This is a read-only
inventory apart from this report. No source, model, installation or cache files
were deleted or moved. The installer build may change delivery-folder sizes after
this snapshot.

The useful files substantially outnumber the actual source leftovers. Most space
is in rebuildable outputs, earlier installers and rollback copies. The current
Electron/React/Rust app needs its native assets; the older Python source still
supports retained tests and verification tools.

## Keep: active source and app inputs

| Path under `K:\IMDAI Project` | Observed size | Why it remains useful |
| --- | ---: | --- |
| `rust/netsec/crates/` | 0.22 MiB, 34 files | Current Rust core, model inference, explanations and application runtime. |
| `rust/netsec/Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml` | About 20 KiB | Reproducible native build configuration. |
| `rust/netsec/web/src/` | 0.06 MiB, 16 files | Current React and TypeScript interface. |
| `rust/netsec/web/public/` | 0.44 MiB, 8 files | Current CSS, font, icon and license assets copied by Vite. |
| `rust/netsec/web/index.html`, package/lock/config files, `tests/` | Small | Active renderer entry, build dependencies and regression checks. |
| `framework/app/desktop/` excluding `node_modules/` | About 0.13 MiB | Electron main/preload, native runtime supervisor, startup surface, icons, installer and build validation. |
| `rust/spikes/capture/`, `rust/spikes/xgboost/` | Not measured in this audit | Despite the historical `spikes` name, the current runtime directly imports these Cargo path dependencies. They are not dead code. |
| `tools/build_native_desktop.py`, `stage_native_app.py`, `verify_native_*.py`, `tools/native_export/` | Small | Current build, sealing, verification and approved export workflow. Export source is useful even though it is not an app runtime dependency. |
| `artifacts/application/native_models_v1/` | 427.85 MiB, 27 files | Approved native model exports, schema and sample/reference metadata used by staging and development. Keep the hidden TabPack auxiliary branch: hybrids require it. |
| `artifacts/llm/qwen/` | Not measured in this audit | Actual local language model and its provenance/license. Staging consumes it. |
| `artifacts/rust_migration/phase1/llm/native-b11564/` | Not measured in this audit | Current CPU llama.cpp executable/DLL source for packaging. Historical-looking parent name does not make these binaries obsolete. |
| `framework/models/xgboost_partial/`, `framework/models/packet_live/` and their referenced checkpoints | Not measured in this audit | Separately identified live-compatible models consumed by the current stage script. Do not infer that they are extra full-schema model choices. |
| `framework/rag/knowledge.json` | Small | Current curated local language context. |
| `run_app.py`, `run_desktop.py` | Small | Developer convenience launchers for the native browser and Electron app. |
| `docs/NATIVE_APP.md`, current context and delivery notes | Small | Current instructions, reproducible commands and architectural decisions. |

## Keep for now: development and packaging outputs

| Path under `K:\IMDAI Project` | Observed size | Decision |
| --- | ---: | --- |
| `rust/netsec/target/release/` | 323.68 MiB | Rebuildable, but development launch/staging currently reads `netsec-runtime.exe` here. Keep through final package/install verification. Later Cargo cleanup is possible if rebuilding is acceptable. |
| `rust/netsec/web/dist/` | 0.69 MiB | Rebuildable, but the runtime and stage script serve/package this compiled UI. It is not interchangeable with the TypeScript source. |
| `rust/netsec/web/node_modules/` | 73.05 MiB | Build-only dependencies, recreated from the lockfile. Also supplies React license files during staging. Keep for offline rebuilds. |
| `framework/app/desktop/node_modules/` | 451.38 MiB | Electron development executable and installer tooling. Reinstallable from the lockfile; not part of source commits. |
| `artifacts/application/package_root/` | 944.21 MiB at inventory | Current sealed runtime staging tree. Contains intentional copies of model/LLM/native assets required to make a self-contained installed app. Keep until installer admission and installed-app checks complete. |
| `runtime_assets.json` | Small | Development resource seal. Regenerate after changing compiled/native resources; removing it prevents the development runtime from starting. |
| `artifacts/desktop/electron-builder-cache/` | 15.43 MiB | Reproducible packaging cache; useful for offline builds. Optional later cleanup, not an app dependency. |
| `artifacts/github_delivery/NetSec/` | 2.45 MiB before delivery sync | Isolated Git checkout and its `.git` history. Keep for the authorized source commit/push; this is not a redundant app installation. |

## Keep: evidence and rollback

| Path under `K:\IMDAI Project` | Observed size | Decision |
| --- | ---: | --- |
| `artifacts/app_electron_react_20261024/source_before.zip` and `source_before_manifest.json` | 2.79 MiB combined | Recoverable pre-migration source and hashes, including retired Tauri source/config. Retain. |
| `artifacts/app_electron_react_20261024/` reports, logs and screenshots | Mostly small | Keep successful final proof plus failure records that explain fixes. Names such as `ownership_red_valid` are historical run labels, not proof that a red failure was actually observed. Read each report's result. |
| `artifacts/app_migration_20261024/native_parity_128_final.json` and related parity/baseline evidence | Small | Native model parity and earlier implementation provenance. Not superseded by UI migration. |
| `artifacts/desktop/releases/NetSec-Setup-0.3.0-x64.exe` and matching blockmap | About 647.3 MiB combined | Retain one known-good previous installer until the 0.4.0 installation is verified and rollback policy is satisfied. |
| `artifacts/desktop/install-backups/` | 1,064.61 MiB | Pre-0.3 installation backup. Preserve through current upgrade; later retention can be reduced only after a newer tested rollback exists. |
| New 0.4.0 installer, checksum, installed verification and release notes | Build in progress during inventory | Keep the exact verified distribution and upload it as a release asset. Its bytes are not represented by a source-code commit alone. |

Verification fixture folders contain hard links to native model weights. For
example, `native_features/fixtures/missing_kan_weights/.../random_forest/weights.bin`
and the older fixture's equivalent each show a link count of three. Their apparent
242 MiB sizes are not independent copies. Never edit these linked fixture weights:
that would modify the same underlying model bytes. Deleting a fixture directory
alone also will not reclaim the full reported model size while other links remain.

## Obsolete or rebuildable candidates after delivery

These are candidates, not deletion instructions. Before cleanup, verify the new
installer, update and test the installed app, retain a working rollback, confirm
source/release upload, and ensure no build or app process owns the target.

| Exact path under `K:\IMDAI Project` | Observed size | Classification and condition |
| --- | ---: | --- |
| `artifacts/app_electron_react_20261024/tauri_retired/target/` | 2,242.83 MiB | Obsolete Tauri compiled output. Current Electron build never uses it. Source recovery is already retained in `source_before.zip`. Strong later cleanup candidate. |
| `artifacts/app_electron_react_20261024/tauri_retired/node_modules/` | 15.75 MiB | Retired Tauri CLI dependency cache. Strong later cleanup candidate with source/lock backup retained. |
| `artifacts/app_electron_react_20261024/tauri_retired/gen/` | 0.29 MiB | Generated Tauri schema output. No active build dependency found. |
| `rust/netsec/target/debug/` | 2,291.96 MiB | Reproducible Rust test/debug output. Does not support the current release launch, but deleting it requires recompilation for later tests. |
| `artifacts/desktop/build-env/` | 504.73 MiB | Legacy Python freezing environment. The native build does not use it. Keep only if preserving easy legacy rebuild capability. |
| `artifacts/desktop/pyinstaller-work/` | 74.36 MiB | Legacy PyInstaller intermediates; rebuildable and not used by native packaging. |
| `artifacts/desktop/staging/` | 34.45 MiB | Legacy Python packaging staging. Current staging is `artifacts/application/package_root/`. |
| `artifacts/desktop/ui-releases/` | 1,732.36 MiB | Superseded 0.3.0 UI build and unpacked tree. Its installer differs in size from the final `releases/NetSec-Setup-0.3.0-x64.exe`; do not call them byte-identical. Keep the final old installer for rollback before removing this earlier build. |
| `artifacts/desktop/releases/attempts/NetSec-Setup-0.3.0-before-clean-freeze.exe` | 646.65 MiB | Explicit pre-final installer attempt, superseded by the final 0.3.0 package. Strong later cleanup candidate. |
| `artifacts/desktop/releases/IMDAI-Security-Setup-0.2.0-x64.exe` and blockmap | About 641.5 MiB combined | Older installer; optional archival retention once a tested 0.3/0.4 rollback exists. |
| `artifacts/desktop/backend/` | 249.32 MiB | Old frozen Python runtime. Not consumed by current Electron `extraResources`. Retain only for old-version diagnosis/rebuild/rollback. |
| `rust/netsec/web/static/` | 0.48 MiB | Original imperative native UI source/assets. Active entry is `src/main.tsx`, public assets are `public/static`, runtime serves `dist/static`, tests import `src/state.ts`. No active reference found to this old directory. Preserve snapshot then retire it deliberately. |
| `rust/netsec/web/startup.html` | 518 bytes | Retired Tauri startup page. Electron uses `framework/app/desktop/loading.html`; no active reference found. |
| Browser `profile/` folders and private `bin/netsec-runtime.exe` copies inside completed verification runs | A few MiB per run | Disposable driver artifacts after reports/screenshots and exact runtime hashes are retained. Clean exact run folders only, not the containing evidence directory wholesale. |
| Python `__pycache__` and `.pytest_cache` | Not fully measured | Regenerable caches. Low-value cleanup compared with old build outputs. |

The first nine large candidate rows total about **7.4 GiB of logical file size**.
This is not a promise of disk space reclaimed: NTFS hard links, compression and
files shared with retained artifacts change physical usage. No cleanup was done.

## Older files that should not be mistaken for trash

- `framework/app/server.py`, `desktop_backend.py`, `model_workspace.py`,
  `labels.py`, `static/` and `templates/` are absent from the new runtime path,
  but existing evaluation suites and legacy verification modules import/use them.
  `stage_native_app.py` also reads the Inter font license from this tree. Removing
  the entire old app source would break currently retained checks.
- `experiment/build_desktop.py` and `experiment/frozen_code_verification.py` are
  legacy Python delivery tools. Their tests remain useful historical/regression
  coverage. They should be labelled legacy rather than silently deleted.
- `framework/app/desktop/verify-native-assets.cjs` is useful at build time. The
  current package allowlist also places it in `app.asar`, though the running app
  does not import it. Excluding that small validator from the distributed app is
  optional packaging refinement, not a release blocker.
- Historical architecture/planning documents preserve decisions and evidence.
  Keep the current override visible (`NATIVE_APP.md` and context section 41)
  instead of treating old proposed roadmaps as current requirements.
- `artifacts/training/`, `dataset/`, trusted model checkpoints, sealed study
  exports, `.venv/`, and isolated `external/` repositories are outside this app
  cleanup scope. Locked test data must remain untouched.
- Installed application folders, Start-menu/Desktop links and shared driver or
  system DLL locations belong to the verified upgrade procedure. This inventory
  does not authorize removing any of them.

## GitHub delivery split

Commit maintained source, tests, dependency locks, build scripts and useful
documentation. Upload the verified setup EXE, checksum and plain-language update
notes as GitHub release assets. Keep `node_modules`, Cargo targets, local browser
profiles, intermediate installers, rollback folders, model/research datasets and
the isolated delivery checkout out of Git history. The setup intentionally carries
approved runtime/model assets so users can install the app; that does not require
committing those raw weights or the full workspace into the source repository.

## Evidence used

- Read the complete `docs/PROJECT_CONTEXT.md` and the current native app guide.
- Checked native Cargo path dependencies and HTTP `web/dist` serving paths.
- Checked the React entry/config/tests and searched active source for old Tauri,
  `web/static` and `startup.html` references.
- Checked Electron's package allowlist, resources and build hooks.
- Checked `stage_native_app.py` inputs and legacy Python imports in evaluation
  and experiment modules.
- Counted files and logical sizes using metadata only; large model contents were
  not opened. Inspected hard-link counts for large verification fixture files.
- No training, packet capture, test-array access or installation changes occurred.
