# NetSec 0.4.0 delivery record

Verified locally on 25 October 2026. The existing per-user NetSec installation
was upgraded from 0.3.0 to 0.4.0 successfully. The installed executable reports
0.4.0 and passed the actual Electron lifecycle and model workflow checks.

## Delivered architecture

Electron 44.5.0, React 19.3.1, TypeScript 7.0.2, Vite 8.3.5 and the native Rust
runtime. The ten visible full-schema models and native explanation implementations
were preserved. The active app has no Python inference process and no Tauri shell.
The existing cream/coral theme is retained.

## Verification recorded

- 610 Python evaluation tests passed, with 17 existing warnings; all five CLI demo runs passed.
- Rust workspace tests passed; two existing unused tensor-field warnings remain.
- Five frontend state tests and four native shell protocol tests passed.
- 38 actual native model, explanation, Qwen, browser and model-isolation checks passed.
- 29 HTTP authentication, input and default-state checks passed.
- Four forced-parent cleanup checks passed.
- 15 desktop checks passed separately in development, the packaged app and the installed app.
- Independent review reproduced and verified the fix for a delayed live-status reply restoring an old verdict after Stop.
- Independent review verified the VC runtime placement beside the Rust executable and rejection of unsealed package files.

Detailed local reports, screenshots, the pre-migration snapshot and installer
records are under `artifacts/app_electron_react_20261024/`. No training, locked
test evaluation or packet capture was performed for these application checks.
The synthetic delayed-response test checks UI state, not detection accuracy.

## Distribution files

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| NetSec-Setup-0.4.0-x64.exe | 774539936 | 618e697064e805ea9fd249fb94718792fc4b3d2cd0c8c2a0ea3368e30e81551b |
| NetSec-App-0.4.0-win-x64.zip | 859798146 | 33bbe93480fa699d8f4e6974efaa639e2d86c298b1f8e61056df8f88633b9548 |

Large distribution binaries belong in GitHub release assets; source, build
instructions, audit and update notes belong in Git. The package is unsigned.
Clean-Windows VM qualification and Microsoft Store certification remain separate
from this machine's successful install and runtime verification.

## Recovery and file audit

The original source is retained in `source_before.zip` and its hash manifest.
The old 0.3.0 installer and the existing user data are retained for rollback.
The intermediate Tauri tree is outside the active build in the recovery folder.
No unrelated cache, research artifact or model was deleted.

See [the file audit](../FILE_AUDIT_20261025.md) for exact keep/cleanup categories
and [the update notes](../UPDATES_0.4.0.md) for the short user-facing summary.

