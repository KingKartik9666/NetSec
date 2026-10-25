# Windows distribution feasibility and runtime inventory

Checked 20 October 2026. Recommendation: **direct signed x64 NSIS EXE first; Store MSI/EXE as a separately qualified channel**. MSIX remains an alternative requiring its own compatibility probe. No channel is currently release-ready. Existing installers were inspected, not installed or replaced.

## Existing artifacts, not new candidates

| Artifact | Bytes | SHA256 | Signature |
| --- | ---: | --- | --- |
| `artifacts/desktop/ui-releases/NetSec-Setup-0.3.0-x64.exe` | 678,059,223 | `a12e502dfebf9624000e8b9c28dae5b06104fe60d3e413beac1e274d460ff60a` | NotSigned |
| `artifacts/desktop/releases/NetSec-Setup-0.3.0-x64.exe` | 678,061,427 | `756ed3598edc12e3fb89618cd87b83cd7cdcc1e0d74ac412981a6210b717f040` | NotSigned |

The same version/filename has different bytes in two directories. Preserve both and use exact path/hash in any test or release decision. Reusing version 0.3.0 for a changed public artifact would defeat immutable release identity. `installers.json` records fresh Authenticode/hash results.

Existing unpacked UI payload: 1,137,734,592 bytes (~1.06 GiB). Backend directory: 261,433,688 bytes. Qwen directory: 491,411,985 bytes, of which weights are 491,400,032 bytes. Setup download is about 647 MiB. These are old-payload inventories, not sizes of a future Rust/React/four-model release. Peak install/upgrade disk use is unknown; extraction and rollback need additional space.

## Dependency and license matrix

Machine-readable inventory: `artifacts/delivery/20261020/dependency-inventory.json` includes all 47 distribution metadata entries found in the frozen notices manifest, 61 notice files, 304 native binary paths/sizes, source requirements, model status and package-asset existence. This is an inventory, not a license clearance or hash-complete SBOM. Build-environment metadata is not proof that every listed module is reachable or bundled. Native/transitive ownership remains a release task.

| Component | Current role / presence | Rights and remaining work | Owner |
| --- | --- | --- | --- |
| Electron | Source pin 44.5.0; development runtime verified. Chromium/Node included by Electron | MIT notice inspected; preserve Chromium/third-party notices, validate supported patched version and signatures | Desktop |
| electron-builder/NSIS | Builder pin 26.15.3, pnpm lockfile. Per-user setup, selectable directory, no app elevation, uninstall metadata configuration | Build tooling, not a customer prerequisite. Pin/cache provenance and validate actual silent install/return codes | Release |
| Embedded Python | Existing PyInstaller backend; `requirements-desktop.txt` pins PyInstaller 6.22.3 and CPU dependencies | Python notice staged; verify exact frozen interpreter and transitive license obligations, not project `.venv` assumptions | Release |
| NumPy/pandas/SciPy/sklearn/XGBoost CPU/Flask/Werkzeug/psutil/Scapy/LIME | Existing inference/backend dependencies, pinned direct build list; Python/native runtime files present | Preserve notices; inspect transitive license compatibility and vulnerabilities. Source range requirements and build pins are different inventories | Runtime/release |
| llama-cpp-python / llama.cpp | Existing native DLLs and Python wrapper, CPU-only generation | Upstream llama.cpp MIT; inventory wrapper and all DLLs/notices separately, verify actual native loading in package | Runtime/release |
| Qwen2.5-0.5B-Instruct Q4_K_M | Pinned repository revision/hash, local weights and LICENSE | Apache-2.0 model metadata and upstream card; retain notices/attribution. No model download occurred | Release |
| Original/partial/binary XGBoost assets and XAI references | Staged by existing builder | Technical availability is not proof of model/data redistribution rights; obtain provenance/rights determination for weights, included validation observations and training references | Owner/research liaison |
| Four-model manifests/checkpoints | Current source RF manifest missing; KAN/HEM app manifests refer to preserved 71-feature assets | Existing frozen backend has **none** of the four-model manifest/config assets. Builder excludes torch. Current package cannot claim full four-model support | Runtime/release |
| HEM / efficient-kan source | Isolated upstream repositories, MIT license files inspected; AfterImage preserves separate MIT attribution | Include notices only for actual shipped code; neural weights and reference-data rights remain separate. No upstream edits | Runtime/release |
| Inter font | Local font file and license in `framework/app/static/fonts/` | Include exact font license and verify packaged asset | Product/release |
| Npcap | External driver prerequisite; not embedded. Setup checks DLL path, optionally downloads/verifies vendor installer and opens normal wizard; silent mode skips helper | No OEM entitlement supplied. Free/demo redistribution and silent install cannot be assumed. Driver presence != permission/capture success | Owner/release |
| Rust | Four spikes, isolated toolchain, no production Rust runtime | Cargo lockfiles exist per spike; dependency/license audit and production toolchain pin still required | Runtime |
| React/TypeScript | Absent from current renderer | Select minimal maintained toolchain after contracts; no runtime packaging claim yet | Frontend |

Sources inspected: `desktop/package.json`, `installer/installer.nsh`, `setup-npcap.ps1`, `experiment/build_desktop.py:80,285`, `verify-build.cjs`, `requirements-desktop.txt`, frozen notice manifest, local Qwen metadata/LICENSE and upstream license files. The inventory does not traverse or publish research arrays.

## Channel conditions

| Channel | Feasibility and required proof |
| --- | --- |
| Direct EXE | Existing per-user NSIS path is reusable. Create a new versioned candidate with complete lawful runtimes; sign/timestamp and verify; execute clean install/upgrade/uninstall, offline operation and process cleanup on supported Windows without developer tools. Current unsigned old setup is not that candidate |
| Store MSI/EXE | Best initial candidate route, but contingent on silent/offline installer tests, immutable versioned HTTPS hosting, publisher/account/policy fields, complete signing and driver disclosure/review. Current Npcap optional download needs channel-specific review; skipping it in silent mode is not Store acceptance |
| Store MSIX | Not a rename/wrapper shortcut. Driver packaging is unsupported; writable install paths, privilege behavior, child processes, native lookup and redirected app data require separate verification. External optional-driver compatibility needs Microsoft review |

Microsoft's [MSI/EXE upload requirements](https://learn.microsoft.com/en-us/windows/apps/publish/publish-your-app/msi/upload-app-packages) require a standalone offline package, silent installer parameters and an immutable versioned HTTPS URL. [Certification guidance](https://learn.microsoft.com/en-us/windows/apps/publish/publish-your-app/msi/app-certification-process) adds standard-user installation, uninstall/metadata checks and driver/service disclosure. These are pending tests, not inferred from an NSIS configuration.

The effective policy on this report date is [Store policy 7.19](https://learn.microsoft.com/en-us/windows/apps/publish/store-policy-archive/store-policy-7-19), effective 14 October 2025. Section 10.2.9 requires signing the installer and all PE files for the URL-based channel; 10.2.4 restricts non-Microsoft drivers with case-by-case consideration and disclosure. The published [policy 7.20](https://learn.microsoft.com/en-us/windows/apps/publish/store-policies) says it becomes effective **22 October 2026**, after this report date. Recheck the applicable version at submission; do not treat publication date as effective date. The certification page's softer signing wording does not replace the policy requirement.

[Microsoft's MSIX preparation guidance](https://learn.microsoft.com/en-us/windows/msix/desktop/desktop-to-uwp-prepare) documents driver, elevation and install-directory constraints. [Npcap OEM redistribution terms](https://npcap.com/oem/redist) distinguish OEM redistribution/silent setup rights from the free/demo installer. [Official Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) and [llama.cpp license](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE) were checked separately from research-asset rights.

## Environment and owner blockers

Initial target: Windows x64. Only Windows 11 x64 build 26300 development execution is demonstrated here. Electron's [platform README](https://github.com/electron/electron/blob/v44.9.0/README.md) is dependency evidence, not an app support matrix. Windows 10, other Windows 11 versions, ARM64, Linux, low-RAM machines and clean standard-user installs are unverified. Establish minimum OS/CPU/RAM/disk from the actual candidate and native libraries before advertising support.

Unresolved owner fields: legal publisher/copyright holder, app license/EULA choice, support/security contacts, model/reference redistribution rights, signing infrastructure, public privacy/support hosting, Store account/identity and driver-policy position. `IMDAI Research Project` and `UNLICENSED` in package metadata are existing values, not proof of a legal publisher identity. No accounts or certificates were accessed.

Stage 1 installed-location probe is pending: launch backend from a disposable installed directory, discover assets, load native libraries, use per-user writable storage, authenticate local communication, close normally/on EOF and force parent failure. Do not replace the owner's existing installation. No installer test is marked passed here.
