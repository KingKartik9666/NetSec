# NetSec for Windows

Download the app from the [NetSec 0.4.0 release](https://github.com/KingKartik9666/NetSec/releases/tag/v0.4.0-windows):

- **Installer:** [NetSec-Setup-0.4.0-x64.exe](https://github.com/KingKartik9666/NetSec/releases/download/v0.4.0-windows/NetSec-Setup-0.4.0-x64.exe)
- **Full app folder:** [NetSec-App-0.4.0-win-x64.zip](https://github.com/KingKartik9666/NetSec/releases/download/v0.4.0-windows/NetSec-App-0.4.0-win-x64.zip)
- **Checksums:** [SHA256SUMS-0.4.0.txt](https://github.com/KingKartik9666/NetSec/releases/download/v0.4.0-windows/SHA256SUMS-0.4.0.txt)

The installer updates an existing installation. To use the ZIP, extract the complete `NetSec` folder and run `NetSec.exe` inside it. Both downloads include the native runtime and local model assets for 64-bit Windows.

This update keeps the current theme, adds a React and TypeScript interface, and runs the backend and model inference through Rust. Ten full-schema models are available. See [the update notes](../docs/UPDATES_0.4.0.md) for the other changes.

The package is unsigned. Packet capture needs the separate Npcap driver; samples and CSV analysis work locally without it.

