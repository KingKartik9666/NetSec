# Phase 1 Windows capture and OS-enumeration spike

This isolated crate is **not a Rust backend or a live-monitor port**. It changes
no runtime routes, UI, model, training worker or release. Default execution is
CPU-only and credential-free. No Npcap installer, capture handle, transmitted
traffic or payload archive is created.

## Implementation

`rust/spikes/capture/` is a standalone Cargo package with its own committed
lockfile and release `panic = "unwind"`. `tools/spike_capture.py` collects a real
psutil snapshot immediately before and after each native probe, then compares
adapter names/addresses/status and TCP/UDP endpoint/PID rows. The resulting
local report includes measurements, differences and explicit unexecuted work.

The Windows adapter uses SDK-generated `windows-sys` types and
`GetAdaptersAddresses`, `GetExtendedTcpTable` and `GetExtendedUdpTable` for IPv4
and IPv6. IPv6 scope suffixes are removed for comparison, matching the current
Python live monitor's address normalization. TCP state names and network-byte
order ports are normalized to psutil's representation. UDP has no remote endpoint
or TCP state. The installed `psutil/_pswindows.py::net_connections` deduplicates
normalized rows with a Python set, so the Rust adapter sorts and deduplicates the
same endpoint/PID rows. The first raw-table comparison, preserved as
`raw-table-comparison.json`, identified six additional duplicate UDP rows rather
than inventing a permission or race explanation. This spike omits process-name lookup and adapter traffic counters;
those remain future port work, not silently fabricated values.

All unsafe lives in `windows_adapter.rs`. OS buffers are aligned, initialized
and bounded to 16 MiB, three resize attempts and 100,000 rows per table. Adapter
and address traversal is bounded to 512 entries; strings to 4,096 units.
Buffer-returned pointers are checked before POD reads. Npcap pointers rely on the
vendor ABI's list/string validity until its matching cleanup function runs.

`GetSystemDirectoryW` identifies the system directory without trusting PATH or
the current directory. Npcap loads dynamically from its `Npcap/wpcap.dll` using
DLL-directory and System32-only dependency search. No SDK/import library or
driver is bundled. Both enumeration and cleanup exports must resolve before
calling the vendor ABI. The library stays loaded until its allocated device list
is freed. Missing files retain the Python missing-driver message exactly;
load/export/enumeration failures return an explicit unavailable state. OS data
does not replace packet capture. The `available_enumeration_only` status means
DLL loading and device enumeration succeeded; it makes no start-time permission
or packet-delivery claim.

The pure replay parser accepts the existing small, invented Ethernet PCAP and
eight sealed windows. It reproduces local IP/MAC ownership, independent TCP bits,
SYN-only versus SYN+ACK and bounded distinct-port counting. It compares full
count dictionaries and rule status against Python. It does not port rule prose,
timers, models or full live lifecycle. Other PCAP encodings/link types and IPv6
extension headers fail explicitly. It is not a complete production parser.

## Reproduction

From the repository root, using an installed compatible Rust toolchain:

```powershell
cargo fmt --manifest-path rust/spikes/capture/Cargo.toml -- --check
cargo clippy --manifest-path rust/spikes/capture/Cargo.toml --all-targets --locked -- -D warnings
cargo test --manifest-path rust/spikes/capture/Cargo.toml --locked
cargo build --manifest-path rust/spikes/capture/Cargo.toml --release --locked
.\.venv\Scripts\python.exe tools/spike_capture.py
```

This checkout uses a project-local portable Rust toolchain under
`artifacts/rust_migration/phase1/toolchain/`. Set `RUSTUP_HOME`/`CARGO_HOME` to its
local `rustup`/`cargo` directories, invoke Cargo by absolute path and use Rust's
bundled GNU GCC as `CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER`. No global PATH or
Windows driver change is required. `--binary <absolute executable>` and
`--output <directory>` make the harness independent of installation paths.
`--repeats` accepts 1–5 snapshots. The missing-driver probe executes even on a
machine with Npcap installed. Unit tests also load an ordinary Windows system DLL
and require missing Npcap exports to fail before any enumeration call.

The exact Windows dependencies are `windows-sys = 0.59.0` and
`libloading = 0.8.6`, using prebuilt `windows-targets = 0.52.6` imports. Initial
0.61.2/0.9.0 builds needed a missing GNU assembler. LLVM-MinGW supplied import
generation, but the combination with Rust's GNU linker then crashed even purely
safe tests when the test harness started a thread. Adding Rust's DLL directory
first did not resolve it. Changing only the binding/import dependencies to these
prebuilt versions allowed all tests to execute. The exact import/linker defect
was not independently diagnosed. Latest raw-DLL bindings therefore remain a
no-go on this recorded toolchain; the pinned, measured bindings are retained.

## Gate evidence

Executed on 12 October 2026 using Rust 1.99.0's native Windows GNU toolchain and
installed psutil 7.0.0. `cargo fmt --check`, `cargo clippy --all-targets --locked
-- -D warnings`, all **10 Rust tests**, and a locked release build passed. Logs
and source hashes are retained under `artifacts/rust_migration/phase1/capture/`.
The parent Phase 1 gate owns the aggregate dependency-license check and complete
Python regression suite; this document does not replace those required checks.

The final harness report is `report.json`; its measured verdict is
**go for enumeration and the covered synthetic replay only**. Production live
capture integration remains unverified and outside this spike's verdict.

| Measured observation | Result |
| --- | --- |
| Actual adapter snapshots | 8/8 names, addresses and up/down statuses equal in each of 3 repeats |
| Actual normalized connection snapshots | 558 native rows each; each equals at least one adjacent psutil snapshot |
| Repeat 1 | psutil before 560, native 558, psutil after 558; native exactly equals after |
| Repeat 2 | before/native/after each 558; native exactly equals before, one changing row differs afterward |
| Repeat 3 | all 558 rows exactly equal before and after |
| Stable-before/after rows missing in Rust | 0 in all three repeats |
| Native rows absent from both adjacent snapshots | 0 in all three repeats |
| Actual Npcap dynamic loading/enumeration | 11 devices, with no error; no capture handle opened |
| Simulated absent Npcap while actually installed | Explicit not-installed result; no capture or substituted devices |
| Missing Npcap exports from an ordinary system DLL | Unit test returns an explicit export error |
| Synthetic replay | 8/8 full count dictionaries and rule statuses equal across 359 invented packets |
| Native adapter call | 6.61–7.25 ms on this PC |
| Native connection-table calls | 1.14–1.28 ms on this PC |
| Dynamic Npcap loading/enumeration | 33.94–37.43 ms on this PC |
| Whole fresh probe process, including launch | 106.67–241.73 ms |
| Synthetic replay parsing and comparison | 1.76 ms, excluding process launch |
| Stripped release executable | 1,209,344 bytes; exact SHA256 in report |

These are small point-in-time software measurements on a busy development PC,
not a sustained performance benchmark, latency guarantee or accuracy result.
The first raw-table comparison remains separately preserved because it revealed
the real UDP multiplicity difference that led to psutil-compatible deduplication.
It is superseded by the final report for normalized parity.

Sequential connection tables can change between calls. The before/after stable
intersection diagnoses differences without asserting an atomic snapshot. Both
implementations read Windows IP Helper data, so matching rows establish software
compatibility rather than independent observational ground truth or detection
accuracy. Address/status matches are also a point-in-time check on this PC.

Opening an Npcap capture handle, passive headers, stop/error/session races,
adapter disappearance during live monitoring and native-process crash isolation
are **not executed** by this enumeration spike. Native DLL crashes cannot be
contained by Rust unwind; an owned process boundary is still required before
production integration. Linux behavior and labelled live accuracy are unverified.
No phase beyond Phase 1 is authorized by these results.

## Primary references

- [Microsoft GetAdaptersAddresses](https://learn.microsoft.com/en-us/windows/win32/api/iphlpapi/nf-iphlpapi-getadaptersaddresses)
- [Microsoft GetExtendedTcpTable](https://learn.microsoft.com/en-us/windows/win32/api/iphlpapi/nf-iphlpapi-getextendedtcptable)
- [Microsoft GetExtendedUdpTable](https://learn.microsoft.com/en-us/windows/win32/api/iphlpapi/nf-iphlpapi-getextendedudptable)
- [Npcap enumeration and cleanup contract](https://npcap.com/guide/wpcap/pcap_findalldevs.html)
- [Npcap API and installation separation](https://npcap.com/guide/npcap-api.html)
- [libloading Windows library API](https://docs.rs/libloading/0.8.6/libloading/os/windows/struct.Library.html)
- [windows-sys SDK binding definitions](https://docs.rs/windows-sys/0.59.0/windows_sys/Win32/NetworkManagement/IpHelper/index.html)
