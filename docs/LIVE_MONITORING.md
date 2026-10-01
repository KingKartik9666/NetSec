# Monitor this PC's network traffic

**Connections & traffic** reads actual OS counters and connection states. **Packet inspection** captures passive headers and runs a separate experimental packet classifier; Windows requires Npcap. Observation starts only when you press **Start monitoring**.

## Start and stop

From the project root in the existing Windows environment:

```powershell
.\.venv\Scripts\python.exe run_desktop.py
```

1. In **Live PC traffic**, select the active **Wi-Fi** or **Ethernet** adapter.
2. Choose **Connections & traffic** and press **Start monitoring**.
3. Browse normally and wait for an observation. Inspect traffic rates, current remote endpoints/processes and rule evidence.
4. Press **Stop** before changing adapters or modes, and when finished. Switching tabs alone does not stop monitoring.

Connections mode updates approximately every two seconds. Its packet rates are OS adapter-counter changes, not individual captured headers. It does not run either saved XGBoost model. Some process names may be unavailable because of OS permissions. Displayed connections belong to the selected adapter and have a remote endpoint.

The optional `run_app.py` browser/developer launcher uses the same interface at `http://127.0.0.1:8501`. See [APP.md](APP.md) for dependencies and desktop packaging status.

## Packet inspection and Npcap

Actual Windows packet capture has now executed on the development PC. The earlier missing-driver checks remain historical evidence; they no longer describe the current machine. Capture on another PC still depends on its driver, adapter and permissions.

1. Install [Npcap from the official download page](https://npcap.com/#download) if absent. The desktop setup integration offers a verified optional download and opens the vendor's interactive wizard.
2. Leave **WinPcap compatibility mode** off, following the official Scapy Windows instructions. Wi-Fi monitor mode is unnecessary.
3. Restart IMDAI after driver installation. Select the active adapter, choose **Packet inspection**, then press **Start monitoring**.
4. Wait for a complete approximately ten-second window. Inspect header metadata, rule evidence, packet-model prediction and signed TreeSHAP. Request additional XAI or language wording explicitly if desired.
5. Press **Stop** afterward.

Npcap is not embedded or silently installed: its separate license and administrator driver approval apply. The helper verifies the pinned official installer checksum and Nmap Software LLC signature before opening it. Internet access is required for the optional download; source Python requirements do not install the driver. Skipping Npcap leaves connection monitoring, Samples and CSV analysis available.

Dependency detection is an initial check. Permission/device errors appear when capture starts; if Windows requires elevated capture privileges, launch the app from an Administrator terminal. An absent driver, failed capture, stopped session or stale observation supplies no healthy verdict. No fixture, offline replay or counter mode silently substitutes for successful live packet capture.

## Rules and model outputs

| Output | Meaning |
| --- | --- |
| Waiting / unassessed | No current assessment |
| No recent traffic activity | No packet evidence; no model prediction |
| No rule thresholds exceeded | Benign-looking within the rule observation; no guarantee of safety |
| Suspicious behavior observed | A rule fired; review its evidence |
| Model predicts normal/suspicious | Separate binary classifier prediction with an uncalibrated score |

Rules in `framework/detection/live_rules.py` flag at least 40 current `SYN_SENT` connections. Packet rules flag a SYN-only burst with at least 100 packets, at least 80% SYN-only and at least 50 SYN-only packets/second, or attempts across at least 25 destination ports for one source/destination pair. High download throughput alone is not treated as an attack. Legitimate activity, failed connections or administrative tools can trigger rules.

Rule evidence is separate from ML prediction and TreeSHAP. Rules and model can disagree. Neither confirms malicious intent. Unknown/zero-day detection, encrypted payload analysis and attack attribution have not been evaluated.

## Separate saved-model paths

| Path | Input | Detector | Measured validation |
| --- | --- | --- | --- |
| Samples / Upload CSV | Original 170 DataSense features | Frozen 84-class XGBoost | 80.0533% on 30,000 validation rows |
| Connections mode | OS counters / states | Explicit rules | No detection accuracy study |
| Packet mode | Seven counts per ten-second window | Rules plus binary XGBoost | 95.4169% on 1,571 DataSense windows |

The CSV model needs sensor/log and protocol features that arbitrary packets do not supply. The app uses the independent packet model rather than inventing those inputs.

The packet study ran three actual CUDA training trials on 20,966 eligible exact ten-second windows. The selected depth-7, seed-42 model achieved macro F1 **0.954158**, false-positive rate **1.2064%** and missed-attack rate **7.6364%** on its validation windows. These are a different binary task from the original 84-class result, not measured live-PC performance. Neither reached 99.97%.

Its inputs are total packet count plus independent TCP ACK, FIN, PSH, RST, SYN and URG counts. SYN+ACK counts toward both SYN and ACK. Live inference accepts nonempty windows within one second of ten seconds. Scheduling jitter is outside the dataset's measured validation; delayed windows skip model prediction. Zero-traffic live windows have no model verdict, although training included 278 zero-count windows.

Within-capture partitions are retained. **764 of 1,571 validation windows share a seven-count vector with training after projection**; equal vectors can arise from different source windows. The publisher's exact extractor was unavailable, so mapping follows published count descriptions without a verified source-equivalence claim. Source provenance is DataSense, not independently verified CIC-IIoT2025. Labelled Windows evaluation, unseen-capture generalization and held-out test scoring remain pending.

Both models infer on CPU. Native TreeSHAP verifies bias plus all feature contributions reconstruct the raw class margin; values are not probability percentage points or causal evidence. Optional real LIME uses training-only references and synthetic perturbations, with local fidelity displayed. Global gain/cover statistics and actual Qwen generation remain separately labelled. Read [APP.md](APP.md) for their limitations.

## Actual observations and their limits

`artifacts/ui_polish_verification/npcap_live_proof.json` records an actual Wi-Fi header capture with **640 observed packets**, 40 retained recent headers and a roughly ten-second model window. Rules were **benign-looking**, while the packet classifier predicted **suspicious** with an uncalibrated score of **90.1977%**. TreeSHAP reconstructed the model margin. This proves capture and saved-model execution and records disagreement; the traffic has no independently validated live labels, so it supplies no accuracy estimate or confirmed attack.

The separate offline UQ-IoT replay used the actual header counter and an independent seven-count oracle. A 108-packet window from a benign-source capture was predicted suspicious at **73.3656%**. Preserve this cross-dataset false-alarm observation, whose source designation is not independently verified per-packet annotation. Its proof is `artifacts/live_verification/pcap_replay_confirmed.json`; the original proof is separately retained. Offline replay is distinct from actual Windows capture.

Earlier Edge browser checks and five recorded-replay renderer checks under `artifacts/live_verification/` verified their then-current code, including the earlier missing-driver state. Those records must not be relabelled as new desktop/capture tests.

## Privacy, lifecycle and reproduction

Capture is passive and nonpromiscuous; callbacks keep traffic involving the selected adapter's own IP/MAC addresses. Bounded in-memory state holds at most 120 history entries, 100 connections and 40 recent headers, plus bounded port sets. Payloads/PCAPs are not saved, observations are not sent externally, attacks are not transmitted and devices are not blocked or isolated.

Sessions stop after one hour. Freshness checks and Stop/failure handling clear obsolete verdicts. Packet explanation requests include observation identity; if the window changed, the request is rejected and the user must refresh evidence. Closing the desktop asks its owned service and monitor to stop.

Implementation: `framework/data/live_monitor.py`, `framework/detection/live_rules.py`, `framework/detection/packet_xgboost.py`, `framework/app/server.py`. The packet study, checkpoints, CPU/TreeSHAP proof and projection diagnostics are under `artifacts/training/packet_live_v1/`. Reproducing training uses `python -m experiment.train_packet_model` in the separate CUDA research environment, not during a presentation.

After changes, run the default demo, all four sample commands and:

```powershell
.\.venv\Scripts\python.exe -m pytest evaluation -v
```

Software checks establish contracts and execution, not operational IDS effectiveness. See [APP.md](APP.md), [ARCHITECTURE.md](ARCHITECTURE.md) and [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).
