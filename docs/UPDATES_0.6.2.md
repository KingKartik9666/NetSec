# What's new in NetSec 0.6.2

Live screening works again, now with the 77-class live model.

- **Live screening scores your traffic.** In packet-header mode (requires Npcap), NetSec scores each 10-second window of this PC's traffic with the live 77-class XGBoost model. Each verdict comes with its exact TreeSHAP explanation and can be explained in plain English or discussed in the chat.
- **Built to match the training data.** The 252 inputs are computed the way the dataset computed them: the device is matched by MAC address, all its frames count, IPv6 is left out, TCP flags are read in 12 bits, and at most the first 1000 ports per direction are kept. On the dataset's validation rows, the app's features and scores match the training code exactly.
- **Ping sweeps count as recon scans here too.** When a window is called recon scan, the explanation shows both recon scan and ping sweep side by side.
- **No false alarms on idle or unfamiliar traffic.**
  - A window with no traffic from this PC shows "No traffic in this window" and is not scored.
  - When the model is less than 20% confident, the window shows "Unfamiliar traffic" instead of an attack label.

Known limits:
- The live model was trained on recordings from an IoT test lab (DataSense). On this PC's own traffic it was only 9-16% confident in a short test, so most windows will show "Unfamiliar traffic". It has not been validated on real traffic from this PC. Treat any verdict as a hint to look closer, not a detection.
- The language model still ships beside the installer (`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`); keep both files in one folder. This build remains unsigned.
