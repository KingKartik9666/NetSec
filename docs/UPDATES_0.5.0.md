# What's new in NetSec 0.5.0

The ten models have been replaced with the newer 77-class versions.

- The eight nmap probe types (ARP/TCP/UDP pings, SYN stealth, OS, port and vulnerability scans) are now reported as one class, **recon scan**. The CSV data cannot reliably tell them apart: the best attempt at separating them scored close to chance. Ping sweeps are still reported separately.
- The default model is now TabM + XGBoost + KAN. On the study's assessment rows it scored 94.82%: about 98.7% on traffic scenarios seen in training and about 90% on unseen ones. These are validation results; the held-out test set has not been used.
- Confidence values have been recalibrated for each model. The "is this an attack?" score now has its own calibration, and HEM-derived shows a recalibrated confidence for its vote.
- CSV files must use the new 445-column layout. Use "Download this example CSV" to get the format. Older 655-column files are rejected, and the error names a missing column.
- Fixed: cancelling an analysis could occasionally report that the cancellation was not confirmed even though it had worked.

The theme, the XAI chat, the local language model and CPU-only offline inference are unchanged. The previous model bundle is kept in an archive folder, not shipped. This Windows build remains unsigned.
