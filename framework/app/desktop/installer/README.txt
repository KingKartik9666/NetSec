NetSec includes its own Python runtime, CPU inference libraries, XGBoost
models, SHAP/LIME explanations, and Qwen2.5-0.5B-Instruct Q4_K_M language model.
No Python, pip, Node.js, GPU runtime or API key is required on the user's PC.

Windows packet inspection uses Npcap, which is licensed separately. The setup
wizard offers to download the official Npcap installer only when absent.
It checks the pinned SHA256 and Nmap Software LLC digital signature before
opening Npcap's interactive wizard. The user must accept that vendor's license
and Windows driver approval. Npcap is not embedded or silently installed.
If skipped/offline, samples, CSV analysis and connection monitoring still work.
Download later: https://npcap.com/#download
Licensing: https://npcap.com/oem/

Uninstalling NetSec does not remove Npcap, which other applications may use.
