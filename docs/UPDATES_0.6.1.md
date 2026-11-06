# What's new in NetSec 0.6.1

The local language model now installs automatically.

- The installer is still about 0.9 GB, because Windows installers of this type cannot hold more than about 2 GB. The 2.4 GB language model ships as a second file, `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`.
- Keep that file in the same folder as `NetSec-Setup-0.6.1-x64.exe`. The installer then copies it in by itself, and the app checks its SHA-256 before first use.
- If the file is missing, NetSec installs and runs without the language model. You can copy the file into `resources\runtime\artifacts\llm\` in the install folder later.
- `SHA256SUMS-0.6.1.txt` lists the hashes of both files.

Everything else is unchanged from 0.6.0, including its known limits.
