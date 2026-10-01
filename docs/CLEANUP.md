# Scoped cleanup after app integration

The cleanup removes only project-owned caches and four verified byte-identical
feature export copies. All raw data, prepared train/validation/test arrays,
candidate model checkpoints, training records, source snapshots, provenance,
metadata JSONs, code, environment and upstream repositories are retained.
The original synthetic demo remains necessary for the project's portable
architecture checks. Every training module is used for experiment reproduction;
the audit found no confidently unused source module to delete.

The exact paths, SHA256s, sizes and recovery mapping are recorded under
`artifacts/app_verification/cleanup_applied.json`. Each removed CSV can be restored
by copying its retained byte-identical counterpart to the old filename:

| Removed `dataset/processed/` CSV | Retained identical CSV |
|---|---|
| hem_benign_20260924T023755779285Z.csv | hem_benign_20260924T024705227721Z.csv |
| hem_scan_20260924T023658037277Z.csv | hem_scan_20260924T024705226723Z.csv |
| hem_syn_20260924T023657175540Z.csv | hem_syn_20260924T035209228481Z.csv |
| hem_syn_20260924T024705227721Z.csv | hem_syn_20260924T035209228481Z.csv |

These are **real HEM/Kitsune feature extractions from offline PCAP**, not
synthetic results. `syn` denotes SYN traffic. Original per-run metadata remains
unchanged; metadata paths referring to removed copies are resolved by the table
above. Numeric data is preserved exactly, with no model/data substitution.

The regeneration-only cache targets are `.pytest_cache` and explicit
`__pycache__` folders in our framework, experiment and evaluation directories.
Python imports/pytest recreate these. No recursive sweep of the project,
`.venv/`, `external/`, dataset or training runs is performed.

The recorded PowerShell operation validates resolved workspace containment,
rejects linked targets, checks each duplicate hash against an existing retained
copy, and uses native `Remove-Item -LiteralPath`. Its dry-run plan was reviewed
before applying deletion. This frees roughly 6.7 MB of duplicate CSV data plus
regenerable caches; the much larger research assets are still needed.

## Desktop cleanup update — 30 September 2026

The historical CSV cleanup above is complete. The later desktop audit found no
additional source module that could safely be removed: runtime, synthetic-demo,
training, preprocessing and evaluation files support execution or reproduction.
Research datasets, splits, checkpoints, upstream repositories and environments
remain in their original locations.

Two scoped removals have current operation records:

- The task-downloaded Qwen 1.5B GGUF (1,117,320,736 bytes) was removed after the
  smaller, checksum-verified 0.5B replacement was installed and executed. See
  `artifacts/ui_polish_verification/llm_cleanup.json`. The 0.5B model and its
  provenance/license remain available for offline inference and packaging.
- The redundant 6,062-byte `pcap_replay_final.json` was removed after preserving
  its exact bytes under `artifacts/ui_polish_verification/cleanup_backup/`.
  `pcap_replay_confirmed.json` differs only in its verification timestamp; both
  the canonical proof and the earlier original replay proof remain. See
  `artifacts/ui_polish_verification/proof_cleanup.json` for restoration details.

The later audit identified eleven generated cache directories in
`artifacts/ui_polish_verification/cleanup_plan.json`. Automatic approval review
rejected the validated recursive PowerShell cache-removal operation with
`blocked by policy`, without a more specific reason. That operation was not
retried through another shell or a move; those cache directories remain. The
current audit does not claim this cache cleanup completed.
