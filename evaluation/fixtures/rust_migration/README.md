# Python reference corpus for the Rust migration

`v1/` is generated from the current Python implementation with XGBoost 3.1.3,
NumPy 2.2.5, pandas 2.2.3, scikit-learn 1.6.1, lime 0.2.0.1, Flask 3.1.3 and
Scapy 2.6.1. It contains tiny **synthetic** CPU checkpoints, checksum-verified
training references, concrete inputs and their actual Python outputs. No real
traffic capture, GGUF generation, research-array loading or Rust execution is
performed by the generator.

From the project environment:

```powershell
.\.venv\Scripts\python.exe tools/make_golden.py --verify
.\.venv\Scripts\python.exe tools/make_golden.py --output artifacts/rust_migration/regenerated
```

Generation refuses an existing destination. Verify file inventories/checksums
and compare the new corpus with `v1/` before intentionally accepting a new
version. Generation fixes seeds, replay times and the HTTP report clock.
Responses are retained byte for byte; Date, Last-Modified and ETag headers are
omitted from HTTP comparison. HEAD bodies stay empty. The manifest pins source
files, package versions and exact generated assets. Run the tool serially in its
own process because packet replay and HTTP fault injection temporarily patch
module bindings, restoring them before returning.

- `models/`: three-class/16-feature `multi:softprob` checkpoint; schema-1/2
  bundles, missing/default routing, all-class margins/TreeSHAP, equal-score
  ordering, LIME/global statistics, promotion selection and CSV edge cases.
  `adapter-analysis.json` covers the current default adapter and XAI failures.
- `packet/`: independent seven-count/two-class synthetic checkpoint, reference
  and window validation/inference. Its required legacy accuracy field is a
  measured synthetic **training-fit** statistic, explicitly not validation.
- `live/`: header-only invented PCAP using documentation addresses; actual
  Python counting, independent flag bits, ownership filtering and rule boundaries.
- `rag/`: exact checked curated corpus, retrieval/abstention/correction and
  lexical grounding positive/negative cases. Attribution/prose inputs are invented.
- `http/`: complete current route table and exact request/response contracts.
  `default_*` invokes the real LazyXGBoostFacade/XGBoostModelAdapter through an
  in-process transport substitute. Other cases preserve supported concrete
  detector injection. Model-workspace metadata and live I/O are explicitly
  labelled transport doubles; the real missing-GGUF provider is also exercised.

Hardcoded DataSense or live-source wording in unchanged runtime bodies is
compatibility text, not synthetic-fixture provenance. A deterministic public
test token is included to reproduce authentication; it is no real session
credential. Historical actual export data are labelled separately in `legacy/`.

Phase 0 supplies a Python oracle. Rust parity, real-artifact parity, live capture,
actual LLM generation and Linux runtime are **not executed** by this corpus.
Phase 1 must measure the port's numerical differences and record the outcome
before claiming validated TreeSHAP or substituting the runtime.
