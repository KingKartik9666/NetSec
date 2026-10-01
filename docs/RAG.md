# Offline retrieval for the local explanation

The desktop assistant retrieves small pieces of project knowledge before asking
the actual Qwen model to explain detector/XAI facts. Retrieval supplies background
about SHAP, LIME, model limits and observed rule/model disagreement. It supplies
no new packets, classifier predictions or research results.

The implementation uses the Python standard library on CPU. It needs no network,
API key, embedding model, vector database or additional language model. It runs
as part of an explicitly requested desktop explanation; the synthetic
`run_demo.py` pipeline still uses TemplateProvider without RAG or LLM inference.

## What the knowledge contains

`framework/rag/knowledge.json` contains twelve original project-curated passages.
They summarize checked sections of `docs/APP.md`, `docs/LIVE_MONITORING.md` and
`framework/xai/model_evidence.py`. This is a small local explanation corpus, not
a cybersecurity encyclopedia, imported research paper collection or web crawler.

Every passage has an identifier, title, scope, search terms, text hash and
source file/section/SHA256. The loader checks its schema and exact passage hashes;
the returned corpus hash describes the exact bytes used. The corpus is bounded
to 128 KiB and 64 entries; the current corpus has twelve entries.

Source hashes refer to the checked curation snapshot, not necessarily the latest
version of a source file. Exact originals and a matching manifest were preserved
in `artifacts/desktop/verification/rag_corpus_sources/` before later documentation
edits. Editing the current docs does not silently change the frozen knowledge.

## How retrieval works

`framework/rag/retrieval.py` tokenizes each passage's title, search terms and text,
filters common stopwords, then builds normalized TF-IDF vectors using smoothed
inverse document frequencies and logarithmic term frequencies. It ranks passages
by cosine similarity. Equal scores are ordered by passage identifier.

The query is derived from the supplied model label, XAI method, two leading
feature names and rule status. Actual rule/model disagreement and available LIME
are added as explicit topics. Packet requests search packet/common passages;
CSV requests search CSV/common passages. The retriever does not infer new traffic
facts from its documents.

Only scores at least **0.14** qualify. At most **two** passages are returned, each
at most **300 UTF-8 bytes**, within a **700-byte** combined context including
identifiers. These lexical scores are relevance measures, not probabilities or
security confidence.

If supported detector/XAI evidence has no match or a top score below **0.32**,
the retriever may make one deterministic query expansion toward the existing
SHAP/uncertainty or rule-disagreement topic. There are at most two retrieval
attempts. An unrelated literal query abstains without expansion. Missing or
invalid corpora, including content/hash mismatches, report an error instead of
inventing passages.

## Generation and checks

`framework/llm/local_model.py` supplies the retrieved text and a compact facts list
to actual Qwen2.5-0.5B-Instruct generation. The facts preserve the selected
prediction and strongest signed SHAP directions. Retrieved background is labelled
separately from measured evidence. No relevant match means an empty background;
the audit reports that outcome explicitly.

`framework/rag/grounding.py` performs deterministic lexical checks for the
predicted label, retained feature names, uncertainty, unsupported percentages,
explicit reversed feature directions, relevant rule/model disagreement,
invented source identifiers and truncation. One corrective regeneration is
permitted if the first output fails:
at most two generation attempts. If both fail, the request reports an error and
returns no substitute template explanation.

These are bounded engineering checks. They are not a learned critic, a trained
Self-RAG model or proof that every sentence follows from its citations. They can
miss paraphrases and semantic errors, or reject acceptable wording. Retrieval
relevance, language correctness and security usefulness remain unbenchmarked.

## What the user can inspect

The local-source cards show the retrieved text, identifier and source provenance.
The JSON response/export preserves query, scope, passage/corpus hashes,
similarities, retrieval attempts, generation checks and whether retrieved content
was used. Displayed sources establish which text entered the prompt; they do
not certify each generated claim.

`/api/assistant/status` reports corpus availability without executing the LLM.
`/api/assistant/explain` performs retrieval and actual generation for a validated
sample, uploaded row or explicitly bound live packet window. Desktop routes stay
protected by the per-launch session token. No evidence is sent externally.

The evaluation suite checks scope, bounded context, deterministic retrieval,
abstention, query correction, provenance, invalid-corpus handling and bounded
generation/retry/unload behavior. Actual Qwen requests for sample 0 and sample
162 returned HTTP 200 with retrieved SHAP/LIME passages supplied to generation,
normal completion and passed lexical checks. Both performed the bounded query
correction and completed in one generation attempt. See
`artifacts/desktop/verification/rag-generation.json` for the executed results.
These two observations establish software execution, not a faithfulness benchmark.
Installed RAG/LLM execution must be read from the current release's separate
verification record; it does not establish improved intrusion-detection accuracy.
