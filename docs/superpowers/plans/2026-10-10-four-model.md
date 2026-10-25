# Four-model extension implementation plan

> For agentic workers: execute incrementally with failure tests and an independent final review.

**Goal:** Preserve NetSec and deliver an isolated four-family research workspace with fresh common-dataset training.
**Architecture:** Add typed adapters and a process-isolated registry around the existing application. Keep research training separate from CPU app inference and optional XAI/language services.
**Tech stack:** Python 3.11, Flask, multiprocessing, existing XGBoost/Torch/efficient-kan/HEM, local HTML/CSS/JS.
**Spec:** docs/superpowers/specs/2026-10-10-four-model-design.md

## Global constraints

- Exactly four families; no model substitution; no RF optimization.
- Original sources, models, data and evidence preserved; upstream code unchanged.
- Default runtime offline, CPU and credential-free; research CUDA required except explicitly documented conventional RF.
- Same eligible training rows and validation rows; test remains locked.
- Optional failures do not prevent prediction or application boot.

## Review focus

- Native crash/hang and concurrent requests: kill/reap only the affected worker.
- Missing explanation: render the genuine prediction without invented evidence.
- Missing schema or corrupted references: reject only that model operation.
- HEM ensemble scores: never fabricate calibrated probabilities or novelty.
- Incompatible comparison protocols: show unavailable/incomparable rather than rank them.

## Task 1 — audit, preservation and shared contracts

Create framework/core/{contracts,config,registry,worker}.py and isolated registry tests.
ModelAdapter methods: health_check()->dict, load()->None,
predict(rows:list[list[float|None]])->list[dict], predict_proba(rows)->list[list[float]],
metadata()->dict. Registry.call(model_id:str, operation:str,payload:dict)->dict|list.
- [ ] Record original hashes/source snapshot and audit/protocol decisions.
- [ ] Write and execute failing crash/timeout/config/lazy-loading tests.
- [ ] Implement bounded worker lifecycle, scoped errors, config and diagnostics.
- [ ] Run registry tests and record evidence.

## Task 2 — authentic adapters and same-dataset training

Create framework/detection/adapters/{common,xgboost,kan,hem,random_forest}.py,
experiment/train_four_models.py, experiment/package_four_models.py, model manifests
and independent adapter tests. Reuse original training implementations without edits.
- [ ] Test schema validation, genuine saved architectures, disabled probability support and CPU reload.
- [ ] Verify dataset/label/row-ID identities and forbidden features.
- [ ] Execute fresh fixed controls and one independent enhancement per eligible family, retaining negative runs.
- [ ] Record common metrics, resources, hashes and selection; package only verified records.

## Task 3 — resilient app, selection, explanations and comparison

Create lazy XGBoost facade; modify startup/server narrowly; add /models template,
static JS/CSS and APIs. Keep existing live/sample/upload routes. Independent
optional monitoring/language initialization and per-window model failure handling.
- [ ] Write failure-injection integration tests for absent models/plugins and missing metrics.
- [ ] Add four-family selector, independent dashboards, capability badges and honest comparison.
- [ ] Decouple XAI and report failure from prediction and keep monitoring rules operational.
- [ ] Run source/browser checks, auth/recovery and complete regression tests.

## Task 4 — review and documentation

- [ ] Independent review adapters, isolation, datasets, research claims and UI.
- [ ] Fix verified findings with regression tests.
- [ ] Execute default demo, benign/ddos/scan/unknown demos and pytest evaluation -v.
- [ ] Update README/context, interfaces/configs/dependency/run guides, references and measured ablation table.
- [ ] Report real native HEM/unknown/installer limitations; never infer release success from source tests.
