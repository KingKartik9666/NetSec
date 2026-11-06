# CSV dataset redesign and unattended training

30 October 2026. This study follows the user's instruction to improve CSV
classification through fresh data and feature work. Live screening is deferred.
Implementation and verification are in progress; this document does not report
new trained scores.

## Evidence and acceptance

The original audit reports are represented by the independently checked evidence
in `artifacts/audit_verification_20261029/REPORT.md`. Their original external
directory is currently unavailable. Recommendations are hypotheses, not commands
or established causes of the accuracy plateau.

The fresh, original-training-only scan is in
`artifacts/research/csv_redesign_20261030_v1/diagnostics/train_structure.json`.
It verifies 480,405 selected training rows, 84 classes, 612 scenarios, 38 devices
and ten window durations. Every row overlaps or touches another window within
its scenario. Of 96,393 rows in conflicting old-representation groups, 96,027
have no observed signal. The remaining 366 require a new-representation check;
their labels are not automatically declared wrong.

Most recon classes have one large whole-network scenario and several much
smaller targeted scenarios. This supports testing scenario imbalance and
generalization, not assuming an information ceiling or a particular gain.

Observable acceptance criteria:

- The original CSV and previous models remain unchanged. The new physical copy
  has the same SHA-256 before processing, with provenance and row-ID joins.
- All learned vocabulary, redundancy decisions and conflict filtering use
  fitting rows only. Metadata and label changes cannot change predictors.
- Temporal roles do not share overlapping/touching windows within a scenario.
  A separate scenario challenge is reserved until selections are frozen.
- Cleaning, feature engineering and scenario weighting have matched controls.
  Each comparison reports accuracy, declared and supported macro F1, per-class
  recall, false alarms, support and excluded-row coverage.
- A failed hypothesis is recorded and can be rejected. New features are not
  accepted merely because they are new, numerous or plausible.
- Training resumes from persisted requests and completed artifacts without
  chat calls. Changed source/data/configuration fails verification. Interrupted
  fits are distinguishable from successful fits and completed model selection.

## What changes, and what is reused

| Finding | Intervention to test | Control and limitation |
| --- | --- | --- |
| Old vocabulary was derived from broader training data before internal folds | Fit a fresh bounded vocabulary inside fitting roles | No inherited 655-column supervised selection; unseen tokens retain explicit semantics |
| Empty windows carry many different scenario labels | Quarantine no-signal fitting rows, preserving the full archive and ledger | Unfiltered fitting control; do not relabel ambiguous observations |
| Old feature vectors still have observed conflicting groups | Recompute conflicts after the new representation, using fitting labels only | No validation-based conflict removal; retain class-support checks |
| Unique port lists lose structure when reduced to memberships | Add observed range, spacing, port bands and directional set relationships | Base and enriched features on identical rows; unique-list statistics are not packet-frequency entropy |
| Aggregated magnitudes vary with window duration and traffic scale | Add normalized directional and summary-shape measurements | Preserve real zeros, signed timing, unknown values and valid extremes |
| Reported counts and capped lists disagree | Retain observed lower bounds and completeness explicitly | Do not infer unobserved ports or repair counts to imagined packet truth |
| One scenario dominates several recon classes | Test capped scenario-balanced resampling within each class | Same class totals; duplicates are reweighting, not new independent observations |
| Overlapping durations and scenarios make row splits optimistic | Purged temporal roles plus a reserved scenario challenge | Report unsupported classes and retained coverage; no independent-row uncertainty claim |
| Old fits had unequal budgets and one selected XGBoost was runtime-truncated | Round-robin model search, explicit budgets and stop reasons | Runtime termination is not convergence; a partial queue is not completed training |

Existing conservative `clean_row` repairs are reused and are not presented as
new discoveries. Existing genuine model trainers, calibration and CPU reload
utilities are reused. HEM remains a closed-set HEM-derived adaptation. No demo
detector, fabricated labels or LLM-generated observations enter this study.

The official publisher describes a synchronized sensor/network dataset and a
multi-objective feature-selection study:
[UNB DataSense dataset page](https://www.unb.ca/cic/datasets/iiot-dataset-2025.html).
That page establishes the publisher's description, not byte identity of this
local CSV or equivalence to its original feature extractor. The paper endpoint
returned HTTP 429 during this refresh; no new extractor semantics were inferred.

## Data boundaries

Original: `dataset/combined_dataset.csv`.

Physical copy:
`dataset/experiments/csv_audit_redesign_20261030_v1/source/combined_dataset.csv`.
The copy receipt records 3,302,457,978 bytes and SHA-256
`4db33c7b73a6e715d1b4484f4d4c77747a5e9355abf70c394c043cd27579a9f9`.
Copying opaque bytes is distinct from selecting records for model fitting.
Processing uses an explicit allowlist of original training row IDs. The
automatic study does not evaluate original validation or open held-out test
arrays. Any later original-validation comparison needs frozen selections and
must acknowledge that this source validation was historically reused.

Rows are archived with their source IDs, original labels and audit decisions.
Device names, addresses, absolute timestamps and scenario identifiers are
available for provenance, grouping and diagnostic tables only. They are not
model features. Duration is derived from window bounds; the absolute bounds
themselves do not enter predictors.

## Architecture and trust boundary

`framework/training/csv_redesign/features.py` owns deterministic row encoding
and the serializable learned schema. `data.py` owns original-train selection,
group/time splits, quarantine and data receipts. `worker.py` owns one isolated
fit/evaluation transaction, while `runner.py` owns scheduling and recovery.
`experiment/train_csv_redesign.py` is the command entry point.

CSV text is untrusted data. Tokens are parsed as data, never executed as Python
or shell commands. Explicit numeric/list fields define the predictor schema.
Study paths are confined to the workspace and model loading uses only local
artifacts named by verified requests. Persisted hashes detect accidental drift;
they are not signatures against an attacker who can rewrite the whole workspace.
No credentials, online API, capture driver or automatic application promotion
belongs in this training path.

The supervised process uses owned child jobs, exclusive locks, heartbeat/state
files, STOP handling and bounded retries. A chat token limit does not stop it;
sleep, shutdown, process failure, exhausted disk space or its configured budget
can. Completed checkpoints and trial evidence remain available after stopping.

## Verification status

The physical source copy has been hash-verified. The former broad-live queue and
narrow plateau pilot were stopped, preserving their artifacts; a fresh process
inspection found no matching training processes.

New feature/data/controller tests, project regression checks, independent review,
an end-to-end smoke run and real-process startup verification are pending.
No new accuracy improvement, complete retraining, native export or installation
update is claimed here.
