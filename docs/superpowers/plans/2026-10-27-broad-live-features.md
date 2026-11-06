# Broad packet feature implementation

Scope: collect broad, bounded Rust header measurements and select a generous
dataset-compatible numeric profile without an arbitrary top-k feature limit.
Existing models, completed studies, held-out tests, external repositories and UI
are protected. No live attack label is inferred from desktop traffic.

Isolation: workspace has no Git repository. Current live Rust/training sources
are hash-snapshotted under artifacts/training/live_features_20261027_v1; no active
training process exists. Changes are additive Python live_features modules and
CLI/tests/docs, with scoped Rust packet/native/window extensions and replay tests.
External repositories and old inference model schemas are untouched.

Contract: 64 raw network numeric columns, six withheld (undefined fragmentation
and interval formulas, three application-dissection protocol counts, defective
destination-MAC count). Four payload summary candidates remain extended-only
because the publisher's protocol payload definition is uncertain. Default broad
core: 54 raw fields plus 14 deterministic derivatives; extended: 58 plus15.
Native capture may report additional live-only measurements, distinctly named.
Publisher-equivalence remains unverified; per-feature availability is explicit.

Selection: use only original training rows and the earlier purged role mapping.
Fit-only constant, entirely missing and exact duplicate checks; preserve useful
feature families and do not prune by an arbitrary correlation threshold or top-k.
Rank groups by permutation importance and test group removals using three purged
chronological folds entirely inside the fitting role. Retain broad features unless
measured ablation supports removal. Freeze ordered profile and drop reasons.
The final original-validation partition and held-out test are not selection data.

Native implementation uses the same checked packet parser, population Welford
statistics, actual packet timestamps, IPv4-specific dataset IP summaries, bounded
cardinality sets, and explicit nulls for absent/truncated/overflow/drop-affected
measurements. Signed timing jitter is preserved. No payload bytes are retained.
Legacy seven-count inference stays separate until new models are validated.

- [x] Read context, isolate and record snapshot hashes; inspect source definitions.
- [x] Prove new selector/parser behavior fails before implementation.
- [x] Implement additive native collection and offline replay.
- [x] Implement broad schema, train-only data projection and conservative selection.
- [x] Run actual feature screening, archive profile and measured comparison.
- [x] Verify Rust/Python oracle agreement, original CSV examples and real PCAP replay.
- [x] Run five demos, Python evaluation, Rust checks and change review.
- [x] Complete 21 matched ablation fits and freeze numeric60 plus port192 profile.
- [x] Verify all-family adaptation smoke, calibration, CPU replay and STOP/resume.
- [x] Prepare detached Optuna queue; see docs/BROAD_LIVE_TRAINING.md and study state.

Closeout, 28 October: broad numeric selection and matched port/group comparisons
are complete. The new 252-feature profile is a research candidate. Independent
review found and the implementation fixed capture-drop/window-boundary ownership;
review then found no blocking issue. A later heartbeat-count correction has its
own failing/passing regression and separate main study revision. Existing full
models and installed application remain protected. Publisher parity and labelled
live accuracy are outstanding research, not completed by these software checks.
