# Verified cleaning of the duplicate DataSense dataset

User request: verify and clean the duplicate, preserving the original. This
protocol is written before producing the new candidate or training a model.

Output: `dataset/experiments/datasense_verified_clean_20261014_v2/`.
Only the existing 480,405 training and 30,000 validation IDs are eligible.
The other 175,266 source rows, including the sealed test, are not profiled or
used to fit rules. Reading the original's bytes for a checksum is not editing it.

The full source copy and original must match their recorded SHA256 before and
after execution. Re-read every selected row directly from the copied CSV and
compare all 94 source cells against the earlier archive's record hashes. Keep
the raw selected cells, labels, source IDs, times and all repairs separately.

Checks cover schema/parse validity; missing/nonfinite values; counts, individual
TCP flags, TTL and summary bounds; label hierarchy; identities and timestamps;
list syntax, count consistency and directional membership; exact duplicates and
contradictory identical model inputs; per-class/device/time/protocol coverage;
split overlap; missing-event windows; signed values; and train/validation shift.
Bounds are semantic rather than fitted outlier cutoffs. High counts and rare
attacks are retained. TCP flags may coexist; their sum is not bounded by packets.

Repair policies:

1. Canonical JSON lists retain every observed valid set member. The all-direction
   list can recover the union of members explicitly present in its directional
   lists. This recovers exported membership, not unseen packet data.
2. The destination-MAC count is an exact copy of the source-MAC count throughout
   the earlier audit arrays. Withhold that suspect reported field. Expose an
   explicitly named observed destination-list count, without asserting complete
   capture cardinality. Keep the original count in the raw archive.
3. Reported counts below observed membership become unknown. Other count/list
   discrepancies are retained as evidence of uncertainty. Unlisted membership
   in an inconsistent list is unknown, even below the prior 1,000-member rule.
4. Nonfinite or semantically impossible values become unknown with a ledger;
   impossible summary ordering masks the affected summary group. Unsupported
   zero-event summaries become unknown. Signed sensor/delta values are retained.
5. Labels never change. Hierarchy/provenance contradictions require review.
   The full-label suffix is a scenario descriptor, possibly naming several
   devices or the whole network. Only the class prefix must match the hierarchy;
   the suffix is not required to equal the row's device name.
   An absent protocol in an attack-labelled window is a coverage finding, not
   enough evidence to relabel it or discard an inconvenient validation record.
6. Preserve the complete cleaned master. Additionally export a conservative
   training subset excluding hard-invalid, signal-free, exact duplicate raw
   records and conflicting identical model-input groups. This subset changes
   the training distribution; it is an optional training experiment, not a
   declaration that every excluded label is wrong. The review queue is reversible.
   All 30,000 validation rows and their labels remain present.
7. Retain the full 655-column prior representation before the explicit repairs;
   recompute affected numeric, list, ratio and presence features. Keep every
   column, avoiding the prior compact view's five-row validation discrepancy.
   No labels, IDs, dates, device identities or label-based review flags become
   predictors. Vocabulary comes from the previously frozen training-only fit.

No model prediction or validation score selects a correction. No training,
test scoring, app replacement, external upload or upstream edit is part of
this cleaning run. It cannot guarantee 90–95% accuracy or prove all annotations
correct. The source PCAPs, aligned sensor logs, capture/attack schedule and exact
exporter are needed to resolve capture coverage and annotation truth. Local
UQ-IoT PCAPs belong to another dataset and cannot fill DataSense gaps.

Verification requires independent CSV reloads, exact selected-row identity and
label preservation, valid canonical lists, finite-or-explicit-missing model
cells, exact strict-subset reconstruction, hashes of all outputs, unchanged
source/input hashes, the default and four sample demos, and the evaluation suite.

The v1 attempt is preserved as superseded: an initially over-strict suffix check
wrongly flagged multi-device scenarios. No labels or full master rows were
changed. The v2 run repeats the complete audit with this check corrected and a
specific multi-device regression fixture; the v1 strict subset must not be used.
