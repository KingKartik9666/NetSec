# Partial input upgrade execution record

Accepted plan: upgrade existing XGBoost to a versioned 170-feature,84-class
model accepting full/partial CSV and seven measured live counts. Live validation
accuracy takes priority, macro F1 breaks ties. Original artifacts and held-out test
remain preserved. Training uses CUDA; app inference stays CPU/offline.

- Canonical mapping/parser/runtime: root.
- Twelve-config study plus four repeat fits: partial_gpu_study.
- Feature coverage, multiclass live UI and CSV export: partial_ui.
- Missing-input SHAP/LIME and grounded local English: partial_xai.
- End-to-end integration, release verification and installer: root.

Ruling: only the existing seven count definitions currently meet the documented
mapping contract. Other measurements remain missing until semantics are verified.
No duplicate feature names, guessed protocols or fabricated sensor logs.

Ruling: original schema-v1 CSV behavior remains supported; the versioned v2
bundle explicitly opts into partial input handling and nullable protocol flags.
Unknown input profiles can produce tentative top-three categories, while their
validation status is clearly unvalidated. No automatic incident confirmation.

Ruling: the workspace has no project Git repository; work stays in its existing
directories. Checkpoints are versioned without overwriting previous studies.

Progress: shared schema, lifecycle, multiclass API, UI, CSV round trip and
missing-aware XAI/LLM are implemented. The earlier integration suite passed 185
tests; the default demo and all four samples passed. Provisional-checkpoint
execution evidence remains preserved and explicitly identifies the unselected
candidate. The original twelve-configuration plus four-repeat CUDA study has now
completed all 16 fits. Its independently executed `AUDIT.json` passed source,
data, checkpoint, metrics, repeat-seed and train-only XAI reference checks. The
selected model's CPU reload probabilities exactly match saved probabilities on
all three validation profiles. No held-out test arrays were opened or scored.

The promoted base model is `partial-v1-fresh_d6_smooth_s42`: seven-count
validation accuracy 70.1464%, top-three accuracy 82.1133%, macro F1 0.387033,
with 1,571 source-validation rows covering 60 of 84 labels. Full-CSV accuracy is
79.3233% and macro F1 0.661233, a disclosed regression from the original
80.0533%/0.696463 full-input checkpoint. See `docs/PARTIAL_TRAINING.md` for all
three profiles, false-positive rates, repeated-seed results and limitations.

Actual selected-base replay and Windows Npcap execution passed on 1 October:
`artifacts/partial_verification/20261001T032332469316Z_117ea9eb/verification.json`.
The replay used 108 recorded headers; the live window used 1,482 host-relevant
Wi-Fi headers. Both independently verified counts, identical 170-column CSV
round trips and predictions, seven measured values/163 nulls, checkpoint reload
and SHAP additivity. Stop cleared both verdicts. The live model's tentative
attack-category hypotheses are unlabelled execution evidence, not confirmed
incidents or classification accuracy. Final installer/model/UI verification is
still pending and must use the final active checkpoint.

The final local English explanation path was also executed through production
HTTP handlers with actual Qwen2.5-0.5B-Instruct Q4_K_M on CPU. Benign sample 162,
sample 0, seven measured inputs reused from the recorded Windows window, and a
two-measurement unvalidated profile all returned HTTP 200. The first three passed
their first generation; the two-measurement profile passed its second permitted
attempt. Proof with actual prompts, model outputs and code hashes:
`artifacts/desktop/verification/release-0.3.0/llm-production-verification.json`.
The small model conservatively joins evidence-derived English fact sentences;
open-ended paraphrasing had dropped alternate category names and limitations.
Retrieval notes remain exactly supplied background, rather than traffic evidence.
This is actual llama.cpp inference, distinct from deterministic fact preparation
and template reports. It is not open-ended security reasoning. CPU threads remain
two, context 2,048 tokens, output at most 200 tokens and generation at most two
attempts. Failed checks still return an error with no substitute text. Diagnostic
codes are excluded from retry prompts, and instruction/diagnostic echo is rejected.
Twenty-nine focused LLM/XAI regression tests passed after the change.

The read-only release review also found that impossible seven-count CSV values
could be labelled a validated input profile. The final coverage guard preserves
the seven-column mapping but marks negative, fractional, empty, excessive or
flag-count-above-total inputs unvalidated. Genuine measured zero flag counts
remain valid. This changes compatibility reporting, not frozen training data.

User extension on 1 October: compare feasible internet-sourced imbalance methods
after the frozen base study, without changing its configuration list or results.
The separate extension compares a matched continuation control, capped random
oversampling, constrained seven-count SMOTE, conservative leaf updates and CUDA
gradient-based sampling. It has at most five initial fits plus four conditional
seed repeats. Synthetic/duplicated training rows are identified separately;
validation/test are never resampled. No method is promoted solely because it is
more complex. Prior GAN results remain preserved and did not beat the original
full-input accuracy baseline.
