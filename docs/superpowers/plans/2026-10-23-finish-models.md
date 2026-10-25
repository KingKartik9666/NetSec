# Finish the frozen all-model study

User: "finish whatevers left, wrap up everything". Execute through verified
completion; do not start another search or promote anything into the application.

Isolation: existing workspace has no Git repository. Additive source modules,
focused tests and a new artifacts/training/all_models_wrapup_20261023_v1 directory.
Protect original protocols, successful/failed attempts, checkpoints, datasets,
upstream code, installed app, and held-out test. No source in sealed studies is
edited. Preserve labels/IDs, frozen selection and role partitioning.

1. Diagnose TabM fold-0 parity failure and retain original failed evidence.
2. Reuse compatible OOF artifacts, recover the failed saved TabM checkpoint only
   after strict numerical and independent CPU replay gates, train missing KAN folds.
3. Assess/calibrate selected models and all fourteen completed seed repeats.
4. Complete three OOF-weighted hybrids, calibration, CPU bundles and inference
   benchmarks. Run all evaluation tests and all five demo commands.
5. Independent review, fix important findings, verify all artifacts and provide a
   final table for ten models, their tradeoffs, seeds and calibration limitations.

Diagnosis: forward hooks localize a 0.1015625 CPU/CUDA encoding difference to
rtdl's piecewise-linear torch.addcmul. Large reciprocal widths and cancelling
biases make fused CUDA arithmetic differ from CPU's operation order. Changing
CUDA encoding inference to separate multiply/add passes the ORIGINAL tolerance:
max probability error drops from 0.0001945496 to 0.0000002384; no sample labels
change. CPU inference, checkpoint weights, bins, selection and tolerance remain
unchanged. Upstream files remain unchanged; adaptation is versioned and explicit.

Reuse ruling: runtime_seconds may differ ONLY if every numerical/training
parameter, arm, seed and fold is identical, recorded stop_reason is early_stopping,
and measured training_seconds is below both caps. The source configuration and
nonbinding-cap difference remain explicit in adoption evidence. No timed-out fit
or full-training prediction is adopted as a completed OOF fit.

Execution ledger (23 October):
- Focused tests passed after the initial missing-module failures established the
  recovery/adoption/export contracts. Full evaluation: 607 passed, 17 warnings;
  all five demo commands exited successfully. Logs: artifacts/training/
  wrapup_20261023_evaluation.log and wrapup_20261023_demo_*.log.
- Independent reviewer day_study_review found no blocking issue in reuse,
  numerical recovery, artifact completion gates or CPU exports.
- New isolated study launched under the persistent supervisor with an 18-hour
  guard. Five compatible existing folds adopted. Saved TabM fold 0 recovered:
  original 256-row discrepancy 0.0001945496; adapted 4096-row maximum difference
  0.0000003874, no changed decisions, original tolerance retained.
- Both new KAN folds completed with independent CPU reload verification (442
  and 420 seconds). All eight required OOF records are present. CPU assessment
  is now running; hybrid construction, repeat assessment and exports remain.
  Existing checkpoints and historical studies remain preserved.

The pipeline completed at 20:43 IST on 23 October: ten models, eight OOF
branches, fourteen repeat assessments, seven standalone and three hybrid CPU
bundles, and twelve completed TPE combiner trials for each hybrid. Fresh hash
verification on 24 October passed for sources, inputs, roles and all outputs.
No training worker/controller remains. Final independent artifact review found
no completion blocker; its calibration, alert-policy, unequal-budget and TabPack
initialization-seed caveats are recorded in CLOSEOUT.md.

Supplementary hybrid CPU benchmarks passed two independent calibrated replay
checks each. Final-closeout project verification passed: 607 tests, 17 warnings,
all five demos, both documented inference examples. All plan steps are complete;
the final-closeout manifest records the checked artifacts. No work is queued.
