# GPU training verification — 29 September 2026

The bounded candidate study completed 63 distinct real-data training trials on
the local NVIDIA RTX 4050. Every completed trial recorded CUDA execution and
verified predictions after reloading its selected checkpoint. There were no
failed trials. Fixtures used by software tests are excluded from this count.

The highest measured single-run validation accuracy is **79.5133%** on the
unchanged 84-class task. The 99.97% target has **not** been achieved. No test
scores or final deployment selection are claimed. See
[measured results](../artifacts/training/RESULTS.md) and
[methodology and commands](GPU_TRAINING.md).

## Executed checks

| Check | Observed result |
|---|---|
| `python -m pytest evaluation -v` | 60 passed in 23.98 seconds; includes CUDA training/reload, HEM vote equivalence, split integrity, resume guards and offline demo checks |
| `python run_demo.py` | Exit 0; synthetic DDoS demonstration |
| All four `--sample` options | Exit 0; BENIGN, DDoS, PORT_SCAN and UNKNOWN as appropriate for the synthetic samples |
| `python -m experiment.predict_trained --index 0` | Saved XGBoost predicts `ddos_rst-fin-flood-port-80`, matching the real validation label |
| Same command with each other selected model record | Forest, KAN and HEM adaptation all reload successfully and predict the same correct label for that observation |
| Resume all four study commands | Exit 0; all trials reused, no new training runs |
| `git diff --stat` in HEM and efficient-kan | No tracked upstream changes |

Commands used the project's `.venv/Scripts/python.exe`. Captured test, demo and
prediction outputs are in `artifacts/training/verification/`. A single correct
prediction verifies the executable path; it is not an accuracy estimate.

## Review and correction

An independent source review traced data preparation, training, validation
selection and checkpoint loading. It also checked temporal interval separation
on the expanded data. It identified a resume-provenance weakness: completed
trial lookup originally omitted the exact prepared-data identity.

Resumption now checks parameters, seed, prepared path, manifest SHA256 and
checkpoint presence. New workers reject missing or changed manifest digests.
The regression test rejects changed hyperparameters, a different prepared
directory, and a changed manifest at the same path. Prediction also verifies
the saved prepared-data identity before loading observations.

Legacy records received their manifest digest in an explicitly marked post-run
audit. Original files, before/after hashes and the audit record are preserved in
`artifacts/training/identity_audit_20260929/`. This is not retroactive proof that
those digests were recorded before training. The audit changed no metrics,
hyperparameters, model weights or prepared arrays.

## Remaining research limits

- These are within-capture temporal validation results; repeated seeds measure
  training randomness, not generalization to independent captures or devices.
- One declared class has no validation support after exclusions. Macro F1
  includes all 84 declared classes; per-class reports disclose support.
- The expanded XGBoost study uses 480,405 training rows. Other selected candidates
  use 120,000 rows, with training budgets shown in the results table.
- HEM uses original neural classes in a documented DataSense closed-set
  adaptation. Original HEM unknown-attack reproduction remains outstanding.
- The default demo uses synthetic data and an offline TemplateProvider. These
  research inference commands execute neither XAI nor a language model.
- Model selection and the requested accuracy target remain open. Raw data,
  earlier code and experiment evidence were retained; cleanup was not performed.

The implementation and review sequence for this bounded study is complete.
The local experiment requires no publishing or deployment step. Further model
development should start from the recorded error patterns and improve observable
features and evaluation evidence, while retaining the locked test partition.
