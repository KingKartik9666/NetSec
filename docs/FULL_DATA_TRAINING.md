# Full-training six-model study

This implements the requested suggestions 4, 5 and 6: expand the existing
candidates to all eligible training rows, run a joint XGBoost parameter search,
and add CatBoost and LightGBM GPU training.

## Meaning of complete dataset

The source CSV has 685,671 observations. This study uses all **480,405 eligible
training rows**, after the existing chronological split and overlap/duplicate
exclusions. The same 30,000 validation observations remain held out for model
selection. Test scoring stays locked. Training on validation/test rows would
invalidate the comparison, so those rows are not added to training.

Trees use 170 numeric and protocol-presence features; KAN and HEM use their
previously selected 71 numeric features. All six candidates use the same row IDs
and 84 fine-grained labels. The task and existing prepared data are unchanged.
No new packet features, hierarchy, label merges or categorical identifiers are
introduced by this study. See [the existing methodology](GPU_TRAINING.md) for
the within-capture evaluation limits and missing validation support.

## Execution

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -u -m experiment.train_full_dataset
.\.venv\Scripts\python.exe -m experiment.summarize_full_dataset
```

The same training command resumes completed trials without retraining them.
The immutable study definition contains the manifest digests, configurations
and seeds. Each new trial has a separate checkpoint directory. Do not run two
controllers against the same study directory concurrently.

The optional environment uses CatBoost 1.2.10 and LightGBM 4.6.0. The tested
Windows LightGBM 4.7.0 wheel reported that its GPU tree learner was disabled;
the study pins the OpenCL-capable 4.6.0 wheel. Keep these dependencies separate
from the default CPU demo requirements.

For one family, for example:

```powershell
.\.venv\Scripts\python.exe -u -m experiment.train_full_dataset --candidates catboost
```

`--search-only` runs seed-42 settings without the repeat seeds. The default
study has 13 search settings and 12 selected-configuration repetitions: 25
completed runs. Settings are selected by unweighted validation macro F1, while
accuracy and per-class results are reported alongside it.

## Backends and settings

| Candidate | Full-training configuration |
|---|---|
| CUDA forest | Previous selected XGBoost standalone forest, retrained on every training row |
| KAN | Previous selected original efficient-kan B-spline architecture; training-only preprocessing |
| HEM adaptation | Original neural classes plus source-equivalent closed-set voting; both networks train on all rows and all 480,405 rows form the exact CUDA KNN reference set |
| XGBoost | Six joint settings: depths 6–12, limits of 500–1,500 boosting rounds, learning rates 0.03–0.08, split penalty, minimum child weight, regularization and row/column sampling |
| CatBoost | CUDA multiclass models with depth 6/8 and limits of 750/1,200 rounds |
| LightGBM | NVIDIA OpenCL GPU histogram training, 31/63 leaves and limits of 500/1,000 rounds |

Boosters use 40-round early stopping on unweighted validation macro F1. Their
round limits are upper bounds, not promises to train every round. The selected
configuration per family is repeated with seeds 123 and 2025 after seed 42.
GPU reductions in CatBoost/LightGBM can vary slightly despite fixed seeds.
CatBoost's native console labels its evaluation score `test`/`bestTest`; in this
workflow that is the validation `eval_set`, not the locked test partition.

LightGBM's Windows GPU backend accelerates histograms through OpenCL, with other
training stages on CPU. CUDA LightGBM is not available natively on Windows.
The implementation locates an NVIDIA OpenCL GPU and requires confirmation from
the actual LightGBM runtime log. CatBoost requires its fitted model to report
GPU training. CPU training fallback is rejected. Preprocessing and metric
calculation may use CPU; saved CatBoost/LightGBM prediction uses CPU.
[Official LightGBM installation documentation](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html#build-cuda-version),
[official Python wheel support](https://github.com/lightgbm-org/LightGBM/blob/main/python-package/README.rst).

## Saved evidence and inference

The study directory is `artifacts/training/datasense_all_models_full_v1/`.
It contains `study.json`, `experiments.json`, per-run logs/models/metrics,
`RESULTS.md`/`.json` after summarization, and selected-checkpoint pointers.
Each completed trial reloads its saved model and checks predictions against the
original selected model on a fixed validation slice.

After summarization, run a saved model on a real validation observation:

```powershell
.\.venv\Scripts\python.exe -m experiment.predict_trained --record artifacts/training/datasense_all_models_full_v1/selected/xgboost.json --index 0
```

Replace `xgboost.json` with `random_forest.json`, `kan.json`, `hem.json`,
`catboost.json` or `lightgbm.json`. This is actual checkpoint inference with the
dataset label printed for comparison. It executes no XAI or LLM.

This HEM experiment is a DataSense closed-set adaptation, not reproduction of
the original unknown-attack study. The default CPU synthetic `run_demo.py`
remains a separate DemoDetector/TemplateProvider software demonstration.

The reports measure attainment of 96% and 99.97% validation accuracy explicitly.
Neither is guaranteed, and neither would by itself establish independent-capture
or final test performance. Earlier checkpoints and original data are retained.
