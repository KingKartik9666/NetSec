"""Build reproducible audit/model/calibration tables from completed artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from experiment.audit_datasense_raw import DEFAULT,PARENT,ROOT
from experiment.datasense_quality import digest,write_json
from experiment.train_datasense_quality import probability_metrics
from framework.training.tracking import classification_metrics


def read(path):return json.loads(path.read_text())


def run(directory=DEFAULT):
    out=directory/'report';out.mkdir(exist_ok=True)
    manifest=read(directory/'04_port_refinement/manifest.json')
    truth=np.load(directory/'04_port_refinement/y_validation.npy')
    records=[];verification=[]
    metrics_columns=['accuracy','macro_precision','macro_recall','macro_f1','weighted_f1','false_positive_rate','attack_missed_as_benign_rate']
    for path in sorted((directory/'runs').glob('*/result.json')):
        result=read(path)
        if result['status']!='completed':continue
        prediction=np.load(path.parent/'validation_predictions.npy')
        probability=path.parent/'validation_probabilities.npy'
        measured=(probability_metrics(truth,np.load(probability),manifest['classes'],manifest['benign_index']) if probability.exists() else
                  classification_metrics(truth,prediction,manifest['classes'],manifest['benign_index']))
        for key in metrics_columns:
            if not np.isclose(result['metrics'][key],measured[key],atol=1e-12,rtol=0):
                raise ValueError(f'Metric recomputation differs {path.name}: {key}')
        checkpoint=path.parent/('model.ubj' if result['family']=='xgboost' else 'model.joblib' if result['family']=='random_forest' else 'model.pt')
        if digest(checkpoint)!=result['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        record={'run':path.parent.name,'family':result['family'],'arm':result['arm'],
                'training_observations':result.get('represented_training_observations',result.get('training_rows')),
                'features':result['feature_count'],'device':result['device'],'elapsed_seconds':result['elapsed_seconds'],
                **{k:measured[k] for k in metrics_columns}}
        records.append(record)
        verification.append({'run':record['run'],'rows':len(truth),'metrics_recomputed':True,
                             'checkpoint_hash_verified':True,'reported_reload_rows':result['reload_rows']})
    with (out/'model_comparison.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    calibration=read(directory/'calibration/summary.json');protocol=read(directory/'calibration/protocol.json')
    correctness=read(directory/'calibration/correctness_confidence_summary.json')
    correctness['hem']=calibration['hem']
    before={'xgboost':read(PARENT/'runs/original_control/result.json')['metrics'],
            'random_forest':read(directory/'runs/random_forest_original_control_s42/result.json')['metrics']}
    for family in ('kan','hem'):
        before[family]=read(ROOT/f'artifacts/training/datasense_all_models_full_v1/selected/{family}.json')['seed_42_record']['metrics']
    comparison=[];catalog=[]
    for family,entry in calibration.items():
        path=directory/entry['model_run'];result=read(path/'result.json')
        current=result['metrics'];old=before[family]
        comparison.append({'family':family,'reference_accuracy':old['accuracy'],'new_accuracy':current['accuracy'],
                           'accuracy_delta_percentage_points':100*(current['accuracy']-old['accuracy']),
                           'reference_macro_f1':old['macro_f1'],'new_macro_f1':current['macro_f1'],
                           'new_macro_recall':current['macro_recall'],'reference_false_positive_rate':old['false_positive_rate'],
                           'new_false_positive_rate':current['false_positive_rate'],
                           'attribution':'Combined preprocessing/recipe comparison; historical neural references, fresh fixed RF control.'})
        candidate=result.get('candidate','04_port_refinement')
        features=read(directory/candidate/'manifest.json')['model_feature_names']
        catalog.append({'family':family,'model_run':entry['model_run'],'candidate':candidate,'feature_names':features,
                        'full_master_column_indexes':[manifest['feature_names'].index(name) for name in features],
                        'checkpoint_sha256':result['checkpoint_sha256'],'calibration_directory':'calibration/'+family,
                        'default_inference_device':'cpu','application_promoted':False,'test_evaluated':False,
                        'hem_scope':'HEM-derived closed-set only' if family=='hem' else None})
    write_json(out/'reference_comparison.json',comparison);write_json(out/'model_catalog.json',catalog)
    write_json(out/'model_verification.json',{'runs':verification,'test_loaded':False})
    pipeline_hashes={}
    for item in catalog:
        model_dir=directory/item['model_run']
        paths=list(model_dir.glob('model.*'))+list(model_dir.glob('preprocessing.joblib'))+list(model_dir.glob('knn_references.npz'))
        paths+=list((directory/'calibration'/item['family']).glob('*.json'))
        paths+=list((directory/'calibration'/item['family']).glob('*.joblib'))
        for path in paths:pipeline_hashes[path.relative_to(directory).as_posix()]=digest(path)
    for path in (ROOT/'external/HEM/model_evaluation/binary_classifier.py',ROOT/'external/HEM/model_evaluation/multiclass_classifier.py',ROOT/'external/efficient-kan/src/efficient_kan/kan.py'):
        pipeline_hashes[path.relative_to(ROOT).as_posix()]=digest(path)
    write_json(out/'pipeline_hashes.json',pipeline_hashes)
    table='| Model | Reference accuracy | New accuracy | Change | New macro F1 | New benign false-alarm rate |\n|---|---:|---:|---:|---:|---:|\n'
    for r in comparison:
        table+=f'| {r["family"]} | {r["reference_accuracy"]:.2%} | {r["new_accuracy"]:.2%} | {r["accuracy_delta_percentage_points"]:+.2f} pp | {r["new_macro_f1"]:.2%} | {r["new_false_positive_rate"]:.3%} |\n'
    ctable='| Model | Assessment accuracy (unchanged) | ECE before | ECE after | Score type |\n|---|---:|---:|---:|---|\n'
    for family,entry in calibration.items():
        a,b=entry['assessment_before'],entry['assessment_after']
        ctable+=f'| {family} | {b["accuracy"]:.2%} | {a["ece_10_equal_width_bins"]:.4f} | {b["ece_10_equal_width_bins"]:.4f} | '+('Hard-vote correctness confidence' if family=='hem' else 'Multiclass probabilities')+' |\n'
    confidence_table='| Model | Decision-confidence ECE before | ECE after | Correctness Brier before | Brier after |\n|---|---:|---:|---:|---:|\n'
    for family,entry in correctness.items():
        a,b=entry['assessment_before'],entry['assessment_after']
        confidence_table+=f'| {family} | {a["ece_10_equal_width_bins"]:.4f} | {b["ece_10_equal_width_bins"]:.4f} | {a["correctness_brier_score"]:.4f} | {b["correctness_brier_score"]:.4f} |\n'
    text=f'''# Cleaned DataSense copy: completed audit and model assessment

All **480,405 training** and **30,000 validation** records retain their original labels and source IDs. The final `04_port_refinement/` master has **655 full features** and **559 compact model inputs**. Full evidence and all original 94 fields remain available in `01_raw_audit/`; original source SHA256 is `{manifest['source_sha256']}`. The other source records and locked test were not audited or fitted.

## Measured models on the same 30,000 validation rows

{table}
RF uses the unchanged 100-tree conventional sklearn configuration and a fresh numeric control. XGBoost retains separate 79-round comparisons and an explicitly labeled deeper/longer research trial; grouping preserves counts and additive loss but changes subsampling units. Neural references are historical seed-42 fits on the same observations; the new runs change preprocessing and duration, and HEM changes PCA capacity. These are combined representation/recipe gains, not a causal estimate of cleaning alone. All attempted results, including regressions, are in `report/model_comparison.csv`. The selected XGBoost recipe is chosen by original full-validation accuracy then macro F1, so selection remains exploratory.

The prior enriched XGBoost reference reached **80.62% accuracy / 76.81% macro F1**. The new selected recipe reaches **80.53% / 77.70%**: macro F1 rises, while accuracy is 0.09 percentage points below that earlier strongest accuracy reference. HEM's benign false-alarm rate rises from 0.44% to 0.60%. RF's attack-missed-as-benign rate rises from 3.35% to 4.16%. These tradeoffs are retained; none of these results is 95% accuracy.

## Calibration assessment

{ctable}
Calibrators fit {protocol['calibration_rows']:,} early within-class validation observations and assess the same {protocol['assessment_rows']:,} later observations before/after. {protocol['boundary_or_copy_exclusions']:,} boundary or repeated-input records are excluded from this comparison only. Fit/assessment coverage is 65/83 classes. This validation set influenced earlier experiments: the assessment is **not independent held-out or external-capture evidence**. Temperature scaling leaves class decisions unchanged; it is judged by log loss, Brier score and reliability. HEM confidence estimates whether its unchanged hard vote is correct; it is not a multiclass ensemble probability. Regressions in any calibration metric remain reported.

Temperature scaling worsens decision reliability for RF and KAN even where another score metric may improve. A second fixed, regularized Platt recipe fits the original top-score logit to observed correctness using only the same calibration rows. Its confidence means the probability that the unchanged class decision is correct, not a replacement multiclass probability distribution. It is fit once without later-assessment tuning; all temperature artifacts remain available.

{confidence_table}
Decision-confidence ECE improves for all four on the later assessment. Correctness Brier improves for three; KAN's rises slightly from 0.112412 to 0.112478. Inspect `calibration/*/correctness_confidence_result.json` (HEM: `result.json`) and `report/calibration_reliability.png`. Small differences should not be treated as statistically independent evidence because windows share captures. Class decision accuracy remains unchanged.

## Audit findings and decisions

- No selected numeric cells were missing/nonfinite; no selected list was malformed. No exact duplicate physical raw observation was found.
- 96,027 training windows have zero measurements and empty lists. They remain in the master and all-records fits; the observed-signal ablation is labeled separately.
- 10,217 training log minima are negative; 1,130 training timing minima are negative. Values, including legitimate zeros and rare attack extremes, remain intact.
- Source and destination MAC reported counts are equal in every audited row, although destination lists disagree with reported counts in 107,129 training and 7,549 validation rows. Original reported counts and independently observed list lengths are both retained.
- Port count/list discrepancies affect 59,285/34,652/52,853 training rows for all/destination/source lists and 4,126/2,447/3,712 validation rows. The 1,000-entry suspected-censoring rule marks unknown absence; it does not replace reported counts or assert verified exporter internals.
- Unsupported zero summary cells are marked unavailable only in the derived model representation; original cells and exact change bitmaps remain in the raw audit.
- 77 exact training duplicate columns and 19 training constants are omitted from compact inputs. One duplicate relationship differs on **five validation rows**; the 655-column master preserves those values. Compact selection is not universally lossless.
- 383,630 exact input/label groups represent all 480,405 training observations, with reversible source mappings and multiplicities. Different labels stay separate. RF/neural fitting uses expanded rows.

## Files to use

`04_port_refinement/cleaned_train.csv.gz` and `cleaned_validation.csv.gz` are complete readable exports. Blank derived numeric cells mean unavailable, not zero. Use `X_train.npy`, `X_validation.npy`, `y_*.npy` and ordered `model_feature_names` for compact fitting. Use `X_full_*.npy` and `feature_names` for maximum retained information. Source row IDs are lineage, never predictors. `label4` is the original target.

For grouped XGBoost inputs, load `all_records_representative_indexes.npy` **with** `all_records_multiplicity.npy`. Never fit representatives with unit weights while claiming full counts. `all_records_inverse.npy` reconstructs every original selected row. Use expanded observations for algorithms whose bootstrap/optimization semantics require them.

`report/model_catalog.json` records the chosen checkpoint, calibration location and full-master column indexes for each family. Trained models accept ordered feature arrays; these artifacts are not silently installed into the active application. See `verification_04_port_refinement/field_dictionary.csv` for all raw zeros and decisions, `feature_dictionary.csv` for all full features, `02_profile/` for exact zero-location bitmaps and the row-level issue/change ledgers for discrepancies.

## Verification and limits

Data verification hashes the untouched 3.3 GB source, frozen IDs/labels/times, all candidate arrays and grouping reconstructions. Model reports independently recompute saved predictions/probabilities and verify checkpoint hashes. CPU checkpoint reload covers all 30,000 rows for RF/XGBoost/KAN and 128 rows for HEM's expensive exact-reference vote. Calibration parameters/scores reload and preserve class decisions. Project demo and evaluation results are in `verification/project_checks.json`.

Two resumed agents completed bounded source and implementation reviews. They did not rerun the primary's full audit. Initial invalid port-token construction and a memory-interrupted RF attempt remain explicitly named historical attempts, excluded from completed results. The original author's extractor, local publisher-byte identity, feature-specific missing sentinels and original label mapping could not be authenticated. The user export's 84 fine labels and 1–10 second overlapping windows are not interchangeable with the paper's reported tasks and protocol. No locked-test, novel-attack, live-capture, 95% accuracy or publication-reproduction claim is made.

Primary research and reproducible commands: `docs/DATA_CLEANING_20261013.md` and `report/source_snapshot/`. This is the completed information-preserving candidate from a bounded study; it does not guarantee maximal possible accuracy on future traffic.
'''
    (directory/'RESULTS.md').write_text(text,encoding='utf-8')
    (directory/'README.md').write_text('# Cleaned DataSense copy\n\nStart with [RESULTS.md](RESULTS.md). Use `04_port_refinement/` for the final complete dataset; the selected model recipes and calibration assessment are recorded under `report/` and `calibration/`.\n',encoding='utf-8')
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    paths=list(ROOT.glob('experiment/*datasense*.py'))+[ROOT/'evaluation/test_datasense_quality.py',ROOT/'docs/DATA_CLEANING_20261013.md']
    source_hashes={}
    for source in paths:
        shutil.copy2(source,snapshot/source.name);source_hashes[str(source.relative_to(ROOT))]=digest(source)
    write_json(out/'source_hashes.json',source_hashes)
    make_plots(out,comparison,correctness)
    print(json.dumps({'completed_runs_verified':len(records),'selected_families':len(catalog),'report':str(directory/'RESULTS.md')}))


def make_plots(out,comparison,calibration):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    x=np.arange(len(comparison));width=.35
    fig,ax=plt.subplots(figsize=(8,4.2))
    ax.bar(x-width/2,[r['reference_accuracy']*100 for r in comparison],width,label='Reference',color='#8a98aa')
    ax.bar(x+width/2,[r['new_accuracy']*100 for r in comparison],width,label='Cleaned data + recipe',color='#137c8b')
    ax.set_xticks(x,[r['family'].replace('random_forest','Random Forest').replace('hem','HEM-derived').replace('kan','KAN').replace('xgboost','XGBoost') for r in comparison])
    ax.set_ylim(0,100);ax.set_ylabel('Validation accuracy (%)');ax.legend(frameon=False)
    ax.set_title('Same 30,000 source-validation rows • exploratory comparisons')
    fig.tight_layout();fig.savefig(out/'validation_comparison.png',dpi=300);fig.savefig(out/'validation_comparison.svg');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(8,7),sharex=True,sharey=True)
    for ax,(family,entry) in zip(axes.flat,calibration.items()):
        ax.plot([0,1],[0,1],'--',color='#999999',lw=1)
        for stage,color,label in [('assessment_before','#8a98aa','Before'),('assessment_after','#137c8b','After')]:
            result=entry[stage];bins=result.get('reliability',result)['bins']
            points=[b for b in bins if b['rows']]
            ax.plot([b['confidence'] for b in points],[b['accuracy'] for b in points],'o-',color=color,label=label)
        ax.set_title(family.replace('hem','HEM correctness'));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.legend(frameon=False)
        ax.set_xlabel('Mean estimated confidence');ax.set_ylabel('Observed correctness')
    fig.suptitle('Later validation calibration assessment • 14,754 rows')
    fig.tight_layout();fig.savefig(out/'calibration_reliability.png',dpi=300);fig.savefig(out/'calibration_reliability.svg');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,default=DEFAULT)
    run(parser.parse_args().directory)
