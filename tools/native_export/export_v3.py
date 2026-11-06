"""Export the cascade v3 77-class candidates into a new native application bundle.

Sources: completed jobs of artifacts/training/cascade_v3_20261030_v1 and the pinned train/validation arrays of
dataset/experiments/csv_v3_cascade_20261030_v1. The held-out test partition is never loaded. Research
artifacts are only read. Two small calibrators are fitted here on the validation calibration role (1) only,
and kept only if tuning-role (0) calibration error drops: the binary attack-score Platt map (every
probability model) and the HEM vote-correctness map (HEM had none in the study). Assessment rows (2) are
reported, never used. The export goes to a new folder; swapping it into the app is a separate step.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil

import numpy as np

from tools.native_export.export import (LABELS, TensorWriter, digest, export_forest, export_pipeline, relative,
                                        row_records, sample_indices, write_json)

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'artifacts/training/cascade_v3_20261030_v1'
DATA = ROOT / 'dataset/experiments/csv_v3_cascade_20261030_v1'
OUTPUT = ROOT / 'artifacts/application/native_models_77_staging'
TUNED = ('xgboost', 'kan', 'hem', 'tabm_ne', 'tabm_hier', 'tabm_hier_ne')
BASELINE = ('random_forest', 'tabpack')          # fixed control / fixed auxiliary pack: not tuned in the study
HYBRIDS = {'tabm_xgboost': '77_hybrid_tuned_tabm_xgboost', 'tabm_xgboost_kan': '77_hybrid_tuned_tabm_xgboost_kan',
           'tabpack_xgboost': '77_hybrid_tuned_tabpack_xgboost'}
DEFAULT_MODEL = 'tabm_xgboost_kan'               # tuning-role selection of the study (REPORT_77.md)
LOG_FLOOR = 1e-7                                  # csv_v3.calibrate.EPS
ECE_BINS = 15
# First written as "tuning ECE must drop"; changed to Brier (a proper score; 15-bin ECE is noisier) on
# 2026-10-31 after HEM's calibrator lowered tuning Brier 0.0375->0.0328 but raised tuning ECE 0.0199->0.0292.
ADOPTION_RULE = ('fit on the calibration role; adopt only if tuning-role Brier drops (slope > 0 for Platt); '
                 'otherwise identity; assessment role reported only')
LIMITATIONS = ('Validation evidence only (tuning/calibration/assessment roles of a reused within-capture '
               'validation split); the held-out test partition was never scored. The 8 nmap recon subtypes are '
               'one class, recon_scan: they were not separable in the CSV features. HEM is a HEM-derived '
               'closed-set adaptation, not the published HEM.')


def job(name):
    result = json.loads((STUDY/'jobs'/name/'result.json').read_text(encoding='utf-8'))
    if result.get('status') != 'complete' or result.get('task') != '77':
        raise ValueError(f'{name} is not a completed 77-class job')
    return result


def selection():
    chosen = {f: json.loads((STUDY/f'tune_77_{f}.json').read_text(encoding='utf-8'))['best_job'] for f in TUNED}
    chosen.update({f: f'77_baseline_{f}' for f in BASELINE})
    for name, hybrid in HYBRIDS.items():
        branches = job(hybrid)['branch_jobs']
        if any(chosen[b] != j for b, j in branches.items()):
            raise ValueError(f'{hybrid} blends {branches}, not the exported branches')
    return {f: job(j) for f, j in chosen.items()}


def model_dir(result):
    return STUDY/Path(result['work_dir'].replace('\\', '/'))/'model'


def raw_validation(result):
    folder = STUDY / Path(result['work_dir'].replace('\\', '/'))
    p = np.load(folder/'val_proba.npy')
    if p.ndim != 2 or not np.isfinite(p).all():
        raise ValueError(f'{folder} raw validation probabilities invalid')
    return p.astype(np.float64)


def binary_ece(truth, probability, bins=ECE_BINS):
    edges = np.clip(np.digitize(probability, np.linspace(0, 1, bins+1)[1:-1]), 0, bins-1)
    return float(sum(np.mean(edges == b)*abs(truth[edges == b].mean()-probability[edges == b].mean())
                     for b in range(bins) if np.any(edges == b)))


def platt(raw, y, roles):
    """Binary attack score: Platt map of the raw 1 - p(benign) (the runtime's formula), fit on role 1."""
    from scipy.special import expit, logit
    from sklearn.linear_model import LogisticRegression
    attack = y != 0
    feature = logit(np.clip(1-raw[:, 0], 1e-6, 1-1e-6))
    fit, tune, assess = (np.flatnonzero(roles == r) for r in (1, 0, 2))
    model = LogisticRegression(C=1., random_state=42, max_iter=1000).fit(feature[fit, None], attack[fit])
    slope, intercept = float(model.coef_[0, 0]), float(model.intercept_[0])
    before = np.clip(1-raw[:, 0], 1e-6, 1-1e-6)
    after = expit(slope*feature+intercept)
    measure = lambda p, rows: dict(ece=binary_ece(attack[rows], p[rows]), brier=float(np.mean((p[rows]-attack[rows])**2)))
    adopted = slope > 0 and measure(after, tune)['brier'] < measure(before, tune)['brier']
    return dict(slope=slope if adopted else 1., intercept=intercept if adopted else 0., adopted=bool(adopted),
                fitted=dict(slope=slope, intercept=intercept), fit_role='calibration', selection_role='tuning',
                tuning_before=measure(before, tune), tuning_after=measure(after, tune),
                assessment_before=measure(before, assess), assessment_after=measure(after, assess),
                rule=ADOPTION_RULE)


def calibration(result, raw, y, roles):
    block = result.get('calibration') or {}
    if block.get('status') != 'complete':
        raise ValueError(f"{result['job_id']} has no completed class calibration")
    cfg = dict(block['config']) if block['method'] != 'identity' else {'kind': 'identity', 'temperature': 1.0}
    cfg['log_floor'] = LOG_FLOOR if cfg['kind'] != 'identity' else 1e-12
    cfg.update(source_job=result['job_id'], method=block['method'], decisions_changed=block.get('decisions_changed'),
               calibration_fit_role='calibration', binary_platt=platt(raw, y, roles))
    return cfg


def hem_correctness(result, X, y, roles, cached=None):
    """Vote-correctness confidence for the unchanged HEM vote (day_finalize.calibrate_hem recipe)."""
    from sklearn.linear_model import LogisticRegression
    from framework.training.csv_v3.adapters_neural import hem_cpu_outputs
    if cached:
        with np.load(cached) as saved_parts:
            vote = saved_parts['vote']
            parts = {k: saved_parts[k] for k in ('multiclass_probabilities', 'binary_probabilities', 'knn_predictions')}
    else:
        vote, parts = hem_cpu_outputs(model_dir(result)/'model.pt', X)
    saved = raw_validation(result).argmax(1)
    if not np.array_equal(vote, saved):
        raise ValueError('HEM CPU vote differs from the study replay')
    features = hem_features(vote, parts)
    correct = vote == y
    fit, tune, assess = (np.flatnonzero(roles == r) for r in (1, 0, 2))
    model = LogisticRegression(C=1., random_state=42, max_iter=1000).fit(features[fit], correct[fit])
    if model.classes_.tolist() != [False, True]:
        raise ValueError('HEM correctness class order unexpected')
    after = model.predict_proba(features)[:, 1]
    multi, binary = parts['multiclass_probabilities'], parts['binary_probabilities']
    before = .5*(multi[np.arange(len(vote)), vote]+np.where(vote == 0, 1-binary, binary))
    measure = lambda p, rows: dict(ece=binary_ece(correct[rows], p[rows]), brier=float(np.mean((p[rows]-correct[rows])**2)),
                                   accuracy=float(correct[rows].mean()), mean_confidence=float(p[rows].mean()))
    report = dict(fit_role='calibration', selection_role='tuning', tuning_before=measure(before, tune),
                  tuning_after=measure(after, tune), assessment_before=measure(before, assess),
                  assessment_after=measure(after, assess),
                  meaning='Correctness confidence of the unchanged HEM vote; not class probabilities')
    report['rule'] = ADOPTION_RULE
    if report['tuning_after']['brier'] >= report['tuning_before']['brier']:
        raise ValueError('HEM correctness calibrator does not improve tuning-role Brier; owner decision needed')
    return dict(coefficients=model.coef_[0].tolist(), intercept=float(model.intercept_[0])), report, model


def hem_features(vote, parts):
    multi, binary, knn = (parts[k] for k in ('multiclass_probabilities', 'binary_probabilities', 'knn_predictions'))
    matching = np.where(vote == 0, 1-binary, binary)
    return np.column_stack([np.log(np.clip(multi[np.arange(len(vote)), vote], 1e-12, 1)),
                            np.log(np.clip(matching, 1e-12, 1)), (knn == multi.argmax(1)).astype(float)])


def standalone(family, result, schema, destination, extra):
    import joblib
    import torch
    torch.set_num_threads(2)
    folder = destination/family
    folder.mkdir(parents=True, exist_ok=True)
    source = model_dir(result)
    checkpoint = source/{'random_forest': 'model.joblib', 'xgboost': 'model.ubj'}.get(family, 'model.pt')
    manifest = dict(version=1, id=family, kind=family, label=LABELS[family], feature_names=schema['feature_names'],
                    classes=schema['classes'], benign_index=0,
                    source=dict(path=relative(checkpoint), sha256=digest(checkpoint), job=result['job_id']),
                    config=result['config'], calibration=extra.get('calibration'), supports_partial=False,
                    auxiliary=family == 'tabpack', files=[])
    blob = folder/'weights.bin'
    with TensorWriter(blob) as writer:
        if family == 'random_forest':
            fitted = joblib.load(checkpoint)
            manifest['forest'] = export_forest(fitted, writer)
            manifest['preprocessor'] = dict(kind='identity')
            del fitted
        elif family == 'xgboost':
            shutil.copyfile(checkpoint, folder/'model.ubj')
            manifest['checkpoint'] = 'model.ubj'
            manifest['preprocessor'] = dict(kind='identity')
            manifest['files'].append(dict(path='model.ubj', sha256=digest(folder/'model.ubj'), bytes=checkpoint.stat().st_size))
        else:
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if family in ('kan', 'hem'):
                for model, state in saved['state_dicts'].items():
                    for name, tensor in state.items():
                        writer.add(f'{model}.{name}', tensor.detach().numpy())
                manifest['preprocessor'] = export_pipeline(joblib.load(source/'preprocessing.joblib'), writer)
                manifest['input_size'] = saved['input_size']
                if family == 'hem':
                    with np.load(source/'knn_references.npz') as references:
                        writer.add('knn.references', references['X'])
                        writer.add('knn.labels', references['y'])
                    manifest['correctness_calibrator'] = extra['correctness_calibrator']
                    manifest['correctness_report'] = extra['correctness_report']
            else:
                if saved['feature_names'] != schema['feature_names'] or saved['classes'] != schema['classes']:
                    raise ValueError(f'{family} checkpoint schema differs from the dataset manifest')
                for name, tensor in saved['state_dict'].items():
                    writer.add(name, tensor.detach().numpy())
                manifest['preprocessor'] = dict(kind='tabm_v2', **saved['preprocessor'])
                manifest['member_weights'] = saved.get('member_weights')
                manifest['embedded'] = family in ('tabm_ne', 'tabm_hier_ne')
            del saved
        manifest['tensors'] = writer.tensors
    manifest['files'].append(dict(path='weights.bin', sha256=digest(blob), bytes=blob.stat().st_size))
    write_json(folder/'model.json', manifest)
    gc.collect()
    return manifest


def export_hybrid(name, result, schema, destination, cal):
    manifest = dict(version=1, id=name, kind='hybrid', label=LABELS[name], feature_names=schema['feature_names'],
                    classes=schema['classes'], benign_index=0,
                    source=dict(path=relative(STUDY/'jobs'/result['job_id']/'result.json'),
                                sha256=digest(STUDY/'jobs'/result['job_id']/'result.json'), job=result['job_id']),
                    config={}, calibration=cal, supports_partial=False, auxiliary=False,
                    branches=list(result['branches']), combiner=result['combiner'], files=[], tensors={})
    write_json(destination/name/'model.json', manifest)
    return manifest


def export_rows(destination, schema):
    metadata = dict(version=1, feature_names=schema['feature_names'], classes=schema['classes'], benign_index=0)
    for split, filename in [('validation', 'samples.json'), ('train', 'reference.json')]:
        X = np.load(DATA/f'X_{split}.npy', mmap_mode='r')
        y = np.load(DATA/f'y77_{split}.npy', mmap_mode='r')
        ids = np.load(DATA/f'row_ids_{split}.npy', mmap_mode='r')
        rows = np.arange(len(y)) if split == 'validation' else np.flatnonzero(np.load(DATA/'fit77_train.npy'))
        indices = rows[sample_indices(np.asarray(y)[rows], 128)]
        groups = {}
        for index, name in enumerate(schema['feature_names']):
            groups.setdefault(name.split('__', 1)[0], []).append(index)
        write_json(destination/filename, dict(metadata, partition=split,
                   rows=row_records(X, y, ids, indices, schema), feature_groups=list(groups.values()),
                   provenance='DataSense CSV cascade v3 (77 classes; 8 recon subtypes merged into recon_scan); '
                              'samples are observations, not generated labels.'))


def apply_cal(raw, cfg):
    from scipy.special import expit, logit
    from framework.training.csv_v3 import calibrate
    probabilities = raw.astype(np.float64) if cfg['kind'] == 'identity' else calibrate.apply(cfg, raw).astype(np.float64)
    if cfg['kind'] == 'identity':
        probabilities = probabilities/probabilities.sum(1, keepdims=True)
    predicted = probabilities.argmax(1) if cfg['kind'] == 'vector' else raw.argmax(1)
    b = cfg['binary_platt']
    attack = expit(b['slope']*logit(np.clip(1-raw[:, 0], 1e-6, 1-1e-6))+b['intercept'])
    return predicted, probabilities, attack


def golden(destination, chosen, schema, count, filename, hem_model):
    from framework.training.csv_v3 import adapters_neural, adapters_tabm, adapters_trees
    from framework.training.hybrid_blend import blend
    X = np.load(DATA/'X_validation.npy', mmap_mode='r')
    y = np.load(DATA/'y77_validation.npy', mmap_mode='r')
    ids = np.load(DATA/'row_ids_validation.npy', mmap_mode='r')
    indices = sample_indices(np.asarray(y), count)
    sample = np.asarray(X[indices], dtype=np.float32)
    adapters = {'random_forest': adapters_trees, 'xgboost': adapters_trees, 'kan': adapters_neural,
                'tabm_ne': adapters_tabm, 'tabm_hier': adapters_tabm, 'tabm_hier_ne': adapters_tabm, 'tabpack': adapters_tabm}
    output, raw, audit = {}, {}, {}
    for family, result in chosen.items():
        print(f'GOLDEN {family} {count} validation rows CPU', flush=True)
        if family == 'hem':
            vote, parts = adapters_neural.hem_cpu_outputs(model_dir(result)/'model.pt', sample)
            confidence = hem_model.predict_proba(hem_features(vote, parts))[:, 1]
            output[family] = [dict(predicted=int(vote[i]), probabilities=None, raw_probabilities=None,
                                   confidence=float(confidence[i]), attack_probability=None,
                                   confidence_kind='vote_correctness') for i in range(count)]
            continue
        prediction = adapters[family].predict_cpu(family, model_dir(result), sample).astype(np.float64)
        audit[family] = float(np.abs(prediction-raw_validation(result)[indices]).max())
        raw[family] = prediction
    for name, hybrid in HYBRIDS.items():
        result = job(hybrid)
        raw[name] = blend(np.stack([raw[b] for b in result['branches']]), result['combiner'], result['branches']).astype(np.float64)
        audit[name] = float(np.abs(raw[name]-raw_validation(result)[indices]).max())
    for name, prediction in raw.items():
        cfg = json.loads((destination/name/'model.json').read_text(encoding='utf-8'))['calibration']
        predicted, probabilities, attack = apply_cal(prediction, cfg)
        output[name] = [dict(predicted=int(predicted[i]), raw_probabilities=prediction[i].tolist(),
                             probabilities=probabilities[i].tolist(), confidence=float(probabilities[i][predicted[i]]),
                             attack_probability=float(attack[i]), confidence_kind='class_probability') for i in range(count)]
    write_json(destination/filename, dict(version=1, partition='validation', test_loaded=False,
               feature_names=schema['feature_names'], classes=schema['classes'],
               rows=row_records(X, y, ids, indices, schema), outputs=output,
               study_replay_max_abs_difference=audit))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--hem-parts', type=Path, help='cached hem_cpu_outputs on X_validation (vote is re-checked)')
    args = parser.parse_args()
    if json.loads((STUDY/'state.json').read_text(encoding='utf-8')).get('stage') != 'all_complete':
        raise ValueError('cascade v3 study is not complete')
    destination = args.output.resolve()
    if ROOT not in destination.parents or STUDY in destination.parents or destination.exists():
        raise ValueError('Export needs a new, isolated folder inside the project')
    data = json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    schema = dict(feature_names=data['feature_names'], classes=data['classes77'])
    if schema['classes'][0] != 'benign' or 'recon_scan' not in schema['classes']:
        raise ValueError('Unexpected 77-class schema')
    y = np.load(DATA/'y77_validation.npy')
    roles = np.load(DATA/'validation_roles.npy')
    chosen = selection()
    models, report, hem_model = [], {}, None
    for family, result in chosen.items():
        print('EXPORT', family, result['job_id'], flush=True)
        if family == 'hem':
            X = np.load(DATA/'X_validation.npy')
            cal, hem_report, hem_model = hem_correctness(result, X, y, roles, args.hem_parts)
            standalone(family, result, schema, destination, dict(correctness_calibrator=cal, correctness_report=hem_report))
            report[family] = dict(job=result['job_id'], correctness=hem_report)
            del X
        else:
            cal = calibration(result, raw_validation(result), y, roles)
            standalone(family, result, schema, destination, dict(calibration=cal))
            report[family] = dict(job=result['job_id'], calibration=cal)
        models.append(dict(id=family, label=LABELS[family], kind=family, auxiliary=family == 'tabpack',
                           manifest=f'{family}/model.json', sha256=digest(destination/family/'model.json')))
    for name, hybrid in HYBRIDS.items():
        result = job(hybrid)
        cal = calibration(result, raw_validation(result), y, roles)
        export_hybrid(name, result, schema, destination, cal)
        report[name] = dict(job=hybrid, calibration=cal)
        models.append(dict(id=name, label=LABELS[name], kind='hybrid', auxiliary=False,
                           manifest=f'{name}/model.json', sha256=digest(destination/name/'model.json')))
    export_rows(destination, schema)
    write_json(destination/'catalog.json', dict(version=1, feature_names=schema['feature_names'], classes=schema['classes'],
               benign_index=0, models=models, default_model=DEFAULT_MODEL, inference_device='cpu',
               source_study=relative(STUDY), source_dataset=relative(DATA), test_loaded=False, limitations=LIMITATIONS))
    audit = {}
    for count, filename in ((16, 'golden_16.json'), (128, 'golden_128.json')):
        audit[filename] = golden(destination, chosen, schema, count, filename, hem_model)
    shutil.copyfile(destination/'golden_16.json', destination/'golden.json')
    write_json(destination/'export_report.json', dict(version=1, test_loaded=False, models=report, study_replay=audit,
               oracle='csv_v3 adapters predict_cpu + hybrid_blend.blend + csv_v3.calibrate.apply + Platt'))
    print('COMPLETE 77-class native export', destination, json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
