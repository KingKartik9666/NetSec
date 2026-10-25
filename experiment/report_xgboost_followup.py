"""Measured comparisons, training-only temperature fit and frozen candidate."""
from __future__ import annotations

import csv
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax

from experiment.xgboost_followup import (ARMS, ROOT, SEEDS, arrays, digest, eligible,
                                        read, temporal_folds)
from framework.training.tracking import classification_metrics, write_json

KEYS = ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
        'false_positive_rate', 'attack_missed_as_benign_rate')


def summarize_arms(records):
    control = [records[f'original_control_s{s}_base'] for s in SEEDS]
    summary = {}
    for arm in ARMS:
        rows = [records[f'{arm}_s{s}_base'] for s in SEEDS]
        summary[arm] = dict(features=len(rows[0]['feature_names']),
            seeds=[dict(seed=s, **{k: r['metrics'][k] for k in KEYS}) for s, r in zip(SEEDS, rows)],
            mean={key: float(np.mean([r['metrics'][key] for r in rows])) for key in KEYS},
            standard_deviation={key: float(np.std([r['metrics'][key] for r in rows], ddof=1)) for key in KEYS},
            paired_change={key: [a['metrics'][key]-b['metrics'][key] for a, b in zip(rows, control)] for key in KEYS},
            guards_pass=all(eligible(a['metrics'], b['metrics']) for a, b in zip(rows, control)))
    allowed = [arm for arm in ARMS if summary[arm]['guards_pass']]
    chosen = max(allowed, key=lambda arm: (summary[arm]['mean']['accuracy'],
                        summary[arm]['mean']['macro_f1'], -summary[arm]['features']))
    return summary, chosen


def calibrate(probabilities, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive finite temperature required')
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Expected normalized probabilities')
    return softmax(np.log(np.clip(p, 1e-30, 1)) / temperature, axis=1)


def fit_temperature(probabilities, truth):
    """Bounded-memory scalar NLL fit; only genuine held-fold predictions enter."""
    truth = np.asarray(truth, dtype=np.int64)
    if len(truth) != len(probabilities) or len(truth) == 0:
        raise ValueError('Misaligned calibration rows')
    def loss(log_temperature):
        total = 0.
        for begin in range(0, len(truth), 4096):
            p = np.asarray(probabilities[begin:begin+4096], dtype=np.float64)
            if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5):
                raise ValueError('Invalid held-fold probabilities')
            logits = np.log(np.clip(p, 1e-30, 1)) / np.exp(log_temperature)
            total += np.sum(logsumexp(logits, axis=1) - logits[np.arange(len(p)), truth[begin:begin+len(p)]])
        return float(total / len(truth))
    optimum = minimize_scalar(loss, bounds=(np.log(.1), np.log(10)), method='bounded')
    if not optimum.success:
        raise RuntimeError('Temperature optimization failed')
    return dict(temperature=float(np.exp(optimum.x)), oof_nll_before=loss(0.),
                oof_nll_after=float(optimum.fun), bounds=[.1, 10.])


def probability_metrics(truth, probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.int64)
    correct = p.argmax(1) == truth
    confidence = p.max(1)
    bins, ece = [], 0.
    for i in range(15):
        low, high = i/15, (i+1)/15
        mask = (confidence >= low) & (confidence < high if i < 14 else confidence <= high)
        row = dict(lower=low, upper=high, rows=int(mask.sum()))
        if mask.any():
            row.update(accuracy=float(correct[mask].mean()), confidence=float(confidence[mask].mean()))
            ece += float(mask.mean() * abs(row['accuracy']-row['confidence']))
        bins.append(row)
    return dict(accuracy=float(correct.mean()), mean_confidence=float(confidence.mean()),
                nll=float(-np.log(np.clip(p[np.arange(len(truth)), truth], 1e-30, 1)).mean()),
                multiclass_brier=float(np.mean(np.sum(p*p, axis=1)-2*p[np.arange(len(truth)), truth]+1)),
                confidence_brier=float(np.mean((confidence-correct)**2)), ece_15_bins=ece, bins=bins)


def finish(directory, source, records, selected, folds):
    directory, source = Path(directory), Path(source)
    manifest = read(source / 'prepared/manifest.json')
    data = arrays(source)
    chosen = selected[0]  # Always seed 42, never the luckiest replicate.
    rows = np.arange(len(data['train']['y']))
    if chosen['recipe']['arm'] != 'original_control':
        rows = rows[np.load(source / 'prepared/training_keep.npy')]
    ids = data['train']['row_ids'][rows]
    if np.any(np.diff(ids) <= 0):
        raise ValueError('Expected distinct ordered training source IDs')
    expected = temporal_folds(*(data['train'][key][rows] for key in
                               ('y', 'window_start', 'window_end', 'feature_hash')))
    out = directory / 'calibration'
    out.mkdir(exist_ok=True)
    oof = np.lib.format.open_memmap(out / 'training_oof_probabilities.npy', mode='w+',
                                  dtype=np.float32, shape=(len(rows), len(manifest['classes'])))
    seen = np.zeros(len(rows), dtype=bool)
    provenance = []
    for number, record in enumerate(folds):
        fitting, held = expected[number]
        actual_fit = np.load(ROOT / record['paths']['fit_ids'])
        actual_held = np.load(ROOT / record['paths']['scoring_ids'])
        np.testing.assert_array_equal(actual_fit, ids[fitting])
        np.testing.assert_array_equal(actual_held, ids[held])
        if np.intersect1d(actual_fit, actual_held).size or seen[held].any():
            raise ValueError('OOF fitting/hold overlap')
        np.testing.assert_array_equal(np.load(ROOT / record['paths']['truth']), data['train']['y'][rows[held]])
        oof[held] = np.load(ROOT / record['paths']['probabilities'], mmap_mode='r')
        seen[held] = True
        provenance.append(dict(fold=number, fitting_rows=len(fitting), held_rows=len(held),
                               fit_ids_sha256=record['artifact_hashes']['fit_ids'],
                               held_ids_sha256=record['artifact_hashes']['scoring_ids']))
    if not seen.all():
        raise ValueError('Incomplete OOF coverage')
    oof.flush()
    np.save(out / 'training_row_ids.npy', ids)
    calibration = dict(fit_temperature(oof, data['train']['y'][rows]),
        method='positive scalar temperature, trained on purged training-only cross-fit predictions',
        folds=provenance, rows=len(rows), validation_labels_used_for_calibrator=False,
        representation='Label-independent vocabulary frozen on outer training; not refitted per fold',
        protocol='Three class-conditional temporal blocks; touching windows and repeated original inputs purged from fitting',
        test_loaded=False)
    write_json(out / 'calibrator.json', calibration)
    p = np.load(ROOT / chosen['paths']['probabilities'])
    y = np.asarray(data['validation']['y'])
    predictions = np.load(ROOT / chosen['paths']['predictions'])
    np.testing.assert_array_equal(p.argmax(1), predictions)
    calibrated = calibrate(p, calibration['temperature'])
    np.testing.assert_array_equal(calibrated.argmax(1), predictions)
    np.save(out / 'validation_probabilities.npy', calibrated)
    np.testing.assert_allclose(calibrate(p, read(out / 'calibrator.json')['temperature']),
                               np.load(out / 'validation_probabilities.npy'), atol=0, rtol=0)
    scores = dict(before=probability_metrics(y, p), after=probability_metrics(y, calibrated),
                  argmax_unchanged=True, calibration_reload_verified=True,
                  scope='Exploratory original validation after recipe selection; not an untouched held-out assessment')
    write_json(out / 'assessment.json', scores)
    # Independent recomputation for every validation result, including all seeds.
    recomputed = {}
    for name, record in records.items():
        if 'fold' in record['recipe']:
            continue
        probabilities = np.load(ROOT / record['paths']['probabilities'])
        pred = np.load(ROOT / record['paths']['predictions'])
        np.testing.assert_array_equal(probabilities.argmax(1), pred)
        measured = classification_metrics(y, pred, manifest['classes'], manifest['benign_index'])
        for key in KEYS:
            if abs(record['metrics'][key]-measured[key]) > 1e-12:
                raise ValueError(f'Recomputed metric mismatch: {name}/{key}')
        recomputed[name] = {key: measured[key] for key in KEYS}
    write_json(directory / 'metric_verification.json', recomputed)
    base = records['original_control_s42_base']
    old_pred = np.load(ROOT / base['paths']['predictions'])
    cm = np.asarray(chosen['metrics']['confusion_matrix']).copy()
    np.fill_diagonal(cm, 0)
    errors = [dict(actual=manifest['classes'][i], predicted=manifest['classes'][j], rows=int(cm[i, j]))
              for i, j in (np.unravel_index(k, cm.shape) for k in np.argsort(-cm.ravel())[:50]) if cm[i, j]]
    with (directory / 'top_confusions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['actual', 'predicted', 'rows'])
        writer.writeheader(); writer.writerows(errors)
    correct_old, correct_new = old_pred == y, predictions == y
    changes = dict(newly_correct=int((~correct_old & correct_new).sum()),
                   newly_incorrect=int((correct_old & ~correct_new).sum()),
                   unchanged_correct=int((correct_old & correct_new).sum()),
                   unchanged_incorrect=int((~correct_old & ~correct_new).sum()),
                   top_confusions=errors)
    write_json(directory / 'error_analysis.json', changes)
    frozen = directory / 'frozen_candidate'
    frozen.mkdir(exist_ok=True)
    for src, dest in [(ROOT / chosen['paths']['model'], frozen / 'model.ubj'),
                      (source / 'prepared/feature_transform.json', frozen / 'feature_transform.json'),
                      (out / 'calibrator.json', frozen / 'calibrator.json')]:
        if dest.exists() and digest(dest) != digest(src):
            raise ValueError(f'Frozen candidate differs: {dest}')
        if not dest.exists():
            shutil.copy2(src, dest)
    bundle = dict(recipe=chosen['recipe'], classes=manifest['classes'], benign_index=manifest['benign_index'],
                  feature_names=chosen['feature_names'], feature_indexes=chosen['feature_indexes'],
                  input='Original canonical numeric/protocol columns followed by audited port/duration/rate columns',
                  array_order='base_feature_names + extra_feature_names from feature_transform.json, then feature_indexes',
                  checkpoint_sha256=digest(frozen / 'model.ubj'),
                  transform_sha256=digest(frozen / 'feature_transform.json'),
                  calibrator_sha256=digest(frozen / 'calibrator.json'),
                  schema_compatible_with_current_app=False, promoted=False, test_loaded=False,
                  status='Frozen research candidate; requires explicit raw-input integration and held-out protocol')
    write_json(frozen / 'bundle.json', bundle)
    comparison, _ = summarize_arms(records)
    final = dict(selected_recipe=chosen['recipe'], selected_metrics=chosen['metrics'],
                 mean_accuracy=float(np.mean([r['metrics']['accuracy'] for r in selected])),
                 mean_macro_f1=float(np.mean([r['metrics']['macro_f1'] for r in selected])),
                 representation_comparison=comparison, calibration=scores,
                 training_fits=sum(not r['adopted'] for r in records.values()),
                 adopted_fits=sum(r['adopted'] for r in records.values()),
                 rows=len(y), target_95_reached=all(r['metrics']['accuracy'] >= .95 for r in selected),
                 candidate=bundle, test_loaded=False)
    write_json(directory / 'summary.json', final)
    lines = ['# XGBoost follow-up results', '',
        'Real CPU fits on the original source training/validation split. All 30,000 validation rows retained.', '',
        '| Representation | Accuracy mean +/- SD (%) | Macro F1 mean +/- SD (%) |',
        '|---|---:|---:|']
    for arm, row in comparison.items():
        lines.append(f"| {arm} | {row['mean']['accuracy']*100:.3f} +/- {row['standard_deviation']['accuracy']*100:.3f} | "
                     f"{row['mean']['macro_f1']*100:.3f} +/- {row['standard_deviation']['macro_f1']*100:.3f} |")
    lines += ['', f"Selected recipe: `{chosen['recipe']}`.",
        f"Three-seed accuracy: {final['mean_accuracy']*100:.3f}%; macro F1: {final['mean_macro_f1']*100:.3f}%.",
        f"Frozen seed-42 checkpoint: accuracy {chosen['metrics']['accuracy']*100:.3f}%, "
        f"macro recall {chosen['metrics']['macro_recall']*100:.3f}%, macro F1 {chosen['metrics']['macro_f1']*100:.3f}%.", '',
        f"Training-only temperature: {calibration['temperature']:.6f}; validation NLL "
        f"{scores['before']['nll']:.6f} -> {scores['after']['nll']:.6f}; ECE "
        f"{scores['before']['ece_15_bins']:.6f} -> {scores['after']['ece_15_bins']:.6f}.",
        'All class decisions are unchanged by calibration. Any worsening probability metrics remain reported.', '',
        f"Against the matched original seed-42 control: {changes['newly_correct']} errors corrected, "
        f"{changes['newly_incorrect']} previously correct rows now incorrect. See top_confusions.csv.", '',
        'All seeds share the same reused validation set. Variation across seeds is not a confidence interval for new captures.',
        'Data filtering changes training distribution; this does not prove quiet windows or their labels are corrupt.',
        'The vocabulary is frozen from original training, including inner hold inputs, without their labels.',
        'Calibration uses purged training folds; earlier validation-based recipe selection remains exploratory.',
        'No locked-test scoring, 95% guarantee, real HEM reproduction, app promotion or live generalization is claimed.', '',
        'Checkpoint and feature mapping: frozen_candidate/. Numerical provenance: protocol.json, records/, runs/, calibration/.']
    (directory / 'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
