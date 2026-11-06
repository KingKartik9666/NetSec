"""Bounded, matched feature ablations on a separate within-training role.

This is exploratory diagnosis. It does not tune on the original validation,
unlock test data, change a frozen profile or promote a deployed checkpoint.
"""
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from framework.training.live_study.protocol import (
    ROOT, FEATURES, read, atomic, digest, fingerprint, relative, verify, versions,
)
from .ports import join_port_features


def wilson_interval(successes, total):
    if total == 0:
        return None
    z = 1.959963984540054
    rate = successes / total
    center = (rate + z * z / (2 * total)) / (1 + z * z / total)
    radius = z * np.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [max(0., float(center - radius)), min(1., float(center + radius))]


def classification_report(y, probabilities):
    n = probabilities.shape[1]
    predicted = probabilities.argmax(1)
    cm = np.bincount(n * y.astype(int) + predicted, minlength=n*n).reshape(n, n)
    denominator = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), denominator, out=np.zeros(n), where=denominator > 0)
    support = cm.sum(1)
    return dict(accuracy=float(np.mean(y == predicted)), macro_f1_all_classes=float(f1.mean()),
                macro_f1_supported_classes=float(f1[support > 0].mean()),
                supported_classes=int((support > 0).sum()), class_support=support.tolist(),
                per_class_recall=np.divide(cm.diagonal(), support, out=np.full(n, np.nan),
                                          where=support > 0), confusion_matrix=cm.tolist())


def probability_report(y, probabilities, classes):
    p = np.asarray(probabilities, float)
    y = np.asarray(y, int)
    if len(y) == 0 or p.shape != (len(y), len(classes)) or classes[0] != 'benign':
        raise ValueError('Nonempty, aligned labelled probabilities with benign index zero required')
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid class probabilities')
    p = p / p.sum(1, keepdims=True)
    full = classification_report(y, p)
    families = list(dict.fromkeys(name.split('_', 1)[0] for name in classes))
    mapping = np.array([families.index(name.split('_', 1)[0]) for name in classes])
    family_p = np.column_stack([p[:, mapping == i].sum(1) for i in range(len(families))])
    family = classification_report(mapping[y], family_p)
    family['classes'] = families
    for report in (full, family):
        report['per_class_recall'] = [float(v) if np.isfinite(v) else None for v in report['per_class_recall']]
    truth = y != 0
    score = 1 - p[:, 0]
    alarm = score >= .5
    benign_n = int((~truth).sum())
    attack_n = int(truth.sum())
    fp = int((alarm & ~truth).sum())
    tp = int((alarm & truth).sum())
    binary = dict(threshold=.5, method='sum of attack probabilities; no separately trained binary classifier',
                  accuracy=float(np.mean(alarm == truth)), benign_windows=benign_n, attack_windows=attack_n,
                  false_alarms=fp, false_positive_rate=fp/benign_n if benign_n else None,
                  false_positive_rate_wilson95=wilson_interval(fp, benign_n),
                  recall=tp/attack_n if attack_n else None,
                  precision=tp/int(alarm.sum()) if alarm.any() else None,
                  average_precision=float(average_precision_score(truth, score)) if attack_n else None,
                  roc_auc=float(roc_auc_score(truth, score)) if attack_n and benign_n else None,
                  brier=float(np.mean((score - truth)**2)))
    return dict(rows=len(y), multiclass=full, family=family, binary=binary,
                negative_log_likelihood=float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)).mean()),
                calibrated=False)


def stress_missing_mss(x, names):
    changed = np.array(x, copy=True)
    columns = [i for i, name in enumerate(names) if name.startswith('network_mss_')]
    if not columns:
        return changed, 0
    affected = np.all(changed[:, columns] == 0, axis=1)
    changed[np.ix_(affected, columns)] = np.nan
    return changed, int(affected.sum())


def _fit(x, y, classes, device, rounds, seed):
    import xgboost as xgb
    import json
    params = dict(objective='multi:softprob', num_class=len(classes), tree_method='hist', device=device,
                  max_depth=5, min_child_weight=2, eta=.08, subsample=.9, colsample_bytree=.9,
                  max_bin=128, nthread=2, seed=seed, reg_lambda=2, eval_metric='mlogloss')
    model = xgb.train(params, xgb.DMatrix(x, label=y, nthread=2), num_boost_round=rounds)
    actual = json.loads(model.save_config())['learner']['generic_param']['device']
    if device == 'cuda' and not actual.startswith('cuda'):
        raise RuntimeError('Explicit CUDA fitting cannot silently use CPU')
    return model, params


def run(study, device='cpu', rounds=100):
    """Seven predeclared arms, three seeds, without automatic winner selection."""
    import xgboost as xgb
    study = Path(study)
    output = study / 'matched_assessment'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'report.json').exists():
        raise FileExistsError('Preserve completed diagnostic evidence; use a new revision')
    manifest = read(study / 'data/manifest.json')
    verify(manifest['array_hashes'])
    profile = read(study / 'selected_profile.json')
    if profile['selection_sha256'] != digest(study / 'selection.json'):
        raise ValueError('Frozen selection changed')
    x = np.load(study / 'data/X_train.npy')
    y = np.load(study / 'data/y_train.npy')
    roles = np.load(study / 'data/roles.npy')
    names = manifest['feature_names']
    classes = manifest['classes']
    selected = profile['feature_names']
    ports, port_names, port_report = join_port_features(study)
    atomic(output / 'port_profile.json', port_report)
    all_x = np.column_stack((x, ports))
    all_names = names + port_names
    arms = {
        'seven_counts': FEATURES,
        'selected60': selected,
        'without_tcp_flag_std': [n for n in selected if n != 'network_tcp-flags_std_deviation'],
        'without_ttl_min': [n for n in selected if n != 'network_ttl_min'],
        'without_dst_fraction': [n for n in selected if n != 'network_dst_packet_fraction'],
        'without_mss': [n for n in selected if not n.startswith('network_mss_')],
        'selected_plus_ports': selected + port_names,
    }
    source_files = [Path(__file__), Path(__file__).with_name('ports.py'), Path(__file__).with_name('selection.py')]
    protocol = dict(arms=arms, seeds=[42, 43, 44], rounds=rounds, device=device,
                    fitting_role=0, assessment_role=1, fit_rows=int((roles == 0).sum()),
                    assess_rows=int((roles == 1).sum()),
                    selection_hash=digest(study / 'selection.json'),
                    manifest_hash=digest(study / 'data/manifest.json'),
                    port_profile_hash=digest(output / 'port_profile.json'),
                    source_hashes={relative(p): digest(p) for p in source_files},
                    limitation='Tune role unused by broad selector but reused in historical seven-count study; exploratory within-capture assessment',
                    calibration_fitted=False, validation_loaded=False, test_loaded=False, live_accuracy_verified=False)
    if (output / 'protocol.json').exists() and read(output / 'protocol.json') != protocol:
        raise ValueError('Diagnostic protocol changed; preserve attempt and create a new revision')
    atomic(output / 'protocol.json', protocol)
    records = []
    for arm, fields in arms.items():
        columns = [all_names.index(name) for name in fields]
        train_x = all_x[roles == 0][:, columns]
        assess_x = all_x[roles == 1][:, columns]
        for seed in protocol['seeds']:
            record_path = output / f'{arm}_{seed}.json'
            if record_path.exists():
                previous = read(record_path)
                if previous['protocol_sha256'] != digest(output / 'protocol.json'):
                    raise ValueError('Stored attempt uses a different protocol')
                verify(previous['artifact_hashes'])
                records.append(previous)
                continue
            start = time.monotonic()
            model, params = _fit(train_x, y[roles == 0], classes, device, rounds, seed)
            p = model.predict(xgb.DMatrix(assess_x, nthread=2))
            checkpoint = output / f'{arm}_{seed}.ubj'
            model.save_model(checkpoint)
            replay = xgb.Booster()
            replay.load_model(checkpoint)
            replay.set_param(dict(device='cpu', nthread=2))
            cpu_p = replay.predict(xgb.DMatrix(assess_x, nthread=2))
            np.testing.assert_allclose(cpu_p, p, atol=1e-6, rtol=1e-5)
            prediction_path = output / f'{arm}_{seed}_probabilities.npy'
            np.save(prediction_path, cpu_p)
            result = probability_report(y[roles == 1], cpu_p, classes)
            stressed, affected = stress_missing_mss(assess_x, fields)
            stress = None
            if affected:
                stress_p = replay.predict(xgb.DMatrix(stressed, nthread=2))
                stress = dict(affected_windows=affected, report=probability_report(y[roles == 1], stress_p, classes),
                              interpretation='Counterfactual zero-summary to absent-MSS stress, not a verified source repair or new labels')
            record = dict(arm=arm, seed=seed, feature_count=len(fields), feature_names=fields,
                          params=params, report=result, missing_mss_stress=stress,
                          cpu_reload_max_abs_error=float(np.max(np.abs(cpu_p-p))),
                          elapsed_seconds=time.monotonic()-start,
                          artifact_hashes={relative(f): digest(f) for f in (checkpoint, prediction_path)},
                          protocol_sha256=digest(output / 'protocol.json'))
            atomic(record_path, record)
            records.append(record)
            print(f'{arm} seed{seed}: accuracy={result["multiclass"]["accuracy"]:.4%}, '
                  f'macroF1={result["multiclass"]["macro_f1_all_classes"]:.4%}', flush=True)
    result = dict(protocol=protocol, records=records, versions=versions(),
                  new_profile_promoted=False, models='Fixed XGBoost feature experiments only; not all ten model families')
    atomic(output / 'report.json', result)
    lines = ['# Matched feature assessment', '', protocol['limitation'], '',
             '| Recipe | Features | Mean accuracy | Mean macro F1 (84 classes) | Binary FPR at 0.5 |',
             '|---|---:|---:|---:|---:|']
    for arm in arms:
        group = [r for r in records if r['arm'] == arm]
        avg = lambda section, key: np.mean([r['report'][section][key] for r in group])
        lines.append(f'| {arm} | {len(arms[arm])} | {avg("multiclass", "accuracy"):.4%} | '
                     f'{avg("multiclass", "macro_f1_all_classes"):.4%} | {avg("binary", "false_positive_rate"):.4%} |')
    lines += ['', 'Seeds 42/43/44, identical 100-round recipes (see protocol for actual rounds), '
              'same fit/assessment rows. CPU checkpoint replay covers all assessment rows.',
              'Binary and family results sum multiclass probabilities; they are not separately trained hierarchical models.',
              'No automatic feature winner or deployment promotion. No final original validation or held-out test was opened.']
    (output / 'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result
