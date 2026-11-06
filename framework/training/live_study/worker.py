"""One isolated fit, CPU replay, tuning evaluation and calibration transaction."""
from __future__ import annotations

import os
from pathlib import Path
import time
import traceback
import numpy as np

from . import calibration
from .data import training_data
from .protocol import ROOT, HYBRIDS, ROLES, atomic, read, relative, digest, fingerprint, verify, verify_study


def predict(record, x):
    import torch
    torch.set_num_threads(2)
    family = record['family']; checkpoint = ROOT/record['checkpoint']
    if family in HYBRIDS:
        from framework.training.hybrid_blend import blend
        branches = record['branches']
        p = np.stack([predict(branches[name], x)['probabilities'] for name in HYBRIDS[family]])
        probability = blend(p, read(checkpoint), HYBRIDS[family])
    elif family == 'hem':
        from framework.training.day_worker import hem_cpu_outputs
        pred, parts = hem_cpu_outputs(checkpoint, x)
        return dict(pred=pred, **parts)
    elif family == 'random_forest':
        import joblib
        model = joblib.load(checkpoint)
        if not np.array_equal(model.classes_, np.arange(record['n_classes'])):
            raise ValueError('Random forest classes differ from the sealed schema')
        probability = model.predict_proba(x).astype(np.float32)
    elif family == 'tabpack':
        from framework.training.hybrid_neural import load, predict as forward
        model, prep, _, weights = load(checkpoint)
        probability = forward(model, prep, x, family, weights)
    elif family.startswith('tabm_'):
        from framework.training.tabm_fit import load_checkpoint, predict as forward
        model, prep, _ = load_checkpoint(checkpoint)
        probability = forward(model, prep, x)
    else:
        from experiment.train_csv_models import reload_outputs
        _, probability = reload_outputs(family, checkpoint, x)
    return dict(pred=probability.argmax(1), probabilities=probability)


def load_outputs(record, role):
    with np.load(ROOT/record['run_directory']/f'{role}_outputs.npz') as a:
        return {k:a[k] for k in a.files}


def verify_result(path, request=None):
    result = read(path)
    if result['status'] not in ('completed','pruned'): raise ValueError('Not an evaluated result')
    if request is not None and result['request_fingerprint'] != request['request_fingerprint']:
        raise ValueError('Result belongs to a different persisted proposal')
    verify(result['artifact_hashes'])
    return result


def _fit(request, x, y, roles, manifest, output):
    family, cfg, seed = request['family'], request['config'], request['seed']
    fit_ids, tune = [np.flatnonzero(roles == i) for i in (0,1)]
    a, b, v, w = x[fit_ids], y[fit_ids], x[tune], y[tune]
    count = len(manifest['classes'])
    print(f'TRAIN {family} seed={seed} fit={len(b)} tune={len(w)}', flush=True)
    if family == 'random_forest':
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from framework.training.trees import class_weights
        model = RandomForestClassifier(**{k:v for k,v in cfg.items() if k not in ('runtime_seconds','weight_strategy')}, random_state=seed)
        started = time.monotonic()
        model.fit(a, b, sample_weight=class_weights(b, count, cfg['weight_strategy'])[b])
        checkpoint = output/'model.joblib'; joblib.dump(model, checkpoint)
        details = dict(backend='scikit-learn bootstrap RandomForestClassifier; new live adaptation',
            training_seconds=time.monotonic()-started, training_device='cpu', trees=len(model.estimators_))
    elif family == 'tabpack':
        from framework.training.hybrid_neural import train
        # make_pack consumes config.seed; repeat seeds must reach the actual model.
        cfg = dict(cfg, seed=seed)
        details = train(family,a,b,v,w,manifest,cfg,seed,output,'cuda')
        checkpoint = output/'model.pt'
    elif family.startswith('tabm_'):
        from framework.training.tabm_fit import train
        model, prep, details = train(a,b,v,w,manifest,request,output,'cuda')
        del model, prep
        checkpoint = output/'model.pt'
    elif family == 'xgboost':
        from framework.training.csv_xgboost import train_csv_xgboost
        details = train_csv_xgboost(a,b,v,w,count,cfg,seed,output)
        checkpoint = output/'model.ubj'
    else:
        from framework.training.neural import train_neural
        details = train_neural(family,a,b,v,w,count,0,cfg,seed,output)
        checkpoint = output/'model.pt'
    for key in ('predictions','probabilities','history','checkpoint_path'): details.pop(key,None)
    return checkpoint, details


def _hybrid(request, study, y, roles, output):
    from framework.training.hybrid_blend import blend
    from .combiner import fit as fit_blend
    selected = read(study/'selected.json')['branches']
    names = HYBRIDS[request['family']]
    records = {name:verify_result(ROOT/selected[name]['result']) for name in names}
    if request['phase'] == 'repeat':
        # Frozen hybrid weights are replayed on independent seeded branch refits.
        records = {name:verify_result(study/f'repeats/{name}/seed{request["seed"]}/result-link.json') for name in names}
    p = np.stack([load_outputs(records[name],'blend')['probabilities'] for name in names])
    cfg = request['config']
    if request['phase'] == 'search':
        recipe = fit_blend(p, y[roles == 2], names, cfg['penalty'] if cfg['kind'] == 'classwise' else None)
        recipe['objective'] = 'independent holdout blending log loss; not OOF'
    else:
        recipe = read(ROOT/cfg['frozen_combiner'])
    checkpoint = output/'combiner.json'; atomic(checkpoint, recipe)
    outputs = {role:dict(probabilities=blend(np.stack([load_outputs(records[name],role)['probabilities']
                for name in names]),recipe,names)) for role in ROLES[1:]}
    for values in outputs.values(): values['pred'] = values['probabilities'].argmax(1)
    return checkpoint, dict(optimizer='SLSQP or L-BFGS-B', weight_fit_role='blend', recipe=recipe), records, outputs


def run(request_path):
    from framework.training import tracking
    tracking.write_json = atomic
    request_path = Path(request_path); output = request_path.parent; request = read(request_path)
    unsigned = {k:v for k,v in request.items() if k != 'request_fingerprint'}
    if fingerprint(unsigned) != request['request_fingerprint']: raise ValueError('Request was edited')
    study = ROOT/request['study']; settings, manifest = verify_study(study)
    if request['protocol_fingerprint'] != settings['protocol_fingerprint'] or request['schema_fingerprint'] != settings['schema_fingerprint']:
        raise ValueError('Request protocol/schema changed')
    x, y, roles = training_data(study)
    started = time.monotonic(); family = request['family']; branches = None
    if family in HYBRIDS:
        checkpoint, details, branches, outputs = _hybrid(request,study,y,roles,output)
    else:
        checkpoint, details = _fit(request,x,y,roles,manifest,output)
        outputs = {}
    record = dict(request, checkpoint=relative(checkpoint), run_directory=relative(output),
        n_classes=len(manifest['classes']))
    if branches: record['branches'] = branches
    for number, role in enumerate(ROLES[1:], 1):
        if role not in outputs: outputs[role] = predict(record, x[roles == number])
        calibration.validate_output(outputs[role], int((roles == number).sum()), len(manifest['classes']))
        np.savez_compressed(output/f'{role}_outputs.npz', **outputs[role])
    correction = calibration.fit(outputs['calibration'], y[roles == 3], outputs['tune'], y[roles == 1])
    atomic(output/'calibration.json', correction)
    score = calibration.measure(outputs['tune'], y[roles == 1], manifest, correction)
    sample = x[roles == 1][:128]
    first = predict(record,sample); second = predict(record,sample)
    for key in first: np.testing.assert_array_equal(first[key],second[key])
    # Separately measured deployment-relevant single-window latency including reload.
    timings = []
    for _ in range(3):
        tick = time.perf_counter(); predict(record,sample[:1]); timings.append(time.perf_counter()-tick)
    import psutil
    memory = psutil.Process().memory_info()
    status = 'pruned' if (output/'PRUNE').exists() else 'completed'
    record.update(status=status, selection_metrics=score, objective=float(score['macro_f1']),
        selection_objective='tuning macro F1 over all 84 declared classes; checkpoint early stop uses tuning accuracy',
        details=details, calibration=correction, elapsed_seconds=time.monotonic()-started,
        cpu_checkpoint_reload_verified=True, cpu_replay_rows=len(sample),
        cold_single_window_seconds=dict(median=float(np.median(timings)), maximum=max(timings), includes_model_load=True),
        peak_working_set_bytes=getattr(memory,'peak_wset',memory.rss),
        test_loaded=False, assessment_used_for_fitting_or_selection=False,
        artifact_hashes={relative(p):digest(p) for p in output.iterdir()
            if p.suffix in ('.pt','.joblib','.ubj','.npz','.json') and p.name not in ('result.json','progress.json','failure.json')})
    atomic(output/'result.json',record)
    print(f'{status.upper()} {family} tuning_macro_f1={record["objective"]:.5f}',flush=True)


def main(request_path):
    try: run(request_path)
    except BaseException as error:
        atomic(Path(request_path).parent/'failure.json', dict(status='failed', error=type(error).__name__,
            message=str(error), traceback=traceback.format_exc(), time=time.time()))
        raise
