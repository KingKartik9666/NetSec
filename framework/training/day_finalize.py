"""Post-freeze assessment, calibration, OOF ensembles and CPU verification."""
from __future__ import annotations

from copy import deepcopy
import gc
from pathlib import Path
import shutil
import time

import numpy as np

from experiment.datasense_quality import digest
from framework.training.hybrid_protocol import ROOT, DATA, relative, verify_hashes, branch_reference
from framework.training.study_supervisor import atomic_write, read
from framework.training.tracking import classification_metrics


def select_record(baseline, candidates, policy):
    reference = baseline['selection_metrics']
    def eligible(row):
        m = row['selection_metrics']
        return (m['false_positive_rate'] <= reference['false_positive_rate']+policy['max_fpr_increase']
            and m['attack_missed_as_benign_rate'] <= reference['attack_missed_as_benign_rate']+policy['max_attack_miss_increase']
            and m['macro_recall'] >= reference['macro_recall']-policy['max_macro_recall_drop'])
    options = [baseline]+[r for r in candidates if eligible(r)]
    return max(options,key=lambda r:(r['selection_metrics']['accuracy'],r['selection_metrics']['macro_f1']))


def freeze_selection(study,protocol):
    target = study/'selected.json'
    if target.exists():
        return read(target)
    selected = {}
    for family,baseline in protocol['baselines'].items():
        candidates = []
        for path in (study/'runs').glob(f'{family}_trial*/attempt_*/result.json'):
            row = read(path)
            if row.get('status')=='completed':
                verify_hashes(row['artifact_hashes'])
                row['run_directory'] = relative(path.parent)
                row['source_result'] = relative(path)
                candidates.append(row)
        selected[family] = select_record(baseline,candidates,protocol['regression_guards'])
    atomic_write(target,selected)
    return selected


def calibrate_hem(destination,parts,prediction,y,roles,checkpoint_hash):
    """Per-tag confidence of unchanged hard votes; no class-probability claim."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from experiment.calibrate_datasense_models import confidence_metrics
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    multi,binary,knn=[parts[k] for k in ('multiclass_probabilities','binary_probabilities','knn_predictions')]
    voted=multi[np.arange(len(y)),prediction]
    matching=np.where(prediction==0,1-binary,binary)
    features=np.column_stack([np.log(np.clip(voted,1e-12,1)),
        np.log(np.clip(matching,1e-12,1)),(knn==multi.argmax(1)).astype(float)])
    correct=prediction==y;fit=np.flatnonzero(roles==1);assess=np.flatnonzero(roles==2)
    model=LogisticRegression(C=1.,random_state=42,max_iter=1000).fit(features[fit],correct[fit])
    confidence=model.predict_proba(features)[:,list(model.classes_).index(True)]
    path=destination/'hem_correctness_calibrator.joblib';joblib.dump(model,path)
    replay=joblib.load(path).predict_proba(features)[:,list(model.classes_).index(True)]
    np.testing.assert_allclose(confidence,replay,atol=0,rtol=0)
    result=dict(source_checkpoint_sha256=checkpoint_hash,calibration_rows=len(fit),assessment_rows=len(assess),
        correctness_before=confidence_metrics(correct[assess],.5*(voted+matching)[assess]),
        correctness_after=confidence_metrics(correct[assess],confidence[assess]),
        calibrator_reload_verified=True,predictions_unchanged=True,
        assessment_used_for_fitting_or_selection=False,test_loaded=False,
        meaning='Correctness confidence of unchanged HEM vote; not multiclass ensemble probabilities')
    atomic_write(destination/'calibration.json',result)
    return result


def evaluate(study,family,record,tag='selected'):
    """No metrics from this function feed back into model selection."""
    from framework.training.day_worker import predict_saved
    from framework.training.tabm_calibration import fit_calibration,apply_calibration
    manifest = read(DATA/'model/manifest.json')
    roles = np.load(study/'validation_roles.npy')
    destination = study/'assessment'/tag/family
    destination.mkdir(parents=True,exist_ok=True)
    if (destination/'result.json').exists() and read(destination/'result.json').get('status')=='completed':
        result = read(destination/'result.json')
        verify_hashes(result['artifact_hashes'])
        return result
    verify_hashes(record['artifact_hashes'])
    X = np.load(DATA/'model/X_validation.npy',mmap_mode='r')
    y = np.load(DATA/'model/y_validation.npy')
    started = time.monotonic()
    values = predict_saved(record,X)
    elapsed = time.monotonic()-started
    prediction = values[0] if family=='hem' else values.argmax(1)
    np.save(destination/'validation_predictions.npy',prediction)
    metrics = classification_metrics(y,prediction,manifest['classes'],0)
    result = dict(family=family,seed=record['seed'],checkpoint=record['checkpoint'],
        config=record['config'],arm=record['arm'],full_validation_metrics=metrics,
        assessment_metrics=classification_metrics(y[roles==2],prediction[roles==2],manifest['classes'],0),
        cpu_validation_seconds=elapsed,validation_rows=len(y),test_loaded=False,
        assessment_used_for_fitting_or_selection=False)
    if family=='hem':
        # Confidence describes correctness of the original HEM vote, not class probabilities.
        np.savez(destination/'validation_components.npz',**values[1])
        result['calibration']=calibrate_hem(destination,values[1],prediction,y,roles,digest(ROOT/record['checkpoint']))
    else:
        np.save(destination/'validation_probabilities.npy',values)
        calibration,calibrated,attack = fit_calibration(values,y,roles,manifest)
        atomic_write(destination/'calibration.json',calibration)
        replay,replay_attack = apply_calibration(values,read(destination/'calibration.json'))
        np.testing.assert_allclose(calibrated,replay,atol=0,rtol=0)
        np.testing.assert_allclose(attack,replay_attack,atol=0,rtol=0)
        result['calibration'] = calibration
    # Independent raw-schema CPU reloads, and measured reload+inference latency.
    sample = X[:128]
    times=[]; previous=None
    for _ in range(2):
        start=time.monotonic(); replay=predict_saved(record,sample);times.append(time.monotonic()-start)
        actual = replay[0] if family=='hem' else replay
        expected = prediction[:128] if family=='hem' else values[:128]
        np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=1e-5)
        if previous is not None:
            np.testing.assert_allclose(actual,previous,atol=0,rtol=0)
        previous=actual
    import psutil
    memory=psutil.Process().memory_info()
    ids=np.load(DATA/'model/row_ids_validation.npy')
    for code,name in ((1,'calibration'),(2,'assessment')):
        np.save(destination/f'{name}_row_ids.npy',ids[roles==code])
    result.update(status='completed',cpu_reload_verified=True,cpu_replay_rows=128,
        reload_and_predict_128_seconds=times,
        peak_working_set_bytes=getattr(memory,'peak_wset',memory.rss),
        artifact_hashes={relative(p):digest(p) for p in destination.iterdir()
            if p.name!='result.json' and p.suffix in ('.json','.npy','.npz','.joblib')})
    atomic_write(destination/'result.json',result)
    return result


def build_hybrid(study,name,names,selected,protocol):
    import optuna
    from framework.training.hybrid_blend import fit_blend,blend,quick_metrics,log_loss
    from framework.training import hybrid_evaluation as publisher
    from framework.training import hybrid_blend
    original_minimize=hybrid_blend.minimize
    def minimize(fun,initial,**kwargs):
        if kwargs.get('method')=='L-BFGS-B':
            kwargs['options']=dict(kwargs.get('options',{}),maxiter=1000)
        return original_minimize(fun,initial,**kwargs)
    hybrid_blend.minimize=minimize
    destination=study/'hybrids'/name
    if (destination/'result.json').exists():
        value=read(destination/'result.json');verify_hashes(value['artifact_hashes']);return value
    fold_ids=[np.load(study/f'folds/{i}/oof.npy') for i in (0,1)]
    y_train=np.load(DATA/'model/y_train.npy')[np.concatenate(fold_ids)]
    roles=np.load(study/'validation_roles.npy');tune=roles==0
    y=np.load(DATA/'model/y_validation.npy')
    jobs={};oof=[];validation=[];branches={}
    for family in names:
        for i in (0,1):
            jobs[f'{family}_fold{i}']=read(study/'oof_registry'/f'{family}_fold{i}.json')
            verify_hashes(jobs[f'{family}_fold{i}']['artifact_hashes'])
        oof.append(np.concatenate([np.load(ROOT/jobs[f'{family}_fold{i}']['run_directory']/'oof_probabilities.npy') for i in (0,1)]))
        validation.append(np.load(study/'assessment/selected'/family/'validation_probabilities.npy'))
        checkpoint=ROOT/selected[family]['checkpoint']
        files=[checkpoint,DATA/'model/manifest.json']
        if family=='kan':
            files.append(checkpoint.parent/'preprocessing.joblib')
        validation_path=study/'assessment/selected'/family/'validation_probabilities.npy'
        branch=dict(family=family,checkpoint=relative(checkpoint),
            schema_manifest=relative(DATA/'model/manifest.json'),
            assets={relative(p):digest(p) for p in files},
            validation_probabilities=relative(validation_path),validation_sha256=digest(validation_path))
        branches[family]=branch
    if 'tabpack' in names:
        # Publisher expects a full-job reference as well as a final branch.
        jobs['tabpack_full']=dict(run_directory=relative((ROOT/selected['tabpack']['checkpoint']).parent))
    a,b=np.stack(oof),np.stack(validation)
    search=destination/'combiner_search';search.mkdir(parents=True,exist_ok=True)
    candidates=[]
    # Equal and learned-global controls always remain candidates.
    global_cfg=fit_blend(a,y_train,names,None)
    equal=deepcopy(global_cfg);equal['weights']=[1/len(names)]*len(names)
    for kind,cfg in (('equal',equal),('global',global_cfg)):
        p=blend(b[:,tune],cfg,names);score=quick_metrics(y[tune],p);score['log_loss']=log_loss(p,y[tune])
        candidates.append(dict(kind=kind,combiner=cfg,tuning_metrics=score))
    storage='sqlite:///'+(study/'combiner.sqlite3').as_posix()
    tpe=optuna.create_study(study_name=name,storage=storage,load_if_exists=True,direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=4))
    for frozen in tpe.trials:
        if frozen.state==optuna.trial.TrialState.RUNNING:
            tpe.tell(frozen.number,state=optuna.trial.TrialState.FAIL)
    while len(tpe.trials)<protocol['combiner_trials'] and time.time()<protocol['deadline']-300:
        if (study/'STOP').exists():
            raise InterruptedError('User stop')
        trial=tpe.ask();penalty=trial.suggest_float('classwise_penalty',1e-4,1.,log=True)
        try:
            cfg=fit_blend(a,y_train,names,penalty)
            p=blend(b[:,tune],cfg,names);score=quick_metrics(y[tune],p);score['log_loss']=log_loss(p,y[tune])
            candidate=dict(combiner=cfg,tuning_metrics=score,penalty=penalty)
            atomic_write(search/f'trial_{trial.number:03d}.json',candidate)
            tpe.tell(trial,score['accuracy']+1e-6*score['macro_f1'])
        except (RuntimeError,FloatingPointError) as exc:
            atomic_write(search/f'failure_{trial.number:03d}.json',dict(error=str(exc)))
            tpe.tell(trial,state=optuna.trial.TrialState.FAIL)
    candidates.extend(read(p) for p in search.glob('trial_*.json'))
    chosen=max(candidates,key=lambda c:(c['tuning_metrics']['accuracy'],c['tuning_metrics']['macro_f1'],-c['tuning_metrics']['log_loss']))
    atomic_write(destination/'combiner_selection.json',dict(chosen=chosen,candidates=candidates,
        fit_partition='training-only forward OOF',selection_partition='frozen tuning',assessment_used=False))
    # Reuse the established publisher with the already selected combiner. The
    # numerical code and saved-bundle CPU verification are unchanged.
    publisher.STUDY=study
    publisher.write=atomic_write
    original=publisher.fit_blend
    original_reference=publisher.branch_reference
    publisher.fit_blend=lambda *_args,**_kwargs:deepcopy(chosen['combiner'])
    publisher.branch_reference=lambda family,_run:branches[family]
    try:
        publisher.finalize(name,names,dict(protocol,final_branches=branches,combiner_penalties=[None]),jobs,destination)
    finally:
        publisher.fit_blend=original;publisher.branch_reference=original_reference
    return read(destination/'result.json')


def report(study,protocol):
    selected=read(study/'selected.json') if (study/'selected.json').exists() else {}
    rows={}
    for family in ('random_forest','xgboost','kan','hem','tabm_ne','tabm_hier','tabm_hier_ne'):
        target=study/'assessment/selected'/family/'result.json'
        baseline=protocol['baselines'][family]
        row=dict(baseline_accuracy=baseline.get('metrics',{}).get('accuracy'),
            selected_checkpoint=selected.get(family,baseline)['checkpoint'],status='pending')
        if target.exists() and read(target).get('status')=='completed':
            value=read(target)
            row.update(status='completed',accuracy=value['full_validation_metrics']['accuracy'],
                macro_f1=value['full_validation_metrics']['macro_f1'],
                assessment_accuracy=value['assessment_metrics']['accuracy'],verification=relative(target))
        rows[family]=row
    for name in protocol['combinations']:
        target=study/'hybrids'/name/'result.json'
        row=dict(status='pending',baseline_accuracy=protocol.get('hybrid_baselines',{}).get(name,{}).get('accuracy'))
        if target.exists():
            value=read(target)
            row.update(status=value['status'],accuracy=value['full_validation_metrics']['accuracy'],
                macro_f1=value['full_validation_metrics']['macro_f1'],
                assessment_accuracy=value['calibration']['before']['accuracy'],verification=relative(target))
        rows[name]=row
    failures=[]
    for pattern in ('runs/*/attempt_*/failure.json','runs/*/attempt_*/interrupted.json',
                    'runs/*/attempt_*/pruned.json','finalization/*/failure.json'):
        for path in study.glob(pattern):
            failures.append(dict(path=relative(path),log=relative(path.parent/('job.log' if 'finalization' in path.parts else 'fit.log')),
                record_type=path.stem,**read(path)))
    for family,row in rows.items():
        job=family if family in protocol['combinations'] else f'evaluate_{family}'
        marker=study/'finalization'/job/'failure.json'
        if row['status']!='completed' and marker.exists():
            row.update(status='failed',failure_log=relative(marker.parent/'job.log'),failure=read(marker))
    seeds={}
    for path in (study/'assessment').glob('seed*/*/result.json'):
        record=read(path)
        if record.get('status')=='completed':
            seeds.setdefault(record['family'],[]).append(dict(seed=record['seed'],
                accuracy=record['full_validation_metrics']['accuracy'],macro_f1=record['full_validation_metrics']['macro_f1']))
    for family,values in seeds.items():
        baseline=rows.get(family,{})
        all_values=[r['accuracy'] for r in values]+([baseline['accuracy']] if 'accuracy' in baseline else [])
        baseline['seed_repeats']=values
        baseline['accuracy_seed_std']=float(np.std(all_values))
        baseline['accuracy_seed_range']=[float(min(all_values)),float(max(all_values))]
    atomic_write(study/'report.json',dict(models=rows,failures=failures,deadline=protocol['deadline'],
        test_loaded=False,updated_at=time.time(),caveat=protocol['caveat']))
    lines=['# All-model day study','',protocol['caveat'],'',
        '| Model | Status | Baseline accuracy | Selected accuracy | Macro F1 | Assessment accuracy |',
        '|---|---|---:|---:|---:|---:|']
    def percent(value):
        return '-' if value is None else f'{100*value:.4f}%'
    for family,row in rows.items():
        lines.append('| '+family+' | '+row['status']+' | '+' | '.join(percent(row.get(k)) for k in
            ('baseline_accuracy','accuracy','macro_f1','assessment_accuracy'))+' |')
    lines+=['','See report.json, selected.json, assessment/, hybrids/, runs/ and verification/ for detailed evidence.',
        'Pending/failed stages are not completed research results. The held-out test remains locked.']
    (study/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return rows
