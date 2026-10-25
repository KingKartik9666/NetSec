"""Finish the frozen study: reuse audited folds, fit missing folds, export/report."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
import time

from experiment import optimize_all_models_day as runner
from experiment.datasense_quality import digest
from experiment.optimize_models import StudyLock,protect_descendants
from framework.training import day_protocol
from framework.training.hybrid_protocol import ROOT,DATA,relative,verify_hashes
from framework.training.study_supervisor import atomic_write,read

SOURCE=ROOT/'artifacts/training/all_models_resume_20261021_v1'
STUDY=ROOT/'artifacts/training/all_models_wrapup_20261023_v1'
MODULE='experiment.finish_all_models'


def initialize():
    if (STUDY/'study.json').exists():
        protocol=read(STUDY/'study.json');day_protocol.verify(protocol);return protocol
    old=read(SOURCE/'study.json');day_protocol.verify(old)
    STUDY.mkdir(parents=True,exist_ok=True)
    for name in ('validation_roles.npy','selected.json','baselines.json','previous_recovery.json'):
        shutil.copy2(SOURCE/name,STUDY/name)
    shutil.copytree(SOURCE/'folds',STUDY/'folds',dirs_exist_ok=True)
    for path in (SOURCE/'oof_registry').glob('*.json'):
        row=read(path);verify_hashes(row['artifact_hashes'])
        atomic_write(STUDY/'oof_registry'/path.name,row)
    # Import successful repeat records as references; large artifacts stay put.
    for path in SOURCE.glob('runs/*_repeat_s*/attempt_*/result.json'):
        row=read(path)
        if row.get('status')=='completed':
            verify_hashes(row['artifact_hashes'])
            atomic_write(STUDY/'runs'/row['job']/'attempt_000/result.json',row)
    protocol=deepcopy(old);now=time.time()
    protocol.update(started_at=now,deadline=now+18*3600,
        started_utc=datetime.fromtimestamp(now,timezone.utc).isoformat(),
        deadline_utc=datetime.fromtimestamp(now+18*3600,timezone.utc).isoformat(),
        per_fit_seconds=3600,max_attempts=3)
    protocol['completion_authorization']='User: finish whatevers left, wrap up everything; no new model search'
    protocol['wrapup']=dict(source_study=relative(SOURCE),source_protocol_sha256=digest(SOURCE/'study.json'),
        frozen_selection_sha256=digest(SOURCE/'selected.json'),numerical_tolerance_unchanged=True,
        numeric_adapter='CUDA PLE inference uses CPU-compatible separate multiply/add',
        searches_reopened=False,original_studies_preserved=True)
    for name in ('experiment/finish_all_models.py','framework/training/finish_models.py','framework/training/model_bundle.py'):
        protocol['source_hashes'][name]=digest(ROOT/name)
    for path in [STUDY/'validation_roles.npy',STUDY/'selected.json',*(STUDY/'folds').rglob('*.npy')]:
        protocol['role_hashes'][relative(path)]=digest(path)
    failed=SOURCE/'runs/tabm_ne_oof0/attempt_000'
    for path in (failed/'model.pt',failed/'request.json',failed/'failure.json'):
        protocol['role_hashes'][relative(path)]=digest(path)
    for name in protocol['source_hashes']:
        target=STUDY/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    atomic_write(STUDY/'study.json',protocol)
    return protocol


def adopt_available(selected):
    from framework.training.finish_models import adopt_oof
    for family in ('xgboost','tabpack'):
        for fold in (0,1):
            target=STUDY/'oof_registry'/f'{family}_fold{fold}.json'
            if target.exists():
                verify_hashes(read(target)['artifact_hashes']);continue
            version='hybrids_optuna_20261018_v1' if family=='xgboost' else 'hybrids_verified_v2_20261018_v2'
            source_study=ROOT/'artifacts/training'/version
            source=source_study/'refits'/f'{family}_fold{fold}.json'
            if not source.exists():
                source=next((source_study/'runs'/f'{family}_fold{fold}').glob('attempt_*/result.json'))
            request={k:deepcopy(selected[family][k]) for k in ('family','arm','seed','config')}
            adopt_oof(STUDY,source,request,fold,source_study)


def recover_fold(protocol,state,selected):
    target=STUDY/'oof_registry/tabm_ne_fold0.json'
    if target.exists():
        verify_hashes(read(target)['artifact_hashes']);return
    output=STUDY/'runs/tabm_ne_oof0_recovered/attempt_000'
    request={k:deepcopy(selected['tabm_ne'][k]) for k in ('family','arm','seed','config')}
    request.update(phase='oof',fold=0,job='tabm_ne_oof0_recovered',study=relative(STUDY))
    output.mkdir(parents=True,exist_ok=True);atomic_write(output/'request.json',request)
    child=runner.launch([MODULE,'--recover',str(output/'request.json')],output/'fit.log')
    runner.state_write(state,status='recovering_numerical_check',active_job=dict(job=request['job'],pid=child.pid))
    while child.poll() is None:
        if (STUDY/'STOP').exists() or time.time()>=protocol['deadline']:
            runner.terminate_tree(child.pid);raise InterruptedError('User stop or deadline')
        runner.state_write(state);time.sleep(5)
    if child.returncode!=0 or not (output/'result.json').exists():
        raise RuntimeError('Saved TabM recovery failed; inspect '+str(output/'fit.log'))
    row=read(output/'result.json');verify_hashes(row['artifact_hashes'])
    atomic_write(target,row)
    runner.state_write(state,active_job=None)


def export_models():
    import numpy as np
    from framework.training.model_bundle import FinalModelPredictor
    from framework.training.tabm_calibration import apply_calibration
    selected=read(STUDY/'selected.json');manifest=read(DATA/'model/manifest.json')
    X=np.load(DATA/'model/X_validation.npy',mmap_mode='r')[:128]
    exports={}
    for family in ('random_forest','xgboost','kan','hem','tabm_ne','tabm_hier','tabm_hier_ne'):
        record=selected[family];assessment=STUDY/'assessment/selected'/family
        value=read(assessment/'result.json')
        if value.get('status')!='completed':
            raise ValueError('Assessment incomplete: '+family)
        verify_hashes(value['artifact_hashes']);verify_hashes(record['artifact_hashes'])
        assets=dict(record['artifact_hashes']);assets[relative(assessment/'result.json')]=digest(assessment/'result.json')
        assets.update(value['artifact_hashes'])
        assets[relative(DATA/'model/manifest.json')]=digest(DATA/'model/manifest.json')
        bundle=dict(status='pending_verification',record={k:record[k] for k in ('family','checkpoint','config','seed','arm')},
            feature_names=manifest['feature_names'],classes=manifest['classes'],benign_index=0,
            calibration=value['calibration'],assets=assets,inference_device='cpu',test_loaded=False,
            research_scope='Candidate only; source-validation assessment, not live or held-out-test validation')
        if family=='hem':
            bundle['correctness_calibrator']=relative(assessment/'hem_correctness_calibrator.joblib')
        destination=STUDY/'exports'/family;atomic_write(destination/'bundle.json',bundle)
        predictor=FinalModelPredictor(destination/'bundle.json',_verification_only=True)
        start=time.monotonic();actual=predictor.predict(X,manifest['feature_names']);seconds=time.monotonic()-start
        expected=np.load(assessment/'validation_predictions.npy')[:128]
        np.testing.assert_array_equal(actual['predictions'],expected)
        if family!='hem':
            raw=np.load(assessment/'validation_probabilities.npy')[:128]
            p,_=apply_calibration(raw,value['calibration'])
            np.testing.assert_allclose(actual['probabilities'],p,atol=2e-6,rtol=1e-5)
        bundle['status']='completed';atomic_write(destination/'bundle.json',bundle)
        exports[family]=dict(bundle=relative(destination/'bundle.json'),bundle_sha256=digest(destination/'bundle.json'),
            cpu_bundle_replay_verified=True,rows=128,reload_and_predict_seconds=seconds,
            checkpoint_bytes=(ROOT/record['checkpoint']).stat().st_size)
        atomic_write(destination/'verification.json',exports[family])
        print('EXPORTED',family,seconds,flush=True)
    atomic_write(STUDY/'exports/manifest.json',exports)


def final_report(protocol):
    from framework.training.day_finalize import report
    rows=report(STUDY,protocol)
    exports=read(STUDY/'exports/manifest.json')
    missing=[f for f,r in rows.items() if r['status']!='completed']
    expected_oof=[f'{f}_fold{i}.json' for f in ('tabm_ne','xgboost','kan','tabpack') for i in (0,1)]
    for name in expected_oof:
        row=read(STUDY/'oof_registry'/name);verify_hashes(row['artifact_hashes'])
    repeat_results=list((STUDY/'assessment').glob('seed*/*/result.json'))
    if missing or len(exports)!=7 or len(repeat_results)!=14:
        raise RuntimeError(f'Wrap-up incomplete: models={missing}, exports={len(exports)}, repeats={len(repeat_results)}')
    for path in list((STUDY/'assessment').glob('*/*/result.json'))+list((STUDY/'hybrids').glob('*/result.json')):
        record=read(path)
        if record.get('status')!='completed':
            raise ValueError('Incomplete assessment/export: '+str(path))
        verify_hashes(record['artifact_hashes'])
    details={};lines=['# Final frozen-model comparison','',protocol['caveat'],'',
        '| Model | Original validation | Selected validation | Macro F1 | Assessment accuracy | FPR | Missed attack |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def pct(value):return '-' if value is None else f'{100*value:.4f}%'
    for name,row in rows.items():
        hybrid=name in protocol['combinations']
        source=STUDY/'hybrids'/name/'result.json' if hybrid else STUDY/'assessment/selected'/name/'result.json'
        value=read(source);metrics=value['full_validation_metrics']
        details[name]=dict(value,export=exports.get(name),comparison=row)
        lines.append('| '+name+' | '+' | '.join(pct(v) for v in (row.get('baseline_accuracy'),metrics['accuracy'],
            metrics['macro_f1'],row['assessment_accuracy'],metrics['false_positive_rate'],metrics['attack_missed_as_benign_rate']))+' |')
    lines+=['','All selections were frozen before these assessments. A selected candidate can regress on assessment or full validation; no assessment-driven retuning or app promotion occurred.',
        '', 'Completed: eight OOF branches, fourteen seed-repeat assessments, seven standalone CPU bundles, three calibrated hybrid CPU bundles.',
        'The numerical audit preserves the original TabM failure and passes the unchanged tolerance using CPU-compatible CUDA encoding arithmetic.',
        'XGBoost OOF artifacts reused only after matching all training parameters/rows/seed and verifying the larger runtime cap was nonbinding.',
        'See FINAL_METRICS.json for per-class reports, confusion matrices, calibration before/after, seed variation, latency, memory, recipes and checkpoint paths.',
        'The held-out test remains locked; no live-network performance is established.']
    (STUDY/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic_write(STUDY/'FINAL_METRICS.json',dict(models=details,protocol_sha256=digest(STUDY/'study.json'),
        test_loaded=False,completed_at=time.time(),repeats=[dict(path=relative(p),result=read(p)) for p in repeat_results]))
    atomic_write(STUDY/'completion.json',dict(status='completed',models=10,standalone_exports=7,hybrids=3,
        oof_fits=8,repeat_assessments=14,test_loaded=False,completed_at=time.time(),
        final_report=relative(STUDY/'FINAL_REPORT.md'),metrics_sha256=digest(STUDY/'FINAL_METRICS.json')))
    return rows


def controller():
    protocol=read(STUDY/'study.json');day_protocol.verify(protocol)
    with StudyLock(STUDY/'controller.lock'):
        handle=protect_descendants()
        state=dict(pid=__import__('os').getpid(),status='starting',active_job=None,test_loaded=False,deadline=protocol['deadline'])
        runner.state_write(state)
        selected=read(STUDY/'selected.json');adopt_available(selected);recover_fold(protocol,state,selected)
        for family in ('kan',):
            for fold in (0,1):
                target=STUDY/'oof_registry'/f'{family}_fold{fold}.json'
                if target.exists():
                    verify_hashes(read(target)['artifact_hashes']);continue
                request={k:deepcopy(selected[family][k]) for k in ('family','arm','seed','config')}
                request.update(phase='oof',fold=fold,job=f'{family}_oof{fold}')
                result=runner.run_worker(protocol,request,state,protocol['deadline'])
                if not result:
                    raise RuntimeError('Missing KAN fold: '+str(fold))
                atomic_write(target,result)
        # Established evaluation and publisher paths; completion is withheld until
        # standalone exports and all postchecks pass as well.
        for family in ('random_forest',*runner.FAMILIES):
            runner.final_job(protocol,state,f'evaluate_{family}','evaluate',family=family,record=selected[family])
        for name,names in protocol['combinations'].items():
            runner.final_job(protocol,state,name,'hybrid',name=name,names=names)
        for path in (STUDY/'runs').glob('*_repeat_s*/attempt_*/result.json'):
            row=read(path)
            runner.final_job(protocol,state,f'evaluate_{row["family"]}_s{row["seed"]}','evaluate',
                family=row['family'],record=row,tag=f'seed{row["seed"]}')
        runner.state_write(state,status='verifying_exports',active_job=None)
        child=runner.launch([MODULE,'--export'],STUDY/'exports.log')
        while child.poll() is None:
            if (STUDY/'STOP').exists() or time.time()>=protocol['deadline']:
                runner.terminate_tree(child.pid);raise InterruptedError('Stopped during export')
            runner.state_write(state,status='exporting_cpu_bundles');time.sleep(5)
        if child.returncode:
            raise RuntimeError('Standalone bundle verification failed; see exports.log')
        rows=final_report(protocol)
        runner.state_write(state,status='completed',completed_models=len(rows),active_job=None)


def configure():
    runner.STUDY=STUDY;runner.MODULE=MODULE;runner.initialize=initialize;runner.controller=controller


if __name__=='__main__':
    configure()
    if '--recover' in sys.argv:
        from framework.training.finish_models import recover_tabm
        path=Path(sys.argv[sys.argv.index('--recover')+1])
        recover_tabm(STUDY,SOURCE/'runs/tabm_ne_oof0/attempt_000',read(path),path.parent)
    elif '--export' in sys.argv:
        export_models()
    else:
        runner.main()
