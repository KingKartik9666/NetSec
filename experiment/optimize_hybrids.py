"""Bayesian TPE optimization and retraining of the three neural/tree hybrids."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from experiment import train_hybrids as runtime
from experiment.optimize_models import StudyLock,protect_descendants
from experiment.train_verified_clean import read,write
from experiment.datasense_quality import digest
from framework.training.hybrid_protocol import ROOT,DATA,relative,verify_hashes
from framework.training.hybrid_hpo_data import STUDY,PREVIOUS,initialize,verify,prior_result,load_oof,validate_adoption
from framework.training.hybrid_hpo import (FAMILIES,trial_study,ask_recipe,tell_result,
    affected_hybrids,score_hybrids)

MODULE='experiment.optimize_hybrids'


def configure():
    runtime.STUDY=STUDY;runtime.MODULE=MODULE
    from framework.training import hybrid_blend,hybrid_evaluation
    hybrid_evaluation.STUDY=STUDY
    original=hybrid_blend.minimize
    def minimize(fun,initial,**kwargs):
        if kwargs.get('method')=='L-BFGS-B':
            kwargs['options']=dict(kwargs.get('options',{}),maxiter=1000)
        return original(fun,initial,**kwargs)
    hybrid_blend.minimize=minimize


def latest(job):
    for path in reversed(sorted((STUDY/'runs'/job).glob('attempt_*/result.json'))):
        result=read(path)
        if result.get('status')=='completed':
            verify_hashes(result['artifact_hashes']);return path,result
    return None


def anchors(protocol,family):
    needed=set(sum(affected_hybrids(family).values(),[]))-{family}
    result={}
    for partner in sorted(needed):
        selected=STUDY/'selected'/f'{partner}.json'
        path=ROOT/read(selected)['search_result'] if selected.exists() else ROOT/protocol['baseline_partners'][partner]['fold0']
        result[partner]=dict(path=relative(path),sha256=digest(path))
    return result


def score_request(request,result):
    probabilities={request['family']:load_oof(result)}
    for family,reference in request['anchors'].items():
        verify_hashes({reference['path']:reference['sha256']})
        probabilities[family]=load_oof(read(ROOT/reference['path']))
    ids=np.load(STUDY/'folds/0/oof.npy')
    y=np.load(DATA/'model/y_train.npy')[ids]
    return score_hybrids(probabilities,y,np.load(STUDY/'meta_roles.npy'),affected_hybrids(request['family']))


def validate_request(request,protocol):
    if request['phase']=='search':
        study=trial_study(STUDY,request['family'])
        trial=next(t for t in study.trials if t.number==request['trial_number'])
        if trial.user_attrs.get('request')!=request:
            raise ValueError('Worker request differs from the durable TPE proposal')
    elif request['phase']=='refit':
        selected=read(STUDY/'selected'/f'{request["family"]}.json')['recipe']
        validate_adoption(request,selected)
        expected=f'{request["family"]}_'+('full' if request['fold'] is None else 'fold1')
        if request['job']!=expected or request['fold'] not in (None,1):
            raise ValueError('Invalid selected refit job')
    else:raise ValueError('Unknown worker phase')


def fit(path):
    from framework.training.hybrid_data import IndexedRows
    from framework.training.hybrid_bundle import branch_predict
    protocol=read(STUDY/'study.json');verify(protocol)
    request=read(path);validate_request(request,protocol);output=path.parent
    family=request['family'];started=time.monotonic()
    if request['phase']=='search' and request['baseline']:
        result=deepcopy(prior_result(protocol,family));validate_adoption(request,result)
        load_oof(result)
        result.update(job=request['job'],phase='search',trial_number=request['trial_number'],
            adopted_baseline=True,source_result=protocol['baseline_partners'][family]['fold0'])
    else:
        manifest=read(DATA/'model/manifest.json')
        X=np.load(DATA/'model/X_train.npy',mmap_mode='r');y=np.load(DATA/'model/y_train.npy')
        ids=np.load(DATA/'model/row_ids_train.npy')
        fold=0 if request['phase']=='search' else request['fold']
        if fold is None:
            fit_ids=np.arange(len(y))
            if request['arm']=='strict':
                strict=np.load(DATA/'model/strict_train/row_ids_train.npy');fit_ids=fit_ids[np.isin(ids,strict)]
            V=np.load(DATA/'model/X_validation.npy',mmap_mode='r');v=np.load(DATA/'model/y_validation.npy')
            tuning=np.load(STUDY/'validation_roles.npy')==0
            early_X,early_y=V[tuning],v[tuning]
            raw=V;prefix='validation';target_ids=np.load(DATA/'model/row_ids_validation.npy')
        else:
            folder=STUDY/f'folds/{fold}'
            fit_ids=np.load(folder/('fit_strict.npy' if request['arm']=='strict' else 'fit.npy'))
            early_ids=np.load(folder/'early_stop.npy');oof=np.load(folder/'oof.npy')
            early_X,early_y=X[early_ids],y[early_ids]
            raw=IndexedRows(X,oof) if family in ('tabm_ne','tabpack') else X[oof]
            prefix='oof';target_ids=ids[oof]
        print(f'TRAIN {request["job"]} rows={len(fit_ids)} early_stop={len(early_y)} phase={request["phase"]} CUDA',flush=True)
        if family in ('tabm_ne','tabpack'):
            from framework.training.hybrid_neural import train
            details=train(family,IndexedRows(X,fit_ids),y[fit_ids],early_X,early_y,
                manifest,request['config'],request['seed'],output,'cuda')
        else:
            from experiment.train_csv_models import train_candidate
            details=train_candidate(family,X[fit_ids],y[fit_ids],early_X,early_y,
                len(manifest['classes']),0,request['config'],request['seed'],output)
            details={k:value for k,value in details.items() if k not in ('predictions','probabilities','checkpoint_path','history')}
        checkpoint=output/('model.ubj' if family=='xgboost' else 'model.pt')
        assets=[checkpoint]+([output/'preprocessing.joblib'] if family=='kan' else [])
        branch=dict(family=family,checkpoint=relative(checkpoint),assets={relative(p):digest(p) for p in assets})
        p=branch_predict(branch,raw)
        if p.shape!=(len(target_ids),len(manifest['classes'])):raise ValueError('Prediction row/class shape mismatch')
        replay=branch_predict(branch,raw[:128])
        np.testing.assert_allclose(p[:len(replay)],replay,atol=2e-6,rtol=1e-5)
        np.testing.assert_array_equal(p[:len(replay)].argmax(1),replay.argmax(1))
        np.save(output/f'{prefix}_probabilities.npy',p);np.save(output/f'{prefix}_row_ids.npy',target_ids)
        result=dict(request,status='completed',run_directory=relative(output),details=details,
            training_rows=len(fit_ids),early_stopping_rows=len(early_y),
            cpu_checkpoint_reload_verified=True,test_loaded=False,
            artifact_hashes={relative(f):digest(f) for f in output.iterdir() if f.suffix in ('.pt','.ubj','.joblib','.npy')})
    if request['phase']=='search':result['score']=score_request(request,result)
    result['elapsed_seconds']=time.monotonic()-started
    # Large historical telemetry stays in the adopted source record/logs.
    if 'details' in result:result['details'].pop('history',None)
    write(output/'result.json',result)
    print('COMPLETE',request['job'],result.get('score',{}).get('mean_accuracy'),flush=True)


def run_job(request,state,protocol):
    result=latest(request['job'])
    if result:return result
    history=runtime.attempts(request['job'])
    if history and (history[-1]/'failure.json').exists():return None
    while not (STUDY/'STOP').exists():
        if not runtime.resources_available(request):
            state.update(status='waiting_for_training_or_memory',heartbeat_at=time.time());write(STUDY/'state.json',state)
            time.sleep(10);continue
        try:
            with ExitStack() as locks:
                for name in ('optimization_24h_v1','csv_specialization_v2','csv_research_v3'):
                    locks.enter_context(StudyLock(ROOT/'artifacts/training'/name/'controller.lock'))
                runtime.launch_fit(request,state,protocol)
            break
        except RuntimeError as exc:
            if 'holds the lock' not in str(exc):raise
            state.update(status='waiting_for_existing_gpu_controller',heartbeat_at=time.time());write(STUDY/'state.json',state);time.sleep(10)
    return latest(request['job'])


def stage_statistics():
    stats={}
    for family in FAMILIES:
        trials=trial_study(STUDY,family).trials
        stats[family]={name:sum(t.state.name==name for t in trials) for name in ('COMPLETE','FAIL','PRUNED','RUNNING','WAITING')}
    return stats


def select_family(family,protocol):
    import optuna
    candidates=[]
    for trial in trial_study(STUDY,family).trials:
        if trial.state==optuna.trial.TrialState.COMPLETE:
            request=trial.user_attrs['request'];found=latest(request['job'])
            if found:candidates.append((found[1]['score']['objective'],trial.number,request,found[0]))
    if not candidates:raise RuntimeError(f'No successful measured recipe for {family}')
    _,number,request,path=max(candidates,key=lambda row:(row[0],-row[1]))
    selection=dict(family=family,trial_number=number,recipe={k:request[k] for k in ('family','arm','seed','config')},
        search_result=relative(path),search_result_sha256=digest(path),score=read(path)['score'])
    target=STUDY/'selected'/f'{family}.json'
    if target.exists() and read(target)!=selection:raise ValueError('Selected recipe changed after family freeze')
    write(target,selection)
    return selection


def adopt_refit(family,suffix,selected,protocol):
    request=selected['recipe'];base=dict(family=family,**protocol['recipes'][family])
    if request!=base:return None
    if suffix=='full' and family!='tabpack':
        branch=protocol['previous_final_branches'][family]
        run=(ROOT/branch['checkpoint']).parent
        source=run/'result.json';name=relative(source)
        verify_hashes({name:protocol['partner_hashes'][name]})
        verify_hashes(branch['assets'])
        result=read(source)
        if result.get('status')!='completed' or not result.get('checkpoint_reload_verified') or result.get('test_loaded'):
            raise ValueError('Historical full model lacks verified completion')
        normalized={}
        for artifact,value in result['artifact_hashes'].items():
            path=(run/artifact).resolve();path.relative_to(run.resolve())
            normalized[relative(path)]=value
        result.update(artifact_hashes=normalized,run_directory=relative(run),
            cpu_checkpoint_reload_verified=True,source_result=name,
            source_result_sha256=protocol['partner_hashes'][name])
    else:result=prior_result(protocol,family,suffix)
    validate_adoption(request,result);verify_hashes(result['artifact_hashes'])
    return dict(result,adopted_baseline=True)


def controller(launch_id):
    with StudyLock(STUDY/'controller.lock'):
        job_handle=protect_descendants()
        protocol=read(STUDY/'study.json');verify(protocol)
        state=dict(status='starting',pid=os.getpid(),launch_id=launch_id,started_at=time.time(),
            active_job=None,planned_trials=4*protocol['trials_per_family'],test_loaded=False)
        write(STUDY/'state.json',state)
        for family in protocol['family_order']:
            while not (STUDY/'STOP').exists():
                study=trial_study(STUDY,family);trials=study.trials
                running=[t for t in trials if t.state.name=='RUNNING']
                if not running and len(trials)>=protocol['trials_per_family']:break
                if running and 'request' not in running[0].user_attrs:
                    tell_result(STUDY,family,running[0].number,state='fail');continue
                request=ask_recipe(STUDY,family,protocol['recipes'][family],anchors(protocol,family))
                if request is None:continue
                state.update(status='optimizing_branch',family=family,trial_number=request['trial_number'],
                    search_statistics=stage_statistics(),heartbeat_at=time.time());write(STUDY/'state.json',state)
                found=run_job(request,state,protocol)
                if (STUDY/'STOP').exists():break
                if found:tell_result(STUDY,family,request['trial_number'],found[1]['score']['objective'])
                else:tell_result(STUDY,family,request['trial_number'],state='fail')
                write(STUDY/'search_summary.json',stage_statistics())
            if (STUDY/'STOP').exists():
                state.update(status='stopped',active_job=None);write(STUDY/'state.json',state);return
            select_family(family,protocol)
        # Checkpoint reuse and refits are explicitly recorded separately from
        # search attempts. Fold-0 models never predict their own fitting rows.
        registry={}
        for family in FAMILIES:
            selected=read(STUDY/'selected'/f'{family}.json')
            first=read(ROOT/selected['search_result']);registry[f'{family}_fold0']=first
            write(STUDY/'refits'/f'{family}_fold0.json',first)
            for suffix,fold in (('fold1',1),('full',None)):
                target=STUDY/'refits'/f'{family}_{suffix}.json'
                if target.exists():record=read(target);verify_hashes(record['artifact_hashes'])
                else:
                    record=adopt_refit(family,suffix,selected,protocol)
                    if record is None:
                        request=dict(selected['recipe'],job=f'{family}_{suffix}',phase='refit',fold=fold)
                        state.update(status='refitting_selected',family=family);write(STUDY/'state.json',state)
                        found=run_job(request,state,protocol)
                        if (STUDY/'STOP').exists():
                            state.update(status='stopped',active_job=None);write(STUDY/'state.json',state);return
                        if not found:raise RuntimeError(f'Selected refit failed: {family}_{suffix}')
                        record=found[1]
                    write(target,record)
                registry[f'{family}_{suffix}']=record
        state.update(status='optimizing_combiners',active_job=None,search_statistics=stage_statistics());write(STUDY/'state.json',state)
        from framework.training.hybrid_hpo_finalize import finalize_all
        finalize_all(protocol,registry)
        failures=sum(row['FAIL'] for row in stage_statistics().values())
        state.update(status='completed_with_search_failures' if failures else 'completed',
            search_statistics=stage_statistics(),completed_hybrids=list(protocol['combinations']),
            completed_selected_models=len(registry),heartbeat_at=time.time())
        write(STUDY/'state.json',state)


def inspect_completion():
    import psutil
    state=read(STUDY/'state.json')
    summary=dict(controller_status=state['status'],controller_pid=state['pid'],
        completed_fits=state.get('completed_selected_models',0),planned_fits=12,
        completed_hybrids=len(state.get('completed_hybrids',[])),checked_at=time.time(),
        event='running',message='Bayesian search or selected retraining continues.')
    if state['status'] in ('completed','completed_with_search_failures'):
        protocol=read(STUDY/'study.json');verify(protocol)
        for family in protocol['family_order']:
            for suffix in ('fold0','fold1','full'):
                result=read(STUDY/'refits'/f'{family}_{suffix}.json')
                if result.get('status')!='completed' or not result.get('cpu_checkpoint_reload_verified') or result.get('test_loaded'):
                    raise ValueError('A selected model lacks the required completion checks')
                verify_hashes(result['artifact_hashes'])
        results={}
        for name in protocol['combinations']:
            result=read(STUDY/'hybrids'/name/'result.json');verify_hashes(result['artifact_hashes'])
            if not result.get('cpu_bundle_replay_verified') or result.get('test_loaded'):
                raise ValueError('Missing hybrid replay verification')
            calibration=result['calibration']
            if calibration.get('assessment_used_for_fitting_or_selection') or not calibration.get('predictions_unchanged'):
                raise ValueError('Calibration verification failed')
            results[name]=dict(validation_accuracy=result['full_validation_metrics']['accuracy'],
                validation_macro_f1=result['full_validation_metrics']['macro_f1'],
                assessment_accuracy=calibration['before']['accuracy'])
        failures=sum(r.get('FAIL',0) for r in state.get('search_statistics',{}).values())
        message=f'Optuna search finished with {failures} failed search trials; all three calibrated hybrids passed saved-bundle checks.'
        summary.update(event='completed',message=message,results=results,failed_search_trials=failures)
    elif state['status'] in ('controller_failed','stopped'):
        summary.update(event='attention',message='Optimization needs attention: '+state.get('error',state['status']))
    else:
        try:
            command=psutil.Process(state['pid']).cmdline()
            identity=(MODULE in command and '_run' in command and '--launch-id' in command
                and command[command.index('--launch-id')+1]==state.get('launch_id'))
        except (psutil.Error,IndexError):identity=False
        if not identity:
            summary.update(event='attention',message='Optimization controller exited before verified completion.')
    return summary


def watch():
    import msvcrt
    monitor_path=PREVIOUS/'completion_monitor/monitor.py'
    spec=importlib.util.spec_from_file_location('_hybrid_hpo_completion',monitor_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.HERE=STUDY/'completion_monitor';module.HERE.mkdir(exist_ok=True)
    with (module.HERE/'monitor.lock').open('a+b') as lock:
        lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        delivered=module.HERE/'last_notification.json'
        previous=module.event_key(read(delivered)) if delivered.exists() and read(delivered).get('notification_exit_code')==0 else None
        while True:
            try:
                summary=inspect_completion();summary['monitor_pid']=os.getpid()
                write(module.HERE/'status.json',summary);key=module.event_key(summary)
                if summary['event'] in ('completed','attention') and key!=previous:
                    if summary['event']=='completed':
                        lines=['# Hybrid optimization completed','',summary['message'],'',
                            '| Hybrid | Validation accuracy | Macro F1 | Assessment accuracy |',
                            '|---|---:|---:|---:|']
                        for name,row in summary['results'].items():
                            lines.append(f"| {name} | {100*row['validation_accuracy']:.2f}% | {100*row['validation_macro_f1']:.2f}% | {100*row['assessment_accuracy']:.2f}% |")
                        lines+=['','Exploratory source-validation results; held-out test remains locked.']
                        (module.HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
                    module.notify(summary);previous=key
                if summary['event']=='completed':return
            except Exception as exc:
                write(module.HERE/'monitor_error.json',dict(error=str(exc),at=time.time()))
            time.sleep(60)


def launch_watch():
    import psutil
    folder=STUDY/'completion_monitor';folder.mkdir(exist_ok=True);launch=folder/'launch.json'
    if launch.exists():
        pid=read(launch)['pid']
        try:
            if MODULE in psutil.Process(pid).cmdline() and '_watch' in psutil.Process(pid).cmdline():return
        except psutil.NoSuchProcess:pass
    flags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    with (folder/'watch.log').open('a',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,'-B','-u','-m',MODULE,'_watch'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=flags,close_fds=True)
    write(launch,dict(pid=child.pid,channel='Windows notification area'))


def main():
    configure()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',choices=('start','resume','initialize','status','stop','_run','_watch'))
    parser.add_argument('--device',choices=('cuda',));parser.add_argument('--launch-id');parser.add_argument('--fit',type=Path)
    args=parser.parse_args()
    if args.fit:
        try:fit(args.fit)
        except Exception as exc:
            write(args.fit.parent/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));raise
    elif args.command in ('initialize','start','resume'):
        if args.command=='start' and args.device!='cuda':parser.error('Use explicit --device cuda')
        if args.command=='resume' and not (STUDY/'study.json').exists():parser.error('No frozen study to resume')
        initialize()
        if args.command!='initialize':runtime.start();launch_watch()
        else:print(STUDY)
    elif args.command=='_run':
        try:controller(args.launch_id)
        except InterruptedError:
            state=read(STUDY/'state.json');state.update(status='stopped',active_job=None)
            write(STUDY/'state.json',state)
        except Exception as exc:
            write(STUDY/'controller_failure.json',dict(error=str(exc),traceback=traceback.format_exc()))
            state=read(STUDY/'state.json') if (STUDY/'state.json').exists() else {}
            state.update(status='controller_failed',error=str(exc));write(STUDY/'state.json',state);raise
    elif args.command=='_watch':watch()
    elif args.command=='stop':(STUDY/'STOP').touch()
    elif args.command=='status':
        print(json.dumps(read(STUDY/'state.json'),indent=2) if (STUDY/'state.json').exists() else 'Not started')
    else:parser.error('Choose a command')


if __name__=='__main__':main()
