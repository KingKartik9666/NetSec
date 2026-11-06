"""Single-writer, round-robin Optuna queue with bounded recoverable attempts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import time
import optuna
import psutil

from .protocol import (ROOT, BRANCHES, HYBRIDS, atomic, read, relative, digest, fingerprint,
                       exclusive, verify_study)
from .processes import Job, launch, terminate
from . import search


class Halt(Exception):
    def __init__(self,status): self.status=status


def make_request(settings,study,family,phase,config,seed=42,**extra):
    result=dict(study=relative(study),family=family,phase=phase,seed=seed,config=config,
        schema_fingerprint=settings['schema_fingerprint'],protocol_fingerprint=settings['protocol_fingerprint'],**extra)
    result['request_fingerprint']=fingerprint(result)
    return result


def pulse(study,state,**changes):
    state.update(changes,heartbeat=time.time(),controller_pid=os.getpid(),
                 controller_created=psutil.Process().create_time())
    atomic(study/'state.json',state)


def check_limits(study,state):
    if (study/'STOP').exists(): raise Halt('stopped')
    if time.time() >= state['deadline']: raise Halt('budget_exhausted')
    if shutil.disk_usage(study).free < 5*1024**3: raise Halt('insufficient_disk_space')


def progress(run):
    try:
        if (run/'progress.json').exists(): return read(run/'progress.json')
        if (run/'epochs.jsonl').exists():
            lines=(run/'epochs.jsonl').read_text(encoding='utf-8').splitlines()
            if lines: return json.loads(lines[-1])
    except (OSError,ValueError): pass  # A trainer may be between writes.
    return {}


def run_job(study,state,request,base,trial=None):
    from .worker import verify_result
    base.mkdir(parents=True,exist_ok=True)
    failures=0;attempts=sorted(base.glob('attempt*'))
    for previous in attempts:
        if (previous/'result.json').exists():
            result=verify_result(previous/'result.json',None if request['phase']=='assessment' else request)
            return result,previous/'result.json'
        if (previous/'failure.json').exists(): failures+=1
        elif not (previous/'interrupted.json').exists():
            # The old controller's Windows job is dead before a new one starts.
            atomic(previous/'failure.json',dict(status='interrupted',reason='controller or machine exited before an atomic result'))
            failures+=1
    while failures < 2:
        check_limits(study,state);verify_study(study)
        number=len(list(base.glob('attempt*')))+1
        run=base/f'attempt{number:02d}';run.mkdir()
        atomic(run/'request.json',request)
        if request['phase']=='assessment':
            command=[sys.executable,str(ROOT/'experiment/train_live_models.py'),'_assess','--study',str(study),'--output',str(run)]
        else:
            command=[sys.executable,str(ROOT/'experiment/train_live_models.py'),'_worker','--request',str(run/'request.json')]
        with Job() as worker_job, (run/'worker.log').open('a',encoding='utf-8') as log:
            command+=['--job-name',worker_job.name]
            process=launch(command,ROOT,log)
            worker_job.assign(process)
            started=time.time();last_reported=-1
            timeout=min(state['deadline'],started+request['config'].get('runtime_seconds',1800)+600)
            pulse(study,state,current=dict(family=request['family'],phase=request['phase'],seed=request['seed'],
                trial=request.get('trial_number'),attempt=number,run_directory=relative(run),worker_pid=process.pid,
                worker_created=psutil.Process(process.pid).create_time(),started_at=started))
            try:
                while process.poll() is None:
                    check_limits(study,state)
                    if time.time() >= timeout:
                        terminate(process)
                        atomic(run/'failure.json',dict(status='failed',reason='bounded worker timeout'))
                        break
                    telemetry=progress(run)
                    if trial is not None and request['family'] in ('xgboost','kan','hem') and telemetry:
                        step=int(telemetry.get('round',telemetry.get('epoch',0)))
                        value=telemetry.get('validation_macro_f1')
                        if value is not None and step > last_reported:
                            trial.report(float(value),step);last_reported=step
                            if trial.should_prune(): (run/'PRUNE').touch()
                    pulse(study,state,worker_progress=telemetry)
                    time.sleep(2 if read(study/'study.json')['smoke'] else 5)
            except Halt as stop:
                # A stopped attempt remains inspectable; resume creates a new attempt.
                terminate(process);atomic(run/'interrupted.json',dict(status=stop.status,time=time.time()))
                raise
            finally:
                terminate(process)
        if (run/'result.json').exists():
            result=verify_result(run/'result.json',None if request['phase']=='assessment' else request)
            return result,run/'result.json'
        if not (run/'failure.json').exists():
            atomic(run/'failure.json',dict(status='failed',exit_code=process.returncode,reason='no evaluated atomic result'))
        failures+=1
        pulse(study,state,last_failure=dict(family=request['family'],run=relative(run),detail=read(run/'failure.json')))
        if failures < 2:
            for _ in range(2 if read(study/'study.json')['smoke'] else 10):
                check_limits(study,state);pulse(study,state);time.sleep(2)
    return None,None


def trial_counts(study,families):
    result={}
    for name in families:
        trials=search.load(study,name).trials
        result[name]={status:sum(t.state.name.lower()==status for t in trials)
            for status in ('complete','pruned','fail','running')}
    return result


def optimize(study,state,settings,families,target):
    while True:
        check_limits(study,state)
        counts=trial_counts(study,families)
        pending=[name for name in families if sum(counts[name][s] for s in ('complete','pruned','fail')) < target]
        pulse(study,state,trial_counts={**state.get('trial_counts',{}),**counts})
        if not pending: return
        family=min(pending,key=lambda name:(0 if counts[name]['running'] else 1,
            sum(counts[name][s] for s in ('complete','pruned','fail')),list(families).index(name)))
        storage,trial,request=search.proposal(study,family)
        state['trial_counts'][family]['running']=1
        result,path=run_job(study,state,request,study/f'runs/{family}/trial{trial.number:04d}',trial)
        if result:
            trial.set_user_attr('result',relative(path));trial.set_user_attr('result_sha256',digest(path))
            trial.set_user_attr('calibration',result['calibration']);trial.set_user_attr('cpu_reload',True)
            storage.tell(trial,result['objective'] if result['status']=='completed' else None,
                state=optuna.trial.TrialState.COMPLETE if result['status']=='completed' else optuna.trial.TrialState.PRUNED)
            previous=state.setdefault('best_tuning',{}).get(family)
            if result['status']=='completed' and (previous is None or result['objective']>previous['macro_f1']):
                score=result['selection_metrics']
                state['best_tuning'][family]=dict(trial=trial.number,macro_f1=result['objective'],accuracy=score['accuracy'],
                    alert_false_positive_rate=score['screening']['false_positive_rate'],
                    alert_attack_recall=score['screening']['attack_recall'],
                    calibrated_correctness_ece=score['calibrated_correctness']['ece_10_equal_width_bins'],
                    cold_single_window_seconds=result['cold_single_window_seconds']['median'],result=relative(path))
        else:
            trial.set_user_attr('failure','two failed attempts; see preserved worker logs')
            storage.tell(trial,state=optuna.trial.TrialState.FAIL)


def freeze(study,families):
    from .worker import verify_result
    selected={}
    for family in families:
        storage=search.load(study,family)
        candidates=[t for t in storage.trials if t.state==optuna.trial.TrialState.COMPLETE]
        if not candidates: raise RuntimeError(f'No usable completed trial for {family}; no substitute permitted')
        winner=max(candidates,key=lambda t:(t.value,-t.number))
        path=ROOT/winner.user_attrs['result']
        if digest(path)!=winner.user_attrs['result_sha256']:raise ValueError('Completed result changed')
        record=verify_result(path)
        selected[family]=dict(result=relative(path),result_sha256=digest(path),trial=winner.number,
            objective=winner.value,config=record['config'],seed=record['seed'])
    return selected


def execute(study,ready=None):
    study=Path(study).resolve()
    if ready:
        deadline=time.time()+20
        while not Path(ready).exists():
            if time.time()>deadline:raise RuntimeError('Supervisor did not assign the owned process job')
            time.sleep(.1)
    with exclusive(study/'controller.lock'):
        settings,_=verify_study(study)
        state=read(study/'state.json') if (study/'state.json').exists() else dict(started_at=time.time(),deadline=time.time()+settings['hours']*3600)
        if state.get('status') in ('completed','budget_exhausted','needs_attention','insufficient_disk_space'):return
        try:
            pulse(study,state,status='running',phase='branch_search',current=None)
            optimize(study,state,settings,BRANCHES,settings['trials_per_branch'])
            selected=read(study/'selected.json') if (study/'selected.json').exists() else dict(branches=freeze(study,BRANCHES))
            atomic(study/'selected.json',selected)
            pulse(study,state,phase='hybrid_search')
            optimize(study,state,settings,HYBRIDS,settings['trials_per_hybrid'])
            if 'hybrids' not in selected:
                selected['hybrids']=freeze(study,HYBRIDS);atomic(study/'selected.json',selected)
            pulse(study,state,phase='seed_repeats')
            for seed in settings['repeat_seeds']:
                for family,entry in {**selected['branches'],**selected['hybrids']}.items():
                    base=study/f'repeats/{family}/seed{seed}'
                    if (base/'result-link.json').exists():
                        from .worker import verify_result
                        verify_result(base/'result-link.json');continue
                    config=entry['config']
                    if family in HYBRIDS:
                        config=dict(frozen_combiner=read(ROOT/entry['result'])['checkpoint'],runtime_seconds=300)
                    request=make_request(settings,study,family,'repeat',config,seed)
                    result,path=run_job(study,state,request,base)
                    if not result:raise RuntimeError(f'Repeated fit failed: {family} seed{seed}')
                    atomic(base/'result-link.json',result)
                    state.setdefault('completed_repeats',[]).append(f'{family}:{seed}')
                    pulse(study,state)
            pulse(study,state,phase='final_assessment')
            request=make_request(settings,study,'all_models','assessment',dict(runtime_seconds=1800))
            result,path=run_job(study,state,request,study/'assessment')
            if not result:raise RuntimeError('Final assessment failed; checkpoints preserved')
            atomic(study/'completion.json',dict(status='completed',finished_at=time.time(),
                branch_trials=trial_counts(study,BRANCHES),hybrid_trials=trial_counts(study,HYBRIDS),
                outputs=10,auxiliary_branches=1,repeat_seeds=settings['repeat_seeds'],assessment_result=relative(path),
                assessment_sha256=digest(path),test_loaded=False,live_accuracy_verified=False,native_integration='pending'))
            pulse(study,state,status='completed',phase='complete',current=None,worker_progress=None)
        except Halt as stop:
            pulse(study,state,status=stop.status,current=None)
        except BaseException as error:
            import traceback
            pulse(study,state,status='needs_attention',error=str(error),traceback=traceback.format_exc(),current=None)
            raise
