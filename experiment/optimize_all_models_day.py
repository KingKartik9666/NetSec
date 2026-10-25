"""Bounded, detached all-model research supervisor. No chat loop is required."""
from __future__ import annotations

import argparse
from copy import deepcopy
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

from experiment.optimize_models import StudyLock, protect_descendants
from framework.training.day_protocol import STUDY, OLD_HYBRIDS, initialize, verify
from framework.training.hybrid_protocol import ROOT, relative
from framework.training.study_supervisor import (FAMILIES, atomic_write, read, phase_at,
    next_family, process_matches, terminate_tree)

MODULE='experiment.optimize_all_models_day'
ENV=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
    CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')


def launch(arguments,log_path,detached=False):
    flags=(subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if detached and os.name=='nt':
        flags|=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('a',encoding='utf-8') as log:
        return subprocess.Popen([sys.executable,'-B','-u','-m',*arguments],cwd=ROOT,
            env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=flags,close_fds=True,start_new_session=detached and os.name!='nt')


def state_write(state,**changes):
    state.update(changes,heartbeat_at=time.time())
    atomic_write(STUDY/'state.json',state)


def resources_available(family):
    import psutil
    from experiment.run_xgboost_followup import memory_available
    for proc in psutil.process_iter(['cmdline']):
        try:
            args=proc.info['cmdline'] or []
            if '--fit' in args or (MODULE in args and '--work' in args):
                return False
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            continue
    headroom=memory_available()
    minimum=2.5 if family.startswith('tabm') or family=='tabpack' else 4.
    return headroom['physical_gib']>=minimum and headroom['commit_gib']>=6.


def progress_value(folder,family):
    path=folder/'progress.json'
    if path.exists():
        value=read(path)
        return value.get('round',value.get('epoch',0)),value.get('validation_accuracy')
    path=folder/'epochs.jsonl'
    if path.exists():
        lines=path.read_text(encoding='utf-8').splitlines()
        if lines:
            try:
                value=json.loads(lines[-1])
            except json.JSONDecodeError:
                return None
            score=value.get('tuning',value.get('early_stopping',{}))
            return value['epoch'],score.get('accuracy')
    return None


def run_worker(protocol,request,state,stage_deadline,trial=None):
    """Restart interrupted attempts; numerical/model failures consume a trial."""
    if time.time()>=min(protocol['deadline'],stage_deadline):
        raise TimeoutError('Study/stage deadline reached before worker launch')
    request=dict(request,study=relative(STUDY))
    parent=STUDY/'runs'/request['job']
    attempts=sorted(parent.glob('attempt_*'))
    for old in attempts:
        if (old/'result.json').exists():
            result=read(old/'result.json')
            if result.get('status')=='completed':
                from framework.training.hybrid_protocol import verify_hashes
                verify_hashes(result['artifact_hashes'])
                return result
    if len(attempts)>=protocol['max_attempts']:
        return None
    # Do not repeatedly rerun a known numerical failure with identical inputs.
    if attempts and (attempts[-1]/'failure.json').exists() and not read(attempts[-1]/'failure.json').get('recoverable'):
        return None
    while not resources_available(request['family']):
        if time.time()>=min(stage_deadline,protocol['deadline']) or (STUDY/'STOP').exists():
            raise TimeoutError('No resource headroom before stage deadline')
        state_write(state,status='waiting_for_resources',job=request['job'])
        time.sleep(10)
    output=parent/f'attempt_{len(attempts):03d}'
    output.mkdir(parents=True)
    atomic_write(output/'request.json',request)
    child=launch([MODULE,'--work',str(output/'request.json')],output/'fit.log')
    state_write(state,status='running',active_job=dict(job=request['job'],family=request['family'],
        pid=child.pid,run_directory=relative(output)))
    started=time.time();last_step=0
    try:
        while child.poll() is None:
            if (STUDY/'STOP').exists() or time.time()>=min(stage_deadline,protocol['deadline']):
                terminate_tree(child.pid)
                atomic_write(output/'interrupted.json',dict(reason='User stop or stage deadline',at=time.time()))
                raise TimeoutError('Worker interrupted at stage boundary')
            if time.time()-started>protocol['per_fit_seconds']:
                terminate_tree(child.pid)
                atomic_write(output/'failure.json',dict(reason='Per-fit watchdog',recoverable=True))
                return None
            if trial is not None:
                progress=progress_value(output,request['family'])
                if progress and progress[0]>last_step and progress[1] is not None:
                    last_step=progress[0];trial.report(progress[1],last_step)
                    # Retain the first three trials per family as full controls.
                    if trial.number>=3 and trial.should_prune():
                        terminate_tree(child.pid)
                        atomic_write(output/'pruned.json',dict(step=last_step,accuracy=progress[1],
                            method='Optuna asynchronous successive halving'))
                        return 'PRUNED'
            state_write(state,elapsed_job_seconds=time.time()-started)
            time.sleep(10)
    finally:
        if child.poll() is None:
            terminate_tree(child.pid)
        state_write(state,active_job=None)
    if (output/'result.json').exists():
        result=read(output/'result.json')
        return result if result.get('status')=='completed' else None
    if not (output/'failure.json').exists():
        atomic_write(output/'failure.json',dict(exit_code=child.returncode,recoverable=True,
            reason='Worker exited without completion'))
    return None


def recover_previous(protocol,state):
    marker=STUDY/'previous_recovery.json'
    if marker.exists():
        return
    limit=min(protocol['started_at']+2*3600,protocol['deadline'])
    launches=0
    while time.time()<limit and not (STUDY/'STOP').exists():
        old=read(OLD_HYBRIDS/'state.json')
        if old['status'] in ('completed','completed_with_search_failures'):
            atomic_write(marker,dict(status=old['status'],completed_at=time.time()))
            return
        pid=old.get('pid')
        if not process_matches(pid,'experiment.resume_hybrids_durable','_run'):
            if launches>=3:
                break
            child=launch(['experiment.resume_hybrids_durable','resume'],STUDY/'recovery.log')
            while child.poll() is None:
                state_write(state,status='recovering_previous_search')
                time.sleep(2)
            launches+=1
            time.sleep(5)
            continue
        state_write(state,status='recovering_previous_search',previous_job=old.get('active_job'))
        time.sleep(10)
    old=read(OLD_HYBRIDS/'state.json')
    if process_matches(old.get('pid'),'experiment.resume_hybrids_durable','_run'):
        (OLD_HYBRIDS/'STOP').touch()
        time.sleep(10)
        if process_matches(old.get('pid'),'experiment.resume_hybrids_durable','_run'):
            terminate_tree(old['pid'])
    atomic_write(marker,dict(status='incomplete_recovery',at=time.time(),error=old.get('error'),
        reason='Recovery budget or bounded restart count exhausted; originals preserved'))


def observations():
    from framework.training.day_search import load_study
    result={}
    for family in FAMILIES:
        result[family]=[dict(value=t.value,seconds=(t.datetime_complete-t.datetime_start).total_seconds()
            if t.datetime_complete and t.datetime_start else 900) for t in load_study(STUDY,family).trials if t.state.is_finished()]
    return result


def search(protocol,state):
    import optuna
    from framework.training.day_search import ask_request,load_study,tell
    limit=protocol['started_at']+14*3600
    while time.time()<limit-300 and not (STUDY/'STOP').exists():
        completed=observations()
        family=next_family(completed)
        # Recover a persisted unfinished ask before allocating a new family.
        for candidate in FAMILIES:
            if any(t.state==optuna.trial.TrialState.RUNNING for t in load_study(STUDY,candidate).trials):
                family=candidate;break
        request=ask_request(STUDY,family,protocol['recipes'][family])
        if request is None:
            continue
        trial=optuna.trial.Trial(load_study(STUDY,family),
            next(t._trial_id for t in load_study(STUDY,family).trials if t.number==request['trial_number']))
        try:
            result=run_worker(protocol,request,state,limit,trial)
        except TimeoutError:
            break
        if result=='PRUNED':
            tell(STUDY,family,request['trial_number'],state='PRUNED')
        elif result:
            score=result['selection_metrics']
            tell(STUDY,family,request['trial_number'],score['accuracy']+1e-6*score['macro_f1'])
        else:
            tell(STUDY,family,request['trial_number'],state='FAIL')
        state_write(state,search_counts={f:len(rows) for f,rows in observations().items()})


def repeats(protocol,state,selected):
    limit=protocol['started_at']+17*3600
    for seed in protocol['repeat_seeds']:
        for family in FAMILIES:
            if time.time()>limit-300 or (STUDY/'STOP').exists():
                return
            record=selected[family]
            request={k:deepcopy(record[k]) for k in ('family','arm','seed','config')}
            request.update(seed=seed,phase='repeat',job=f'{family}_repeat_s{seed}')
            try:
                run_worker(protocol,request,state,limit)
            except TimeoutError:
                return


def oof(protocol,state,selected):
    from framework.training.hybrid_protocol import verify_hashes
    limit=protocol['started_at']+21*3600
    for family in ('tabm_ne','xgboost','kan','tabpack'):
        for fold in (0,1):
            destination=STUDY/'oof_registry'/f'{family}_fold{fold}.json'
            if destination.exists():
                verify_hashes(read(destination)['artifact_hashes']);continue
            if time.time()>limit-300 or (STUDY/'STOP').exists():
                return
            record=selected[family]
            request={k:deepcopy(record[k]) for k in ('family','arm','seed','config')}
            request.update(phase='oof',fold=fold,job=f'{family}_oof{fold}')
            try:
                result=run_worker(protocol,request,state,limit)
            except TimeoutError:
                return
            if result:
                atomic_write(destination,result)


def final_job(protocol,state,job,kind,**kwargs):
    request=dict(job=job,kind=kind,study=relative(STUDY),**kwargs)
    folder=STUDY/'finalization'/job
    marker=folder/'completed.json'
    if marker.exists():
        return
    if time.time()>=protocol['deadline']-60 or (STUDY/'STOP').exists():
        return
    folder.mkdir(parents=True,exist_ok=True)
    atomic_write(folder/'request.json',request)
    child=launch([MODULE,'--final',str(folder/'request.json')],folder/'job.log')
    state_write(state,status='finalizing',active_job=dict(job=job,pid=child.pid,run_directory=relative(folder)))
    try:
        while child.poll() is None:
            if time.time()>=protocol['deadline']-30 or (STUDY/'STOP').exists():
                terminate_tree(child.pid);return
            state_write(state);time.sleep(10)
    finally:
        if child.poll() is None:
            terminate_tree(child.pid)
        state_write(state,active_job=None)
    atomic_write(folder/('completed.json' if child.returncode==0 else 'failure.json'),dict(exit_code=child.returncode))


def finalize(protocol,state,selected):
    from framework.training.day_finalize import report
    for family in ('random_forest',*FAMILIES):
        final_job(protocol,state,f'evaluate_{family}','evaluate',family=family,record=selected[family])
    for name,names in protocol['combinations'].items():
        required=[STUDY/'oof_registry'/f'{f}_fold{i}.json' for f in names for i in (0,1)]
        if all(p.exists() for p in required):
            final_job(protocol,state,name,'hybrid',name=name,names=names)
    # Assess repeats only after all selections are sealed, never select by seed.
    for path in (STUDY/'runs').glob('*_repeat_s*/attempt_*/result.json'):
        row=read(path)
        if row.get('status')=='completed':
            final_job(protocol,state,f'evaluate_{row["family"]}_s{row["seed"]}','evaluate',
                family=row['family'],record=row,tag=f'seed{row["seed"]}')
    rows=report(STUDY,protocol)
    finished=sum(r['status']=='completed' for r in rows.values())
    state_write(state,status='completed' if finished==10 else 'finished_with_gaps',completed_models=finished)


def controller():
    protocol=read(STUDY/'study.json');verify(protocol)
    state=dict(pid=os.getpid(),started_at=protocol['started_at'],deadline=protocol['deadline'],
        status='starting',test_loaded=False,active_job=None)
    with StudyLock(STUDY/'controller.lock'):
        job_handle=protect_descendants()
        state_write(state)
        recover_previous(protocol,state)
        if not (STUDY/'selected.json').exists() and phase_at(protocol)=='search':
            search(protocol,state)
        if (STUDY/'STOP').exists():
            state_write(state,status='stopped');return
        from framework.training.day_finalize import freeze_selection
        selected=freeze_selection(STUDY,protocol)
        if phase_at(protocol) in ('search','repeats'):
            repeats(protocol,state,selected)
        if phase_at(protocol) in ('search','repeats','oof'):
            oof(protocol,state,selected)
        finalize(protocol,state,selected)


def supervise():
    """Own the controller independently of Codex turns, with bounded restarts."""
    protocol=read(STUDY/'study.json');verify(protocol)
    with StudyLock(STUDY/'supervisor.lock'):
        job_handle=protect_descendants()
        if os.name=='nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            restarts=0
            while time.time()<protocol['deadline'] and not (STUDY/'STOP').exists():
                child=launch([MODULE,'_controller'],STUDY/'controller.log')
                atomic_write(STUDY/'supervisor.json',dict(pid=os.getpid(),controller_launcher_pid=child.pid,
                    deadline=protocol['deadline'],restarts=restarts,status='running',heartbeat_at=time.time()))
                while child.poll() is None and time.time()<protocol['deadline']:
                    if (STUDY/'STOP').exists():
                        terminate_tree(child.pid);break
                    atomic_write(STUDY/'supervisor.json',dict(pid=os.getpid(),controller_launcher_pid=child.pid,
                        deadline=protocol['deadline'],restarts=restarts,status='running',heartbeat_at=time.time()))
                    time.sleep(15)
                if child.poll() is None:
                    terminate_tree(child.pid)
                current=read(STUDY/'state.json') if (STUDY/'state.json').exists() else {}
                if current.get('status') in ('completed','finished_with_gaps','stopped') or (STUDY/'STOP').exists():
                    break
                restarts+=1
                if restarts>3:
                    atomic_write(STUDY/'supervisor.json',dict(pid=os.getpid(),status='needs_attention',
                        restarts=restarts,error='Controller failed repeatedly; see controller.log',deadline=protocol['deadline']))
                    return
                time.sleep(min(60,10*restarts))
            if time.time()>=protocol['deadline']:
                from framework.training.day_finalize import report
                rows=report(STUDY,protocol)
                atomic_write(STUDY/'state.json',dict(status='deadline_reached',pid=os.getpid(),
                    deadline=protocol['deadline'],test_loaded=False,completed_models=sum(r['status']=='completed' for r in rows.values())))
            atomic_write(STUDY/'supervisor.json',dict(pid=os.getpid(),status='exited',deadline=protocol['deadline'],restarts=restarts))
        finally:
            if os.name=='nt':
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def final_worker(path):
    from framework.training.day_finalize import evaluate,build_hybrid
    request=read(path);protocol=read(STUDY/'study.json');verify(protocol)
    if request['kind']=='evaluate':
        evaluate(STUDY,request['family'],request['record'],request.get('tag','selected'))
    else:
        build_hybrid(STUDY,request['name'],request['names'],read(STUDY/'selected.json'),protocol)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',choices=('start','resume','initialize','status','stop','_supervise','_controller'))
    parser.add_argument('--work',type=Path);parser.add_argument('--final',type=Path)
    args=parser.parse_args()
    if args.work:
        from framework.training.day_worker import fit
        try:
            fit(args.work)
        except Exception as exc:
            atomic_write(args.work.parent/'failure.json',dict(error=str(exc),type=type(exc).__name__,
                traceback=traceback.format_exc(),recoverable=isinstance(exc,(PermissionError,MemoryError))))
            raise
    elif args.final:
        final_worker(args.final)
    elif args.command in ('start','resume','initialize'):
        protocol=initialize()
        if args.command=='initialize':
            print(protocol['deadline_utc']);return
        if time.time()>=protocol['deadline']:
            raise TimeoutError('Original 24-hour deadline already elapsed; create a new version')
        with StudyLock(STUDY/'launch.lock'):
            with StudyLock(STUDY/'supervisor.lock'):
                pass
            (STUDY/'STOP').unlink(missing_ok=True)
            child=launch([MODULE,'_supervise'],STUDY/'supervisor.log',detached=True)
            atomic_write(STUDY/'launch.json',dict(pid=child.pid,at=time.time(),deadline=protocol['deadline']))
        print(f'Supervisor launcher PID {child.pid}; deadline {protocol["deadline_utc"]}')
    elif args.command=='_supervise':
        supervise()
    elif args.command=='_controller':
        controller()
    elif args.command=='stop':
        (STUDY/'STOP').touch()
    elif args.command=='status':
        for name in ('supervisor','state'):
            if (STUDY/f'{name}.json').exists():
                print(name,json.dumps(read(STUDY/f'{name}.json'),indent=2))


if __name__=='__main__':
    main()
