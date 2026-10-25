"""Open-ended, detached full-CSV research. No app promotion or locked-test access.

Usage: python -m experiment.optimize_csv_research start|status|stop|resume
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiment import optimize_models as lifecycle
from experiment.csv_research_policy import (FAMILIES, PIPELINE, fingerprint, recipe_key,
     build_recipes, capacity_trial, score, success_report, propose_tpe, report_trial, tell_trial)

ROOT=lifecycle.ROOT
STUDY=ROOT/'artifacts/training/csv_research_v3'
PREPARED=ROOT/'dataset/prepared/datasense_full_protocols'
MODULE='experiment.optimize_csv_research'
WORKER='experiment.train_csv_research'
PREVIOUS=ROOT/'artifacts/training/csv_specialization_v2'
DATABASE=STUDY/'optuna.sqlite3'


def initial_state() -> dict:
    return {'started':time.time(),'deadline':None,'status':'created','phase':'solo_and_pairs',
            'active_trial':None,'target_accuracy':.95,'test_loaded':False,'live_training_started':False,
            'rf_fixed':True,'queue_index':0,'tuning_index':{f:0 for f in FAMILIES},
            'capacity_stage':{f:0 for f in FAMILIES},'capacity_at_count':{f:0 for f in FAMILIES},
            'pending_requests':[],'postprocess_keys':[],'combination_keys':[]}


def cuda_verified(row: dict) -> bool:
    if not str(row.get('device','')).startswith('cuda'):
        return False
    if row.get('gpu_training_verified') is True:
        return True
    evidence=row.get('device_evidence',{})
    devices=evidence.get('parameter_devices',[])
    return bool(devices) and all(str(d).startswith('cuda') for d in devices)


def trial_budget(request: dict) -> float:
    budget=float(request.get('watchdog_seconds',1200))
    if not 0<budget<=14400:
        raise ValueError('Per-fit watchdog must be positive and at most four hours')
    return budget


def oom_retry(request: dict) -> dict:
    row=deepcopy(request)
    config=row['config']
    if row['model_id']=='xgboost':
        config['max_bin']=max(64,int(config.get('max_bin',256))//2)
        config['max_depth']=max(3,int(config.get('max_depth',6))-2)
    else:
        config['batch_size']=max(32,int(config.get('batch_size',512))//2)
        if row['model_id']=='hem':
            config['knn_batch_size']=max(1,int(config.get('knn_batch_size',64))//2)
    row.pop('optuna_trial_number',None)
    row.pop('optuna_study_name',None)
    row['variant']=str(row.get('variant','trial'))+'_memory_recovery'
    row['decision']={'reason':'GPU memory recovery; all real rows retained', 'parent_fingerprint':fingerprint(request)}
    return row


def all_locks():
    stack=ExitStack()
    try:
        for folder in (lifecycle.STUDY,PREVIOUS,STUDY):
            stack.enter_context(lifecycle.StudyLock(folder/'controller.lock'))
    except Exception:
        stack.close()
        raise
    return stack


def pin_inputs() -> dict:
    paths=[PREPARED/'manifest.json']
    paths += [PREPARED/f'{kind}_{split}.npy' for kind in ('X','y','row_ids') for split in ('train','validation')]
    paths += [ROOT/p for p in ('experiment/optimize_csv_research.py','experiment/train_csv_research.py',
            'experiment/csv_research_policy.py','experiment/train_csv_models.py','experiment/optimize_models.py',
            'framework/training/research_features.py','framework/training/research_augmentation.py',
            'framework/training/augmentation.py','framework/training/research_losses.py',
            'framework/training/neural.py','framework/training/csv_xgboost.py','framework/training/trees.py',
            'framework/training/partial_objectives.py','framework/training/csv_monitoring.py',
            'framework/training/fixed_random_forest.py','requirements-research.txt')]
    for upstream in ('HEM','efficient-kan'):
        paths+=list((ROOT/'external'/upstream).rglob('*.py'))
    return {p.relative_to(ROOT).as_posix():lifecycle.digest(p) for p in paths}


def prepare(request: dict, definition: dict) -> dict:
    if request['model_id'] not in FAMILIES or request.get('input_profile','full_csv')!='full_csv':
        raise ValueError('Only three upgraded full-CSV families are eligible; RF is fixed')
    row=deepcopy(request)
    row.update(input_profile='full_csv',research_policy='csv_research_v3',prepared=PREPARED.relative_to(ROOT).as_posix(),
               prepared_manifest_sha256=definition['prepared_manifest_sha256'],protocol_id=definition['protocol_id'],
               train_row_ids_sha256=definition['train_row_ids_sha256'],validation_row_ids_sha256=definition['validation_row_ids_sha256'],
               dataset_train_sha256=definition['dataset_train_sha256'])
    for key in ('dataset_labels_sha256','dataset_validation_sha256','dataset_validation_labels_sha256'):
        if key in definition:
            row[key]=definition[key]
    row['pipeline']={**PIPELINE,**row.get('pipeline',{})}
    if row['model_id']=='hem' and row['pipeline']['feature_selection'] in ('mi64','mi128'):
        dimension=int(row['pipeline']['feature_selection'][2:])
        row['config']['pca_components']=min(int(row['config'].get('pca_components',32)),dimension)
    row.setdefault('seed',42)
    row.setdefault('watchdog_seconds',1200)
    row['config']['runtime_seconds']=min(float(row['config'].get('runtime_seconds',trial_budget(row)-180)),trial_budget(row)-180)
    row['fingerprint']=fingerprint(row)
    row['recipe_key']=recipe_key(row)
    return row


def request_of(row: dict) -> dict:
    keys=('model_id','variant','config','seed','pipeline','input_profile','research_policy','prepared',
          'prepared_manifest_sha256','protocol_id','train_row_ids_sha256','validation_row_ids_sha256',
          'dataset_train_sha256','dataset_labels_sha256','dataset_validation_sha256',
          'dataset_validation_labels_sha256','watchdog_seconds','architecture_provenance')
    return {k:deepcopy(row[k]) for k in keys if k in row}


def adoption_request(row: dict, definition: dict) -> dict:
    """Verify original evidence before adding the new study's recipe identity."""
    for key in ('prepared_manifest_sha256','protocol_id','train_row_ids_sha256','validation_row_ids_sha256'):
        if row.get(key) != definition[key]:
            raise ValueError(f'Adopted fit input identity differs: {key}')
    if row.get('input_profile') != 'full_csv' or row.get('checkpoint_reload_verified') is not True or not cuda_verified(row):
        raise ValueError('Adopted CSV fit lacks verified CPU reload/CUDA evidence')
    return prepare(request_of(row),definition)


def verify_prior_inputs(prior_definition: dict) -> None:
    """Same row IDs are insufficient: match prior raw measurements and labels."""
    hashes={Path(name).as_posix():value for name,value in prior_definition['source_input_hashes'].items()}
    paths=[PREPARED/'manifest.json'] + [PREPARED/f'{kind}_{split}.npy'
             for kind in ('X','y','row_ids') for split in ('train','validation')]
    for path in paths:
        name=path.relative_to(ROOT).as_posix()
        if hashes.get(name) != lifecycle.digest(path):
            raise ValueError(f'Prior training input identity changed: {name}')


def verify_result(path: Path) -> dict | None:
    result=lifecycle.verified(path)
    if result is None:
        return None
    assets=path.parent/'bundle_integrity.json'
    if assets.exists():
        for name,expected in json.loads(assets.read_text()).items():
            asset=path.parent/name
            if not asset.is_file() or lifecycle.digest(asset)!=expected:
                raise ValueError(f'Research bundle corruption: {asset}')
    return result


def completed_results(*,verify: bool=False) -> list[dict]:
    rows=[]
    adopted=STUDY/'adopted_results.json'
    if adopted.exists():
        for record in json.loads(adopted.read_text()):
            path=ROOT/record['path']
            if lifecycle.digest(path)!=record['sha256']:
                raise ValueError('Original adopted result changed')
            row=verify_result(path) if verify else json.loads(path.read_text())
            if row:
                row={**row,**record['request'],'adopted_from':record['path']}
                row['cuda_verified']=cuda_verified(row)
                rows.append(row)
    for path in sorted((STUDY/'runs').glob('*/attempt_*/result.json')):
        row=verify_result(path) if verify else json.loads(path.read_text())
        if row and row.get('status')=='completed': rows.append(row)
    return rows


def initialize() -> None:
    STUDY.mkdir(parents=True,exist_ok=True)
    with all_locks():
        if (STUDY/'study.json').exists():
            return
        previous=json.loads((PREVIOUS/'state.json').read_text())
        if previous.get('status')=='running' or previous.get('active_trial'):
            raise RuntimeError('Stop the previous CUDA controller before transition')
        from experiment.optimize_csv_models import completed_results as previous_results
        old_rows=previous_results(verify=True)
        if not all(any(r['model_id']==f and r['variant']=='control' for r in old_rows) for f in FAMILIES):
            raise ValueError('Missing completed CSV controls')
        prior_definition=json.loads((PREVIOUS/'study.json').read_text())
        verify_prior_inputs(prior_definition)
        definition={'protocol_id':prior_definition['protocol_id'],
            'prepared_manifest_sha256':lifecycle.digest(PREPARED/'manifest.json'),
            'train_row_ids_sha256':lifecycle.digest(PREPARED/'row_ids_train.npy'),
            'validation_row_ids_sha256':lifecycle.digest(PREPARED/'row_ids_validation.npy'),
            'dataset_train_sha256':lifecycle.digest(PREPARED/'X_train.npy'),
            'dataset_labels_sha256':lifecycle.digest(PREPARED/'y_train.npy'),
            'dataset_validation_sha256':lifecycle.digest(PREPARED/'X_validation.npy'),
            'dataset_validation_labels_sha256':lifecycle.digest(PREPARED/'y_validation.npy'),
            'controls':{},'deadline':None,'target_accuracy':.95,'fixed_rf':prior_definition['fixed_rf']}
        controls={f:next(r for r in old_rows if r['model_id']==f and r['variant']=='control') for f in FAMILIES}
        definition['controls']={f:adoption_request(r,definition) for f,r in controls.items()}
        adopted=[]
        # Locate actual prior result files by checkpoint, without changing their bytes.
        prior_paths=list((PREVIOUS/'runs').glob('*/attempt_*/result.json'))
        older=ROOT/'artifacts/training/csv_specialization_v1/runs'
        prior_paths+=list(older.glob('*/attempt_*/result.json'))
        for row in old_rows:
            path=next((p for p in prior_paths if json.loads(p.read_text()).get('checkpoint_path')==row['checkpoint_path']),None)
            if path is None: raise ValueError('Cannot find adopted checkpoint record')
            adopted.append({'path':path.relative_to(ROOT).as_posix(),'sha256':lifecycle.digest(path),
                            'request':adoption_request(row,definition)})
        lifecycle.write(STUDY/'adopted_results.json',adopted)
        definition['adopted_sha256']=lifecycle.digest(STUDY/'adopted_results.json')
        queue,rejected=build_recipes(definition['controls'])
        queue=[prepare(r,definition) for r in queue]
        # Interleave families within solo/pair phases to avoid starvation.
        interleaved=[]
        for phase in ('control','solo_','pair_'):
            groups={f:[r for r in queue if r['model_id']==f and (r['variant']==phase if phase=='control' else r['variant'].startswith(phase))] for f in FAMILIES}
            for i in range(max(map(len,groups.values()))):
                interleaved += [groups[f][i] for f in FAMILIES if i<len(groups[f])]
        definition['queue']=interleaved
        definition['incompatibilities']=rejected
        definition['source_input_hashes']=pin_inputs()
        for name in definition['source_input_hashes']:
            if name.endswith('.py'):
                target=STUDY/'source_snapshot'/name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(ROOT/name,target)
        lifecycle.write(STUDY/'study.json',definition)
        lifecycle.write(STUDY/'state.json',initial_state())
        lifecycle.write(STUDY/'recovery.json',lifecycle.register_recovery(MODULE,recovery_command='recover'))


def handoff(state: dict) -> None:
    rows=completed_results()
    state['completed_fit_records']=dict(Counter(r['model_id'] for r in rows))
    state['updated_at']=time.time()
    lifecycle.write(STUDY/'state.json',state)
    lines=['# Open-ended full-CSV research handoff','',
      'Optimize XGBoost, efficient-KAN and HEM-derived closed-set models. Conventional RF fixed.',
      'Target: one family >=95% full 84-category validation accuracy in seeds42/43/44.',
      'No overall deadline. Per-fit watchdogs:20min/1h/4h. CUDA fits sequential; CPU reload verified.',
      'Real training/validation partitions unchanged; no locked-test loading, live training or app promotion.',
      f"Status:{state['status']}; phase:{state.get('phase')}; PID:{state.get('pid')}",
      f"Completed fits:{state['completed_fit_records']}",f"Active:{json.dumps(state.get('active_trial'))}",'',
      'Commands from the project root:','```powershell',
      '.venv/Scripts/python.exe -B -m experiment.optimize_csv_research status',
      '.venv/Scripts/python.exe -B -m experiment.optimize_csv_research stop',
      '.venv/Scripts/python.exe -B -m experiment.optimize_csv_research resume','```','',
      'Do not resume retired optimize_models or optimize_csv_models controllers.',
      'Recipe requests/results/logs and Optuna SQLite:artifacts/training/csv_research_v3/.',
      'Synthetic additions are labelled training augmentations, not independently captured traffic.',
      'Next:finish solo/pairs, supported combinations, Optuna and capacity trials, seed repeats.',
      'Separate future live stage remains blocked; reaching this one-family goal does not open its prior all-four gate.']
    for family in FAMILIES:
        candidates=[r for r in rows if r['model_id']==family]
        if candidates:
            best=max(candidates,key=score); m=best['metrics']
            lines.append(f"- {family}: best provisional accuracy {m['accuracy']:.6f}, macro recall {m['macro_recall']:.6f}, F1 {m['macro_f1']:.6f}; {best['variant']}")
    (ROOT/'docs/RESUME_CONTEXT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run_trial(request: dict,state: dict,definition: dict) -> dict | None:
    request=prepare(request,definition)
    key=request['fingerprint']
    for row in completed_results():
        if row['fingerprint']==key:
            origin=ROOT/row['adopted_from'] if row.get('adopted_from') else next((STUDY/'runs'/key).glob('attempt_*/result.json'))
            verify_result(origin)
            return row
    directory=STUDY/'runs'/key
    previous=sorted(directory.glob('attempt_*/request.json'))
    if previous:
        immutable=json.loads(previous[0].read_text())
        if fingerprint(immutable)!=key:
            raise ValueError('Immutable recipe identity changed')
        # Interrupted attempts resume only on explicit restart; ordinary failures
        # are isolated once instead of repeatedly spending GPU time.
        last=previous[-1].parent
        failure=last/'failure.json'
        if failure.exists() and json.loads(failure.read_text()).get('kind') not in ('interrupted','controller_restart'):
            return None
        request=immutable
    if (STUDY/'STOP').exists(): return None
    attempt=directory/f'attempt_{len(previous):03d}'
    attempt.mkdir(parents=True)
    lifecycle.write(attempt/'request.json',request)
    budget=trial_budget(request); deadline=time.time()+budget
    lifecycle.write(attempt/'execution.json',{'watchdog_seconds':budget,'deadline':deadline,'overall_deadline':None})
    state['active_trial']={'request':request,'directory':attempt.relative_to(ROOT).as_posix(),'started':time.time(),'budget_seconds':budget}
    with (attempt/'fit.log').open('a',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,'-B','-u','-m',WORKER,'--fit',str(attempt/'request.json')],
             cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'IMDAI_FIT_DEADLINE':str(deadline)},
             creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,start_new_session=os.name!='nt')
        state['active_trial']['pid']=child.pid
        handoff(state)
        reason=None; step=-1
        while child.poll() is None:
            if (STUDY/'STOP').exists(): reason={'kind':'interrupted','message':'Explicit Stop'}
            elif time.time()>=deadline: reason={'kind':'timeout','message':'Per-fit watchdog expired'}
            if reason:
                lifecycle.terminate_tree(child); break
            progress=attempt/'progress.json'
            if request.get('optuna_trial_number') is not None and progress.exists():
                try:
                    p=json.loads(progress.read_text()); current=p.get('epoch',p.get('round',-1))
                    if current>step:
                        step=current
                        if report_trial(DATABASE,request['model_id'],request['optuna_trial_number'],step,p['validation_accuracy']):
                            (attempt/'PRUNE').touch()
                except (OSError,json.JSONDecodeError): pass
            time.sleep(1)
        result=verify_result(attempt/'result.json')
        if result is None:
            failure_file=attempt/'worker_failure.json'
            failure=json.loads(failure_file.read_text()) if failure_file.exists() else {'kind':'failed','message':f'Worker exit {child.returncode}'}
            if reason: failure=reason
            failure.update(exit_code=child.returncode,test_loaded=False)
            lifecycle.write(attempt/'failure.json',failure)
            if failure['kind']=='cuda_oom':
                retry=prepare(oom_retry(request),definition)
                if retry['fingerprint']!=key:
                    state['pending_requests'].append(retry)
            terminal='pruned' if failure['kind']=='pruned' else 'fail'
        else: terminal='complete'
    if request.get('optuna_trial_number') is not None and not (reason and reason['kind']=='interrupted'):
        tell_trial(DATABASE,request['model_id'],request['optuna_trial_number'],result['metrics']['accuracy'] if result else None,state=terminal)
    state['active_trial']=None
    handoff(state)
    return result


def plateau(rows: list[dict]) -> bool:
    ordered=sorted(rows,key=lambda r:r.get('completed_at',0))
    if len(ordered)<7: return False
    return max(score(r)[0] for r in ordered[-6:])-max(score(r)[0] for r in ordered[:-6])<.001


def next_capacity(rows: list[dict], state: dict) -> dict | None:
    """Inspect stagnation at fit boundaries, even while solo/pairs are queued."""
    for family in FAMILIES:
        group=[r for r in rows if r['model_id']==family and r['seed']==42]
        if not plateau(group) or len(group)-state['capacity_at_count'][family]<6:
            continue
        leader=max(group,key=score)
        latest=max(group,key=lambda r:r.get('completed_at',0))
        history=latest.get('history',[])
        observed=next((p for p in reversed(history) if p.get('training_accuracy') is not None),{})
        train=observed.get('training_accuracy'); validation=observed.get('validation_accuracy')
        gap=float(train-validation) if train is not None and validation is not None else None
        trial=capacity_trial(family,request_of(leader),state['capacity_stage'][family])
        trial['decision']={'reason':'Six stagnant completed fits: capacity ablation interleaved before next recipe',
             'best_validation_accuracy':score(leader)[0], 'training_monitor_accuracy':train,
             'training_monitor_rows':observed.get('training_monitor_rows'),
             'training_validation_gap':gap, 'monitor_is_sampled':True,
             'diagnosis':'monitor_unavailable' if gap is None else 'possible_overfitting' if gap>.03 else 'capacity_probe'}
        if gap is not None and gap>.03:
            if family=='xgboost':
                trial['config']['learning_rate']=float(trial['config'].get('learning_rate',.08))*.5
                trial['config']['reg_lambda']=max(6.,float(trial['config'].get('reg_lambda',1.)))
            else:
                trial['config']['weight_decay']=max(.001,float(trial['config'].get('weight_decay',.0001)))
        state['capacity_stage'][family]+=1
        state['capacity_at_count'][family]=len(group)
        return trial
    return None


def supported_combinations(rows: list[dict],definition: dict,state: dict) -> list[dict]:
    proposals=[]
    for family in FAMILIES:
        base=definition['controls'][family]
        control=next(r for r in rows if r['fingerprint']==base['fingerprint'])
        improvements=[r for r in rows if r['model_id']==family and r['seed']==42 and
                      r['variant'].startswith('solo_') and score(r)>score(control)]
        if len(improvements)<3: continue
        combined=request_of(base); merged={}; pipe={}
        for row in sorted(improvements,key=score,reverse=True):
            for k,v in row['config'].items():
                if v!=base['config'].get(k): merged.setdefault(k,v)
            for k,v in row.get('pipeline',{}).items():
                if v!=base['pipeline'].get(k): pipe.setdefault(k,v)
        combined['config'].update(merged); combined['pipeline'].update(pipe)
        combined['variant']='supported_higher_combination'
        combined=prepare(combined,definition)
        if combined['fingerprint'] not in state['combination_keys']:
            state['combination_keys'].append(combined['fingerprint']); proposals.append(combined)
    return proposals


def postprocess_trials(rows: list[dict],definition: dict,state: dict) -> list[dict]:
    proposals=[]
    for family in FAMILIES:
        group=[r for r in rows if r['model_id']==family and r['seed']==42 and
               r.get('pipeline',{}).get('ensemble','none')=='none' and r.get('pipeline',{}).get('calibration','none')=='none']
        leader=max(group,key=score)
        for kind in (('ensemble',) if family=='hem' else ('ensemble','calibration')):
            request=request_of(leader); request['pipeline'][kind]='seed_average' if kind=='ensemble' else 'temperature_oof'
            request['variant']='postprocess_'+kind; request['watchdog_seconds']=14400
            request=prepare(request,definition)
            if request['fingerprint'] not in state['postprocess_keys']:
                state['postprocess_keys'].append(request['fingerprint']); proposals.append(request)
    return proposals


def controller() -> None:
    with all_locks():
        job=lifecycle.protect_descendants()
        state=json.loads((STUDY/'state.json').read_text())
        if state.get('status')=='success': return
        if (STUDY/'STOP').exists():
            state.update(status='stopped',active_trial=None); handoff(state); return
        state.update(status='running',pid=os.getpid())
        if os.name=='nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            definition=json.loads((STUDY/'study.json').read_text())
            if pin_inputs()!=definition['source_input_hashes']:
                raise ValueError('Training source/input changed; preserve study and version before resuming')
            if lifecycle.digest(STUDY/'adopted_results.json')!=definition['adopted_sha256']:
                raise ValueError('Adopted-result index changed')
            completed_results(verify=True)
            # Job-object cleanup guarantees an interrupted predecessor cannot
            # leave a competing CUDA descendant behind after recovery.
            if state.get('active_trial'):
                old=ROOT/state['active_trial']['directory']
                if not (old/'result.json').exists():
                    lifecycle.write(old/'failure.json',{'kind':'controller_restart','message':'Recovered interrupted attempt'})
                state['active_trial']=None
            handoff(state)
            while not (STUDY/'STOP').exists():
                rows=completed_results()
                controls={f:next(r for r in rows if r['fingerprint']==definition['controls'][f]['fingerprint']) for f in FAMILIES}
                report=success_report(rows,controls)
                lifecycle.write(STUDY/'report.json',report)
                if report['success']:
                    completed_results(verify=True)
                    state.update(status='success',phase='repeat_validated_target',active_trial=None)
                    handoff(state); return
                # Freeze near-target repeats as soon as they qualify. Other
                # studies keep running if any seed or metric guard fails.
                repeats=[]
                for family in FAMILIES:
                    group=[r for r in rows if r['model_id']==family and r['seed']==42]
                    leader=max(group,key=score)
                    if score(leader)[0]>=.95:
                        for seed in (43,44):
                            repeat=request_of(leader); repeat.update(seed=seed,variant='target_repeat')
                            repeat=prepare(repeat,definition)
                            if not any(r['fingerprint']==repeat['fingerprint'] for r in rows): repeats.append(repeat)
                repeats=[r for r in repeats if not any((STUDY/'runs'/r['fingerprint']).glob('attempt_*/failure.json'))]
                if state.get('scheduled_request'):
                    trial=state['scheduled_request']
                elif repeats: trial=repeats[0]; state['phase']='target_seed_repeats'
                elif state['pending_requests']: trial=state['pending_requests'].pop(0)
                elif (capacity := next_capacity(rows,state)) is not None:
                    trial=capacity; state['phase']='capacity_escalation'
                elif state['queue_index']<len(definition['queue']):
                    trial=definition['queue'][state['queue_index']]
                    state['phase']='solo_and_pairs'
                    state['scheduled_queue_index']=state['queue_index']
                else:
                    proposals=supported_combinations(rows,definition,state)
                    if not proposals and not state.get('postprocess_initial_done'):
                        proposals=postprocess_trials(rows,definition,state)
                        state['postprocess_initial_done']=True
                    if proposals:
                        state['pending_requests']+=proposals[1:]; trial=proposals[0]; state['phase']='supported_combination_or_postprocess'
                    else:
                        family=FAMILIES[state.get('tuning_family_index',0)%len(FAMILIES)]
                        state['tuning_family_index']=state.get('tuning_family_index',0)+1
                        group=[r for r in rows if r['model_id']==family and r['seed']==42]
                        leader=max(group,key=score)
                        trial=propose_tpe(DATABASE,family,request_of(leader),rows,state['tuning_index'][family])
                        state['tuning_index'][family]+=1; state['phase']='optuna_tuning'
                trial=prepare(trial,definition)
                # Persist the chosen request BEFORE fit creation; resume never
                # silently advances past a fit that had not completed.
                state['scheduled_request']=trial
                handoff(state)
                result=run_trial(trial,state,definition)
                if not (STUDY/'STOP').exists():
                    if state.get('scheduled_queue_index') is not None:
                        state['queue_index']=state.pop('scheduled_queue_index')+1
                    state.pop('scheduled_request',None)
                handoff(state)
            state.update(status='stopped',active_trial=None)
        except Exception as exc:
            state.update(status='failed',error=f'{type(exc).__name__}: {exc}',active_trial=None)
            raise
        finally:
            if os.name=='nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            handoff(state)


def process_alive(pid: int | None) -> bool:
    if not pid: return False
    import psutil
    try:
        process=psutil.Process(pid)
        return process.is_running() and MODULE in ' '.join(process.cmdline())
    except (psutil.NoSuchProcess,psutil.AccessDenied): return False


def launch() -> None:
    with all_locks():
        state=json.loads((STUDY/'state.json').read_text())
        if state['status']=='success': print('Repeat-validated target already reached'); return
        marker=STUDY/'launch.json'
        if marker.exists() and process_alive(json.loads(marker.read_text()).get('pid')):
            raise RuntimeError('Research controller already launching')
        (STUDY/'STOP').unlink(missing_ok=True)
        flags=(subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW) if os.name=='nt' else 0
        with (STUDY/'controller.log').open('a',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-B','-u','-m',MODULE,'_run'],cwd=ROOT,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,close_fds=True,creationflags=flags,start_new_session=os.name!='nt')
        lifecycle.write(marker,{'pid':child.pid,'started':time.time()})
        print(json.dumps({'launched_pid':child.pid,'overall_deadline':None,'target':.95}))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('start','status','stop','resume','recover','_run'))
    command=parser.parse_args().command
    if command=='start': initialize(); launch()
    elif command=='resume':
        if (STUDY/'study.json').exists(): launch()
    elif command=='recover':
        if (STUDY/'study.json').exists() and not (STUDY/'STOP').exists(): launch()
    elif command=='stop':
        STUDY.mkdir(parents=True,exist_ok=True); (STUDY/'STOP').touch(); print('Explicit research Stop requested')
    elif command=='status':
        print((STUDY/'state.json').read_text() if (STUDY/'state.json').exists() else 'Research not started')
    else: controller()


if __name__=='__main__': main()
