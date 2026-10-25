"""TPE combiner search and reuse of the verified calibrated bundle publisher."""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from experiment.train_verified_clean import read,write
from experiment.datasense_quality import digest
from framework.training.hybrid_protocol import ROOT,DATA,relative,branch_reference,verify_hashes
from framework.training.hybrid_hpo_data import STUDY,PREVIOUS
from framework.training.hybrid_hpo import trial_study
from framework.training.hybrid_blend import fit_blend,blend,quick_metrics,log_loss


def candidate(p_oof,y_oof,p_tune,y_tune,names,penalty):
    cfg=fit_blend(p_oof,y_oof,names,penalty)
    p=blend(p_tune,cfg,names)
    score=quick_metrics(y_tune,p);score['log_loss']=log_loss(p,y_tune)
    return dict(combiner=cfg,tuning_metrics=score,
        objective=score['accuracy']+1e-6*score['macro_f1'],penalty=penalty)


def optimize_combiner(name,names,p_oof,y_oof,p_tune,y_tune,protocol):
    import optuna
    directory=STUDY/'combiner_search'/name;directory.mkdir(parents=True,exist_ok=True)
    global_path=directory/'global.json'
    if not global_path.exists():write(global_path,candidate(p_oof,y_oof,p_tune,y_tune,names,None))
    study_name='combiner_'+name
    while True:
        study=trial_study(STUDY,study_name);trials=study.trials
        unfinished=[t for t in trials if t.state==optuna.trial.TrialState.RUNNING]
        if not unfinished and len(trials)>=protocol['combiner_trials']:break
        if (STUDY/'STOP').exists():raise InterruptedError('Stopped during combiner search')
        if unfinished:
            trial=unfinished[0]
            if 'penalty' not in trial.params:
                study.tell(trial.number,state=optuna.trial.TrialState.FAIL);continue
            number,penalty=trial.number,trial.params['penalty']
        else:
            if not trials:
                study.enqueue_trial({'penalty':.01});study.enqueue_trial({'penalty':.1})
            trial=study.ask();number=trial.number
            penalty=trial.suggest_float('penalty',.0001,1.,log=True)
        path=directory/f'trial_{number:03d}.json'
        try:
            if path.exists():
                result=read(path)
                if result.get('trial_number')!=number or result.get('penalty')!=penalty:
                    raise ValueError('Persisted combiner trial identity changed')
                if result.get('status')=='failed':
                    study.tell(number,state=optuna.trial.TrialState.FAIL);continue
                if result.get('status')!='completed' or not np.isfinite(result.get('objective',np.nan)):
                    raise ValueError('Incomplete persisted combiner score')
            else:result=candidate(p_oof,y_oof,p_tune,y_tune,names,penalty)
            result.update(status='completed',trial_number=number)
            write(path,result);study.tell(number,result['objective'])
        except (RuntimeError,FloatingPointError) as exc:
            write(path,dict(status='failed',trial_number=number,penalty=penalty,error=str(exc)))
            study.tell(number,state=optuna.trial.TrialState.FAIL)
    candidates=[read(global_path)]+[read(p) for p in directory.glob('trial_*.json') if read(p).get('status')=='completed']
    chosen=max(candidates,key=lambda r:(r['tuning_metrics']['accuracy'],r['tuning_metrics']['macro_f1'],-r['tuning_metrics']['log_loss']))
    return chosen,dict(study_name=study_name,planned_trials=protocol['combiner_trials'],
        trial_states={state.name:sum(t.state==state for t in trial_study(STUDY,study_name).trials)
                      for state in (optuna.trial.TrialState.COMPLETE,optuna.trial.TrialState.FAIL)},
        selected_trial=chosen.get('trial_number'),selected_penalty=chosen['penalty'],
        artifacts={relative(p):digest(p) for p in directory.glob('*.json')})


def finalize_all(protocol,registry):
    from framework.training import hybrid_evaluation as publisher
    publisher.STUDY=STUDY
    final_branches={family:branch_reference(family,ROOT/registry[f'{family}_full']['run_directory'])
                    for family in protocol['family_order']}
    y=np.load(DATA/'model/y_train.npy');validation_y=np.load(DATA/'model/y_validation.npy')
    role=np.load(STUDY/'validation_roles.npy');tune=role==0
    train_ids=np.load(DATA/'model/row_ids_train.npy')
    fold_ids=[np.load(STUDY/f'folds/{i}/oof.npy') for i in (0,1)]
    for name,names in protocol['combinations'].items():
        destination=STUDY/'hybrids'/name
        if (destination/'result.json').exists():
            existing=read(destination/'result.json');verify_hashes(existing['artifact_hashes'])
            if not existing.get('cpu_bundle_replay_verified'):raise ValueError('Incomplete saved hybrid')
            continue
        p_oof=[];p_tune=[]
        for family in names:
            chunks=[]
            for i in (0,1):
                record=registry[f'{family}_fold{i}'];verify_hashes(record['artifact_hashes'])
                run=ROOT/record['run_directory']
                np.testing.assert_array_equal(np.load(run/'oof_row_ids.npy'),train_ids[fold_ids[i]])
                chunks.append(np.load(run/'oof_probabilities.npy'))
            p_oof.append(np.concatenate(chunks))
            branch=final_branches[family]
            verify_hashes({branch['validation_probabilities']:branch['validation_sha256']})
            p_tune.append(np.load(ROOT/branch['validation_probabilities'])[tune])
        p_oof=np.stack(p_oof);p_tune=np.stack(p_tune)
        chosen,search=optimize_combiner(name,names,p_oof,y[np.concatenate(fold_ids)],
            p_tune,validation_y[tune],protocol)
        destination.mkdir(parents=True,exist_ok=True)
        write(destination/'optimizer_selection.json',search)
        old=read(PREVIOUS/'hybrids'/name/'result.json')
        write(destination/'previous_control.json',dict(full_validation_metrics=old['full_validation_metrics'],
            assessment=old['calibration']['before'],source=relative(PREVIOUS/'hybrids'/name/'result.json')))
        selected=deepcopy(chosen['combiner']);del p_oof,p_tune
        original=publisher.fit_blend
        def selected_blend(probabilities,labels,branch_names,penalty):
            if branch_names!=selected['names'] or len(labels)!=selected['fitting_rows']:
                raise ValueError('Selected combiner input identities differ')
            return deepcopy(selected)
        try:
            publisher.fit_blend=selected_blend
            publisher.finalize(name,names,dict(protocol,final_branches=final_branches,
                combiner_penalties=[chosen['penalty']]),registry,destination)
        finally:publisher.fit_blend=original
