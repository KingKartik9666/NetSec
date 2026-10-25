"""Durable studies must survive sharing locks without losing their deadline."""
import ctypes
import json
import os
import threading
import time

import pytest


@pytest.mark.skipif(os.name != 'nt', reason='Real Windows sharing violation')
def test_json_replacement_survives_temporary_windows_reader(tmp_path):
    from framework.training.study_supervisor import atomic_write
    path = tmp_path/'state.json'
    path.write_text('{"old": true}', encoding='utf-8')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                  ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0, None)
    assert handle not in (None, ctypes.c_void_p(-1).value)
    release = threading.Timer(.15, lambda: kernel.CloseHandle(handle))
    release.start()
    try:
        atomic_write(path, {'new': True})
    finally:
        release.join()
    assert json.loads(path.read_text()) == {'new': True}
    assert not list(tmp_path.glob('*.tmp'))


def test_json_rejects_nonfinite_without_replacing_valid_state(tmp_path):
    from framework.training.study_supervisor import atomic_write
    path = tmp_path/'state.json'
    atomic_write(path, {'valid': 1})
    with pytest.raises(ValueError):
        atomic_write(path, {'bad': float('nan')})
    assert json.loads(path.read_text()) == {'valid': 1}


def test_deadline_and_stage_boundaries_do_not_reset_on_restart():
    from framework.training.study_supervisor import phase_at
    protocol = {'started_at': 100., 'deadline': 100.+86400}
    assert phase_at(protocol, 101.) == 'search'
    assert phase_at(protocol, 100.+15*3600) == 'repeats'
    assert phase_at(protocol, 100.+18*3600) == 'oof'
    assert phase_at(protocol, 100.+22*3600) == 'finalize'
    assert phase_at(protocol, protocol['deadline']) == 'expired'


def test_scheduler_gives_every_family_coverage_before_exploitation():
    from framework.training.study_supervisor import next_family, FAMILIES
    observations = {f: [] for f in FAMILIES}
    for _ in range(3*len(FAMILIES)):
        family = next_family(observations)
        observations[family].append({'value': .8, 'seconds': 50})
    assert {len(v) for v in observations.values()} == {3}
    observations[FAMILIES[0]][-1]['value'] = .9
    assert next_family(observations) in FAMILIES


def test_resume_reuses_persisted_trial_request(tmp_path):
    from framework.training.day_search import ask_request, tell, load_study
    base = {'family': 'tabm_hier', 'arm': 'full', 'seed': 42, 'config': {
        'k':16,'width':256,'depth':3,'dropout':.1,'d_embedding':12,'n_bins':16,
        'scaling':'log_robust','lr':.002,'weight_decay':.0003,'weight_power':0.,
        'smoothing':0.,'subtype_weight':1.,'batch_size':512,'epochs':60,
        'patience':8,'runtime_seconds':1200}}
    first = ask_request(tmp_path, 'tabm_hier', base)
    assert ask_request(tmp_path, 'tabm_hier', base) == first
    tell(tmp_path, 'tabm_hier', first['trial_number'], .8)
    second = ask_request(tmp_path, 'tabm_hier', base)
    assert second['trial_number'] != first['trial_number']
    assert len(load_study(tmp_path, 'tabm_hier').trials) == 2


def test_role_access_never_opens_test_or_assessment_for_search():
    from framework.training.day_worker import role_indices
    import numpy as np
    roles = np.array([-1,0,1,2,0,2])
    np.testing.assert_array_equal(role_indices(roles, 'search'), [1,4])
    with pytest.raises(ValueError):
        role_indices(roles, 'test')


def test_accuracy_gain_cannot_silently_override_attack_recall_guard():
    from framework.training.day_finalize import select_record
    baseline = {'selection_metrics': {'accuracy':.8,'macro_f1':.7,'macro_recall':.7,
        'false_positive_rate':.01,'attack_missed_as_benign_rate':.02}}
    worse = {'selection_metrics':dict(baseline['selection_metrics'],accuracy=.9,
        attack_missed_as_benign_rate=.05)}
    good = {'selection_metrics':dict(baseline['selection_metrics'],accuracy=.82)}
    policy = {'max_fpr_increase':.005,'max_attack_miss_increase':.005,'max_macro_recall_drop':.01}
    assert select_record(baseline,[worse],policy) is baseline
    assert select_record(baseline,[worse,good],policy) is good


def test_expired_run_refuses_to_launch_another_worker(tmp_path):
    from experiment.optimize_all_models_day import run_worker
    with pytest.raises(TimeoutError):
        run_worker({'deadline':0}, {'job':'never','family':'kan'}, {}, 0)


def test_oof_uses_strict_fit_ids_and_never_self_predictions(tmp_path):
    from framework.training.day_worker import fitting_rows
    import numpy as np
    fold=tmp_path/'folds/0';fold.mkdir(parents=True)
    np.save(fold/'fit.npy',[0,1,2,3]);np.save(fold/'fit_strict.npy',[0,2])
    np.save(fold/'early_stop.npy',[4]);np.save(fold/'oof.npy',[5,6])
    np.testing.assert_array_equal(fitting_rows(tmp_path,0,'strict'),[0,2])
    np.testing.assert_array_equal(fitting_rows(tmp_path,0,'full'),[0,1,2,3])
    np.save(fold/'fit_strict.npy',[0,5])
    with pytest.raises(ValueError,match='overlap'):
        fitting_rows(tmp_path,0,'strict')


def test_saved_worker_request_rejects_parameter_tampering(tmp_path):
    from framework.training.day_worker import validate_request
    from framework.training.study_supervisor import atomic_write
    selected={'kan':{'family':'kan','arm':'strict','config':{'lr':.001}}}
    atomic_write(tmp_path/'selected.json',selected)
    request=dict(selected['kan'],phase='repeat',seed=43)
    validate_request(request,tmp_path)
    with pytest.raises(ValueError,match='differs'):
        validate_request(dict(request,config={'lr':.9}),tmp_path)


def test_successive_halving_prunes_an_inferior_measured_curve(tmp_path):
    import optuna
    from framework.training.day_search import load_study
    study=load_study(tmp_path,'tabm_hier')
    winner=study.ask();winner.report(.9,6)
    assert not winner.should_prune()
    study.tell(winner,.9)
    loser=study.ask();loser.report(.5,6)
    assert loser.should_prune()
    study.tell(loser,state=optuna.trial.TrialState.PRUNED)


def test_all_family_search_spaces_build_effective_model_configurations():
    from framework.training.day_search import sample
    from framework.training.study_supervisor import FAMILIES
    from experiment.train_tabm import configurations as tabm
    from experiment.train_verified_clean import configurations as originals
    from framework.training.tabpack_backend import pack_configuration
    from framework.training.tabm_data import TabMPreprocessor
    from framework.training.tabm_models import TabMClassifier
    import numpy as np
    import optuna
    import torch
    bases={}
    for family in FAMILIES:
        if family=='tabpack':
            row=dict(family=family,config=pack_configuration(),seed=42,arm='full')
        else:
            row=next(r for r in [*tabm(),*originals()] if r['family']==family)
        bases[family]=row
    for family,base in bases.items():
        trial=optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42)).ask()
        request=sample(trial,family,base)
        assert request['family']==family and request['seed']==42
        assert request['config']['runtime_seconds']==1500
        if family.startswith('tabm_'):
            prep=TabMPreprocessor('log_robust',8).fit(np.arange(240,dtype=np.float32).reshape(80,3),['a','b','c'])
            model=TabMClassifier(family,prep,4,request['config'])
            p=model.predict_proba(torch.from_numpy(prep.transform(np.ones((3,3),dtype=np.float32))))
            assert p.shape==(3,4)
            np.testing.assert_allclose(p.detach().numpy().sum(1),1,atol=2e-6)


def test_hem_calibration_tags_do_not_share_mutable_source_records(tmp_path):
    from framework.training.day_finalize import calibrate_hem
    import numpy as np
    y=np.tile([0,1,2],30);pred=y.copy();pred[::5]=(pred[::5]+1)%3
    p=np.full((90,3),.1);p[np.arange(90),pred]=.8
    parts=dict(multiclass_probabilities=p,binary_probabilities=1-p[:,0],knn_predictions=pred)
    roles=np.repeat([0,1,2],30)
    a=calibrate_hem(tmp_path/'selected',parts,pred,y,roles,'hash-a')
    b=calibrate_hem(tmp_path/'seed43',parts,pred,y,roles,'hash-b')
    assert a['source_checkpoint_sha256']=='hash-a'
    assert b['source_checkpoint_sha256']=='hash-b'
    assert a['assessment_used_for_fitting_or_selection'] is False
    assert a['calibration_rows']==b['calibration_rows']==30


def test_hem_neighbor_replay_explicitly_stays_on_cpu():
    from framework.training.day_worker import cpu_neighbors
    import numpy as np
    X=np.array([[0.,0.],[2.,2.]],dtype=np.float32)
    actual=cpu_neighbors(X,X,np.array([0,1]),2,1,1)
    np.testing.assert_array_equal(actual,[0,1])


def test_final_report_distinguishes_failed_exports_and_hybrid_baselines(tmp_path,monkeypatch):
    from framework.training import day_finalize as final
    from framework.training.study_supervisor import atomic_write
    monkeypatch.setattr(final,'relative',lambda p:str(p))
    families=('random_forest','xgboost','kan','hem','tabm_ne','tabm_hier','tabm_hier_ne')
    protocol=dict(baselines={f:{'metrics':{'accuracy':.8},'checkpoint':'original.pt'} for f in families},
        combinations={'tabm_xgboost':['tabm_ne','xgboost']},
        hybrid_baselines={'tabm_xgboost':{'accuracy':.81}},deadline=1,caveat='exploratory')
    folder=tmp_path/'finalization/tabm_xgboost';folder.mkdir(parents=True)
    atomic_write(folder/'failure.json',{'exit_code':1})
    rows=final.report(tmp_path,protocol)
    assert rows['tabm_xgboost']['status']=='failed'
    assert rows['tabm_xgboost']['baseline_accuracy']==.81
    assert 'job.log' in rows['tabm_xgboost']['failure_log']
    assert final.read(tmp_path/'report.json')['failures']


def test_supervisor_restarts_real_failed_children_with_a_bound(tmp_path,monkeypatch):
    import subprocess
    import sys
    from experiment import optimize_all_models_day as runner
    from framework.training.study_supervisor import atomic_write,read
    atomic_write(tmp_path/'study.json',{'deadline':time.time()+60})
    monkeypatch.setattr(runner,'STUDY',tmp_path)
    monkeypatch.setattr(runner,'verify',lambda _p:None)
    monkeypatch.setattr(runner,'protect_descendants',lambda:None)
    monkeypatch.setattr(runner.time,'sleep',lambda _seconds:None)
    children=[]
    def fail_child(*_args,**_kwargs):
        child=subprocess.Popen([sys.executable,'-c','raise SystemExit(7)'],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        children.append(child)
        return child
    monkeypatch.setattr(runner,'launch',fail_child)
    runner.supervise()
    assert len(children)==4
    assert all(child.poll()==7 for child in children)
    assert read(tmp_path/'supervisor.json')['status']=='needs_attention'
