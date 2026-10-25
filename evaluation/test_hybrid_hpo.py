"""Bayesian search must optimize the hybrid on aligned training-only rows."""
import numpy as np
import pytest


def base_recipes():
    from experiment.train_tabm import configurations as tabm_configs
    from experiment.train_verified_clean import configurations
    from framework.training.tabpack_backend import pack_configuration
    original=configurations()
    return dict(tabm_ne={k:tabm_configs()[0][k] for k in ('config','arm','seed')},
        xgboost={k:next(r for r in original if r['family']=='xgboost' and r['arm']=='full')[k] for k in ('config','arm','seed')},
        kan={k:next(r for r in original if r['family']=='kan' and r['arm']=='strict')[k] for k in ('config','arm','seed')},
        tabpack=dict(config=pack_configuration(),arm='full',seed=42))


def test_parameter_spaces_affect_effective_recipes():
    import optuna
    from framework.training.hybrid_hpo import sample_recipe
    recipes=base_recipes()
    for family in ('tabm_ne','xgboost','kan','tabpack'):
        study=optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=3))
        a=sample_recipe(study.ask(),family,recipes[family])
        b=sample_recipe(study.ask(),family,recipes[family])
        assert a!=b
        assert a['config']['runtime_seconds']>0
        assert a['seed']==42
        if family=='tabpack':
            assert len(a['config']['lr'])==a['config']['pack_size']==8


def test_tpe_request_recovery_is_exact(tmp_path):
    from framework.training.hybrid_hpo import trial_study, ask_recipe, tell_result
    base=base_recipes()['tabm_ne']
    first=ask_recipe(tmp_path,'tabm_ne',base)
    assert ask_recipe(tmp_path,'tabm_ne',base)==first
    tell_result(tmp_path,'tabm_ne',first['trial_number'],.8)
    second=ask_recipe(tmp_path,'tabm_ne',base)
    assert second['trial_number']!=first['trial_number']
    assert len(trial_study(tmp_path,'tabm_ne').trials)==2


def test_hybrid_objective_uses_meta_fit_and_meta_score_roles():
    from framework.training.hybrid_hpo import score_hybrids
    y=np.arange(60)%3
    good=np.full((60,3),.05);good[np.arange(60),y]=.9
    other=np.roll(good,1,axis=1)
    roles=np.repeat([0,1],30)
    result=score_hybrids({'tabm_ne':good,'xgboost':other},y,roles,
                         {'pair':['tabm_ne','xgboost']})
    assert result['mean_accuracy']==1.
    assert result['meta_fit_rows']==30 and result['meta_score_rows']==30
    assert result['objective_partition']=='original_train'


def test_meta_partition_purges_windows_and_duplicates():
    from framework.training.hybrid_hpo_data import meta_partition
    y=np.repeat(np.arange(3),50);starts=np.tile(np.arange(50)*10,3);ends=starts+5
    h=np.array([str(i) for i in range(150)]);h[40]=h[2]
    roles=meta_partition(y,starts,ends,h)
    assert roles[40]==-1
    assert not np.intersect1d(h[roles==0],h[roles==1]).size
    for c in range(3):
        assert ends[(y==c)&(roles==0)].max()<starts[(y==c)&(roles==1)].min()


def test_adoption_rejects_changed_recipe():
    from framework.training.hybrid_hpo_data import validate_adoption
    record={'config':{'lr':.001},'arm':'full','seed':42,'family':'tabm_ne'}
    validate_adoption(record,record)
    with pytest.raises(ValueError,match='recipe'):
        validate_adoption(dict(record,config={'lr':.002}),record)


def test_standalone_baseline_adoption_normalizes_and_pins_record(tmp_path,monkeypatch):
    from experiment import optimize_hybrids as runner
    from experiment.train_verified_clean import write
    from experiment.datasense_quality import digest
    from framework.training.hybrid_protocol import verify_hashes
    run=tmp_path/'old_run';run.mkdir()
    checkpoint=run/'model.pt';checkpoint.write_bytes(b'verified checkpoint')
    recipe=dict(family='tabm_ne',arm='full',seed=42,config={'lr':.001})
    record=dict(recipe,status='completed',checkpoint_reload_verified=True,test_loaded=False,
                artifact_hashes={'model.pt':digest(checkpoint)})
    write(run/'result.json',record)
    protocol=dict(recipes={'tabm_ne':{k:recipe[k] for k in ('arm','seed','config')}},
        previous_final_branches={'tabm_ne':{'checkpoint':'old_run/model.pt',
            'assets':{'old_run/model.pt':digest(checkpoint)}}},
        partner_hashes={'old_run/result.json':digest(run/'result.json')})
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'relative',lambda p:p.resolve().relative_to(tmp_path).as_posix())
    monkeypatch.setattr(runner,'verify_hashes',lambda pins:verify_hashes(pins,root=tmp_path))
    adopted=runner.adopt_refit('tabm_ne','full',{'recipe':recipe},protocol)
    assert adopted['run_directory']=='old_run'
    assert adopted['cpu_checkpoint_reload_verified']
    assert adopted['artifact_hashes']=={'old_run/model.pt':digest(checkpoint)}
    write(run/'result.json',dict(record,checkpoint_reload_verified=False))
    with pytest.raises(ValueError,match='Pinned asset changed'):
        runner.adopt_refit('tabm_ne','full',{'recipe':recipe},protocol)


def test_combiner_resume_preserves_failed_record(tmp_path,monkeypatch):
    import optuna
    from framework.training import hybrid_hpo_finalize as finalizer
    from framework.training.hybrid_hpo import trial_study
    from experiment.train_verified_clean import read,write
    monkeypatch.setattr(finalizer,'STUDY',tmp_path)
    monkeypatch.setattr(finalizer,'relative',lambda p:p.relative_to(tmp_path).as_posix())
    directory=tmp_path/'combiner_search/pair';directory.mkdir(parents=True)
    global_result=dict(combiner={},tuning_metrics=dict(accuracy=.8,macro_f1=.7,log_loss=.5),
                       objective=.8000007,penalty=None)
    write(directory/'global.json',global_result)
    study=trial_study(tmp_path,'combiner_pair');trial=study.ask()
    penalty=trial.suggest_float('penalty',.0001,1.,log=True)
    failure=dict(status='failed',trial_number=trial.number,penalty=penalty,error='bounded optimizer failed')
    write(directory/'trial_000.json',failure)
    chosen,metadata=finalizer.optimize_combiner('pair',['a','b'],None,None,None,None,{'combiner_trials':1})
    assert chosen==global_result
    assert metadata['trial_states']['FAIL']==1
    assert read(directory/'trial_000.json')==failure
    assert trial_study(tmp_path,'combiner_pair').trials[0].state==optuna.trial.TrialState.FAIL


def test_completion_detects_reused_controller_pid(tmp_path,monkeypatch):
    from experiment import optimize_hybrids as runner
    from experiment.train_verified_clean import write
    import psutil
    write(tmp_path/'state.json',dict(status='optimizing_branch',pid=123,launch_id='original'))
    monkeypatch.setattr(runner,'STUDY',tmp_path)
    class OtherProcess:
        def __init__(self,pid):pass
        def cmdline(self):return ['python','-m',runner.MODULE,'_run','--launch-id','different']
    monkeypatch.setattr(psutil,'Process',OtherProcess)
    summary=runner.inspect_completion()
    assert summary['event']=='attention'
    assert 'exited' in summary['message']
