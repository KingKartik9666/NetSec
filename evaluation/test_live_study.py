import numpy as np
import pytest

from framework.training.live_study.protocol import atomic, exclusive, read
from framework.training.live_study.data import split_roles, valid_counts


def fixture():
    n=400
    x=np.column_stack([np.arange(n)+1,np.zeros((n,6))]).astype(np.float32)
    y=np.repeat([0,1],n//2)
    start=np.r_[np.arange(n//2),10000+np.arange(n//2)].astype(np.int64)*5_000_000_000
    return x,y,start,start+10_000_000_000


def test_split_purges_overlap_and_projected_duplicates():
    x,y,start,end=fixture();x[160]=x[10]
    roles=split_roles(x,y,start,end)
    assert roles[160]==-1
    for a in range(4):
        for b in range(a+1,4):
            left=np.flatnonzero(roles==a);right=np.flatnonzero(roles==b)
            assert not set(map(tuple,x[left])).intersection(map(tuple,x[right]))
            for label in (0,1):
                l=left[y[left]==label];r=right[y[right]==label]
                assert end[l].max()<start[r].min()
    np.testing.assert_array_equal(roles,split_roles(x,y,start,end))


def test_counts_reject_unmeasured_invalid_and_zero_windows():
    x=np.asarray([[5,3,0,0,0,2,0],[0,0,0,0,0,0,0],[2,3,0,0,0,0,0],
                  [1,.5,0,0,0,0,0],[3,np.nan,0,0,0,0,0]],np.float32)
    assert valid_counts(x).tolist()==[True,False,False,False,False]


def test_exclusive_controller_lock_and_atomic_state(tmp_path):
    with exclusive(tmp_path/'lock'):
        with pytest.raises(OSError):
            with exclusive(tmp_path/'lock'): pass
    atomic(tmp_path/'state.json',{'status':'running'})
    atomic(tmp_path/'state.json',{'status':'completed'})
    assert read(tmp_path/'state.json')=={'status':'completed'}
    assert not list(tmp_path.glob('*.tmp'))


def test_calibration_does_not_fit_on_tuning_targets():
    from framework.training.live_study.calibration import fit, apply
    p=np.array([[.9,.1],[.8,.2],[.6,.4],[.4,.6],[.2,.8],[.1,.9]])
    output={'probabilities':p,'pred':p.argmax(1)}
    truth=np.array([0,0,1,0,1,1])
    first=fit(output,truth,output,truth)
    second=fit(output,truth,output,1-truth)
    assert first['temperature']==second['temperature']
    assert first['binary_platt']==second['binary_platt']
    np.testing.assert_array_equal(apply(output,first)[2].argmax(1),output['pred'])


def test_persistent_proposal_reuses_running_trial(tmp_path,monkeypatch):
    from framework.training.live_study import search
    atomic(tmp_path/'study.json',{'smoke':True,'schema_fingerprint':'schema','protocol_fingerprint':'protocol'})
    monkeypatch.setattr(search,'relative',lambda path:str(path))
    _,trial,request=search.proposal(tmp_path,'random_forest')
    _,again,recovered=search.proposal(tmp_path,'random_forest')
    assert trial.number==again.number==0
    assert request==recovered
    assert len(search.load(tmp_path,'random_forest').trials)==1


def test_corrupted_checkpoint_fails_closed(tmp_path,monkeypatch):
    from framework.training.live_study import protocol
    from framework.training.live_study.worker import verify_result
    monkeypatch.setattr(protocol,'ROOT',tmp_path)
    checkpoint=tmp_path/'checkpoint';checkpoint.write_bytes(b'original checkpoint')
    atomic(tmp_path/'result.json',dict(status='completed',request_fingerprint='a',
        artifact_hashes={'checkpoint':protocol.digest(checkpoint)}))
    verify_result(tmp_path/'result.json',{'request_fingerprint':'a'})
    checkpoint.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError,match='changed'):verify_result(tmp_path/'result.json')


def test_stop_and_budget_do_not_start_more_work(tmp_path):
    import time
    from framework.training.live_study.controller import check_limits,Halt
    (tmp_path/'STOP').touch()
    with pytest.raises(Halt) as error:check_limits(tmp_path,{'deadline':time.time()+60})
    assert error.value.status=='stopped'
    (tmp_path/'STOP').unlink()
    with pytest.raises(Halt) as error:check_limits(tmp_path,{'deadline':time.time()-1})
    assert error.value.status=='budget_exhausted'


def test_owned_job_closes_parent_and_grandchild(tmp_path):
    import os,sys,time,subprocess,json
    import psutil
    from framework.training.live_study.processes import Job,launch,terminate
    if os.name!='nt':pytest.skip('Windows owned job contract')
    code="""import subprocess,sys,time,json
from pathlib import Path
folder=Path(sys.argv[1])
while not (folder/'ready').exists():time.sleep(.02)
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
(folder/'child.json').write_text(json.dumps({'pid':child.pid}))
time.sleep(30)
"""
    parent=None;child=None
    try:
        with (tmp_path/'log').open('w') as log:
            with Job() as job:
                parent=launch([sys.executable,'-c',code,str(tmp_path)],tmp_path,log)
                job.assign(parent);(tmp_path/'ready').touch()
                deadline=time.time()+10
                while not (tmp_path/'child.json').exists() and time.time()<deadline:time.sleep(.05)
                child=psutil.Process(json.loads((tmp_path/'child.json').read_text())['pid'])
                assert child.is_running()
                closed_at=time.monotonic()
            parent.wait(timeout=10)
            child.wait(timeout=10)
            # Job termination may report exit code zero on Windows. The contract
            # is prompt termination of both still-sleeping owned processes.
            assert time.monotonic()-closed_at<10
            assert parent.poll() is not None
    finally:
        if parent:terminate(parent)
        if child and child.is_running():child.kill()


def test_class_blend_retains_global_weights_for_absent_classes():
    from framework.training.live_study.combiner import fit
    from framework.training.hybrid_blend import blend
    rng=np.random.default_rng(42)
    p=rng.dirichlet([2,2,1],size=(2,80))
    y=np.tile([0,1],40)
    config=fit(p,y,['a','b'],1.)
    np.testing.assert_allclose(np.array(config['weights'])[:,2],config['global_weights'])
    assert config['classwise_supported_indices']==[0,1]
    assert np.isfinite(blend(p,config,['a','b'])).all()


def test_terminate_redirector_also_stops_its_child(tmp_path):
    import sys,time,json,psutil
    from framework.training.live_study.processes import launch,terminate
    code="""import subprocess,sys,time,json
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
Path(sys.argv[1]).write_text(json.dumps({'pid':child.pid}))
time.sleep(30)
"""
    parent=None;child=None
    try:
        with (tmp_path/'log').open('w') as log:
            parent=launch([sys.executable,'-c',code,str(tmp_path/'child.json')],tmp_path,log)
            deadline=time.time()+10
            while not (tmp_path/'child.json').exists() and time.time()<deadline:time.sleep(.05)
            child=psutil.Process(json.loads((tmp_path/'child.json').read_text())['pid'])
            terminate(parent)
            child.wait(timeout=5)
            assert parent.poll() is not None
    finally:
        if parent:terminate(parent)
        if child and child.is_running():child.kill()


def test_worker_self_join_survives_redirector_race(tmp_path):
    import os,sys,time,json,psutil
    from framework.training.live_study.processes import Job,launch,terminate
    from framework.training.live_study.protocol import ROOT
    if os.name!='nt':pytest.skip('Windows named job contract')
    code="""import subprocess,sys,time,json,os
from pathlib import Path
from framework.training.live_study.processes import join_current
join_current(sys.argv[1])
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
Path(sys.argv[2]).write_text(json.dumps({'pid':child.pid,'parent':os.getpid()}))
time.sleep(30)
"""
    process=None;child=None
    try:
        with (tmp_path/'log').open('w') as log:
            with Job() as job:
                # Deliberately do not assign the redirector: the real interpreter
                # must join itself before starting any child work.
                process=launch([sys.executable,'-c',code,job.name,str(tmp_path/'child.json')],ROOT,log)
                deadline=time.time()+10
                while not (tmp_path/'child.json').exists() and time.time()<deadline:time.sleep(.05)
                record=json.loads((tmp_path/'child.json').read_text())
                child=psutil.Process(record['pid']);psutil.Process(record['parent']).kill()
                assert child.is_running()
            child.wait(timeout=5)
            process.wait(timeout=5)
    finally:
        if process:terminate(process)
        if child and child.is_running():child.kill()
