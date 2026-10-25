"""Leakage and probability contracts for the three requested hybrids."""
import json
import numpy as np
import pytest


def test_forward_folds_purge_windows_and_duplicate_vectors():
    from framework.training.hybrid_data import forward_folds
    labels = np.repeat(np.arange(3), 100)
    starts = np.tile(np.arange(100) * 10, 3)
    ends = starts + 5
    hashes = np.array([f'{i:064d}' for i in range(300)])
    hashes[65] = hashes[10]
    folds = forward_folds(labels, starts, ends, hashes)
    for f in folds:
        for a, b in [('fit', 'early_stop'), ('early_stop', 'oof')]:
            assert not np.intersect1d(f[a], f[b]).size
            assert not np.intersect1d(hashes[f[a]], hashes[f[b]]).size
            for c in range(3):
                assert ends[f[a]][labels[f[a]] == c].max() < starts[f[b]][labels[f[b]] == c].min()
        assert not np.intersect1d(hashes[f['fit']], hashes[f['oof']]).size
    assert 65 not in folds[0]['oof']
    assert not np.intersect1d(folds[0]['oof'], folds[1]['oof']).size


@pytest.mark.parametrize('scaling',['log_robust','quantile'])
def test_indexed_training_rows_preserve_exact_preprocessing_and_batches(scaling):
    from framework.training.hybrid_data import IndexedRows
    from framework.training.tabm_data import TabMPreprocessor
    X=np.random.default_rng(2).normal(size=(50,3)).astype('float32');X[::4,1]=np.nan
    ids=np.array([20,1,34,7,19,9,4,33,21,10,3,5,8,40,11])
    view=IndexedRows(X,ids); dense=X[ids]
    np.testing.assert_array_equal(view[:,1],dense[:,1])
    np.testing.assert_array_equal(view[[4,1,3]],dense[[4,1,3]])
    prep=TabMPreprocessor(scaling,8).fit(view,['a','b','c'])
    wanted=TabMPreprocessor(scaling,8).fit(dense,['a','b','c'])
    assert prep.to_dict()==wanted.to_dict()
    np.testing.assert_array_equal(prep.transform(view[:5]),wanted.transform(dense[:5]))


def test_blend_discards_wrong_branch_and_roundtrips():
    from framework.training.hybrid_blend import fit_blend, blend
    y = np.array([0, 1] * 30)
    p = np.where(np.arange(2)[None, :] == y[:, None], .9, .1)
    cfg = fit_blend(np.stack([p, p[:, ::-1]]), y, ['good', 'bad'])
    assert cfg['weights'][0] > .99
    q = blend(np.stack([p, p[:, ::-1]]), json.loads(json.dumps(cfg)), ['good', 'bad'])
    np.testing.assert_array_equal(q.argmax(1), y)
    np.testing.assert_allclose(q.sum(1), 1)
    with pytest.raises(ValueError, match='order'):
        blend(np.stack([p, p]), cfg, ['bad', 'good'])


def test_class_blend_is_finite_with_absent_class():
    from framework.training.hybrid_blend import fit_blend, blend
    rng = np.random.default_rng(4)
    p = rng.dirichlet(np.ones(3), size=(2, 60))
    cfg = fit_blend(p, np.arange(60) % 2, ['a', 'b'], penalty=.1)
    q = blend(p, cfg, ['a', 'b'])
    assert q.shape == (60, 3)
    assert np.isfinite(q).all()
    np.testing.assert_allclose(q.sum(1), 1, atol=1e-6)


@pytest.mark.parametrize('bad', [np.full((2, 5, 3), np.nan), np.ones((2, 5, 3)), np.full((2, 5, 3), -1.)])
def test_invalid_probabilities_rejected(bad):
    from framework.training.hybrid_blend import fit_blend
    with pytest.raises(ValueError):
        fit_blend(bad, np.zeros(5, dtype=int), ['a', 'b'])


def test_upstream_tabpack_forward_backward_and_serialization(tmp_path):
    import torch
    from framework.training.tabpack_backend import make_pack, pack_configuration, upstream_pins
    torch.set_num_threads(2)
    model, optimizer = make_pack(7, 3, pack_configuration(42), 'cpu')
    x = torch.randn(11, 7)
    y = torch.arange(11) % 3
    z = model(x, None)
    assert z.shape == (8, 11, 3)
    assert not torch.equal(z[0], z[1])
    loss = torch.nn.functional.cross_entropy(z.reshape(-1, 3), y.repeat(8))
    loss.backward()
    optimizer.step()
    model.eval()
    reference = model(x, None).detach()
    torch.save(model.state_dict(), tmp_path/'pack.pt')
    other, _ = make_pack(7, 3, pack_configuration(42), 'cpu')
    other.load_state_dict(torch.load(tmp_path/'pack.pt', weights_only=True))
    other.eval()
    torch.testing.assert_close(other(x, None), reference, rtol=0, atol=0)
    assert len(upstream_pins()) >= 4


def test_numeric_replay_reports_margin_changes_without_hiding_them():
    from framework.training.hybrid_neural import parity_diagnostics
    a = np.array([[.50001, .49999], [.1, .9]])
    b = np.array([[.49999, .50001], [.1, .9]])
    result = parity_diagnostics(a, b)
    assert result['changed_decisions'] == 1
    assert result['max_absolute_difference'] < .0001
    assert result['changed_reference_margins'][0] < .0001


def test_changed_asset_is_rejected(tmp_path):
    from framework.training.hybrid_protocol import verify_hashes
    from experiment.datasense_quality import digest
    f = tmp_path/'checkpoint'; f.write_bytes(b'original')
    pins = {'checkpoint':digest(f)}
    verify_hashes(pins, tmp_path)
    f.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        verify_hashes(pins, tmp_path)


def test_upstream_source_change_is_rejected(tmp_path):
    from framework.training.tabpack_backend import PINS, upstream_pins
    for name in PINS:
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='source changed'):
        upstream_pins(tmp_path)


def test_strict_arm_rejects_class_loss():
    from framework.training.hybrid_data import strict_fit_indices
    y = np.array([0,0,1,1]); ids = np.arange(4)
    np.testing.assert_array_equal(strict_fit_indices(ids, ids, [0,2], y, 2), [0,2])
    with pytest.raises(ValueError, match='class coverage'):
        strict_fit_indices(ids, ids, [0,1], y, 2)


def test_bundle_rejects_incomplete_or_misaligned_schema(tmp_path):
    from framework.training.hybrid_bundle import HybridPredictor
    f = tmp_path/'bundle.json'
    f.write_text(json.dumps({'status':'pending'}))
    with pytest.raises(ValueError, match='not complete'):
        HybridPredictor(f)
    f.write_text(json.dumps({'status':'completed','combiner':{'names':['a','b']},'branches':{'b':{},'a':{}}}))
    with pytest.raises(ValueError, match='order'):
        HybridPredictor(f)


def test_completed_and_failed_attempts_do_not_restart(tmp_path, monkeypatch):
    from experiment import train_hybrids as runner
    monkeypatch.setattr(runner, 'STUDY', tmp_path)
    assert runner.needs_attempt('a')
    out = tmp_path/'runs/a/attempt_000'; out.mkdir(parents=True)
    (out/'interrupted.json').write_text('{}')
    assert runner.needs_attempt('a')
    (out/'failure.json').write_text('{}')
    assert not runner.needs_attempt('a')
    (out/'result.json').write_text('{}')
    assert not runner.needs_attempt('a')


def test_greedy_members_can_exclude_harmful_member():
    from framework.training.hybrid_blend import greedy_members
    y = np.array([0,1]*10)
    p = np.where(y[:,None] == np.arange(2)[None], .9, .1)
    w = greedy_members(np.stack([p, p[:,::-1]]), y)
    assert w == [1., 0.]


def _saved_bundle_fixture(tmp_path):
    from experiment.datasense_quality import digest
    schema = tmp_path/'schema.json'
    schema.write_text(json.dumps({'feature_names':['a','b'], 'classes':['benign','attack']}))
    checkpoint = tmp_path/'model'; checkpoint.write_bytes(b'fixture')
    branch = dict(family='xgboost', checkpoint=str(checkpoint), schema_manifest=str(schema),
                  assets={str(p):digest(p) for p in (schema,checkpoint)})
    bundle = dict(status='completed', branches={'one':branch,'two':branch},
        classes=['benign','attack'], feature_names=['a','b'],
        combiner={'names':['one','two'],'n_classes':2})
    path = tmp_path/'bundle.json'; path.write_text(json.dumps(bundle))
    result = tmp_path/'result.json'
    result.write_text(json.dumps(dict(status='completed',artifact_hashes={str(path):digest(path)})))
    return path,bundle,result


def test_bundle_requires_verified_completion_record(tmp_path):
    from framework.training.hybrid_bundle import HybridPredictor
    path, _, result = _saved_bundle_fixture(tmp_path)
    result.unlink()
    with pytest.raises(ValueError, match='completion'):
        HybridPredictor(path)


@pytest.mark.parametrize('field', ['classes','feature_names'])
def test_same_length_schema_permutation_is_rejected(tmp_path, field):
    from framework.training.hybrid_bundle import HybridPredictor
    from experiment.datasense_quality import digest
    path,bundle,result = _saved_bundle_fixture(tmp_path)
    bundle[field].reverse(); path.write_text(json.dumps(bundle))
    result.write_text(json.dumps(dict(status='completed',artifact_hashes={str(path):digest(path)})))
    with pytest.raises(ValueError, match='schema'):
        HybridPredictor(path)


def test_existing_hybrid_is_hash_checked_before_adoption(tmp_path, monkeypatch):
    from framework.training import hybrid_evaluation as ev
    from framework.training.hybrid_protocol import verify_hashes
    folder=tmp_path/'hybrids/pair'; folder.mkdir(parents=True)
    (folder/'result.json').write_text(json.dumps(dict(status='completed',artifact_hashes={'missing.npy':'bad'})))
    monkeypatch.setattr(ev,'STUDY',tmp_path)
    monkeypatch.setattr(ev,'completed_jobs',lambda: {})
    monkeypatch.setattr(ev,'verify_hashes',lambda pins: verify_hashes(pins,tmp_path))
    with pytest.raises((ValueError,FileNotFoundError)):
        ev.finalize_available({'combinations':{'pair':['a','b']}})


def test_start_releases_controller_lock_before_spawning(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from experiment import train_hybrids as runner
    from experiment.optimize_models import StudyLock
    monkeypatch.setattr(runner,'STUDY',tmp_path)
    monkeypatch.setattr(runner,'wait_for_controller',lambda *a,**k: None,raising=False)
    observed=[]
    def spawn(*args,**kwargs):
        with StudyLock(tmp_path/'controller.lock'):
            observed.append('child acquired controller lock')
        return SimpleNamespace(pid=99999999)
    monkeypatch.setattr(runner.subprocess,'Popen',spawn)
    runner.start()
    assert observed == ['child acquired controller lock']


def test_startup_reports_child_failure(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from experiment import train_hybrids as runner
    monkeypatch.setattr(runner,'STUDY',tmp_path)
    with pytest.raises(RuntimeError,match='startup'):
        runner.wait_for_controller(SimpleNamespace(poll=lambda:7),'launch-id',timeout=.1)


@pytest.mark.parametrize('fail_replay', [True,False])
def test_composite_publication_waits_for_replay(tmp_path,monkeypatch,fail_replay):
    from framework.training import hybrid_evaluation as ev
    from framework.training.hybrid_protocol import verify_hashes
    from experiment.train_verified_clean import write
    from experiment.datasense_quality import digest
    data=tmp_path/'data'; study=tmp_path/'study'
    (data/'model').mkdir(parents=True); study.mkdir()
    y=np.arange(60)%3; p=np.full((60,3),.1,dtype=np.float32); p[np.arange(60),y]=.8
    write(data/'model/manifest.json',{'classes':['benign','a','b'],'feature_names':['f'],'benign_index':0})
    for split in ('train','validation'):
        for kind,values in [('row_ids',np.arange(60)),('y',y),('X',np.arange(60)[:,None])]:
            np.save(data/f'model/{kind}_{split}.npy',values)
    np.save(study/'validation_roles.npy',np.repeat([0,1,2],20))
    write(study/'study.json',{'fixture':True})
    jobs={}; branches={}
    for i in (0,1):
        folder=study/'folds'/str(i); folder.mkdir(parents=True)
        np.save(folder/'oof.npy',np.arange(i*20,(i+1)*20))
    for family in ('a','b'):
        full=tmp_path/family; full.mkdir()
        np.save(full/'validation_probabilities.npy',p)
        branches[family]={'validation_probabilities':str(full/'validation_probabilities.npy'),
                         'validation_sha256':digest(full/'validation_probabilities.npy')}
        for i in (0,1):
            run=full/str(i); run.mkdir()
            ids=np.arange(i*20,(i+1)*20)
            np.save(run/'oof_row_ids.npy',ids); np.save(run/'oof_probabilities.npy',p[ids])
            jobs[f'{family}_fold{i}']={'run_directory':str(run)}
    class Replay:
        def __init__(self,*args,**kwargs): pass
        def predict_proba(self,X,*args,**kwargs):
            if fail_replay: raise AssertionError('intentional replay failure')
            return p[X[:,0].astype(int)]
    for key,value in [('ROOT',tmp_path),('DATA',data),('STUDY',study),('HybridPredictor',Replay)]:
        monkeypatch.setattr(ev,key,value)
    monkeypatch.setattr(ev,'verify_hashes',lambda pins: verify_hashes(pins,tmp_path))
    protocol={'final_branches':branches,'combiner_penalties':[None], 'caveat':'software fixture'}
    destination=study/'hybrids/pair'
    if fail_replay:
        with pytest.raises(AssertionError,match='intentional replay'):
            ev.finalize('pair',['a','b'],protocol,jobs,destination)
        assert not (destination/'result.json').exists()
        assert json.loads((destination/'bundle.json').read_text())['status'] != 'completed'
    else:
        ev.finalize('pair',['a','b'],protocol,jobs,destination)
        result=json.loads((destination/'result.json').read_text())
        assert result['status']=='completed'
        verify_hashes(result['artifact_hashes'],tmp_path)
