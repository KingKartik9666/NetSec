"""Real TabM math, input compatibility and experiment boundary checks."""
import numpy as np
import pytest
import torch


def fixture_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(96, 5)).astype('float32')
    X[:, 1] = (X[:, 1] > 0).astype(float)
    X[::3, 1] = np.nan
    X[:, 2] = 7
    X[:, 3] = np.nan
    y = np.where(X[:, 0] < -.3, 0, np.where(X[:, 0] > .3, 2, 1))
    return X, y, ['signed_rate', 'protocol__present_tcp', 'constant', 'unknown', 'count']


@pytest.mark.parametrize('scaling', ['log_robust', 'quantile'])
def test_transform_preserves_unknown_flags_constants_and_schema(scaling):
    from framework.training.tabm_data import TabMPreprocessor
    X, _, names = fixture_data()
    p = TabMPreprocessor(scaling=scaling, n_bins=8).fit(X, names)
    z = p.transform(X)
    assert z.shape == (96, 10) and np.isfinite(z).all()
    assert np.array_equal(z[:, 6], np.isnan(X[:, 1]))
    assert 1 not in p.embedded_indices and 2 not in p.embedded_indices and 3 not in p.embedded_indices
    original = p.to_dict()
    altered = X[:2].copy(); altered[0, 0] = 1e20
    p.transform(altered)
    assert p.to_dict() == original
    with pytest.raises(ValueError, match='schema'):
        p.transform(X, names[::-1])
    q = TabMPreprocessor.from_dict(original)
    np.testing.assert_array_equal(q.transform(X), z)


def test_hierarchical_loss_equals_member_nll_and_handles_all_benign():
    from framework.training.tabm_models import member_loss, member_probabilities
    logits = torch.randn(8, 3, 4, requires_grad=True)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    p = member_probabilities(logits, hierarchical=True)
    expected = -p.gather(-1, y[:, None, None].expand(-1, 3, 1)).log().mean()
    torch.testing.assert_close(member_loss(logits, y, True), expected)
    loss = member_loss(logits, torch.zeros(8, dtype=torch.long), True)
    loss.backward()
    assert torch.isfinite(loss) and torch.count_nonzero(logits.grad[..., 1:]) == 0


def test_zero_inflated_counts_keep_nonzero_embedding_resolution():
    from framework.training.tabm_data import TabMPreprocessor
    values = np.concatenate([np.zeros(1000),np.arange(1,51)])[:,None]
    p = TabMPreprocessor(n_bins=16).fit(values,['sparse_count'])
    assert p.embedded_indices == [0] and len(p.bins[0]) > 10
    assert np.isfinite(p.transform(np.array([[0.],[1.],[1e20],[np.nan]]))).all()


def test_memberwise_products_are_composed_before_averaging():
    from framework.training.tabm_models import member_probabilities
    logits = torch.tensor([[[6., 6., -6.], [-6., -6., 6.]]])
    p = member_probabilities(logits, True).mean(1)
    torch.testing.assert_close(p.sum(1), torch.ones(1))
    wrong = logits[..., 0].sigmoid().mean(1) * logits[..., 1:].softmax(-1).mean(1)[:, 0]
    assert abs(float(p[0, 1] - wrong[0])) > .2


@pytest.mark.parametrize('family', ['tabm_ne', 'tabm_hier', 'tabm_hier_ne'])
def test_real_models_train_and_reload(family, tmp_path):
    from framework.training.tabm_data import TabMPreprocessor
    from framework.training.tabm_models import TabMClassifier, member_loss
    torch.manual_seed(42)
    torch.set_num_threads(2)
    X, y, names = fixture_data()
    p = TabMPreprocessor(n_bins=4).fit(X, names)
    config = dict(k=2, width=16, depth=2, dropout=0., d_embedding=4)
    m = TabMClassifier(family, p, 3, config)
    x = torch.from_numpy(p.transform(X))
    opt = torch.optim.AdamW(m.parameters(), lr=.01)
    before = member_loss(m(x), torch.from_numpy(y), m.hierarchical).item()
    for _ in range(5):
        opt.zero_grad(); loss = member_loss(m(x), torch.from_numpy(y), m.hierarchical)
        loss.backward(); opt.step()
    assert member_loss(m(x), torch.from_numpy(y), m.hierarchical).item() < before
    m.eval(); a = m.predict_proba(x)
    path = tmp_path / 'weights.pt'; torch.save(m.state_dict(), path)
    b = TabMClassifier(family, p, 3, config)
    b.load_state_dict(torch.load(path, weights_only=True)); b.eval()
    torch.testing.assert_close(a, b.predict_proba(x))


def test_recipe_selection_cannot_use_assessment_scores():
    from experiment.train_tabm import configurations, choose_best
    configs = configurations()
    assert len(configs) == 24 and {c['family'] for c in configs} == {'tabm_ne','tabm_hier','tabm_hier_ne'}
    rows = [dict(family='tabm_ne', phase='search', selection_metrics=dict(accuracy=.7,macro_f1=.6),
                 metrics=dict(accuracy=.99)),
            dict(family='tabm_ne', phase='search', selection_metrics=dict(accuracy=.8,macro_f1=.5),
                 metrics=dict(accuracy=.1))]
    assert choose_best(rows,'tabm_ne') is rows[1]


def test_calibration_parameters_ignore_assessment_labels_and_reload():
    from framework.training.tabm_calibration import fit_calibration, apply_calibration
    rng = np.random.default_rng(42)
    y = np.tile(np.arange(3), 60)
    roles = np.repeat(np.arange(3),60)
    p = rng.dirichlet([1,1,1],size=180)
    p[np.arange(180),y] += .7
    p /= p.sum(1,keepdims=True)
    manifest = dict(classes=['benign','a','b'],benign_index=0)
    c, after, binary = fit_calibration(p,y,roles,manifest)
    changed = y.copy(); changed[roles==2] = (changed[roles==2]+1)%3
    d, _, _ = fit_calibration(p,changed,roles,manifest)
    assert c['temperature']==d['temperature'] and c['binary_platt']==d['binary_platt']
    assert c['raw_attack_threshold']==d['raw_attack_threshold']
    replay, replay_binary = apply_calibration(p,c)
    np.testing.assert_array_equal(replay,after)
    np.testing.assert_array_equal(replay_binary,binary)
    np.testing.assert_array_equal(after.argmax(1),p.argmax(1))


def test_interrupted_attempt_can_resume_without_overwriting_failure(tmp_path,monkeypatch):
    from experiment import train_tabm as study
    monkeypatch.setattr(study,'STUDY',tmp_path)
    request = dict(recipe='model',seed=42)
    assert study.attempt_needed(request)
    attempt = tmp_path/'runs/model_s42/attempt_000';attempt.mkdir(parents=True)
    study.write(attempt/'interrupted.json',dict(reason='User stop'))
    assert study.attempt_needed(request)
    study.write(attempt/'failure.json',dict(reason='Numerical failure'))
    assert not study.attempt_needed(request)


@pytest.mark.parametrize('family',['tabm_ne','tabm_hier','tabm_hier_ne'])
def test_fit_loop_saves_reloadable_cpu_checkpoint(family,tmp_path):
    from experiment.train_tabm import configurations
    from framework.training.tabm_fit import train,predict,load_checkpoint
    X,y,names = fixture_data()
    request = next(c for c in configurations() if c['family']==family)
    request['config'].update(k=2,width=16,depth=2,epochs=2,batch_size=32,runtime_seconds=30,n_bins=4)
    manifest = dict(feature_names=names,classes=['benign','attack_a','attack_b'],benign_index=0)
    model,prep,result = train(X[:72],y[:72],X[72:],y[72:],manifest,request,tmp_path)
    assert result['checkpoint_reload_verified'] and result['epochs']==2
    reloaded,replay_prep,_ = load_checkpoint(tmp_path/'model.pt')
    np.testing.assert_array_equal(predict(model,prep,X),predict(reloaded,replay_prep,X))
