"""Finalization may reuse only compatible folds and must preserve CPU behavior."""
import numpy as np
import pytest


def test_early_stopped_fold_can_reuse_a_nonbinding_runtime_cap():
    from framework.training.finish_models import compatible_recipe
    requested=dict(family='xgboost',arm='strict',seed=42,config={'max_depth':9,'runtime_seconds':1500})
    old=dict(requested,config={'max_depth':9,'runtime_seconds':2400},
             details={'stop_reason':'early_stopping','training_seconds':1000})
    assert compatible_recipe(requested,old)
    assert not compatible_recipe(requested,dict(old,details={'stop_reason':'runtime_limit','training_seconds':1000}))
    assert not compatible_recipe(requested,dict(old,config={'max_depth':8,'runtime_seconds':2400}))
    assert not compatible_recipe(requested,dict(old,details={'stop_reason':'early_stopping','training_seconds':1600}))


def test_separated_encoding_matches_cpu_upstream_layout():
    import torch
    from rtdl_num_embeddings import _PiecewiseLinearEncodingImpl
    from framework.training.finish_models import separated_encoding
    impl=_PiecewiseLinearEncodingImpl([torch.tensor([0.,1.,1.000001,2.]),torch.tensor([0.,1.])])
    x=torch.tensor([[.2,.5],[1.0000005,2.]])
    torch.testing.assert_close(separated_encoding(impl,x),impl(x),rtol=0,atol=0)


def test_saved_fold_adoption_rejects_schema_or_recipe_mismatch():
    from framework.training.finish_models import validate_saved_fold
    request=dict(family='tabm_ne',arm='strict',seed=42,config={'width':384})
    saved=dict(family='tabm_ne',seed=42,config={'width':384},feature_names=['a'],classes=['benign','attack'])
    manifest=dict(feature_names=['a'],classes=['benign','attack'])
    validate_saved_fold(request,request,saved,manifest)
    with pytest.raises(ValueError):
        validate_saved_fold(request,request,dict(saved,feature_names=['other']),manifest)
    with pytest.raises(ValueError):
        validate_saved_fold(request,dict(request,arm='full'),saved,manifest)


def test_final_bundle_replays_cpu_predictions_and_rejects_column_reordering(tmp_path):
    import json,joblib
    from sklearn.ensemble import RandomForestClassifier
    from experiment.datasense_quality import digest
    from framework.training.model_bundle import FinalModelPredictor
    X=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]],dtype=np.float32)
    model=RandomForestClassifier(n_estimators=5,random_state=42).fit(X,[0,0,1,1])
    checkpoint=tmp_path/'model.joblib';joblib.dump(model,checkpoint)
    bundle=tmp_path/'bundle.json'
    bundle.write_text(json.dumps(dict(status='completed',feature_names=['a','b'],classes=['benign','attack'],
        record=dict(family='random_forest',checkpoint=str(checkpoint)),
        assets={str(checkpoint):digest(checkpoint)},calibration={'temperature':1.,'binary_platt':{'slope':1.,'intercept':0.}})))
    predictor=FinalModelPredictor(bundle)
    actual=predictor.predict(X,['a','b'])
    np.testing.assert_array_equal(actual['predictions'],model.predict(X))
    np.testing.assert_allclose(actual['probabilities'],model.predict_proba(X),atol=1e-7)
    with pytest.raises(ValueError,match='order'):
        predictor.predict(X,['b','a'])
