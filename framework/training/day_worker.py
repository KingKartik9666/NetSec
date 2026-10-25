"""Authorized training-only workers and CPU inference for the day study."""
from __future__ import annotations

import gc
from pathlib import Path
import time

import numpy as np

from experiment.datasense_quality import digest
from experiment.train_verified_clean import training_paths
from framework.training.hybrid_protocol import ROOT, DATA, relative, verify_hashes
from framework.training.study_supervisor import atomic_write, read
from framework.training.tracking import classification_metrics


def role_indices(roles, phase):
    if phase not in ('search','repeat','oof','finalize'):
        raise ValueError('Unsupported role; no held-out test access')
    return np.flatnonzero(np.asarray(roles) == 0)


def validate_request(request, study):
    phase=request['phase']
    if request['family']=='random_forest':
        raise ValueError('RF is fixed and cannot enter tuning')
    if phase=='search':
        from framework.training.day_search import load_study
        trial=next(t for t in load_study(study,request['family']).trials if t.number==request['trial_number'])
        if trial.user_attrs['request']!={k:v for k,v in request.items() if k!='study'}:
            raise ValueError('Worker request differs from persistent Optuna proposal')
    elif phase in ('repeat','oof'):
        selected=read(study/'selected.json')[request['family']]
        if any(request[k]!=selected[k] for k in ('family','arm','config')):
            raise ValueError('Refit differs from frozen selected recipe')
        if phase=='oof' and (request['fold'] not in (0,1) or request['seed']!=selected['seed']):
            raise ValueError('Invalid OOF role or seed')
        if phase=='repeat' and request['seed'] not in (43,44):
            raise ValueError('Invalid repeat seed')
    else:
        raise ValueError('Unknown fit phase')


def fitting_rows(study,fold,arm):
    if fold not in (0,1) or arm not in ('full','strict'):
        raise ValueError('Unknown training fold or arm')
    folder=Path(study)/f'folds/{fold}'
    ids=np.load(folder/('fit_strict.npy' if arm=='strict' else 'fit.npy'))
    early=np.load(folder/'early_stop.npy');oof=np.load(folder/'oof.npy')
    if np.intersect1d(ids,early).size or np.intersect1d(ids,oof).size or np.intersect1d(early,oof).size:
        raise ValueError('Fitting, early-stop and OOF rows overlap')
    return ids


def source_checkpoint(result, run):
    named = result.get('checkpoint_path')
    if named:
        path = Path(named)
        return path if path.is_absolute() else ROOT/path
    return run/('model.joblib' if result['family']=='random_forest' else
                'model.ubj' if result['family']=='xgboost' else 'model.pt')


def cpu_neighbors(X,references,labels,n_classes,k,batch_size):
    from framework.training.neural import cuda_knn_predict
    return cuda_knn_predict(X,references,labels,n_classes,k,batch_size,device='cpu')


def hem_cpu_outputs(checkpoint,X):
    import joblib
    import torch
    from scipy.special import softmax,expit
    from framework.training.neural import _models,_clean,hem_closed_set_vote
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    transformed=joblib.load(checkpoint.parent/'preprocessing.joblib').transform(_clean(X)).astype(np.float32)
    models=_models('hem',saved['input_size'],saved['n_classes'],saved['config'],torch,device='cpu')
    logits={}
    for name,model in models.items():
        model.load_state_dict(saved['state_dicts'][name]);model.eval()
        with torch.inference_mode():
            logits[name]=np.concatenate([model(torch.from_numpy(transformed[i:i+256])).numpy()
                for i in range(0,len(X),256)])
    multi=softmax(logits['multiclass'],axis=1);binary=expit(logits['binary'][:,0])
    with np.load(checkpoint.parent/'knn_references.npz') as refs:
        knn=cpu_neighbors(transformed,refs['X'],refs['y'],saved['n_classes'],
            saved['config']['knn_k'],saved['config']['knn_batch_size'])
    pred=hem_closed_set_vote(multi.argmax(1),binary>=.5,knn,saved['benign_index'])
    return pred,dict(multiclass_probabilities=multi,binary_probabilities=binary,knn_predictions=knn)


def predict_saved(record, X):
    import torch
    torch.set_num_threads(2)
    family = record['family']
    checkpoint = ROOT/record['checkpoint']
    if family == 'random_forest':
        import joblib
        return joblib.load(checkpoint).predict_proba(X)
    if family in ('tabm_ne','tabpack'):
        from framework.training.hybrid_neural import load, predict
        model, prep, saved, weights = load(checkpoint)
        p = predict(model, prep, X, family, weights)
    elif family.startswith('tabm_'):
        from framework.training.tabm_fit import load_checkpoint, predict
        model, prep, saved = load_checkpoint(checkpoint)
        p = predict(model, prep, X)
    else:
        from experiment.train_csv_models import reload_outputs
        if family == 'hem':
            return hem_cpu_outputs(checkpoint,X)
        _, p = reload_outputs(family, checkpoint, X)
    gc.collect()
    return p


def fit(request_path):
    from framework.training import tracking
    tracking.write_json=atomic_write
    request_path = Path(request_path)
    request = read(request_path)
    output = request_path.parent
    study = ROOT/request['study']
    protocol = read(study/'study.json')
    validate_request(request,study)
    verify_hashes(protocol['source_hashes'])
    verify_hashes(protocol['input_hashes'])
    verify_hashes(protocol['role_hashes'])
    manifest = read(DATA/'model/manifest.json')
    family = request['family']
    roles = np.load(study/'validation_roles.npy')
    paths = training_paths(DATA, request['arm'])
    X, y = [np.load(paths[k], mmap_mode='r') for k in ('X_train','y_train')]
    V, v = [np.load(paths[k], mmap_mode='r') for k in ('X_validation','y_validation')]
    tune = role_indices(roles, request['phase'])
    if request['phase'] == 'oof':
        from framework.training.hybrid_data import IndexedRows
        fold = request['fold']
        full_X = np.load(DATA/'model/X_train.npy', mmap_mode='r')
        full_y = np.load(DATA/'model/y_train.npy', mmap_mode='r')
        fit_ids = fitting_rows(study,fold,request['arm'])
        early = np.load(study/f'folds/{fold}/early_stop.npy')
        X = IndexedRows(full_X,fit_ids) if family.startswith('tabm') or family=='tabpack' else full_X[fit_ids]
        y = full_y[fit_ids]
        tune_X, tune_y = full_X[early], full_y[early]
    else:
        tune_X, tune_y = V[tune], v[tune]
    started = time.monotonic()
    print(f'TRAIN {family} {request["phase"]} rows={len(y)} tuning={len(tune_y)} CUDA', flush=True)
    if family == 'tabpack':
        from framework.training.hybrid_neural import train
        details = train(family,X,y,tune_X,tune_y,manifest,request['config'],request['seed'],output,'cuda')
        checkpoint = output/'model.pt'
    elif family.startswith('tabm_'):
        from framework.training.tabm_fit import train
        model, prep, details = train(X,y,tune_X,tune_y,manifest,request,output,'cuda')
        del model, prep
        checkpoint = output/'model.pt'
    else:
        from experiment.train_csv_models import train_candidate
        details = train_candidate(family,X,y,tune_X,tune_y,len(manifest['classes']),0,
            request['config'],request['seed'],output)
        details.pop('predictions',None)
        details.pop('probabilities',None)
        details.pop('history',None)
        checkpoint = Path(details.pop('checkpoint_path'))
    record = dict(request,checkpoint=relative(checkpoint))
    prediction = predict_saved(record, tune_X)
    pred = prediction[0] if family=='hem' else prediction.argmax(1)
    score = classification_metrics(tune_y,pred,manifest['classes'],0)
    # Search exports only its tuning predictions/metrics, never assessment results.
    np.save(output/'tuning_predictions.npy',pred)
    if family != 'hem':
        np.save(output/'tuning_probabilities.npy',prediction)
    if request['phase']=='oof':
        ids = np.load(study/f'folds/{request["fold"]}/oof.npy')
        oof_X = full_X[ids]
        p = predict_saved(record, oof_X)
        np.save(output/'oof_probabilities.npy',p)
        np.save(output/'oof_row_ids.npy',np.load(DATA/'model/row_ids_train.npy')[ids])
    # Same-device, two independent reloads; numerical GPU checks stay in backend.
    sample = tune_X[:128]
    first, second = predict_saved(record,sample), predict_saved(record,sample)
    if family=='hem':
        np.testing.assert_array_equal(first[0],second[0])
    else:
        np.testing.assert_allclose(first,second,atol=0,rtol=0)
    import psutil
    memory = psutil.Process().memory_info()
    result = dict(record,status='completed',run_directory=relative(output),selection_metrics=score,details=details,
        training_rows=len(y),tuning_rows=len(tune_y),elapsed_seconds=time.monotonic()-started,
        cpu_checkpoint_reload_verified=True,cpu_replay_rows=len(sample),
        peak_working_set_bytes=getattr(memory,'peak_wset',memory.rss),test_loaded=False,
        assessment_used_for_fitting_or_selection=False,
        artifact_hashes={relative(p):digest(p) for p in output.iterdir()
            if p.suffix in ('.pt','.joblib','.ubj','.npy','.npz')})
    atomic_write(output/'result.json',result)
    print(f'COMPLETE {family} tuning_accuracy={score["accuracy"]:.6f}',flush=True)
