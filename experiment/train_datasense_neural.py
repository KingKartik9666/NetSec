"""Fixed-epoch genuine KAN / upstream-derived HEM on the audited copy.

No validation stopping or model selection. CUDA is an explicit research option;
the default CPU execution and the deployed application remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np

from experiment.audit_datasense_raw import DEFAULT
from experiment.datasense_quality import digest,write_json
from experiment.train_datasense_quality import probability_metrics
from framework.training.tracking import classification_metrics


def fit(family,directory=DEFAULT,candidate='04_port_refinement',device='cpu',epochs=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from framework.training.neural import (_fit_preprocessing,_models,_seed,class_weights,
        cuda_knn_predict,hem_closed_set_vote,HEM_SHA,KAN_SHA)
    from framework.training.research_losses import multiclass_loss,binary_loss
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Explicit CUDA request unavailable; no fallback')
    _seed(torch,42)
    prepared=directory/candidate
    manifest=json.loads((prepared/'manifest.json').read_text())
    out=directory/'runs'/f'{family}_all_records_s42_{candidate}'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    X=np.load(prepared/'X_train.npy',mmap_mode='r');V=np.load(prepared/'X_validation.npy',mmap_mode='r')
    y=np.load(prepared/'y_train.npy');v=np.load(prepared/'y_validation.npy')
    n=len(manifest['classes']);benign=manifest['benign_index']
    config={'scaling':'log_minmax','hidden':[96,48] if family=='kan' else [512,256,32,16],
            'binary_hidden':[1024,256,64,16],'pca_components':64,'knn_k':1,
            'grid_size':5,'spline_order':3,'batch_size':512,'lr':.002 if family=='kan' else .001,
            'weight_decay':.001,'weight_strategy':'sqrt_balanced' if family=='kan' else 'none',
            'epochs':epochs or (24 if family=='kan' else 12)}
    if family=='hem':config.update(pca_solver='randomized',pca_iterated_power=2,preprocessing_execution='chunked all-training statistics')
    request={'family':family,'candidate':candidate,'arm':'all_records','seed':42,
             'config':config,'device':device,'training_rows':len(y),'validation_rows':len(v),
             'feature_count':X.shape[1],'candidate_manifest_sha256':digest(prepared/'manifest.json'),
             'test_loaded':False,'validation_model_selection':False,
             'selection':'Final fixed epoch; validation telemetry never changes fitting.',
             'scope':'Genuine upstream efficient-KAN; HEM-derived closed-set adaptation, not original unknown-attack protocol.'}
    write_json(out/'request.json',request)
    started=time.monotonic()
    if family=='hem':
        from experiment.datasense_preprocessing import fit_streamed_preprocessing
        preprocessing,train,val=fit_streamed_preprocessing(family,X,V,config,42,out)
    else:
        preprocessing,train,val=_fit_preprocessing(family,X,V,config,42)
    joblib.dump(preprocessing,out/'preprocessing.joblib')
    request['transformed_feature_count']=train.shape[1]
    if family=='hem':request['pca_explained_variance_ratio_sum']=float(preprocessing.pca.explained_variance_ratio_.sum())
    write_json(out/'request.json',request)
    models=_models(family,train.shape[1],n,config,torch,device=device)
    weights=torch.as_tensor(class_weights(y,n,config['weight_strategy']),device=device)
    binary_y=(y!=benign).astype(np.int64)
    bw=torch.as_tensor(class_weights(binary_y,2,'none'),device=device)
    parameters=[p for model in models.values() for p in model.parameters()]
    optimizer=torch.optim.AdamW(parameters,lr=config['lr'],weight_decay=config['weight_decay'])
    rng=np.random.default_rng(42);history=[]
    for epoch in range(1,config['epochs']+1):
        for model in models.values():model.train()
        loss_sum=0.;order=rng.permutation(len(y))
        for start in range(0,len(y),config['batch_size']):
            ids=order[start:start+config['batch_size']]
            batch=torch.as_tensor(train[ids],device=device)
            target=torch.as_tensor(y[ids],dtype=torch.long,device=device)
            optimizer.zero_grad(set_to_none=True)
            if family=='kan':
                loss=multiclass_loss(models['kan'](batch),target,weights,loss='native_ce')
            else:
                bt=torch.as_tensor(binary_y[ids],dtype=torch.long,device=device)
                loss=multiclass_loss(models['multiclass'](batch),target,weights,loss='native_ce')
                loss+=binary_loss(models['binary'](batch)[:,0],bt,bw,loss='native_ce')
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,5.);optimizer.step()
            loss_sum+=float(loss.detach())*len(ids)
        entry={'epoch':epoch,'train_loss':loss_sum/len(y),'elapsed_seconds':time.monotonic()-started}
        history.append(entry);write_json(out/'progress.json',entry)
        print(f'{family} fixed epoch {epoch}/{config["epochs"]}: loss={entry["train_loss"]:.5f}, elapsed={entry["elapsed_seconds"]:.1f}s',flush=True)
    checkpoint=out/'model.pt'
    torch.save({'candidate':family,'config':config,'seed':42,'input_size':train.shape[1],
                'n_classes':n,'benign_index':benign,'upstream_commit':KAN_SHA if family=='kan' else HEM_SHA,
                'state_dicts':{name:{k:t.detach().cpu() for k,t in model.state_dict().items()} for name,model in models.items()}},checkpoint)
    def output(matrix):
        values={name:[] for name in models}
        for model in models.values():model.eval()
        with torch.no_grad():
            for start in range(0,len(matrix),512):
                batch=torch.as_tensor(matrix[start:start+512],device=device)
                for name,model in models.items():values[name].append(model(batch).cpu().numpy())
        return {name:np.concatenate(v) for name,v in values.items()}
    logits=output(val)
    if family=='kan':
        z=logits['kan'];p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
        predicted=p.argmax(1);np.save(out/'validation_probabilities.npy',p)
        metrics=probability_metrics(v,p,manifest['classes'],benign)
    else:
        # Authentic references only, preserving every supplied training observation.
        np.savez(out/'knn_references.npz',X=train,y=y,train_indices=np.arange(len(y)))
        print(f'HEM exact KNN: all {len(y):,} original training references',flush=True)
        knn=cuda_knn_predict(val,train,y,n,1,batch_size=64,device=device)
        multi=logits['multiclass'].argmax(1);binary=(logits['binary'][:,0]>=0).astype(np.int64)
        predicted=hem_closed_set_vote(multi,binary,knn,benign)
        np.savez(out/'validation_components.npz',multiclass_logits=logits['multiclass'],
                 binary_logits=logits['binary'][:,0],knn_predictions=knn)
        metrics=classification_metrics(v,predicted,manifest['classes'],benign)
        metrics.update(top_three_accuracy=None,confidence_note='Hard voting has no multiclass probability API; component probabilities are not ensemble probabilities.')
    np.save(out/'validation_predictions.npy',predicted)
    from experiment.train_csv_models import reload_outputs
    count=len(v) if family=='kan' else min(128,len(v))
    cpu,p_cpu=reload_outputs(family,checkpoint,V[:count])
    np.testing.assert_array_equal(cpu,predicted[:count])
    if family=='kan':np.testing.assert_allclose(p_cpu,p,atol=2e-6,rtol=1e-5)
    result={**request,'status':'completed','metrics':metrics,'elapsed_seconds':time.monotonic()-started,
            'checkpoint_sha256':digest(checkpoint),'checkpoint_reload_verified':True,
            'reload_device':'cpu','reload_rows':count,'history':history,
            'parameter_devices':sorted({str(p.device) for p in parameters}),
            'knn_reference_rows':len(y) if family=='hem' else None,
            'research_status':'Exploratory within-capture validation; no locked test or live-model promotion.'}
    write_json(out/'result.json',result)
    print(json.dumps({k:metrics[k] for k in ('accuracy','macro_f1','macro_recall','false_positive_rate')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=['kan','hem'],required=True)
    parser.add_argument('--directory',type=Path,default=DEFAULT)
    parser.add_argument('--candidate',choices=['03_candidate','04_port_refinement'],default='04_port_refinement')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--epochs',type=int)
    args=parser.parse_args();fit(args.family,args.directory,args.candidate,args.device,args.epochs)
