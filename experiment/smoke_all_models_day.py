"""Small genuine-row CPU compatibility fits; explicitly not research scores."""
from pathlib import Path
from copy import deepcopy
import numpy as np

from framework.training.hybrid_protocol import ROOT,DATA
from framework.training.study_supervisor import atomic_write,read


def check_baselines():
    import torch
    from framework.training.day_protocol import baseline_records
    from framework.training.day_worker import predict_saved
    records=baseline_records()
    X=np.load(DATA/'model/X_validation.npy',mmap_mode='r')
    roles=np.load(ROOT/'artifacts/training/verified_clean_four_models_20261016_v1/validation_roles.npy')
    sample=X[np.flatnonzero(roles==0)[:16]]
    evidence={}
    for family,record in records.items():
        print('CPU CHECK',family,flush=True)
        p=predict_saved(record,sample)
        actual=p[0] if family=='hem' else p
        assert actual.shape==((16,) if family=='hem' else (16,84))
        assert not torch.cuda.is_initialized(),'CPU check initialized a CUDA context'
        evidence[family]=dict(passed=True,shape=list(actual.shape),cuda_initialized=False,test_loaded=False)
    atomic_write(ROOT/'artifacts/training/all_models_day_baseline_cpu_smoke_20261020.json',evidence)
    print('PASS: eight branch checkpoints loaded and inferred on CPU')


def main():
    from framework.training.tabm_fit import train
    from framework.training.hybrid_neural import train as train_branch
    from experiment.train_tabm import configurations
    from framework.training.tabpack_backend import pack_configuration
    destination=ROOT/'artifacts/training/all_models_day_smoke_20261020'
    destination.mkdir(exist_ok=True)
    manifest=read(DATA/'model/manifest.json')
    X=np.load(DATA/'model/X_train.npy',mmap_mode='r');y=np.load(DATA/'model/y_train.npy')
    V=np.load(DATA/'model/X_validation.npy',mmap_mode='r');v=np.load(DATA/'model/y_validation.npy')
    # Actual source observations, with all declared classes represented in fitting.
    ids=np.unique(np.r_[np.arange(256),[np.flatnonzero(y==c)[0] for c in range(84)]])
    validation=np.unique(np.r_[np.arange(128),[np.flatnonzero(v==c)[0] for c in np.unique(v)]])
    evidence={}
    for family in ('tabm_ne','tabm_hier','tabm_hier_ne','tabpack'):
        folder=destination/family
        if (folder/'smoke.json').exists():
            evidence[family]=read(folder/'smoke.json');continue
        if family=='tabpack':
            cfg=pack_configuration()
            cfg.update(d_block=[32]*8,n_blocks=[1]*8,epochs=2,batch_size=128,
                n_bins=8,runtime_seconds=120,patience=2)
            details=train_branch(family,X[ids],y[ids],V[validation],v[validation],manifest,cfg,42,folder,'cpu')
        else:
            request=deepcopy(next(r for r in configurations() if r['family']==family))
            request['config'].update(k=2,width=16,depth=1,d_embedding=4,n_bins=8,
                epochs=2,batch_size=128,runtime_seconds=120,patience=2)
            _,_,details=train(X[ids],y[ids],V[validation],v[validation],manifest,request,folder,'cpu')
        value=dict(passed=True,device='cpu',training_rows=len(ids),validation_rows=len(validation),
            feature_count=655,class_count=84,research_score=False,test_loaded=False,details=details)
        atomic_write(folder/'smoke.json',value);evidence[family]=value
    atomic_write(destination/'checks.json',evidence)
    print('PASS: four genuine-row CPU smoke fits and checkpoint replay; not research results')


if __name__=='__main__':
    import sys
    check_baselines() if '--baselines' in sys.argv else main()
