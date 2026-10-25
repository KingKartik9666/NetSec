"""Evidence-preserving OOF recovery and numerical audit for frozen models."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import time
import types

import numpy as np

from experiment.datasense_quality import digest
from framework.training.hybrid_protocol import ROOT,DATA,relative,verify_hashes
from framework.training.study_supervisor import atomic_write,read


def compatible_recipe(request,source):
    if any(request[k]!=source[k] for k in ('family','arm','seed')):
        return False
    a,b=deepcopy(request['config']),deepcopy(source['config'])
    if a==b:
        return True
    cap_a,cap_b=a.pop('runtime_seconds',None),b.pop('runtime_seconds',None)
    details=source.get('details',{})
    return (a==b and cap_a is not None and cap_b is not None
        and details.get('stop_reason')=='early_stopping'
        and 0<details.get('training_seconds',float('inf'))<min(cap_a,cap_b))


def separated_encoding(module,x):
    """Same encoding/layout, CPU-compatible operation order; no tolerance change."""
    import torch
    x=module.weight*x[...,None]
    x=x+module.bias
    if x.shape[-1]>1:
        x=torch.cat([x[...,:1].clamp_max(1.),x[...,1:-1].clamp(0.,1.),
            x[...,-1:].clamp_min(0.) if module.single_bin_mask is None else
            torch.where(module.single_bin_mask[...,None],x[...,-1:],x[...,-1:].clamp_min(0.))],dim=-1)
    return x


def cpu_order_on_cuda(model):
    if model.embeddings is not None:
        implementation=model.embeddings.impl
        implementation.forward=types.MethodType(separated_encoding,implementation)


def validate_saved_fold(request,source,saved,manifest):
    if any(request[k]!=source[k] for k in ('family','arm','seed','config')):
        raise ValueError('Failed source request does not match frozen recipe')
    if any(saved[k]!=request[k] for k in ('family','seed','config')):
        raise ValueError('Saved checkpoint differs from frozen recipe')
    if any(saved[k]!=manifest[k] for k in ('feature_names','classes')):
        raise ValueError('Saved checkpoint ordered schema differs')


def adopt_oof(study,source_path,request,fold,source_study):
    """Reuse real OOF only after identity, recipe, byte and row-order checks."""
    source=read(source_path)
    if source.get('status')!='completed' or not source.get('cpu_checkpoint_reload_verified'):
        raise ValueError('Source fold is not CPU-verified complete')
    if not compatible_recipe(request,source):
        raise ValueError('Source fold recipe is incompatible')
    verify_hashes(source['artifact_hashes'])
    for name in ('fit','fit_strict','early_stop','oof'):
        np.testing.assert_array_equal(np.load(study/f'folds/{fold}/{name}.npy'),
            np.load(source_study/f'folds/{fold}/{name}.npy'))
    ids=np.load(study/f'folds/{fold}/oof.npy')
    run=ROOT/source['run_directory']
    np.testing.assert_array_equal(np.load(run/'oof_row_ids.npy'),np.load(DATA/'model/row_ids_train.npy')[ids])
    p=np.load(run/'oof_probabilities.npy',mmap_mode='r')
    if p.shape!=(len(ids),84) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1.,atol=3e-6):
        raise ValueError('Invalid adopted OOF probabilities')
    source.update(adoption=dict(source_result=relative(source_path),source_result_sha256=digest(source_path),
        requested_recipe=request,original_config=source['config'],
        only_nonbinding_runtime_cap_differs=source['config']!=request['config'],fold_role_arrays_identical=True))
    atomic_write(study/'oof_registry'/f'{request["family"]}_fold{fold}.json',source)
    return source


def recover_tabm(study,source,request,output):
    """Recover the finished fit; keep its historical failed verification intact."""
    import torch
    from framework.training.tabm_fit import load_checkpoint,predict
    from framework.training.hybrid_neural import parity_diagnostics
    from framework.training.hybrid_data import IndexedRows
    from framework.training.day_worker import fitting_rows
    from framework.training.tracking import classification_metrics
    started=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    protocol=read(study/'study.json')
    verify_hashes(protocol['input_hashes']);verify_hashes(protocol['source_hashes']);verify_hashes(protocol['role_hashes'])
    source=Path(source);checkpoint=source/'model.pt'
    source_request=read(source/'request.json');manifest=read(DATA/'model/manifest.json')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    cpu,prep,saved=load_checkpoint(checkpoint)
    validate_saved_fold(request,source_request,saved,manifest)
    fitting_rows(study,request['fold'],request['arm'])
    for name in ('fit','fit_strict','early_stop','oof'):
        np.testing.assert_array_equal(np.load(study/f'folds/{request["fold"]}/{name}.npy'),
            np.load(ROOT/source_request['study']/f'folds/{request["fold"]}/{name}.npy'))
    X=np.load(DATA/'model/X_train.npy',mmap_mode='r');y=np.load(DATA/'model/y_train.npy')
    early=np.load(study/f'folds/{request["fold"]}/early_stop.npy')
    sample=X[early[:256]]
    gpu,gpu_prep,_=load_checkpoint(checkpoint,'cuda')
    native=predict(gpu,gpu_prep,sample,'cuda');reference=predict(cpu,prep,sample)
    original=parity_diagnostics(native,reference)
    cpu_order_on_cuda(gpu)
    aligned=predict(gpu,gpu_prep,sample,'cuda')
    np.testing.assert_allclose(aligned,reference,atol=5e-5,rtol=5e-4)
    np.testing.assert_array_equal(aligned.argmax(1),reference.argmax(1))
    # A larger deterministic sample excludes accidental passage on the first batch.
    audit_ids=early[np.linspace(0,len(early)-1,min(4096,len(early)),dtype=int)]
    a=predict(gpu,gpu_prep,X[audit_ids],'cuda');b=predict(cpu,prep,X[audit_ids])
    np.testing.assert_allclose(a,b,atol=5e-5,rtol=5e-4)
    np.testing.assert_array_equal(a.argmax(1),b.argmax(1))
    del gpu
    torch.cuda.empty_cache()
    shutil.copy2(checkpoint,output/'model.pt')
    p=predict(cpu,prep,IndexedRows(X,np.load(study/f'folds/{request["fold"]}/oof.npy')))
    np.save(output/'oof_probabilities.npy',p)
    ids=np.load(study/f'folds/{request["fold"]}/oof.npy')
    np.save(output/'oof_row_ids.npy',np.load(DATA/'model/row_ids_train.npy')[ids])
    other,other_prep,_=load_checkpoint(output/'model.pt')
    replay=predict(other,other_prep,X[ids[:128]])
    np.testing.assert_allclose(p[:128],replay,atol=2e-6,rtol=1e-5)
    duplicate,duplicate_prep,_=load_checkpoint(output/'model.pt')
    duplicate_p=predict(duplicate,duplicate_prep,X[ids[:128]])
    np.testing.assert_allclose(replay,duplicate_p,atol=0,rtol=0)
    early_p=predict(cpu,prep,IndexedRows(X,early))
    diagnostic=dict(source_checkpoint=relative(checkpoint),source_sha256=digest(checkpoint),
        original_failure=relative(source/'failure.json'),original_native_gpu_cpu=original,
        aligned_gpu_cpu=parity_diagnostics(aligned,reference),extended_rows=len(audit_ids),
        extended_aligned_gpu_cpu=parity_diagnostics(a,b),original_atol=5e-5,original_rtol=5e-4,
        cpu_computation_unchanged=True,weights_and_bins_unchanged=True,
        adaptation='CUDA PLE inference uses separate multiply/add to match CPU operation order')
    atomic_write(output/'numerical_audit.json',diagnostic)
    result=dict(request,status='completed',run_directory=relative(output),checkpoint=relative(output/'model.pt'),
        cpu_checkpoint_reload_verified=True,cpu_replay_rows=128,training_rows=len(fitting_rows(study,request['fold'],request['arm'])),
        selection_metrics=classification_metrics(y[early],early_p.argmax(1),manifest['classes'],0),
        recovery=diagnostic,elapsed_seconds=time.monotonic()-started,test_loaded=False,
        artifact_hashes={relative(p):digest(p) for p in output.iterdir() if p.suffix in ('.pt','.npy','.json') and p.name!='result.json'})
    atomic_write(output/'result.json',result)
    return result
