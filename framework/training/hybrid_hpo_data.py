"""Frozen provenance and purged inner roles for Optuna hybrid retraining."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path
import shutil
import numpy as np

from experiment.train_verified_clean import read,write
from experiment.datasense_quality import digest,calibration_partition
from framework.training.hybrid_protocol import (ROOT,DATA,STUDY as PREVIOUS,
    COMBINATIONS,verify_hashes,verify_protocol,relative)

STUDY=ROOT/'artifacts/training/hybrids_optuna_20261018_v1'
NEW_SOURCES=('experiment/optimize_hybrids.py','framework/training/hybrid_hpo.py',
    'framework/training/hybrid_hpo_data.py','framework/training/hybrid_hpo_finalize.py',
    'artifacts/training/hybrids_verified_v2_20261018_v2/completion_monitor/monitor.py')


def meta_partition(labels,starts,ends,hashes):
    return calibration_partition(labels,starts,ends,hashes)


def validate_adoption(request,result):
    if any(request[k]!=result[k] for k in ('family','config','arm','seed')):
        raise ValueError('Adopted branch recipe differs')


def verify(protocol):
    for key in ('input_hashes','source_hashes','fold_hashes','partner_hashes'):
        verify_hashes(protocol[key])
    for name,value in protocol['packages'].items():
        if importlib.metadata.version(name)!=value:raise ValueError(f'Pinned package changed: {name}')


def initialize():
    STUDY.mkdir(parents=True,exist_ok=True)
    if (STUDY/'study.json').exists():
        protocol=read(STUDY/'study.json');verify(protocol);return protocol
    old=read(PREVIOUS/'study.json');verify_protocol(old)
    previous_state=read(PREVIOUS/'state.json')
    if previous_state['status']!='completed':raise ValueError('Previous hybrid study is not complete')
    for name in ('validation_roles.npy',):shutil.copy2(PREVIOUS/name,STUDY/name)
    shutil.copytree(PREVIOUS/'folds',STUDY/'folds',dirs_exist_ok=True)
    partners={};partner_hashes={relative(PREVIOUS/'study.json'):digest(PREVIOUS/'study.json')}
    for family in ('tabm_ne','xgboost','kan','tabpack'):
        refs={}
        for suffix in ('fold0','fold1')+ (('full',) if family=='tabpack' else ()):
            paths=list((PREVIOUS/'runs'/f'{family}_{suffix}').glob('attempt_*/result.json'))
            successful=[p for p in paths if read(p).get('status')=='completed']
            if len(successful)!=1:raise ValueError('Ambiguous or absent baseline fit')
            path=successful[0];result=read(path);verify_hashes(result['artifact_hashes'])
            refs[suffix]=relative(path)
            partner_hashes[relative(path)]=digest(path)
            partner_hashes.update(result['artifact_hashes'])
        partners[family]=refs
    for branch in old['final_branches'].values():
        run=(ROOT/branch['checkpoint']).parent
        path=run/'result.json';result=read(path)
        if result.get('status')!='completed' or not result.get('checkpoint_reload_verified') or result.get('test_loaded'):
            raise ValueError('Historical full model lacks verified completion')
        partner_hashes[relative(path)]=digest(path)
        for name,value in result['artifact_hashes'].items():
            artifact=(run/name).resolve();artifact.relative_to(run.resolve())
            partner_hashes[relative(artifact)]=value
    verify_hashes(partner_hashes)
    oof=np.load(STUDY/'folds/0/oof.npy')
    y=np.load(DATA/'model/y_train.npy');s=np.load(DATA/'model/window_start_train.npy')
    e=np.load(DATA/'model/window_end_train.npy');h=np.load(DATA/'audit/train/model_input_sha256.npy')
    roles=meta_partition(y[oof],s[oof],e[oof],h[oof]);np.save(STUDY/'meta_roles.npy',roles)
    if any((roles==i).sum()<1000 for i in (0,1)):raise ValueError('Insufficient purged meta role support')
    fold_files=list((STUDY/'folds').rglob('*.npy'))+[STUDY/'validation_roles.npy',STUDY/'meta_roles.npy']
    source_names=sorted(set(old['source_hashes'])|set(NEW_SOURCES))
    source_hashes={name:digest(ROOT/name) for name in source_names}
    for name in source_names:
        target=STUDY/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    protocol=dict(version=1,device='cuda',trials_per_family=12,combiner_trials=12,
        family_order=['tabm_ne','xgboost','kan','tabpack'],sampler='Optuna TPESampler',
        startup_trials=4,class_blend_maxiter=1000,per_fit_seconds=7200,
        recipes=old['recipes'],previous_final_branches=old['final_branches'],baseline_partners=partners,
        combinations=COMBINATIONS,input_hashes=old['input_hashes'],source_hashes=source_hashes,
        fold_hashes={relative(p):digest(p) for p in fold_files},partner_hashes=partner_hashes,
        packages=dict(old['packages'],optuna=importlib.metadata.version('optuna')),
        meta_roles={str(i):int((roles==i).sum()) for i in (-1,0,1)},
        validation_roles=old['validation_roles'],test_loaded=False,auto_promote=False,
        caveat=old['caveat'],previous_study=relative(PREVIOUS),
        objective='Mean affected-hybrid accuracy + 1e-6 macro F1 on purged training meta-score rows')
    write(STUDY/'study.json',protocol);return protocol


def prior_result(protocol,family,suffix='fold0'):
    path=ROOT/protocol['baseline_partners'][family][suffix]
    verify_hashes({relative(path):protocol['partner_hashes'][relative(path)]})
    result=read(path);verify_hashes(result['artifact_hashes']);return result


def load_oof(result,fold=0):
    run=ROOT/result['run_directory']
    expected=np.load(DATA/'model/row_ids_train.npy')[np.load(STUDY/f'folds/{fold}/oof.npy')]
    np.testing.assert_array_equal(np.load(run/'oof_row_ids.npy'),expected)
    verify_hashes(result['artifact_hashes'])
    return np.load(run/'oof_probabilities.npy')
