"""Freeze the authorized day budget, data roles, recipes, and source evidence."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import shutil
import time

import numpy as np

from experiment.datasense_quality import digest
from framework.training.hybrid_protocol import ROOT, DATA, relative, verify_hashes
from framework.training.study_supervisor import FAMILIES, atomic_write, read

STUDY = ROOT/'artifacts/training/all_models_day_20261020_v1'
OLD_HYBRIDS = ROOT/'artifacts/training/hybrids_optuna_20261018_v1'
ORIGINAL = ROOT/'artifacts/training/verified_clean_four_models_20261016_v1'
TABM = ROOT/'artifacts/training/tabm_verified_v2_20261016_v1'
SOURCES = ('experiment/optimize_all_models_day.py','experiment/resume_hybrids_durable.py',
    'framework/training/study_supervisor.py','framework/training/day_protocol.py',
    'framework/training/day_search.py','framework/training/day_worker.py',
    'framework/training/day_finalize.py',
    'external/HEM/model_evaluation/binary_classifier.py',
    'external/HEM/model_evaluation/multiclass_classifier.py',
    'external/efficient-kan/src/efficient_kan/kan.py')


def normalize_result(path):
    from framework.training.day_worker import source_checkpoint
    path = Path(path)
    row = read(path)
    if row.get('status') != 'completed' or row.get('test_loaded'):
        raise ValueError('Invalid historical candidate')
    pins = {}
    for name,value in row['artifact_hashes'].items():
        artifact = ROOT/name if (ROOT/name).exists() else path.parent/name
        pins[relative(artifact)] = value
    verify_hashes(pins)
    checkpoint = source_checkpoint(row,path.parent)
    row.update(checkpoint=relative(checkpoint),run_directory=relative(path.parent),
        source_result=relative(path),source_result_sha256=digest(path),artifact_hashes=pins)
    return row


def baseline_records():
    records = {}
    for folder,families in ((ORIGINAL,('random_forest','xgboost','kan','hem')),
                            (TABM,('tabm_ne','tabm_hier','tabm_hier_ne'))):
        candidates = [p for p in (folder/'runs').glob('*/attempt_*/result.json')
            if read(p).get('status')=='completed']
        for family in families:
            eligible = [p for p in candidates if read(p)['family']==family and read(p)['seed']==42]
            if family=='random_forest':
                eligible = [p for p in eligible if read(p)['recipe']=='random_forest_full_c0' and read(p)['seed']==42]
            chosen = max(eligible,key=lambda p:(read(p)['selection_metrics']['accuracy'],
                                                read(p)['selection_metrics']['macro_f1']))
            records[family] = normalize_result(chosen)
    # TabPack is an auxiliary trained branch of the tenth (hybrid) model.
    folder = ROOT/'artifacts/training/hybrids_verified_v2_20261018_v2/runs/tabpack_full'
    records['tabpack'] = normalize_result(next(folder.glob('attempt_*/result.json')))
    # The original branch result has no standalone metrics. Derive its control
    # strictly on the frozen tuning role; do not inspect assessment labels here.
    from framework.training.tracking import classification_metrics
    roles=np.load(ORIGINAL/'validation_roles.npy');tuning=np.flatnonzero(roles==0)
    run=ROOT/records['tabpack']['run_directory']
    probabilities=np.load(run/'validation_probabilities.npy',mmap_mode='r')
    truth=np.load(DATA/'model/y_validation.npy')[tuning]
    manifest=read(DATA/'model/manifest.json')
    records['tabpack']['selection_metrics']=classification_metrics(truth,probabilities[tuning].argmax(1),manifest['classes'],0)
    return records


def initialize():
    if (STUDY/'study.json').exists():
        protocol = read(STUDY/'study.json')
        verify(protocol)
        return protocol
    STUDY.mkdir(parents=True,exist_ok=True)
    original = read(ORIGINAL/'study.json')
    input_hashes = {relative(DATA/name):value for name,value in original['input_hashes'].items()}
    verify_hashes(input_hashes)
    verify_hashes(read(OLD_HYBRIDS/'study.json')['fold_hashes'])
    shutil.copy2(OLD_HYBRIDS/'validation_roles.npy',STUDY/'validation_roles.npy')
    shutil.copytree(OLD_HYBRIDS/'folds',STUDY/'folds',dirs_exist_ok=True)
    baselines = baseline_records()
    recipes = {f:{k:deepcopy(baselines[f][k]) for k in ('family','arm','seed','config')} for f in FAMILIES}
    for f in ('tabm_ne','xgboost','kan'):
        chosen = OLD_HYBRIDS/'selected'/f'{f}.json'
        if chosen.exists():
            recipes[f] = read(chosen)['recipe']
    for recipe in recipes.values():
        recipe['config']['runtime_seconds'] = 1500
    old = read(OLD_HYBRIDS/'study.json')
    hybrid_baselines={}
    for name in old['combinations']:
        path=ROOT/'artifacts/training/hybrids_verified_v2_20261018_v2/hybrids'/name/'result.json'
        result=read(path);verify_hashes(result['artifact_hashes'])
        hybrid_baselines[name]=dict(accuracy=result['full_validation_metrics']['accuracy'],
            macro_f1=result['full_validation_metrics']['macro_f1'],
            assessment_accuracy=result['calibration']['before']['accuracy'],
            source_result=relative(path),source_result_sha256=digest(path))
    source_names = sorted(set(SOURCES)|set(old['source_hashes']))
    source_hashes = {name:digest(ROOT/name) for name in source_names}
    for name in source_names:
        target = STUDY/'source_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    now = time.time()
    protocol = dict(version=1,started_at=now,deadline=now+86400,
        started_utc=datetime.fromtimestamp(now,timezone.utc).isoformat(),
        deadline_utc=datetime.fromtimestamp(now+86400,timezone.utc).isoformat(),
        device='cuda',families=list(FAMILIES),fixed_rf=True,recipes=recipes,
        baselines=baselines,hybrid_baselines=hybrid_baselines,source_hashes=source_hashes,input_hashes=input_hashes,
        role_hashes={relative(p):digest(p) for p in [STUDY/'validation_roles.npy',*(STUDY/'folds').rglob('*.npy')]},
        packages={name:importlib.metadata.version(name) for name in ('optuna','torch','xgboost','numpy','scipy','scikit-learn',
            'tabm','rtdl-num-embeddings','joblib')},
        search_hours=14,repeat_end_hours=17,oof_end_hours=21,
        sampler='Optuna multivariate group TPE; deterministic ask-index seeds',
        pruner='Asynchronous successive halving; per-family tuning accuracy; minimum 6 epochs / 60 rounds',
        minimum_trials_per_family=3,per_fit_seconds=2100,max_attempts=2,
        repeat_seeds=[43,44],combiner_trials=12,
        regression_guards=dict(max_fpr_increase=.005,max_attack_miss_increase=.005,max_macro_recall_drop=.01),
        roles=old['validation_roles'],combinations=old['combinations'],
        test_loaded=False,auto_promote=False,
        caveat='Previously reused within-capture validation; exploratory comparisons, not independent final-test or live results.')
    atomic_write(STUDY/'study.json',protocol)
    atomic_write(STUDY/'baselines.json',baselines)
    return protocol


def verify(protocol):
    verify_hashes(protocol['source_hashes'])
    verify_hashes(protocol['input_hashes'])
    verify_hashes(protocol['role_hashes'])
    for name,version in protocol['packages'].items():
        if importlib.metadata.version(name)!=version:
            raise ValueError('Pinned package changed: '+name)
    if protocol.get('test_loaded') or not protocol.get('fixed_rf'):
        raise ValueError('Invalid study safety policy')
