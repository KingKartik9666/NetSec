"""Immutable inputs, fold roles and branch recipes for the hybrid study."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path
import shutil

import numpy as np

from experiment.datasense_quality import digest
from experiment.train_verified_clean import read, write
from framework.training.hybrid_data import forward_folds, strict_fit_indices
from framework.training.tabpack_backend import pack_configuration, upstream_pins
from framework.training.tracking import validate_manifest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'dataset/experiments/datasense_verified_clean_20261014_v2'
STUDY = ROOT/'artifacts/training/hybrids_verified_v2_20261018_v2'
ORIGINAL = ROOT/'artifacts/training/verified_clean_four_models_20261016_v1'
TABM = ROOT/'artifacts/training/tabm_verified_v2_20261016_v1'
COMBINATIONS = {
    'tabm_xgboost': ['tabm_ne', 'xgboost'],
    'tabm_xgboost_kan': ['tabm_ne', 'xgboost', 'kan'],
    'tabpack_xgboost': ['tabpack', 'xgboost'],
}
SOURCES = (
    'experiment/train_hybrids.py', 'framework/training/hybrid_data.py',
    'framework/training/hybrid_blend.py', 'framework/training/hybrid_neural.py',
    'framework/training/hybrid_protocol.py', 'framework/training/hybrid_bundle.py',
    'framework/training/hybrid_evaluation.py',
    'framework/training/tabpack_backend.py', 'framework/training/tabm_data.py',
    'framework/training/tabm_models.py', 'framework/training/tabm_fit.py',
    'framework/training/tabm_calibration.py', 'framework/training/neural.py',
    'framework/training/csv_xgboost.py', 'framework/training/trees.py',
    'framework/training/tracking.py', 'framework/training/csv_monitoring.py',
    'framework/data/research_data.py', 'experiment/train_csv_models.py',
    'experiment/train_verified_clean.py', 'experiment/optimize_models.py',
    'experiment/datasense_quality.py', 'experiment/calibrate_datasense_models.py',
    'experiment/train_datasense_quality.py',
)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def verify_hashes(pins, root=ROOT):
    for name, expected in pins.items():
        if digest(Path(root)/name) != expected:
            raise ValueError(f'Pinned asset changed: {name}')


def branch_reference(family, run):
    run = Path(run)
    checkpoint = run/('model.ubj' if family == 'xgboost' else 'model.pt')
    schema = DATA/'model/manifest.json'
    files = [checkpoint, schema]
    if family == 'kan':
        files.append(run/'preprocessing.joblib')
    return dict(family=family, checkpoint=relative(checkpoint), schema_manifest=relative(schema),
        assets={relative(p):digest(p) for p in files},
        validation_probabilities=relative(run/'validation_probabilities.npy'),
        validation_sha256=digest(run/'validation_probabilities.npy'))


def verify_protocol(protocol, include_data=True):
    verify_hashes(protocol['source_hashes'])
    verify_hashes(protocol['fold_hashes'])
    if include_data:
        verify_hashes(protocol['input_hashes'])
    for family, branch in protocol['final_branches'].items():
        verify_hashes(branch['assets'])
        verify_hashes({branch['validation_probabilities']:branch['validation_sha256']})
    for name, version in protocol['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Pinned package changed: {name}')
    upstream_pins()


def initialize():
    STUDY.mkdir(parents=True, exist_ok=True)
    if (STUDY/'study.json').exists():
        protocol = read(STUDY/'study.json'); verify_protocol(protocol); return protocol
    manifest = read(DATA/'model/manifest.json'); validate_manifest(manifest)
    if len(manifest['feature_names']) != 655 or len(manifest['classes']) != 84 or manifest['benign_index'] != 0:
        raise ValueError('Verified v2 schema differs')
    for name in ('integrity', 'independent_feature_replay'):
        if not read(DATA/f'verification/{name}.json')['passed']:
            raise ValueError('Dataset verification gate failed')
    paths = [DATA/'model/manifest.json', DATA/'verification/integrity.json',
        DATA/'verification/independent_feature_replay.json', DATA/'verification/output_hashes.json']
    paths += [DATA/f'model/{kind}_{split}.npy' for split in ('train', 'validation')
              for kind in ('X', 'y', 'row_ids', 'window_start', 'window_end')]
    paths += [DATA/f'audit/{split}/model_input_sha256.npy' for split in ('train', 'validation')]
    paths.append(DATA/'model/strict_train/row_ids_train.npy')
    expected = read(DATA/'verification/output_hashes.json')
    input_hashes = {}
    for p in paths:
        value = digest(p)
        if p.relative_to(DATA).as_posix() in expected and value != expected[p.relative_to(DATA).as_posix()]:
            raise ValueError(f'Verified dataset changed: {p.name}')
        input_hashes[relative(p)] = value
    train_ids = np.load(DATA/'model/row_ids_train.npy')
    val_ids = np.load(DATA/'model/row_ids_validation.npy')
    if np.intersect1d(train_ids, val_ids).size or len(np.unique(train_ids)) != len(train_ids):
        raise ValueError('Dataset row identity mismatch')
    roles = np.load(ORIGINAL/'validation_roles.npy')
    np.testing.assert_array_equal(roles, np.load(TABM/'validation_roles.npy'))
    # Verify the original partition bytes against its sealed protocol.
    if digest(ORIGINAL/'validation_roles.npy') != read(ORIGINAL/'study.json')['roles_sha256']:
        raise ValueError('Original validation roles changed')
    for prior in (ORIGINAL, TABM):
        if read(prior/'study.json')['input_hashes']['model/manifest.json'] != digest(DATA/'model/manifest.json'):
            raise ValueError('Selected branch study uses a different ordered schema')
    np.save(STUDY/'validation_roles.npy', roles)
    y = np.load(DATA/'model/y_train.npy')
    folds = forward_folds(y, np.load(DATA/'model/window_start_train.npy'),
        np.load(DATA/'model/window_end_train.npy'), np.load(DATA/'audit/train/model_input_sha256.npy'))
    strict = np.load(DATA/'model/strict_train/row_ids_train.npy')
    fold_hashes = {relative(STUDY/'validation_roles.npy'):digest(STUDY/'validation_roles.npy')}
    counts = []
    for index, fold in enumerate(folds):
        fold['fit_strict'] = strict_fit_indices(fold['fit'], train_ids, strict, y, 84)
        folder = STUDY/'folds'/str(index); folder.mkdir(parents=True, exist_ok=True)
        report = {}
        for role, ids in fold.items():
            np.save(folder/f'{role}.npy', ids)
            np.save(folder/f'{role}_row_ids.npy', train_ids[ids])
            report[role] = dict(rows=len(ids), unsupported_classes=np.flatnonzero(np.bincount(y[ids], minlength=84)==0).tolist())
        for p in folder.glob('*.npy'):
            fold_hashes[relative(p)] = digest(p)
        counts.append(report)
    branches, recipes = {}, {}
    for family, source in (('tabm_ne', TABM), ('xgboost', ORIGINAL), ('kan', ORIGINAL)):
        selected_path = source/'selected'/f'{family}.json'
        selected = read(selected_path); run = ROOT/selected['run_directory']
        if selected['status'] != 'completed' or selected['family'] != family:
            raise ValueError('Invalid selected branch result')
        for name, expected_hash in selected['artifact_hashes'].items():
            if name in ('model.pt', 'model.ubj', 'preprocessing.joblib', 'validation_probabilities.npy') and digest(run/name) != expected_hash:
                raise ValueError('Selected branch bytes changed')
        input_hashes[relative(selected_path)] = digest(selected_path)
        branches[family] = branch_reference(family, run)
        recipes[family] = dict(config=selected['config'], arm=selected['arm'], seed=42)
    recipes['tabpack'] = dict(config=pack_configuration(42), arm='full', seed=42)
    jobs = [dict(job=f'{family}_fold{fold}', family=family, fold=fold, **recipes[family])
        for family in ('tabm_ne', 'xgboost', 'kan', 'tabpack') for fold in (0, 1)]
    jobs.append(dict(job='tabpack_full', family='tabpack', fold=None, **recipes['tabpack']))
    sources = list(SOURCES)+[relative(p) for p in (ROOT/'external/efficient-kan/src/efficient_kan').glob('*.py')]
    sources += ['external/tabpack/'+p for p in upstream_pins()]
    source_hashes = {name:digest(ROOT/name) for name in sources}
    for name in sources:
        target = STUDY/'source_snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    protocol = dict(version=2, device='cuda', seed=42, jobs=jobs, planned_fits=len(jobs),
        combinations=COMBINATIONS, recipes=recipes, final_branches=branches,
        input_hashes=input_hashes, source_hashes=source_hashes, fold_hashes=fold_hashes,
        packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','torch','xgboost','scikit-learn','tabm','rtdl-num-embeddings','joblib')},
        folds=counts, validation_roles={str(i):int((roles==i).sum()) for i in (-1,0,1,2)},
        combiner_penalties=[None, .01, .1], per_fit_seconds=7200, test_loaded=False,
        memory_minimum_gib={'neural_physical':2.5,'other_physical':4.,'commit':6.},
        memory_policy='Neural fitting streams selected rows/columns; controller avoids torch; eligible jobs may run out of queue order.',
        auto_promote=False, caveat='Previously reused within-capture validation; exploratory comparison, not independent final test.')
    write(STUDY/'study.json', protocol)
    return protocol
