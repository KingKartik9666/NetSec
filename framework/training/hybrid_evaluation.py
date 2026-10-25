"""Select OOF-fitted combiners, calibrate, assess and export each hybrid."""
from __future__ import annotations

import numpy as np

from experiment.train_verified_clean import read, write
from experiment.datasense_quality import digest
from framework.training.hybrid_protocol import ROOT, DATA, STUDY, branch_reference, verify_hashes
from framework.training.hybrid_blend import blend, fit_blend, log_loss, quick_metrics
from framework.training.hybrid_bundle import HybridPredictor
from framework.training.tabm_calibration import fit_calibration, apply_calibration
from framework.training.tracking import classification_metrics


def completed_jobs():
    jobs = {}
    for path in sorted((STUDY/'runs').glob('*/attempt_*/result.json')):
        result = read(path)
        if result.get('status') == 'completed':
            verify_hashes(result['artifact_hashes'])
            if result['job'] in jobs:
                raise ValueError('Duplicate successful job; reconcile explicitly')
            jobs[result['job']] = result
    return jobs


def finalize_available(protocol):
    jobs = completed_jobs()
    for name, names in protocol['combinations'].items():
        destination = STUDY/'hybrids'/name
        if (destination/'result.json').exists():
            completed = read(destination/'result.json')
            if completed.get('status') != 'completed' or not completed.get('artifact_hashes'):
                raise ValueError('Invalid hybrid completion record')
            verify_hashes(completed['artifact_hashes'])
            HybridPredictor(destination/'bundle.json')
            continue
        required = [f'{family}_fold{i}' for family in names for i in (0, 1)]
        if 'tabpack' in names:
            required.append('tabpack_full')
        if any(j not in jobs for j in required):
            continue
        finalize(name, names, protocol, jobs, destination)


def finalize(name, names, protocol, jobs, destination):
    destination.mkdir(parents=True, exist_ok=True)
    manifest = read(DATA/'model/manifest.json')
    row_ids = np.load(DATA/'model/row_ids_train.npy')
    train_y = np.load(DATA/'model/y_train.npy')
    fold_ids = [np.load(STUDY/f'folds/{i}/oof.npy') for i in (0, 1)]
    oof_ids = np.concatenate(fold_ids)
    oof = []
    branches = {}
    validation = []
    for family in names:
        parts = []
        for i in (0, 1):
            run = ROOT/jobs[f'{family}_fold{i}']['run_directory']
            np.testing.assert_array_equal(np.load(run/'oof_row_ids.npy'), row_ids[fold_ids[i]])
            parts.append(np.load(run/'oof_probabilities.npy'))
        oof.append(np.concatenate(parts))
        if family == 'tabpack':
            branch = branch_reference(family, ROOT/jobs['tabpack_full']['run_directory'])
        else:
            branch = protocol['final_branches'][family]
        branches[family] = branch
        verify_hashes({branch['validation_probabilities']:branch['validation_sha256']})
        validation.append(np.load(ROOT/branch['validation_probabilities']))
    p_oof = np.stack(oof); del oof
    p_val = np.stack(validation); del validation
    roles = np.load(STUDY/'validation_roles.npy'); tune = roles == 0
    y = np.load(DATA/'model/y_validation.npy')
    candidates = []
    for penalty in protocol['combiner_penalties']:
        cfg = fit_blend(p_oof, train_y[oof_ids], names, penalty)
        p = blend(p_val[:, tune], cfg, names)
        score = quick_metrics(y[tune], p)
        score['log_loss'] = log_loss(p, y[tune])
        candidates.append(dict(combiner=cfg, tuning_metrics=score))
    chosen = max(candidates, key=lambda c:(c['tuning_metrics']['accuracy'],
        c['tuning_metrics']['macro_f1'], -c['tuning_metrics']['log_loss']))
    cfg = chosen['combiner']; p = blend(p_val, cfg, names)
    calibration, calibrated, binary = fit_calibration(p, y, roles, manifest)
    bundle = dict(status='pending_verification', name=name, branches=branches, combiner=cfg,
        classes=manifest['classes'], feature_names=manifest['feature_names'], benign_index=0,
        calibration=calibration, protocol_sha256=digest(STUDY/'study.json'),
        inference_device='cpu', test_loaded=False, research_scope=protocol['caveat'])
    write(destination/'bundle.json', bundle)
    np.save(destination/'validation_probabilities.npy', p)
    np.save(destination/'calibrated_probabilities.npy', calibrated)
    np.save(destination/'calibrated_attack_probability.npy', binary)
    np.save(destination/'combiner_fit_row_ids.npy', row_ids[oof_ids])
    val_ids = np.load(DATA/'model/row_ids_validation.npy')
    for role, role_name in ((1, 'calibration'), (2, 'assessment')):
        np.save(destination/f'{role_name}_row_ids.npy', val_ids[roles==role])
    replay_cal, replay_binary = apply_calibration(p, read(destination/'bundle.json')['calibration'])
    np.testing.assert_allclose(calibrated, replay_cal, rtol=0, atol=0)
    np.testing.assert_allclose(binary, replay_binary, rtol=0, atol=0)
    # Authentic raw observations exercise all saved branches + learned combiner.
    sample_ids = np.linspace(0, len(y)-1, 128, dtype=int)
    X = np.load(DATA/'model/X_validation.npy', mmap_mode='r')
    replay = HybridPredictor(destination/'bundle.json', _verification_only=True).predict_proba(X[sample_ids], manifest['feature_names'], calibrated=False)
    np.testing.assert_allclose(p[sample_ids], replay, atol=2e-5, rtol=1e-4)
    np.testing.assert_array_equal(p[sample_ids].argmax(1), replay.argmax(1))
    bundle['status'] = 'completed'
    write(destination/'bundle.json', bundle)
    comparison = {}
    for family, branch_p in zip(names, p_val):
        comparison[family] = {role_name:quick_metrics(y[roles==role], branch_p[roles==role])
            for role, role_name in ((0,'tuning'), (2,'assessment'))}
    equal = p_val.mean(0)
    result = dict(status='completed', name=name, combiner=cfg, candidates=candidates,
        calibration=calibration, branch_baselines=comparison,
        equal_weights={role_name:quick_metrics(y[roles==role], equal[roles==role])
            for role, role_name in ((0,'tuning'),(2,'assessment'))},
        tuning_metrics=chosen['tuning_metrics'],
        full_validation_metrics=classification_metrics(y, p.argmax(1), manifest['classes'], 0),
        cpu_bundle_replay_verified=True, cpu_replay_rows=len(sample_ids),
        combiner_fit_rows=len(oof_ids), combiner_fit_partition='training-only forward OOF',
        calibration_fit_partition='frozen validation role1',
        assessment_used_for_fitting_or_selection=False, test_loaded=False,
        caveat=protocol['caveat'], protocol_sha256=digest(STUDY/'study.json'))
    result['artifact_hashes'] = {str(f.relative_to(ROOT).as_posix()):digest(f)
        for f in destination.iterdir() if f.suffix in ('.npy','.json') and f.name != 'result.json'}
    write(destination/'result.json', result)
    write(STUDY/'results.json', {p.parent.name:read(p) for p in (STUDY/'hybrids').glob('*/result.json')})
    print(f'FINALIZED {name}: tuning={chosen["tuning_metrics"]["accuracy"]:.6f}, assessment={calibration["before"]["accuracy"]:.6f}', flush=True)
