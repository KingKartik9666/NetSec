"""Isolated CSV fits using existing model backends, with explicit data views.

Only this module adapts the redesigned schema to established model trainers.
It never imports a source-validation loader or changes the application models.
"""
from __future__ import annotations

from pathlib import Path
import time
import traceback

import numpy as np

from framework.training.live_study import calibration
from framework.training.live_study.protocol import ROOT, HYBRIDS, atomic, read, relative, digest
from framework.training.live_study.worker import _fit, predict as backend_predict, verify_result


def arm_view(arm, manifest, roles, unfiltered):
    """Select columns and fit rows without changing any evaluation role."""
    changed = np.array(roles, copy=True)
    if arm == 'base_unfiltered':
        changed[np.asarray(unfiltered, bool)] = 0
    if arm.startswith('base_'):
        columns = manifest['base_feature_indices']
    else:
        columns = manifest['selected_feature_indices']
        if arm.startswith('enrich_no_'):
            group = arm.removeprefix('enrich_no_')
            excluded = set(manifest['feature_groups'][group])
            columns = [i for i in columns if i not in excluded]
    if not columns or len(columns) != len(set(columns)):
        raise ValueError('Feature view is empty or duplicated')
    return list(columns), changed


def balanced_fit_indices(y, roles, scenarios, seed):
    """Cap inverse scenario weight at four and preserve every class row count.

    This resamples real fitting observations, not synthetic data or relabeling.
    Its duplicate multiplicities are exported for review.
    """
    rng = np.random.default_rng(seed)
    fitting = np.flatnonzero(roles == 0)
    selected = []
    for label in np.unique(y[fitting]):
        ids = fitting[y[fitting] == label]
        _, inverse, counts = np.unique(scenarios[ids], return_inverse=True, return_counts=True)
        weights = np.minimum(4., len(ids) / (len(counts) * counts[inverse]))
        selected.append(rng.choice(ids, size=len(ids), replace=True, p=weights/weights.sum()))
    result = np.concatenate(selected)
    return result[rng.permutation(len(result))]


def predict(record, x):
    if record['family'] in HYBRIDS:
        from framework.training.hybrid_blend import blend
        names = HYBRIDS[record['family']]
        probability = blend(np.stack([predict(record['branches'][name], x)['probabilities']
                                     for name in names]), read(ROOT/record['checkpoint']), names)
        return dict(pred=probability.argmax(1), probabilities=probability)
    return backend_predict(record, np.asarray(x[:, record['feature_indices']], np.float32))


def _hybrid(request, study, y, roles, output):
    from framework.training.hybrid_blend import blend
    from framework.training.live_study.combiner import fit
    names = HYBRIDS[request['family']]
    selected = read(study/'selected.json')['branches']
    records = {}
    for name in names:
        path = (study/f'repeats/{name}/seed{request["seed"]}/result-link.json'
                if request['phase'] == 'repeat' else ROOT/selected[name]['result'])
        records[name] = verify_result(path)
        if records[name]['arm'] != request['arm']:
            raise ValueError('Hybrid branches must use the same frozen representation')
    def outputs(role):
        values = []
        for name in names:
            with np.load(ROOT/records[name]['run_directory']/f'{role}_outputs.npz', allow_pickle=False) as archive:
                values.append(archive['probabilities'])
        return np.stack(values)
    config = request['config']
    recipe = (read(ROOT/config['frozen_combiner']) if request['phase'] == 'repeat' else
              fit(outputs('blend'), y[roles == 2], names,
                  config['penalty'] if config['kind'] == 'classwise' else None))
    checkpoint = output/'combiner.json'
    atomic(checkpoint, recipe)
    predicted = {}
    for role in ('tune', 'blend', 'calibration'):
        p = blend(outputs(role), recipe, names)
        predicted[role] = dict(pred=p.argmax(1), probabilities=p)
    return checkpoint, dict(optimizer='SLSQP or L-BFGS-B', fit_role='blend'), records, predicted


def run(request_path):
    from .runner import validate_request, verify_study, selection_value
    from framework.training import tracking
    tracking.write_json = atomic
    request_path = Path(request_path).resolve()
    output = request_path.parent
    request = read(request_path)
    validate_request(request)
    study = ROOT/request['study']
    settings, manifest, data = verify_study(study)
    if request['protocol_fingerprint'] != settings['protocol_fingerprint']:
        raise ValueError('Request study protocol differs')
    x = np.load(data/'X_train.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(data/'y_train.npy', mmap_mode='r', allow_pickle=False)
    roles = np.load(data/'roles.npy', mmap_mode='r', allow_pickle=False)
    unfiltered = np.load(data/'fit_unfiltered.npy', mmap_mode='r', allow_pickle=False)
    columns, fit_roles = arm_view(request['arm'], manifest, roles, unfiltered)
    started = time.monotonic()
    if request['phase'] == 'assessment':
        return assess(request, study, settings, manifest, x, y, roles, output)
    outputs, branches = {}, None
    if request['family'] in HYBRIDS:
        checkpoint, details, branches, outputs = _hybrid(request, study, y, roles, output)
    else:
        local_manifest = dict(manifest, feature_names=[manifest['feature_names'][i] for i in columns])
        selected = np.flatnonzero((fit_roles >= 0) & (fit_roles <= 1))
        if request['arm'] == 'enriched_scenario_balanced':
            scenarios = np.load(data/'scenario_codes.npy', mmap_mode='r', allow_pickle=False)
            draw = balanced_fit_indices(y, fit_roles, scenarios, request['seed'])
            selected = np.concatenate((draw, np.flatnonzero(fit_roles == 1)))
            np.save(output/'fitting_resample_indices.npy', draw, allow_pickle=False)
            atomic(output/'resampling.json', dict(method='within-class inverse scenario frequency; cap 4x',
                original_fitting_rows=int((fit_roles == 0).sum()), drawn_rows=len(draw),
                unique_original_rows=len(np.unique(draw)), synthetic_rows=0, labels_changed=False))
        # Only fit/tune materialize. Challenge is not read into model calls.
        fit_x = np.asarray(x[np.ix_(selected, columns)], np.float32)
        checkpoint, details = _fit(request, fit_x, y[selected], fit_roles[selected], local_manifest, output)
        del fit_x
        if request['family'] == 'random_forest':
            details['backend'] = 'scikit-learn bootstrap RandomForestClassifier; CSV redesign research'
    record = dict(request, checkpoint=relative(checkpoint), run_directory=relative(output),
                  n_classes=len(manifest['classes']), feature_indices=columns,
                  encoder=relative(data/'encoder.json'))
    if branches:
        record['branches'] = branches
    for number, role in enumerate(('tune', 'blend', 'calibration'), 1):
        if role not in outputs:
            outputs[role] = predict(record, x[roles == number])
        calibration.validate_output(outputs[role], int((roles == number).sum()), len(manifest['classes']))
        np.savez_compressed(output/f'{role}_outputs.npz', **outputs[role])
    correction = calibration.fit(outputs['calibration'], y[roles == 3], outputs['tune'], y[roles == 1])
    atomic(output/'calibration.json', correction)
    score = calibration.measure(outputs['tune'], y[roles == 1], manifest, correction)
    sample = x[np.flatnonzero(roles == 1)[:128]]
    first, second = predict(record, sample), predict(record, sample)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    timings = []
    for _ in range(3):
        tick = time.perf_counter()
        predict(record, sample[:1])
        timings.append(time.perf_counter()-tick)
    status = 'pruned' if (output/'PRUNE').exists() else 'completed'
    record.update(status=status, selection_metrics=score, objective=selection_value({'selection_metrics': score}),
        selection_objective='tune accuracy; final selection additionally guards macro F1 and false alarms',
        details=details, calibration=correction, elapsed_seconds=time.monotonic()-started,
        cpu_checkpoint_reload_verified=True, cpu_replay_rows=len(sample),
        cold_single_window_seconds=dict(median=float(np.median(timings)), maximum=max(timings)),
        original_validation_loaded=False, test_loaded=False, challenge_used=False,
        artifact_hashes={relative(p): digest(p) for p in output.iterdir()
                        if p.suffix in ('.pt', '.joblib', '.ubj', '.npz', '.npy', '.json')
                        and p.name not in ('result.json', 'progress.json', 'failure.json')})
    atomic(output/'result.json', record)
    print(f'{status.upper()} {request["family"]} arm={request["arm"]} accuracy={score["accuracy"]:.6f}', flush=True)


def assess(request, study, settings, manifest, x, y, roles, output):
    """Open the internal scenario challenge only after selections are sealed."""
    selected = read(study/'selected.json')
    if not (study/'selection_seal.json').exists():
        raise ValueError('Selection must be frozen before internal challenge assessment')
    seal = read(study/'selection_seal.json')
    if digest(study/'selected.json') != seal['selected_sha256']:
        raise ValueError('Selection changed after freeze')
    challenge = np.flatnonzero(roles == 4)
    if not len(challenge):
        raise ValueError('No internal challenge observations')
    report = {}
    for family, entry in {**selected['branches'], **selected['hybrids']}.items():
        record = verify_result(ROOT/entry['result'])
        values = predict(record, x[challenge])
        report[family] = calibration.measure(values, y[challenge], manifest, record['calibration'])
        np.savez_compressed(output/f'{family}_challenge.npz', **values)
    result = dict(status='completed', request_fingerprint=request['request_fingerprint'],
        metrics=report, challenge_rows=len(challenge), scope='original-training scenario holdout; exploratory',
        original_validation_loaded=False, test_loaded=False, selection_used_challenge=False,
        artifact_hashes={relative(p): digest(p) for p in output.glob('*.npz')})
    atomic(output/'result.json', result)


def main(request_path):
    try:
        run(request_path)
    except BaseException as error:
        atomic(Path(request_path).parent/'failure.json', dict(status='failed', error=type(error).__name__,
            message=str(error), traceback=traceback.format_exc(), time=time.time()))
        raise
