"""CPU assessment of frozen broad candidates; no assessment-driven refitting."""
from pathlib import Path
import csv
import time

import numpy as np

from framework.training.live_study.protocol import (
    ROOT, BRANCHES, HYBRIDS, read, atomic, digest, relative, verify, verify_study,
)
from framework.training.live_study.worker import predict, verify_result
from framework.training.live_study.calibration import apply, measure, validate_output
from framework.training.live_features.schema import core_names, derive
from .data import (PREVIOUS, NUMERIC, CLEAN, CLEAN_HASHES, load_arrays, validate_arrays,
                   aligned_positions, assert_metadata, select_columns)


def frozen_records(study, settings):
    selected = read(study / 'selected.json')
    if set(selected['branches']) != set(BRANCHES) or set(selected.get('hybrids', {})) != set(HYBRIDS):
        raise ValueError('Assessment requires frozen selections for all branches and hybrids')
    primary, repeats = {}, {}
    for name, entry in {**selected['branches'], **selected['hybrids']}.items():
        path = ROOT / entry['result']
        if digest(path) != entry['result_sha256']:
            raise ValueError('Frozen selection result changed')
        primary[name] = verify_result(path)
        repeats[name] = {str(seed): verify_result(study / f'repeats/{name}/seed{seed}/result-link.json')
                         for seed in settings['repeat_seeds']}
        for seed, record in [('42', primary[name]), *repeats[name].items()]:
            if (record['status'] != 'completed' or record['family'] != name or record['seed'] != int(seed)
                    or record['protocol_fingerprint'] != settings['protocol_fingerprint']
                    or record['schema_fingerprint'] != settings['schema_fingerprint']
                    or not record['cpu_checkpoint_reload_verified']):
                raise ValueError('All selected and repeated models need completed compatible CPU replay')
    return primary, repeats


def load_validation(study, manifest, output):
    """Identity-join only historically admitted validation, after selection closes."""
    old = read(PREVIOUS / 'manifest.json')
    numeric_manifest, clean_manifest = read(NUMERIC / 'manifest.json'), read(CLEAN / 'manifest.json')
    classes = manifest['classes']
    if any(item['classes'] != classes for item in (old, numeric_manifest, clean_manifest)):
        raise ValueError('Assessment class order differs from training')
    pins = {name: sha for name, sha in {**old['input_hashes'], **old['array_hashes']}.items()
            if '_validation.npy' in Path(name).name}
    clean_hashes = read(CLEAN_HASHES)
    for field in ('X', 'y', 'row_ids', 'window_start', 'window_end'):
        key = f'model/{field}_validation.npy'
        pins[relative(CLEAN.parent / key)] = clean_hashes[key]
    verify(pins)
    admitted = load_arrays(PREVIOUS, 'validation')
    validate_arrays(admitted, old['feature_names'], classes)
    numeric, clean = load_arrays(NUMERIC, 'validation'), load_arrays(CLEAN, 'validation')
    validate_arrays(numeric, numeric_manifest['feature_names'], classes, exact_windows=False)
    validate_arrays(clean, clean_manifest['feature_names'], classes, exact_windows=False)
    a, b = (aligned_positions(source['row_ids'], admitted['row_ids']) for source in (numeric, clean))
    assert_metadata(numeric, admitted, a)
    assert_metadata(clean, admitted, b)
    raw = np.asarray(numeric['X'][a], dtype=np.float32)
    index = {name: i for i, name in enumerate(numeric_manifest['feature_names'])}
    derivatives = derive(raw, numeric_manifest['feature_names'], (admitted['window_end']-admitted['window_start']) / 1e9)
    names = core_names()
    broad = np.column_stack([raw[:, index[name]] if name in index else derivatives[name] for name in names])
    port_names = [name for name in manifest['feature_names'] if name not in names]
    positions = [clean_manifest['feature_names'].index(name) for name in port_names]
    ports = clean['X'][np.ix_(b, positions)]
    x = select_columns(broad, names, ports, port_names, manifest['feature_names'])
    novel = np.load(PREVIOUS / 'novel_validation.npy', allow_pickle=False)
    if novel.dtype != bool or novel.shape != admitted['y'].shape:
        raise ValueError('Historical novelty mask differs from admitted validation')
    # This is the conservative old seven-count mask, not new 252-feature novelty.
    for name, value in dict(X=x, y=admitted['y'], row_ids=admitted['row_ids'], novel=novel).items():
        np.save(output / f'{name}_validation.npy', value, allow_pickle=False)
    atomic(output / 'validation_provenance.json', dict(rows=len(x), feature_names=manifest['feature_names'],
        input_hashes=pins, test_loaded=False, selected_frozen_before_load=True,
        novel_reference_scope=manifest['novel_reference_scope']))
    return x, np.asarray(admitted['y']), np.asarray(admitted['row_ids']), novel


def csv_features(row):
    return ['' if np.isnan(value) else float(value) for value in row]


def assess(study, output):
    study, output = Path(study), Path(output)
    settings, manifest = verify_study(study)
    records, repeats = frozen_records(study, settings)
    selection_hash = digest(study / 'selected.json')
    output.mkdir(parents=True, exist_ok=True)
    x, y, ids, novel = load_validation(study, manifest, output)
    results, exports = {}, {}
    review = output / 'validation_cases.csv'
    with review.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'source_row_id', 'csv_label', 'prediction', 'confidence',
                         'attack_probability', 'novel_seven_count_vector', *manifest['feature_names']])
        for name, record in records.items():
            values = predict(record, x)
            validate_output(values, len(y), len(manifest['classes']))
            np.savez_compressed(output / f'{name}_validation_outputs.npz', **values)
            correction = record['calibration']
            confidence, attack, _ = apply(values, correction)
            for i in range(len(y)):
                writer.writerow([name, int(ids[i]), manifest['classes'][y[i]], manifest['classes'][values['pred'][i]],
                                 float(confidence[i]), float(attack[i]), bool(novel[i]), *csv_features(x[i])])
            results[name] = dict(full=measure(values, y, manifest, correction),
                novel_vectors=measure({k: v[novel] for k, v in values.items()}, y[novel], manifest, correction) if novel.any() else None,
                repeated_seeds={})
            for seed, repeat in repeats[name].items():
                replay = predict(repeat, x)
                validate_output(replay, len(y), len(manifest['classes']))
                np.savez_compressed(output / f'{name}_seed{seed}_validation_outputs.npz', **replay)
                results[name]['repeated_seeds'][seed] = dict(full=measure(replay, y, manifest, repeat['calibration']),
                    novel_vectors=measure({k: v[novel] for k, v in replay.items()}, y[novel], manifest, repeat['calibration']) if novel.any() else None)
            exports[name] = dict(family=name, checkpoint=record['checkpoint'],
                calibration=relative(ROOT / record['run_directory'] / 'calibration.json'),
                source_result=relative(ROOT / record['run_directory'] / 'result.json'),
                artifact_hashes=record['artifact_hashes'], auxiliary=name == 'tabpack',
                backend='CPU Python checkpoint; native Rust export pending', deployment_status='research_candidate_not_promoted')
            print(f'ASSESSED {name} CSV accuracy={results[name]["full"]["accuracy"]:.5f}', flush=True)
    if digest(study / 'selected.json') != selection_hash:
        raise ValueError('Selection changed during assessment')
    atomic(output / 'assessment.json', dict(status='completed', rows=len(y), novel_rows=int(novel.sum()),
        models=results, scope='historically reused within-capture CSV validation; not live accuracy',
        limitation=manifest['warning'], selection_digest=selection_hash, selected_frozen_before_assessment=True,
        novel_reference_scope=manifest['novel_reference_scope'], test_loaded=False,
        source_extractor_equivalence_verified=False, unlabelled_npcap_accuracy=None))
    export_path = study / 'exports/manifest.json'
    atomic(export_path, dict(feature_names=manifest['feature_names'], feature_profile=settings['feature_profile'],
        feature_profile_sha256=settings['feature_profile_sha256'], window_seconds=10, classes=manifest['classes'],
        benign_index=0, models=exports, preprocessing='per-checkpoint fit-only transform; preserve unknown NaN',
        zero_traffic_policy='no prediction', invalid_feature_policy='reject structurally invalid window',
        deployment_status='research_candidate_not_promoted', test_loaded=False))
    files = [p for p in output.iterdir() if p.suffix in ('.json', '.csv', '.npy', '.npz')
             and p.name not in ('result.json', 'failure.json')]
    atomic(output / 'result.json', dict(status='completed', phase='assessment', finished_at=time.time(),
        test_loaded=False, artifact_hashes={relative(p): digest(p) for p in files + [export_path]}))
