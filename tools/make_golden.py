"""Freeze reproducible Python reference fixtures for the staged Rust migration.

Run with the project environment: python tools/make_golden.py --output <new-dir>.
This fits only tiny synthetic CPU models. It never reads research arrays, starts
capture, invokes a language model, or changes deployed/training artifacts.
Existing output is refused; --verify only validates an existing sealed corpus.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import platform
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def json_safe(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True,
                               ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def file_inventory(directory: Path) -> dict:
    return {path.relative_to(directory).as_posix(): {'sha256': digest(path), 'bytes': path.stat().st_size}
            for path in sorted(directory.rglob('*')) if path.is_file() and path != directory / 'manifest.json'}


def seal_manifest(directory: Path, summary: dict):
    write_json(directory / 'manifest.json', {'schema_version': 1, **summary, 'files': file_inventory(directory)})


def verify_manifest(directory: Path) -> dict:
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    actual = file_inventory(directory)
    if set(actual) != set(manifest['files']):
        raise ValueError('Golden fixture file set differs from the sealed manifest')
    for name, info in actual.items():
        if info != manifest['files'][name]:
            raise ValueError(f'Golden fixture checksum differs: {name}')
    return {'files': len(actual), 'bytes': sum(item['bytes'] for item in actual.values()), 'verified': True}


def csv_bytes(header, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def record_call(function, *args):
    try:
        return function(*args)
    except Exception as error:
        # Characterization records existing failures, including unexpected
        # runtime bugs; callers/tests distinguish their type from valid output.
        return {'error_type': type(error).__name__, 'error': str(error)}


def _csv_goldens(folder, bundle, full):
    from framework.detection.trained_xgboost import transform_csv
    names, numeric = bundle['feature_names'], bundle['numeric_features']
    fields = list(bundle['protocol_vocabulary'])
    prepared = csv_bytes(names, [full])
    raw = csv_bytes([*numeric, *fields], [[*full[:len(numeric)], "['tcp', 'tcp']", "['tcp']", "['other']"]])
    cases = [
        ('v1_prepared', 1, prepared), ('v2_prepared', 2, prepared),
        ('v1_raw_python_lists', 1, raw), ('v2_raw_python_lists', 2, raw),
        ('reordered_with_label', 2, csv_bytes(['label4', *reversed(names)], [['ignored', *reversed(full)]])),
        ('utf8_bom', 2, b'\xef\xbb\xbf' + prepared),
        ('v2_observed_empty_protocol', 2, csv_bytes([numeric[0], fields[0]], [[10, '[]']])),
        ('v2_missing_protocol', 2, csv_bytes([numeric[0], fields[0]], [[10, '']])),
        ('v2_unknown_protocol', 2, csv_bytes([numeric[0], fields[0]], [[10, "['future-protocol']"]])),
        ('v2_numeric_missing', 2, csv_bytes(numeric[:2], [[10, '']])),
        ('v2_measured_zero', 2, csv_bytes(numeric[:2], [[10, 0]])),
        ('v1_partial_rejected', 1, csv_bytes(numeric[:1], [[10]])),
        ('duplicate_headers', 2, b'network_packets_all_count,network_packets_all_count\n1,2\n'),
        ('empty_header', 2, b'network_packets_all_count,\n1,2\n'),
        ('malformed_width', 2, b'network_packets_all_count\n1,2\n'),
        ('malformed_protocol', 2, csv_bytes([numeric[0], fields[0]], [[10, "['tcp'"]])),
        ('non_list_protocol', 2, csv_bytes([numeric[0], fields[0]], [[10, "{'tcp':1}"]])),
        ('mixed_protocol_forms', 2, csv_bytes([numeric[0], fields[0], names[len(numeric)]], [[10, '[]', 1]])),
        ('infinity', 2, csv_bytes(numeric[:1], [['inf']])),
        ('overflow', 2, csv_bytes(numeric[:1], [['1e100']])),
        ('unknown_measurement', 2, b'network_typo\n1\n'),
        ('no_measurements', 2, csv_bytes(numeric[:2], [['', '']])),
        ('invalid_protocol_flag', 2, csv_bytes([numeric[0], names[-1]], [[10, 2]])),
        ('maximum_rows', 2, csv_bytes(numeric[:1], [[10]] * 500)),
        ('too_many_rows', 2, csv_bytes(numeric[:1], [[10]] * 501)),
        ('invalid_utf8', 2, b'\xff\xfe'), ('empty_file', 2, b''),
    ]
    input_dir = folder / 'csv'
    input_dir.mkdir()
    output = []
    for name, version, content in cases:
        path = input_dir / (name + ('.bin' if name == 'invalid_utf8' else '.csv'))
        path.write_bytes(content)
        spec = {**bundle, 'schema_version': version, 'partial_input_supported': version == 2}
        def transform():
            values = transform_csv(content, spec)
            return {'rows': json_safe(values), 'row_count': len(values)}
        output.append({'id': name, 'schema_version': version,
                       'input_file': path.relative_to(folder).as_posix(), 'output': record_call(transform)})
    write_json(folder / 'csv.json', {'synthetic': True, 'cases': output})
    return len(output)


def _reference(folder, values, names, categorical):
    import numpy as np
    path = folder / 'xai_reference.npz'
    medians = np.nanmedian(values, axis=0).astype(np.float32)
    np.savez_compressed(path, training=values, medians=medians)
    write_json(path.with_suffix('.json'), {'feature_names': names, 'sha256': digest(path),
               'categorical_indices': categorical, 'rows': len(values),
               'kind': 'Synthetic migration training fixture; not research observations'})
    return path


def _selection_goldens(folder):
    """Record selection separately from loading; no research folders are read."""
    from tempfile import TemporaryDirectory
    from framework.detection.trained_xgboost import default_bundle
    cases = []
    definitions = [
        ('original', {}),
        ('unpromoted', {'xgboost_partial': {'promoted': False}}),
        ('partial', {'xgboost_partial': {'promoted': True, 'promotion': {'improved': True}}}),
        ('enhanced', {name: {'promoted': True, 'promotion': {'improved': True}}
                      for name in ['xgboost_partial', 'xgboost_partial_enhanced']}),
        ('enhanced_not_improved', {'xgboost_partial_enhanced': {'promoted': True, 'promotion': {'improved': False}},
                                   'xgboost_partial': {'promoted': True, 'promotion': {'improved': True}}}),
        ('corrupt_selected_json', {'xgboost_partial_enhanced': '{invalid'}),
    ]
    for name, inputs in definitions:
        with TemporaryDirectory(prefix='imdai-selection-') as temporary:
            root = Path(temporary)
            for variant, contents in inputs.items():
                path = root / 'framework/models' / variant / 'bundle.json'
                path.parent.mkdir(parents=True)
                if isinstance(contents, str):
                    path.write_text(contents, encoding='utf-8')
                else:
                    write_json(path, contents)
            result = record_call(lambda: {'selected': default_bundle(root).relative_to(root).as_posix()})
            cases.append({'id': name, 'input_bundles': inputs, 'output': result})
    write_json(folder / 'selection.json', {'synthetic': True, 'cases': cases})
    return len(cases)


def _adapter_goldens(folder, values):
    from unittest.mock import patch
    from framework.detection.adapters.xgboost import XGBoostModelAdapter
    cases = []
    for name, enabled, fail in [('normal', True, False), ('explanation_disabled', False, False),
                                ('explanation_failed', True, True)]:
        adapter = XGBoostModelAdapter({'bundle_path': str(folder / 'bundle-v2.json'),
                                      'plugins': {'explanation': enabled}}, ROOT)
        source = {'kind': 'Synthetic adapter contract fixture', 'synthetic': True}
        if fail:
            with patch.object(adapter, '_explain', side_effect=RuntimeError('Injected explanation failure')):
                result = adapter.analyze(values, source)
        else:
            result = adapter.analyze(values, source)
        cases.append({'id': name, 'input': {'values': values, 'source': source,
                      'plugins': {'explanation': enabled}, 'injected_failure': fail}, 'output': result})
    write_json(folder / 'adapter-analysis.json', {'synthetic': True, 'cases': cases,
        'execution': 'Actual XGBoostModelAdapter used by the default process worker; failure injection explicitly marked'})
    return len(cases)


def _packet_goldens(output, training, xgb):
    import numpy as np
    from framework.detection.packet_xgboost import PacketXGBoost, FEATURE_NAMES, STATS_KEYS, CLASSES
    folder = output / 'packet'
    folder.mkdir()
    X = training[:, :7].copy()
    y = (X[:, 5] > X[:, 0] * .5).astype(np.int32)
    model = xgb.train({'objective': 'multi:softprob', 'num_class': 2, 'seed': 42,
                       'device': 'cpu', 'nthread': 2, 'max_depth': 2, 'min_child_weight': 0},
                      xgb.DMatrix(X, label=y, nthread=2), num_boost_round=4)
    path = folder / 'model.ubj'
    model.save_model(path)
    # This measured fit statistic exists solely because PacketXGBoost requires
    # this legacy field; the surrounding fixture explicitly disclaims validation.
    fit_accuracy = float((model.predict(xgb.DMatrix(X, nthread=2)).argmax(1) == y).mean())
    write_json(folder / 'bundle.json', {'schema_version': 1, 'checkpoint': 'model.ubj',
        'checkpoint_sha256': digest(path), 'feature_names': FEATURE_NAMES, 'stats_keys': STATS_KEYS,
        'classes': CLASSES, 'window_seconds': 10, 'live_validation': False, 'test_evaluated': False,
        'validation_metrics': {'accuracy': fit_accuracy}, 'synthetic': True,
        'metric_scope': 'Synthetic training-set fit only; field required by runtime, not validation accuracy'})
    _reference(folder, X, FEATURE_NAMES, [])
    detector = PacketXGBoost(folder / 'bundle.json')
    rows = [("observed", {**dict(zip(STATS_KEYS, map(int, X[0]))), 'seconds': 10.0}),
            ("lower_duration", {**dict(zip(STATS_KEYS, map(int, X[0]))), 'seconds': 9.0}),
            ("upper_duration", {**dict(zip(STATS_KEYS, map(int, X[0]))), 'seconds': 11.0}),
            ("late", {**dict(zip(STATS_KEYS, map(int, X[0]))), 'seconds': 11.01}),
            ("flag_exceeds_total", {**dict(zip(STATS_KEYS, map(int, X[0]))), 'ack': int(X[0, 0] + 1)}),
            ("empty", {**{key: 0 for key in STATS_KEYS}, 'seconds': 10.0})]
    write_json(folder / 'predictions.json', {'synthetic': True,
        'limitations': 'Actual Python inference on a tiny synthetic CPU checkpoint; not capture or research validation.',
        'cases': [{'id': name, 'input': stats, 'output': record_call(detector.predict, stats)} for name, stats in rows]})
    return len(rows)


def generate_model_goldens(output_dir: Path):
    import numpy as np
    import xgboost as xgb
    from framework.detection.packet_xgboost import FEATURE_NAMES
    from framework.detection.trained_xgboost import XGBoostDetector
    from framework.xai.model_evidence import explain_additional

    if xgb.__version__ != '3.1.3':
        raise RuntimeError(f'Golden generation requires xgboost 3.1.3, found {xgb.__version__}')
    output = Path(output_dir)
    if any((output / name).exists() or (output / name).is_symlink() for name in ['models', 'packet']):
        raise FileExistsError('Model golden output already exists; choose a new directory')
    folder = output / 'models'
    folder.mkdir(parents=True)
    numeric = [*FEATURE_NAMES, 'network_signal', 'log_sensor']
    vocabulary = {'network_protocols_all': ['tcp', 'udp'],
                  'network_protocols_src': ['tcp'], 'network_protocols_dst': ['tcp']}
    names = numeric + [f'{field}__present_{token}' for field, tokens in vocabulary.items() for token in [*tokens, 'UNSEEN']]
    classes = ['benign', 'synthetic_probe', 'synthetic_flood']
    rng = np.random.default_rng(42)
    X = np.zeros((120, len(names)), dtype=np.float32)
    X[:, 0] = rng.integers(10, 201, len(X))
    X[:, 1:7] = np.floor(rng.random((len(X), 6)) * X[:, :1])
    X[:, 7] = np.tile([-3., 0., 3.], 40)
    X[:, 8] = rng.normal(size=len(X))
    X[:, 9:] = rng.integers(0, 2, (len(X), len(names) - 9))
    y = np.tile(np.arange(3), 40)
    # Missing is associated with class 2 and zero with class 1. The regression
    # asserts different predictions so the fixture tests real default routing.
    X[(y == 2) & (np.arange(len(X)) % 2 == 0), 7] = np.nan
    X[::7, 8] = np.nan
    X[::11, 10:] = np.nan
    params = {'objective': 'multi:softprob', 'num_class': len(classes), 'max_depth': 3,
              'min_child_weight': 0, 'eta': .3, 'seed': 42, 'device': 'cpu', 'nthread': 2,
              'tree_method': 'hist'}
    matrix = xgb.DMatrix(X, label=y, nthread=2)
    booster = xgb.train(params, matrix, num_boost_round=8)
    checkpoint = folder / 'model.ubj'
    booster.save_model(checkpoint)
    full = [100., 30., 1., 2., 1., 10., 0., 0., .25, 1., 0., 0., 1., 0., 1., 0.]
    anchors = {'full': full, 'missing_numeric': [*full[:7], None, *full[8:]],
               'measured_zero': full.copy(), 'missing_protocols': [*full[:9], *([None] * 7)],
               'packet_counts': [*full[:7], *([None] * 9)],
               'sparse_zero': [0., *([None] * 15)]}
    bundle = {'schema_version': 2, 'partial_input_supported': True, 'checkpoint': 'model.ubj',
        'checkpoint_sha256': digest(checkpoint), 'feature_names': names, 'numeric_features': numeric,
        'protocol_vocabulary': vocabulary, 'classes': classes, 'benign_index': 0, 'synthetic': True,
        'model': {'name': 'Synthetic migration XGBoost fixture', 'run_id': 'migration_synthetic_s42',
                  'validation_accuracy': None, 'macro_f1': None, 'training_rows': len(X),
                  'features': len(names), 'classes': len(classes), 'seed': 42,
                  'selection_basis': 'Synthetic compatibility fixture; no research metrics'},
        'samples': [{'id': i, 'label': classes[i % 3], 'source_row': i,
                     'features': row, 'synthetic': True} for i, row in enumerate(anchors.values())]}
    write_json(folder / 'bundle-v2.json', bundle)
    write_json(folder / 'bundle-v1.json', {**bundle, 'schema_version': 1, 'partial_input_supported': False})
    write_json(folder / 'training.json', {'synthetic': True, 'seed': 42, 'features': names,
               'rows': X, 'labels': y, 'params': params, 'rounds': 8})
    reference = _reference(folder, X, names, list(range(len(numeric), len(names))))
    detector = XGBoostDetector(folder / 'bundle-v2.json')
    predictions, additional = [], []
    for name, values in anchors.items():
        row = detector.validate_features(values).reshape(1, -1)
        matrix = xgb.DMatrix(row, nthread=2)
        prediction = detector.predict_batch(row)[0]
        predictions.append({'id': name, 'input': values, 'prediction': prediction,
            'probabilities': detector.booster.predict(matrix, strict_shape=True)[0],
            'margins': detector.booster.predict(matrix, strict_shape=True, output_margin=True)[0],
            'contributions': detector.booster.predict(matrix, strict_shape=True, pred_contribs=True, approx_contribs=False)[0],
            'analysis': detector.analyze(values, {'kind': 'Synthetic migration fixture', 'synthetic': True, 'case': name})})
        if name in {'full', 'missing_numeric', 'packet_counts'}:
            additional.append({'id': name, 'input': values, 'prediction': prediction,
                'output': explain_additional(detector.booster, row[0], names, classes, reference)})
    write_json(folder / 'predictions.json', {'synthetic': True, 'cases': predictions})
    write_json(folder / 'additional.json', {'synthetic': True, 'cases': additional})
    tie_booster = xgb.train({**params, 'base_score': .5}, xgb.DMatrix(X, label=y, nthread=2), num_boost_round=0)
    tie_booster.save_model(folder / 'ties.ubj')
    write_json(folder / 'bundle-ties.json', {**bundle, 'checkpoint': 'ties.ubj', 'checkpoint_sha256': digest(folder / 'ties.ubj')})
    tie_detector = XGBoostDetector(folder / 'bundle-ties.json')
    write_json(folder / 'ties.json', {'synthetic': True, 'input': full,
        'prediction': tie_detector.predict_batch(np.asarray([full], dtype=np.float32))[0],
        'purpose': 'Equal scores select the first maximum and stable class-index top-three order'})
    csv_count = _csv_goldens(folder, bundle, full)
    selection_count = _selection_goldens(folder)
    adapter_count = _adapter_goldens(folder, full)
    packet_count = _packet_goldens(output, X, xgb)
    return detector, {'xgboost_version': xgb.__version__, 'prediction_cases': len(predictions),
        'lime_cases': len(additional), 'csv_cases': csv_count, 'packet_model_cases': packet_count,
        'selection_cases': selection_count,
        'adapter_cases': adapter_count,
        'max_shap_reconstruction_error': max(case['analysis']['explanation']['reconstruction_error'] for case in predictions),
        'fixture_shape': {'features': len(names), 'classes': len(classes), 'training_rows': len(X)}}


def source_inventory() -> dict:
    paths = [path for path in (ROOT / 'framework').rglob('*') if path.is_file()
             and path.suffix in {'.py', '.js', '.cjs', '.html', '.css', '.json'}
             and not set(path.relative_to(ROOT).parts).intersection({'models', 'node_modules', '__pycache__'})]
    paths += list((ROOT / 'tools').glob('*.py'))
    paths += [ROOT / 'run_app.py', ROOT / 'run_desktop.py']
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(paths)}


def generate(output: Path) -> dict:
    from tools.golden_http import generate_http_goldens
    from tools.golden_observations import generate_observation_goldens
    if output.exists() or output.is_symlink():
        raise FileExistsError('Refusing existing output; use --verify or choose a new version directory')
    before = source_inventory()
    detector, models = generate_model_goldens(output)
    observations = generate_observation_goldens(output)
    http = generate_http_goldens(output, detector)
    if source_inventory() != before:
        raise RuntimeError('Runtime/tool source changed while generating fixtures; corpus not sealed')
    versions = {name: importlib.metadata.version(name) for name in
                ['xgboost', 'numpy', 'pandas', 'scikit-learn', 'lime', 'Flask', 'scapy']}
    summary = {'synthetic': True, 'python_version': platform.python_version(), 'packages': versions,
        'models': models, 'observations': observations, 'http': http, 'sources': before,
        'executed': {'python_synthetic_reference': True, 'rust_parity': False,
                     'real_artifact_parity': False, 'live_capture': False, 'llm_generation': False},
        'limitations': ['Reduced synthetic 16-feature/3-class model preserves artifact layout, not research task size.',
            'HTTP transport doubles are explicitly marked; saved synthetic XGBoost executes real inference.',
            'Hardcoded DataSense/validation wording in captured runtime outputs is compatibility text, not fixture provenance.',
            'No Rust parity numbers exist in Phase 0. No detection accuracy or language-quality claims.']}
    seal_manifest(output, summary)
    return verify_manifest(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluation/fixtures/rust_migration/v1')
    parser.add_argument('--verify', action='store_true', help='Check sealed file hashes without regenerating')
    args = parser.parse_args(argv)
    result = verify_manifest(args.output) if args.verify else generate(args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
