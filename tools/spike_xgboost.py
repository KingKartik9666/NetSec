"""Measure the isolated Rust spike against native XGBoost; no fitting/promotion.

Native model files are hash-verified and read in place. Only comparison inputs
and software parity records are saved in the ignored migration artifact tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.make_golden import write_json


def compare(binary: Path, checkpoint: Path, expected: str, rows: list, directory: Path):
    started = time.perf_counter()
    actual_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual_hash != expected:
        raise ValueError('Checkpoint bytes differ from the frozen bundle')
    directory.mkdir(parents=True, exist_ok=False)
    input_file = directory / 'rows.json'
    write_json(input_file, {'rows': rows})
    child_started = time.perf_counter()
    child = subprocess.run([str(binary), '--checkpoint', str(checkpoint), '--sha256', expected,
                            '--rows', str(input_file)], capture_output=True, text=True, timeout=120)
    rust_seconds = time.perf_counter() - child_started
    if child.returncode:
        raise RuntimeError(f'Rust spike failed: {child.stderr.strip()}')
    rust = json.loads(child.stdout)
    booster = xgb.Booster(model_file=str(checkpoint))
    booster.set_param({'device': 'cpu', 'nthread': 2})
    values = np.asarray([[np.nan if value is None else value for value in row] for row in rows], dtype=np.float32)
    matrix = xgb.DMatrix(values, nthread=2)
    python_started = time.perf_counter()
    probabilities = booster.predict(matrix, strict_shape=True)
    margins = booster.predict(matrix, output_margin=True, strict_shape=True)
    shap = booster.predict(matrix, pred_contribs=True, approx_contribs=False, strict_shape=True)
    python_seconds = time.perf_counter() - python_started
    predictions = np.asarray([row['probabilities'] for row in rust['results']])
    rust_margins = np.asarray([row['margins'] for row in rust['results']])
    rust_shap = np.asarray([row['contributions'] for row in rust['results']])
    expected_top = np.argsort(-probabilities, kind='stable', axis=1)[:, :3]
    score_error = float(np.max(np.abs(predictions - probabilities)))
    margin_error = float(np.max(np.abs(rust_margins - margins)))
    shap_error = float(np.max(np.abs(rust_shap - shap)))
    reconstructed = float(np.max(np.abs(rust_shap.sum(axis=2) - rust_margins)))
    argmax_same = [row['argmax'] for row in rust['results']] == probabilities.argmax(axis=1).tolist()
    top_same = [row['top_three'] for row in rust['results']] == expected_top.tolist()
    result = {'checkpoint_sha256': expected, 'checkpoint_bytes': checkpoint.stat().st_size,
        'rows': len(rows), 'features': rust['features'], 'classes': rust['classes'], 'trees': rust['trees'],
        'max_probability_difference': score_error, 'max_margin_difference': margin_error,
        'max_shap_difference': shap_error, 'max_reconstruction_error': reconstructed,
        'identical_argmax': argmax_same, 'identical_top_three': top_same,
        'rust_process_load_and_all_outputs_seconds': rust_seconds,
        'python_preloaded_predict_and_shap_seconds': python_seconds,
        'elapsed_seconds': time.perf_counter() - started,
        'passed': score_error <= 1e-5 and margin_error <= 1e-5 and shap_error <= 1e-4 and argmax_same and top_same,
        'timing_note': 'Rust includes process startup, decoding and checksum; Python excludes model loading. Not a comparable performance benchmark.'}
    write_json(directory / 'python.json', {'probabilities': probabilities, 'margins': margins, 'contributions': shap})
    write_json(directory / 'rust.json', rust)
    write_json(directory / 'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT/'rust/spikes/xgboost/target/release/netsec-xgboost-spike.exe')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/rust_migration/phase1/xgboost')
    parser.add_argument('--real', action='store_true', help='Also compare existing app bundles without changing them')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Choose a fresh parity artifact directory')
    args.output.mkdir(parents=True)
    binary = args.binary.resolve()
    fixture = ROOT/'evaluation/fixtures/rust_migration/v1'
    cases = {}
    for name, bundle_file, rows in [
        ('synthetic_multiclass', fixture/'models/bundle-v2.json',
         [row['input'] for row in json.loads((fixture/'models/predictions.json').read_text())['cases']]),
        ('synthetic_ties', fixture/'models/bundle-ties.json',
         [json.loads((fixture/'models/ties.json').read_text())['input']]),
        ('synthetic_packet', fixture/'packet/bundle.json',
         [[10, 3, 1, 2, 0, 2, 0], [100, 0, 0, 0, 0, 100, 0]]),
    ]:
        bundle = json.loads(bundle_file.read_text())
        cases[name] = compare(binary, (bundle_file.parent/bundle['checkpoint']).resolve(),
                              bundle['checkpoint_sha256'], rows, args.output/name)
        cases[name]['synthetic'] = True
    if args.real:
        for name in ['xgboost_app', 'xgboost_partial', 'xgboost_partial_enhanced', 'packet_live']:
            path = ROOT/'framework/models'/name/'bundle.json'
            if not path.is_file():
                cases[name] = {'executed': False, 'reason': 'Bundle not present', 'passed': None}
                continue
            bundle = json.loads(path.read_text())
            if name == 'packet_live':
                rows = [[108, 62, 4, 20, 0, 5, 0], [640, 400, 10, 80, 2, 50, 0]]
                source = 'Synthetic count vectors through actual saved packet checkpoint; not a capture'
            else:
                selected = np.unique(np.linspace(0, len(bundle['samples']) - 1,
                                                min(12, len(bundle['samples'])), dtype=int))
                rows = [bundle['samples'][int(index)]['features'] for index in selected]
                rows.extend([[None if i % 3 == phase else value for i, value in enumerate(row)]
                             for phase, row in enumerate(rows[:3])])
                packet_names = set(json.loads((fixture/'packet/bundle.json').read_text())['feature_names'])
                rows.append([value if name in packet_names else None
                             for name, value in zip(bundle['feature_names'], rows[0])])
                source = 'Existing included source-validation samples plus deterministic missing-mask diagnostic; no new labels or test arrays'
            cases[name] = compare(binary, (path.parent/bundle['checkpoint']).resolve(),
                                  bundle['checkpoint_sha256'], rows, args.output/name)
            cases[name].update(synthetic=False, input_provenance=source)
    passed = all(case['passed'] for case in cases.values() if case.get('passed') is not None)
    write_json(args.output/'summary.json', {'schema_version': 1, 'python_xgboost': xgb.__version__,
        'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(), 'cases': cases,
        'go': passed, 'scope': 'Native numeric gbtree/multi:softprob inference and path-dependent TreeSHAP parity only; not detection accuracy.',
        'out_of_scope': ['Categorical splits', 'DART', 'vector leaves', 'SHAP interactions', 'training', 'application migration']})
    print(json.dumps({'go': passed, 'cases': cases}, sort_keys=True))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
