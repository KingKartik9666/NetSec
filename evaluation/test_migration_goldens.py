"""Reference corpus checks for the Python-to-Rust port; no research accuracy claims."""
import json

import numpy as np
import pytest


def _models(path):
    from tools.make_golden import generate_model_goldens
    return generate_model_goldens(path)


def _bytes(path):
    return {p.relative_to(path).as_posix(): p.read_bytes()
            for p in path.rglob('*') if p.is_file()}


def test_saved_models_reload_and_missing_routing_is_distinct_from_zero(tmp_path):
    from framework.detection.trained_xgboost import XGBoostDetector
    detector, summary = _models(tmp_path)
    fixture = json.loads((tmp_path / 'models/predictions.json').read_text())
    assert summary['xgboost_version'] == '3.1.3'
    assert fixture['synthetic'] is True
    assert len(fixture['cases']) >= 6
    cases = {case['id']: case for case in fixture['cases']}
    assert cases['missing_numeric']['input'] != cases['measured_zero']['input']
    assert cases['missing_numeric']['probabilities'] != cases['measured_zero']['probabilities']
    reloaded = XGBoostDetector(tmp_path / 'models/bundle-v2.json')
    for case in fixture['cases']:
        values = detector.validate_features(case['input'])
        assert reloaded.predict_batch(values.reshape(1, -1))[0] == case['prediction']
        contributions = np.asarray(case['contributions'], dtype=np.float64)
        np.testing.assert_allclose(contributions.sum(axis=1), case['margins'], atol=1e-4, rtol=1e-5)
        assert case['analysis']['reasoning']['llm_executed'] is False
    ties = json.loads((tmp_path / 'models/ties.json').read_text())
    assert ties['prediction']['label'] == detector.classes[0]
    assert [v['label'] for v in ties['prediction']['top_classes']] == detector.classes
    selection = {case['id']: case['output'] for case in
                 json.loads((tmp_path / 'models/selection.json').read_text())['cases']}
    assert selection['unpromoted']['selected'] == 'framework/models/xgboost_app/bundle.json'
    assert selection['partial']['selected'] == 'framework/models/xgboost_partial/bundle.json'
    assert selection['enhanced']['selected'] == 'framework/models/xgboost_partial_enhanced/bundle.json'
    assert selection['corrupt_selected_json']['error_type'] == 'JSONDecodeError'
    adapters = {case['id']: case['output'] for case in
                json.loads((tmp_path / 'models/adapter-analysis.json').read_text())['cases']}
    assert adapters['normal']['capability_errors'] == []
    for name in ['explanation_disabled', 'explanation_failed']:
        assert adapters[name]['prediction'] == adapters['normal']['prediction']
        assert adapters[name]['explanation'] is None and adapters[name]['reasoning'] is None
        assert adapters[name]['capability_errors'][0]['capability'] == 'explain'


def test_csv_goldens_cover_schema_versions_lists_nulls_and_errors(tmp_path):
    _models(tmp_path)
    cases = json.loads((tmp_path / 'models/csv.json').read_text())['cases']
    cases = {case['id']: case for case in cases}
    assert cases['v1_raw_python_lists']['output']['rows']
    empty = cases['v2_observed_empty_protocol']['output']['rows'][0]
    missing = cases['v2_missing_protocol']['output']['rows'][0]
    assert empty != missing
    for name in ['duplicate_headers', 'malformed_width', 'malformed_protocol',
                 'mixed_protocol_forms', 'infinity', 'unknown_measurement',
                 'too_many_rows', 'no_measurements', 'invalid_protocol_flag']:
        assert cases[name]['output']['error_type'] == 'ValueError'
    assert cases['maximum_rows']['output']['row_count'] == 500
    # Existing runtime bug: decoding fails before its local `import csv`, then
    # its exception tuple raises UnboundLocalError. Freeze, do not fix in phase 0.
    assert cases['invalid_utf8']['output']['error_type'] == 'UnboundLocalError'


def test_lime_and_packet_fixture_use_actual_models_and_references(tmp_path):
    _models(tmp_path)
    additional = json.loads((tmp_path / 'models/additional.json').read_text())
    for case in additional['cases']:
        result = case['output']
        assert result['lime']['perturbations'] == 2048
        assert np.isfinite(result['lime']['local_fidelity_r2'])
        assert result['lime']['model_score'] == pytest.approx(case['prediction']['score'])
        assert result['statistics']['model_parameters']['device'] == 'cpu'
    packet = json.loads((tmp_path / 'packet/predictions.json').read_text())
    assert packet['synthetic'] is True
    assert packet['cases'][0]['output']['explanation']['reconstruction_error'] < 1e-4
    assert packet['cases'][-1]['output']['error_type'] == 'ValueError'


def test_generation_is_reproducible_and_never_rewrites_existing_fixtures(tmp_path):
    from tools.make_golden import generate_model_goldens
    one, two = tmp_path / 'one', tmp_path / 'two'
    _models(one)
    _models(two)
    assert _bytes(one) == _bytes(two)
    original = _bytes(one)
    with pytest.raises(FileExistsError):
        generate_model_goldens(one)
    assert _bytes(one) == original


def test_manifest_verification_rejects_tampered_and_extra_files(tmp_path):
    from tools.make_golden import seal_manifest, verify_manifest
    tmp_path.joinpath('sample.json').write_text('{"synthetic":true}', encoding='utf-8')
    seal_manifest(tmp_path, {'synthetic': True})
    verify_manifest(tmp_path)
    tmp_path.joinpath('extra.txt').write_text('unexpected', encoding='utf-8')
    with pytest.raises(ValueError, match='file set'):
        verify_manifest(tmp_path)
    tmp_path.joinpath('extra.txt').unlink()
    tmp_path.joinpath('sample.json').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='checksum'):
        verify_manifest(tmp_path)
