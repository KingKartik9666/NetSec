"""Partial feature alignment is a measured-input contract, independent of labels."""
import json
import numpy as np
import pytest

from evaluation.test_ids_app import detector
from framework.detection.trained_xgboost import XGBoostDetector, transform_csv


@pytest.fixture
def partial_detector(detector):
    bundle = dict(detector.bundle, schema_version=2, partial_input_supported=True)
    detector.bundle_path.write_text(json.dumps(bundle))
    return XGBoostDetector(detector.bundle_path)


def test_partial_csv_aligns_names_keeps_zero_and_missing_flags(partial_detector):
    content = b'network_b,network_protocols_src__present_UNSEEN,label4\n0,1,ignore\n,0,ignore\n'
    X = transform_csv(content, partial_detector.bundle)
    np.testing.assert_equal(X, [[np.nan, 0, np.nan, 1, np.nan], [np.nan, np.nan, np.nan, 0, np.nan]])
    prediction = partial_detector.predict_batch(X)
    assert len(prediction) == 2 and all(len(p['top_classes']) == 3 for p in prediction)
    assert all(p['tentative'] for p in prediction)


@pytest.mark.parametrize('content', [b'network_a\nhello\n', b'network_a\ninf\n',
    b'network_a,network_a\n1,2\n', b'network_a,network_b\n,\n', b'wrong_column\n1\n',
    b'network_protocols_src__present_UNSEEN\n2\n', b'network_a\n1,2\n',
    b'network_a,network_b\n1\n', b'network_a\n1e80\n',
    b'network_protocols_all,network_protocols_all__present_UNSEEN\n[],0\n'])
def test_partial_csv_rejects_invalid_or_empty_measurements(partial_detector, content):
    with pytest.raises(ValueError):
        transform_csv(content, partial_detector.bundle)


def test_raw_protocol_missing_is_different_from_observed_empty_list(partial_detector):
    X = transform_csv(b'network_a,network_protocols_all\n1,\n2,[]\n', partial_detector.bundle)
    assert np.isnan(X[0, 2]) and X[1, 2] == 0
    assert np.isnan(X[:, 3:]).all()


def test_missing_inputs_remain_visible_in_actual_report(partial_detector):
    report = partial_detector.analyze([0, None, None, 1, None], {'kind': 'Partial test fixture'})
    assert report['features'] == [0, None, None, 1, None]
    coverage = report['feature_coverage']
    assert (coverage['observed'], coverage['missing'], coverage['total']) == (2, 3, 5)
    assert coverage['profile'] == 'partial_custom' and not coverage['validated_profile']
    assert report['explanation']['reconstruction_error'] < 1e-4


def test_live_vector_csv_round_trip_uses_only_shared_column_names():
    from framework.data.feature_schema import PACKET_COLUMNS, packet_vector, vector_csv
    names = ['log_absent', *PACKET_COLUMNS, 'network_protocols_all__present_UNSEEN',
             'network_protocols_src__present_UNSEEN', 'network_protocols_dst__present_UNSEEN']
    bundle = {'schema_version': 2, 'partial_input_supported': True, 'feature_names': names,
              'numeric_features': names[:8], 'protocol_vocabulary': {f'network_protocols_{p}': [] for p in ['all','src','dst']}}
    values = packet_vector({'packets': 12, 'ack': 7, 'fin': 0, 'psh': 3, 'rst': 0, 'syn_all': 2, 'urg': 0, 'seconds': 10}, names)
    assert np.isfinite(values).sum() == 7
    np.testing.assert_equal(transform_csv(vector_csv(values, names), bundle)[0], values)
    with pytest.raises(ValueError):
        packet_vector({'packets': 0, 'seconds': 10}, names)
    with pytest.raises(ValueError, match='every measured count'):
        packet_vector({'packets': 12, 'seconds': 10}, names)


def test_default_bundle_requires_explicit_successful_promotion(tmp_path):
    from framework.detection.trained_xgboost import default_bundle
    baseline = tmp_path / 'framework/models/xgboost_app/bundle.json'
    partial = tmp_path / 'framework/models/xgboost_partial/bundle.json'
    enhanced = tmp_path / 'framework/models/xgboost_partial_enhanced/bundle.json'
    for path in [baseline, partial, enhanced]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'promoted': False, 'promotion': {'improved': True}}))
    assert default_bundle(tmp_path) == baseline
    partial.write_text(json.dumps({'promoted': True, 'promotion': {'improved': True}}))
    assert default_bundle(tmp_path) == partial
    enhanced.write_text(json.dumps({'promoted': True, 'promotion': {'improved': False}}))
    assert default_bundle(tmp_path) == partial
    enhanced.write_text(json.dumps({'promoted': True, 'promotion': {'improved': True}}))
    assert default_bundle(tmp_path) == enhanced
    enhanced.write_text('{"promoted":')
    with pytest.raises(json.JSONDecodeError):
        default_bundle(tmp_path)  # Corruption must not silently select another model.
@pytest.mark.parametrize('counts', [
    [-1, 0, 0, 0, 0, 0, 0],
    [10, 11, 0, 0, 0, 0, 0],
    [10, 1.5, 0, 0, 0, 0, 0],
    [2**24 + 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
])
def test_invalid_packet_count_values_are_outside_evaluated_profile(counts):
    from framework.data.feature_schema import PACKET_COLUMNS, feature_coverage
    names = list(PACKET_COLUMNS) + ['log_unavailable']
    bundle = {'feature_names': names, 'live_profile_validation': {'validation_rows': 1571}}
    coverage = feature_coverage([*counts, np.nan], bundle)
    assert coverage['profile'] == 'packet_counts_7'
    assert coverage['validated_profile'] is False
    assert 'packet' in coverage['validation_scope'].lower()


def test_valid_packet_counts_with_genuine_zero_flags_keep_profile_validation():
    from framework.data.feature_schema import PACKET_COLUMNS, feature_coverage
    bundle = {'feature_names': [*PACKET_COLUMNS, 'log_unavailable'],
              'live_profile_validation': {'validation_rows': 1571}}
    coverage = feature_coverage([10, 0, 0, 0, 0, 0, 0, np.nan], bundle)
    assert coverage['validated_profile'] is True
