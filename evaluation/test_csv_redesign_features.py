"""CSV encoder contracts: observed evidence, fitting-only vocab, safe replay."""
import copy
import importlib
import json
import math

import numpy as np
import pytest


def api():
    try:
        return importlib.import_module('framework.training.csv_redesign.features')
    except ModuleNotFoundError:
        pytest.fail('The new CSV encoder has not been implemented')


def raw_row():
    module = api()
    row = {name: '0' for name in module.NUMERIC_FIELDS}
    row.update({name: '[]' for name in module.LISTS})
    row.update(device_name='fixture', device_mac='00:11:22:33:44:55',
               label1='benign', label2='benign', label3='benign', label4='benign',
               label_full='benign_fixture', timestamp='old',
               timestamp_start='2026-01-01T00:00:00Z',
               timestamp_end='2026-01-01T00:00:02Z',
               network_packets_all_count='10', network_packets_src_count='6',
               network_packets_dst_count='4', network_ports_all='[80,81,82,50000]',
               network_ports_all_count='4', network_ports_src='[80,81,82]',
               network_ports_src_count='3', network_ports_dst='[80,50000]',
               network_ports_dst_count='2')
    return row


def feature_map(encoder, row):
    return dict(zip(encoder.feature_names, encoder.transform(row)))


def test_new_port_geometry_is_distinct_set_evidence_and_not_packet_frequency():
    row = raw_row()
    encoder = api().CsvEncoder.fit([row], min_port_support=1)
    values = feature_map(encoder, row)
    assert values['network_ports_src__listed_span'] == 2
    assert values['network_ports_src__listed_adjacent_gap_fraction'] == 1
    assert values['network_ports_src__listed_well_known_fraction'] == 1
    assert values['network_ports_dst__listed_dynamic_fraction'] == .5
    assert not any('entropy' in key for key in values)
    assert encoder.feature_groups['port_geometry']
    assert encoder.feature_groups['scan_shape']
    assert not set(encoder.base_indices) & set(encoder.feature_groups['port_geometry'])


def test_incomplete_membership_does_not_invent_absence_or_full_set_overlap():
    row = raw_row()
    encoder = api().CsvEncoder.fit([row], min_port_support=1)
    row['network_ports_src'] = '[80]'
    row['network_ports_src_count'] = '3'
    values = feature_map(encoder, row)
    assert values['network_ports_src__present_80'] == 1
    assert math.isnan(values['network_ports_src__present_81'])
    assert values['network_ports_src__membership_incomplete'] == 1
    assert math.isnan(values['network_ports__complete_direction_jaccard'])
    assert values['network_ports_src__listed_span'] == 0
    assert math.isnan(values['network_ports_src__listed_gap_mean'])


def test_training_vocabulary_is_frozen_and_transform_preserves_unseen():
    row = raw_row()
    encoder = api().CsvEncoder.fit([row], min_port_support=1)
    before = json.dumps(encoder.to_dict(), sort_keys=True)
    row['network_ports_src'] = '[443]'
    row['network_ports_src_count'] = '1'
    values = feature_map(encoder, row)
    assert values['network_ports_src__present_UNSEEN'] == 1
    assert 'network_ports_src__present_443' not in values
    assert json.dumps(encoder.to_dict(), sort_keys=True) == before
    np.testing.assert_array_equal(encoder.transform(row),
                                  api().CsvEncoder.from_dict(json.loads(before)).transform(row))


def test_labels_identity_and_absolute_clock_do_not_change_predictors():
    row = raw_row()
    encoder = api().CsvEncoder.fit([row], min_port_support=1)
    altered = {k: v for k, v in row.items()
               if not k.startswith('label') and k not in ('device_name', 'device_mac')}
    altered.update(timestamp_start='2027-02-01T12:00:00Z',
                   timestamp_end='2027-02-01T12:00:02Z', source_row_id='999')
    np.testing.assert_array_equal(encoder.transform(row), encoder.transform(altered))
    assert not any(key.startswith(('label', 'device', 'timestamp', 'source_row'))
                   for key in encoder.feature_names)
    other = copy.deepcopy(row)
    other.update(label4='made_up', device_mac='secret', label_full='attack_secret')
    assert api().CsvEncoder.fit([other], min_port_support=1).to_dict() == encoder.to_dict()


def test_signed_measurements_and_measured_zero_are_not_clipped():
    row = raw_row()
    row.update({'network_time-delta_min': '-1', 'network_time-delta_avg': '0',
                'network_time-delta_max': '1', 'network_time-delta_std_deviation': '.5'})
    encoder = api().CsvEncoder.fit([row], min_port_support=1)
    values = feature_map(encoder, row)
    assert values['network_time-delta_min'] == -1
    assert values['network_window-size_avg'] == 0
    assert values['network_packets_all_count__per_second'] == 5
    assert math.isnan(values['log_data-ranges_avg'])


def test_no_identity_vocabulary_and_suspect_destination_mac_count_withheld():
    row = raw_row()
    row.update(network_macs_all='["00:11:22:33:44:55"]', network_macs_all_count='1',
               network_macs_src='[]', network_macs_src_count='0',
               network_macs_dst='["00:11:22:33:44:55"]', network_macs_dst_count='7')
    encoder = api().CsvEncoder.fit([row], min_port_support=1)
    values = feature_map(encoder, row)
    assert 'network_macs_dst_count' not in values
    assert values['network_macs_dst__observed_distinct_count'] == 1
    assert not any('00:11' in name for name in values)


def test_malformed_list_never_executes_and_is_unknown(tmp_path):
    row = raw_row()
    encoder = api().CsvEncoder.fit([row], min_port_support=1)
    marker = tmp_path / 'must_not_exist'
    row['network_ports_src'] = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    vector, facts, _ = encoder.encode_row(row)
    assert facts['hard_invalid']
    assert not marker.exists()
    assert math.isnan(vector[encoder.feature_names.index('network_ports_src__present_80')])


def test_schema_rejects_tampered_numeric_names_and_nonfinite_bounds():
    encoder = api().CsvEncoder.fit([raw_row()], min_port_support=1)
    bad = encoder.to_dict()
    bad['feature_names'][0] = 'label4'
    with pytest.raises(ValueError, match='schema'):
        api().CsvEncoder.from_dict(bad)
    with pytest.raises(ValueError):
        api().CsvEncoder.fit([], max_ports_per_direction=0)


def test_all_observed_protocols_survive_rare_frequency():
    row = raw_row()
    row.update(network_protocols_all='["rtsp"]', network_protocols_all_count='1',
               network_protocols_src='["rtsp"]', network_protocols_src_count='1')
    encoder = api().CsvEncoder.fit([row], min_port_support=100)
    assert 'network_protocols_src__present_rtsp' in encoder.feature_names
    assert feature_map(encoder, row)['network_protocols_src__present_rtsp'] == 1
