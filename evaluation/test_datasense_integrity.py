"""Synthetic fixtures for cleaning semantics and leakage-preserving selection."""
import csv
import gzip
import json

import numpy as np
import pytest

from experiment.audit_datasense_raw import LISTS
from experiment.datasense_integrity import clean_row, timestamp_ns, validate_numeric
from experiment.verify_clean_datasense import input_groups, interval_statistics, recompute_derived


def fixture_row():
    row = {'device_name': 'sensor', 'device_mac': '00:11:22:33:44:55',
           'label1': 'attack', 'label2': 'recon', 'label3': 'host-disc-udp-ping',
           'label4': 'recon_host-disc-udp-ping', 'label_full': 'attack_recon_host-disc-udp-ping_sensor',
           'timestamp_start': '2025-01-01T00:00:00.123000Z',
           'timestamp_end': '2025-01-01T00:00:10.123000Z'}
    row['timestamp'] = row['timestamp_start'] + '_' + row['timestamp_end']
    for field in LISTS:
        row[field] = '[]'; row[field + '_count'] = '0'
    row.update(network_packets_all_count='10', network_packets_src_count='4',
               network_packets_dst_count='6', log_messages_count='0',
               **{'network_time-delta_min': '-0.1', 'network_time-delta_avg': '0.1',
                  'network_time-delta_max': '1', 'network_time-delta_std_deviation': '0.2',
                  'network_window-size_min': '0', 'network_window-size_avg': '0',
                  'network_window-size_max': '0', 'network_window-size_std_deviation': '0',
                  'log_data-ranges_min': '0', 'log_data-ranges_avg': '0',
                  'log_data-ranges_max': '0', 'log_data-ranges_std_deviation': '0'})
    for d in ('all', 'src', 'dst'):
        row[f'network_protocols_{d}'] = "['tcp']"; row[f'network_protocols_{d}_count'] = '1'
    numeric = [k for k in row if k not in LISTS and k not in
               ('device_name', 'device_mac', 'label_full', 'label1', 'label2', 'label3', 'label4',
                'timestamp', 'timestamp_start', 'timestamp_end')]
    return row, numeric


def test_signed_data_zero_window_and_unsupported_summaries_are_distinct():
    row, numeric = fixture_row()
    clean, facts, _, events = clean_row(row, numeric, LISTS)
    assert clean['network_time-delta_min'] == '-0.1'
    assert clean['network_window-size_min'] == '0'
    assert clean['log_data-ranges_min'] == ''
    assert facts['negative_delta'] and not facts['hard_invalid']
    assert any(e['rule'] == 'zero_event_summary_unknown' for e in events)


def test_malformed_and_nonfinite_values_are_unknown_not_guessed():
    row, numeric = fixture_row()
    row['network_packets_src_count'] = 'oops'
    row['network_time-delta_max'] = 'inf'
    row['network_ports_src'] = "__import__('os').getcwd()"
    clean, facts, observed, _ = clean_row(row, numeric, LISTS)
    assert clean['network_packets_src_count'] == clean['network_time-delta_max'] == ''
    assert clean['network_ports_src'] == '' and observed['network_ports_src'] is None
    assert facts['hard_invalid']


def test_wrong_counts_and_directions_do_not_create_new_measurements():
    row, numeric = fixture_row()
    row.update(network_macs_src="['00:11:22:33:44:55']", network_macs_src_count='1',
               network_macs_dst="['00:11:22:33:44:55','ff:ff:ff:ff:ff:ff']", network_macs_dst_count='1',
               network_macs_all="['00:11:22:33:44:55']", network_macs_all_count='2',
               network_ports_all='[80]', network_ports_all_count='3000',
               network_ports_src='[80,443]', network_ports_src_count='2',
               network_ports_dst='[53]', network_ports_dst_count='1')
    clean, facts, observed, events = clean_row(row, numeric, LISTS)
    assert clean['network_macs_dst_count'] == ''
    assert observed['network_ports_all'] == (53, 80, 443)
    assert clean['network_ports_all_count'] == '3000'
    assert facts['recovered_members'] == 3
    assert 'network_ports_all' in facts['incomplete_fields']
    assert any(e['rule'] == 'recover_observed_directional_members' for e in events)


def test_small_list_under_count_and_fractional_counts_are_detected():
    row, numeric = fixture_row()
    row['network_ports_src'] = '[1,2,3]'; row['network_ports_src_count'] = '2'
    row['network_packets_src_count'] = '3.5'
    clean, facts, _, _ = clean_row(row, numeric, LISTS)
    assert clean['network_ports_src_count'] == clean['network_packets_src_count'] == ''
    assert facts['hard_invalid']


def test_absent_udp_is_review_only_and_never_a_label_repair():
    row, numeric = fixture_row()
    clean, facts, _, _ = clean_row(row, numeric, LISTS)
    assert facts['udp_evidence_gap'] and not facts['hard_invalid']
    for field in ('label_full', 'label1', 'label2', 'label3', 'label4'):
        assert clean[field] == row[field]


def test_label_contradiction_and_bad_time_require_review():
    row, numeric = fixture_row()
    row['label4'] = 'benign'
    row['timestamp_end'] = row['timestamp_start']
    clean, facts, _, _ = clean_row(row, numeric, LISTS)
    assert facts['hard_invalid'] and clean['label4'] == 'benign'


def test_multi_device_or_whole_network_scenario_is_not_a_label_error():
    row, numeric = fixture_row()
    for suffix in ('sensor--mqtt-broker', 'whole-network', 'another-device'):
        row['label_full'] = 'attack_recon_host-disc-udp-ping_' + suffix
        _, facts, _, events = clean_row(row, numeric, LISTS)
        assert not facts['hard_invalid']
        assert not any(e['rule'] == 'label_hierarchy_requires_source_review' for e in events)


def test_tcp_flags_can_coexist_but_one_flag_cannot_exceed_packets():
    assert not validate_numeric({'network_packets_all_count': 10,
                                'network_tcp-flags-syn_count': 8, 'network_tcp-flags-ack_count': 8})
    assert 'network_tcp-flags-syn_count' in validate_numeric({'network_packets_all_count': 10,
                                                           'network_tcp-flags-syn_count': 11})


def test_summary_order_and_ttl_bounds_are_semantic():
    issues = validate_numeric({'network_ttl_min': 257, 'network_ttl_avg': 256,
                               'network_ttl_max': 255, 'network_ttl_std_deviation': 0})
    assert len(issues) == 4
    assert not validate_numeric({'log_data-ranges_min': -10, 'log_data-ranges_avg': -5,
                                 'log_data-ranges_max': 0, 'log_data-ranges_std_deviation': 2})


def test_nan_equal_conflict_groups_preserve_same_label_repetition():
    X = np.array([[0, np.nan], [-0., np.nan], [1, 2], [1, 2], [3, 4]], dtype=np.float32)
    _, first, inverse, counts, conflict = input_groups(X, np.array([0, 1, 2, 2, 3]))
    np.testing.assert_array_equal(conflict[inverse], [True, True, False, False, False])
    assert len(first) == 3 and counts.sum() == 5


def test_interval_coverage_merges_overlaps_without_inventing_gaps():
    result = interval_statistics(np.array([0, 5, 25])*10**9, np.array([10, 15, 30])*10**9)
    assert result['union_seconds'] == 20 and result['span_seconds'] == 30
    assert result['gaps'] == result['overlapping_windows'] == 1


def test_timezone_is_explicit_and_nanosecond_conversion_exact():
    assert timestamp_ns('2025-01-01T00:00:00.123000Z') % 10**9 == 123000000
    assert timestamp_ns('2025-01-01T05:30:00.123000+05:30') == timestamp_ns('2025-01-01T00:00:00.123000Z')
    with pytest.raises(ValueError, match='Timezone'):
        timestamp_ns('2025-01-01T00:00:00')


def test_recomputed_port_membership_keeps_known_positives_and_unknown_absence(tmp_path):
    row, numeric = fixture_row()
    row['network_ports_src'] = '[80]'
    row['network_ports_src_count'] = '100'
    for name, value in {'network_packet-size_std_deviation': '2', 'network_packet-size_avg': '100',
                        'network_payload-length_avg': '50'}.items():
        row[name] = value; numeric.append(name)
    for flag in ('ack', 'fin', 'psh', 'rst', 'syn', 'urg'):
        name = f'network_tcp-flags-{flag}_count'; row[name] = '0'; numeric.append(name)
    names = numeric + ['network_ports_src__present_80', 'network_ports_src__present_443',
        'network_ports_src__present_UNSEEN', 'window_duration_seconds',
        'quality_no_packets_observed', 'quality_no_logs_observed', 'quality_negative_packet_delta',
        'network_src_packet_fraction', 'network_dst_packet_fraction',
        'network_packet_size_coefficient_of_variation', 'network_payload_fraction',
        'network_packets_per_reported_ip', 'network_packets_per_reported_port']
    names += [f'network_tcp-flags-{flag}__fraction_of_packets' for flag in ('ack', 'fin', 'psh', 'rst', 'syn', 'urg')]
    names += [field + '__per_second' for field in ('network_packets_all_count', 'network_packets_src_count',
                                                  'network_packets_dst_count', 'log_messages_count')]
    X = np.lib.format.open_memmap(tmp_path/'X.npy', mode='w+', dtype='float32', shape=(1,len(names)))
    for name in numeric:
        X[0, names.index(name)] = float(row[name])
    path = tmp_path/'clean.csv.gz'
    with gzip.open(path, 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, ['source_row_id'] + list(row))
        writer.writeheader(); writer.writerow({'source_row_id': 123, **row})
    np.save(tmp_path/'window_start.npy', [0]); np.save(tmp_path/'window_end.npy', [10*10**9])
    incomplete = np.zeros((1,len(LISTS)), dtype=bool)
    incomplete[0,LISTS.index('network_ports_src')] = True
    recompute_derived(X, names, numeric, incomplete, [123], tmp_path, path)
    assert X[0,names.index('network_ports_src__present_80')] == 1
    assert np.isnan(X[0,names.index('network_ports_src__present_443')])
    assert np.isnan(X[0,names.index('network_ports_src__present_UNSEEN')])
    assert X[0,names.index('network_packets_all_count__per_second')] == 1
    assert X[0,names.index('network_payload_fraction')] == .5
