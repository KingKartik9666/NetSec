"""Conservative, row-local repairs for the isolated DataSense copy.

Missing measurements stay missing. A scenario label is never inferred from a
prediction, protocol, device identity, or another row's label.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import re

from experiment.audit_datasense_raw import kind, parsed

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
SUMMARY_SUFFIXES = ('_min', '_avg', '_max', '_std_deviation')
MAC = re.compile(r'[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5}')


def timestamp_ns(text):
    value = datetime.fromisoformat(text.replace('Z', '+00:00'))
    if value.tzinfo is None:
        raise ValueError('Timezone is required; do not guess it')
    delta = value - EPOCH
    return ((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds) * 1000


def validate_numeric(values):
    """Return invalid fields with reasons, without using distribution cutoffs."""
    invalid = {}
    for field, value in values.items():
        if not math.isfinite(value):
            invalid[field] = 'invalid_or_nonfinite_numeric'
        elif field.endswith('_count') or field == 'network_fragmented-packets':
            if value < 0 or value != math.floor(value):
                invalid[field] = 'negative_or_fractional_count'
        elif field.endswith('_std_deviation') and value < 0:
            invalid[field] = 'negative_standard_deviation'
        elif field.startswith(('network_header-length_', 'network_ip-length_',
                               'network_mss_', 'network_packet-size_',
                               'network_payload-length_', 'network_ttl_',
                               'network_window-size_')) and value < 0:
            invalid[field] = 'negative_unsigned_measurement'
        if field.startswith('network_ttl_') and not field.endswith('_std_deviation'):
            if math.isfinite(value) and not 0 <= value <= 255:
                invalid[field] = 'ttl_outside_byte_range'
    for group in {f[:-4] for f in values if f.endswith('_avg')}:
        fields = [group + suffix for suffix in SUMMARY_SUFFIXES]
        if not all(f in values for f in fields):
            continue
        low, avg, high, _ = [values[f] for f in fields]
        tolerance = 1e-10 * max(1., abs(low), abs(high))
        if low > avg + tolerance or avg > high + tolerance:
            for field in fields:
                invalid[field] = 'summary_order_contradiction'
    packets = values.get('network_packets_all_count', math.nan)
    for field, value in values.items():
        if field.startswith('network_tcp-flags-') and field.endswith('_count') and value > packets:
            invalid[field] = 'individual_flag_count_exceeds_packets'
    return invalid


def clean_row(row, numeric, lists):
    """Return cleaned cells, row facts, parsed observed sets, and change events.

    Original 94 cells are separately archived by the caller. Events identify
    repairs versus unresolved review findings. Blank numeric cells mean unknown.
    """
    clean = dict(row)
    events = []
    fatal = False

    def event(rule, fields, severity='review', before=None, after=None):
        events.append({'rule': rule, 'fields': fields, 'severity': severity,
                       'before': before, 'after': after})

    def patch(rule, changes):
        changes = {k: v for k, v in changes.items() if clean[k] != v}
        if changes:
            before = {k: clean[k] for k in changes}
            clean.update(changes)
            event(rule, list(changes), 'repair', before, changes)

    values = {}
    for field in numeric:
        try:
            values[field] = float(row[field])
        except (ValueError, TypeError):
            values[field] = math.nan
    invalid = validate_numeric(values)
    for reason in sorted(set(invalid.values())):
        patch(reason, {f: '' for f, r in invalid.items() if r == reason})
    fatal |= bool(invalid)

    observed, incomplete = {}, set()
    for field in lists:
        try:
            tokens, count = parsed(row[field], kind(field))
            observed[field] = tokens
            # A standard JSON set representation; order carries no asserted meaning.
            canonical = json.dumps(tokens, separators=(',', ':'))
            clean[field] = canonical
            if count != len(tokens):
                event('duplicate_list_members_removed', [field], 'repair', count, len(tokens))
        except ValueError:
            observed[field] = None
            patch('invalid_list_unknown', {field: ''})
            incomplete.add(field)
            fatal = True
            continue
        reported = values.get(field + '_count', math.nan)
        if reported != len(tokens):
            incomplete.add(field)
            event('reported_count_disagrees_with_list', [field, field + '_count'],
                  before=reported if math.isfinite(reported) else None, after=len(tokens))

    # Reconstruct only membership actually present in one of the three lists.
    # This is still a lower bound on coverage of the original packet capture.
    recovered = 0
    for group in ('ips', 'macs', 'ports', 'protocols'):
        fields = [f'network_{group}_{d}' for d in ('all', 'src', 'dst')]
        if any(observed.get(f) is None for f in fields):
            continue
        combined = tuple(sorted(set().union(*(observed[f] for f in fields))))
        old = observed[fields[0]]
        if combined != old:
            added = sorted(set(combined).difference(old))
            recovered += len(added)
            clean[fields[0]] = json.dumps(combined, separators=(',', ':'))
            observed[fields[0]] = combined
            incomplete.add(fields[0])
            event('recover_observed_directional_members', [fields[0]], 'repair', None, added)
        if set(observed[fields[0]]) != set(observed[fields[1]]).union(observed[fields[2]]):
            event('directional_membership_not_recoverable', fields)

    # This source column exactly copies the source-direction count. Do not
    # pretend to know the complete destination cardinality from a suspect field.
    dst = 'network_macs_dst'
    if dst in observed and observed[dst] is not None:
        patch('suspect_destination_mac_count_withheld', {dst + '_count': ''})
        if values[dst + '_count'] != len(observed[dst]):
            incomplete.add(dst)

    for field, tokens in observed.items():
        if tokens is not None:
            count_field = field + '_count'
            if values.get(count_field, math.inf) < len(tokens):
                # Approximate/exported counts cannot be repaired to capture truth.
                patch('count_below_observed_lower_bound', {count_field: ''})
                incomplete.add(field)

    # Zero-event summaries have no supporting observations. Positive or signed
    # data in a nonempty window stay intact, including valid zero TCP windows.
    unsupported = {}
    for field in numeric:
        count_field = 'log_messages_count' if field.startswith('log_') else 'network_packets_all_count'
        if field.endswith(SUMMARY_SUFFIXES) and values.get(count_field) == 0 and values[field] == 0:
            unsupported[field] = ''
    patch('zero_event_summary_unknown', unsupported)

    packets = values.get('network_packets_all_count', math.nan)
    src, dst_count = (values.get('network_packets_' + d + '_count', math.nan) for d in ('src', 'dst'))
    if all(math.isfinite(v) for v in (packets, src, dst_count)) and packets != src + dst_count:
        fatal = True
        event('packet_direction_totals_contradict',
              ['network_packets_all_count', 'network_packets_src_count', 'network_packets_dst_count'])

    try:
        start, end = (timestamp_ns(row['timestamp_' + p]) for p in ('start', 'end'))
        if end <= start:
            raise ValueError('Nonpositive window')
        duration = (end - start) / 1e9
        composite = row['timestamp_start'] + '_' + row['timestamp_end']
        patch('timestamp_composite_reconstructed', {'timestamp': composite})
    except (ValueError, TypeError, OverflowError):
        start, end, duration = 0, 0, math.nan
        fatal = True
        event('invalid_window_timestamp', ['timestamp', 'timestamp_start', 'timestamp_end'])

    if not MAC.fullmatch(row.get('device_mac', '')) or not row.get('device_name', '').strip():
        fatal = True
        event('invalid_device_provenance', ['device_mac', 'device_name'])
    benign = row.get('label1') == 'benign'
    expected = 'benign' if benign else row.get('label2', '') + '_' + row.get('label3', '')
    hierarchy_ok = row.get('label1') in ('benign', 'attack') and row.get('label4') == expected
    if benign:
        hierarchy_ok &= row.get('label2') == row.get('label3') == 'benign' and row.get('label_full', '').startswith('benign_')
    else:
        # The suffix describes a scenario and can name multiple devices, or the
        # whole network. It is not required to equal this observation's device.
        prefix = '_'.join(row.get(f, '') for f in ('label1', 'label2', 'label3')) + '_'
        hierarchy_ok &= row.get('label_full', '').startswith(prefix) and len(row.get('label_full', '')) > len(prefix)
    if not hierarchy_ok:
        fatal = True
        event('label_hierarchy_requires_source_review', ['label_full', 'label1', 'label2', 'label3', 'label4'])
    no_signal = (all(v == 0 for v in values.values()) and all(v == () for v in observed.values()))
    negative_delta = values.get('network_time-delta_min', 0) < 0
    if negative_delta:
        event('signed_packet_delta_retained', ['network_time-delta_min'])
    if no_signal:
        event('no_observed_measurements', numeric + lists)
    protocols = observed.get('network_protocols_all')
    udp_gap = row.get('label4') == 'recon_host-disc-udp-ping' and protocols is not None and 'udp' not in protocols
    if udp_gap:
        event('udp_scenario_without_exported_udp_evidence', ['label4', 'network_protocols_all'])
    facts = {'hard_invalid': fatal, 'no_signal': no_signal, 'negative_delta': negative_delta,
             'list_uncertainty': bool(incomplete), 'recovered_members': recovered,
             'udp_evidence_gap': udp_gap, 'start': start, 'end': end, 'duration': duration,
             'numeric_missing_cells': sum(clean[f] == '' for f in numeric),
             'incomplete_fields': sorted(incomplete)}
    return clean, facts, observed, events
