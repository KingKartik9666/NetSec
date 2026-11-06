"""Fresh fitting-only CSV encoding with explicit observed-set semantics.

The conservative row repairs are reused from ``datasense_integrity``. New
features describe *listed distinct ports*, direction overlap and summary shape;
they do not reconstruct packet frequencies or assume complete capped lists.
No target, scenario, device identifier, address or absolute clock is a predictor.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import math
from urllib.parse import quote

import numpy as np

from experiment.audit_datasense_raw import LISTS, META, kind, parsed
from experiment.datasense_integrity import clean_row


SUMMARY_GROUPS = ('header-length', 'ip-flags', 'ip-length', 'mss', 'packet-size',
                  'payload-length', 'tcp-flags', 'time-delta', 'ttl', 'window-size')
SUMMARY_SUFFIXES = ('avg', 'max', 'min', 'std_deviation')
FLAGS = ('ack', 'fin', 'psh', 'rst', 'syn', 'urg')
NUMERIC_FIELDS = tuple(sorted(
    [f'log_data-ranges_{suffix}' for suffix in SUMMARY_SUFFIXES]
    + ['log_data-types_count', 'log_interval-messages', 'log_messages_count']
    + [f'network_{group}_{suffix}' for group in SUMMARY_GROUPS for suffix in SUMMARY_SUFFIXES]
    + [f'network_{group}_{direction}_count' for group in ('ips', 'macs', 'ports', 'protocols', 'packets')
       for direction in ('all', 'dst', 'src')]
    + [f'network_tcp-flags-{flag}_count' for flag in FLAGS]
    + ['network_fragmentation-score', 'network_fragmented-packets', 'network_interval-packets']))
VOCAB_FIELDS = ('log_data-types', 'network_ports_src', 'network_ports_dst',
                'network_protocols_all', 'network_protocols_src', 'network_protocols_dst')
NAN = float('nan')
SCHEMA_VERSION = 1


def _clean(row: Mapping[str, str]):
    missing = set(NUMERIC_FIELDS).difference(row)
    if missing:
        raise ValueError(f'Missing required numeric CSV columns: {sorted(missing)}')
    # Metadata may be absent during inference. The reused validator then records
    # provenance/label warnings in facts, which never enter the feature vector.
    source = {name: '' for name in META + LISTS}
    source.update(row)
    return clean_row(source, list(NUMERIC_FIELDS), LISTS)


def row_facts(row: Mapping[str, str]) -> dict:
    """Row-local quality/provenance facts; never a learned predictor mapping."""
    return _clean(row)[1]


def _number(text):
    try:
        value = float(text)
        return value if math.isfinite(value) else NAN
    except (TypeError, ValueError):
        return NAN


def _ratio(numerator, denominator):
    return (numerator / denominator if math.isfinite(numerator)
            and math.isfinite(denominator) and denominator > 0 else NAN)


def _token_name(token):
    return quote(str(token), safe='-_.')


def _derived(clean, facts, observed, vocabularies):
    """Return stable ordered (name, value, group) entries for one raw row."""
    values = {name: _number(clean[name]) for name in NUMERIC_FIELDS}
    incomplete = set(facts['incomplete_fields'])
    entries = []

    def add(name, value, group):
        entries.append((name, float(value), group))

    def ratio(name, numerator, denominator, group):
        add(name, _ratio(numerator, denominator), group)

    for name in NUMERIC_FIELDS:
        if name == 'network_macs_dst_count':
            tokens = observed['network_macs_dst']
            add('network_macs_dst__observed_distinct_count',
                len(tokens) if tokens is not None else NAN, 'base_numeric')
        else:
            add(name, values[name], 'base_numeric')

    for field in LISTS:
        tokens = observed[field]
        count = len(tokens) if tokens is not None else NAN
        add(field + '__listed_distinct', count, 'base_lists')
        add(field + '__reported_minus_listed', values[field + '_count'] - count, 'base_lists')
        add(field + '__membership_incomplete', field in incomplete or tokens is None, 'base_lists')

    duration = facts['duration']
    add('window_duration_seconds', duration, 'base_rates')
    packets = values['network_packets_all_count']
    for field in ('network_packets_all_count', 'network_packets_src_count',
                  'network_packets_dst_count', 'log_messages_count'):
        ratio(field + '__per_second', values[field], duration, 'base_rates')
    for flag in FLAGS:
        ratio(f'network_tcp-flags-{flag}__fraction_of_packets',
              values[f'network_tcp-flags-{flag}_count'], packets, 'base_rates')
    for direction in ('src', 'dst'):
        ratio(f'network_{direction}_packet_fraction',
              values[f'network_packets_{direction}_count'], packets, 'base_rates')
    for name, numerator, denominator in (
            ('network_packet_size_coefficient_of_variation', 'network_packet-size_std_deviation', 'network_packet-size_avg'),
            ('network_payload_fraction', 'network_payload-length_avg', 'network_packet-size_avg'),
            ('network_packets_per_reported_ip', 'network_packets_all_count', 'network_ips_all_count'),
            ('network_packets_per_reported_port', 'network_packets_all_count', 'network_ports_all_count')):
        ratio(name, values[numerator], values[denominator], 'base_rates')
    add('quality_no_packets_observed', packets == 0 if math.isfinite(packets) else NAN, 'base_rates')
    logs = values['log_messages_count']
    add('quality_no_logs_observed', logs == 0 if math.isfinite(logs) else NAN, 'base_rates')
    delta = values['network_time-delta_min']
    add('quality_negative_packet_delta', delta < 0 if math.isfinite(delta) else NAN, 'base_rates')

    for field, vocabulary in vocabularies.items():
        tokens = observed[field]
        token_set = set(tokens or ())
        unknown = tokens is None or field in incomplete
        for token in vocabulary:
            add(field + '__present_' + _token_name(token),
                1 if token in token_set else (NAN if unknown else 0), 'membership')
        add(field + '__present_UNSEEN', 1 if token_set.difference(vocabulary)
            else (NAN if unknown else 0), 'membership')

    # Statistics concern sorted *unique listed values*, not packet-port counts.
    for direction in ('all', 'src', 'dst'):
        field = f'network_ports_{direction}'
        tokens = observed[field]
        ports = np.asarray(tokens or (), dtype=np.float64)
        gaps = np.diff(ports)
        stats = {
            'min': ports[0] if len(ports) else NAN,
            'max': ports[-1] if len(ports) else NAN,
            'span': ports[-1] - ports[0] if len(ports) else NAN,
            'median': float(np.median(ports)) if len(ports) else NAN,
            'iqr': float(np.percentile(ports, 75) - np.percentile(ports, 25)) if len(ports) else NAN,
            'gap_mean': float(np.mean(gaps)) if len(gaps) else NAN,
            'gap_max': float(np.max(gaps)) if len(gaps) else NAN,
            'adjacent_gap_fraction': float(np.mean(gaps == 1)) if len(gaps) else NAN,
            'well_known_fraction': float(np.mean(ports < 1024)) if len(ports) else NAN,
            'registered_fraction': float(np.mean((ports >= 1024) & (ports < 49152))) if len(ports) else NAN,
            'dynamic_fraction': float(np.mean(ports >= 49152)) if len(ports) else NAN,
        }
        for stat, value in stats.items():
            add(field + '__listed_' + stat, value, 'port_geometry')

    for group in ('ips', 'macs', 'ports', 'protocols'):
        src_field, dst_field = (f'network_{group}_{d}' for d in ('src', 'dst'))
        source, destination = observed[src_field], observed[dst_field]
        valid = source is not None and destination is not None
        src_set, dst_set = set(source or ()), set(destination or ())
        intersection, union = len(src_set & dst_set), len(src_set | dst_set)
        stem = f'network_{group}__'
        add(stem + 'listed_direction_intersection', intersection if valid else NAN, 'direction_overlap')
        add(stem + 'listed_src_only', len(src_set - dst_set) if valid else NAN, 'direction_overlap')
        add(stem + 'listed_dst_only', len(dst_set - src_set) if valid else NAN, 'direction_overlap')
        complete = valid and not ({src_field, dst_field} & incomplete)
        add(stem + 'complete_direction_jaccard',
            _ratio(intersection, union) if complete else NAN, 'direction_overlap')
        add(stem + 'direction_overlap_complete', complete, 'direction_overlap')

    # Ratio names explicitly identify reported counts; the exporter may censor
    # membership independently, so these are descriptors rather than flow truth.
    for direction in ('all', 'src', 'dst'):
        count = values[f'network_packets_{direction}_count']
        for group in ('ips', 'ports'):
            field = f'network_{group}_{direction}_count'
            ratio(field + '__per_packet', values[field], count, 'scan_shape')
            ratio(field + '__per_second', values[field], duration, 'scan_shape')
        ratio(f'network_ports_{direction}__per_reported_ip',
              values[f'network_ports_{direction}_count'], values[f'network_ips_{direction}_count'], 'scan_shape')
    ratio('network_packet_direction_balance', values['network_packets_src_count']
          - values['network_packets_dst_count'], packets, 'scan_shape')
    for left, right in (('syn', 'ack'), ('rst', 'syn'), ('fin', 'syn')):
        ratio(f'network_tcp_flags_{left}_minus_{right}__per_packet',
              values[f'network_tcp-flags-{left}_count'] - values[f'network_tcp-flags-{right}_count'],
              packets, 'scan_shape')
    for group in ('packet-size', 'header-length', 'ip-length', 'mss', 'payload-length',
                  'time-delta', 'ttl', 'window-size'):
        stem = 'network_' + group
        low, high, avg, std = (values[stem + suffix] for suffix in ('_min', '_max', '_avg', '_std_deviation'))
        width = high - low
        add(stem + '__range', width, 'summary_shape')
        ratio(stem + '__range_over_abs_average', width, abs(avg), 'summary_shape')
        position = ((avg - low) / width if width > 0 else 0) if math.isfinite(width) and math.isfinite(avg) else NAN
        add(stem + '__average_position_in_range', position, 'summary_shape')
        if group != 'packet-size':  # Already in base under its historical name.
            ratio(stem + '__std_over_abs_average', std, abs(avg), 'summary_shape')
    return entries


class CsvEncoder:
    """Serializable fresh vocabulary and deterministic per-row transform."""

    def __init__(self, vocabularies: dict, *, training_rows: int,
                 max_ports_per_direction: int = 256, min_port_support: int = 3):
        if type(max_ports_per_direction) is not int or not 1 <= max_ports_per_direction <= 4096:
            raise ValueError('Port vocabulary limit must be in [1, 4096]')
        if type(min_port_support) is not int or min_port_support < 1:
            raise ValueError('Port support must be a positive integer')
        if type(training_rows) is not int or training_rows < 1:
            raise ValueError('Encoder requires at least one fitting row')
        if set(vocabularies) != set(VOCAB_FIELDS):
            raise ValueError('Unexpected vocabulary schema')
        self.vocabularies = {}
        for field in VOCAB_FIELDS:
            tokens = vocabularies[field]
            if not isinstance(tokens, (list, tuple)):
                raise ValueError('Invalid vocabulary schema')
            limit = max_ports_per_direction if kind(field) == 'ports' else 128
            if len(tokens) > limit:
                raise ValueError('Vocabulary exceeds bounded schema')
            if kind(field) == 'ports':
                valid = all(type(token) is int and 0 <= token <= 65535 for token in tokens)
            else:
                valid = all(isinstance(token, str) and 0 < len(token) <= 200 for token in tokens)
            if not valid or len(tokens) != len(set(tokens)):
                raise ValueError('Invalid vocabulary schema')
            self.vocabularies[field] = tuple(sorted(tokens))
        self.training_rows = training_rows
        self.max_ports_per_direction = max_ports_per_direction
        self.min_port_support = min_port_support
        blank = {name: '' for name in NUMERIC_FIELDS}
        observed = {field: None for field in LISTS}
        entries = _derived(blank, {'incomplete_fields': LISTS, 'duration': NAN}, observed, self.vocabularies)
        self.feature_names = [name for name, _, _ in entries]
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError('Duplicate feature names in schema')
        self.feature_groups = {}
        for index, (_, _, group) in enumerate(entries):
            self.feature_groups.setdefault(group, []).append(index)
        self.feature_groups['flag_bitmasks'] = [i for i, name in enumerate(self.feature_names)
            if name.startswith(('network_ip-flags_', 'network_tcp-flags_'))]
        self.base_indices = [i for i, (_, _, group) in enumerate(entries)
                             if group.startswith('base_') or group == 'membership']
        self.enriched_indices = list(range(len(entries)))

    @classmethod
    def fit(cls, rows: Iterable[Mapping[str, str]], *, max_ports_per_direction=256,
            min_port_support=3):
        if type(max_ports_per_direction) is not int or not 1 <= max_ports_per_direction <= 4096:
            raise ValueError('Port vocabulary limit must be in [1, 4096]')
        if type(min_port_support) is not int or min_port_support < 1:
            raise ValueError('Port support must be a positive integer')
        counts = {field: Counter() for field in VOCAB_FIELDS}
        n = 0
        for row in rows:
            n += 1
            observed = {}
            for field in VOCAB_FIELDS:
                observed[field] = parsed(row.get(field, ''), kind(field))[0]
            # Reuse the canonical direction-union rule without labels or counts.
            observed['network_protocols_all'] = tuple(sorted(set().union(
                *(observed[f'network_protocols_{d}'] for d in ('all', 'src', 'dst')))))
            for field, tokens in observed.items():
                counts[field].update(tokens)
                if kind(field) != 'ports' and len(counts[field]) > 128:
                    raise ValueError('Protocol/log vocabulary exceeds bounded schema')
        vocabulary = {}
        for field, count in counts.items():
            if kind(field) == 'ports':
                candidates = sorted((token for token in count if count[token] >= min_port_support),
                                    key=lambda token: (-count[token], token))
                vocabulary[field] = sorted(candidates[:max_ports_per_direction])
            else:
                vocabulary[field] = sorted(count)
        return cls(vocabulary, training_rows=n, max_ports_per_direction=max_ports_per_direction,
                   min_port_support=min_port_support)

    def encode_row(self, row: Mapping[str, str]):
        clean, facts, observed, events = _clean(row)
        entries = _derived(clean, facts, observed, self.vocabularies)
        values = np.asarray([value for _, value, _ in entries], dtype=np.float64)
        if np.isinf(values).any() or np.any(np.abs(values[np.isfinite(values)]) > np.finfo(np.float32).max):
            raise ValueError('Derived feature exceeds finite float32 range')
        return values.astype(np.float32), facts, events

    def transform(self, row: Mapping[str, str]) -> np.ndarray:
        return self.encode_row(row)[0]

    def to_dict(self):
        return {'version': SCHEMA_VERSION, 'training_rows': self.training_rows,
                'max_ports_per_direction': self.max_ports_per_direction,
                'min_port_support': self.min_port_support,
                'vocabularies': {field: list(tokens) for field, tokens in self.vocabularies.items()},
                'feature_names': list(self.feature_names),
                'base_indices': list(self.base_indices),
                'feature_groups': {group: list(indices) for group, indices in self.feature_groups.items()}}

    @classmethod
    def from_dict(cls, data: dict):
        if not isinstance(data, dict) or data.get('version') != SCHEMA_VERSION:
            raise ValueError('Unsupported encoder schema version')
        encoder = cls(data['vocabularies'], training_rows=data['training_rows'],
                      max_ports_per_direction=data['max_ports_per_direction'], min_port_support=data['min_port_support'])
        if data != encoder.to_dict():
            raise ValueError('Encoder schema does not match fixed feature definitions')
        return encoder
