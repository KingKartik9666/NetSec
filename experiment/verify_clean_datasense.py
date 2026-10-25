"""Create and verify a conservatively cleaned DataSense copy, without fitting.

Run: python -B -m experiment.verify_clean_datasense [--stage all|scan|finish|verify]
Every real-data stage is restricted to the existing train/validation whitelist.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from experiment.audit_datasense_raw import LISTS, META, PARENT, ROOT, SPLITS, allocate, json_writer
from experiment.datasense_integrity import clean_row, timestamp_ns, validate_numeric
from experiment.datasense_quality import digest, parse_list, selected_rows, write_json

BASE = PARENT / 'deep_clean_v2'
PRIOR = BASE / '04_port_refinement'
RAW = BASE / '01_raw_audit'
DEFAULT = ROOT / 'dataset/experiments/datasense_verified_clean_20261014_v2'
POLICY = ROOT / 'docs/DATASET_VERIFICATION_20261014.md'
FACTS = ['hard_invalid', 'no_signal', 'negative_delta', 'list_uncertainty',
         'recovered_members', 'udp_evidence_gap', 'start', 'end', 'duration', 'numeric_missing_cells']


def json_load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checkpoint(out, stage, **values):
    path = out / 'state.json'
    state = json_load(path) if path.exists() else {}
    state.update(stage=stage, updated_at=datetime.now(timezone.utc).isoformat(), **values)
    temporary = out / '.state.tmp'
    write_json(temporary, state)
    temporary.replace(path)


def input_paths():
    paths = [ROOT / 'dataset/combined_dataset.csv', PARENT / 'source/combined_dataset.csv',
             PARENT / 'copy_manifest.json', PARENT / 'prepared/manifest.json',
             RAW / 'scan_summary.json', PRIOR / 'manifest.json', POLICY,
             Path(__file__), Path(__file__).with_name('datasense_integrity.py'),
             Path(__file__).with_name('datasense_quality.py'), Path(__file__).with_name('audit_datasense_raw.py')]
    for split in SPLITS:
        paths += [PARENT / f'prepared/{p}_{split}.npy'
                  for p in ('row_ids', 'y', 'window_start', 'window_end', 'feature_hash')]
        paths += [RAW / f'record_hash_{split}.npy', PRIOR / f'X_full_{split}.npy']
    return paths


def hashes(paths):
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def initialize(out):
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}; resume only a completed stage')
    out.mkdir(parents=True)
    for folder in ('raw', 'cleaned', 'model', 'audit', 'verification', 'source_snapshot'):
        (out / folder).mkdir()
    checkpoint(out, 'hash_inputs', status='running', selected_rows_done=0, selected_rows_total=510405)
    pinned = hashes(input_paths())
    expected = json_load(PARENT / 'copy_manifest.json')['sha256']
    for path in (ROOT / 'dataset/combined_dataset.csv', PARENT / 'source/combined_dataset.csv'):
        if pinned[path.relative_to(ROOT).as_posix()] != expected:
            raise ValueError('Original/source copy checksum mismatch')
    write_json(out / 'input_hashes.json', pinned)
    for path in (POLICY, Path(__file__), Path(__file__).with_name('datasense_integrity.py')):
        shutil.copy2(path, out / 'source_snapshot' / path.name)
    write_json(out / 'scope.json', {
        'source_sha256': expected, 'source_rows': 685671,
        'selected_rows': {'train': 480405, 'validation': 30000}, 'excluded_source_rows': 175266,
        'sealed_test_arrays_opened': False, 'labels_changed': 0, 'training_run': False,
        'source': (PARENT / 'source/combined_dataset.csv').relative_to(ROOT).as_posix(),
        'policy': POLICY.relative_to(ROOT).as_posix(),
        'coverage_limit': 'Only original train/validation IDs; no claim of a full held-out-data audit.'})


def model_schema():
    schema = json_load(PRIOR / 'manifest.json')
    names = schema['feature_names'][:]
    names[names.index('network_macs_dst_count')] = 'network_macs_dst__observed_distinct_count'
    names = [n.replace('__capped_list', '__membership_incomplete') for n in names]
    schema.update(version=3, feature_names=names, model_feature_names=names,
                  model_feature_indexes=list(range(len(names))),
                  numeric_zero_masking='Unknown when zero supporting events; invalid cells also unknown.',
                  censored_list_rule='Any recorded/listed disagreement or directional recovery; absence is unknown.',
                  row_grouping='No compression in master; strict training selection is separately recorded.',
                  redundancy='Full features retained; no validation-derived selection.',
                  calibration='None fitted for this representation.',
                  deployment_status='Research dataset only; prior checkpoints are incompatible with this schema.')
    # Old artifact hashes and preparation claims cannot describe the new bytes.
    for key in ('artifacts', 'training_arms', 'no_signal_rule'):
        schema.pop(key, None)
    return schema


def update_model(vector, names, index, numeric, clean, facts, observed):
    for field in numeric:
        if field == 'network_macs_dst_count':
            tokens = observed.get('network_macs_dst')
            vector[index['network_macs_dst__observed_distinct_count']] = len(tokens) if tokens is not None else np.nan
        else:
            vector[index[field]] = float(clean[field]) if clean[field] else np.nan
    incomplete = set(facts['incomplete_fields'])
    for field in LISTS:
        tokens = observed[field]
        length = len(tokens) if tokens is not None else np.nan
        vector[index[field + '__listed_distinct']] = length
        reported = float(clean[field + '_count']) if clean[field + '_count'] else np.nan
        vector[index[field + '__reported_minus_listed']] = reported - length
        vector[index[field + '__membership_incomplete']] = field in incomplete or tokens is None
    return vector


def recompute_derived(matrix, names, numeric, membership, ids, facts_path, clean_path):
    """Rebuild list presence and dependent rates in bounded chunks/one CSV pass."""
    idx = {name: i for i, name in enumerate(names)}
    presence = {}
    for field in LISTS:
        columns = [(i, n.split('__present_', 1)[1]) for i, n in enumerate(names)
                   if n.startswith(field + '__present_')]
        presence[field] = [(i, int(t) if 'ports_' in field and t != 'UNSEEN' else t) for i, t in columns]
    with gzip.open(clean_path, 'rt', encoding='utf-8', newline='') as stream:
        for i, row in enumerate(csv.DictReader(stream)):
            if int(row['source_row_id']) != ids[i]:
                raise ValueError('Cleaned row alignment changed')
            for j, field in enumerate(LISTS):
                if not presence[field]:
                    continue
                category = 'log' if field == 'log_data-types' else field.split('_')[1]
                tokens = set(parse_list(row[field], category)[0]) if row[field] else None
                known = {token for _, token in presence[field] if token != 'UNSEEN'}
                unknown = bool(membership[i, j]) or tokens is None
                for col, token in presence[field]:
                    present = tokens is not None and (bool(tokens - known) if token == 'UNSEEN' else token in tokens)
                    matrix[i, col] = 1 if present else (np.nan if unknown else 0)
    starts = np.load(facts_path / 'window_start.npy', mmap_mode='r')
    ends = np.load(facts_path / 'window_end.npy', mmap_mode='r')
    for start in range(0, len(matrix), 2048):
        sl = slice(start, start + 2048)
        block = matrix[sl]
        def col(name):
            return block[:, idx[name]]
        def ratio(name, numerator, denominator):
            result = np.full(len(block), np.nan, dtype=np.float32)
            np.divide(numerator, denominator, out=result,
                      where=np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0))
            block[:, idx[name]] = result
        duration = (ends[sl] - starts[sl]) / 1e9
        block[:, idx['window_duration_seconds']] = duration
        packets = col('network_packets_all_count')
        for field in ('network_packets_all_count', 'network_packets_src_count',
                      'network_packets_dst_count', 'log_messages_count'):
            ratio(field + '__per_second', col(field), duration)
        for flag in ('ack', 'fin', 'psh', 'rst', 'syn', 'urg'):
            ratio(f'network_tcp-flags-{flag}__fraction_of_packets', col(f'network_tcp-flags-{flag}_count'), packets)
        for direction in ('src', 'dst'):
            ratio(f'network_{direction}_packet_fraction', col(f'network_packets_{direction}_count'), packets)
        for name, numerator, denominator in (
            ('network_packet_size_coefficient_of_variation', 'network_packet-size_std_deviation', 'network_packet-size_avg'),
            ('network_payload_fraction', 'network_payload-length_avg', 'network_packet-size_avg'),
            ('network_packets_per_reported_ip', 'network_packets_all_count', 'network_ips_all_count'),
            ('network_packets_per_reported_port', 'network_packets_all_count', 'network_ports_all_count')):
            ratio(name, col(numerator), col(denominator))
        block[:, idx['quality_no_packets_observed']] = packets == 0
        block[:, idx['quality_no_logs_observed']] = col('log_messages_count') == 0
        block[:, idx['quality_negative_packet_delta']] = col('network_time-delta_min') < 0
        if np.isinf(block).any():
            raise ValueError('Model input overflow')
    matrix.flush()


def scan(out):
    if not out.exists():
        initialize(out)
    if (out / 'audit/scan.json').exists():
        raise FileExistsError('Scan is complete; use finish or verify')
    scope = json_load(RAW / 'scan_summary.json')
    schema = model_schema()
    numeric, headers, names = scope['numeric_columns'], scope['raw_columns'], schema['feature_names']
    index = {name: i for i, name in enumerate(names)}
    ids = {s: np.load(PARENT / f'prepared/row_ids_{s}.npy') for s in SPLITS}
    y = {s: np.load(PARENT / f'prepared/y_{s}.npy', mmap_mode='r') for s in SPLITS}
    if np.intersect1d(ids['train'], ids['validation']).size:
        raise ValueError('Partition membership overlaps')
    selector = np.zeros(685671, dtype=np.uint8)
    positions = np.zeros(685671, dtype=np.int32)
    arrays, expected, row_counts, issue_counts = {}, {}, Counter(), {s: Counter() for s in SPLITS}
    device_codes = {}
    coverage = {s: defaultdict(Counter) for s in SPLITS}
    for code, split in enumerate(SPLITS, 1):
        n = len(ids[split]); selector[ids[split]] = code; positions[ids[split]] = np.arange(n)
        path = out / 'audit' / split; path.mkdir()
        arrays[split] = {
            'model': allocate(out / 'model', f'X_{split}', 'float32', (n, len(names))),
            'membership': allocate(path, 'list_incomplete', 'bool', (n, len(LISTS))),
            'hard_invalid': allocate(path, 'hard_invalid', 'bool', (n,)),
            'no_signal': allocate(path, 'no_signal', 'bool', (n,)),
            'device': allocate(path, 'device_codes', 'int16', (n,)),
            'start': allocate(path, 'window_start', 'int64', (n,)),
            'end': allocate(path, 'window_end', 'int64', (n,)),
        }
        original = np.load(PRIOR / f'X_full_{split}.npy', mmap_mode='r')
        for a in range(0, n, 2048):
            arrays[split]['model'][a:a+2048] = original[a:a+2048]
        del original
        expected[split] = np.load(RAW / f'record_hash_{split}.npy', mmap_mode='r')
        for prefix in ('row_ids', 'y', 'window_start', 'window_end'):
            shutil.copy2(PARENT / f'prepared/{prefix}_{split}.npy', out / f'model/{prefix}_{split}.npy')
    checkpoint(out, 'scan_source', status='running')
    started = time.monotonic()
    canonical_counts = {s: Counter() for s in SPLITS}
    mac_pattern = {s: Counter() for s in SPLITS}
    with ExitStack() as stack:
        raw_writers, clean_writers, fact_writers = {}, {}, {}
        for split in SPLITS:
            def writer(path, fields):
                return json_writer(stack.enter_context(gzip.open(path, 'wt', encoding='utf-8', newline='', compresslevel=1)), fields)
            raw_writers[split] = writer(out / f'raw/selected_{split}.csv.gz', ['source_row_id'] + headers)
            extra = [f'{f}__observed_distinct' for f in LISTS] + ['quality_numeric_missing_cells', 'quality_list_membership_incomplete']
            clean_writers[split] = writer(out / f'cleaned/cleaned_{split}.csv.gz', ['source_row_id'] + headers + extra)
            fact_writers[split] = writer(out / f'audit/row_facts_{split}.csv.gz', ['source_row_id', 'device_name', 'label4'] + FACTS)
        ledger = writer(out / 'audit/change_and_review_ledger.csv.gz', ['split', 'source_row_id', 'rule', 'severity', 'fields', 'before', 'after'])
        for row_id, row in selected_rows(PARENT / 'source/combined_dataset.csv',
                                        np.sort(np.concatenate(list(ids.values()))), 685671):
            split = SPLITS[int(selector[row_id]) - 1]; i = int(positions[row_id])
            row_hash = hashlib.sha256(json.dumps([row[f] for f in headers], separators=(',', ':')).encode()).digest()
            if row_hash != expected[split][i].tobytes().ljust(32, b'\0'):
                raise ValueError(f'Archived raw cell mismatch: {split} source ID {row_id}')
            if row['label4'] != schema['classes'][int(y[split][i])]:
                raise ValueError('Frozen target changed')
            raw_writers[split].writerow({'source_row_id': row_id, **row})
            clean, facts, observed, events = clean_row(row, numeric, LISTS)
            mac_pattern[split]['rows'] += 1
            mac_pattern[split]['dst_count_equals_src_count'] += float(row['network_macs_dst_count']) == float(row['network_macs_src_count'])
            mac_pattern[split]['dst_count_disagrees_with_list'] += float(row['network_macs_dst_count']) != len(observed['network_macs_dst'] or ())
            for field in LISTS:
                canonical_counts[split][field] += clean[field] != row[field]
            clean_writers[split].writerow({'source_row_id': row_id, **clean,
                **{f'{f}__observed_distinct': len(observed[f]) if observed[f] is not None else '' for f in LISTS},
                'quality_numeric_missing_cells': facts['numeric_missing_cells'],
                'quality_list_membership_incomplete': int(facts['list_uncertainty'])})
            fact_writers[split].writerow({'source_row_id': row_id, 'device_name': row['device_name'],
                                         'label4': row['label4'], **{k: facts[k] for k in FACTS}})
            for event in events:
                issue_counts[split][event['rule']] += 1
                ledger.writerow({'split': split, 'source_row_id': row_id,
                    **{k: json.dumps(v, separators=(',', ':'), allow_nan=False) if k in ('fields', 'before', 'after') else v
                       for k, v in event.items()}})
            a = arrays[split]
            update_model(a['model'][i], names, index, numeric, clean, facts, observed)
            a['membership'][i] = [f in facts['incomplete_fields'] or observed[f] is None for f in LISTS]
            for key in ('hard_invalid', 'no_signal', 'start', 'end'):
                a[key][i] = facts[key]
            identity = (row['device_name'], row['device_mac'])
            if identity not in device_codes:
                device_codes[identity] = len(device_codes)
            a['device'][i] = device_codes[identity]
            key = (row['label4'], row['device_name'], row['timestamp_start'][:10], facts['duration'])
            record = coverage[split][key]
            record['rows'] += 1
            for fact in ('hard_invalid', 'no_signal', 'udp_evidence_gap', 'negative_delta', 'list_uncertainty', 'recovered_members'):
                record[fact] += int(facts[fact])
            record['rows_with_packets'] += float(row['network_packets_all_count']) > 0
            record['rows_with_logs'] += float(row['log_messages_count']) > 0
            for proto in ('tcp', 'udp', 'icmp', 'arp', 'mqtt'):
                record['observed_' + proto] += proto in (observed.get('network_protocols_all') or ())
            row_counts[split] += 1
            if sum(row_counts.values()) % 10000 == 0:
                checkpoint(out, 'scan_source', selected_rows_done=sum(row_counts.values()))
                print(f'Audited {sum(row_counts.values()):,}/510,405 rows in {time.monotonic()-started:.1f}s', flush=True)
    for split in SPLITS:
        if row_counts[split] != len(ids[split]):
            raise ValueError('Incomplete source scan')
        for key in ('start', 'end'):
            np.testing.assert_array_equal(arrays[split][key],
                np.load(PARENT / f'prepared/window_{key}_{split}.npy', mmap_mode='r'))
        for array in arrays[split].values():
            array.flush()
        checkpoint(out, 'recompute_derived_' + split, selected_rows_done=510405)
        recompute_derived(arrays[split]['model'], names, numeric, arrays[split]['membership'], ids[split],
                          out / 'audit' / split, out / f'cleaned/cleaned_{split}.csv.gz')
    coverage_rows = []
    for split in SPLITS:
        for (label, device, date, duration), counts in coverage[split].items():
            coverage_rows.append({'split': split, 'label4': label, 'device': device, 'date': date,
                                  'duration_seconds': duration, **counts})
    with (out / 'audit/coverage_class_device_date_duration.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = json_writer(stream, list(coverage_rows[0]))
        writer.writerows(coverage_rows)
    write_json(out / 'audit/devices.json', [{'code': c, 'device_name': k[0], 'device_mac': k[1]} for k, c in device_codes.items()])
    write_json(out / 'model/manifest.json', schema)
    write_json(out / 'audit/scan.json', {'rows': dict(row_counts), 'issues': {s: dict(c) for s, c in issue_counts.items()},
        'canonicalized_list_cells': {s: dict(c) for s, c in canonical_counts.items()},
        'destination_mac_count_pattern': {s: dict(c) for s, c in mac_pattern.items()},
        'all_selected_raw_cells_match_prior_archive': True, 'elapsed_seconds': time.monotonic()-started})
    checkpoint(out, 'scan_complete', status='running')


def input_groups(matrix, labels):
    """Exact numeric-vector conflict groups; canonical NaN/zero and collision check."""
    labels = np.asarray(labels)
    row_hash = np.empty(len(matrix), dtype='V32')
    for start in range(0, len(matrix), 2048):
        block = np.array(matrix[start:start+2048], dtype=np.float32, order='C', copy=True)
        block[block == 0] = 0; block[np.isnan(block)] = np.nan
        for j, row in enumerate(block):
            row_hash[start+j] = hashlib.sha256(row.tobytes()).digest()
    _, first, inverse, counts = np.unique(row_hash, return_index=True, return_inverse=True, return_counts=True)
    low = np.full(len(first), np.iinfo(np.int64).max, dtype=np.int64)
    high = np.full(len(first), -1, dtype=np.int64)
    np.minimum.at(low, inverse, labels); np.maximum.at(high, inverse, labels)
    for start in range(0, len(matrix), 2048):
        np.testing.assert_array_equal(matrix[first[inverse[start:start+2048]]], matrix[start:start+2048])
    return row_hash, first, inverse, counts, low != high


def interval_statistics(starts, ends):
    """Coverage of selected windows, not a claim of missing capture packets."""
    if not len(starts):
        return {'windows': 0, 'union_seconds': 0, 'span_seconds': 0, 'gaps': 0, 'overlapping_windows': 0}
    order = np.argsort(starts, kind='stable')
    beginning, finish = int(starts[order[0]]), int(ends[order[0]])
    first = beginning; total = gaps = overlaps = 0
    for i in order[1:]:
        a, b = int(starts[i]), int(ends[i])
        if a > finish:
            total += finish - beginning; gaps += 1; beginning, finish = a, b
        else:
            overlaps += a < finish; finish = max(finish, b)
    total += finish - beginning
    return {'windows': len(starts), 'union_seconds': total / 1e9, 'span_seconds': (finish-first)/1e9,
            'gaps': gaps, 'overlapping_windows': overlaps}


def finish(out):
    if not (out / 'audit/scan.json').exists():
        raise ValueError('Complete scan first')
    if (out / 'summary.json').exists():
        raise FileExistsError('Finish is complete; use verify')
    checkpoint(out, 'duplicates_and_coverage', status='running')
    schema = json_load(out / 'model/manifest.json')
    duplicates, temporal, class_rows, conflict_rows, group_hashes = {}, [], [], [], {}
    masks = {}
    for split in SPLITS:
        X = np.load(out / f'model/X_{split}.npy', mmap_mode='r')
        y = np.load(out / f'model/y_{split}.npy', mmap_mode='r')
        ids = np.load(out / f'model/row_ids_{split}.npy', mmap_mode='r')
        row_hash, first, inverse, counts, conflicts = input_groups(X, y)
        group_hashes[split] = row_hash
        conflict_mask = conflicts[inverse]
        np.save(out / f'audit/{split}/conflicting_inputs.npy', conflict_mask)
        np.save(out / f'audit/{split}/model_input_sha256.npy', row_hash)
        hard = np.load(out / f'audit/{split}/hard_invalid.npy')
        empty = np.load(out / f'audit/{split}/no_signal.npy')
        raw_hash = np.load(RAW / f'record_hash_{split}.npy', mmap_mode='r')
        _, raw_first = np.unique(raw_hash, return_index=True)
        raw_duplicate = np.ones(len(y), dtype=bool); raw_duplicate[raw_first] = False
        masks[split] = ~hard & ~empty & ~conflict_mask & ~raw_duplicate
        np.save(out / f'audit/{split}/quality_eligible_mask.npy', masks[split])
        duplicates[split] = {'rows': len(y), 'distinct_model_inputs': len(first),
            'duplicate_model_input_excess': int((counts-1).sum()), 'conflicting_model_input_groups': int(conflicts.sum()),
            'rows_in_conflicting_groups': int(conflict_mask.sum()), 'duplicate_raw_excess': int(raw_duplicate.sum()),
            'hard_invalid_rows': int(hard.sum()), 'no_signal_rows': int(empty.sum()),
            'quality_eligible_rows': int(masks[split].sum()), 'validation_filter_applied': False}
        for group in np.flatnonzero(conflicts):
            members = np.flatnonzero(inverse == group)
            labels, amounts = np.unique(y[members], return_counts=True)
            conflict_rows.append({'split': split, 'group': int(group), 'rows': len(members),
                'source_ids': ids[members].tolist(), 'labels': {schema['classes'][int(k)]: int(n) for k, n in zip(labels, amounts)}})
        device = np.load(out / f'audit/{split}/device_codes.npy', mmap_mode='r')
        starts = np.load(out / f'audit/{split}/window_start.npy', mmap_mode='r')
        ends = np.load(out / f'audit/{split}/window_end.npy', mmap_mode='r')
        for c, label in enumerate(schema['classes']):
            rows = y == c
            class_rows.append({'split': split, 'label4': label, 'rows': int(rows.sum()),
                'quality_eligible_rows': int((rows & masks[split]).sum()),
                'no_signal_rows': int((rows & empty).sum()), 'conflicting_rows': int((rows & conflict_mask).sum()),
                'hard_invalid_rows': int((rows & hard).sum()), 'devices': int(len(np.unique(device[rows])))})
            for d in np.unique(device[rows]):
                selected = rows & (device == d) & ~hard
                temporal.append({'split': split, 'label4': label, 'device_code': int(d),
                                 **interval_statistics(starts[selected], ends[selected])})
        if split == 'train':
            selected = np.flatnonzero(masks[split])
            strict = out / 'model/strict_train'; strict.mkdir()
            target = allocate(strict, 'X_train', 'float32', (len(selected), X.shape[1]))
            for a in range(0, len(selected), 2048):
                target[a:a+2048] = X[selected[a:a+2048]]
            target.flush()
            np.save(strict / 'y_train.npy', y[selected]); np.save(strict / 'row_ids_train.npy', ids[selected])
            np.save(strict / 'master_indexes.npy', selected)
            with gzip.open(out / 'cleaned/cleaned_train.csv.gz', 'rt', encoding='utf-8', newline='') as source, \
                 gzip.open(out / 'cleaned/strict_train.csv.gz', 'wt', encoding='utf-8', newline='', compresslevel=1) as dest, \
                 gzip.open(out / 'audit/training_review_queue.csv.gz', 'wt', encoding='utf-8', newline='', compresslevel=1) as review:
                reader = csv.DictReader(source); writer = json_writer(dest, reader.fieldnames)
                queue = json_writer(review, ['source_row_id', 'label4', 'reasons'])
                for i, row in enumerate(reader):
                    if masks[split][i]:
                        writer.writerow(row)
                    else:
                        reasons = [name for name, mask in [('hard_invalid', hard), ('no_observed_signal', empty),
                                  ('conflicting_identical_inputs', conflict_mask), ('exact_raw_duplicate', raw_duplicate)] if mask[i]]
                        queue.writerow({'source_row_id': row['source_row_id'], 'label4': row['label4'], 'reasons': '|'.join(reasons)})
        print(f'{split}: {duplicates[split]}', flush=True)
    for filename, rows in [('class_coverage.csv', class_rows), ('selected_window_coverage.csv', temporal)]:
        with (out / 'audit' / filename).open('w', encoding='utf-8', newline='') as stream:
            json_writer(stream, list(rows[0])).writerows(rows)
    write_json(out / 'audit/input_conflicts.json', conflict_rows)
    cross = int(np.intersect1d(group_hashes['train'], group_hashes['validation']).size)
    # Existing purge is class-conditional; independently validate its intervals.
    overlap_classes = []
    train_y, val_y = [np.load(out / f'model/y_{s}.npy') for s in SPLITS]
    train_end = np.load(out / 'model/window_end_train.npy')
    val_start = np.load(out / 'model/window_start_validation.npy')
    for c, label in enumerate(schema['classes']):
        if np.any(train_y == c) and np.any(val_y == c) and train_end[train_y == c].max() >= val_start[val_y == c].min():
            overlap_classes.append(label)
    summary = {'rows': {s: duplicates[s]['rows'] for s in SPLITS}, 'features': len(schema['feature_names']),
        'duplicates_and_conflicts': duplicates, 'train_validation_shared_model_vectors': cross,
        'class_conditional_temporal_overlap': overlap_classes,
        'strict_train_rows': int(masks['train'].sum()),
        'strict_train_classes': int(len(np.unique(train_y[masks['train']]))),
        'validation_rows_retained': len(val_y), 'labels_changed': 0, 'test_arrays_loaded': False,
        'missing_capture_data_reconstructed': False, 'model_trained': False,
        'annotation_truth_verified': False, 'source_capture_available': False}
    write_json(out / 'summary.json', summary)
    checkpoint(out, 'cleaning_complete', status='running')


def verify(out):
    checkpoint(out, 'independent_verification', status='running')
    schema = json_load(out / 'model/manifest.json')
    scope = json_load(RAW / 'scan_summary.json')
    numeric = scope['numeric_columns']; names = schema['feature_names']; idx = {n: i for i, n in enumerate(names)}
    results = {}
    for split in SPLITS:
        X = np.load(out / f'model/X_{split}.npy', mmap_mode='r')
        ids = np.load(out / f'model/row_ids_{split}.npy', mmap_mode='r')
        y = np.load(out / f'model/y_{split}.npy', mmap_mode='r')
        missing = 0; rows = 0
        for start in range(0, len(X), 2048):
            if np.isinf(X[start:start+2048]).any():
                raise ValueError('Infinite output')
            missing += int(np.isnan(X[start:start+2048]).sum())
        with gzip.open(out / f'cleaned/cleaned_{split}.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
            for i, row in enumerate(csv.DictReader(stream)):
                if int(row['source_row_id']) != ids[i] or row['label4'] != schema['classes'][int(y[i])]:
                    raise ValueError('Output ID/label mismatch')
                vals = {n: float(row[n]) for n in numeric if row[n]}
                if validate_numeric(vals):
                    raise ValueError('Unresolved invalid numeric output')
                if row['network_macs_dst_count']:
                    raise ValueError('Suspect source count was retained')
                for field in LISTS:
                    if row[field]:
                        kind = 'log' if field == 'log_data-types' else field.split('_')[1]
                        tokens, length = parse_list(row[field], kind)
                        if len(tokens) != length or len(tokens) != int(row[field + '__observed_distinct']):
                            raise ValueError('List normalization/count mismatch')
                        if field == 'network_macs_dst' and X[i, idx['network_macs_dst__observed_distinct_count']] != len(tokens):
                            raise ValueError('Observed MAC count did not reconstruct')
                for field in numeric:
                    if field == 'network_macs_dst_count':
                        continue
                    expected = np.float32(float(row[field])) if row[field] else np.nan
                    actual = X[i, idx[field]]
                    if not (actual == expected or np.isnan(actual) and np.isnan(expected)):
                        raise ValueError('Numeric CSV/model mismatch')
                rows += 1
        if rows != len(ids):
            raise ValueError('Incomplete CSV')
        for prefix in ('row_ids', 'y', 'window_start', 'window_end'):
            if digest(out / f'model/{prefix}_{split}.npy') != digest(PARENT / f'prepared/{prefix}_{split}.npy'):
                raise ValueError('Partition provenance changed')
        results[split] = {'rows_reloaded': rows, 'numeric_cells_checked': rows * len(numeric),
                          'explicit_missing_model_cells': missing, 'partition_labels_ids_unchanged': True}
        print(f'Independent CSV verification: {split} {rows:,} rows', flush=True)
    master = np.load(out / 'model/X_train.npy', mmap_mode='r')
    strict = np.load(out / 'model/strict_train/X_train.npy', mmap_mode='r')
    selected = np.load(out / 'model/strict_train/master_indexes.npy')
    for a in range(0, len(selected), 2048):
        np.testing.assert_array_equal(strict[a:a+2048], master[selected[a:a+2048]])
    pinned, current = json_load(out / 'input_hashes.json'), hashes(input_paths())
    if current != pinned:
        raise ValueError('Source or pinned input changed during cleaning')
    write_json(out / 'verification/integrity.json', {'passed': True, 'partitions': results,
        'strict_training_reconstruction': True, 'source_and_input_hashes_unchanged': True,
        'labels_changed': 0, 'test_arrays_opened': False})
    artifacts = {p.relative_to(out).as_posix(): digest(p) for p in out.rglob('*')
                 if p.is_file() and p.name not in ('state.json', 'output_hashes.json') and p.suffix != '.log'}
    write_json(out / 'verification/output_hashes.json', artifacts)
    checkpoint(out, 'verified', status='completed', completed_selected_rows=510405)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    parser.add_argument('--stage', choices=('all', 'scan', 'finish', 'verify'), default='all')
    args = parser.parse_args()
    if args.stage in ('all', 'scan') and args.directory.exists():
        raise FileExistsError('Scan requires a fresh directory; partial attempts are preserved')
    try:
        if args.stage in ('all', 'scan'):
            scan(args.directory)
        if args.stage in ('all', 'finish'):
            finish(args.directory)
        if args.stage in ('all', 'verify'):
            verify(args.directory)
    except BaseException as error:
        if args.directory.exists():
            checkpoint(args.directory, 'failed', status='failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
