"""Audit all whitelisted raw DataSense cells, preserving originals on disk.

Stages are immutable and restricted to the existing train/validation IDs.
The source CSV is traversed to locate those IDs; no other record is profiled.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from datetime import datetime
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiment.datasense_quality import digest, no_signal_mask, parse_list, selected_rows, write_json, zero_bitmap

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'dataset/experiments/xgboost_data_audit_20261012_v1'
DEFAULT = PARENT / 'deep_clean_v2'
SPLITS = ('train', 'validation')
META = ['device_name', 'device_mac', 'label_full', 'label1', 'label2', 'label3', 'label4',
        'timestamp', 'timestamp_start', 'timestamp_end']
LISTS = ['log_data-types'] + [f'network_{group}_{direction}'
        for group in ('ips', 'macs', 'ports', 'protocols') for direction in ('all', 'dst', 'src')]


def kind(field):
    return 'log' if field == 'log_data-types' else field.split('_')[1]


@lru_cache(maxsize=1024)
def small_parse(text, category):
    return parse_list(text, category)


def parsed(text, category):
    return small_parse(text, category) if len(text) < 1024 else parse_list(text, category)


def json_writer(stream, fields):
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    return writer


def allocate(directory, name, dtype, shape):
    return np.lib.format.open_memmap(directory / f'{name}.npy', mode='w+', dtype=dtype, shape=shape)


def scan(directory, parent=PARENT):
    stage = directory / '01_raw_audit'
    if stage.exists():
        raise FileExistsError(f'Refusing to overwrite {stage}')
    source = parent / 'source/combined_dataset.csv'
    copy = json.loads((parent / 'copy_manifest.json').read_text())
    print('Verifying copied source SHA256...', flush=True)
    if source.stat().st_size != copy['bytes'] or digest(source) != copy['sha256']:
        raise ValueError('Original copied source failed integrity')
    manifest = json.loads((parent / 'prepared/manifest.json').read_text())
    with source.open(encoding='utf-8-sig', newline='') as stream:
        headers = next(csv.reader(stream))
    numeric = [name for name in headers if name not in META + LISTS]
    if numeric != manifest['feature_names'][:len(numeric)]:
        raise ValueError('Raw numeric schema differs from frozen feature order')
    ids = {s: np.load(parent / f'prepared/row_ids_{s}.npy') for s in SPLITS}
    ys = {s: np.load(parent / f'prepared/y_{s}.npy', mmap_mode='r') for s in SPLITS}
    if np.intersect1d(ids['train'], ids['validation']).size:
        raise ValueError('Train/validation overlap')
    selector = np.zeros(manifest['source_rows'], dtype=np.uint8)
    positions = np.zeros(manifest['source_rows'], dtype=np.int32)
    for code, split in enumerate(SPLITS, 1):
        selector[ids[split]] = code
        positions[ids[split]] = np.arange(len(ids[split]))
    stage.mkdir(parents=True)
    state = {'source_sha256': copy['sha256'], 'raw_columns': headers, 'numeric_columns': numeric,
             'list_columns': LISTS, 'rows': {s: len(ids[s]) for s in SPLITS},
             'scope': 'Only original train/validation whitelist. Other source records traversed but not analyzed.',
             'test_arrays_loaded': False, 'rules_fitted_on': 'train', 'issues': {}, 'input_hashes': {}}
    for split in SPLITS:
        for prefix in ('row_ids', 'y', 'window_start', 'window_end'):
            path = parent / f'prepared/{prefix}_{split}.npy'
            state['input_hashes'][f'{prefix}_{split}'] = digest(path)
    write_json(stage / 'scope.json', state)
    arrays = {}
    for split in SPLITS:
        n = len(ids[split])
        arrays[split] = {
            'numeric': allocate(stage, f'numeric_{split}', 'float64', (n, len(numeric))),
            'lengths': allocate(stage, f'list_lengths_{split}', 'int32', (n, len(LISTS))),
            'list_hash': allocate(stage, f'list_hash_{split}', 'S16', (n, len(LISTS))),
            'record_hash': allocate(stage, f'record_hash_{split}', 'S32', (n,)),
        }
    counts = {s: Counter() for s in SPLITS}
    categorical = {s: {f: Counter() for f in META[:7]} for s in SPLITS}
    vocab = {f: Counter() for f in LISTS if kind(f) in ('ports', 'protocols', 'log')}
    seen = Counter()
    started = time.monotonic()
    with ExitStack() as stack:
        writers = {}
        for split in SPLITS:
            stream = stack.enter_context(gzip.open(stage / f'raw_selected_{split}.csv.gz', 'wt', encoding='utf-8', newline='', compresslevel=1))
            writers[split] = json_writer(stream, ['source_row_id'] + headers)
        issue_stream = stack.enter_context(gzip.open(stage / 'cell_issues.csv.gz', 'wt', encoding='utf-8', newline='', compresslevel=1))
        issues = json_writer(issue_stream, ['split', 'source_row_id', 'column', 'issue', 'raw_value'])
        def issue(split, row_id, field, code, value):
            counts[split][f'{field}:{code}'] += 1
            issues.writerow(dict(split=split, source_row_id=row_id, column=field, issue=code, raw_value=value))
        for row_id, row in selected_rows(source, np.sort(np.concatenate(list(ids.values()))), manifest['source_rows']):
            split = SPLITS[int(selector[row_id]) - 1]
            index = int(positions[row_id])
            expected = manifest['classes'][int(ys[split][index])]
            if row['label4'] != expected or (row['label1'] == 'benign') != (expected == 'benign'):
                raise ValueError(f'Frozen label mismatch at {row_id}')
            if row['label4'] != ('benign' if row['label1'] == 'benign' else row['label2'] + '_' + row['label3']):
                issue(split, row_id, 'label4', 'hierarchy_disagreement', row['label4'])
            if row['timestamp'] != row['timestamp_start'] + '_' + row['timestamp_end']:
                issue(split, row_id, 'timestamp', 'composite_disagreement', row['timestamp'])
            writers[split].writerow({'source_row_id': row_id, **row})
            arrays[split]['record_hash'][index] = hashlib.sha256(json.dumps(list(row.values()), separators=(',', ':')).encode()).digest()
            for field in META[:7]:
                categorical[split][field][row[field]] += 1
            values = []
            for field in numeric:
                text = row[field]
                try:
                    value = float(text)
                except ValueError:
                    value = float('nan')
                    issue(split, row_id, field, 'unparseable_numeric', text)
                if not np.isfinite(value):
                    issue(split, row_id, field, 'nonfinite_numeric', text)
                values.append(value)
            arrays[split]['numeric'][index] = values
            sets = {}
            for j, field in enumerate(LISTS):
                try:
                    tokens, raw_length = parsed(row[field], kind(field))
                except ValueError:
                    arrays[split]['lengths'][index, j] = -1
                    arrays[split]['list_hash'][index, j] = b''
                    issue(split, row_id, field, 'invalid_list', row[field])
                    continue
                sets[field] = tokens
                arrays[split]['lengths'][index, j] = len(tokens)
                canonical = json.dumps(tokens, separators=(',', ':'))
                arrays[split]['list_hash'][index, j] = hashlib.blake2b(canonical.encode(), digest_size=16).digest()
                if raw_length != len(tokens):
                    issue(split, row_id, field, 'repeated_list_elements', str(raw_length - len(tokens)))
                count_field = field + '_count'
                if count_field in row and float(row[count_field]) != len(tokens):
                    issue(split, row_id, count_field, 'list_count_disagreement', row[count_field])
                if split == 'train' and field in vocab:
                    vocab[field].update(tokens)
            for group in ('ips', 'macs', 'ports', 'protocols'):
                a, b, c = (f'network_{group}_{x}' for x in ('all', 'src', 'dst'))
                if all(f in sets for f in (a, b, c)) and set(sets[a]) != set(sets[b]).union(sets[c]):
                    issue(split, row_id, a, 'all_not_src_dst_union', 'See raw selected record')
            seen[split] += 1
            if sum(seen.values()) % 25000 == 0:
                print(f'Raw audit: {sum(seen.values()):,}/510,405 selected rows; {time.monotonic()-started:.1f}s', flush=True)
        for split in SPLITS:
            if seen[split] != len(ids[split]):
                raise ValueError('Incomplete selected partition')
            for array in arrays[split].values():
                array.flush()
    state['issues'] = {s: dict(c) for s, c in counts.items()}
    state['elapsed_seconds'] = time.monotonic() - started
    write_json(stage / 'scan_summary.json', state)
    write_json(stage / 'categorical_counts.json', categorical)
    write_json(stage / 'training_list_vocabulary_counts.json', {f: {str(t): n for t, n in sorted(v.items(), key=lambda x: (-x[1], str(x[0])))} for f, v in vocab.items()})
    print('Raw scan complete; no records or measurements changed.', flush=True)


def profile(directory, parent=PARENT):
    stage = directory / '01_raw_audit'
    out = directory / '02_profile'
    if out.exists():
        raise FileExistsError(f'Refusing overwrite: {out}')
    scope = json.loads((stage / 'scan_summary.json').read_text())
    numeric = scope['numeric_columns']
    out.mkdir()
    results = {}
    with (out / 'column_profile.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = ['split', 'column', 'type', 'rows', 'zero_or_empty', 'missing_or_invalid', 'negative', 'distinct',
                  'minimum', 'q01', 'q25', 'median', 'q75', 'q99', 'maximum', 'most_frequent', 'mode_rows', 'mode_fraction']
        writer = json_writer(stream, fields)
        for split in SPLITS:
            x = np.load(stage / f'numeric_{split}.npy', mmap_mode='r')
            lengths = np.load(stage / f'list_lengths_{split}.npy', mmap_mode='r')
            zeros = allocate(out, f'zero_cells_{split}', 'uint8', (len(x), (len(numeric)+7)//8))
            for start in range(0, len(x), 4096):
                zeros[start:start+4096] = zero_bitmap(x[start:start+4096])
            zeros.flush()
            signals = no_signal_mask(x, lengths)
            np.save(out / f'no_observed_signal_{split}.npy', signals)
            ids = np.load(parent / f'prepared/row_ids_{split}.npy', mmap_mode='r')
            y = np.load(parent / f'prepared/y_{split}.npy', mmap_mode='r')
            results[split] = {'rows': len(x), 'all_numeric_zero_and_lists_empty': int(signals.sum()), 'numeric': {}}
            for i, field in enumerate(numeric):
                col = x[:, i]
                finite = col[np.isfinite(col)]
                unique, n = np.unique(finite, return_counts=True)
                mode = int(np.argmax(n)) if len(n) else 0
                quantiles = np.quantile(finite, [0,.01,.25,.5,.75,.99,1]).tolist() if len(finite) else [None]*7
                report = dict(split=split, column=field, type='numeric', rows=len(x), zero_or_empty=int((col == 0).sum()),
                    missing_or_invalid=int((~np.isfinite(col)).sum()), negative=int((col < 0).sum()), distinct=len(unique),
                    **dict(zip(['minimum','q01','q25','median','q75','q99','maximum'],quantiles)),
                    most_frequent=float(unique[mode]) if len(unique) else None, mode_rows=int(n[mode]) if len(n) else 0,
                    mode_fraction=float(n[mode]/len(x)) if len(n) else 0)
                writer.writerow(report)
                results[split]['numeric'][field] = report
            hashes = np.load(stage / f'list_hash_{split}.npy', mmap_mode='r')
            for j, field in enumerate(LISTS):
                unique, n = np.unique(hashes[:, j], return_counts=True)
                writer.writerow(dict(split=split, column=field, type='list', rows=len(x), zero_or_empty=int((lengths[:,j] == 0).sum()),
                    missing_or_invalid=int((lengths[:,j] < 0).sum()), distinct=len(unique), mode_rows=int(n.max()), mode_fraction=float(n.max()/len(x))))
            categorical = json.loads((stage / 'categorical_counts.json').read_text())[split]
            for field, counts in categorical.items():
                top = max(counts, key=counts.get)
                writer.writerow(dict(split=split,column=field,type='metadata',rows=len(x),zero_or_empty=counts.get('',0),
                    distinct=len(counts),most_frequent=top,mode_rows=counts[top],mode_fraction=counts[top]/len(x)))
            for field in META[7:]:
                writer.writerow(dict(split=split,column=field,type='provenance_timestamp',rows=len(x)))
            record_hash = np.load(stage / f'record_hash_{split}.npy', mmap_mode='r')
            _, inverse, counts = np.unique(record_hash, return_inverse=True, return_counts=True)
            results[split]['exact_raw_duplicate_excess_sha256'] = int((counts - 1).sum())
            np.save(out / f'exact_record_group_{split}.npy', inverse)
            np.save(out / f'exact_record_multiplicity_{split}.npy', counts)
            idx = {name:i for i,name in enumerate(numeric)}
            checks = {}
            def count_mask(name, mask):
                checks[name] = {'rows':int(mask.sum()), 'example_source_row_ids':ids[np.flatnonzero(mask)[:10]].tolist()}
                np.save(out / f'{name}_{split}.npy', mask)
            p = x[:,idx['network_packets_all_count']]
            src, dst = (x[:,idx[f'network_packets_{d}_count']] for d in ('src','dst'))
            count_mask('packet_direction_sum_disagreement',p != src+dst)
            count_mask('packet_direction_exceeds_all',(src>p)|(dst>p))
            for name in numeric:
                if name.endswith('_count') or name == 'network_fragmented-packets':
                    c = x[:,idx[name]]
                    if np.any((c<0)|(c!=np.floor(c))):
                        count_mask(name+'_invalid_count',(c<0)|(c!=np.floor(c)))
            groups = sorted({f[:-4] for f in numeric if f.endswith('_avg')})
            for group in groups:
                low,avg,high,sd=(x[:,idx[group+s]] for s in ('_min','_avg','_max','_std_deviation'))
                tol = np.maximum(1.,np.maximum(np.abs(low),np.abs(high))) * 1e-10
                count_mask(group+'_invalid_summary',(low>avg+tol)|(avg>high+tol)|(sd<0))
            count_mask('negative_packet_delta', x[:,idx['network_time-delta_min']]<0)
            results[split]['checks'] = checks
            results[split]['no_signal_classes'] = {str(c):int(np.sum(signals & (y==c))) for c in np.unique(y)}
            print(f'Profile complete: {split}, {len(x):,} rows, {int(signals.sum()):,} empty measurements', flush=True)
    write_json(out / 'profile_summary.json', results)
    write_json(out / 'zero_bitmap_schema.json', {'columns':numeric,'bitorder':'little','row_order':'frozen row_ids_<split>.npy',
        'meaning':'One bit per raw numeric cell equal to zero. NaN is not zero. All raw values remain in 01_raw_audit.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('scan','profile'))
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    args = parser.parse_args()
    {'scan':scan,'profile':profile}[args.stage](args.directory)


if __name__ == '__main__':
    main()
