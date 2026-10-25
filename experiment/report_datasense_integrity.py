"""Coverage, provenance and transparent results for the verified dataset copy."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from experiment.audit_datasense_raw import SPLITS, json_writer
from experiment.datasense_quality import digest, write_json
from experiment.verify_clean_datasense import DEFAULT, RAW, json_load


def provenance(out):
    """Check coincident device/windows independently of feature-vector equality."""
    result, keys = {}, {}
    for split in SPLITS:
        folder = out / 'audit' / split
        y = np.load(out / f'model/y_{split}.npy')
        device, start, end = [np.load(folder / (name + '.npy'))
                              for name in ('device_codes', 'window_start', 'window_end')]
        k = np.rec.fromarrays([device, start, end], names='device,start,end')
        unique, inverse, counts = np.unique(k, return_inverse=True, return_counts=True)
        low = np.full(len(unique), 100000, dtype=np.int64); high = np.full(len(unique), -1, dtype=np.int64)
        np.minimum.at(low, inverse, y); np.maximum.at(high, inverse, y)
        result[split] = {'coincident_device_window_groups': int((counts > 1).sum()),
                         'excess_rows_at_same_device_window': int((counts-1).sum()),
                         'device_window_groups_with_different_labels': int((low != high).sum()),
                         'rows_in_different_label_device_windows': int((low[inverse] != high[inverse]).sum())}
        keys[split] = k
    result['shared_train_validation_device_windows'] = int(np.intersect1d(*keys.values()).size)
    write_json(out / 'audit/provenance_window_checks.json', result)
    return result


def shift(out):
    """Describe numeric shift with frozen training quantiles; never delete tails."""
    schema = json_load(out / 'model/manifest.json')
    numeric = json_load(RAW / 'scope.json')['numeric_columns']
    names = schema['feature_names']; idx = {n: i for i, n in enumerate(names)}
    matrices = {s: np.load(out / f'model/X_{s}.npy', mmap_mode='r') for s in SPLITS}
    rows = []
    for original in numeric:
        name = 'network_macs_dst__observed_distinct_count' if original == 'network_macs_dst_count' else original
        columns = {s: np.array(x[:,idx[name]], dtype=np.float64) for s, x in matrices.items()}
        finite = {s: x[np.isfinite(x)] for s, x in columns.items()}
        row = {'feature': name}
        for split in SPLITS:
            x = finite[split]
            row.update({split + '_rows': len(columns[split]), split + '_missing': int(np.isnan(columns[split]).sum()),
                        split + '_zero': int((x == 0).sum()), split + '_negative': int((x < 0).sum()),
                        split + '_median': float(np.median(x)) if len(x) else '',
                        split + '_min': float(x.min()) if len(x) else '',
                        split + '_max': float(x.max()) if len(x) else ''})
        if len(finite['train']) and len(finite['validation']):
            cuts = np.unique(np.quantile(finite['train'], np.linspace(0, 1, 11)))[1:-1]
            edges = np.concatenate([[-np.inf], cuts, [np.inf]])
            a, b = [np.histogram(finite[s], bins=edges)[0].astype(float) for s in SPLITS]
            a = (a+.5)/(a.sum()+.5*len(a)); b = (b+.5)/(b.sum()+.5*len(b))
            row['population_stability_index'] = float(np.sum((b-a)*np.log(b/a)))
            row['validation_outside_train_range'] = int(((finite['validation'] < finite['train'].min()) |
                                                        (finite['validation'] > finite['train'].max())).sum())
        else:
            row['population_stability_index'] = ''; row['validation_outside_train_range'] = ''
        rows.append(row)
    with (out / 'audit/numeric_coverage_and_shift.csv').open('w', encoding='utf-8', newline='') as stream:
        json_writer(stream, list(rows[0])).writerows(rows)
    return rows


def coverage(out):
    totals, combinations = defaultdict(Counter), defaultdict(Counter)
    with (out / 'audit/coverage_class_device_date_duration.csv').open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['split'], row['label4'])
            for field, value in row.items():
                if field not in ('split', 'label4', 'device', 'date', 'duration_seconds'):
                    totals[key][field] += int(value)
            combinations[(row['split'], row['label4'], row['device'])]['rows'] += int(row['rows'])
    class_rows = [{'split': s, 'label4': c, **v} for (s,c),v in totals.items()]
    with (out / 'audit/protocol_evidence_by_class.csv').open('w', encoding='utf-8', newline='') as stream:
        json_writer(stream, list(class_rows[0])).writerows(class_rows)
    classes = json_load(out / 'model/manifest.json')['classes']
    devices = sorted({d for s,c,d in combinations})
    all_pairs = []
    for c in classes:
        for d in devices:
            all_pairs.append({'label4': c, 'device': d,
                              **{s + '_rows': combinations[(s,c,d)]['rows'] for s in SPLITS}})
    with (out / 'audit/class_device_support.csv').open('w', encoding='utf-8', newline='') as stream:
        json_writer(stream, list(all_pairs[0])).writerows(all_pairs)
    return {'classes': len(classes), 'devices': len(devices),
            'possible_class_device_pairs': len(all_pairs),
            'class_device_pairs_without_train_rows': sum(r['train_rows'] == 0 for r in all_pairs),
            'class_device_pairs_without_validation_rows': sum(r['validation_rows'] == 0 for r in all_pairs),
            'validation_rows_from_class_device_pairs_absent_in_train': sum(r['validation_rows'] for r in all_pairs if r['train_rows'] == 0),
            'note': 'A missing class/device combination may be outside the source experiment design; no rows are invented.'}


def report(out):
    summary = json_load(out / 'summary.json'); scan = json_load(out / 'audit/scan.json')
    integrity = json_load(out / 'verification/integrity.json')
    if not integrity['passed']:
        raise ValueError('Integrity verification must pass before reporting')
    proof = provenance(out); shift_rows = shift(out); support = coverage(out)
    write_json(out / 'audit/coverage_summary.json', support)
    total = sum(summary['rows'].values())
    issues = {s: scan['issues'][s] for s in SPLITS}
    count = lambda rule: sum(issues[s].get(rule, 0) for s in SPLITS)
    mac_disagreements = sum(v['dst_count_disagrees_with_list'] for v in scan['destination_mac_count_pattern'].values())
    recovered = count('recover_observed_directional_members')
    scan_checks = {
        'source_rows_traversed': 685671, 'selected_rows_audited': total, 'raw_columns_per_row': 94,
        'selected_raw_cells_verified': total*94, 'class_count': support['classes'], 'device_count': support['devices'],
        'destination_mac_disagreement_rows': mac_disagreements,
        'rows_with_directional_membership_recovery_events': recovered,
        'reported_counts_below_observed_membership_events': count('count_below_observed_lower_bound'),
        'unresolved_udp_scenario_rows': count('udp_scenario_without_exported_udp_evidence'),
        'signed_packet_delta_rows_retained': count('signed_packet_delta_retained'),
        'source_materials_status': 'User reports matching files elsewhere; exact location not yet supplied in this run.',
        'provenance': proof,
    }
    write_json(out / 'audit/check_summary.json', scan_checks)
    lines = [
        '# Verified cleaned dataset copy', '',
        f'All **{total:,} permitted rows x 94 raw fields = {total*94:,} cells** were re-read from the duplicate and checked against the archived source records.',
        'The original CSV and existing experiment inputs passed before/after SHA256 checks. Original labels and split IDs are unchanged. The locked test and other unselected source rows were not profiled.', '',
        '## Delivered files', '',
        '- `raw/selected_train.csv.gz` and `selected_validation.csv.gz`: exact selected source cell archives.',
        '- `cleaned/cleaned_train.csv.gz` and `cleaned_validation.csv.gz`: complete cleaned masters, including quality and observed-count fields.',
        '- `cleaned/strict_train.csv.gz`: optional conservative training subset. All excluded records remain in the raw/master files and the review queue.',
        '- `model/`: full 655-feature matrices, original labels/IDs, explicit schema, and an aligned `strict_train/` subset. Prior checkpoints must not be used with this changed schema.',
        '- `audit/change_and_review_ledger.csv.gz`: row IDs, rule, severity, affected fields, and repair evidence.',
        '- `audit/`: class/device/protocol/time support, numeric shift, exact input conflicts, provenance and the training review queue.',
        '- `verification/`: integrity results, output hashes and executed project checks.', '',
        '## Concrete corrections and unresolved evidence', '',
        f'- Destination-MAC reported count copies source-MAC count in all {total:,} rows; it disagrees with observed destination membership in {mac_disagreements:,}. The suspect field is withheld and the model uses the explicitly named observed-list count.',
        f'- {recovered:,} directional-membership recovery events add only members explicitly recorded in another direction. Large or inconsistent lists retain unknown absence; missing source packets are not reconstructed.',
        f'- {count("count_below_observed_lower_bound"):,} other reported-count fields fall below the observed lower bound and are withheld. Reported approximations remain in the original archive.',
        f'- Unsupported zero-event summaries become explicit missing values. {count("signed_packet_delta_retained"):,} negative packet-delta rows, signed sensor values, legitimate zero windows and extreme measured traffic are retained.',
        f'- {count("udp_scenario_without_exported_udp_evidence"):,} UDP-ping scenario rows lack exported UDP protocol evidence. Their labels remain unchanged; this is a coverage/annotation review finding, not proof of a wrong label.', '',
        '| Partition | Master rows | No signal | Conflicting-input rows | Hard-invalid rows | Quality-eligible rows |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for split in SPLITS:
        v = summary['duplicates_and_conflicts'][split]
        lines.append(f'| {split} | {v["rows"]:,} | {v["no_signal_rows"]:,} | {v["rows_in_conflicting_groups"]:,} | {v["hard_invalid_rows"]:,} | {v["quality_eligible_rows"]:,} |')
    lines += ['', 'Reason counts overlap. Quality eligibility is a training-subset proposal; **all 30,000 validation rows remain in evaluation**. Identical measurements with different labels establish ambiguity in this representation, not which label is true.', '',
        f'The conservative training subset contains **{summary["strict_train_rows"]:,} rows and {summary["strict_train_classes"]}/84 classes**. It changes the distribution and has not been shown to improve model accuracy.', '',
        '## Coverage and remaining limits', '',
        f'Observed support covers {support["classes"]} classes and {support["devices"]} devices. Of {support["possible_class_device_pairs"]:,} possible class/device pairs, {support["class_device_pairs_without_train_rows"]:,} have no training rows and {support["class_device_pairs_without_validation_rows"]:,} have no validation rows. Missing combinations may be outside the capture design; fabricated rows would not repair coverage.',
        f'Exact shared cleaned model vectors across train/validation: {summary["train_validation_shared_model_vectors"]}. Class-conditional temporal-overlap classes: {len(summary["class_conditional_temporal_overlap"])}. Shared exact device/window keys: {proof["shared_train_validation_device_windows"]}.',
        'Window coverage describes only the selected partitions. Gaps can result from sampling and intentional purging; they do not establish missing packets in the original recording. Numeric shift describes distribution differences and does not justify deleting tails.',
        'The user reports that matching source files exist elsewhere; their location is needed for capture-level verification. The local UQ-IoT packet captures belong to another dataset. Missing packets, scenario labels, exporter semantics and capture completeness cannot be proven or reconstructed from this CSV alone.',
        '**This is a verified CSV cleanup, not certification of an error-free ground truth dataset. No new model was trained and no accuracy improvement or 90–95% result is claimed.**', '',
        '## Reproduction', '', 'From the project root with the project Python environment:', '', '```powershell',
        'python -B -m experiment.verify_clean_datasense --directory dataset/experiments/<new-output-directory>',
        'python -B -m experiment.report_datasense_integrity --directory dataset/experiments/<new-output-directory>',
        '```', '',
        'Fresh directories are mandatory for a new scan. Completed stages can be checked with `--stage verify`; partial outputs are preserved rather than overwritten.', '']
    (out / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(out / 'verification/report_hashes.json', {p.relative_to(out).as_posix(): digest(p)
        for p in [out/'RESULTS.md', *sorted((out/'audit').glob('*.csv')), out/'audit/check_summary.json',
                  out/'audit/coverage_summary.json', out/'audit/provenance_window_checks.json']})
    print(json.dumps(scan_checks, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    report(parser.parse_args().directory)
