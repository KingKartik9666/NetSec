"""Compare original whitelisted training CSV cells with the live candidates."""
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment.datasense_quality import selected_rows, parse_list, censored_presence
from framework.training.live_features.data import DEFAULT
from framework.training.live_features.schema import RAW_NAMES, WITHHELD, PAYLOAD, derive
from framework.training.live_features.ports import join_port_features
from framework.training.live_study.protocol import ROOT, read, atomic, digest, relative


def review(study=DEFAULT):
    study = Path(study)
    output = study / 'verification'
    output.mkdir(parents=True, exist_ok=True)
    manifest = read(study / 'data/manifest.json')
    names = manifest['feature_names']
    x = np.load(study / 'data/X_train.npy', mmap_mode='r')
    y = np.load(study / 'data/y_train.npy')
    roles = np.load(study / 'data/roles.npy')
    ids = np.load(study / 'data/row_ids_train.npy')
    fit = np.flatnonzero(roles == 0)
    chosen = []
    families = set()
    for i in fit:
        family = manifest['classes'][y[i]].split('_', 1)[0]
        if family not in families:
            families.add(family)
            chosen.append(int(i))
    for name in ('network_packets_all_count', 'network_time-delta_min', 'network_tcp-flags_std_deviation', 'network_mss_max'):
        values = x[fit, names.index(name)]
        for position in (np.nanargmin(values), np.nanargmax(values)):
            if int(fit[position]) not in chosen:
                chosen.append(int(fit[position]))
    chosen = chosen[:16]
    indexes = {int(ids[i]): i for i in chosen}
    raw_names = [name for name in RAW_NAMES if name not in WITHHELD and name not in PAYLOAD]
    ports, port_names, _ = join_port_features(study)
    detail = output / 'csv_cell_comparison.csv'
    counts = dict(rows=0, raw_cells=0, derived_cells=0, port_cells=0)
    with detail.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_row_id', 'class', 'feature', 'source_or_recomputed', 'prepared', 'match'])
        for row_id, row in selected_rows(ROOT / 'dataset/combined_dataset.csv', sorted(indexes)):
            i = indexes[row_id]
            label = manifest['classes'][y[i]]
            raw = np.array([[float(row[name]) for name in raw_names]], np.float32)
            derived = derive(raw, raw_names, 10.)
            raw_values = dict(zip(raw_names, raw[0]))
            expected = {**raw_values, **{k: v[0] for k, v in derived.items()}}
            for name in names:
                value = expected[name]
                actual = x[i, names.index(name)]
                np.testing.assert_allclose(value, actual, rtol=1e-6, atol=1e-7, equal_nan=True,
                                           err_msg=f'CSV row{row_id}, {name}')
                writer.writerow([row_id, label, name, value, actual, True])
                counts['raw_cells' if name in raw_names else 'derived_cells'] += 1
            for direction in ('src', 'dst'):
                field = f'network_ports_{direction}'
                tokens, _ = parse_list(row[field], 'ports')
                chosen_ports = [n for n in port_names if n.startswith(field + '__present_')]
                vocabulary = [int(n.rsplit('_', 1)[1]) for n in chosen_ports]
                expected_ports = censored_presence(tokens, vocabulary, float(row[field + '_count']))[:-1]
                for name, value in zip(chosen_ports, expected_ports):
                    actual = ports[i, port_names.index(name)]
                    np.testing.assert_array_equal(value, actual, err_msg=f'CSV row{row_id}, {name}')
                    writer.writerow([row_id, label, name, value, actual, True])
                    counts['port_cells'] += 1
            counts['rows'] += 1
    if counts['rows'] != len(indexes):
        raise ValueError('Not all selected training rows were found')
    report = dict(status='passed', **counts, families=sorted(families), source_row_ids=sorted(indexes),
                  all_rows_original_fitting_role=True, all_numeric_cells_match=True,
                  interpretation='Direct CSV cell comparison; not publisher packet-extractor parity or label authentication',
                  payloads_saved=False, validation_loaded=False, test_loaded=False,
                  detail=relative(detail), detail_sha256=digest(detail),
                  source_csv_sha256=digest(ROOT / 'dataset/combined_dataset.csv'))
    atomic(output / 'csv_review.json', report)
    print(report)


if __name__ == '__main__':
    review()
