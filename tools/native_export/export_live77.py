"""Export the live 77-class XGBoost model (live77) into a new SHA-pinned live bundle (integration plan step 3).

    python -m tools.native_export.export_live77 [--output artifacts/application/live_models_v1/live77_xgboost]

Writes bundle.json, a byte copy of model.ubj and golden_live77.json into a new folder (an existing one is refused).
Owner decisions of 2026-11-03 override the plan: 10 s windows (window_seconds 10), bundle location, model_id, label
and provenance note. The label and note come from the contract and must pass ``shippable`` (provenance_review
ship_allowed, no stale 1 s claim, Live.tsx caveat filter); otherwise nothing is exported. The pipeline the app must reproduce is raw softprob -> temperature calibration -> class merge
(recon_ping-sweep into recon_scan) -> argmax, lowest index on ties.

Only validation arrays are read (all rows for the study replay; golden rows from roles 0/1/2 only). The held-out test
partition is never opened: every input goes through ``guard``. Nothing is fitted; research artifacts are only read.
The golden file is the parity target for the Rust loader (plan step 4): rows, raw softprob and margins (CPU xgboost
3.1.3), calibrate.apply output, merged vector, decisions, and exact TreeSHAP for the decided class and classes 70/71.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

from tools.native_export.export import digest, relative, sample_indices, write_json

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'artifacts/training/live77_20261031_v1'
JOB = STUDY / 'jobs/77_live_final'
MODEL = JOB / 'attempt_0/model'
DATA = ROOT / 'dataset/experiments/csv_v3_cascade_20261030_v1'
RULES = ROOT / 'artifacts/training/upgrades2_live_20261101_v1/live_output_rules.json'
SPEC = ROOT / 'artifacts/training/class_merge_20261101/spec.json'
CONTRACT = ROOT / 'framework/training/csv_v3/live77_contract.json'
DECISIONS = ROOT / 'artifacts/app_v0_6_20261102/live77_decisions.json'
OUTPUT = ROOT / 'artifacts/application/live_models_v1/live77_xgboost'   # owner decision (defaults_chosen_by_app_chat)
PROTECTED = (STUDY, ROOT / 'dataset', DATA, ROOT / 'framework/models', ROOT / 'artifacts/training')
CHECKPOINT_SHA = '2ebc7f2da863d79d65b03ed785752b6b48d1bd273f0f446fc9235d059ac272e3'
CHECKPOINT_BYTES = 5165479
RULES_VALUE_SHA = 'b29e6a19fb6e1dad478358ea92cedd850746467ef72d37b3454ca734de2c4e46'
XGBOOST_VERSION = '3.1.3'
LOG_FLOOR = 1e-7                       # csv_v3.calibrate.EPS; does not bind (alpha/K = 2.83e-3), written for symmetry
ROLES = (0, 1, 2)                      # tuning, calibration, assessment; role -1 rows are not golden candidates
ROLE_NAMES = {0: 'tuning', 1: 'calibration', 2: 'assessment'}
GOLDEN_ROWS = 128
SHAP_TOLERANCE = 1e-4
TEST_NAME = re.compile(r'(^|[^a-z])test', re.I)
STALE_WINDOW = re.compile(r'\b1[ -]?s(econd)?\b')    # a 1 s window claim; the owner set 10 s windows
CAVEAT = re.compile(r'unverif|untested|provisional|limitation|caveat|experimental|accuracy_verified|not[ _-]?verified'
                    r'|unvalidated', re.I)          # Live.tsx caveat filter; the shipped provenance text must survive it


def shippable(contract, d):
    """The contract's label and note may ship: review allows it, no stale 1 s claim, not hidden by the Live.tsx filter,
    and traceable to the decisions file (verbatim, or its app-chat default kept verbatim as superseded)."""
    pr, label, note = contract['provenance_review'], contract['label'], contract['provenance_note']
    traced = ((label, note) == (d['label'], d['provenance_note']) if pr['status'] == 'accepted' else
              pr['status'] == 'corrected_under_owner_window_override' and (label, note) == (pr['corrected_label'], pr['corrected_provenance_note'])
              and (pr['superseded']['label'], pr['superseded']['provenance_note']) == (d['label'], d['provenance_note']))
    return (pr['ship_allowed'] is True and traced and not STALE_WINDOW.search(f'{label} {note}')
            and not CAVEAT.search(label) and not CAVEAT.search(note) and 'held-out test was not used' in note)


def guard(path):
    """Every input passes here: a path naming a test partition is refused before it is opened."""
    path = Path(path)
    if any(TEST_NAME.search(part) for part in path.parts):
        raise ValueError(f'refusing to open {path}: the held-out test partition is locked')
    return path


def read_json(path):
    return json.loads(guard(path).read_text(encoding='utf-8'))


def sources():
    """Pinned sources, cross-checked against each other. Any disagreement stops the export."""
    schema, result, rules = read_json(MODEL/'model_schema.json'), read_json(JOB/'result.json'), read_json(RULES)
    spec, contract, decisions = read_json(SPEC), read_json(CONTRACT), read_json(DECISIONS)
    value, d = rules['value'], decisions['defaults_chosen_by_app_chat']
    checks = {
        'checkpoint sha': digest(guard(MODEL/'model.ubj')) == schema['checkpoint_sha256'] == CHECKPOINT_SHA,
        'checkpoint bytes': (MODEL/'model.ubj').stat().st_size == CHECKPOINT_BYTES,
        'schema': (schema['family'], schema['task'], schema['n_features'], schema['n_classes'], schema['benign_index'])
                  == ('xgboost', '77', 252, 77, 0) and schema['classes'][0] == 'benign',
        'result': result['status'] == 'complete' and result['job_id'] == value['final_job'] == '77_live_final',
        'rules value sha': rules['sha256'] == RULES_VALUE_SHA and value['logit_adjustment'] is None,
        'calibration': value['calibration']['kind'] == result['calibration']['config']['kind'] == 'temperature'
                       and all(value['calibration'][k] == result['calibration']['config'][k] for k in ('temperature', 'alpha')),
        'merge spec': spec['classes'] == schema['classes'] and (spec['source_index'], spec['target_index']) == (71, 70)
                      and all(value['class_merge'][k] == spec[k] for k in ('name', 'version', 'source_class', 'target_class', 'rule')),
        'contract': contract['feature_names'] == schema['feature_names'] and contract['model']['checkpoint_sha256'] == CHECKPOINT_SHA,
        'window': decisions['window']['length_seconds'] == contract['window']['length_seconds'] == 10,
        'provenance': contract['model_id'] == d['model_id'] and shippable(contract, d),
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise ValueError(f'live77 sources disagree ({", ".join(bad)}); rebuild the contract or ask the owner')
    return dict(schema=schema, result=result, rules=rules, spec=spec, contract=contract, decisions=decisions)


def load_booster(path):
    import xgboost as xgb
    if xgb.__version__ != XGBOOST_VERSION:
        raise ValueError(f'golden needs xgboost {XGBOOST_VERSION}, found {xgb.__version__}')
    booster = xgb.Booster()
    booster.load_model(str(guard(path)))
    booster.set_param({'nthread': 1})       # deterministic CPU replay
    if json.loads(booster.save_config())['learner']['objective']['name'] != 'multi:softprob':
        raise ValueError('live77 checkpoint is not multi:softprob')
    return booster


def golden_rows(X, y, roles, raw_decision, decision, names, n=GOLDEN_ROWS):
    """Deterministic validation rows from roles 0/1/2: every merge flip (all raw-argmax ping-sweep rows among
    them), true ping-sweep rows, NaN-pattern edges, 10 s windows, then a class-stratified fill per role."""
    keep = np.isin(roles, ROLES)
    col = names.index
    missing = np.isnan(X)
    ports = {d: [i for i, m in enumerate(names) if m.startswith(f'network_ports_{d}__present_') and not m.endswith('UNSEEN')]
             for d in ('src', 'dst')}
    flags = {'merge_flip': decision != raw_decision, 'raw_argmax_ping_sweep': raw_decision == 71, 'true_ping_sweep': y == 71,
             'zero_packets': X[:, col('quality_no_packets_observed')] == 1,
             # presence NaN marks a list capped at 1000 (validation: NaN rows == capped rows, convention checks)
             'src_ports_capped': missing[:, ports['src']].any(1), 'dst_ports_capped': missing[:, ports['dst']].any(1),
             'window_10s': X[:, col('window_duration_seconds')] == 10}
    tags = {}

    def add(rows, tag):
        for r in rows:
            tags.setdefault(int(r), set()).add(tag)

    add(np.flatnonzero(keep & flags['merge_flip']), 'merge_flip')
    for role in ROLES:
        mine = keep & (roles == role)
        add(np.flatnonzero(mine & flags['true_ping_sweep'])[:2], 'true_ping_sweep')
        for tag in ('zero_packets', 'src_ports_capped', 'dst_ports_capped'):
            add(np.flatnonzero(mine & flags[tag])[:1], tag)
        count = np.where(mine, missing.sum(1), -1)
        add([int(np.argmax(count))], 'most_missing')
        tens = np.flatnonzero(mine & flags['window_10s'] & ~np.isin(np.arange(len(y)), list(tags)))
        add(tens[sample_indices(y[tens], 4)], 'window_10s')
    left = n - len(tags)
    if left < len(ROLES):
        raise ValueError('golden edge rows exceed the golden size')
    for k, role in enumerate(ROLES):
        free = np.flatnonzero(keep & (roles == role) & ~np.isin(np.arange(len(y)), list(tags)))
        add(free[sample_indices(y[free], left // len(ROLES) + (k < left % len(ROLES)))], 'stratified')
    rows = np.array(sorted(tags), dtype=np.int64)
    for name, flag in flags.items():
        add(rows[flag[rows]], name)
    return rows, {int(r): sorted(t) for r, t in tags.items()}


def export(destination):
    from framework.training.csv_v3 import calibrate, class_merge, live77
    import xgboost as xgb
    src = sources()
    schema, result, rules, spec, contract, decisions = (src[k] for k in ('schema', 'result', 'rules', 'spec', 'contract', 'decisions'))
    names, classes = schema['feature_names'], schema['classes']
    if live77.feature_names(list(live77.EXACT)) != names:
        raise ValueError('live77 groups no longer give the schema feature order')
    columns = live77.columns(list(live77.EXACT))
    X = np.ascontiguousarray(np.load(guard(DATA/'X_validation.npy'), mmap_mode='r')[:, columns], dtype=np.float32)
    y = np.load(guard(DATA/'y77_validation.npy'))
    roles = np.load(guard(DATA/'validation_roles.npy'))
    ids = np.load(guard(DATA/'row_ids_validation.npy'))
    if not (len(X) == len(y) == len(roles) == len(ids)):
        raise ValueError('validation arrays are not aligned')
    booster = load_booster(MODEL/'model.ubj')
    raw = booster.predict(xgb.DMatrix(X, missing=np.nan))
    cfg = dict(kind='temperature', temperature=rules['value']['calibration']['temperature'],
               alpha=rules['value']['calibration']['alpha'], log_floor=LOG_FLOOR)
    cal = calibrate.apply(cfg, raw)
    merged = class_merge.merge_proba(cal, classes)
    raw_decision, decision = raw.argmax(1), merged.argmax(1)
    if not np.array_equal(cal.argmax(1), raw_decision):
        raise ValueError('temperature calibration changed a decision')
    y_merged = class_merge.merge_labels(y, classes)
    scored = np.isin(roles, ROLES)
    study_raw, study_cal = np.load(guard(JOB/'attempt_0/val_proba.npy')), np.load(guard(JOB/'val_proba_cal.npy'))
    study = dict(rows=int(scored.sum()), raw_max_abs_difference=float(np.abs(raw[scored]-study_raw[scored]).max()),
                 calibrated_max_abs_difference=float(np.abs(cal[scored]-study_cal[scored]).max()),
                 sources={relative(JOB/'attempt_0/val_proba.npy'): digest(JOB/'attempt_0/val_proba.npy'),
                          relative(JOB/'val_proba_cal.npy'): digest(JOB/'val_proba_cal.npy')})
    replay = {}
    for role in ROLES:
        s, name = roles == role, ROLE_NAMES[role]
        replay[name] = dict(role=role, rows=int(s.sum()), raw_accuracy=float(np.mean(raw_decision[s] == y[s])),
                            merged_accuracy=float(np.mean(decision[s] == y_merged[s])),
                            result_json_accuracy=result['report'][name]['accuracy'])
        if replay[name]['raw_accuracy'] != replay[name]['result_json_accuracy'] or replay[name]['rows'] != result['report'][name]['rows']:
            raise ValueError(f'CPU replay of the {name} role differs from result.json')

    rows, tags = golden_rows(X, y, roles, raw_decision, decision, names)
    sample = X[rows]
    dm = xgb.DMatrix(sample, missing=np.nan)
    margins, contribs = booster.predict(dm, output_margin=True), booster.predict(dm, pred_contribs=True)
    if not np.array_equal(booster.predict(dm), raw[rows]):
        raise ValueError('golden rows do not replay the batch prediction')
    reconstruction = float(np.abs(contribs.sum(-1)-margins).max())
    if reconstruction > SHAP_TOLERANCE:
        raise ValueError(f'TreeSHAP reconstruction error {reconstruction} exceeds {SHAP_TOLERANCE}')
    records = []
    for j, i in enumerate(rows):
        explained = sorted({int(decision[i]), 70, 71})
        records.append(dict(
            validation_row=int(i), row_id=str(ids[i]), role=int(roles[i]), label_index=int(y[i]), label=classes[int(y[i])],
            label_merged_index=int(y_merged[i]), tags=tags[int(i)],
            values=[float(v) if np.isfinite(v) else None for v in sample[j]],
            raw=raw[i].tolist(), margins=margins[j].tolist(), calibrated=cal[i].tolist(), merged=merged[i].tolist(),
            raw_decision=int(raw_decision[i]), calibrated_decision=int(cal[i].argmax()), decision=int(decision[i]),
            decision_class=classes[int(decision[i])], merge_flip=bool(decision[i] != raw_decision[i]),
            contributions={str(k): contribs[j, k].tolist() for k in explained}))
    accuracy = {str(role): dict(rows=int(sum(roles[rows] == role)),
                                raw=float(np.mean(raw_decision[rows][roles[rows] == role] == y[rows][roles[rows] == role])),
                                merged=float(np.mean(decision[rows][roles[rows] == role] == y_merged[rows][roles[rows] == role])))
                for role in ROLES}
    golden = dict(version=1, model_id=contract['model_id'], partition='validation', roles_used=list(ROLES), test_loaded=False,
                  xgboost_version=XGBOOST_VERSION, predictor='CPU, nthread 1, DMatrix missing=NaN, float32 rows',
                  feature_names=names, classes=classes, calibration=cfg,
                  class_merge=dict(source_index=71, target_index=70), shap='exact TreeSHAP (pred_contribs); 253 = 252 features + bias',
                  tolerance=dict(probability=1e-6, shap=SHAP_TOLERANCE), shap_reconstruction_max_abs=reconstruction,
                  number_types=dict(values='float32', raw='float32', margins='float32', calibrated='float32', contributions='float32',
                                    merged='float64 sums of the float32 calibrated values (class_merge.merge_proba upcasts); '
                                           'a float32 replay differs by up to ~3e-8, so compare within tolerance.probability, '
                                           'not bit-for-bit'),
                  selection=('roles 0/1/2 only: every merge flip, up to 2 true ping-sweep rows, the first zero-packet, '
                             'src-capped and dst-capped row and the most-missing row, and 4 class-spread 10 s rows per role, '
                             'then a class-stratified fill split evenly over the roles (export.sample_indices)'),
                  role_replay=replay, study_replay=study, golden_accuracy=accuracy,
                  note='golden accuracy is over deliberately hard rows; role_replay is the full-role replay of result.json',
                  rows=records)

    destination.mkdir(parents=True)
    shutil.copyfile(MODEL/'model.ubj', destination/'model.ubj')
    if digest(destination/'model.ubj') != CHECKPOINT_SHA:
        raise ValueError('copied checkpoint differs')
    write_json(destination/'golden_live77.json', golden)
    m = rules['value']['class_merge']
    w = decisions['window']
    bundle = dict(
        version=1, model_id=contract['model_id'], kind='xgboost', label=contract['label'],
        provenance_note=contract['provenance_note'], provenance_review=contract['provenance_review'],
        feature_names=names, classes=classes, benign_index=0, n_features=len(names), n_classes=len(classes),
        checkpoint='model.ubj', checkpoint_sha256=CHECKPOINT_SHA, bytes=CHECKPOINT_BYTES,
        xgboost_version=XGBOOST_VERSION, objective='multi:softprob',
        calibration=cfg, logit_adjustment=None,
        class_merge=dict(name=m['name'], version=m['version'], source_class=m['source_class'], target_class=m['target_class'],
                         source_index=spec['source_index'], target_index=spec['target_index'], rule=m['rule'],
                         applies='after calibration, on the displayed 77-class vector',
                         spec=relative(SPEC), spec_sha256=digest(SPEC)),
        window_seconds=w['length_seconds'], window=dict(contract['window'], scoring=w['scoring']),
        feature_contract=dict(path=relative(CONTRACT), sha256=digest(CONTRACT), version=contract['version'],
                              feature_names_sha256=contract['feature_names_sha256']),
        output_rules=dict(path=relative(RULES), file_sha256=digest(RULES), value_sha256=rules['sha256'], frozen=rules['frozen']),
        output=dict(order=['raw softprob (multi:softprob)', 'temperature calibration (calibrate.apply)',
                           'class merge recon_ping-sweep into recon_scan', 'argmax of the merged vector, lowest index on ties'],
                    score_kind='calibrated_probability_after_class_merge', raw_probabilities_kept_for_audit=True,
                    binary_attack_probability=None,
                    binary_note='no live77 binary calibration exists; 1 - p(benign) must not be shown as a calibrated attack probability'),
        explanation=dict(kind='exact_treeshap', reconstruction_tolerance=SHAP_TOLERANCE,
                         after_merge=decisions['rules']['shap_after_merge'], margins='raw margins, not probabilities'),
        source=dict(job=result['job_id'], checkpoint=relative(MODEL/'model.ubj'),
                    schema=relative(MODEL/'model_schema.json'), schema_sha256=digest(MODEL/'model_schema.json'),
                    result=relative(JOB/'result.json'), result_sha256=digest(JOB/'result.json'),
                    decisions=relative(DECISIONS), decisions_sha256=digest(DECISIONS),
                    validation_data=relative(DATA)),
        validation_evidence=dict(scope='DataSense validation roles; the held-out test partition was not scored',
                                 **{ROLE_NAMES[r]: replay[ROLE_NAMES[r]]['raw_accuracy'] for r in ROLES}),
        golden=dict(path='golden_live77.json', sha256=digest(destination/'golden_live77.json'), rows=len(records),
                    merged_comparison='merged is float64; compare within 1e-6, not bit-for-bit'),
        test_evaluated=False, live_validation=False, publisher_equivalence_verified=False)
    write_json(destination/'bundle.json', bundle)
    if (destination/'bundle.json').stat().st_size > 1024*1024:
        raise ValueError('bundle.json exceeds the 1 MiB live bundle bound')
    return dict(bundle_sha256=digest(destination/'bundle.json'), golden_rows=len(records),
                flips=int(sum(r['merge_flip'] for r in records)), shap_reconstruction_max_abs=reconstruction,
                study_replay=study['raw_max_abs_difference'], role_replay={k: v['raw_accuracy'] for k, v in replay.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists() or any(p == destination or p in destination.parents for p in PROTECTED):
        raise SystemExit(f'Export needs a new folder outside the research and retired bundle folders: {destination}')
    print(json.dumps(dict(export(destination), output=str(destination))), flush=True)


if __name__ == '__main__':
    main()
