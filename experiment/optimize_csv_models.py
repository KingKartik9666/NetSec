"""Resumable CSV-only upgrade study. Usage: start|status|stop|resume.

Inherits the authorized study deadline; never starts live-model training or
promotes models automatically. Holds the old controller lock to exclude mixed
training while sequential CUDA fits execute.
"""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiment import optimize_models as lifecycle
from experiment.csv_study_policy import (FAMILIES, TARGET, adaptive_trial, controls,
                                          independent_trials, ranking, request,
                                          supported_combination)
from experiment.csv_adaptive_policy import adaptive_decision

ROOT = lifecycle.ROOT
OLD = lifecycle.STUDY
PRIOR_CSV = ROOT / 'artifacts/training/csv_specialization_v1'
STUDY = ROOT / 'artifacts/training/csv_specialization_v2'
PREPARED = ROOT / 'dataset/prepared/datasense_full_protocols'
MODULE = 'experiment.optimize_csv_models'


def transition_state(previous: dict, *, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    if previous['deadline'] <= now:
        raise ValueError('Original authorized deadline expired; no automatic extension')
    return dict(started=previous['started'], csv_started=now, deadline=previous['deadline'],
                search_deadline=previous['search_deadline'], status='created', phase='csv_controls',
                active_trial=None, target_accuracy=TARGET, optimized_families=list(FAMILIES),
                input_profile='full_csv', test_loaded=False,
                live_stage={'status': 'blocked', 'reason': 'CSV training and independent validation first',
                            'requires_independent_labels': True, 'rf_fixed': True,
                            'previous_all_four_minimum_accuracy': .95})


def prepare_request(trial: dict, prepared: Path, protocol_id: str) -> dict:
    if trial.get('input_profile') != 'full_csv' or trial['model_id'] not in FAMILIES:
        raise ValueError('CSV study only accepts full_csv requests for the three upgraded families')
    return {**trial, 'prepared': prepared.relative_to(ROOT).as_posix(),
            'prepared_manifest_sha256': lifecycle.digest(prepared / 'manifest.json'), 'protocol_id': protocol_id}


def completed_results(verify: bool = False) -> list[dict]:
    rows = []
    adopted = STUDY / 'adopted_results.json'
    if adopted.exists():
        for record in json.loads(adopted.read_text()):
            path = ROOT / record['path']
            if lifecycle.digest(path) != record['sha256']:
                raise ValueError('Adopted CSV result changed; preserve and investigate')
            result = lifecycle.verified(path) if verify else json.loads(path.read_text())
            if result:
                result.setdefault('completed_at', path.stat().st_mtime)
                rows.append(result)
    for path in sorted((STUDY / 'runs').glob('*/attempt_*/result.json')):
        result = lifecycle.verified(path) if verify else json.loads(path.read_text())
        if result and result.get('status') == 'completed':
            result.setdefault('completed_at', path.stat().st_mtime)
            rows.append(result)
    return rows


def summary(result: dict) -> dict:
    return {k: result[k] for k in ('model_id', 'variant', 'seed', 'config', 'metrics',
                                  'checkpoint_path', 'checkpoint_sha256')}


def handoff(state: dict) -> None:
    rows = completed_results()
    state['completed_fit_records'] = dict(Counter(r['model_id'] for r in rows))
    lifecycle.write(STUDY / 'state.json', state)
    lines = ['# CSV-only study handoff', '',
             'Current request: four CSV models first, then four separately trained live versions.',
             'Upgrade XGBoost, genuine KAN and GitHub-derived HEM. Conventional RF stays fixed in both modes.',
             'RF is a documented conventional baseline; the paper did not disclose exact settings.',
             'Mixed CSV/live training is cancelled. No masked views or live-profile selection in this study.',
             'All three upgraded CSV families use 170 ordered features and original real labels.',
             'Target: 96% full-CSV validation accuracy per upgraded family; no guaranteed outcome.',
             'Default app inference remains CPU/offline; research fitting uses CUDA; test locked.',
             f"State: {state['status']}; phase: {state['phase']}; PID: {state.get('pid')}",
             f"Deadline UTC (unchanged): {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(state['deadline']))}",
             f"Completed fits: {state['completed_fit_records']}",
             f"Active trial: {json.dumps(state.get('active_trial'))}", '',
             'Run from the project root:', '```powershell',
             '.venv/Scripts/python.exe -B -m experiment.optimize_csv_models status',
             '.venv/Scripts/python.exe -B -m experiment.optimize_csv_models resume',
             '.venv/Scripts/python.exe -B -m experiment.optimize_csv_models stop', '```', '',
             'Do not resume experiment.optimize_models: its mixed objective was superseded.',
             'Original evidence: artifacts/training/optimization_24h_v1/ (preserved).',
             'Current evidence: artifacts/training/csv_specialization_v2/.',
             'CSV v1 checkpoints are hash-verified and adopted without retraining identical controls.',
             'Round/epoch telemetry: accuracy, macro precision/recall/F1, benign false alarms and missed attacks.',
             'Adaptive decisions are recorded in immutable requests and decisions.jsonl; plateau means',
             'less than 0.1 percentage point accuracy gain over six completed trials (minimum seven fits).',
             'No automatic promotion, live training or installed-app update.',
             'Next: finish CSV ablations/adaptive search and seed repeats, review validation,',
             'then explicitly integrate accepted checkpoints and independently validate deployment.',
             'Prior all-four 95% live gate remains; fixed RF currently does not satisfy it.', '']
    for family in FAMILIES:
        candidates = [r for r in rows if r['model_id'] == family and r['seed'] == 42]
        if candidates:
            best = max(candidates, key=ranking)
            lines.append(f"- {family}: best provisional full accuracy {best['metrics']['accuracy']:.6f}, "
                         f"macro F1 {best['metrics']['macro_f1']:.6f}; {best['variant']}")
    (ROOT / 'docs/RESUME_CONTEXT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def pin_inputs() -> dict:
    pins = lifecycle.pin_inputs()
    for name in ('experiment/optimize_models.py', 'experiment/optimize_csv_models.py',
                 'experiment/train_csv_models.py', 'experiment/csv_study_policy.py',
                 'framework/training/csv_xgboost.py', 'framework/training/trees.py',
                 'framework/training/tracking.py', 'experiment/csv_adaptive_policy.py',
                 'framework/training/csv_monitoring.py'):
        pins[name] = lifecycle.digest(ROOT / name)
    return pins


def initialize() -> None:
    """Create immutable study definition without resetting the original clock."""
    with lifecycle.StudyLock(OLD / 'controller.lock'), lifecycle.StudyLock(STUDY / 'controller.lock'):
        if (STUDY / 'state.json').exists():
            return
        old_state = json.loads((OLD / 'state.json').read_text())
        if (PRIOR_CSV / 'state.json').exists():
            prior = json.loads((PRIOR_CSV / 'state.json').read_text())
            if prior.get('active_trial') or prior.get('status') not in {'stopped', 'completed', 'failed', 'incomplete'}:
                raise RuntimeError('Stop CSV v1 before activating adaptive CSV v2')
        if old_state.get('active_trial') or old_state.get('status') not in {'stopped', 'completed', 'failed'}:
            raise RuntimeError('Stop the mixed study before starting CSV specialization')
        state = transition_state(old_state)
        from experiment.train_four_models import verify_common_partitions
        numeric = ROOT / 'dataset/prepared/datasense_full_train'
        identity = verify_common_partitions(numeric, PREPARED)
        original = json.loads((lifecycle.BASE / 'study.json').read_text())
        if identity != original['identity']:
            raise ValueError('Original CSV train/validation partitions changed')
        rf_path = OLD / 'prerequisites/random_forest_fixed_baseline/attempt_000/result.json'
        fixed = lifecycle.verified(rf_path)
        if not fixed or fixed.get('model_id') != 'random_forest':
            raise ValueError('Verified fixed conventional RF baseline is required')
        pins = pin_inputs()
        for name in pins:
            if name.endswith('.py'):
                destination = STUDY / 'source_snapshot' / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, destination)
        adopted = []
        for path in sorted((PRIOR_CSV / 'runs').glob('*/attempt_*/result.json')):
            result = lifecycle.verified(path)
            if result and result.get('input_profile') == 'full_csv' and result.get('checkpoint_reload_verified'):
                if result.get('protocol_id') != original['protocol_id']:
                    raise ValueError('Prior CSV result uses different observations')
                adopted.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': lifecycle.digest(path)})
        lifecycle.write(STUDY / 'adopted_results.json', adopted)
        definition = dict(protocol_id=original['protocol_id'], identity=identity, feature_count=170,
                          controls=controls(), trials=independent_trials(), target_accuracy=TARGET,
                          fixed_rf=summary(fixed), input_profile='full_csv', test_loaded=False,
                          selection='full validation accuracy, then macro F1; controls retain exact ties',
                          source_input_hashes=pins, parent_deadline=state['deadline'],
                          adopted_results_sha256=lifecycle.digest(STUDY / 'adopted_results.json'))
        lifecycle.write(STUDY / 'study.json', definition)
        lifecycle.write(STUDY / 'state.json', state)
        lifecycle.write(OLD / 'csv_supersession.json', {'successor': str(STUDY.relative_to(ROOT)),
                        'reason': 'User cancelled mixed CSV/live training', 'deadline_reset': False})
        lifecycle.write(STUDY / 'recovery.json', lifecycle.register_recovery(MODULE))
        handoff(state)


def frozen_trial(directory: Path, proposed: dict) -> dict:
    """Never reinterpret an already attempted trial after a later result arrives."""
    saved = sorted(directory.glob('attempt_*/request.json'))
    if not saved:
        return proposed
    original = json.loads(saved[0].read_text())
    for key in ('model_id', 'variant', 'seed', 'input_profile'):
        if original[key] != proposed[key]:
            raise ValueError('Persisted trial identity differs')
    return {key: original[key] for key in
            ('model_id', 'variant', 'config', 'seed', 'input_profile', 'research_policy', 'decision') if key in original}


def run_trial(trial: dict, state: dict, definition: dict) -> dict | None:
    directory = STUDY / 'runs' / f"{trial['model_id']}_{trial['variant']}_s{trial['seed']}"
    trial = frozen_trial(directory, trial)
    for adopted in completed_results():
        if all(adopted.get(k) == trial.get(k) for k in ('model_id', 'variant', 'seed', 'config', 'input_profile')):
            return adopted
    if trial.get('decision') and not directory.exists():
        with (STUDY / 'decisions.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps({'time': time.time(), **trial}) + '\n')
    prepared = prepare_request(trial, PREPARED, definition['protocol_id'])
    return lifecycle.run_fit(prepared, directory, state, study=STUDY,
                             worker_module='experiment.train_csv_models', on_update=handoff)


def stopped(state: dict, *, search: bool = False) -> bool:
    deadline = state['search_deadline'] if search else state['deadline']
    return time.time() >= deadline or (STUDY / 'STOP').exists()


def final_report(rows: list[dict], fixed_rf: dict, repeat_plan: dict | None = None) -> dict:
    report = dict(models={}, fixed_rf=fixed_rf, target_accuracy=TARGET, test_loaded=False,
                  live_training_started=False, promoted=False)
    for family in FAMILIES:
        seed42 = [r for r in rows if r['model_id'] == family and r['seed'] == 42]
        base = next((r for r in seed42 if r['variant'] == 'control'), None)
        if not base:
            report['models'][family] = {'status': 'control_incomplete'}
            continue
        leader = max(seed42, key=ranking)
        if repeat_plan and family in repeat_plan:
            chosen = repeat_plan[family]['candidate']['checkpoint_path']
            leader = next(r for r in seed42 if r['checkpoint_path'] == chosen)
        repeats = {kind: [r for r in rows if r['model_id'] == family and r['variant'] == 'repeat_' + kind]
                   for kind in ('control', 'candidate')}
        matched = {kind: {r['seed']: r for r in values} for kind, values in repeats.items()}
        configs = {'control': base['config'], 'candidate': leader['config']}
        repeat_complete = all(seed in matched[kind] and matched[kind][seed]['config'] == configs[kind]
                              for kind in matched for seed in (43, 44))
        improved = repeat_complete and all(ranking(matched['candidate'][seed]) > ranking(matched['control'][seed])
                                          for seed in (43, 44))
        tradeoff = (leader['metrics']['macro_recall'] < base['metrics']['macro_recall'] - .01 or
                    (leader['metrics'].get('false_positive_rate') or 0.) > (base['metrics'].get('false_positive_rate') or 0.) + .005)
        report['models'][family] = dict(control=summary(base), leader=summary(leader),
                    repeats={k: [summary(r) for r in v] for k, v in repeats.items()},
                    repeats_complete=repeat_complete,
                    promotion_eligible=ranking(leader) > ranking(base) and improved and not tradeoff,
                    recall_or_false_alarm_tradeoff=tradeoff,
                    target_repeated=repeat_complete and ranking(leader)[0] >= TARGET
                    and all(ranking(matched['candidate'][s])[0] >= TARGET for s in (43, 44)))
    return report


def terminal_status(report: dict, *, stopped: bool, expired: bool) -> str:
    if stopped:
        return 'stopped'
    if expired:
        return 'deadline_reached'
    return 'completed' if all(r.get('repeats_complete') for r in report['models'].values()) else 'incomplete'


def controller() -> None:
    # Cross-study exclusion also prevents the retired mixed controller starting.
    with lifecycle.StudyLock(OLD / 'controller.lock'), lifecycle.StudyLock(STUDY / 'controller.lock'):
        job = lifecycle.protect_descendants()
        state = json.loads((STUDY / 'state.json').read_text())
        if stopped(state):
            state.update(status='stopped' if (STUDY / 'STOP').exists() else 'deadline_reached', active_trial=None)
            handoff(state)
            return
        state.update(status='running', pid=os.getpid(), active_trial=None)
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            definition = json.loads((STUDY / 'study.json').read_text())
            if pin_inputs() != definition['source_input_hashes']:
                raise ValueError('CSV inputs/source changed; preserve evidence and version the study')
            if lifecycle.digest(STUDY / 'adopted_results.json') != definition['adopted_results_sha256']:
                raise ValueError('Adopted-result index changed')
            completed_results(verify=True)
            state['phase'] = 'csv_controls'
            for family, config in definition['controls'].items():
                if stopped(state):
                    break
                run_trial(request(family, 'control', config), state, definition)
            rows = completed_results()
            if not all(any(r['model_id'] == f and r['variant'] == 'control' for r in rows) for f in FAMILIES):
                if stopped(state):
                    state.update(status='stopped' if (STUDY / 'STOP').exists() else 'deadline_reached', active_trial=None)
                    return
                raise RuntimeError('CSV control incomplete; inspect failed attempt and resume within deadline')
            state['phase'] = 'independent_ablation_search'
            for trial in definition['trials']:
                if stopped(state, search=True):
                    break
                run_trial(trial, state, definition)
            state['phase'] = 'justified_combinations'
            rows = completed_results()
            for family in FAMILIES:
                if stopped(state, search=True):
                    break
                family_rows = [r for r in rows if r['model_id'] == family and r['seed'] == 42]
                base = next(r for r in family_rows if r['variant'] == 'control')
                trial = supported_combination(family, base, family_rows)
                if trial:
                    run_trial(trial, state, definition)
            # Continued bounded search changes one/two factors around measured
            # leaders. Persisted requests make restart independent of queue timing.
            state['phase'] = 'independent_ablation_search'
            for index in range(60):
                if stopped(state, search=True):
                    break
                rows = completed_results()
                if all(max(ranking(r)[0] for r in rows if r['model_id'] == f) >= TARGET for f in FAMILIES):
                    break
                for family in FAMILIES:
                    if stopped(state, search=True):
                        break
                    family_rows = [r for r in rows if r['model_id'] == family and r['seed'] == 42]
                    folder = STUDY / 'runs' / f'{family}_adaptive_{index:03d}_s42'
                    saved = sorted(folder.glob('attempt_*/request.json'))
                    if saved:
                        saved_request = json.loads(saved[0].read_text())
                        trial = {k: saved_request[k] for k in
                                 ('model_id', 'variant', 'config', 'seed', 'input_profile', 'research_policy', 'decision')
                                 if k in saved_request}
                    else:
                        leader = max(family_rows, key=ranking)
                        if ranking(leader)[0] >= TARGET:
                            continue
                        seen = {json.dumps(r['config'], sort_keys=True) for r in family_rows}
                        trial = adaptive_decision(family, family_rows, index, seen)
                    if trial:
                        run_trial(trial, state, definition)
            state['phase'] = 'repeat_leaders_and_controls'
            rows = completed_results()
            plan_path = STUDY / 'repeat_plan.json'
            if plan_path.exists():
                repeat_plan = json.loads(plan_path.read_text())
            else:
                repeat_plan = {}
                for family in FAMILIES:
                    candidates = [r for r in rows if r['model_id'] == family and r['seed'] == 42]
                    repeat_plan[family] = {'control': summary(next(r for r in candidates if r['variant'] == 'control')),
                                           'candidate': summary(max(candidates, key=ranking))}
                lifecycle.write(plan_path, repeat_plan)
            for family in FAMILIES:
                family_rows = [r for r in rows if r['model_id'] == family and r['seed'] == 42]
                base = next(r for r in family_rows if r['variant'] == 'control')
                leader = next(r for r in family_rows if r['checkpoint_path'] == repeat_plan[family]['candidate']['checkpoint_path'])
                for seed in (43, 44):
                    for name, candidate in [('control', base), ('candidate', leader)]:
                        if stopped(state):
                            break
                        run_trial(request(family, 'repeat_' + name, candidate['config'], seed), state, definition)
            report = final_report(completed_results(), definition['fixed_rf'], repeat_plan)
            lifecycle.write(STUDY / 'report.json', report)
            state.update(status=terminal_status(report, stopped=(STUDY / 'STOP').exists(),
                                                expired=time.time() >= state['deadline']), active_trial=None)
        except Exception as exc:
            state.update(status='failed', error=f'{type(exc).__name__}: {exc}', active_trial=None)
            raise
        finally:
            if os.name == 'nt':
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            handoff(state)


def launch() -> None:
    with lifecycle.StudyLock(OLD / 'controller.lock'), lifecycle.StudyLock(STUDY / 'controller.lock'):
        state = json.loads((STUDY / 'state.json').read_text())
        if state['deadline'] <= time.time() or state.get('status') == 'completed':
            print('Study finished or deadline expired; no automatic extension')
            return
        (STUDY / 'STOP').unlink(missing_ok=True)
    flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if os.name == 'nt' else 0
    with (STUDY / 'controller.log').open('a', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, '-B', '-u', '-m', MODULE, '_run'], cwd=ROOT,
                                  stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                  close_fds=True, creationflags=flags, start_new_session=os.name != 'nt')
    print(json.dumps({'launched_pid': child.pid, 'deadline': state['deadline'], 'input_profile': 'full_csv'}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['start', 'status', 'stop', 'resume', '_run'])
    args = parser.parse_args()
    if args.command == 'start':
        initialize()
        launch()
    elif args.command == 'resume':
        if (STUDY / 'state.json').exists():
            launch()
    elif args.command == 'stop':
        STUDY.mkdir(parents=True, exist_ok=True)
        (STUDY / 'STOP').touch()
        print('CSV stop requested; owned fit will be terminated')
    elif args.command == 'status':
        print((STUDY / 'state.json').read_text() if (STUDY / 'state.json').exists() else 'CSV study not started')
    else:
        controller()


if __name__ == '__main__':
    main()
