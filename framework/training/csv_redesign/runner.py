"""Sealed, restartable CSV redesign experiments independent of a chat session."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import optuna
import psutil

from framework.training.live_study.protocol import (ROOT, BRANCHES, HYBRIDS, atomic, read,
    relative, digest, fingerprint, verify, exclusive, source_pins, versions)
from framework.training.live_study.controller import Halt, pulse, check_limits, progress, make_request
from framework.training.live_study.processes import Job, launch, terminate
from framework.training.live_study.worker import verify_result
from framework.training.broad_live.search import load as load_optimizer, configuration as established_configuration

DEFAULT = ROOT/'artifacts/training/csv_redesign_20261030_v1'
DEFAULT_DATA = ROOT/'dataset/experiments/csv_audit_redesign_20261030_v1/processed'
CLI = ROOT/'experiment/train_csv_redesign.py'
ARMS = ('base_unfiltered', 'base_clean', 'enriched_clean', 'enriched_scenario_balanced',
        'enrich_no_port_geometry', 'enrich_no_scan_shape')
TERMINAL = {'completed', 'budget_exhausted', 'stopped', 'needs_attention', 'insufficient_disk_space'}


def study_path(path):
    path = Path(path).resolve()
    parent = (ROOT/'artifacts/training').resolve()
    if path.parent != parent or not path.name.startswith('csv_redesign_'):
        raise ValueError('CSV study must be a csv_redesign_* directory under artifacts/training')
    return path


def validate_request(request):
    if fingerprint({k: v for k, v in request.items() if k != 'request_fingerprint'}) != request['request_fingerprint']:
        raise ValueError('Persisted request was edited')


def selection_value(record):
    return float(record['selection_metrics']['accuracy'])


def _guard(candidate, baseline):
    return (candidate['macro_f1'] >= baseline['macro_f1'] - .0025
            and candidate['false_positive_rate'] is not None
            and baseline['false_positive_rate'] is not None
            and candidate['false_positive_rate'] <= baseline['false_positive_rate'] + .002)


def select_representation(records):
    """Predeclared matched-seed rule; no choice based on challenge scores."""
    if 'base_clean' not in records:
        raise ValueError('A matched base_clean control is required')
    seed_sets = [{r['seed'] for r in values} for values in records.values()]
    if not seed_sets[0] or any(s != seed_sets[0] for s in seed_sets) or any(
            len(values) != len(seed_sets[0]) for values in records.values()):
        raise ValueError('Each arm requires the identical complete seed set')
    summaries = {}
    for arm, values in records.items():
        summaries[arm] = {key: float(np.mean([v['selection_metrics'][key] for v in values]))
                          for key in ('accuracy', 'macro_f1', 'false_positive_rate')}
    baseline = summaries['base_clean']
    eligible = ['base_clean']
    for arm, score in summaries.items():
        if arm == 'base_clean':
            continue
        # Positive accuracy effect in every matched seed, not just best-seed luck.
        matched = {r['seed']: r['selection_metrics'] for r in records['base_clean']}
        improves = all(r['selection_metrics']['accuracy'] > matched[r['seed']]['accuracy']
                       and _guard(r['selection_metrics'], matched[r['seed']]) for r in records[arm])
        if improves and _guard(score, baseline):
            eligible.append(arm)
    chosen = max(eligible, key=lambda arm: (summaries[arm]['accuracy'], summaries[arm]['macro_f1']))
    return dict(arm=chosen, means=summaries, eligible=eligible, seeds=sorted(seed_sets[0]),
        rule='accuracy improves in every seed; macro F1 loss <=0.0025 and FPR increase <=0.002 per seed',
        selection_role='tune only', challenge_used=False, original_validation_used=False,
        conclusion='exploratory matched fitting comparison, not evidence of unseen-network accuracy')


def configuration(trial, family, smoke=False):
    cfg = established_configuration(trial, family, smoke)
    if family in HYBRIDS:
        cfg['runtime_seconds'] = 120 if smoke else 900
    elif not smoke:
        cfg['runtime_seconds'] = 7200
        if family == 'xgboost':
            cfg.update(n_estimators=1800, early_stopping_rounds=120)
        elif family != 'random_forest':
            cfg.update(epochs=180, patience=20)
    return cfg


def pilot_config(smoke=False):
    return dict(n_estimators=8 if smoke else 600, early_stopping_rounds=4 if smoke else 100,
        max_depth=5, learning_rate=.06, min_child_weight=2., subsample=.85,
        colsample_bytree=.9, reg_lambda=3., reg_alpha=.01, max_bin=128, nthread=2,
        weight_strategy='none', runtime_seconds=120 if smoke else 7200)


def verify_study(study):
    study = study_path(study)
    settings = read(study/'study.json')
    if fingerprint({k: v for k, v in settings.items() if k != 'protocol_fingerprint'}) != settings['protocol_fingerprint']:
        raise ValueError('Sealed CSV settings changed')
    data = (ROOT/settings['data_directory']).resolve()
    if not data.is_relative_to((ROOT/'dataset/experiments').resolve()) and not settings['smoke']:
        raise ValueError('CSV data path escaped dataset experiments')
    verify(settings['source_hashes'])
    if versions() != settings['versions']:
        raise ValueError('Training library versions changed')
    manifest = read(data/'manifest.json')
    if fingerprint(manifest) != settings['data_manifest_fingerprint']:
        raise ValueError('CSV data manifest changed')
    verify(manifest['array_hashes'])
    verify(manifest['input_hashes'])
    return settings, manifest, data


def prepare(study=DEFAULT, data=DEFAULT_DATA, *, hours=48., branch_trials=12, hybrid_trials=12, smoke=False):
    study = study_path(study)
    data = Path(data).resolve()
    if not 0 < hours <= 168 or not 1 <= branch_trials <= 100 or not 1 <= hybrid_trials <= 100:
        raise ValueError('Use explicit bounded hours and trial counts')
    if (study/'study.json').exists():
        return verify_study(study)[0]
    if not (data/'manifest.json').exists():
        if smoke:
            raise FileNotFoundError('Create smoke data before sealing')
        from .data import prepare as prepare_data
        if data != DEFAULT_DATA:
            raise ValueError('Automatic preparation only supports the default isolated data copy')
        prepare_data()
    manifest = read(data/'manifest.json')
    required = ('X_train.npy', 'y_train.npy', 'roles.npy', 'fit_unfiltered.npy', 'scenario_codes.npy', 'encoder.json')
    for name in required:
        if not (data/name).is_file():
            raise ValueError(f'Prepared data missing {name}')
    verify(manifest['array_hashes'])
    verify(manifest['input_hashes'])
    for name in required:
        if relative(data/name) not in {**manifest['array_hashes'], **manifest['input_hashes']}:
            raise ValueError(f'Prepared input is not hash sealed: {name}')
    roles = np.load(data/'roles.npy', allow_pickle=False)
    y = np.load(data/'y_train.npy', allow_pickle=False)
    if any(not (roles == i).any() for i in range(5)):
        raise ValueError('All five independent roles must be nonempty')
    if set(y[roles == 0]) != set(range(len(manifest['classes']))):
        raise ValueError('Every declared class needs cleaned fitting support')
    study.mkdir(parents=True, exist_ok=True)
    settings = dict(kind='csv_audit_redesign', smoke=smoke, hours=hours,
        trials_per_branch=branch_trials, trials_per_hybrid=hybrid_trials,
        pilot_seeds=[42] if smoke else [42, 43, 44], repeat_seeds=[43] if smoke else [43, 44],
        arms=list(ARMS), data_directory=relative(data), data_manifest_fingerprint=fingerprint(manifest),
        schema_fingerprint=fingerprint(manifest['feature_names']), source_hashes=source_pins(), versions=versions(),
        source_validation_policy='locked; no loader in the automatic loop', test_policy='locked',
        selection_rule='accuracy primary; guard macro F1 and false-positive regression',
        original_labels_changed=False, application_promotion=False)
    settings['protocol_fingerprint'] = fingerprint(settings)
    atomic(study/'study.json', settings)
    return settings


def run_job(study, state, request, base, trial=None):
    """Bounded subprocess attempts; completed results require matching hashes."""
    validate_request(request)
    base.mkdir(parents=True, exist_ok=True)
    attempts = sorted(base.glob('attempt*'))
    failures = 0
    for previous in attempts:
        if (previous/'result.json').exists():
            return verify_result(previous/'result.json', request), previous/'result.json'
        if (previous/'failure.json').exists():
            failures += 1
        elif not (previous/'interrupted.json').exists():
            atomic(previous/'failure.json', dict(status='interrupted', reason='owner exited before atomic result'))
            failures += 1
    while failures < 2:
        check_limits(study, state)
        verify_study(study)
        run = base/f'attempt{len(list(base.glob("attempt*")))+1:02d}'
        run.mkdir()
        atomic(run/'request.json', request)
        with Job() as job, (run/'worker.log').open('a', encoding='utf-8') as log:
            process = launch([sys.executable, str(CLI), '_worker', '--request', str(run/'request.json'),
                              '--job-name', job.name], ROOT, log)
            job.assign(process)
            started = time.time()
            timeout = min(state['deadline'], started+request['config'].get('runtime_seconds', 7200)+900)
            pulse(study, state, current=dict(family=request['family'], arm=request['arm'], phase=request['phase'],
                seed=request['seed'], run_directory=relative(run), worker_pid=process.pid,
                worker_created=psutil.Process(process.pid).create_time(), started_at=started))
            last_step = -1
            try:
                while process.poll() is None:
                    check_limits(study, state)
                    if time.time() >= timeout:
                        atomic(run/'failure.json', dict(status='failed', reason='bounded worker timeout'))
                        break
                    telemetry = progress(run)
                    if trial is not None and request['family'] in ('xgboost', 'kan', 'hem') and telemetry:
                        step = int(telemetry.get('round', telemetry.get('epoch', 0)))
                        value = telemetry.get('validation_accuracy')
                        if value is not None and step > last_step:
                            trial.report(float(value), step)
                            last_step = step
                            if trial.should_prune():
                                (run/'PRUNE').touch()
                    pulse(study, state, worker_progress=telemetry)
                    time.sleep(2)
            except Halt as stopped:
                atomic(run/'interrupted.json', dict(status=stopped.status, time=time.time()))
                raise
            finally:
                terminate(process)
        check_limits(study, state)
        if (run/'result.json').exists():
            return verify_result(run/'result.json', request), run/'result.json'
        if not (run/'failure.json').exists():
            atomic(run/'failure.json', dict(status='failed', exit_code=process.returncode, reason='no atomic evaluated result'))
        failures += 1
        pulse(study, state, last_failure=read(run/'failure.json'))
    return None, None


def pilot(study, state, settings):
    output = study/'representation.json'
    if output.exists():
        result = read(output)
        verify(result['result_hashes'])
        return result
    records = {arm: [] for arm in settings['arms']}
    pins = {}
    for seed in settings['pilot_seeds']:
        for arm in settings['arms']:
            request = make_request(settings, study, 'xgboost', 'pilot', pilot_config(settings['smoke']), seed, arm=arm)
            result, path = run_job(study, state, request, study/f'pilot/{arm}/seed{seed}')
            if result is None or result['status'] != 'completed':
                raise RuntimeError(f'Matched pilot did not finish: {arm} seed{seed}')
            records[arm].append(result)
            pins[relative(path)] = digest(path)
    result = select_representation(records)
    result['result_hashes'] = pins
    atomic(output, result)
    return result


def counts(study, families):
    return {name: {state: sum(t.state.name.lower() == state for t in load_optimizer(study, name).trials)
                   for state in ('complete', 'pruned', 'fail', 'running')} for name in families}


def proposal(study, settings, family, arm):
    optimizer = load_optimizer(study, family)
    pending = [t for t in optimizer.trials if t.state == optuna.trial.TrialState.RUNNING]
    if len(pending) > 1:
        raise ValueError('Multiple pending proposals for one family')
    trial = optuna.trial.Trial(optimizer, pending[0]._trial_id) if pending else optimizer.ask()
    if pending and 'request' in pending[0].user_attrs:
        request = pending[0].user_attrs['request']
        validate_request(request)
        return optimizer, trial, request
    cfg = configuration(trial, family, settings['smoke'])
    request = make_request(settings, study, family, 'search', cfg, 42, arm=arm, trial_number=trial.number)
    trial.set_user_attr('request', request)
    return optimizer, trial, request


def optimize(study, state, settings, families, target, arm):
    while True:
        check_limits(study, state)
        totals = counts(study, families)
        pulse(study, state, trial_counts={**state.get('trial_counts', {}), **totals})
        def finished(name):
            return sum(totals[name][s] for s in ('complete', 'pruned', 'fail'))
        pending = [name for name in families if finished(name) < target]
        if not pending:
            return
        family = min(pending, key=lambda name: (not totals[name]['running'], finished(name), list(families).index(name)))
        optimizer, trial, request = proposal(study, settings, family, arm)
        state['trial_counts'][family]['running'] = 1
        result, path = run_job(study, state, request, study/f'runs/{family}/trial{trial.number:04d}', trial)
        if result is None:
            optimizer.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue
        trial.set_user_attr('result', relative(path))
        trial.set_user_attr('result_sha256', digest(path))
        finished_state = optuna.trial.TrialState.COMPLETE if result['status'] == 'completed' else optuna.trial.TrialState.PRUNED
        optimizer.tell(trial, result['objective'] if result['status'] == 'completed' else None, state=finished_state)


def freeze(study, families):
    selected = {}
    for family in families:
        candidates = [t for t in load_optimizer(study, family).trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not candidates:
            raise RuntimeError(f'No completed genuine model for {family}')
        records = []
        for trial in candidates:
            path = ROOT/trial.user_attrs['result']
            if digest(path) != trial.user_attrs['result_sha256']:
                raise ValueError('Completed trial result changed')
            records.append((trial, verify_result(path)))
        reference = records[0][1]['selection_metrics']
        guarded = [(trial, result) for trial, result in records if _guard(result['selection_metrics'], reference)]
        trial, result = max(guarded or records[:1], key=lambda pair: (selection_value(pair[1]),
                                                    pair[1]['selection_metrics']['macro_f1'], -pair[0].number))
        selected[family] = dict(result=trial.user_attrs['result'], result_sha256=trial.user_attrs['result_sha256'],
            config=result['config'], seed=result['seed'], arm=result['arm'], objective=result['objective'],
            selection_guard_reference_trial=records[0][0].number)
    return selected


def execute(study=DEFAULT):
    study = study_path(study)
    with exclusive(study/'controller.lock'):
        settings, _, _ = verify_study(study)
        state = read(study/'state.json') if (study/'state.json').exists() else dict(
            started_at=time.time(), deadline=time.time()+settings['hours']*3600)
        if state.get('status') in TERMINAL - {'stopped'}:
            return
        try:
            pulse(study, state, status='running', phase='matched_data_and_feature_comparison', current=None)
            representation = pilot(study, state, settings)
            arm = representation['arm']
            pulse(study, state, phase='branch_search', selected_arm=arm)
            optimize(study, state, settings, BRANCHES, settings['trials_per_branch'], arm)
            selected = read(study/'selected.json') if (study/'selected.json').exists() else dict(branches=freeze(study, BRANCHES))
            atomic(study/'selected.json', selected)
            pulse(study, state, phase='hybrid_search')
            optimize(study, state, settings, HYBRIDS, settings['trials_per_hybrid'], arm)
            if 'hybrids' not in selected:
                selected['hybrids'] = freeze(study, HYBRIDS)
                atomic(study/'selected.json', selected)
            pulse(study, state, phase='seed_repeats')
            for seed in settings['repeat_seeds']:
                for family, entry in {**selected['branches'], **selected['hybrids']}.items():
                    base = study/f'repeats/{family}/seed{seed}'
                    cfg = entry['config']
                    if family in HYBRIDS:
                        cfg = dict(frozen_combiner=read(ROOT/entry['result'])['checkpoint'], runtime_seconds=900)
                    request = make_request(settings, study, family, 'repeat', cfg, seed, arm=arm)
                    result, _ = run_job(study, state, request, base)
                    if result is None:
                        raise RuntimeError(f'Seed repeat failed: {family} seed{seed}')
                    atomic(base/'result-link.json', result)
            atomic(study/'selection_seal.json', dict(selected_sha256=digest(study/'selected.json'),
                representation_sha256=digest(study/'representation.json'), sealed_at=time.time()))
            pulse(study, state, phase='frozen_internal_challenge')
            request = make_request(settings, study, 'all_models', 'assessment', dict(runtime_seconds=3600), arm=arm)
            result, path = run_job(study, state, request, study/'assessment')
            if result is None:
                raise RuntimeError('Frozen internal challenge assessment failed')
            atomic(study/'completion.json', dict(status='completed', assessment_result=relative(path),
                assessment_sha256=digest(path), finished_at=time.time(), original_validation_loaded=False,
                test_loaded=False, installed_models_changed=False, branch_trials=counts(study, BRANCHES),
                hybrid_trials=counts(study, HYBRIDS), repeat_seeds=settings['repeat_seeds']))
            pulse(study, state, status='completed', phase='complete', current=None)
        except Halt as halt:
            pulse(study, state, status=halt.status, current=None)
        except BaseException as error:
            pulse(study, state, status='needs_attention', current=None, error=str(error), traceback=traceback.format_exc())
            raise


def supervise(study=DEFAULT):
    """Bounded restart watchdog; Windows jobs own only this study's children."""
    study = study_path(study)
    lock = exclusive(study/'supervisor.lock')
    try:
        lock.__enter__()
    except OSError:
        return
    failures = 0
    try:
        while not (study/'STOP').exists():
            state = read(study/'state.json') if (study/'state.json').exists() else {}
            if state.get('status') in TERMINAL:
                return
            with Job() as job, (study/'controller.log').open('a', encoding='utf-8') as log:
                child = launch([sys.executable, str(CLI), '_controller', '--study', str(study), '--job-name', job.name], ROOT, log)
                job.assign(child)
                started = time.time()
                try:
                    while child.poll() is None:
                        atomic(study/'supervisor.json', dict(status='watching', pid=os.getpid(),
                            created=psutil.Process().create_time(), controller_pid=child.pid,
                            heartbeat=time.time(), controller_restarts=failures))
                        state = read(study/'state.json') if (study/'state.json').exists() else {}
                        if time.time()-max(started, state.get('heartbeat', started)) > 600:
                            break
                        time.sleep(5)
                finally:
                    terminate(child)
            state = read(study/'state.json') if (study/'state.json').exists() else {}
            if state.get('status') in TERMINAL:
                return
            failures += 1
            if failures >= 3:
                atomic(study/'state.json', dict(state, status='needs_attention',
                    error='Three unexpected controller exits', heartbeat=time.time()))
                return
    finally:
        atomic(study/'supervisor.json', dict(status='exited', pid=os.getpid(), heartbeat=time.time(), restarts=failures))
        lock.__exit__(None, None, None)


def start(study=DEFAULT):
    study = study_path(study)
    verify_study(study)
    state = read(study/'state.json') if (study/'state.json').exists() else {}
    if state.get('status') in TERMINAL - {'stopped'}:
        raise ValueError('Study reached a terminal state; inspect evidence before creating a new revision')
    (study/'STOP').unlink(missing_ok=True)
    if state.get('status') == 'stopped':
        atomic(study/'state.json', dict(state, status='resuming'))
    with (study/'supervisor.log').open('a', encoding='utf-8') as log:
        process = launch([sys.executable, str(CLI), 'supervise', '--study', str(study)], ROOT, log)
    return dict(launcher_pid=process.pid, study=relative(study), status='launched; verify heartbeat and worker identity')


def status(study=DEFAULT):
    study = study_path(study)
    result = {key: read(study/f'{key}.json') if (study/f'{key}.json').exists() else None
              for key in ('state', 'supervisor', 'representation', 'completion')}
    state = result['state'] or {}
    result['heartbeat_age_seconds'] = time.time()-state['heartbeat'] if 'heartbeat' in state else None
    pid = state.get('controller_pid')
    try:
        process = psutil.Process(pid) if pid else None
        result['controller_identity_alive'] = bool(process and abs(process.create_time()-state['controller_created']) < .1
            and 'train_csv_redesign.py' in ' '.join(process.cmdline()))
    except (psutil.Error, KeyError):
        result['controller_identity_alive'] = False
    return result
