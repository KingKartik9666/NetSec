"""Broad adaptation phases, reusing owned workers and calibration transactions."""
from pathlib import Path
import sys
import time
import traceback

import optuna

from framework.training.live_study.protocol import (
    ROOT, BRANCHES, HYBRIDS, read, atomic, relative, digest, exclusive, verify_study,
)
from framework.training.live_study.controller import (
    Halt, check_limits, pulse, run_job, make_request, freeze,
)
from framework.training.live_study.worker import verify_result
from framework.training.live_study.processes import Job, launch, terminate
from . import search


def finished_counts(counts):
    return sum(counts[key] for key in ('complete', 'pruned', 'fail'))


def counts(study, families):
    return {name: {state: sum(t.state.name.lower() == state for t in search.load(study, name).trials)
                   for state in ('complete', 'pruned', 'fail', 'running')} for name in families}


def optimize(study, state, families, target):
    while True:
        check_limits(study, state)
        totals = counts(study, families)
        pulse(study, state, trial_counts={**state.get('trial_counts', {}), **totals})
        pending = [name for name in families if finished_counts(totals[name]) < target]
        if not pending:
            return
        family = min(pending, key=lambda name: (not totals[name]['running'],
                                               finished_counts(totals[name]), list(families).index(name)))
        optimizer, trial, request = search.proposal(study, family)
        state['trial_counts'][family]['running'] = 1
        result, path = run_job(study, state, request, study / f'runs/{family}/trial{trial.number:04d}', trial)
        if result is None:
            trial.set_user_attr('failure', 'Two bounded failed attempts; worker logs preserved')
            optimizer.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue
        trial.set_user_attr('result', relative(path))
        trial.set_user_attr('result_sha256', digest(path))
        trial.set_user_attr('calibration', result['calibration'])
        trial.set_user_attr('cpu_reload', True)
        completed = result['status'] == 'completed'
        optimizer.tell(trial, result['objective'] if completed else None,
                       state=optuna.trial.TrialState.COMPLETE if completed else optuna.trial.TrialState.PRUNED)
        previous = state.setdefault('best_tuning', {}).get(family)
        if completed and (previous is None or result['objective'] > previous['macro_f1']):
            score = result['selection_metrics']
            state['best_tuning'][family] = dict(trial=trial.number, macro_f1=result['objective'],
                accuracy=score['accuracy'], alert_false_positive_rate=score['screening']['false_positive_rate'],
                alert_attack_recall=score['screening']['attack_recall'],
                cold_single_window_seconds=result['cold_single_window_seconds']['median'], result=relative(path))


def run_assessment(study, state):
    """Keep final validation outside the controller heartbeat and fitting worker."""
    base = study / 'assessment'
    base.mkdir(exist_ok=True)
    attempts = sorted(base.glob('attempt*'))
    for path in attempts:
        if (path / 'result.json').exists():
            verify_result(path / 'result.json')
            return path / 'result.json'
    failures = sum((p / 'failure.json').exists() for p in attempts)
    while failures < 2:
        check_limits(study, state)
        verify_study(study)
        output = base / f'attempt{len(list(base.glob("attempt*")))+1:02d}'
        output.mkdir()
        with Job() as job, (output / 'worker.log').open('a', encoding='utf-8') as log:
            process = launch([sys.executable, str(ROOT / 'experiment/train_broad_live.py'), '_assess',
                              '--study', str(study), '--output', str(output), '--job-name', job.name], ROOT, log)
            job.assign(process)
            started = time.time()
            try:
                while process.poll() is None:
                    check_limits(study, state)
                    if time.time()-started > 3600:
                        raise TimeoutError('Final broad assessment exceeded one hour')
                    pulse(study, state, current=dict(phase='final_assessment', worker_pid=process.pid,
                                                    run_directory=relative(output), started_at=started))
                    time.sleep(3)
            except Halt as stop:
                atomic(output / 'interrupted.json', dict(status=stop.status, time=time.time()))
                raise
            finally:
                terminate(process)
        if (output / 'result.json').exists():
            verify_result(output / 'result.json')
            return output / 'result.json'
        if not (output / 'failure.json').exists():
            atomic(output / 'failure.json', dict(status='failed', exit_code=process.returncode))
        failures += 1
    raise RuntimeError('Final assessment failed twice; inspect preserved logs')


def execute(study, ready=None):
    study = Path(study).resolve()
    if ready:
        deadline = time.time()+20
        while not Path(ready).exists():
            if time.time() > deadline:
                raise RuntimeError('Supervisor failed to assign process ownership')
            time.sleep(.1)
    with exclusive(study / 'controller.lock'):
        settings, _ = verify_study(study)
        state = read(study / 'state.json') if (study / 'state.json').exists() else dict(
            started_at=time.time(), deadline=time.time()+settings['hours']*3600)
        if state.get('status') in ('completed', 'budget_exhausted', 'needs_attention', 'insufficient_disk_space'):
            return
        try:
            pulse(study, state, status='running', phase='branch_search', current=None)
            optimize(study, state, BRANCHES, settings['trials_per_branch'])
            selected = read(study / 'selected.json') if (study / 'selected.json').exists() else dict(branches=freeze(study, BRANCHES))
            atomic(study / 'selected.json', selected)
            pulse(study, state, phase='hybrid_search')
            optimize(study, state, HYBRIDS, settings['trials_per_hybrid'])
            if 'hybrids' not in selected:
                selected['hybrids'] = freeze(study, HYBRIDS)
                atomic(study / 'selected.json', selected)
            pulse(study, state, phase='seed_repeats')
            for seed in settings['repeat_seeds']:
                for family, entry in {**selected['branches'], **selected['hybrids']}.items():
                    base = study / f'repeats/{family}/seed{seed}'
                    if (base / 'result-link.json').exists():
                        verify_result(base / 'result-link.json')
                        continue
                    config = entry['config']
                    if family in HYBRIDS:
                        config = dict(frozen_combiner=read(ROOT / entry['result'])['checkpoint'], runtime_seconds=300)
                    request = make_request(settings, study, family, 'repeat', config, seed)
                    result, _ = run_job(study, state, request, base)
                    if result is None:
                        raise RuntimeError(f'Repeated fit failed for {family} seed{seed}')
                    atomic(base / 'result-link.json', result)
                    state.setdefault('completed_repeats', []).append(f'{family}:{seed}')
                    pulse(study, state)
            pulse(study, state, phase='final_assessment', current=None)
            result_path = run_assessment(study, state)
            atomic(study / 'completion.json', dict(status='completed', finished_at=time.time(),
                branch_trials=counts(study, BRANCHES), hybrid_trials=counts(study, HYBRIDS),
                outputs=10, auxiliary_branches=1, repeat_seeds=settings['repeat_seeds'],
                feature_profile=settings['feature_profile'], assessment_result=relative(result_path),
                assessment_sha256=digest(result_path), test_loaded=False, live_accuracy_verified=False,
                native_integration='pending; existing installed models unchanged'))
            pulse(study, state, status='completed', phase='complete', current=None, worker_progress=None)
        except Halt as stop:
            pulse(study, state, status=stop.status, current=None)
        except BaseException as error:
            pulse(study, state, status='needs_attention', error=str(error), traceback=traceback.format_exc(), current=None)
            raise
