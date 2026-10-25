"""Detached, bounded CUDA optimization; fixed RF is never optimized.

Usage: python -m experiment.optimize_models start|status|stop|resume
Results are source-validation evidence, never labelled live-PC accuracy.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/training/optimization_24h_v1'
BASE = ROOT / 'artifacts/training/four_models_v2'
POLICY_VERSION = 'architecture_v3'


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verified(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('status') != 'completed' or 'checkpoint_path' not in data:
        return None
    checkpoint = ROOT / data['checkpoint_path']
    if not checkpoint.is_file() or digest(checkpoint) != data['checkpoint_sha256']:
        raise ValueError(f'Checkpoint integrity failed: {path}; preserve evidence')
    assets_path = path.parent / 'integrity.json'
    if assets_path.exists():
        for name, expected in json.loads(assets_path.read_text()).items():
            if not (path.parent / name).is_file() or digest(path.parent / name) != expected:
                raise ValueError(f'Inference asset integrity failed: {name}')
    return data


def seal_assets(directory: Path) -> None:
    assets = {p.name: digest(p) for p in directory.iterdir()
              if p.suffix in ('.pt', '.ubj', '.joblib', '.npz')}
    write(directory / 'integrity.json', assets)


def pin_inputs() -> dict:
    """Pin actual train/validation inputs and source, never the locked test."""
    paths = []
    for folder in ('datasense_full_train', 'datasense_full_protocols'):
        parent = ROOT / 'dataset/prepared' / folder
        paths += [parent / 'manifest.json']
        paths += [parent / f'{kind}_{split}.npy' for kind in ('X', 'y', 'row_ids')
                  for split in ('train', 'validation')]
    paths += [ROOT / name for name in ('experiment/train_four_models.py',
              'experiment/train_partial_xgboost.py', 'framework/training/neural.py',
              'framework/training/fixed_random_forest.py', 'framework/training/partial_objectives.py')]
    for folder in ('external/HEM', 'external/efficient-kan'):
        paths += list((ROOT / folder).rglob('*.py'))
    return {str(p.relative_to(ROOT)): digest(p) for p in paths if p.is_file()}


def protect_descendants() -> int | None:
    """Kill all owned descendants if this Windows controller dies or crashes."""
    if os.name != 'nt':
        return None
    from ctypes import wintypes
    class BASIC(ctypes.Structure):
        _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                    ('flags', wintypes.DWORD), ('minimum', ctypes.c_size_t),
                    ('maximum', ctypes.c_size_t), ('active', wintypes.DWORD),
                    ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD),
                    ('scheduling', wintypes.DWORD)]
    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]
    class EXTENDED(ctypes.Structure):
        _fields_ = [('basic', BASIC), ('io', IO), ('process_memory', ctypes.c_size_t),
                    ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t),
                    ('peak_job', ctypes.c_size_t)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = kernel.CreateJobObjectW(None, None)
    limits = EXTENDED()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not job or not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())
    # Deliberately retain the non-inheritable handle until process termination.
    # Kernel closes it on death and terminates every remaining fit descendant.
    return job


class StudyLock:
    """OS lock automatically released on process death, including reboot."""
    def __init__(self, path: Path):
        self.path = path
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a+b')
        try:
            if self.path.stat().st_size == 0:
                self.stream.write(b'0')
                self.stream.flush()
            self.stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise RuntimeError('Another study controller holds the lock') from exc
        return self

    def __exit__(self, *_):
        if os.name == 'nt':
            import msvcrt
            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.stream, fcntl.LOCK_UN)
        self.stream.close()


def remaining(state: dict) -> float:
    return max(0., state['deadline'] - time.time())


def ranking(result: dict) -> tuple[float, float]:
    metrics = (result['profile_metrics']['packet_counts_7']
               if result['model_id'] == 'xgboost' else result['metrics'])
    return ((metrics['accuracy'], metrics['macro_f1']) if result['model_id'] == 'xgboost'
            else (metrics['macro_f1'], metrics['accuracy']))


def search_configs(originals: dict, allow_architecture: bool = True) -> list[dict]:
    """Round-robin independent ablations; combinations follow completed evidence."""
    changes = {
        'xgboost': [('learning_rate', .04), ('learning_rate', .12), ('reg_lambda', 6.),
                    ('reg_alpha', .2), ('subsample', 1.), ('colsample_bynode', 1.),
                    ('min_child_weight', 1.), ('weight_strategy', 'balanced'),
                    ('weight_strategy', 'none'), ('smoothing', .1)],
        'kan': [('lr', .001), ('lr', .003), ('weight_decay', .0001),
                ('weight_decay', .01), ('spline_regularization', .0001),
                ('spline_regularization', .001), ('weight_strategy', 'balanced'),
                ('weight_strategy', 'none'), ('epochs', 24), ('batch_size', 1024)],
        'hem': [('lr', .0005), ('lr', .002), ('weight_decay', .001),
                ('weight_decay', .01), ('weight_strategy', 'sqrt_balanced'),
                ('weight_strategy', 'balanced'), ('epochs', 16), ('batch_size', 1024)],
    }
    trials = []
    for index in range(max(map(len, changes.values()))):
        for model in ('xgboost', 'kan', 'hem'):
            if index >= len(changes[model]):
                continue
            key, value = changes[model][index]
            config = copy.deepcopy(originals[model])
            config[key] = value
            trials.append(dict(model_id=model, variant=f'ablation_{index:02d}_{key}',
                               config=config, seed=42))
    if not allow_architecture:
        return trials
    capacity = {
        'xgboost': [('max_depth', 4), ('max_depth', 8), ('max_depth', 10),
                    ('max_depth', 12), ('num_boost_round', 800), ('num_boost_round', 1200)],
        'kan': [('hidden', [128, 64]), ('hidden', [192, 96]),
                ('hidden', [128, 64, 32]), ('grid_size', 7), ('grid_size', 9)],
        'hem': [('hidden', [768, 384, 128, 64]), ('binary_hidden', [512, 128, 32]),
                ('pca_components', 24), ('pca_components', 32),
                ('knn_k', 3), ('knn_k', 5)],
    }
    architecture = []
    for index in range(max(map(len, capacity.values()))):
        for model in ('xgboost', 'kan', 'hem'):
            if index >= len(capacity[model]):
                continue
            key, value = capacity[model][index]
            config = copy.deepcopy(originals[model])
            config[key] = value
            architecture.append(dict(model_id=model, variant=f'capacity_v3_{index:02d}_{key}',
                                     config=config, seed=42, architecture_modified=True,
                                     research_policy=POLICY_VERSION))
    # Give capacity changes an early turn within the same persisted budget.
    mixed = []
    for index in range(max(len(trials), len(architecture))):
        if index < len(architecture):
            mixed.append(architecture[index])
        if index < len(trials):
            mixed.append(trials[index])
    return mixed


def terminate_tree(child: subprocess.Popen) -> None:
    """Only terminate the owned trial, including Windows venv redirector children."""
    if child.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        import signal
        os.killpg(child.pid, signal.SIGKILL)
    child.wait(timeout=20)


def run_fit(request: dict, directory: Path, state: dict, *, study: Path | None = None,
            worker_module: str = 'experiment.fit_optimization', on_update=None) -> dict | None:
    """Retries use new attempt directories, retaining interrupted trial evidence."""
    study = STUDY if study is None else study
    notify = handoff if on_update is None else on_update
    for candidate in sorted(directory.glob('attempt_*/result.json')):
        result = verified(candidate)
        if result:
            if any(result.get(key) != value for key, value in request.items()):
                raise ValueError('Completed attempt differs from immutable request')
            return result
    if remaining(state) <= 0 or (study / 'STOP').exists():
        return None
    attempt = directory / f'attempt_{len(list(directory.glob("attempt_*"))):03d}'
    attempt.mkdir(parents=True)
    write(attempt / 'request.json', request)
    budget = min(1200., remaining(state))
    if state.get('phase') in ('independent_ablation_search', 'justified_combinations'):
        budget = min(budget, max(0., state['search_deadline'] - time.time()))
    if budget <= 0:
        return None
    deadline = time.time() + budget
    write(attempt / 'execution.json', {'deadline': deadline, 'budget_seconds': budget})
    with (attempt / 'fit.log').open('a', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, '-B', '-u', '-m',
                                  worker_module, '--fit', str(attempt / 'request.json')],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 env={**os.environ, 'IMDAI_FIT_DEADLINE': str(deadline)},
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                                 start_new_session=os.name != 'nt')
        state['active_trial'] = {'request': request, 'directory': str(attempt.relative_to(ROOT)),
                                 'pid': child.pid, 'started': time.time(), 'budget_seconds': budget}
        write(study / 'state.json', state)
        notify(state)
        started = time.monotonic()
        reason = None
        while child.poll() is None:
            if (study / 'STOP').exists() or remaining(state) <= 0:
                reason = 'stop requested or persisted study deadline reached'
            elif time.monotonic() - started >= budget:
                reason = '20-minute per-fit budget exceeded'
            if reason:
                terminate_tree(child)
                break
            time.sleep(1)
        if reason or child.returncode:
            write(attempt / 'failure.json', {'reason': reason or 'fit failed; inspect log',
                                           'exit_code': child.returncode, 'test_loaded': False})
    state['active_trial'] = None
    write(study / 'state.json', state)
    result = verified(attempt / 'result.json')
    if result:
        seal_assets(attempt)
    notify(state)
    return result


def request_for(trial: dict, definition: dict) -> dict:
    prepared = ROOT / ('dataset/prepared/datasense_full_protocols'
                       if trial['model_id'] == 'xgboost' else 'dataset/prepared/datasense_full_train')
    return {**trial, 'prepared': prepared.relative_to(ROOT).as_posix(),
            'prepared_manifest_sha256': digest(prepared / 'manifest.json'),
            'protocol_id': definition['protocol_id']}


def handoff(state: dict) -> None:
    lines = ['# Resumable optimization handoff', '',
             'Exactly four families: fixed conventional RF, XGBoost, genuine KAN and GitHub HEM.',
             'RF remains fixed. XGBoost/KAN/HEM capacity changes now authorized; original controls preserved.',
             'Modified HEM architectures are labelled HEM-derived variants. CUDA research; CPU app; test locked.',
             'Target: 95%+ accuracy, not a promised outcome. Full and seven-feature metrics remain separate.',
             f"State: {state.get('status')}; phase: {state.get('phase')}; controller PID: {state.get('pid')}",
             f"Started UTC: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(state['started']))}",
             f"Deadline UTC: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(state['deadline']))}",
             f"Active trial: {json.dumps(state.get('active_trial'))}", '',
             'Run from the project root:', '```powershell',
             '.venv/Scripts/python.exe -B -m experiment.optimize_models status',
             '.venv/Scripts/python.exe -B -m experiment.optimize_models resume',
             '.venv/Scripts/python.exe -B -m experiment.optimize_models stop', '```', '',
             'Resume never resets the deadline. Results and logs: artifacts/training/optimization_24h_v1/.',
             'Original study: artifacts/training/four_models_v2/. Check hashes before using results.',
             'After training: audit seed repeats, explicitly promote validated models, verify CPU app reload,',
             'run demos and full evaluation suite, then separately rebuild/install desktop.',
             'No automatic installed-app update or model promotion. Native HEM unknown-stage reproduction',
             'remains distinct from this closed-set DataSense adaptation.', '',
             'Deferred next stage: four Npcap/live versions using selected CSV architectures',
             '(eight deployments, four families). Do not start until CSV training',
             'is complete and the agreed 95% validation gate passes. Ground-truth lab/PCAP labels',
             'are required; Npcap and classifier predictions do not establish labels.',
             'Live gate: ALL FOUR CSV models individually need >=95% source-validation accuracy.',
             'RF stays fixed. Comparing unlabelled Npcap traffic to CIC-IIoT-2025 is compatibility',
             'analysis, not ground truth. Independent compatible labelled lab captures/PCAP datasets',
             'must supply labels; never self-label from predictions. Live stage remains blocked.']
    lines += ['', 'Recorded completed trial summaries (checkpoint integrity is checked on resume):']
    for folder in (BASE, STUDY):
        for result_path in sorted(folder.rglob('result.json')):
            try:
                result = json.loads(result_path.read_text())
                if result.get('status') == 'completed' and 'metrics' in result:
                    metrics = result['metrics']
                    live = result.get('profile_metrics', {}).get('packet_counts_7', {})
                    lines.append(f"- {result_path.relative_to(ROOT)}: full accuracy "
                                 f"{metrics['accuracy']:.6f}, macro F1 {metrics['macro_f1']:.6f}; "
                                 f"seven-feature accuracy {live.get('accuracy', 'not supported')}")
            except (ValueError, KeyError):
                continue
    (ROOT / 'docs/RESUME_CONTEXT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def register_recovery(module_name: str = 'experiment.optimize_models', *, recovery_command: str = 'resume') -> dict:
    """Current-user logon recovery; no admin, credentials or visible console."""
    if os.name != 'nt':
        return {'registered': False, 'reason': 'Windows logon recovery only'}
    startup = Path(os.environ['APPDATA']) / 'Microsoft/Windows/Start Menu/Programs/Startup'
    startup.mkdir(parents=True, exist_ok=True)
    script = startup / 'IMDAI-Training-Resume.vbs'
    pythonw = Path(sys.executable).with_name('pythonw.exe')
    if recovery_command not in ('resume', 'recover'):
        raise ValueError('Unsupported training recovery command')
    command = subprocess.list2cmdline([str(pythonw), '-B', '-m', module_name, recovery_command])
    script.write_text('Set shell = CreateObject("WScript.Shell")\n'
                      + 'shell.CurrentDirectory = "' + str(ROOT).replace('"', '""') + '"\n'
                      + 'shell.Run "' + command.replace('"', '""') + '", 0, False\n', encoding='utf-8')
    return {'registered': True, 'path': str(script), 'trigger': 'current-user logon'}


def controller() -> None:
    with StudyLock(STUDY / 'controller.lock'):
        owned_job = protect_descendants()
        state = json.loads((STUDY / 'state.json').read_text())
        if remaining(state) <= 0:
            state.update(status='deadline_reached', active_trial=None)
            write(STUDY / 'state.json', state)
            handoff(state)
            return
        state.update(pid=os.getpid(), status='running')
        write(STUDY / 'state.json', state)
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            pinned = pin_inputs()
            pinned_path = STUDY / 'input_source_integrity.json'
            if pinned_path.exists() and json.loads(pinned_path.read_text()) != pinned:
                raise ValueError('Training input/source hashes changed; incompatible resume')
            if not pinned_path.exists():
                write(pinned_path, pinned)
            definition = json.loads((BASE / 'study.json').read_text())
            definition_hash = digest(BASE / 'study.json')
            if state.get('base_study_sha256', definition_hash) != definition_hash:
                raise ValueError('Original study definition changed; preserve and investigate')
            state['base_study_sha256'] = definition_hash
            from experiment.train_four_models import verify_common_partitions
            identity = verify_common_partitions(ROOT / 'dataset/prepared/datasense_full_train',
                                                ROOT / 'dataset/prepared/datasense_full_protocols')
            if identity != definition['identity']:
                raise ValueError('Training/validation identity changed')
            originals = {t['model_id']: t['config'] for t in definition['configurations']
                         if t['variant'] == 'control'}
            state['phase'] = 'complete_existing_study'
            for trial in definition['configurations']:
                base_result = BASE / 'runs' / f"{trial['model_id']}_{trial['variant']}_s42" / 'result.json'
                if verified(base_result):
                    seal_assets(base_result.parent)
                    continue
                completed = run_fit(request_for({**trial, 'seed': 42}, definition),
                                    STUDY / 'prerequisites' / f"{trial['model_id']}_{trial['variant']}", state)
                if not completed:
                    state['phase'] = 'prerequisites_incomplete'
                    raise RuntimeError(f"Prerequisite incomplete: {trial['model_id']} {trial['variant']}")
            controls = {}
            for model in originals:
                controls[model] = verified(BASE / 'runs' / f'{model}_control_s42/result.json')
                if controls[model] is None:
                    for path in sorted((STUDY / 'prerequisites' / f'{model}_control').glob('attempt_*/result.json')):
                        controls[model] = verified(path)
                        if controls[model]:
                            break
                if controls[model] is None:
                    raise RuntimeError(f'Missing valid matched control: {model}')
            trials = search_configs(originals)
            policy = {'version': POLICY_VERSION, 'rf_fixed': True, 'configurations': trials,
                      'fit_wrapper_sha256': digest(ROOT / 'experiment/fit_optimization.py'),
                      'controller_sha256': digest(Path(__file__)), 'test_loaded': False}
            policy_path = STUDY / f'policies/{POLICY_VERSION}.json'
            if policy_path.exists() and json.loads(policy_path.read_text()) != policy:
                raise ValueError('Architecture policy changed; create a separately versioned policy')
            if not policy_path.exists():
                write(policy_path, policy)
            state['research_policy'] = policy['version']
            state['policy_upgrade_pending'] = False
            candidates = {model: [result] for model, result in controls.items()}
            for model in originals:
                enhanced = verified(BASE / 'runs' / f'{model}_enhanced_s42/result.json')
                if enhanced is None:
                    for path in sorted((STUDY / 'prerequisites' / f'{model}_enhanced').glob('attempt_*/result.json')):
                        enhanced = verified(path)
                        if enhanced:
                            break
                if enhanced:
                    candidates[model].append(enhanced)
            state['phase'] = 'independent_ablation_search'
            for trial in trials:
                if time.time() >= state['search_deadline'] or (STUDY / 'STOP').exists():
                    break
                result = run_fit(request_for(trial, definition),
                                 STUDY / 'runs' / f"{trial['model_id']}_{trial['variant']}_s42", state)
                if result:
                    candidates[trial['model_id']].append(result)
            # Combine only independently improving parameter changes, with an explicit record.
            state['phase'] = 'justified_combinations'
            for model, results in candidates.items():
                if time.time() >= state['search_deadline'] or (STUDY / 'STOP').exists():
                    break
                improved = sorted([r for r in results if ranking(r) > ranking(controls[model])],
                                  key=ranking, reverse=True)
                config = copy.deepcopy(originals[model])
                changes = {}
                for result in improved:
                    for key, value in result['config'].items():
                        if value != originals[model].get(key) and key not in changes:
                            changes[key] = value
                if len(changes) < 2:
                    continue
                config.update(changes)
                trial = dict(model_id=model, variant='supported_combination', config=config, seed=42)
                architecture_keys = {'max_depth', 'num_boost_round', 'hidden', 'binary_hidden',
                                     'pca_components', 'knn_k', 'grid_size', 'spline_order'}
                trial['architecture_modified'] = bool(architecture_keys.intersection(changes))
                trial['research_policy'] = POLICY_VERSION
                result = run_fit(request_for(trial, definition), STUDY / 'runs' / f'{model}_combination_v3_s42', state)
                if result:
                    results.append(result)
            state['phase'] = 'repeat_leaders_and_controls'
            report = {'test_loaded': False, 'target_accuracy': .95, 'models': {}}
            for model, results in candidates.items():
                leader = max(results, key=ranking)
                repeats = {'control': [], 'candidate': []}
                for seed in (43, 44):
                    for variant, selected in [('control', controls[model]), ('candidate', leader)]:
                        trial = dict(model_id=model, variant=f'repeat_{variant}', config=selected['config'], seed=seed)
                        trial['architecture_modified'] = selected.get('architecture_modified', False)
                        trial['research_policy'] = POLICY_VERSION
                        result = run_fit(request_for(trial, definition),
                                         STUDY / 'repeats' / f'{model}_{variant}_v3_s{seed}', state)
                        if result:
                            repeats[variant].append(result)
                repeat_improved = (len(repeats['control']) == len(repeats['candidate']) == 2
                                   and all(ranking(b) > ranking(a) for a, b in
                                           zip(repeats['control'], repeats['candidate'])))
                report['models'][model] = {'control': controls[model], 'leader': leader,
                                          'repeats': repeats, 'promotion_eligible':
                                          ranking(leader) > ranking(controls[model]) and repeat_improved,
                                          'promoted': False}
                write(STUDY / 'report.json', report)
            state['live_stage'] = {'status': 'blocked', 'minimum_full_validation_accuracy': .95,
                                   'requires_all_four': True, 'requires_independent_labels': True,
                                   'reason': 'Await CSV acceptance and independently labelled compatible captures'}
            state.update(status='stopped' if (STUDY / 'STOP').exists() else
                         ('deadline_reached' if remaining(state) <= 0 else 'completed'), active_trial=None)
        except Exception as exc:
            state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if os.name == 'nt':
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            write(STUDY / 'state.json', state)
            handoff(state)


def launch() -> None:
    # Briefly acquire the same OS lock: a duplicate launch never clears STOP.
    with StudyLock(STUDY / 'controller.lock'):
        state = json.loads((STUDY / 'state.json').read_text())
        if remaining(state) <= 0 or state.get('status') == 'completed':
            print('Study is finished or expired; deadline will not reset.')
            return
        (STUDY / 'STOP').unlink(missing_ok=True)
    with (STUDY / 'controller.log').open('a', encoding='utf-8') as log:
        flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                 | subprocess.CREATE_NO_WINDOW) if os.name == 'nt' else 0
        child = subprocess.Popen([sys.executable, '-B', '-u', '-m', 'experiment.optimize_models', '_run'],
                                 cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 close_fds=True, creationflags=flags, start_new_session=os.name != 'nt')
    print(json.dumps({'launched_pid': child.pid, 'state': str(STUDY / 'state.json')}))


def refresh_after_fit() -> None:
    """Activate new controller code at a completed fit boundary, retaining deadline.

    A detached watcher holds no training Job Object; its launched controller
    creates its own. The active fit completes before STOP is requested.
    """
    with StudyLock(STUDY / 'refresh.lock'):
        transition = json.loads((STUDY / 'refresh_request.json').read_text())
        target = transition.get('active_directory')
        target_pid = transition.get('active_pid')
        while True:
            state = json.loads((STUDY / 'state.json').read_text())
            if remaining(state) <= 0:
                write(STUDY / 'refresh_result.json', {'status': 'expired', 'deadline_reset': False})
                return
            active = state.get('active_trial')
            changed = not active or active.get('directory') != target
            # A saved result may precede the wrapper's final metadata write.
            # Wait for actual worker exit, not only checkpoint availability.
            complete = bool(target and verified(ROOT / target / 'result.json')
                            and not trial_process_alive(target_pid, target))
            # Failures also permit transition, but never manufacture completion.
            terminal = state.get('status') in ('failed', 'stopped', 'completed')
            if changed or complete or terminal:
                (STUDY / 'STOP').touch()
                break
            time.sleep(.2)
        while remaining(state) > 0:
            try:
                with StudyLock(STUDY / 'controller.lock'):
                    state = json.loads((STUDY / 'state.json').read_text())
                    write(STUDY / 'policy_transition.json', {'previous_state': state,
                         'new_policy': POLICY_VERSION, 'deadline_reset': False,
                         'original_deadline': state['deadline']})
                    # An explicitly requested policy upgrade may continue a
                    # completed finite search, within the original deadline.
                    state.update(status='stopped', policy_upgrade_pending=True)
                    write(STUDY / 'state.json', state)
                launch()
                write(STUDY / 'refresh_result.json', {'status': 'activated', 'deadline_reset': False})
                return
            except RuntimeError as exc:
                if 'holds the lock' not in str(exc):
                    raise
                time.sleep(.2)
            state = json.loads((STUDY / 'state.json').read_text())


def queue_refresh() -> None:
    state = json.loads((STUDY / 'state.json').read_text())
    with StudyLock(STUDY / 'refresh.lock'):
        write(STUDY / 'refresh_request.json', {'active_directory':
              (state.get('active_trial') or {}).get('directory'),
              'active_pid': (state.get('active_trial') or {}).get('pid'),
              'requested': time.time(), 'policy': POLICY_VERSION,
              'deadline': state['deadline']})
    with (STUDY / 'refresh.log').open('a', encoding='utf-8') as log:
        flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                 | subprocess.CREATE_NO_WINDOW) if os.name == 'nt' else 0
        child = subprocess.Popen([sys.executable, '-B', '-u', '-m', 'experiment.optimize_models', '_refresh'],
                                 cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 close_fds=True, creationflags=flags, start_new_session=os.name != 'nt')
    print(json.dumps({'queued_refresh_pid': child.pid, 'after_current_fit': True, 'deadline_reset': False}))


def trial_process_alive(pid: int | None, directory: str) -> bool:
    """Read-only ownership check; a recycled PID is not the old trial."""
    if pid is None:
        return False
    import psutil
    try:
        process = psutil.Process(pid)
        command = process.cmdline()
        expected = (ROOT / directory / 'request.json').resolve()
        return any(Path(arg).resolve() == expected for arg in command if 'request.json' in arg)
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True  # Defer transition if ownership cannot be checked.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['start', 'status', 'stop', 'resume', 'refresh', '_run', '_refresh'])
    args = parser.parse_args()
    if args.command == 'start':
        STUDY.mkdir(parents=True, exist_ok=True)
        if not (STUDY / 'state.json').exists():
            now = time.time()
            write(STUDY / 'state.json', {'started': now, 'deadline': now + 24 * 3600,
                                       'search_deadline': now + 18 * 3600, 'status': 'created',
                                       'target_accuracy': .95, 'test_loaded': False})
        write(STUDY / 'recovery.json', register_recovery())
        launch()
    elif args.command == 'resume':
        if (STUDY / 'state.json').exists():
            launch()
    elif args.command == 'stop':
        STUDY.mkdir(parents=True, exist_ok=True)
        (STUDY / 'STOP').touch()
        print('Stop requested; owned active trial will be terminated.')
    elif args.command == 'status':
        print((STUDY / 'state.json').read_text() if (STUDY / 'state.json').exists() else 'Not started')
    elif args.command == 'refresh':
        queue_refresh()
    elif args.command == '_refresh':
        refresh_after_fit()
    else:
        controller()


if __name__ == '__main__':
    main()
