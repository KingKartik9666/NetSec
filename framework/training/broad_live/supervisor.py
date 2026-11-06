"""Hidden watchdog for the broad CLI; owned jobs survive chat termination."""
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import psutil

from framework.training.live_study.protocol import ROOT, atomic, read, exclusive, digest
from framework.training.live_study.processes import Job, launch, terminate
from framework.training.live_study.supervisor import TERMINAL, remove_startup

CLI = ROOT / 'experiment/train_broad_live.py'


def install_startup(study):
    if os.name != 'nt':
        raise RuntimeError('Logon recovery is Windows-only')
    study = Path(study).resolve()
    identity = read(study / 'study.json')['protocol_fingerprint'][:12]
    directory = Path(os.environ['APPDATA']) / 'Microsoft/Windows/Start Menu/Programs/Startup'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'NetSec-live-study-{identity}.vbs'
    command = subprocess.list2cmdline([sys.executable, str(CLI), 'supervise', '--study', str(study)])
    quote = lambda value: '"' + str(value).replace('"', '""') + '"'
    text = 'Set shell = CreateObject("WScript.Shell")\n' + 'shell.CurrentDirectory = ' + quote(ROOT) + '\n'
    text += 'shell.Run ' + quote(command) + ', 0, False\n'
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise FileExistsError('Unrelated Startup entry will not be overwritten')
    path.write_text(text, encoding='utf-8')
    atomic(study / 'startup.json', dict(path=str(path), sha256=digest(path), kind='hidden broad study logon recovery'))
    return str(path)


def supervise(study):
    study = Path(study).resolve()
    lock = exclusive(study / 'supervisor.lock')
    try:
        lock.__enter__()
    except OSError:
        return
    restarts = 0
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        while not (study / 'STOP').exists():
            state = read(study / 'state.json') if (study / 'state.json').exists() else {}
            if state.get('status') in TERMINAL:
                if state['status'] == 'completed':
                    remove_startup(study)
                return
            ready = study / f'.job-ready-{uuid.uuid4().hex}'
            with Job() as job, (study / 'controller.log').open('a', encoding='utf-8') as log:
                process = launch([sys.executable, str(CLI), '_controller', '--study', str(study),
                                  '--ready', str(ready), '--job-name', job.name], ROOT, log)
                try:
                    job.assign(process)
                    atomic(ready, dict(pid=process.pid, created=psutil.Process(process.pid).create_time()))
                    started = time.time()
                    while process.poll() is None:
                        atomic(study / 'supervisor.json', dict(status='watching', pid=os.getpid(),
                            created=psutil.Process().create_time(), controller_pid=process.pid,
                            heartbeat=time.time(), controller_restarts=restarts))
                        state = read(study / 'state.json') if (study / 'state.json').exists() else {}
                        if time.time()-max(started, state.get('heartbeat', started)) > 240:
                            terminate(process)
                            break
                        time.sleep(5)
                finally:
                    terminate(process)
                    ready.unlink(missing_ok=True)
            state = read(study / 'state.json') if (study / 'state.json').exists() else {}
            if state.get('status') in TERMINAL:
                continue
            restarts += 1
            if restarts >= 3:
                state.update(status='needs_attention', error='Three unexpected controller exits; inspect controller.log', heartbeat=time.time())
                atomic(study / 'state.json', state)
                return
            for _ in range(15):
                if (study / 'STOP').exists():
                    return
                time.sleep(2)
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        atomic(study / 'supervisor.json', dict(status='exited', pid=os.getpid(), heartbeat=time.time(), controller_restarts=restarts))
        lock.__exit__(None, None, None)
