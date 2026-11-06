"""Detached watchdog, bounded controller recovery and optional logon resume."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys
import time
import uuid

from .protocol import ROOT, atomic, read, exclusive, digest
from .processes import Job, launch, terminate

TERMINAL = {'completed','budget_exhausted','stopped','needs_attention','insufficient_disk_space'}


def install_startup(study):
    if os.name != 'nt':raise RuntimeError('Logon resume is Windows-only')
    study=Path(study).resolve()
    identity=read(study/'study.json')['protocol_fingerprint'][:12]
    startup=Path(os.environ['APPDATA'])/'Microsoft/Windows/Start Menu/Programs/Startup'
    startup.mkdir(parents=True,exist_ok=True)
    path=startup/f'NetSec-live-study-{identity}.vbs'
    import subprocess
    command=subprocess.list2cmdline([sys.executable,str(ROOT/'experiment/train_live_models.py'),'supervise','--study',str(study)])
    def vb(value):return '"'+str(value).replace('"','""')+'"'
    content='Set shell = CreateObject("WScript.Shell")\n'+'shell.CurrentDirectory = '+vb(ROOT)+'\n'+'shell.Run '+vb(command)+', 0, False\n'
    if path.exists() and path.read_text(encoding='utf-8') != content:
        raise FileExistsError('Refusing to overwrite an unrelated Startup entry')
    path.write_text(content,encoding='utf-8')
    atomic(study/'startup.json',dict(path=str(path),sha256=digest(path),kind='current-user logon; hidden; obeys STOP and deadline'))
    return str(path)


def remove_startup(study):
    marker=Path(study)/'startup.json'
    if not marker.exists():return
    saved=read(marker);path=Path(saved['path'])
    expected=Path(os.environ['APPDATA'])/'Microsoft/Windows/Start Menu/Programs/Startup'
    if path.parent.resolve()!=expected.resolve() or not path.name.startswith('NetSec-live-study-'):
        raise ValueError('Startup path is outside this study scope')
    if path.exists() and digest(path)==saved['sha256']:path.unlink()


def _supervise(study):
    import psutil
    failures=0
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        while True:
            if (study/'STOP').exists():return
            state=read(study/'state.json') if (study/'state.json').exists() else {}
            if state.get('status') in TERMINAL:
                if state['status']=='completed':remove_startup(study)
                return
            ready=study/f'.job-ready-{uuid.uuid4().hex}'
            with Job() as job, (study/'controller.log').open('a',encoding='utf-8') as log:
                process=launch([sys.executable,str(ROOT/'experiment/train_live_models.py'),'_controller',
                    '--study',str(study),'--ready',str(ready),'--job-name',job.name],ROOT,log)
                try:
                    job.assign(process)
                    atomic(ready,dict(pid=process.pid,created=psutil.Process(process.pid).create_time()))
                    started=time.time()
                    while process.poll() is None:
                        atomic(study/'supervisor.json',dict(status='watching',pid=os.getpid(),
                            created=psutil.Process().create_time(),controller_pid=process.pid,
                            heartbeat=time.time(),controller_restarts=failures))
                        state=read(study/'state.json') if (study/'state.json').exists() else {}
                        if time.time()-max(started,state.get('heartbeat',started)) > 240:
                            terminate(process);break
                        time.sleep(5)
                finally:
                    terminate(process);ready.unlink(missing_ok=True)
            state=read(study/'state.json') if (study/'state.json').exists() else {}
            if state.get('status') in TERMINAL:continue
            failures+=1
            if failures>=3:
                state.update(status='needs_attention',error='Three unexpected controller exits; inspect controller.log',heartbeat=time.time())
                atomic(study/'state.json',state);return
            atomic(study/'supervisor.json',dict(status='restarting',pid=os.getpid(),heartbeat=time.time(),controller_restarts=failures))
            for _ in range(15):
                if (study/'STOP').exists():return
                time.sleep(2)
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        atomic(study/'supervisor.json',dict(status='exited',pid=os.getpid(),heartbeat=time.time(),controller_restarts=failures))


def supervise(study):
    study=Path(study).resolve()
    # Lock acquisition alone handles the harmless second-launch case. Errors in
    # the running supervisor are not silently swallowed as "already running".
    lock=exclusive(study/'supervisor.lock')
    try:lock.__enter__()
    except OSError:return
    try:_supervise(study)
    finally:lock.__exit__(None,None,None)
