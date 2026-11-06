"""Own only this study's child process tree; never stop unrelated Python jobs."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import subprocess
import uuid


class Job:
    """Windows closes the entire owned process tree if the supervisor crashes."""
    def __init__(self):
        self.handle = None
        self.name = 'Local\\NetSecLive-'+uuid.uuid4().hex
        if os.name != 'nt': return
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        class Basic(ctypes.Structure):
            _fields_ = [('PerProcessUserTimeLimit',ctypes.c_longlong),('PerJobUserTimeLimit',ctypes.c_longlong),
                ('LimitFlags',wintypes.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),
                ('MaximumWorkingSetSize',ctypes.c_size_t),('ActiveProcessLimit',wintypes.DWORD),
                ('Affinity',ctypes.c_size_t),('PriorityClass',wintypes.DWORD),('SchedulingClass',wintypes.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name,ctypes.c_ulonglong) for name in
                ('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
        class Extended(ctypes.Structure):
            _fields_ = [('BasicLimitInformation',Basic),('IoInfo',IO),('ProcessMemoryLimit',ctypes.c_size_t),
                ('JobMemoryLimit',ctypes.c_size_t),('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]
        kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR];kernel.CreateJobObjectW.restype=wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
        kernel.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        self.kernel=kernel;self.handle=kernel.CreateJobObjectW(None,self.name)
        if not self.handle: raise ctypes.WinError(ctypes.get_last_error())
        info=Extended();info.BasicLimitInformation.LimitFlags=0x2000  # KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(self.handle,9,ctypes.byref(info),ctypes.sizeof(info)):
            self.close();raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, process):
        if self.handle and not self.kernel.AssignProcessToJobObject(self.handle,int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle);self.handle=None

    def __enter__(self): return self
    def __exit__(self,*args): self.close()


def join_current(name):
    """The real interpreter joins before work, even behind a venv redirector.

    The controller/watchdog retains the only long-lived job handle. A worker
    cannot keep its own job alive after its owner exits.
    """
    if os.name!='nt':return
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenJobObjectW.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.LPCWSTR]
    kernel.OpenJobObjectW.restype=wintypes.HANDLE
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.IsProcessInJob.argtypes=[wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)]
    kernel.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenJobObjectW(0x0001|0x0004,False,name)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    try:
        current=kernel.GetCurrentProcess();member=wintypes.BOOL()
        if not kernel.IsProcessInJob(current,handle,ctypes.byref(member)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not member.value and not kernel.AssignProcessToJobObject(handle,current):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:kernel.CloseHandle(handle)


def launch(command, cwd, log):
    env=dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
             OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return subprocess.Popen(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
        env=env,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)


def terminate(process):
    if process.poll() is None:
        # A Windows venv python.exe can be a redirector with a real interpreter
        # child. Killing only the redirector would leave an old GPU fit alive.
        import psutil
        try:children=psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:children=[]
        for child in reversed(children):
            try:child.terminate()
            except psutil.NoSuchProcess:pass
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.kill();process.wait(timeout=10)
        _,alive=psutil.wait_procs(children,timeout=5)
        for child in alive:
            try:child.kill()
            except psutil.NoSuchProcess:pass
        psutil.wait_procs(alive,timeout=5)
