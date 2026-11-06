"""Detached watchdog for upgrade round 2: runs U2 (fix77_ext) and then the live upgrades after fix77 stage 1 ends.

    python -m experiment.upgrades2_queue start     # spawns the watchdog detached (log: artifacts/training/upgrades2_queue/queue.log)
    python -m experiment.upgrades2_queue status
    python -m experiment.upgrades2_queue stop      # STOP in the queue folder and in the four upgrades2 study folders

Plan: ml_pipeline/notes_20261101/upgrades2_plan.md ("everything here queues after" the fix77 stage-1 chain).
The watchdog never touches the fix77 chain, its processes, STOP files or state files: it only reads
``chain_pid.json`` and the two ``state.json`` files. It waits until the chain process is gone and no training
controller or runner job runs; then it proceeds only if BOTH fix77 studies report phase 'done' (error, stopped, a
crashed 'running' or a missing state writes BLOCKED.md and exits). Then, strictly one at a time, for each of
[fix77_ext, upgrades2_live]: wait for >= 3600 MiB free GPU memory and no other controller, run the driver's smoke
(own smoke study), and only if the smoke ends in phase 'done' run the real driver; any failure writes BLOCKED.md with
the reason and stops the queue. This module sets no environment variable at import and is deliberately not a
controller marker (the fix77 drivers and the upgrades2 drivers must not see the waiting watchdog as a GPU job).

Operating rule: while the queue is armed, do NOT start any other training controller by hand. Since 2026-11-02 the
fix77 drivers list this round's drivers (``fix77.CONTROLLER_MARKERS``) and refuse to start while one is alive; the
live77, live77_upgrades, cascade and fs77 drivers have no such check, so they cannot see a driver that is between
jobs (and none of them sees the waiting queue). The upgrades2 drivers re-check before every training job
(``upgrades2.guard_run_job``) and wait, so a second GPU fit is never started from this side, but those older drivers
would not wait for them.

Restart and stop: ``start`` clears old STOP files (queue + the four upgrades2 studies); drivers launched by the
queue keep their STOP (``UPGRADES2_QUEUED=1``), so a ``stop`` issued while a driver starts is honoured. A driver
that honours STOP (exit 0, phase 'stopped') ends the queue as 'stopped', not BLOCKED. ``chain_alive`` matches the
recorded pid by create time against ``chain_pid.json['started']`` (pid reuse after a reboot is not the chain).
State files are written atomically; unreadable state/pid files block the queue with the reason.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from framework.training.csv_v3 import upgrades2 as U

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "artifacts/training/upgrades2_queue"
TRAINING = ROOT / "artifacts/training"
FIX77_STUDIES = {"fix77_tabm_ne": TRAINING / "fix77_tabm_ne_20261101_v1", "fix77_live": TRAINING / "fix77_live_20261101_v1"}
CHAIN_PID = FIX77_STUDIES["fix77_tabm_ne"] / "chain_pid.json"
CHAIN_MARKER = "experiment.train_fix77_tabm"
STEPS = (
    ("fix77_ext", "experiment.train_fix77_ext", TRAINING / "fix77_ext_smoke", TRAINING / "fix77_ext_20261101_v1"),
    ("upgrades2_live", "experiment.train_upgrades2_live", TRAINING / "upgrades2_live_smoke",
     TRAINING / "upgrades2_live_20261101_v1"),
)
MIN_FREE_MIB = 3600
POLL_SECONDS = 60
NOTE_EVERY = 1800                 # seconds between "still waiting" log lines
ENV_KEYS = ("CASCADE_STUDY", "CASCADE_SMOKE_STUDY", "CASCADE_DATA", "CASCADE_FEATURE_GROUPS", "CASCADE_SMOKE",
            "LIVE77_UPGRADE_STUDY")


# ------------------------------------------------------------------------------------------------ helpers
def log(message: str, queue: Path = QUEUE) -> None:
    queue.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    with open(queue / "queue.log", "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def set_state(queue: Path = QUEUE, **update) -> None:
    """Merge ``update`` into queue_state.json (atomic write; an unreadable old file is replaced, never fatal)."""
    path = queue / "queue_state.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(state, dict):
            raise ValueError("not an object")
    except (OSError, ValueError) as exc:
        state = {"previous_state_unreadable": repr(exc)}
    state.update(update, updated=time.strftime("%Y-%m-%d %H:%M:%S"), pid=os.getpid())
    U.write_json_atomic(path, state)


def blocked(reason: str, queue: Path = QUEUE) -> None:
    queue.mkdir(parents=True, exist_ok=True)
    (queue / "BLOCKED.md").write_text(
        f"# Upgrade round 2 queue blocked\n\n{time.strftime('%Y-%m-%d %H:%M:%S')}: {reason}\n\n"
        "Nothing after this point was started. Resolve the cause, delete this file, and run "
        "`python -m experiment.upgrades2_queue start` again (finished jobs are cached).\n", encoding="utf-8")
    log(f"BLOCKED: {reason}", queue)
    set_state(queue, phase="blocked", reason=reason)


def stopped(queue: Path = QUEUE) -> bool:
    return (queue / "STOP").exists()


START_TOLERANCE = 5.0             # seconds between the chain's Popen and its recorded 'started'


def chain_alive(pid_file: Path = CHAIN_PID, tolerance: float = START_TOLERANCE) -> bool:
    """True while the fix77 chain process recorded in chain_pid.json still runs.

    Pid reuse is checked by create time: the process is the chain only if it was created within ``tolerance`` s of
    the recorded 'started' and the machine has not rebooted since. A mismatch, a reboot, or a process whose create
    time cannot be read (not ours, so not the chain) is not alive; the fix77 phase checks still guard the real chain.
    An unreadable pid file raises ValueError (the watchdog turns it into BLOCKED.md)."""
    if not pid_file.exists():
        return False
    import psutil
    try:
        record = json.loads(pid_file.read_text(encoding="utf-8"))
        pid, started = int(record["pid"]), record.get("started")
        started = None if started is None else float(started)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot read {pid_file}: {exc!r}") from exc
    if started is not None and psutil.boot_time() > started:
        return False             # rebooted after the chain started: the recorded pid is not the chain
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return False
    if started is not None:
        try:
            created = proc.create_time()
        except psutil.NoSuchProcess:
            return False
        except psutil.AccessDenied:
            return False         # the chain is our own process; one we cannot inspect is a reused pid
        if abs(created - started) > tolerance:
            return False
    try:
        return CHAIN_MARKER in proc.cmdline()
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True                  # create time matched (it is the chain), or a legacy file: assume alive


def wait_for_fix77(studies: dict = FIX77_STUDIES, queue: Path = QUEUE, poll: float = POLL_SECONDS,
                   alive=chain_alive, busy=U.other_controllers) -> bool:
    """True when fix77 stage 1 ended with every study 'done'; False when blocked or stopped (BLOCKED.md written
    when blocked)."""
    last_note = 0.0
    while True:
        if stopped(queue):
            log("STOP present while waiting for fix77; exiting", queue)
            set_state(queue, phase="stopped")
            return False
        try:
            status, reason = U.fix77_gate(alive(), busy(), {k: U.read_phase(p) for k, p in studies.items()})
        except (OSError, ValueError) as exc:
            blocked(f"cannot read the fix77 chain or state files: {exc}", queue)
            return False
        if status == "go":
            log(reason, queue)
            return True
        if status == "blocked":
            blocked(reason, queue)
            return False
        if time.time() - last_note >= NOTE_EVERY:
            log(f"waiting: {reason}", queue)
            last_note = time.time()
        time.sleep(poll)


def wait_ready(queue: Path = QUEUE, poll: float = POLL_SECONDS) -> bool:
    """Wait for >= MIN_FREE_MIB free GPU memory and no other controller or job. False if STOP appears."""
    last_note = 0.0
    while True:
        if stopped(queue):
            log("STOP present; exiting", queue)
            set_state(queue, phase="stopped")
            return False
        free, busy = U.gpu_free_mib(), U.other_controllers()
        if free is not None and free >= MIN_FREE_MIB and not busy:
            log(f"GPU ready: {free} MiB free, no other controller", queue)
            return True
        if time.time() - last_note >= NOTE_EVERY:
            log(f"waiting for the GPU: free {free} MiB (need {MIN_FREE_MIB}), other controllers {busy}", queue)
            last_note = time.time()
        time.sleep(poll)


def driver_env(smoke: bool, queued: bool = True) -> dict:
    """Clean environment for a driver; ``queued`` marks it as launched by the queue (it then keeps its STOP)."""
    env = os.environ.copy()
    for key in ENV_KEYS + (U.QUEUED_ENV,):
        env.pop(key, None)
    if smoke:
        env["CASCADE_SMOKE"] = "1"
    if queued:
        env[U.QUEUED_ENV] = "1"
    return env


def run_driver(module: str, smoke: bool, queue: Path = QUEUE) -> int:
    log(f"starting {module} ({'smoke' if smoke else 'real'})", queue)
    with open(queue / f"{module.split('.')[-1]}_{'smoke' if smoke else 'real'}.log", "a", encoding="utf-8") as out:
        code = subprocess.run([sys.executable, "-m", module, "run"], cwd=ROOT, env=driver_env(smoke),
                              stdout=out, stderr=subprocess.STDOUT).returncode
    log(f"{module} ({'smoke' if smoke else 'real'}) exited {code}", queue)
    return code


def study_folders() -> list[Path]:
    return [s for _, _, a, b in STEPS for s in (a, b)]


# ------------------------------------------------------------------------------------------------ watchdog
def watch(queue: Path = QUEUE) -> int:
    """The queue. STOP is not cleared here (``start`` clears it), so a stop issued during start-up is honoured."""
    queue.mkdir(parents=True, exist_ok=True)
    try:
        return _watch(queue)
    except Exception as exc:                  # noqa: BLE001 - record why the watchdog died, then re-raise
        blocked(f"watchdog error: {exc!r} (see the queue logs)", queue)
        raise


def _watch(queue: Path) -> int:
    log("watchdog started", queue)
    set_state(queue, phase="waiting_for_fix77")
    if not wait_for_fix77(queue=queue):
        return 1
    for name, module, smoke_study, real_study in STEPS:
        for smoke, study in ((True, smoke_study), (False, real_study)):
            mode = "smoke" if smoke else "real"
            set_state(queue, phase="waiting_for_gpu", step=name, mode=mode)
            if not wait_ready(queue):
                return 1
            set_state(queue, phase="running", step=name, mode=mode)
            code = run_driver(module, smoke, queue)
            phase = U.read_phase(study)
            if code == 0 and phase == "stopped":
                log(f"{name} {mode} honoured STOP; queue stopped (restart resumes from cached jobs)", queue)
                set_state(queue, phase="stopped", step=name, mode=mode)
                return 1
            if code != 0 or phase != "done":
                blocked(f"{name} {mode} run ended with exit code {code} and phase {phase!r} "
                        f"(see {study.relative_to(ROOT).as_posix()} and the queue logs)", queue)
                return 1
            log(f"{name} {mode}: phase done", queue)
            if stopped(queue):
                log("STOP present after the step; exiting", queue)
                set_state(queue, phase="stopped", step=name, mode=mode)
                return 1
    set_state(queue, phase="done", step=None, mode=None)
    log("all upgrade-round-2 steps done", queue)
    return 0


def _watch_alive(queue: Path = QUEUE) -> int | None:
    path = queue / "queue_pid.json"
    if not path.exists():
        return None
    import psutil
    try:
        pid = int(json.loads(path.read_text(encoding="utf-8"))["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    try:
        return pid if "experiment.upgrades2_queue" in psutil.Process(pid).cmdline() else None
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["start", "status", "stop", "_watch"])
    args = parser.parse_args(argv)
    QUEUE.mkdir(parents=True, exist_ok=True)
    if args.command == "_watch":
        return watch()
    if args.command == "start":
        if _watch_alive():
            print(f"not started: the watchdog already runs (pid {_watch_alive()})")
            return 1
        if (QUEUE / "BLOCKED.md").exists():
            print(f"not started: {QUEUE / 'BLOCKED.md'} exists; resolve it and delete the file first")
            return 1
        for folder in [QUEUE] + study_folders():       # an explicit start clears old stops (fix77 untouched)
            (folder / "STOP").unlink(missing_ok=True)
        out = open(QUEUE / "watchdog_stdout.log", "a", encoding="utf-8")
        flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0   # DETACHED_PROCESS | NEW_PROCESS_GROUP
        proc = subprocess.Popen([sys.executable, "-m", "experiment.upgrades2_queue", "_watch"], cwd=ROOT,
                                stdout=out, stderr=subprocess.STDOUT, creationflags=flags,
                                env=driver_env(False, queued=False))
        U.write_json_atomic(QUEUE / "queue_pid.json", {"pid": proc.pid, "started": time.time()})
        print(f"watchdog started, pid {proc.pid}; log: {QUEUE / 'queue.log'}")
    elif args.command == "stop":
        for folder in [QUEUE] + study_folders():
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "STOP").write_text("stop requested\n", encoding="utf-8")
        print("STOP written (queue and upgrades2 studies; fix77 studies untouched)")
    else:
        state = QUEUE / "queue_state.json"
        print(state.read_text(encoding="utf-8") if state.exists() else "no queue state yet")
        print(f"watchdog alive: {_watch_alive()}")
        if (QUEUE / "BLOCKED.md").exists():
            print((QUEUE / "BLOCKED.md").read_text(encoding="utf-8"))
        for name, path in FIX77_STUDIES.items():
            print(f"{name}: phase {U.read_phase(path)!r}")
        try:
            alive = chain_alive()
        except ValueError as exc:
            alive = f"unknown ({exc})"
        print(f"fix77 chain alive: {alive}")
        for name, _, smoke_study, real_study in STEPS:
            print(f"{name}: smoke phase {U.read_phase(smoke_study)!r}, real phase {U.read_phase(real_study)!r}")
        print(f"GPU free MiB: {U.gpu_free_mib()}; other controllers: {U.other_controllers()}")
        if (QUEUE / "queue.log").exists():
            print("".join((QUEUE / "queue.log").read_text(encoding="utf-8").splitlines(keepends=True)[-10:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
