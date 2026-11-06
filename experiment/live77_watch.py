"""Watchdog for the live 77 study: restart the controller after a crash, then run the queued upgrade tests.

    python -m experiment.live77_watch start          # detached (log: <study>/watch.log)
    python -m experiment.live77_watch start resume   # detached: wait for the running controller to exit (e.g. after a
                                                     # STOP for a code change), restart it, then keep watching

Owner 2026-10-31: "dont stop current training after first run completes just add them to queue". The watchdog
never stops a running controller. When the controller process is gone:
* state phase "done"    -> run ``experiment.train_live77_upgrades run`` once if that module exists, then exit;
* state phase "stopped" -> exit (an explicit STOP is respected);
* otherwise (error/crash) -> move jobs whose result.json says "failed" to ``failed_jobs/`` (kept as evidence)
  and restart the controller, at most MAX_RESTARTS times.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("CASCADE_STUDY", "artifacts/training/live77_20261031_v1")

from framework.training.csv_v3 import runner  # noqa: E402

STUDY = runner.STUDY
MAX_RESTARTS = 5
POLL_SECONDS = 60


def _log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def _alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return f" {pid} " in out


def _pid(name: str) -> int | None:
    path = STUDY / name
    return json.loads(path.read_text(encoding="utf-8"))["pid"] if path.exists() else None


def _phase() -> str | None:
    path = STUDY / "state.json"
    return json.loads(path.read_text(encoding="utf-8")).get("phase") if path.exists() else None


def _quarantine_failed() -> list[str]:
    moved = []
    for result in (STUDY / "jobs").glob("*/result.json"):
        if json.loads(result.read_text(encoding="utf-8")).get("status") == "failed":
            target = STUDY / "failed_jobs" / f"{result.parent.name}_{time.strftime('%Y%m%d_%H%M%S')}"
            target.parent.mkdir(exist_ok=True)
            shutil.move(str(result.parent), str(target))
            moved.append(result.parent.name)
    return moved


def watch() -> None:
    restarts = 0
    while True:
        pid = _pid("controller_pid.json")
        while pid is not None and _alive(pid):
            time.sleep(POLL_SECONDS)
        phase = _phase()
        if phase == "done":
            if importlib.util.find_spec("experiment.train_live77_upgrades") and not (STUDY / "upgrades_done").exists():
                _log("main run done; starting queued upgrade tests")
                code = subprocess.run([sys.executable, "-m", "experiment.train_live77_upgrades", "run"],
                                      cwd=runner.ROOT).returncode
                _log(f"upgrade tests exited with {code}")
            else:
                _log("main run done; no queued upgrade module (or already done)")
            return
        if phase == "stopped":
            _log("controller stopped by STOP file; watchdog exits")
            return
        if restarts >= MAX_RESTARTS:
            _log(f"controller ended in phase {phase!r}; restart limit reached, watchdog exits")
            return
        moved = _quarantine_failed()
        restarts += 1
        _log(f"controller ended in phase {phase!r}; moved failed jobs {moved}; restart {restarts}/{MAX_RESTARTS}")
        subprocess.run([sys.executable, "-m", "experiment.train_live77", "start"], cwd=runner.ROOT)
        time.sleep(10)


def resume() -> None:
    pid = _pid("controller_pid.json")
    while pid is not None and _alive(pid):
        time.sleep(20)
    _log("controller exited; restarting it (resume)")
    subprocess.run([sys.executable, "-m", "experiment.train_live77", "start"], cwd=runner.ROOT)
    time.sleep(10)
    watch()


def main(argv=None) -> int:
    args = argv or sys.argv[1:] or ["run"]
    command = args[0]
    if command == "start":
        log = open(STUDY / "watch.log", "a", encoding="utf-8")
        flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0
        mode = "resume" if args[1:] == ["resume"] else "run"
        proc = subprocess.Popen([sys.executable, "-m", "experiment.live77_watch", mode], cwd=runner.ROOT,
                                stdout=log, stderr=subprocess.STDOUT, creationflags=flags, env=os.environ.copy())
        (STUDY / "watch_pid.json").write_text(json.dumps({"pid": proc.pid, "started": time.time()}))
        print(f"watchdog started, pid {proc.pid}; log: {STUDY / 'watch.log'}")
    elif command == "resume":
        resume()
    else:
        watch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
