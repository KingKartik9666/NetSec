"""CLI for the cascade v3 study (77-class -> 8-class recon specialist -> 84-class).

    python -m experiment.train_cascade_v3 start            # detached controller: 77, 8, 84xgb control, cascade report
    python -m experiment.train_cascade_v3 run [--tasks 77] # controller in the foreground
    python -m experiment.train_cascade_v3 status
    python -m experiment.train_cascade_v3 stop             # STOP file; the controller halts between jobs

Research GPU training is explicit here; application and demo inference remain CPU-only and offline.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from framework.training.csv_v3 import runner


def _log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "start", "status", "stop", "_fit", "_predict"])
    parser.add_argument("request", nargs="?")
    parser.add_argument("--tasks", default=",".join(runner.TASK_ORDER))
    args = parser.parse_args(argv)

    if args.command == "_fit":
        runner.worker_fit(Path(args.request))
        return 0
    if args.command == "_predict":
        runner.worker_predict(Path(args.request))
        return 0
    if args.command == "run":
        (runner.STUDY / "STOP").unlink(missing_ok=True)
        runner.Controller(log=_log).run([t for t in args.tasks.split(",") if t])
        return 0
    if args.command == "start":
        runner.STUDY.mkdir(parents=True, exist_ok=True)
        (runner.STUDY / "STOP").unlink(missing_ok=True)
        log = open(runner.STUDY / "controller.log", "a", encoding="utf-8")
        flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0   # DETACHED_PROCESS | NEW_PROCESS_GROUP
        proc = subprocess.Popen([sys.executable, "-m", "experiment.train_cascade_v3", "run", "--tasks", args.tasks],
                                cwd=runner.ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        (runner.STUDY / "controller_pid.json").write_text(json.dumps({"pid": proc.pid, "started": time.time()}))
        print(f"controller started, pid {proc.pid}; log: {runner.STUDY / 'controller.log'}")
        return 0
    if args.command == "stop":
        (runner.STUDY / "STOP").write_text("stop requested\n", encoding="utf-8")
        print("STOP written; the controller exits after the current job")
        return 0
    state = runner.STUDY / "state.json"
    print(state.read_text(encoding="utf-8") if state.exists() else "no state yet")
    jobs = sorted((runner.STUDY / "jobs").glob("*/result.json")) if (runner.STUDY / "jobs").exists() else []
    print(f"{len(jobs)} finished jobs")
    for path in jobs[-12:]:
        r = json.loads(path.read_text(encoding="utf-8"))
        rep = r.get("report", {})
        acc = rep.get("assessment", {}).get("accuracy")
        print(f"  {r['job_id']:<45} {r['status']:<9} tune={r.get('tune_value', float('nan')):.4f} "
              f"assess={'-' if acc is None else f'{100*acc:.2f}%'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
