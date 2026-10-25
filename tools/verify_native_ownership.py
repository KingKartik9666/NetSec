"""Verify OS cleanup after forcibly terminating a real native inference parent.

Only this driver's processes are terminated. Models and prepared sample bytes
are read-only; Python is a test driver, never the application's inference path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import psutil

from tools.verify_native_app import Evidence, NativeRuntime, ROOT, RELEASE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=RELEASE)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = Evidence(args.output)
    binary = evidence.folder / "bin/netsec-runtime.exe"
    binary.parent.mkdir(exist_ok=True)
    shutil.copy2(args.binary, binary)
    runtime = NativeRuntime(args.root, binary, evidence, "forced-parent")
    children = []
    try:
        runtime.start()
        sample = runtime.json("/api/v1/samples")["samples"][0]
        runtime.json("/api/v1/jobs", {"model_ids": ["random_forest"], "rows": [sample["row"]] * 500})
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            children = psutil.Process(runtime.process.pid).children(recursive=True)
            workers = [p for p in children if "--predict-worker" in p.cmdline()]
            if workers:
                break
            time.sleep(0.01)
        else:
            raise RuntimeError("A genuine native prediction worker was not observed")
        # A paused genuine worker models a stuck native operation deterministically;
        # it must still be retired by OS process ownership when its parent dies.
        for worker in workers:
            worker.suspend()
        evidence.check("real native worker active before forced parent termination", True,
                       worker_pids=[p.pid for p in workers], paused_to_model_stuck_worker=True)
        runtime.process.kill()
        runtime.process.wait(timeout=10)
        _, alive = psutil.wait_procs(children, timeout=5)
        evidence.check("forced runtime termination leaves no descendants", not alive,
                       survivor_pids=[p.pid for p in alive])
    except Exception as error:
        evidence.check("ownership verification completed", False, error=str(error))
    finally:
        # Cleanup the observed identities only, including the intentionally red
        # pre-fix case. psutil checks creation time before sending termination.
        for child in children:
            try:
                if child.is_running():
                    child.kill()
            except psutil.NoSuchProcess:
                pass
        runtime.close(expected_exit_codes=(0, 1))
    evidence.data["passed"] = all(c["passed"] for c in evidence.data["checks"])
    evidence.save()
    print(json.dumps({"passed": evidence.data["passed"], "report": str(evidence.folder / "report.json")}))
    return 0 if evidence.data["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
