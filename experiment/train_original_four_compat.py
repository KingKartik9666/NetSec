"""Resume the frozen four-model study with Windows-safe resource detection.

The original study source and recipes remain byte-for-byte preserved. This
adapter changes controller scheduling only and records its source adoption.
"""
from __future__ import annotations

import os
import sys
from experiment import train_verified_clean as original

MODULE = 'experiment.train_original_four_compat'
ADAPTER_ID = 'windows_redirector_v1'


def available_resources(gpu=False):
    import psutil
    from experiment.run_xgboost_followup import memory_available
    if gpu:
        for proc in psutil.process_iter(['pid','cmdline']):
            if proc.pid == os.getpid():
                continue
            command = proc.info['cmdline'] or []
            # Idle controllers and their .venv redirectors consume no CUDA fit slot.
            if '--fit' in command and any('experiment.train_' in arg for arg in command):
                return False
    memory = memory_available()
    return memory['physical_gib'] >= (3.5 if gpu else 4.) and memory['commit_gib'] >= 6.


def adoption_record():
    source = original.ROOT/'experiment/train_original_four_compat.py'
    checksum = original.digest(source)
    directory = original.STUDY/'controller_adapters'
    path = directory/f'{ADAPTER_ID}.json'
    record = dict(adapter=ADAPTER_ID, source='experiment/train_original_four_compat.py',
        source_sha256=checksum, original_controller_source_sha256=original.digest(original.ROOT/'experiment/train_verified_clean.py'),
        behavior='Resource probe excludes idle controllers and Windows redirectors; active --fit workers still serialize CUDA.',
        frozen_recipes_unchanged=True, original_study_source_unchanged=True, test_loaded=False)
    if path.exists() and original.read(path) != record:
        raise ValueError('Controller adapter changed; preserve this adoption and version a new adapter')
    original.write(path,record)
    snapshot=original.STUDY/'source_snapshot/experiment/train_original_four_compat.py'
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    if snapshot.exists() and original.digest(snapshot) != checksum:
        raise ValueError('Controller adapter snapshot changed')
    if not snapshot.exists():
        snapshot.write_bytes(source.read_bytes())
    return dict(adapter=ADAPTER_ID,source=record['source'],source_sha256=checksum)


def main():
    if any(arg in sys.argv for arg in ('start','resume','_run','--fit')):
        adoption = adoption_record()
        launch = original.launch_trial

        def annotated_launch(request,state,protocol):
            return launch(dict(request,runtime_controller=adoption),state,protocol)

        original.launch_trial = annotated_launch
    original.available_resources = available_resources
    original.MODULE = MODULE
    original.main()


if __name__ == '__main__':
    main()
