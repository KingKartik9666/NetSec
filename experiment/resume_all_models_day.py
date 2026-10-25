"""Explicit post-pause continuation; the original sealed day remains intact."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import shutil
import time

from experiment import optimize_all_models_day as runner
from experiment.datasense_quality import digest
from framework.training import day_protocol
from framework.training.hybrid_protocol import ROOT,relative,verify_hashes
from framework.training.study_supervisor import atomic_write,read

ORIGINAL=ROOT/'artifacts/training/all_models_day_20261020_v1'
STUDY=ROOT/'artifacts/training/all_models_resume_20261021_v1'
MODULE='experiment.resume_all_models_day'


def continuation_budget(original,pause,now):
    elapsed=max(0.,pause['requested_at']-original['started_at'])
    remaining=original['deadline']-original['started_at']-elapsed
    if remaining<=0:
        raise TimeoutError('Original active-time budget was already exhausted')
    return dict(started_at=now-elapsed,deadline=now+remaining,
        active_elapsed_seconds=elapsed,remaining_seconds_at_resume=remaining,resumed_at=now)


def initialize():
    if (STUDY/'study.json').exists():
        protocol=read(STUDY/'study.json');day_protocol.verify(protocol);return protocol
    original=read(ORIGINAL/'study.json');day_protocol.verify(original)
    pause=read(ORIGINAL/'pause_request.json')
    if not (ORIGINAL/'STOP').exists():
        raise RuntimeError('Original study must remain paused during versioned continuation')
    selected=read(ORIGINAL/'selected.json')
    for record in selected.values():
        verify_hashes(record['artifact_hashes'])
    STUDY.mkdir(parents=True,exist_ok=True)
    for name in ('validation_roles.npy','selected.json','baselines.json','previous_recovery.json'):
        shutil.copy2(ORIGINAL/name,STUDY/name)
    shutil.copytree(ORIGINAL/'folds',STUDY/'folds',dirs_exist_ok=True)
    # Successful repeat/OOF results can be referenced without copying models or
    # rewriting original attempts. The actual paused run has no successful repeats.
    imports={}
    for path in ORIGINAL.glob('runs/*_repeat_s*/attempt_*/result.json'):
        record=read(path)
        if record.get('status')=='completed':
            imports[record['job']]=dict(source_result=relative(path),sha256=digest(path),result=record)
    protocol=deepcopy(original)
    protocol.update(continuation_budget(original,pause,time.time()))
    protocol['started_utc']=datetime.fromtimestamp(protocol['started_at'],timezone.utc).isoformat()
    protocol['deadline_utc']=datetime.fromtimestamp(protocol['deadline'],timezone.utc).isoformat()
    protocol['continuation']=dict(original_study=relative(ORIGINAL),
        original_protocol_sha256=digest(ORIGINAL/'study.json'),pause_request_sha256=digest(ORIGINAL/'pause_request.json'),
        authorization='User: resume everything; restore unused active time after explicit pause',
        inherited_selection_sha256=digest(ORIGINAL/'selected.json'),new_search=False,
        original_deadline_unchanged=True,imported_successful_repeats=imports)
    protocol['source_hashes'][relative(Path(__file__))]=digest(__file__)
    for path in [STUDY/'validation_roles.npy',*(STUDY/'folds').rglob('*.npy')]:
        protocol['role_hashes'][relative(path)]=digest(path)
    for name,value in protocol['source_hashes'].items():
        target=STUDY/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    atomic_write(STUDY/'study.json',protocol)
    atomic_write(STUDY/'continuation.json',dict(protocol['continuation'],
        remaining_seconds=protocol['remaining_seconds_at_resume'],deadline_utc=protocol['deadline_utc']))
    return protocol


def configure():
    runner.STUDY=STUDY
    runner.MODULE=MODULE
    runner.initialize=initialize


if __name__=='__main__':
    configure()
    runner.main()
