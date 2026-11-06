"""Durable live-screening study CLI. Default operation is offline status only."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from framework.training.live_study.protocol import (DEFAULT, ROOT, BRANCHES, HYBRIDS, FEATURES,
    atomic, read, relative, fingerprint, source_pins, versions, exclusive, verify_study)


def initialize(study,hours,smoke):
    from framework.training.live_study.data import prepare
    study=Path(study).resolve()
    if not study.is_relative_to(ROOT/'artifacts/training'):
        raise ValueError('Study outputs must stay under artifacts/training')
    with exclusive(study/'controller.lock'):
        if (study/'study.json').exists():
            settings,_=verify_study(study)
            if settings['hours']!=hours or settings['smoke']!=smoke:
                raise ValueError('Existing study settings differ; use a new revision')
            return settings
        manifest=prepare(study)
        if hours <= 0 or hours > 168:raise ValueError('Budget must be between zero and 168 hours')
        settings=dict(version=1,created_at=time.time(),hours=hours,smoke=smoke,
            trials_per_branch=1 if smoke else 16,trials_per_hybrid=2 if smoke else 24,
            repeat_seeds=[43] if smoke else [43,44],branches=list(BRANCHES),hybrids=HYBRIDS,
            objective='tuning macro F1; calibration fitted on independent training role',
            cpu_inference=True,training_device='explicit CUDA except sklearn RF and blend optimization',
            test_locked=True,allow_pseudolabels=False,network_required=False,
            data_manifest_fingerprint=fingerprint(manifest),
            schema_fingerprint=fingerprint(dict(feature_names=FEATURES,classes=manifest['classes'],window_seconds=10)),
            source_hashes=source_pins(),versions=versions())
        settings['protocol_fingerprint']=fingerprint(settings)
        atomic(study/'study.json',settings)
        return settings


def status(study):
    import json
    import psutil
    state=read(study/'state.json') if (study/'state.json').exists() else dict(status='not_started')
    def alive(pid,created):
        try:return psutil.Process(pid).create_time()==created
        except (psutil.Error,TypeError):return False
    supervisor=read(study/'supervisor.json') if (study/'supervisor.json').exists() else {}
    state.update(study=relative(study),supervisor_alive=alive(supervisor.get('pid'),supervisor.get('created')),
        controller_alive=alive(state.get('controller_pid'),state.get('controller_created')),
        heartbeat_age_seconds=time.time()-state['heartbeat'] if 'heartbeat' in state else None,
        stop_requested=(study/'STOP').exists())
    if 'deadline' in state:state['remaining_hours']=max(0,(state['deadline']-time.time())/3600)
    print(json.dumps(state,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',default='status',choices=['init','status','supervise','stop','resume','install-logon','remove-logon','_worker','_controller','_assess'])
    parser.add_argument('--study',type=Path,default=DEFAULT)
    parser.add_argument('--hours',type=float,default=24)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--request',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--ready',type=Path)
    parser.add_argument('--job-name')
    args=parser.parse_args();study=args.study.resolve()
    if args.job_name:
        from framework.training.live_study.processes import join_current
        join_current(args.job_name)
    if args.command=='init':
        initialize(study,args.hours,args.smoke);print(f'Initialized {study}')
    elif args.command=='status':status(study)
    elif args.command=='stop':
        study.mkdir(parents=True,exist_ok=True);(study/'STOP').touch();print('STOP requested; only this study worker will stop.')
    elif args.command=='resume':
        # Explicit resume never changes hashes or the original wall-clock budget.
        verify_study(study)
        with exclusive(study/'supervisor.lock'),exclusive(study/'controller.lock'):
            state=read(study/'state.json')
            if state['status']=='completed':raise ValueError('Study is already complete')
            if time.time()>=state['deadline']:raise ValueError('Budget expired; preserve this study and initialize an explicit new revision')
            (study/'STOP').unlink(missing_ok=True);state.update(status='resuming',current=None)
            atomic(study/'state.json',state)
        from framework.training.live_study.supervisor import supervise
        supervise(study)
    elif args.command in ('install-logon','remove-logon','supervise'):
        from framework.training.live_study.supervisor import install_startup,remove_startup,supervise
        if args.command=='install-logon':verify_study(study);print(install_startup(study))
        elif args.command=='remove-logon':remove_startup(study)
        else:supervise(study)
    elif args.command=='_controller':
        from framework.training.live_study.controller import execute
        execute(study,args.ready)
    elif args.command=='_worker':
        from framework.training.live_study.worker import main as worker
        worker(args.request)
    elif args.command=='_assess':
        from framework.training.live_study.report import assess
        assess(study,args.output)


if __name__=='__main__':main()
