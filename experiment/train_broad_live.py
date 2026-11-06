"""Durable broad-feature tuning; status is the default, CUDA fitting is explicit."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.training.live_study.protocol import (
    BRANCHES, HYBRIDS, atomic, read, relative, digest, fingerprint, source_pins, versions, exclusive, verify_study,
)

DEFAULT = ROOT / 'artifacts/training/broad_live_models_20261028_v2'
PROFILE = ROOT / 'artifacts/training/live_features_20261027_v1/training_profile.json'


def initialize(study, profile=PROFILE, hours=24, smoke=False, cuda=False):
    from framework.training.broad_live.data import prepare
    study, profile = Path(study).resolve(), Path(profile).resolve()
    if not study.is_relative_to(ROOT / 'artifacts/training'):
        raise ValueError('Study must be isolated under artifacts/training')
    if not cuda:
        raise ValueError('This research queue requires explicit --cuda; default app inference stays CPU')
    if not 0 < hours <= 168:
        raise ValueError('Budget must be positive and no greater than168 hours')
    with exclusive(study / 'controller.lock'):
        if (study / 'study.json').exists():
            settings, _ = verify_study(study)
            if settings['hours'] != hours or settings['smoke'] != smoke or settings['feature_profile_sha256'] != digest(profile):
                raise ValueError('Existing study protocol differs; use a new revision')
            return settings
        manifest = prepare(study, profile)
        settings = dict(version=2, created_at=time.time(), hours=hours, smoke=smoke,
            trials_per_branch=1 if smoke else 32, trials_per_hybrid=2 if smoke else 48,
            repeat_seeds=[43] if smoke else [43, 44], branches=list(BRANCHES), hybrids=HYBRIDS,
            feature_profile=relative(profile), feature_profile_sha256=digest(profile),
            feature_count=len(manifest['feature_names']),
            objective='tuning macro F1; raw class argmax or HEM hard vote; calibration separate',
            search='Multivariate group TPE,8 startup trials, MedianPruner; deterministic persisted proposals',
            cpu_inference=True, training_device='explicit CUDA except RF and weight/calibration optimization',
            test_locked=True, allow_pseudolabels=False, network_required=False,
            data_manifest_fingerprint=fingerprint(manifest),
            schema_fingerprint=fingerprint(dict(feature_names=manifest['feature_names'], classes=manifest['classes'], window_seconds=10)),
            source_hashes=source_pins(), versions=versions())
        settings['protocol_fingerprint'] = fingerprint(settings)
        atomic(study / 'study.json', settings)
        return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='status', choices=[
        'init', 'status', 'supervise', 'stop', 'resume', 'install-logon', 'remove-logon', '_controller', '_assess'])
    parser.add_argument('--study', type=Path, default=DEFAULT)
    parser.add_argument('--profile', type=Path, default=PROFILE)
    parser.add_argument('--hours', type=float, default=24)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--ready', type=Path)
    parser.add_argument('--job-name')
    args = parser.parse_args()
    study = args.study.resolve()
    if args.job_name:
        from framework.training.live_study.processes import join_current
        join_current(args.job_name)
    if args.command == 'init':
        settings = initialize(study, args.profile, args.hours, args.smoke, args.cuda)
        print(f'Initialized {settings["feature_count"]}-feature study: {study}')
    elif args.command == 'status':
        from experiment.train_live_models import status
        status(study)
    elif args.command == 'stop':
        study.mkdir(parents=True, exist_ok=True)
        (study / 'STOP').touch()
        print('STOP requested for this broad study only')
    elif args.command == 'resume':
        verify_study(study)
        with exclusive(study / 'supervisor.lock'), exclusive(study / 'controller.lock'):
            state = read(study / 'state.json')
            if state['status'] == 'completed' or time.time() >= state['deadline']:
                raise ValueError('Study complete or budget expired; preserve it and use a new revision')
            (study / 'STOP').unlink(missing_ok=True)
            state.update(status='resuming', current=None)
            atomic(study / 'state.json', state)
        from framework.training.broad_live.supervisor import supervise
        supervise(study)
    elif args.command in ('supervise', 'install-logon', 'remove-logon'):
        from framework.training.broad_live.supervisor import supervise, install_startup, remove_startup
        if args.command == 'supervise':
            supervise(study)
        elif args.command == 'install-logon':
            verify_study(study)
            print(install_startup(study))
        else:
            remove_startup(study)
    elif args.command == '_controller':
        from framework.training.broad_live.controller import execute
        execute(study, args.ready)
    elif args.command == '_assess':
        from framework.training.broad_live.report import assess
        try:
            assess(study, args.output)
        except BaseException as error:
            import traceback
            atomic(args.output / 'failure.json', dict(status='failed', error=str(error), traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
