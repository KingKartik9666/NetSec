"""Run via python -m experiment.investigate_plateau prepare|run|status|stop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from framework.training.plateau.data import read_json
from framework.training.plateau.runner import STUDY, initialize, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'status', 'stop'))
    parser.add_argument('--study', type=Path, default=STUDY)
    args = parser.parse_args()
    if args.command == 'prepare':
        p = initialize(args.study)
        print(json.dumps({key: p[key] for key in ('fit_rows', 'evaluation_rows', 'rounds', 'jobs')}, indent=2))
    elif args.command == 'run':
        run(args.study)
    elif args.command == 'stop':
        if not (args.study/'protocol.json').is_file():
            raise FileNotFoundError('Study has not been prepared')
        (args.study/'STOP').touch()
        print('STOP requested; active fit checks after each boosting round.')
    else:
        state = read_json(args.study/'state.json')
        import psutil
        try:
            process = psutil.Process(state['pid'])
            state['process_identity_matches'] = abs(process.create_time()-state['process_created']) < .05
        except (KeyError, psutil.NoSuchProcess):
            state['process_identity_matches'] = False
        print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
