"""Matched feature ablations; CPU by default, explicit optional CUDA research."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from framework.training.live_features.data import DEFAULT
from framework.training.live_features.assessment import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=DEFAULT)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--rounds', type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 1000:
        parser.error('rounds must be between1 and1000')
    run(args.study, args.device, args.rounds)
