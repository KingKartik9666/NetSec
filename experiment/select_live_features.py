"""Prepare broad Npcap candidates and conservatively select inside fitting data."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from framework.training.live_features.data import DEFAULT,prepare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--extended-research',action='store_true')
    parser.add_argument('--screen',action='store_true')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--rounds',type=int,default=100)
    args=parser.parse_args()
    if not 1<=args.rounds<=1000:parser.error('rounds must be1..1000')
    manifest=prepare(args.output.resolve(),args.extended_research)
    print(f'Prepared {manifest["candidate_features"]} candidate features across {manifest["rows"]} original training windows.',flush=True)
    if args.screen:
        from framework.training.live_features.screening import run
        run(args.output.resolve(),args.device,args.rounds)


if __name__=='__main__':main()
