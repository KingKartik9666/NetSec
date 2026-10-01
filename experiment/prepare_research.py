"""Prepare the real local DataSense CSV without invoking any model training."""
import argparse
from pathlib import Path

from framework.data.research_data import prepare

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "dataset/combined_dataset.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset/prepared/datasense_v1")
    parser.add_argument("--train-cap", type=int, default=120000)
    parser.add_argument("--eval-cap", type=int, default=30000)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    prepare(args.source, args.output, args.artifacts, args.train_cap, args.eval_cap)


if __name__ == "__main__":
    main()
