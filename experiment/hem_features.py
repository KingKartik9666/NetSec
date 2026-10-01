"""Run real HEM/Kitsune feature extraction on existing UQ-IoT packet captures."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from framework.data.hem_features import CAPTURE_NAMES, extract_capture, find_capture
from framework.data.loader import PROJECT_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--sample", choices=CAPTURE_NAMES, default="syn")
    source.add_argument("--pcap", type=Path, help="Read an explicit local capture instead.")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum packets read (default 1000).")
    parser.add_argument("--output", type=Path, help="New CSV path; existing files are never overwritten.")
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or PROJECT_ROOT / "dataset" / "processed" / f"hem_{args.sample}_{stamp}.csv"
    try:
        pcap = args.pcap or find_capture(args.sample)
        print("REAL HEM / KITSUNE FEATURE EXTRACTION")
        print(f"Input: {pcap}\nPacket limit: {args.limit}")
        print("Executing original external/HEM feature statistics code on CPU...")
        result = extract_capture(pcap, output, args.limit)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Feature extraction error: {exc}\n")
    print(f"PASS: {result['packets_read']} packets read; {result['feature_rows']} feature rows.")
    print(f"{result['feature_count']} HEM/Kitsune features per row; skipped: {result['skipped_packets']}.")
    print(f"Out-of-order timestamps skipped: {result['out_of_order_packets']}")
    print(f"Decoded protocols: {result['protocol_counts']}")
    print(f"Features: {result['output_csv']}")
    print(f"Provenance: {Path(result['output_csv']).with_suffix('.metadata.json')}")
    print("Stage completed: real PCAP -> original HEM feature engine -> feature CSV.")
    print("Trained HEM classification, XAI and LLM inference are NOT executed by this command.")
    print("The capture name is not a prediction or proof that every packet is an attack.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
