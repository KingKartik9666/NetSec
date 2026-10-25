"""Compatibility launcher for the native browser app; no Python inference."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    binary = ROOT / "rust/netsec/target/release/netsec-runtime.exe"
    if not binary.is_file():
        parser.error("Build the Rust runtime first; see docs/NATIVE_APP.md")
    return subprocess.call([str(binary), "--root", str(ROOT), "--port", str(args.port), "--browser"], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
