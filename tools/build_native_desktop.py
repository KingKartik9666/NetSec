"""Build the Electron/React/Rust Windows application from approved local assets.

Python orchestrates developer tooling only. It is never copied into the app.
Model export/training and network downloads are not performed by this command.
"""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installer", action="store_true", help="Also build the Windows NSIS installer")
    args = parser.parse_args()
    node = shutil.which("node")
    cargo = shutil.which("cargo") or str(Path(os.environ["USERPROFILE"]) / ".cargo/bin/cargo.exe")
    if not node or not Path(cargo).is_file():
        raise SystemExit("Install Node.js and the pinned Rust MSVC toolchain first; see docs/NATIVE_APP.md.")
    web = ROOT / "rust/netsec/web"
    desktop = ROOT / "framework/app/desktop"
    # Keep extraction and rename on the workspace volume. The user's redirected
    # Windows cache can otherwise produce EXDEV during NSIS/7zip extraction.
    os.environ.setdefault("ELECTRON_BUILDER_CACHE", str(ROOT / "artifacts/desktop/native-builder-cache"))

    def run(command, cwd=ROOT):
        subprocess.run([str(value) for value in command], cwd=cwd, check=True)

    run([node, web / "node_modules/typescript/bin/tsc", "--noEmit"], web)
    run([node, web / "node_modules/vite/bin/vite.js", "build"], web)
    run([cargo, "build", "--locked", "--release", "-p", "netsec-runtime"], ROOT / "rust/netsec")
    run([sys.executable, ROOT / "tools/stage_native_app.py", "--seal-only"])
    run([sys.executable, ROOT / "tools/stage_native_app.py"])
    if args.installer:
        run([node, desktop / "node_modules/electron-builder/out/cli/cli.js", "--win", "nsis", "--x64", "--publish", "never"], desktop)


if __name__ == "__main__":
    main()
