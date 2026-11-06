"""Build the Electron/React/Rust Windows application from approved local assets.

Python orchestrates developer tooling only. It is never copied into the app.
Model export/training and network downloads are not performed by this command.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
        # The language model is too large for NSIS (about 2 GB); it ships as a pack beside the installer, which
        # copies it in (installer/installer.nsh). The file name in the include must match the installed model.
        model = ROOT / "artifacts/llm/model.json"
        pack = None
        if model.is_file():
            pack = ROOT / "artifacts/llm" / json.loads(model.read_text(encoding="utf-8"))["file"]
            if pack.name not in (desktop / "installer/installer.nsh").read_text(encoding="utf-8"):
                raise SystemExit(f"installer/installer.nsh does not install {pack.name}; update the model pack file name.")
        run([node, desktop / "node_modules/electron-builder/out/cli/cli.js", "--win", "nsis", "--x64", "--publish", "never"], desktop)
        releases = ROOT / "artifacts/desktop/releases"
        version = json.loads((desktop / "package.json").read_text(encoding="utf-8"))["version"]
        files = [releases / f"NetSec-Setup-{version}-x64.exe"]
        if pack:
            target = releases / pack.name
            if not target.exists():
                try:
                    os.link(pack, target)  # same volume: no second 2.4 GB copy
                except OSError:
                    shutil.copy2(pack, target)
            files.append(target)
        sums = "".join(f"{digest(path)}  {path.name}\n" for path in files)
        (releases / f"SHA256SUMS-{version}.txt").write_text(sums, encoding="ascii")


if __name__ == "__main__":
    main()
