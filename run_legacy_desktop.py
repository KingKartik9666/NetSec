"""Launch NetSec's Electron desktop app from the existing project environment."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    desktop = ROOT / "framework/app/desktop"
    binary = desktop / "node_modules/electron/dist" / ("electron.exe" if sys.platform == "win32" else "electron")
    if not binary.is_file():
        raise SystemExit("Install desktop dependencies first: npm install --prefix framework/app/desktop")
    env = {**os.environ, "IMDAI_PYTHON": sys.executable}
    env.pop("ELECTRON_RUN_AS_NODE", None)
    return subprocess.call([str(binary), str(desktop)], cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
