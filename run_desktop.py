"""Developer convenience launcher; the installed Electron app needs no Python."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    desktop = ROOT / "framework/app/desktop"
    binary = desktop / "node_modules/electron/dist/electron.exe"
    if not binary.is_file():
        raise SystemExit("Install desktop build dependencies first; see docs/NATIVE_APP.md.")
    return subprocess.call([str(binary), str(desktop)], cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
