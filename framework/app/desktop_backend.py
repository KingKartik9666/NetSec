"""Owned desktop service: random loopback port, per-launch authentication, EOF cleanup."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading


def main():
    from multiprocessing import freeze_support
    freeze_support()
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="Explicit XGBoost bundle override")
    parser.add_argument("--model-config", type=Path, help="Optional four-model configuration JSON")
    args = parser.parse_args()
    token = os.environ.pop("IMDAI_SESSION_TOKEN", "")
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("The desktop launcher must supply a fresh session token")
    from werkzeug.serving import make_server
    from framework.app.server import create_app
    from framework.core.registry import ModelRegistry
    registry = ModelRegistry(root, args.model_config)
    if args.bundle:
        registry.configuration["models"]["xgboost"]["bundle_path"] = str(args.bundle.resolve())
    app = create_app(registry=registry, session_token=token)
    server = make_server("127.0.0.1", 0, app, threaded=True)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    # Keep the launcher pipe private. On Windows a spawned interpreter can
    # block during stdio initialization while we read the same stdin pipe.
    # Redirect fd 0 before serving requests, but keep commands/EOF on the copy.
    with os.fdopen(os.dup(sys.stdin.fileno()), encoding="utf-8") as commands:
        with open(os.devnull) as empty:
            os.dup2(empty.fileno(), sys.stdin.fileno())
        worker.start()
        print(json.dumps({"type": "ready", "port": server.server_port}), flush=True)
        try:
            for line in commands:
                if line.strip() == "shutdown":
                    break
        finally:
            app.extensions["close_resources"]()
            server.shutdown()
            server.server_close()
            worker.join(timeout=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
