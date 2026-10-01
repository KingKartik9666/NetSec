"""Owned desktop service: random loopback port, per-launch authentication, EOF cleanup."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading


def main():
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    from framework.detection.trained_xgboost import default_bundle
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=default_bundle(root))
    args = parser.parse_args()
    token = os.environ.pop("IMDAI_SESSION_TOKEN", "")
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("The desktop launcher must supply a fresh session token")
    from werkzeug.serving import make_server
    from framework.app.server import create_app
    from framework.data.live_monitor import LiveMonitor
    from framework.detection.trained_xgboost import XGBoostDetector
    detector = XGBoostDetector(args.bundle)
    monitor = LiveMonitor(detector=detector)
    app = create_app(detector, monitor=monitor, session_token=token)
    server = make_server("127.0.0.1", 0, app, threaded=True)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    print(json.dumps({"type": "ready", "port": server.server_port}), flush=True)
    try:
        for line in sys.stdin:
            if line.strip() == "shutdown":
                break
    finally:
        monitor.stop()
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
