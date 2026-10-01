"""Launch local intrusion analysis with opt-in PC monitoring and TreeSHAP."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    from framework.detection.trained_xgboost import default_bundle
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--bundle", type=Path, default=default_bundle(ROOT))
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    from framework.detection.trained_xgboost import XGBoostDetector
    from framework.app.server import create_app
    try:
        detector = XGBoostDetector(args.bundle)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved XGBoost: {detector.bundle['model']['run_id']}")
    print("CPU inference | exact TreeSHAP | offline template | unknown attacks not evaluated")
    print("Live PC monitoring starts only when you press Start; packet inspection needs Npcap on Windows.")
    print(f"Open http://127.0.0.1:{args.port} in your browser. Stop with Ctrl+C.")
    create_app(detector).run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
