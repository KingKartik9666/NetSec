"""Launch local intrusion analysis with opt-in PC monitoring and TreeSHAP."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    from multiprocessing import freeze_support
    freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--bundle", type=Path, help="Explicit XGBoost bundle override; other model families are unchanged")
    parser.add_argument("--model-config", type=Path, help="Optional four-model configuration JSON")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    from framework.core.registry import ModelRegistry
    from framework.app.server import create_app
    registry = ModelRegistry(ROOT, args.model_config)
    if args.bundle:
        registry.configuration["models"]["xgboost"]["bundle_path"] = str(args.bundle.resolve())
    print("Four independent model families | lazy CPU inference | optional explanations | offline")
    print("Open /models for model selection and recorded comparisons. Model failures remain scoped.")
    print("Live PC monitoring starts only when you press Start; packet inspection needs Npcap on Windows.")
    print(f"Open http://127.0.0.1:{args.port} in your browser. Stop with Ctrl+C.")
    app = create_app(registry=registry)
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
    finally:
        app.extensions["close_resources"]()


if __name__ == "__main__":
    main()
