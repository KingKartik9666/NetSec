"""Run actual local generation on saved validation evidence and measure process RSS."""
import json
from pathlib import Path
import threading
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    from framework.detection.trained_xgboost import XGBoostDetector
    from framework.app.server import create_app
    detector = XGBoostDetector(ROOT / "framework/models/xgboost_app/bundle.json")
    client = create_app(detector).test_client()
    process = psutil.Process()
    baseline = process.memory_info().rss
    peaks = [baseline]
    done = threading.Event()
    def sample():
        while not done.wait(.02):
            peaks.append(process.memory_info().rss)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        response = client.post("/api/assistant/explain", json={"kind": "sample", "sample_id": 0})
    finally:
        done.set()
        thread.join()
    result = {"http_status": response.status_code, "result": response.json,
              "wall_seconds": round(time.monotonic() - started, 2),
              "process_baseline_rss_mib": round(baseline / 1048576, 1),
              "process_peak_rss_mib": round(max(peaks) / 1048576, 1),
              "process_peak_increase_mib": round((max(peaks) - baseline) / 1048576, 1),
              "process_after_rss_mib": round(process.memory_info().rss / 1048576, 1),
              "provider_after": client.get("/api/assistant/status").json,
              "measurement_scope": "One actual saved-validation request including LIME, Python process and CPU LLM; RSS is not dedicated model RAM or a benchmark of other PCs."}
    output = ROOT / "artifacts/desktop/verification/local_llm.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)
    if response.status_code != 200 or not response.json.get("llm_executed"):
        raise RuntimeError("Actual local generation failed")


if __name__ == "__main__":
    main()
