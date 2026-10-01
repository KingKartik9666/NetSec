"""Execute actual local Qwen with retrieved background and measured saved-model XAI."""
import json
from pathlib import Path

from framework.app.server import create_app
from framework.detection.trained_xgboost import XGBoostDetector

ROOT = Path(__file__).resolve().parents[1]


def main():
    detector = XGBoostDetector(ROOT / "framework/models/xgboost_app/bundle.json")
    client = create_app(detector).test_client()
    proof = {"status": client.get("/api/assistant/status").get_json(), "cases": []}
    for sample_id in [0, 162]:
        response = client.post("/api/assistant/explain", json={"kind": "sample", "sample_id": sample_id})
        result = response.get_json()
        proof["cases"].append({"sample_id": sample_id, "http_status": response.status_code, "result": result})
        print(json.dumps({"sample_id": sample_id, "http_status": response.status_code,
                          "text": result.get("text"), "error": result.get("error"),
                          "correction": result.get("correction"), "rag": result.get("rag")}, indent=2), flush=True)
        path = ROOT / "artifacts/desktop/verification/rag-generation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(proof, indent=2), encoding="utf-8")
        assert response.status_code == 200, result
        assert result["llm_executed"] and result["rag"]["used_in_generation"]
        assert result["correction"]["generation_attempts"][-1]["passed"]
        assert result["rag"]["context"] == "\n".join(f"[{p['id']}] {p['text']}" for p in result["rag"]["passages"])
        assert client.get("/api/assistant/status").get_json()["loaded"] is False


if __name__ == "__main__":
    main()
