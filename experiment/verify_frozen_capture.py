"""Opt-in executed packet capture proof against the actual frozen CPU backend."""
import json
import os
from pathlib import Path
import queue
import secrets
import subprocess
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    token = secrets.token_hex(32)
    executable = ROOT / "artifacts/desktop/backend/imdai-backend/imdai-backend.exe"
    environment = {**os.environ, "IMDAI_SESSION_TOKEN": token, "IMDAI_LLM_DIR": str(ROOT / "artifacts/llm/qwen")}
    for key in ["PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"]:
        environment.pop(key, None)
    out = ROOT / "artifacts/desktop/verification"
    out.mkdir(parents=True, exist_ok=True)
    ready = queue.Queue()
    with (out / "capture-stderr.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen([str(executable)], cwd=executable.parent, env=environment,
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True)
        threading.Thread(target=lambda: [ready.put(line) for line in child.stdout], daemon=True).start()
        try:
            startup = json.loads(ready.get(timeout=60))
            origin = f"http://127.0.0.1:{startup['port']}"
            def request(path, data=None):
                headers = {"X-IMDAI-Session": token, "Content-Type": "application/json"}
                body = None if data is None else json.dumps(data).encode()
                with urllib.request.urlopen(urllib.request.Request(origin + path, data=body, headers=headers), timeout=60) as response:
                    return json.load(response)
            capabilities = request("/api/live/capabilities")
            assert capabilities["packet_capture_available"], capabilities["packet_capture_message"]
            request("/api/live/start", {"interface": capabilities["recommended_interface"], "mode": "packets"})
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                snapshot = request("/api/live/status")
                if snapshot["error"]:
                    raise RuntimeError(snapshot["error"])
                prediction = snapshot.get("model_prediction")
                if prediction and "score" in prediction:
                    break
                time.sleep(.5)
            else:
                raise RuntimeError("No nonempty packet model observation before deadline")
            identity = prediction["observation_id"]
            evidence_request = {"kind": "live", "observation_id": identity}
            additional = request("/api/xai", evidence_request)
            language = request("/api/assistant/explain", evidence_request)
            assert additional["source"]["observation_id"] == language["evidence"]["source"]["observation_id"] == identity
            exported = request("/api/live/report")
            stopped = request("/api/live/stop", {})
            assert not stopped["running"] and stopped["model_prediction"] is None
            result = {"backend": str(executable), "actual_capture": snapshot,
                      "additional_xai": additional, "actual_local_llm": language,
                      "export": exported, "stop_clears_prediction": True,
                      "research_scope": "Executed passive header capture and model/XAI/language software; traffic has no incident ground truth and no live-accuracy claim."}
            (out / "frozen-capture.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps({"packets": snapshot["metrics"]["packets_observed"], "rule_status": snapshot["assessment"]["status"],
                              "model_prediction": prediction["label"], "llm_text": language["text"],
                              "llm_seconds": language["elapsed_seconds"], "proof": str(out / "frozen-capture.json")}), flush=True)
        finally:
            child.stdin.close()
            if child.wait(timeout=15) != 0:
                raise RuntimeError("Frozen capture backend did not exit cleanly")


if __name__ == "__main__":
    main()
