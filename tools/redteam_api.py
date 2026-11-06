"""The NetSec runtime's loopback HTTP API, as the live red-team runner (tools/redteam_chat.py) uses it.

Every request path and body field the runner sends is in this one module, so a change to transport.rs/chat.rs request
fields (for example during the XAI-to-English redesign) is adjusted here only. Stdlib only; loopback only.
"""
from __future__ import annotations

import http.client
import json
import time
from urllib.parse import urlsplit

#: Server messages (chat.rs / evidence.rs) for a turn the server refuses to start. Such a turn shows no model text.
REJECTIONS = (
    "choose an xai method",            # evidence::CHOOSE_XAI: no selected XAI, no model call (owner, 3 Nov)
    "initial explanation takes no message",
    "ask a nonempty question",
    "conversation changed",            # stale revision
    "already has an answer in progress",
    "language model is busy",          # HTTP 429 from the language slot
    "twenty-turn limit",
)


class ApiError(RuntimeError):
    def __init__(self, path, status, data):
        self.path, self.status, self.data = path, status, data
        super().__init__(f"{path}: HTTP {status}: {data}")

    @property
    def message(self):
        return self.data.get("error", "") if isinstance(self.data, dict) else str(self.data)


def error_text(data):
    return data.get("error", "") if isinstance(data, dict) else str(data or "")


def is_rejection(status, data):
    """True when the server refused to start a turn for one of its documented reasons (nothing was generated or shown)."""
    return status in (400, 409, 422, 429, 503) and any(r in error_text(data).lower() for r in REJECTIONS)


class ChatAPI:
    """Authenticated client for one runtime: `x-imdai-session` token and the exact loopback Host header (transport.rs `protect`)."""

    def __init__(self, base_url, token, timeout=30):
        parts = urlsplit(base_url)
        if parts.hostname not in ("127.0.0.1", "localhost") or not parts.port:
            raise ValueError("The red-team runner only talks to a loopback runtime URL such as http://127.0.0.1:PORT")
        self.host, self.port, self.token, self.timeout = parts.hostname, parts.port, token, timeout
        self.url = f"http://{self.host}:{self.port}"

    # ------------------------------------------------------------ transport
    def request(self, path, body=None, timeout=None):
        payload = json.dumps(body, allow_nan=False).encode() if body is not None else None
        headers = {"Host": f"{self.host}:{self.port}", "x-imdai-session": self.token}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout or self.timeout)
        try:
            connection.request("POST" if payload is not None else "GET", path, payload, headers)
            response = connection.getresponse()
            raw = response.read()
            kind = response.getheader("Content-Type", "")
            data = json.loads(raw) if "json" in kind and raw else raw.decode("utf-8", errors="replace")
            return response.status, data
        finally:
            connection.close()

    def json(self, path, body=None, timeout=None):
        status, data = self.request(path, body, timeout)
        if status not in (200, 202):
            raise ApiError(path, status, data)
        return data

    # ------------------------------------------------------------ read-only views
    def catalog(self):
        return self.json("/api/v1/catalog")

    def samples(self):
        return self.json("/api/v1/samples")["samples"]

    def settings(self):
        return self.json("/api/v1/settings")

    def wait_language_verified(self, timeout=300):
        """A request before the background model hash pass ends gets the template; wait for it (as verify_native_app does)."""
        deadline = time.monotonic() + timeout
        while True:
            language = self.settings().get("language", {})
            if language.get("integrity") != "verifying" or time.monotonic() > deadline:
                return language
            time.sleep(.25)

    # ------------------------------------------------------------ detector jobs and XAI
    def create_job(self, model_ids, *, sample_id=None, rows=None, csv=None):
        body = {"model_ids": list(model_ids)}
        if sample_id is not None:
            body["sample_id"] = sample_id
        if rows is not None:
            body["rows"] = rows
        if csv is not None:
            body["csv"] = csv
        return self.json("/api/v1/jobs", body)

    def wait_job(self, job_id, timeout=300):
        """Same return shape as verify_native_app.NativeRuntime.wait_job: (view, progress, seconds)."""
        started, view = time.monotonic(), None
        while time.monotonic() - started < timeout:
            view = self.json(f"/api/v1/jobs/{job_id}")
            if view["status"] in ("completed", "failed", "cancelled"):
                return view, [], time.monotonic() - started
            time.sleep(.15)
        raise TimeoutError(f"job {job_id} did not finish: {view}")

    def explain(self, job_id, model_id, row_index, method, timeout=300):
        """Select and generate one XAI method's explanation (the explainer only runs on a selected XAI)."""
        return self.json("/api/v1/explain", {"job_id": job_id, "model_id": model_id, "row_index": row_index, "method": method}, timeout=timeout)

    # ------------------------------------------------------------ chat
    def create_context(self, job_id, model_id, row_index, explanation_id):
        body = {"job_id": job_id, "model_id": model_id, "row_index": row_index}
        if explanation_id is not None:
            body["explanation_id"] = explanation_id
        return self.json("/api/v1/chat/context", body)

    def view(self, context_id):
        return self.json(f"/api/v1/chat/contexts/{context_id}")

    def reset(self, context_id):
        return self.json("/api/v1/chat/reset", {"context_id": context_id})

    def send(self, context_id, revision, message=None, initial=False, include_message=True):
        """Start one turn; returns (status, data) without raising, because a refused start is a result the red-team records."""
        body = {"context_id": context_id, "revision": revision, "initial": bool(initial)}
        if include_message:
            body["message"] = message
        return self.request("/api/v1/chat", body)

    def wait_idle(self, context_id, timeout=150, on_view=None, poll=.2):
        """Poll until the context's turn has finished (`busy` false); `on_view` sees each busy view (pending sentences)."""
        deadline = time.monotonic() + timeout
        while True:
            view = self.view(context_id)
            if not view.get("busy"):
                return view
            if on_view is not None:
                on_view(view)
            if time.monotonic() > deadline:
                raise TimeoutError(f"chat turn on {context_id} did not finish within {timeout} s")
            time.sleep(poll)
