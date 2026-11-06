"""Offline checks for the english_matrix.json summaries in tools/verify_native_app.py; no runtime or model is started."""
from __future__ import annotations

import pytest

from tools.verify_native_app import cache_miss_row, latency_summary, percentile, turn_ok


def turn(first=1.0, total=2.0, device="gpu", **extra):
    return dict({"first_token_seconds": first, "seconds": total, "device": device, "llm_executed": True, "labelled": True, "passed": True}, **extra)


def test_percentile_uses_nearest_rank():
    assert percentile([], 0.95) is None
    assert percentile([3.0], 0.95) == 3.0
    assert percentile([float(v) for v in range(1, 21)], 0.95) == 19.0
    assert percentile([float(v) for v in range(1, 21)], 0.5) == 10.0
    assert percentile([5.0, 1.0, 3.0], 0.5) == 3.0


def test_turn_requires_model_labels_and_first_token_within_limit():
    assert turn_ok(turn())
    assert not turn_ok(turn(first=15.01))
    assert not turn_ok(turn(first=None))
    assert not turn_ok(dict(turn(), llm_executed=False))
    assert not turn_ok(dict(turn(), labelled=False))


def test_latency_summary_reports_first_token_and_total_separately():
    rows = [turn(first=v, total=v + 1) for v in (1.0, 2.0, 3.0, 4.0)] + [turn(first=None, total=60.0, llm_executed=False)]
    summary = latency_summary(rows)
    assert summary["turns"] == 5 and summary["first_token_p50"] == 2.0 and summary["first_token_p95"] == 4.0
    assert summary["total_p95"] == 60.0 and summary["verifier_pass_rate"] == 1.0 and summary["turn_pass_rate"] == 0.8
    assert summary["first_token_missing"] == 1 and summary["passed"] is False
    assert latency_summary([turn(first=v) for v in (1.0, 14.0)])["passed"] is True


def miss(first, evaluated, total, cached, pair="benign/ddos"):
    return {"pair": pair, "first_token_seconds": first, "evaluated": evaluated, "total": total, "cached": cached}


def test_cache_miss_row_flags_partial_hits_and_estimates_full_misses():
    shared = {"benign/ddos": 380, "ddos/scan": 380}
    turns = [miss(10.0, 1000, 1380, 380), miss(12.0, 1200, 1580, 380, "ddos/scan"), miss(2.0, 100, 1500, 1400)]
    row = cache_miss_row(turns, shared)
    assert [t["partial_hit"] for t in row["turns"]] == [False, False, True] and row["excluded_partial_hits"] == 1
    assert row["first_token_p95"] == 12.0 and row["passed"] is True and row["status"] == "measured"
    # The partial-hit exclusion applies to the p95 only (plan): the prompt rate and the full-miss estimate use every turn of the row.
    # Rate = 2300 evaluated tokens / 24 s; estimate = each turn's total prompt tokens / rate; p95 = 1580 / (2300 / 24).
    assert row["prompt_tokens_per_second"] == pytest.approx(2300 / 24)
    estimate = row["different_class_count_estimate"]
    assert estimate["label"] == "estimate, not measured" and estimate["p95_seconds"] == pytest.approx(1580 * 24 / 2300) and estimate["passed"] is False
    assert row["passed_with_estimate"] is False
    # Every turn a partial hit (llama.cpp host prompt cache): the row fails by rule, but the full-miss estimate is still reported.
    flagged = cache_miss_row([miss(7.0, 568, 1328, 760), miss(3.5, 301, 1070, 769)], {"benign/ddos": 742})
    assert flagged["status"] == "failed: every turn is a partial hit" and flagged["passed"] is False
    assert flagged["different_class_count_estimate"]["p95_seconds"] == pytest.approx(1328 * 10.5 / 869)


def test_cache_miss_row_fails_when_all_flagged_or_mostly_unknown():
    assert cache_miss_row([miss(2.0, 10, 1500, 1490)], {"benign/ddos": 380})["status"] == "failed: every turn is a partial hit"
    unknown = [miss(9.0, 1000, None, None), miss(9.0, 1000, None, None), miss(9.0, 1000, 1380, 380), miss(9.0, 1000, 1380, 380)]
    row = cache_miss_row(unknown, {"benign/ddos": 380})
    assert [t["cache_unknown"] for t in row["turns"]] == [True, True, False, False]
    assert row["status"] == "failed: unmeasured (cache count unknown for more than a quarter of turns)" and row["passed"] is False
    assert cache_miss_row(unknown[1:], {"benign/ddos": 380})["status"] == "failed: unmeasured (cache count unknown for more than a quarter of turns)"
    assert cache_miss_row(unknown[2:] + unknown[:1] + [miss(9.0, 1000, 1380, 380)] * 2, {"benign/ddos": 380})["status"] == "measured"


def test_latency_summary_reports_time_to_first_verified_sentence():
    rows = [turn(first=1.0, first_sentence_seconds=v) for v in (1.5, 2.5, 3.5)] + [turn(first=1.0, first_sentence_seconds=None)]
    summary = latency_summary(rows)
    assert summary["first_sentence_p50"] == 2.5 and summary["first_sentence_p95"] == 3.5 and summary["first_sentence_missing"] == 1


def test_cache_miss_estimate_labels_its_turns_and_counts_unknown_totals():
    estimate = cache_miss_row([miss(10.0, 1000, 1380, 380), miss(9.0, 900, None, None), miss(2.0, 100, 1500, 1400)], {"benign/ddos": 380})["different_class_count_estimate"]
    assert "every row turn with a known total" in estimate["method"] and "kept" not in estimate["method"]
    assert estimate["turns"] == 2 and estimate["turns_without_total"] == 1 and len(estimate["seconds"]) == 2


def test_runtime_env_keeps_the_shipped_cache_cap_unless_a_row_sets_it(monkeypatch):
    from tools.verify_native_app import runtime_env
    monkeypatch.setenv("NETSEC_LLM_CACHE_RAM", "512")
    monkeypatch.setenv("NETSEC_DESKTOP", "1")
    env = runtime_env("token", {"NETSEC_LLM_DEVICE": "cpu"})
    assert env["IMDAI_SESSION_TOKEN"] == "token" and env["NETSEC_LLM_DEVICE"] == "cpu"
    assert "NETSEC_LLM_CACHE_RAM" not in env and "NETSEC_DESKTOP" not in env, "an inherited cache override would change the shipped cap"
    assert runtime_env("token", {"NETSEC_LLM_CACHE_RAM": "0"})["NETSEC_LLM_CACHE_RAM"] == "0"


def test_cache_miss_row_is_measured_with_the_host_prompt_cache_off(monkeypatch):
    """Task 7 ruling 1: the cache-miss row runs in its own CPU runtime with NETSEC_LLM_CACHE_RAM=0, after the in-order runtime has
    exited (one llama-server at a time); the interleaved turns under the shipped cap are information only."""
    import tools.verify_native_app as app
    started, checks = [], []

    class Runtime:
        def __init__(self, root, binary, evidence, name="runtime", env=None):
            assert all(r.closed for r in started), "one runtime (and so one llama-server) at a time"
            self.name, self.env, self.closed = name, dict(env or {}), False
            started.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.closed = True

        def json(self, path, body=None, timeout=30):
            return {"samples": [{"id": "s"}]} if path.endswith("/samples") else {"id": "job"}

        def wait_job(self, job_id, timeout=300):
            return {"id": job_id}, None, None

    class Evidence:
        def check(self, name, passed, **details):
            checks.append((name, bool(passed)))
            return bool(passed)

        def artifact(self, name, data):
            self.saved = (name, data)

    def turns(runtime, chosen):
        cached = 760 if runtime.env.get("NETSEC_LLM_CACHE_RAM") != "0" else 380  # the host cache restores each context's own prompt
        return [miss(3.0 if cached > 380 else 12.0, 400 if cached > 380 else 1000, 1380, cached)], {"benign/ddos": 380}, ["log"]

    monkeypatch.setattr(app, "NativeRuntime", Runtime)
    monkeypatch.setattr(app, "llama_servers", lambda: [])
    monkeypatch.setattr(app, "gpu_state", lambda: {"available": True, "contended": False})
    monkeypatch.setattr(app, "verify_english", lambda runtime, *a: {"device": "cpu", "runtime": runtime.name})
    monkeypatch.setattr(app, "wait_language_verified", lambda runtime, timeout=300: {"integrity": "verified"})
    monkeypatch.setattr(app, "matrix_samples", lambda runtime, samples: {"benign": {}, "ddos": {}})
    monkeypatch.setattr(app, "cache_miss_turns", turns)
    matrix, evidence = {}, Evidence()
    app.verify_english_cpu("bin", evidence, matrix)
    assert [r.env for r in started] == [{"NETSEC_LLM_DEVICE": "cpu"}, {"NETSEC_LLM_DEVICE": "cpu", "NETSEC_LLM_CACHE_RAM": "0"}]
    assert all(r.closed for r in started) and matrix["cpu"]["runtime"] == started[0].name
    row = matrix["cpu_cache_miss"]
    assert row["cache_ram_mib"] == 0 and row["status"] == "measured" and row["excluded_partial_hits"] == 0 and row["first_token_p95"] == 12.0
    info = matrix["cpu_interleaved_shipped_cap"]
    assert info["label"] == "information only" and info["first_token_p95"] == 3.0 and info["cache_ram_mib"] == 1024
    assert ("CPU cache-miss row: different-sample first token p95 <= 15 s (partial hits excluded)", True) in checks
    assert not [name for name, _ in checks if "shipped" in name or "information" in name], "the shipped-cap interleaved p95 is never a pass/fail check"
    assert evidence.saved[0] == "english_matrix.json"


def test_information_only_interleaved_failure_does_not_block_the_cache_miss_row(monkeypatch):
    """T7 fix-1 review: the shipped-cap interleaved p95 is information only, so an error there must not abort the required
    cache-miss row or lose english_matrix.json; the error is recorded in the information row instead."""
    import tools.verify_native_app as app
    started, checks = [], []

    class Runtime:
        def __init__(self, root, binary, evidence, name="runtime", env=None):
            self.name, self.env = name, dict(env or {})
            started.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def json(self, path, body=None, timeout=30):
            return {"samples": [{"id": "s"}]} if path.endswith("/samples") else {"id": "job"}

        def wait_job(self, job_id, timeout=300):
            return {"id": job_id}, None, None

    class Evidence:
        def check(self, name, passed, **details):
            checks.append((name, bool(passed)))
            return bool(passed)

        def artifact(self, name, data):
            self.saved = (name, data)

    def turns(runtime, chosen):
        if runtime.env.get("NETSEC_LLM_CACHE_RAM") != "0":
            raise TimeoutError("chat turn did not finish within 150 s")
        return [miss(12.0, 1000, 1380, 380)], {"benign/ddos": 380}, ["log"]

    monkeypatch.setattr(app, "NativeRuntime", Runtime)
    monkeypatch.setattr(app, "llama_servers", lambda: [])
    monkeypatch.setattr(app, "gpu_state", lambda: {"available": True, "contended": False})
    monkeypatch.setattr(app, "verify_english", lambda runtime, *a: {"device": "cpu"})
    monkeypatch.setattr(app, "wait_language_verified", lambda runtime, timeout=300: {"integrity": "verified"})
    monkeypatch.setattr(app, "matrix_samples", lambda runtime, samples: {"benign": {}, "ddos": {}})
    monkeypatch.setattr(app, "cache_miss_turns", turns)
    matrix, evidence = {}, Evidence()
    app.verify_english_cpu("bin", evidence, matrix)
    info = matrix["cpu_interleaved_shipped_cap"]
    assert info["label"] == "information only" and "150 s" in info["error"]
    assert matrix["cpu_cache_miss"]["status"] == "measured" and len(started) == 2
    assert ("CPU cache-miss row: different-sample first token p95 <= 15 s (partial hits excluded)", True) in checks
    assert evidence.saved[0] == "english_matrix.json"


class _ChatRuntime:
    """A fake runtime for one finished chat turn per context, and stored explanations per method (no model is started)."""

    def __init__(self, sentences, failing=()):
        self.sentences, self.failing, self.calls = sentences, set(failing), []

    def json(self, path, body=None, timeout=30):
        self.calls.append((path, body))
        if path == "/api/v1/jobs":
            return {"id": "job"}
        if path == "/api/v1/explain":
            if body["method"] in self.failing:
                raise RuntimeError("explain failed: " + body["method"])
            return {"explanation_id": f"{body['method']}-{body['row_index']}", "method": body["method"]}
        if path == "/api/v1/chat/context":
            return {"id": f"ctx-{body['explanation_id']}", "xai_method": body["explanation_id"].split("-")[0]}
        if path == "/api/v1/chat":
            return {}
        return {"revision": 0, "busy": False, "messages": [{"reply": {"status": "completed", "sentences": self.sentences, "llm_executed": True,
                "device": "gpu", "grounding": {"kind": "initial", "passed": True, "method": "lime"}}}]}

    def wait_job(self, job_id, timeout=300):
        return {"id": job_id}, None, None


def test_chat_turn_accepts_only_the_fixed_netsec_footer_as_netsec_text():
    """Owner redesign 3 Nov: the one fixed footer is NetSec text; any other netsec-labelled sentence is mislabelled."""
    from tools.verify_native_app import FOOTER, chat_turn
    assert FOOTER == "XAI shows how the model reached its answer, not whether it is correct."
    ok = [{"text": "A.", "source": "model", "part": "shows"}, {"text": "B.", "source": "template", "part": "why"},
          {"text": FOOTER, "source": "netsec", "part": "footer"}]
    turn = chat_turn(_ChatRuntime(ok), "c", {"initial": True})
    assert turn["labelled"] is True and turn["sentences"][2] == {"text": FOOTER, "source": "netsec", "part": "footer"}
    assert turn["method"] == "lime"
    for bad in ([{"text": "Model text.", "source": "netsec", "part": "footer"}], [{"text": FOOTER, "source": "model", "part": "why"}],
                [ok[0], {"text": FOOTER, "source": "netsec", "part": "footer"}, {"text": FOOTER, "source": "netsec", "part": "footer"}]):
        assert chat_turn(_ChatRuntime(bad), "c", {"initial": True})["labelled"] is False, bad


def test_method_matrix_runs_one_initial_turn_per_offline_method_and_sample():
    """Each sample gets one initial explanation per XAI method whose explanation can be generated offline; a method that
    cannot be generated is recorded with its error and no chat turn."""
    from tools.verify_native_app import METHODS, method_matrix
    assert METHODS == ["shap", "lime", "anchors", "counterfactual", "permutation", "ale"]
    from tools.verify_native_app import FOOTER
    runtime = _ChatRuntime([{"text": "A.", "source": "model", "part": "shows"}, {"text": FOOTER, "source": "netsec", "part": "footer"}], failing={"ale"})
    chosen = {"benign": {"row": 1, "label": "benign", "source_row_id": "b"}, "ddos": {"row": 2, "label": "ddos_icmp-flood", "source_row_id": "d"}}
    rows = method_matrix(runtime, chosen)
    assert [(r["sample"], r["method"]) for r in rows] == [(s, m) for s in chosen for m in METHODS]
    failed = [r for r in rows if r["method"] == "ale"]
    assert all(r["xai_error"] and "status" not in r for r in failed)
    done = [r for r in rows if r["method"] != "ale"]
    assert all(r["requested_kind"] == "initial" and r["labelled"] and r["xai_method"] == r["method"] and r["source_row_id"] for r in done)
    contexts = [b for p, b in runtime.calls if p == "/api/v1/chat/context"]
    assert len(contexts) == 10 and all(b["explanation_id"].startswith(b["explanation_id"].split("-")[0]) for b in contexts)


def test_served_bundle_must_label_netsec_text_and_require_a_selected_xai():
    """Redesign round 1 review, finding 7: a sealed UI built before the redesign labels the NetSec footer as Model text."""
    from tools.verify_native_app import CHOOSE_XAI, NETSEC_LABEL, bundle_labels
    current = f"const a='{NETSEC_LABEL}';function b(s){{if(s.source==='netsec')return a;}}const c='{CHOOSE_XAI}';"
    assert bundle_labels(current) == {"passed": True, "netsec_label": True, "choose_xai": True}
    old = "function b(s){if(s.source!=='template')return'Model';return'NetSec template'}"
    assert bundle_labels(old) == {"passed": False, "netsec_label": False, "choose_xai": False}
    assert not bundle_labels(f"const a='{NETSEC_LABEL}';")["passed"]
