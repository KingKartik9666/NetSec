"""Exercise the actual owned native runtime, without training or packet capture.

Python is only the HTTP/browser test driver. Prediction and language generation
must execute through netsec-runtime and its owned native children. All failure
fixtures are isolated; original checkpoint, sample and reference bytes are read
only. A private, hash-identical executable copy avoids locking a concurrent build.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import traceback

import psutil

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "rust/netsec/target/release/netsec-runtime.exe"
PHASES = ("http", "models", "xai", "english", "browser", "partial", "lifecycle", "live77")
# Owner decision 3 Nov 2026: the SHA-pinned live77 bundle (bundle.json + model.ubj); live/models.rs loads it lazily on the
# first scored packet window, so the verifier checks it offline against the Rust pins and never starts a capture.
LIVE77 = Path("artifacts/application/live_models_v1/live77_xgboost")
LIVE77_DECISIONS = Path("artifacts/app_v0_6_20261102/live77_decisions.json")  # owner decisions: label, note, model id


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Evidence:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.data = {"started_utc": datetime.now(timezone.utc).isoformat(), "checks": [],
                     "runtime": "native Rust; Python is only the verification driver",
                     "training_executed": False, "test_arrays_opened": False,
                     "packet_capture_started": False}

    def check(self, name, passed, **details):
        item = dict(name=name, passed=bool(passed), **details)
        self.data["checks"].append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        self.save()
        return bool(passed)

    def save(self):
        (self.folder / "report.json").write_text(json.dumps(self.data, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def artifact(self, name, data):
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return str(path)


def llama_servers():
    """Every llama-server.exe on the machine (the one-server rule and the orphan check read this)."""
    found = []
    for process in psutil.process_iter(["pid", "name", "ppid", "create_time"]):
        if (process.info["name"] or "").lower() == "llama-server.exe":
            found.append(dict(process.info))
    return found


def runtime_env(token, extra):
    """The runtime's environment: the session token and this runtime's own settings. An inherited NETSEC_LLM_CACHE_RAM is dropped,
    so every row measures the shipped --cache-ram cap unless the row sets it (the cache-miss row sets 0; Task 7 ruling 1)."""
    env = dict(os.environ, IMDAI_SESSION_TOKEN=token, **extra)
    env.pop("NETSEC_DESKTOP", None)
    if "NETSEC_LLM_CACHE_RAM" not in extra:
        env.pop("NETSEC_LLM_CACHE_RAM", None)
    return env


class NativeRuntime:
    def __init__(self, root, binary, evidence, name="runtime", env=None):
        self.root, self.binary, self.evidence, self.name = Path(root), Path(binary), evidence, name
        self.token = secrets.token_hex(32)
        self.process = None
        self.descendants = {}
        self.monitor_stop = threading.Event()
        self.extra_env = dict(env or {})

    def start(self, timeout=60):
        env = runtime_env(self.token, self.extra_env)
        self.preexisting_llama = {p["pid"] for p in llama_servers()}
        self.stderr = (self.evidence.folder / f"{self.name}.stderr.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen([str(self.binary), "--root", str(self.root), "--port", "0", "--parent-stdin"],
            cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
            text=True, encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        events = queue.Queue()

        def read_stdout():
            for line in self.process.stdout:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if value.get("type") == "ready":
                    events.put(value)

        self.reader = threading.Thread(target=read_stdout, daemon=True)
        self.reader.start()
        started = time.monotonic()
        while time.monotonic() - started < timeout:
            if self.process.poll() is not None:
                self.stderr.flush()
                raise RuntimeError(f"Native startup exited {self.process.returncode}; inspect {self.name}.stderr.log")
            try:
                ready = events.get(timeout=0.1)
                self.port = ready["port"]
                self.url = f"http://127.0.0.1:{self.port}"
                self.evidence.check(f"{self.name}: native readiness", ready.get("backend") == "rust" and ready["pid"] == self.process.pid,
                    readiness=ready, startup_seconds=time.monotonic() - started, executable_sha256=digest(self.binary))
                self.monitor = threading.Thread(target=self._watch_children, daemon=True)
                self.monitor.start()
                return self
            except queue.Empty:
                pass
        raise TimeoutError("Native runtime produced no readiness record before its startup deadline")

    def _watch_children(self):
        while not self.monitor_stop.wait(0.03):
            try:
                for child in psutil.Process(self.process.pid).children(recursive=True):
                    self.descendants[child.pid] = {"pid": child.pid, "created": child.create_time(), "name": child.name()}
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return

    def request(self, path, body=None, *, method=None, auth=True, headers=None, timeout=30, raw=None):
        payload = json.dumps(body, allow_nan=False).encode() if body is not None else raw
        fields = {"Host": f"127.0.0.1:{self.port}"}
        if auth:
            fields["x-imdai-session"] = self.token
        if payload is not None:
            fields["Content-Type"] = "application/json"
        fields.update(headers or {})
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            connection.request(method or ("POST" if payload is not None else "GET"), path, payload, fields)
            response = connection.getresponse()
            data, response_headers = response.read(), dict(response.getheaders())
            content = response_headers.get("content-type", response_headers.get("Content-Type", ""))
            value = json.loads(data) if "application/json" in content else data.decode("utf-8", errors="replace")
            return response.status, response_headers, value
        finally:
            connection.close()

    def json(self, path, body=None, timeout=30):
        status, _, data = self.request(path, body, timeout=timeout)
        if status not in (200, 202):
            raise RuntimeError(f"{path}: HTTP {status}: {data}")
        return data

    def wait_job(self, job_id, timeout=300):
        started, progress, view = time.monotonic(), [], None
        while time.monotonic() - started < timeout:
            view = self.json(f"/api/v1/jobs/{job_id}")
            count = view["progress"]["completed"]
            if not progress or progress[-1] != count:
                progress.append(count)
            if view["status"] in ("completed", "failed", "cancelled"):
                return view, progress, time.monotonic() - started
            time.sleep(0.15)
        raise TimeoutError(f"Native job did not finish: {view}")

    def close(self, expected_exit_codes=(0,)):
        if self.process is None:
            return
        started = time.monotonic()
        if self.process.poll() is None:
            self.process.stdin.close()  # Real parent EOF, not a simulated shutdown.
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.evidence.check(f"{self.name}: parent EOF shutdown", False, reason="Runtime exceeded 30-second shutdown deadline")
                for child in psutil.Process(self.process.pid).children(recursive=True):
                    child.kill()
                self.process.kill()
                self.process.wait(timeout=10)
        self.monitor_stop.set()
        if hasattr(self, "monitor"):
            self.monitor.join(timeout=1)
        survivors = []
        # Windows can retire an owned conhost just after the application handle
        # signals. Give the OS a bounded grace period before calling it orphaned.
        grace = time.monotonic() + 2
        while True:
            survivors = []
            for info in self.descendants.values():
                try:
                    process = psutil.Process(info["pid"])
                    if process.create_time() == info["created"] and process.is_running():
                        survivors.append(info)
                except psutil.NoSuchProcess:
                    pass
            if not survivors or time.monotonic() >= grace:
                break
            time.sleep(0.05)
        # The language server is the one child that holds VRAM and a loopback port: it must die with the runtime (Review Focus 5).
        llama = [info for info in self.descendants.values() if info["name"].lower() == "llama-server.exe"]
        leftover = [p for p in llama_servers() if p["pid"] not in getattr(self, "preexisting_llama", set()) and p["ppid"] == self.process.pid]
        llama_orphans = [info for info in survivors if info["name"].lower() == "llama-server.exe"]
        llama_orphans += [p for p in leftover if p["pid"] not in {info["pid"] for info in llama_orphans}]
        self.evidence.check(f"{self.name}: no llama-server.exe outlives the runtime", not llama_orphans,
            llama_children=llama, orphaned_llama_servers=llama_orphans)
        for info in survivors + [p for p in leftover if p["pid"] not in {s["pid"] for s in survivors}]:
            try:
                psutil.Process(info["pid"]).kill()  # Only an exact child identity created by this test.
            except psutil.NoSuchProcess:
                pass
        label = "parent EOF shutdown" if expected_exit_codes == (0,) else "expected forced termination"
        self.evidence.check(f"{self.name}: {label} and no orphans", self.process.returncode in expected_exit_codes and not survivors,
            exit_code=self.process.returncode, shutdown_seconds=time.monotonic()-started,
            owned_children=list(self.descendants.values()), orphaned_children=survivors)
        self.stderr.close()

    def __enter__(self):
        try:
            return self.start()
        except Exception:
            self.close()
            raise

    def __exit__(self, *_):
        self.close()


def verify_http(runtime, evidence):
    denied = [
        ("missing session", {"auth": False}),
        ("wrong session", {"headers": {"x-imdai-session": "0"*64}}),
        ("foreign host", {"headers": {"Host": "foreign.invalid"}}),
        ("foreign origin", {"headers": {"Origin": "https://foreign.invalid"}}),
        ("cross-site fetch", {"headers": {"sec-fetch-site": "cross-site"}}),
    ]
    for name, options in denied:
        status, headers, body = runtime.request("/api/v1/catalog", **options)
        evidence.check(name+" refused", status == 403 and isinstance(body, dict), status=status, body=body)
    status, headers, _ = runtime.request(f"/bootstrap?session={runtime.token}", auth=False)
    cookie = headers.get("set-cookie", headers.get("Set-Cookie", ""))
    evidence.check("session bootstrap cookie", status == 303 and "HttpOnly" in cookie and "SameSite=Strict" in cookie,
        status=status, cookie_flags={"HttpOnly":"HttpOnly" in cookie,"SameSite=Strict":"SameSite=Strict" in cookie})
    status, _, _ = runtime.request("/api/v1/catalog", auth=False, headers={"Cookie":cookie.split(";")[0]})
    evidence.check("HttpOnly session authenticates browser requests", status == 200, status=status)
    status, headers, html = runtime.request("/")
    evidence.check("themed page and security headers", status == 200 and "NetSec" in html and
        "frame-ancestors 'none'" in headers.get("content-security-policy", "") and headers.get("x-content-type-options") == "nosniff",
        status=status, content_type=headers.get("content-type"), cache_control=headers.get("cache-control"))
    script = re.search(r'<script[^>]+src="([^"]+)"', html)
    evidence.check("built React entrypoint is served", bool(script) and script.group(1).startswith('/static/'))
    _, _, bundle = runtime.request(script.group(1)) if script else (0, {}, "")
    labels = bundle_labels(bundle if isinstance(bundle, str) else "")
    evidence.check("served UI labels the footer as NetSec text and requires a selected XAI", labels["passed"], **labels)
    for path, expected in [(script.group(1) if script else '/missing-entry.js',200),("/static/native.css",200),("/static/fonts/InterVariable.woff2",200),
                           ("/missing-page",404),("/static/missing.js",404),("/static/%2e%2e/%2e%2e/Cargo.toml",404)]:
        status, _, _ = runtime.request(path)
        evidence.check("static/route "+path, status == expected, status=status)
    status, headers, body = runtime.request("/api/v1/jobs", raw=b"{bad json", method="POST")
    evidence.check("malformed JSON has JSON error response", status == 400 and isinstance(body,dict) and "application/json" in headers.get("content-type",""),status=status)
    status, _, body = runtime.request("/api/v1/jobs", raw=b"{}", method="POST", headers={"Content-Type":"text/plain"})
    evidence.check("wrong request content type refused", status == 415 and isinstance(body,dict), status=status)
    catalog, samples = runtime.json("/api/v1/catalog"), runtime.json("/api/v1/samples")["samples"]
    names = catalog["schema"]["feature_names"]
    invalid = [
        ("no input", {"model_ids":["xgboost"]}),
        ("ambiguous inputs", {"model_ids":["xgboost"],"sample_id":samples[0]["id"],"rows":[samples[0]["row"]]}),
        ("unknown model", {"model_ids":["invented-model"],"sample_id":samples[0]["id"]}),
        ("wrong row dimension", {"model_ids":["xgboost"],"rows":[[1]]}),
        ("unmeasured row", {"model_ids":["xgboost"],"rows":[[None]*len(names)]}),
        ("too many rows", {"model_ids":["xgboost"],"rows":[samples[0]["row"]]*501}),
        ("CSV missing features", {"model_ids":["xgboost"],"csv":"unknown\n1\n"}),
        ("CSV duplicate headers", {"model_ids":["xgboost"],"csv":",".join(names+[names[0]])+"\n"}),
    ]
    for name, request in invalid:
        status, _, body = runtime.request("/api/v1/jobs", request)
        evidence.check(name+" rejected before inference", status == 400 and isinstance(body,dict),status=status)
    live = runtime.json("/api/v1/live/status")
    evidence.check("monitoring remains stopped by default", live.get("running") is False and live.get("observation") is None, status=live)
    return catalog, samples


def verify_models(runtime, evidence, catalog, samples):
    started = runtime.json("/api/v1/jobs", {"mode":"extensive","sample_id":samples[0]["id"]})
    result, progress, seconds = runtime.wait_job(started["id"])
    evidence.artifact("all_models_job.json", result)
    expected = {model["id"] for model in catalog["models"]}
    actual = {model["model_id"] for model in result["models"]}
    evidence.check("all ten visible models execute independently", result["status"] == "completed" and len(expected) == 10
        and actual == expected and all(m["status"] == "completed" and len(m["predictions"]) == 1 for m in result["models"]),
        models=[{"id":m["model_id"],"status":m["status"],"error":m["error"],"elapsed_ms":m["elapsed_ms"]} for m in result["models"]],
        progress=progress, elapsed_seconds=seconds)
    for model in result["models"]:
        if model["status"] != "completed":
            continue
        p=model["predictions"][0]
        probabilities=p.get("probabilities")
        valid=p.get("label") in catalog["schema"]["classes"] and isinstance(p.get("class_index"),int)
        if probabilities is not None:
            valid=valid and len(probabilities)==len(catalog["schema"]["classes"]) and all(math.isfinite(v) and 0<=v<=1 for v in probabilities) and abs(sum(probabilities)-1)<1e-5
        evidence.check("authentic prediction shape "+model["model_id"],valid, label=p.get("label"),score_kind=p.get("score_kind"),probabilities_available=probabilities is not None)
    names=catalog["schema"]["feature_names"]
    order=list(reversed(range(len(names))))
    row=samples[0]["row"]
    csv=",".join(names[i] for i in order)+",label4\r\n"+",".join("" if row[i] is None else str(row[i]) for i in order)+",not_a_feature\r\n"
    created=runtime.json("/api/v1/jobs",{"model_ids":["xgboost"],"csv":csv})
    csv_result,_,_=runtime.wait_job(created["id"])
    reference=next(m["predictions"][0] for m in result["models"] if m["model_id"]=="xgboost")
    observed=csv_result["models"][0]["predictions"][0]
    evidence.check("CSV matches exact names and excludes extra labels",observed==reference,
        source="Included source-validation observation; no held-out test data",origin=csv_result["input"]["origin"])
    job=runtime.json("/api/v1/jobs",{"model_ids":["xgboost","random_forest","hem"],"rows":[row]*32})
    deadline=time.monotonic()+90
    while time.monotonic()<deadline:
        current=runtime.json("/api/v1/jobs/"+job["id"])
        if any(m["status"]=="completed" for m in current["models"]):
            break
        time.sleep(0.05)
    completed={m["model_id"]:m["predictions"] for m in current["models"] if m["status"]=="completed"}
    try:
        runtime.json(f'/api/v1/jobs/{job["id"]}/cancel',{})
        evidence.check("cancel request returns its acknowledged state",True)
    except OSError as error:
        # A failed response is still a product failure; inspect the independent
        # job state rather than abandoning subsequent verification phases.
        evidence.check("cancel request returns its acknowledged state",False,error=str(error))
    time.sleep(0.5)
    final=runtime.json("/api/v1/jobs/"+job["id"])
    evidence.check("cancel preserves already completed model results",final["status"]=="cancelled" and bool(completed)
        and all(next(m for m in final["models"] if m["model_id"]==key)["predictions"]==value for key,value in completed.items())
        and not any(m["status"] in ("queued","running") for m in final["models"]),
        completed_before_cancel=list(completed),statuses={m["model_id"]:m["status"] for m in final["models"]})
    evidence.artifact("cancelled_job.json",final)
    return result


def verify_xai(runtime,evidence,job):
    explanations={}
    for method in ("shap","lime","anchors","counterfactual","permutation","ale"):
        before=time.monotonic()
        status,_,result=runtime.request("/api/v1/explain",{"job_id":job["id"],"model_id":"xgboost","method":method},timeout=180)
        evidence.artifact(f"xai/{method}.json",result)
        valid=status==200 and result.get("method")==method and result.get("job_id")==job["id"] and result.get("model_id")=="xgboost"
        if status==200:
            valid=valid and isinstance(result.get("features"),list) and isinstance(result.get("quality"),dict) and bool(result.get("explanation_id"))
            evaluations=result["quality"].get("evaluations",0)
            valid=valid and evaluations<=256
            explanations[method]=result
        evidence.check("actual native XAI "+method,valid,status=status,elapsed_seconds=time.monotonic()-before,
            method_label=result.get("method_label"),scope=result.get("scope"),state=result.get("details",{}).get("state"),
            quality=result.get("quality"),error=result.get("error"))
    # Generic Shapley explains the selected final hybrid, not an unlabeled branch.
    for model in ("tabm_ne","tabm_xgboost_kan"):
        status,_,result=runtime.request("/api/v1/explain",{"job_id":job["id"],"model_id":model,"method":"shap"},timeout=180)
        evidence.artifact(f"xai/{model}_shap.json",result)
        evidence.check("final selected model explanation "+model,status==200 and result.get("model_id")==model and result.get("method")=="shap",
            status=status,scope=result.get("scope"),quality=result.get("quality"),error=result.get("error"))
    for request in [dict(job_id="invented",model_id="xgboost",method="shap"),
                    dict(job_id=job["id"],model_id="xgboost",row_index=999,method="shap"),
                    dict(job_id=job["id"],model_id="xgboost",method="fake-method")]:
        status,_,_=runtime.request("/api/v1/explain",request)
        evidence.check("invalid XAI binding refused",status==400,status=status,kind="unknown job/row/method")
    return explanations


FIRST_TOKEN_LIMIT=15.0  # seconds from the request to the first token (owner, 31 Oct 2026)
GPU_FREE_MIB=3600  # shared GPU rule: the language server goes on GPU only with this much free VRAM
QUESTIONS=[("initial",{"initial":True}),("simpler",{"message":"Explain it simpler"}),("summary",{"message":"Summarize the scores"}),("detail",{"message":"More detail please"})]
PAIRS=[("benign","ddos"),("ddos","scan"),("scan","unknown"),("unknown","benign")]
FOLLOW_UP={"message":"Which features influenced this result?"}
# Owner redesign 3 Nov 2026: the fixed NetSec footer closes every non-refusal answer, and each XAI method has its own explainer preset.
FOOTER="XAI shows how the model reached its answer, not whether it is correct."
# The served UI must label the footer as NetSec text and refuse to start without a selected XAI (redesign round 1 review, finding 7:
# a bundle built before the redesign labels every non-template sentence, the footer included, as Model text).
NETSEC_LABEL="NetSec text"
CHOOSE_XAI="Choose an XAI method and generate its explanation first; the plain-English explainer only explains a selected XAI result."
def bundle_labels(js):
    found={"netsec_label":NETSEC_LABEL in js,"choose_xai":CHOOSE_XAI in js}
    return {"passed":all(found.values()),**found}
METHODS=["shap","lime","anchors","counterfactual","permutation","ale"]
SHIPPED_CACHE_RAM_MIB=1024  # server.rs CACHE_RAM_MIB: llama-server --cache-ram as shipped (Task 7 ruling 3)
CACHE_MISS_ENV={"NETSEC_LLM_DEVICE":"cpu","NETSEC_LLM_CACHE_RAM":"0"}  # host prompt cache off, so a miss is measured (Task 7 ruling 1)


def percentile(values,q):
    """Nearest-rank percentile of the known values; None when there are none."""
    values=sorted(v for v in values if v is not None)
    return values[max(0,math.ceil(q*len(values))-1)] if values else None


def turn_ok(turn):
    """A model reply whose sentences are all labelled and whose first token arrived within the limit."""
    first=turn.get("first_token_seconds")
    return bool(turn.get("llm_executed") and turn.get("labelled") and first is not None and first<=FIRST_TOKEN_LIMIT)


def latency_summary(turns):
    firsts=[t.get("first_token_seconds") for t in turns if t.get("first_token_seconds") is not None]
    sentences=[t.get("first_sentence_seconds") for t in turns if t.get("first_sentence_seconds") is not None]
    ok=[turn_ok(t) for t in turns];p95=percentile(firsts,.95)
    return {"turns":len(turns),"first_token_p50":percentile(firsts,.5),"first_token_p95":p95,
            "first_sentence_p50":percentile(sentences,.5),"first_sentence_p95":percentile(sentences,.95),"first_sentence_missing":len(turns)-len(sentences),
            "total_p50":percentile([t.get("seconds") for t in turns],.5),"total_p95":percentile([t.get("seconds") for t in turns],.95),
            "first_token_missing":len(turns)-len(firsts),
            "verifier_pass_rate":sum(bool(t.get("passed")) for t in turns)/len(turns) if turns else None,
            "turn_pass_rate":sum(ok)/len(turns) if turns else None,
            "passed":bool(turns) and all(ok) and p95 is not None and p95<=FIRST_TOKEN_LIMIT}


def cache_miss_row(turns,shared):
    """CPU cache-miss row (plan Task 7): flags partial hits (cached > the pair's shared prefix S) and unknown cache counts,
    reports first-token p95 over the kept turns, and the labelled different-class_count estimate (whole prompt re-evaluated)."""
    rows=[]
    for t in turns:
        s,cached=shared.get(t["pair"]),t.get("cached")
        rows.append(dict(t,partial_hit=cached is not None and s is not None and cached>s,cache_unknown=cached is None))
    kept=[r for r in rows if not r["partial_hit"]];unknown=sum(r["cache_unknown"] for r in rows)
    if not rows:status="failed: no turns"
    elif not kept:status="failed: every turn is a partial hit"
    elif unknown*4>len(rows):status="failed: unmeasured (cache count unknown for more than a quarter of turns)"
    else:status="measured"
    p95=percentile([r.get("first_token_seconds") for r in kept],.95)
    # The partial-hit exclusion applies to the p95 only; the rate and the full-miss estimate use every turn of the row.
    timed=[r for r in rows if r.get("evaluated") and r.get("first_token_seconds")]
    # First-token seconds include the HTTP and verifier overhead, so the rate is a slight underestimate and the estimate conservative.
    rate=sum(r["evaluated"] for r in timed)/sum(r["first_token_seconds"] for r in timed) if timed else None
    estimates=[r["total"]/rate for r in rows if rate and r.get("total") is not None]
    estimate_p95=percentile(estimates,.95)
    passed=status=="measured" and p95 is not None and p95<=FIRST_TOKEN_LIMIT and all(r.get("first_token_seconds") is not None for r in kept)
    estimate={"label":"estimate, not measured","method":"total prompt tokens of every row turn with a known total / measured prompt rate (a full cache miss, such as a live77 packet-window turn that shares no cached prefix, re-evaluates the whole prompt)",
              "turns":len(estimates),"turns_without_total":len(rows)-len(estimates),
              "seconds":estimates,"p95_seconds":estimate_p95,"passed":estimate_p95 is not None and estimate_p95<=FIRST_TOKEN_LIMIT}
    return {"status":status,"turns":rows,"kept_turns":len(kept),"excluded_partial_hits":len(rows)-len(kept),"cache_unknown_turns":unknown,
            "shared_prefix_tokens":shared,"first_token_p50":percentile([r.get("first_token_seconds") for r in kept],.5),"first_token_p95":p95,
            "prompt_tokens_per_second":rate,"different_class_count_estimate":estimate,"passed":passed,"passed_with_estimate":passed and estimate["passed"]}


def gpu_state():
    """Free VRAM and load before a language server starts; another GPU job makes the measurement "contended"."""
    try:
        out=subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.total,utilization.gpu","--format=csv,noheader,nounits"],
                           capture_output=True,text=True,timeout=15,check=True).stdout
        used,total,util=(float(v) for v in out.strip().splitlines()[0].split(","))
    except (OSError,ValueError,IndexError,subprocess.SubprocessError) as error:
        return {"available":False,"error":str(error)}
    # Desktop baseline is about 1341-1430 MiB with no compute job (ledger); more than 1900 MiB or a busy GPU means another job runs.
    return {"available":True,"memory_used_mib":used,"memory_total_mib":total,"memory_free_mib":total-used,"utilization_percent":util,
            "contended":used>1900 or util>20,"contention_rule":"memory.used > 1900 MiB or utilization > 20% before the server starts"}


def labelled_ok(sentences):
    """Every sentence is Model or NetSec template text, except the one fixed NetSec footer, which is labelled NetSec text and is last."""
    netsec=[i for i,s in enumerate(sentences) if s.get("source")=="netsec"]
    return bool(sentences) and all(s.get("source") in ("model","template") for i,s in enumerate(sentences) if i not in netsec) and (
        not netsec or (netsec==[len(sentences)-1] and sentences[-1].get("text")==FOOTER)) and not any(
        s.get("text")==FOOTER and s.get("source")!="netsec" for s in sentences)


def chat_turn(runtime,context_id,body,timeout=150):
    """One streamed chat turn, polled to completion; returns the audit fields the matrix needs."""
    revision=runtime.json("/api/v1/chat/contexts/"+context_id)["revision"];started=time.monotonic()
    runtime.json("/api/v1/chat",dict(body,context_id=context_id,revision=revision))
    deadline=started+timeout
    while True:
        view=runtime.json("/api/v1/chat/contexts/"+context_id)
        if not view["busy"]:break
        if time.monotonic()>deadline:raise TimeoutError("chat turn did not finish within %s s"%timeout)
        time.sleep(.1)
    reply=view["messages"][-1]["reply"];grounding=reply.get("grounding",{});tokens=grounding.get("prompt_tokens") or {}
    sentences=reply.get("sentences") or []
    return {"status":reply.get("status"),"kind":grounding.get("kind"),"device":reply.get("device"),"llm_executed":reply.get("llm_executed"),
            "labelled":labelled_ok(sentences),"passed":grounding.get("passed"),"method":grounding.get("method"),
            "first_violation":grounding.get("first_violation"),"first_token_seconds":grounding.get("first_token_seconds"),
            "first_sentence_seconds":grounding.get("first_sentence_seconds"),"seconds":grounding.get("seconds"),"wall_seconds":time.monotonic()-started,
            "model_sentences":grounding.get("model_sentences"),"template_sentences":grounding.get("template_sentences"),
            "prompt_chars":grounding.get("prompt_chars"),"prior_answers":grounding.get("prior_answers"),
            "evaluated":tokens.get("evaluated"),"total":tokens.get("total"),"cached":tokens.get("cached"),"violation":grounding.get("violation"),
            "text":reply.get("text"),"sentences":[{"text":s.get("text"),"source":s.get("source"),"part":s.get("part")} for s in sentences]}


def matrix_samples(runtime,samples):
    """benign, ddos and scan by validation label; the 77-class catalog has no "unknown" class, so "unknown" is the
    validation row the detector is least sure about (lowest top score), never a held-out test row."""
    chosen={"benign":next(s for s in samples if s["label"]=="benign"),
            "ddos":next((s for s in samples if s["label"]=="ddos_icmp-flood"),None) or next(s for s in samples if s["label"].startswith("ddos")),
            "scan":next(s for s in samples if s["label"]=="recon_scan")}
    job=runtime.json("/api/v1/jobs",{"model_ids":["xgboost"],"rows":[s["row"] for s in samples]})
    job,_,_=runtime.wait_job(job["id"])
    scores=[p["score"] for p in job["models"][0]["predictions"]]
    taken={s["source_row_id"] for s in chosen.values()}
    chosen["unknown"]=next(samples[i] for i in sorted(range(len(samples)),key=lambda i:scores[i]) if samples[i]["source_row_id"] not in taken)
    return chosen


def open_contexts(runtime,rows):
    """A fresh xgboost job over `rows` (a new binding, so no earlier chat history), TreeSHAP for each row and its chat context."""
    job=runtime.json("/api/v1/jobs",{"model_ids":["xgboost"],"rows":[s["row"] for s in rows]})
    job,_,_=runtime.wait_job(job["id"])
    contexts=[]
    for index in range(len(rows)):
        binding={"job_id":job["id"],"model_id":"xgboost","row_index":index}
        xai=runtime.json("/api/v1/explain",dict(binding,method="shap"),timeout=180)
        contexts.append(runtime.json("/api/v1/chat/context",dict(binding,explanation_id=xai["explanation_id"])))
    return contexts


def english_matrix(runtime,chosen):
    """The four kinds (initial, simpler, summary, detail) on each of the four samples, in order on one context per sample."""
    rows=[]
    for (name,sample),context in zip(chosen.items(),open_contexts(runtime,list(chosen.values()))):
        for kind,body in QUESTIONS:
            rows.append(dict(chat_turn(runtime,context["id"],body),sample=name,sample_label=sample["label"],source_row_id=sample["source_row_id"],requested_kind=kind))
    return rows


def method_matrix(runtime,chosen,methods=METHODS):
    """Owner redesign 3 Nov 2026 (per-method row): for each sample, one initial explanation per XAI method whose explanation can be
    generated offline, each on its own context pinned to that method's stored explanation. A method that cannot be generated is
    recorded with its error and no chat turn."""
    rows=[]
    job=runtime.json("/api/v1/jobs",{"model_ids":["xgboost"],"rows":[s["row"] for s in chosen.values()]})
    job,_,_=runtime.wait_job(job["id"])
    for index,(name,sample) in enumerate(chosen.items()):
        binding={"job_id":job["id"],"model_id":"xgboost","row_index":index}
        for method in methods:
            row=dict(sample=name,sample_label=sample["label"],source_row_id=sample["source_row_id"],method=method,requested_kind="initial")
            try:xai=runtime.json("/api/v1/explain",dict(binding,method=method),timeout=300)
            except Exception as error:rows.append(dict(row,xai_error=str(error)));continue
            context=runtime.json("/api/v1/chat/context",dict(binding,explanation_id=xai["explanation_id"]))
            rows.append(dict(chat_turn(runtime,context["id"],{"initial":True}),**dict(row,method=method,xai_method=context.get("xai_method"),
                xai_state=(xai.get("details") or {}).get("state"))))
    return rows


def cache_miss_turns(runtime,chosen):
    """Interleaves two contexts on different samples so each follow-up finds the other context's prompt in the one slot,
    carrying two earlier Detail answers. S = cached count of B's initial sent straight after A's initial."""
    measured,shared,log=[],{},[]
    for a,b in PAIRS:
        pair=f"{a}/{b}";ca,cb=open_contexts(runtime,[chosen[a],chosen[b]])
        def step(context,side,role,body):
            t=dict(chat_turn(runtime,context["id"],body),pair=pair,sample=side,role=role);log.append(t);return t
        step(ca,a,"initial",{"initial":True});shared[pair]=step(cb,b,"initial",{"initial":True})["cached"]
        for _ in range(2):
            step(ca,a,"prior detail",QUESTIONS[3][1]);step(cb,b,"prior detail",QUESTIONS[3][1])
        measured.append(step(ca,a,"cache-miss follow-up",FOLLOW_UP));measured.append(step(cb,b,"cache-miss follow-up",FOLLOW_UP))
    return measured,shared,log


def wait_language_verified(runtime,timeout=300):
    """By design a request before the background hash pass ends gets "Template (model unavailable)"; measure only after it."""
    deadline=time.monotonic()+timeout
    while True:
        language=runtime.json("/api/v1/settings")["language"]
        if language.get("integrity")!="verifying" or time.monotonic()>deadline:return language
        time.sleep(.25)


def verify_english(runtime,evidence,job,explanations,device,environment):
    language=wait_language_verified(runtime)
    evidence.check("language model files verified by the background hash pass",language.get("integrity")=="verified",language=language)
    explanation=explanations.get("shap")
    if explanation:
        forged=dict(explanation,explanation_id="0"*64)
        status,_,_=runtime.request("/api/v1/english",{"job_id":job["id"],"model_id":"xgboost","explanation":forged})
        evidence.check("forged English evidence refused",status==400,status=status)
    before=time.monotonic()
    body={"job_id":job["id"],"model_id":"xgboost"}
    if explanation:body["explanation"]=explanation
    status,_,result=runtime.request("/api/v1/english",body,timeout=150)
    evidence.artifact("english.json",result)
    if not (Path(runtime.root)/"artifacts/llm/model.json").is_file():
        # No model installed: the request must fail honestly instead of substituting template text as LLM output.
        evidence.check("absent local language model reported, not substituted",status==503 and result.get("llm_executed") is not True,
            status=status,error=result.get("error"))
        return None
    grounding=result.get("grounding",{}) if isinstance(result,dict) else {}
    sentences=(result.get("sentences") or []) if isinstance(result,dict) else []
    first=grounding.get("first_token_seconds")
    evidence.check("actual local language model generation, labelled sentences, first token within 15 s",status==200 and result.get("llm_executed") is True
        and labelled_ok(sentences) and first is not None and first<=FIRST_TOKEN_LIMIT
        and result.get("inference_device")==device,status=status,device=result.get("device"),expected_device=device,first_token_seconds=first,
        total_seconds=grounding.get("seconds"),wall_seconds=time.monotonic()-before,verifier_passed=grounding.get("passed"),first_violation=grounding.get("first_violation"))
    samples=runtime.json("/api/v1/samples")["samples"]
    rows=english_matrix(runtime,matrix_samples(runtime,samples))
    summary=latency_summary(rows)
    evidence.check(f"english matrix on {device}: 4 samples x 4 kinds, llm_executed, labelled, first token p95 <= 15 s",
        summary["passed"] and all(r["device"]==device for r in rows),devices=sorted({str(r["device"]) for r in rows}),
        contended=environment.get("contended"),**{k:v for k,v in summary.items() if k!="passed"})
    return {"device":device,"environment":environment,"turns":rows,"summary":summary}


def verify_english_cpu(binary,evidence,matrix):
    """CPU fallback rows in their own runtimes after the GPU runtime has exited (one llama-server at a time): the in-order matrix
    with the shipped --cache-ram cap, then the cache-miss row in a second runtime with the host prompt cache off (Task 7 ruling 1)."""
    if llama_servers():
        evidence.check("CPU english matrix: no other llama-server is running",False,running=llama_servers());return
    with NativeRuntime(ROOT,binary,evidence,"cpu-language",env={"NETSEC_LLM_DEVICE":"cpu"}) as runtime:
        environment=gpu_state()
        job=runtime.json("/api/v1/jobs",{"model_ids":["xgboost"],"sample_id":runtime.json("/api/v1/samples")["samples"][0]["id"]})
        job,_,_=runtime.wait_job(job["id"])
        matrix["cpu"]=verify_english(runtime,evidence,job,{},"cpu",environment)
        # Information only: with the shipped cap the host cache restores each context's own prompt, so these turns are partial hits.
        # An error here is recorded, never raised, so it cannot abort the required cache-miss row below (T7 fix-1 review).
        try:
            measured,shared,log=cache_miss_turns(runtime,matrix_samples(runtime,runtime.json("/api/v1/samples")["samples"]))
            info=cache_miss_row(measured,shared)
            matrix["cpu_interleaved_shipped_cap"]={"label":"information only","cache_ram_mib":SHIPPED_CACHE_RAM_MIB,"environment":environment,
                "first_token_p50":percentile([t.get("first_token_seconds") for t in measured],.5),"first_token_p95":percentile([t.get("first_token_seconds") for t in measured],.95),
                "partial_hits":info["excluded_partial_hits"],"first_token_p95_without_partial_hits":info["first_token_p95"],
                **{k:info[k] for k in ("shared_prefix_tokens","turns")},"sequence":log}
        except Exception as error:
            matrix["cpu_interleaved_shipped_cap"]={"label":"information only","cache_ram_mib":SHIPPED_CACHE_RAM_MIB,"environment":environment,
                "error":f"{type(error).__name__}: {error}"}
    if llama_servers():
        evidence.check("CPU cache-miss row: no other llama-server is running",False,running=llama_servers());evidence.artifact("english_matrix.json",matrix);return
    with NativeRuntime(ROOT,binary,evidence,"cpu-cache-miss",env=CACHE_MISS_ENV) as runtime:
        environment=gpu_state();wait_language_verified(runtime)
        chosen=matrix_samples(runtime,runtime.json("/api/v1/samples")["samples"])
        measured,shared,log=cache_miss_turns(runtime,chosen)
        row=cache_miss_row(measured,shared);row["sequence"]=log;row["environment"]=environment;row["cache_ram_mib"]=int(CACHE_MISS_ENV["NETSEC_LLM_CACHE_RAM"])
        row["note"]=("Different-sample interleaved pairs, never pooled with the in-order rows; same-sample follow-ups are not cache misses. "
                     "Measured in its own runtime with NETSEC_LLM_CACHE_RAM=0 (llama-server --cache-ram 0), so no host cache restores a context's prompt.")
        matrix["cpu_cache_miss"]=row
        evidence.check("CPU cache-miss row: different-sample first token p95 <= 15 s (partial hits excluded)",row["passed"],status=row["status"],
            first_token_p95=row["first_token_p95"],kept_turns=row["kept_turns"],excluded_partial_hits=row["excluded_partial_hits"],
            cache_unknown_turns=row["cache_unknown_turns"],shared_prefix_tokens=shared)
        estimate=row["different_class_count_estimate"]
        evidence.check("CPU different class_count full miss <= 15 s (estimate, not measured)",estimate["passed"],p95_seconds=estimate["p95_seconds"],
            prompt_tokens_per_second=row["prompt_tokens_per_second"],label=estimate["label"])
    evidence.artifact("english_matrix.json",matrix)


def failure_fixture(evidence):
    fixture=evidence.folder/"fixtures/missing_kan_weights"
    source=ROOT/"artifacts/application/native_models_v1"
    target=fixture/"artifacts/application/native_models_v1"
    for path in source.rglob("*"):
        if not path.is_file() or path.name.startswith("golden"):
            continue
        relative=path.relative_to(source)
        if relative==Path("kan/weights.bin"):
            continue
        out=target/relative;out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():continue
        if path.suffix in (".bin",".ubj"):
            os.link(path,out)  # Never write a linked file; only native reads.
        else:
            shutil.copy2(path,out)
    # Shared byte validation must pass; only the selected KAN model fails loading.
    from tools.stage_native_app import seal
    paths=[p.relative_to(fixture) for p in fixture.rglob("*") if p.is_file() and p.name!="runtime_assets.json"]
    seal(fixture,paths)
    return fixture


def verify_partial(binary,evidence,sample_id):
    fixture=failure_fixture(evidence)
    with NativeRuntime(fixture,binary,evidence,"missing-model-fixture") as runtime:
        job=runtime.json("/api/v1/jobs",{"model_ids":["xgboost","kan","tabm_hier"],"sample_id":sample_id})
        final,_,_=runtime.wait_job(job["id"])
        evidence.artifact("partial_failure_job.json",final)
        statuses={m["model_id"]:m["status"] for m in final["models"]}
        evidence.check("one missing model preserves other genuine predictions",final["status"]=="completed"
            and statuses=={"xgboost":"completed","kan":"failed","tabm_hier":"completed"},statuses=statuses,
            failure=next(m["error"] for m in final["models"] if m["model_id"]=="kan"),fixture=str(fixture),original_artifacts_changed=False)


def rust_pins(source=ROOT):
    """The live77 constants live/models.rs enforces, the display label live.rs model_label gives the id, and the Live
    page caveat filter (liveView.ts); read from source so this check cannot drift from what the runtime pins."""
    runtime=source/"rust/netsec/crates/runtime/src"
    pins=dict(re.findall(r'^(?:pub )?const (\w+): [^=\n]+= "?([^";\n]*)"?;',(runtime/"live/models.rs").read_text(encoding="utf-8"),re.M))
    pins["label"]=re.search(r'models::LIVE77_MODEL_ID => Ok\("([^"]+)"\)',(runtime/"live.rs").read_text(encoding="utf-8")).group(1)
    pins["caveat"]=re.search(r'export const caveat=/(.+)/i;',(source/"rust/netsec/web/src/liveView.ts").read_text(encoding="utf-8")).group(1)
    return pins


def live77_bundle(root=ROOT,pins=None,source=ROOT):
    """Plan step 9: the live77 bundle's SHA, schema, classes, merge, window, provenance and checkpoint, checked offline
    against the guards live/models.rs load_bundle enforces. Nothing executes: no capture and no inference (the Rust
    loader itself, its golden replay and TreeSHAP run in cargo test live::models). Only shipped bytes are read."""
    result={"bundle":(LIVE77/"bundle.json").as_posix(),"packet_capture_started":False,"model_executed":False,"test_arrays_opened":False,
            "caveat_filter":None,"failures":[],"error":None,"checks":{},
            "rust_loader_coverage":"the Rust load path (load_bundle guards, Booster::load_checkpoint, golden replay, exact TreeSHAP) is covered by "
                                   "cargo test -p netsec-runtime live::models; it is not executed by this phase (live77 loads lazily on a captured window)"}
    try:
        pins=pins or rust_pins(source);caveat=result["caveat_filter"]=re.compile(pins["caveat"],re.I)
        path=root/LIVE77/"bundle.json";raw=path.read_bytes();b=json.loads(raw)
        names,classes,merge=b.get("feature_names") or [],b.get("classes") or [],b.get("class_merge") or {}
        joined=lambda values:hashlib.sha256("\n".join(values).encode()).hexdigest()
        catalog=json.loads((source/"artifacts/application/native_models_v1/catalog.json").read_text(encoding="utf-8"))["classes"]
        spec_path=source/"artifacts/training/class_merge_20261101/spec.json";spec=json.loads(spec_path.read_text(encoding="utf-8"))
        temperature=(b.get("calibration") or {}).get("temperature");window=int(pins["WINDOW_SECONDS"])
        checkpoint=(path.parent/str(b.get("checkpoint") or "")).resolve()
        inside=checkpoint.parent==path.parent.resolve() and checkpoint.is_file()
        label,note=str(b.get("label") or ""),str(b.get("provenance_note") or "")
        decisions_path=source/LIVE77_DECISIONS;decided=json.loads(decisions_path.read_text(encoding="utf-8"))["defaults_chosen_by_app_chat"]
        checks=[("bundle at most 1 MiB",len(raw)<=1024*1024),
            ("bundle SHA256 equals the Rust pin",hashlib.sha256(raw).hexdigest()==pins["LIVE77_BUNDLE_SHA"]),
            ("model id, kind and objective",b.get("model_id")==pins["LIVE77_MODEL_ID"] and b.get("kind")=="xgboost" and b.get("objective")=="multi:softprob"),
            ("252 feature names in the pinned order",len(names)==252==b.get("n_features") and joined(names)==pins["LIVE77_NAMES_SHA"]),
            ("77 classes in the pinned order, equal to the CSV catalog",len(classes)==77==b.get("n_classes") and joined(classes)==pins["LIVE77_CLASSES_SHA"] and classes==catalog),
            ("benign index 0, recon_scan 70, recon_ping-sweep 71",b.get("benign_index")==0 and classes[:1]==["benign"] and classes[70:72]==["recon_scan","recon_ping-sweep"]),
            ("class merge ping_sweep_into_recon_scan v1, 71 -> 70, pinned spec",merge.get("name")=="ping_sweep_into_recon_scan" and merge.get("version")==1
                and (merge.get("source_class"),merge.get("source_index"),merge.get("target_class"),merge.get("target_index"))==("recon_ping-sweep",71,"recon_scan",70)
                and merge.get("spec_sha256")==pins["CLASS_MERGE_SPEC_SHA"]==digest(spec_path) and (spec.get("source_index"),spec.get("target_index"))==(71,70)),
            ("10 s window",window==10 and b.get("window_seconds")==window and (b.get("window") or {}).get("length_seconds")==window),
            ("test_evaluated and live_validation false",b.get("test_evaluated") is False and b.get("live_validation") is False),
            ("temperature calibration, no logit adjustment, no binary attack probability",(b.get("calibration") or {}).get("kind")=="temperature"
                and isinstance(temperature,(int,float)) and math.isfinite(temperature) and temperature>0 and b.get("logit_adjustment") is None
                and (b.get("output") or {}).get("binary_attack_probability") is None),
            ("checkpoint inside the bundle folder; bytes and SHA256 match the bundle and the Rust pin",inside and checkpoint.stat().st_size==b.get("bytes")
                and digest(checkpoint)==b.get("checkpoint_sha256")==pins["LIVE77_CHECKPOINT_SHA"]),
            ("label equals the runtime label; label and provenance note survive the Live caveat filter",label==pins["label"] and bool(note.strip())
                and not caveat.search(label) and not caveat.search(note)),
            ("label and provenance note equal the owner's decisions file (defaults_chosen_by_app_chat)",(label,note)==(decided.get("label"),decided.get("provenance_note"))
                and b.get("model_id")==decided.get("model_id")),
            ("bundle pins the current owner decisions file (sha256)",(b.get("source") or {}).get("decisions_sha256")==digest(decisions_path))]
        result["checks"]={name:bool(ok) for name,ok in checks};result["failures"]=[name for name,ok in checks if not ok]
        result.update(bundle_sha256=hashlib.sha256(raw).hexdigest(),checkpoint_sha256=b.get("checkpoint_sha256"),features=len(names),classes=len(classes),
            merge=f'{merge.get("source_class")} ({merge.get("source_index")}) -> {merge.get("target_class")} ({merge.get("target_index")})',
            window_seconds=b.get("window_seconds"),label=label,provenance_note=note)
    except (OSError,ValueError,KeyError,TypeError,AttributeError,IndexError) as error:
        result["error"]=f"{type(error).__name__}: {error}"
    result["passed"]=result["error"] is None and bool(result["checks"]) and not result["failures"]
    return result


def verify_live77(evidence,root=ROOT):
    """Offline phase: no runtime, capture or inference starts; the stopped-by-default live check stays in the http phase."""
    result=live77_bundle(root);passed=result.pop("passed")
    result["caveat_filter"]=result["caveat_filter"].pattern if result["caveat_filter"] else None
    return evidence.check("live77 bundle SHA, classes and merge load against the Rust pins (offline; no capture)",passed,**result)


def verify_browser(runtime,evidence):
    from playwright.sync_api import sync_playwright, expect
    with sync_playwright() as p:
        browser=p.chromium.launch(channel="msedge",headless=True)
        context=browser.new_context(viewport={"width":1440,"height":1100},device_scale_factor=1)
        page=context.new_page();errors=[]
        page.on("pageerror",lambda error:errors.append(str(error)))
        page.goto(runtime.url+"/bootstrap?session="+runtime.token,wait_until="networkidle")
        page.screenshot(path=str(evidence.folder/"browser_home_1440.png"),full_page=True)
        evidence.check("browser renders themed home",page.locator("#home-view").is_visible() and not errors,
            javascript_errors=errors.copy(),background=page.locator("body").evaluate("el=>getComputedStyle(el).backgroundColor"))
        page.goto(runtime.url+"/models",wait_until="networkidle")
        expect(page.locator('#sample-select')).to_be_enabled()
        expect(page.locator('.model-choice')).to_have_count(10)
        evidence.check("browser catalog exposes ten real models",page.locator(".model-choice").count()==10,count=page.locator(".model-choice").count())
        page.locator("#run-analysis").click()
        expect(page.locator('#job-status')).to_have_text('COMPLETED',timeout=180000)
        page.locator("#explain-model").click()
        expect(page.locator('#xai-content')).to_contain_text('TreeSHAP',timeout=180000)
        page.screenshot(path=str(evidence.folder/"browser_analysis_1440.png"),full_page=True)
        evidence.check("browser actual prediction and explanation",page.locator(".model-result-title").count()>0
            and page.locator(".xai-feature-list li").count()>0 and not errors,javascript_errors=errors.copy(),result=page.locator(".model-result-title").first.inner_text())
        desktop_overflow=page.evaluate("document.documentElement.scrollWidth>innerWidth")
        page.set_viewport_size({"width":390,"height":844})
        page.screenshot(path=str(evidence.folder/"browser_analysis_390.png"),full_page=True)
        mobile_overflow=page.evaluate("document.documentElement.scrollWidth>innerWidth")
        evidence.check("browser desktop/mobile have no horizontal overflow",not desktop_overflow and not mobile_overflow,
            desktop_overflow=desktop_overflow,mobile_overflow=mobile_overflow)
        # Native shell dispatches this real lifecycle event after backend EOF.
        page.evaluate("window.dispatchEvent(new CustomEvent('netsec-runtime-failed',{detail:{reason:'Verification of owned backend exit'}}))")
        evidence.check("themed backend-exit recovery is visible",page.locator("#connection-error").is_visible()
            and "restart" in page.locator("#connection-message").inner_text().lower())
        page.screenshot(path=str(evidence.folder/"browser_recovery_390.png"),full_page=True)
        context.close();browser.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases",default=",".join(PHASES))
    parser.add_argument("--binary",type=Path,default=RELEASE)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    phases=set(args.phases.split(","))
    if phases-set(PHASES):parser.error("Unknown verification phase")
    folder=args.output or ROOT/"artifacts/app_migration_20261024/native_verification"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence=Evidence(folder)
    private=evidence.folder/"bin/netsec-runtime.exe"
    if phases-{"live77"}:  # the offline live77 phase never runs the executable, so it needs no private copy
        private.parent.mkdir(exist_ok=True);shutil.copy2(args.binary,private)
        evidence.check("private test executable matches actual release",digest(args.binary)==digest(private),source=str(args.binary),sha256=digest(private))
    catalog=samples=job=None;explanations={};matrix={};env={};device="gpu";environment={}
    if "english" in phases:
        # Shared GPU rules: at most one llama-server machine-wide, and GPU only with enough free VRAM (otherwise CPU, recorded).
        if llama_servers():
            evidence.check("english: no other llama-server is running (one-server rule)",False,running=llama_servers());phases-={"english"}
        else:
            environment=gpu_state()
            if not (environment.get("available") and environment["memory_free_mib"]>=GPU_FREE_MIB):
                device="cpu";env={"NETSEC_LLM_DEVICE":"cpu"}
                evidence.check("GPU english matrix measured (needs >= %d MiB free VRAM)"%GPU_FREE_MIB,False,environment=environment)
    try:
        if "live77" in phases:verify_live77(evidence)
        if phases-{"live77"}:
            with NativeRuntime(ROOT,private,evidence,env=env) as runtime:
                catalog=runtime.json("/api/v1/catalog");samples=runtime.json("/api/v1/samples")["samples"]
                if "http" in phases:verify_http(runtime,evidence)
                if phases & {"models","xai","english"}:
                    if "models" in phases:job=verify_models(runtime,evidence,catalog,samples)
                    else:
                        created=runtime.json("/api/v1/jobs",{"mode":"extensive","sample_id":samples[0]["id"]})
                        job,_,_=runtime.wait_job(created["id"])
                    if "xai" in phases:explanations=verify_xai(runtime,evidence,job)
                    if "english" in phases:matrix[device]=verify_english(runtime,evidence,job,explanations,device,environment)
                if "browser" in phases:verify_browser(runtime,evidence)
            if "partial" in phases:verify_partial(private,evidence,samples[0]["id"])
            if "english" in phases and device=="gpu":verify_english_cpu(private,evidence,matrix)
            elif matrix:evidence.artifact("english_matrix.json",matrix)
    except Exception as error:
        evidence.check("verification driver completed",False,error=str(error),traceback=traceback.format_exc())
    evidence.data["finished_utc"]=datetime.now(timezone.utc).isoformat()
    evidence.data["passed"]=all(item["passed"] for item in evidence.data["checks"])
    evidence.save()
    print(json.dumps({"passed":evidence.data["passed"],"checks":len(evidence.data["checks"]),"report":str(evidence.folder/"report.json")}),flush=True)
    return 0 if evidence.data["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
