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
PHASES = ("http", "models", "xai", "english", "browser", "partial", "lifecycle")


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


class NativeRuntime:
    def __init__(self, root, binary, evidence, name="runtime"):
        self.root, self.binary, self.evidence, self.name = Path(root), Path(binary), evidence, name
        self.token = secrets.token_hex(32)
        self.process = None
        self.descendants = {}
        self.monitor_stop = threading.Event()

    def start(self, timeout=60):
        env = dict(os.environ, IMDAI_SESSION_TOKEN=self.token)
        env.pop("NETSEC_DESKTOP", None)
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
        for info in survivors:
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


def verify_english(runtime,evidence,job,explanations):
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
    evidence.check("actual local Qwen generation and grounding",status==200 and result.get("llm_executed") is True
        and result.get("grounding",{}).get("passed") is True and bool(result.get("text")) and result.get("inference_device")=="cpu",
        status=status,elapsed_seconds=time.monotonic()-before,provider=result.get("provider"),attempts=result.get("attempts"),error=result.get("error"))


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
    private=evidence.folder/"bin/netsec-runtime.exe";private.parent.mkdir(exist_ok=True)
    shutil.copy2(args.binary,private)
    evidence.check("private test executable matches actual release",digest(args.binary)==digest(private),source=str(args.binary),sha256=digest(private))
    catalog=samples=job=None;explanations={}
    try:
        with NativeRuntime(ROOT,private,evidence) as runtime:
            catalog=runtime.json("/api/v1/catalog");samples=runtime.json("/api/v1/samples")["samples"]
            if "http" in phases:verify_http(runtime,evidence)
            if phases & {"models","xai","english"}:
                if "models" in phases:job=verify_models(runtime,evidence,catalog,samples)
                else:
                    created=runtime.json("/api/v1/jobs",{"mode":"extensive","sample_id":samples[0]["id"]})
                    job,_,_=runtime.wait_job(created["id"])
                if "xai" in phases:explanations=verify_xai(runtime,evidence,job)
                if "english" in phases:verify_english(runtime,evidence,job,explanations)
            if "browser" in phases:verify_browser(runtime,evidence)
        if "partial" in phases:verify_partial(private,evidence,samples[0]["id"])
    except Exception as error:
        evidence.check("verification driver completed",False,error=str(error),traceback=traceback.format_exc())
    evidence.data["finished_utc"]=datetime.now(timezone.utc).isoformat()
    evidence.data["passed"]=all(item["passed"] for item in evidence.data["checks"])
    evidence.save()
    print(json.dumps({"passed":evidence.data["passed"],"checks":len(evidence.data["checks"]),"report":str(evidence.folder/"report.json")}),flush=True)
    return 0 if evidence.data["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
