"""Verify the actual local UI without starting capture or language generation.

Uses the project's optional Playwright installation and installed Microsoft Edge.
Screenshots are viewport captures; output is separate from research results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

from playwright.sync_api import sync_playwright


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8502")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    origin = urlsplit(base)
    if origin.scheme != "http" or origin.hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("Verification requires a local HTTP app URL")
    out = Path(__file__).resolve().parents[1] / "artifacts" / "ui_redesign"
    out.mkdir(parents=True, exist_ok=True)
    result = {"passed": False, "base_url": base,
              "scope": "Actual local CPU CSV inference and UI; no live capture or LLM generation",
              "browser": "Installed Microsoft Edge via Playwright", "checks": [],
              "page_errors": [], "external_requests": [], "forbidden_actions": [],
              "viewports": [], "screenshots": []}

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        result["checks"].append(name)

    def api(path):
        with urlopen(base + path, timeout=15) as response:
            return json.load(response)

    def no_overflow(page, name):
        dimensions = page.evaluate("""() => ({viewport: innerWidth,
            width: document.documentElement.scrollWidth,
            offenders: [...document.querySelectorAll('body *')].filter(e => {
              const s = getComputedStyle(e), r = e.getBoundingClientRect();
              return s.display !== 'none' && s.visibility !== 'hidden' && r.width > 0
                && (r.left < -1 || r.right > innerWidth + 1);
            }).slice(0, 12).map(e => ({tag: e.tagName, id: e.id, class: e.className,
              right: e.getBoundingClientRect().right}))})""")
        result["viewports"].append({"name": name, **dimensions})
        check(f"{name}: no horizontal document overflow", dimensions["width"] <= dimensions["viewport"] + 1)

    def screenshot(page, name):
        path = out / f"{name}.png"
        page.screenshot(path=str(path), full_page=False, animations="disabled")
        result["screenshots"].append(str(path))

    try:
        initial = api("/api/live/status")
        assistant = api("/api/assistant/status")
        check("Monitoring is stopped before opening the page", not initial["running"] and initial["model_prediction"] is None)
        check("Local language model is not automatically loaded", not assistant.get("loaded", False) and not assistant.get("automatic_generation", False))
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="msedge", headless=True)
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 1000},
                                        device_scale_factor=1, accept_downloads=True)
                page.set_default_timeout(20000)
                page.on("pageerror", lambda error: result["page_errors"].append(str(error)))

                def record_request(request):
                    parsed = urlsplit(request.url)
                    if parsed.scheme not in {"blob", "data"} and (parsed.scheme, parsed.netloc) != (origin.scheme, origin.netloc):
                        result["external_requests"].append(request.url)
                    if request.method == "POST" and parsed.path in {"/api/live/start", "/api/assistant/explain"}:
                        result["forbidden_actions"].append(parsed.path)

                page.on("request", record_request)
                def open_workspace(route):
                    response = page.goto(base + route, wait_until="domcontentloaded")
                    check(f"{route}: page is directly addressable", response.status == 200)
                    page.wait_for_function("() => document.documentElement.dataset.modelState === 'ready'")
                    page.wait_for_function("() => !document.getElementById('sample-select').disabled")
                    page.evaluate("document.fonts.ready")

                sizes = [("desktop", 1440, 1000), ("compact-1100", 1100, 900),
                         ("compact-1024", 1024, 900), ("tablet", 768, 1024),
                         ("mobile", 375, 812), ("landscape", 812, 375)]
                response = page.goto(base, wait_until="networkidle")
                check("Home loads separately with NetSec branding", response.status == 200 and "NetSec" in page.title())
                check("Home does not load analysis controls or call analysis APIs", page.locator("#analyze-button").count() == 0 and page.locator('script[src="/static/app.js"]').count() == 0)
                for name, width, height in sizes:
                    page.set_viewport_size({"width": width, "height": height})
                    page.wait_for_timeout(100)
                    no_overflow(page, "home-" + name)
                    check(f"Home {name}: all three destinations are visible", page.locator(".route-card").count() == 3 and all(page.locator(f'.route-card[href="{r}"]').is_visible() for r in ["/live", "/csv", "/samples"]))
                    if name in {"desktop", "mobile"}:
                        screenshot(page, "home-" + name)
                page.set_viewport_size({"width": 1440, "height": 1000})
                cards = page.locator(".route-card").evaluate_all("cards => cards.map(c => c.getBoundingClientRect().height)")
                check("Home cards have matching heights", max(cards) - min(cards) < 2)

                for route, mode in [("/live", "live"), ("/csv", "upload"), ("/samples", "sample")]:
                    open_workspace(route)
                    check(f"{route}: exactly its own source is visible", all(page.locator(f"#{m}-panel").is_visible() == (m == mode) for m in ["live", "sample", "upload"]))
                    check(f"{route}: both XAI and LLM sections are present", page.locator("#extra-xai-title").is_visible() and page.locator("#assistant-title").is_visible())
                    check(f"{route}: NetSec title links home", page.locator('.studio-header .brand').get_attribute("href") == "/" and "NetSec" in page.locator('.studio-header .brand').inner_text())
                    page.reload(wait_until="domcontentloaded")
                    page.wait_for_function("() => document.documentElement.dataset.modelState === 'ready'")
                    check(f"{route}: reload retains the selected workspace", page.locator("body").get_attribute("data-workspace-mode") == mode and page.locator(f"#{mode}-panel").is_visible())
                    for name, width, height in sizes:
                        page.set_viewport_size({"width": width, "height": height})
                        page.evaluate("scrollTo({top: 0, behavior: 'instant'})")
                        page.wait_for_timeout(100)
                        no_overflow(page, route[1:] + "-" + name)
                        check(f"{route} {name}: all navigation destinations remain visible", all(page.locator(f'.studio-nav a[href="{r}"]').is_visible() for r in ["/", "/live", "/csv", "/samples"]))
                        if name in {"desktop", "mobile"}:
                            screenshot(page, route[1:] + "-" + name)
                    page.set_viewport_size({"width": 1440, "height": 1000})
                    report_selector = "#live-report-panel" if mode == "live" else "#report-panel"
                    layout = page.evaluate("""selector => {
                        const a = document.querySelector('.input-panel').getBoundingClientRect();
                        const b = document.querySelector(selector).getBoundingClientRect();
                        const cards = [...document.querySelectorAll('.insight-card')].map(c => c.getBoundingClientRect());
                        return {aligned: Math.abs(a.left-b.left)<2 && Math.abs(a.right-b.right)<2,
                          gap: b.top-a.bottom, equal: Math.abs(cards[0].top-cards[1].top)<2 && Math.abs(cards[0].bottom-cards[1].bottom)<2};
                    }""", report_selector)
                    check(f"{route}: source/result panels align with consistent spacing", layout["aligned"] and 18 <= layout["gap"] <= 22)
                    check(f"{route}: XAI and LLM cards have matching bounds", layout["equal"])
                    page.locator('.studio-header .brand').click()
                    page.wait_for_url(base + "/")
                    check(f"{route}: clicking NetSec returns home", page.locator("#home-title").is_visible())
                check("Saved CPU model is ready", api("/api/status")["execution"]["device"] == "cpu")
                check("Opening all pages keeps monitoring stopped", not api("/api/live/status")["running"])
                open_workspace("/samples")
                sample_id = page.locator("#sample-select option").evaluate_all("options => options.find(o => o.value !== '').value")
                page.locator("#sample-select").select_option(sample_id)
                page.locator("#analyze-button").click()
                page.locator("#report-content .shap-row").first.wait_for()
                check("Real sample prediction and signed TreeSHAP are rendered", page.locator(".prediction-label").count() == 1 and page.locator("#report-content .shap-row").count() > 0)
                check("Sample comparison label is retained", page.locator(".ground-truth").count() == 1)
                page.locator("#report-tab-1").click()
                check("Assessment explicitly retains offline-template reasoning", "No LLM inference" in page.locator(".assessment-label").inner_text())
                page.locator("#report-tab-2").click()
                check("All 170 feature contributions are available", page.locator("#report-view-2 .feature-table tbody tr").count() == 170)
                page.locator("#report-tab-0").focus()
                for key, selected in [("ArrowRight", 1), ("ArrowRight", 2), ("ArrowLeft", 1), ("Home", 0), ("End", 2)]:
                    page.keyboard.press(key)
                    page.wait_for_function("id => document.activeElement.id === id && document.getElementById(id).getAttribute('aria-selected') === 'true'", arg=f"report-tab-{selected}")
                    check(f"Keyboard {key} selects report tab {selected}", page.locator(f"#report-view-{selected}").is_visible())
                page.locator("#report-tab-0").click()
                check("Model analysis enables XAI and preserves optional LLM control", page.locator("#extra-xai-button").is_enabled() and page.locator("#assistant-button").is_visible())
                no_overflow(page, "desktop-analysis")
                screenshot(page, "desktop-analysis")
                with page.expect_download() as download:
                    page.locator("#download-report").click()
                report_path = out / "sample-report.json"
                download.value.save_as(str(report_path))
                report = json.loads(report_path.read_text(encoding="utf-8"))
                check("JSON export retains exact SHAP additivity and source provenance", report["explanation"]["reconstruction_error"] < 1e-4 and len(report["explanation"]["all_features"]) == 170 and bool(report["source"]["validation_label"]))
                check("JSON report truthfully identifies no default LLM inference", report["reasoning"]["llm_executed"] is False)
                page.locator("#extra-xai-button").click()
                page.locator("#extra-xai-content").wait_for(state="visible", timeout=90000)
                check("On-request XAI renders actual LIME evidence", page.locator(".lime-row").count() > 0)
                page.locator("#extra-xai-tab-1").click()
                check("XAI statistics retain all 170 native model features", page.locator("#extra-xai-view-1 tbody tr").count() == 170)
                page.locator("#extra-xai-tab-0").click()
                page.set_viewport_size({"width": 375, "height": 812})
                page.locator("#report-panel").scroll_into_view_if_needed()
                no_overflow(page, "mobile-analysis")
                screenshot(page, "mobile-analysis")

                page.set_viewport_size({"width": 1440, "height": 1000})
                open_workspace("/csv")
                page.locator("#upload-panel").wait_for(state="visible")
                check("Changing source clears the previous prediction", page.locator("#report-content").is_hidden())
                with urlopen(base + "/api/example.csv", timeout=15) as response:
                    example = response.read()
                page.locator("#csv-file").set_input_files({"name": "examples.csv", "mimeType": "text/csv", "buffer": example})
                page.locator("#upload-button").click()
                page.locator(".batch-row").first.wait_for()
                check("Example CSV produces eight actual row predictions", page.locator(".batch-row").count() == 8)
                page.locator(".batch-row").nth(1).click()
                page.locator("#report-content .shap-row").first.wait_for()
                check("Uploaded row has its own exact SHAP without sample labels", page.locator(".ground-truth").count() == 0 and page.locator("#report-content .shap-row").count() > 0)
                evidence_before_exit = page.locator("#report-content").inner_text()
                page.locator(".session-exit").click()
                check("Exit opens a themed screen without discarding the selected evidence", page.locator('#lifecycle-screen[data-state="exit"]').is_visible())
                page.keyboard.press("Escape")
                check("Cancel restores the exact workspace and report", page.locator("#lifecycle-screen").is_hidden() and page.locator("#report-content").inner_text() == evidence_before_exit and urlsplit(page.url).path == "/csv")
                page.locator("#csv-file").set_input_files({"name": "invalid.csv", "mimeType": "text/csv", "buffer": b"wrong_column\n1\n"})
                page.locator("#upload-button").click()
                page.locator("#global-error").wait_for(state="visible")
                check("Invalid CSV clears stale prediction and batch rows", page.locator("#report-content").is_hidden() and page.locator(".batch-row").count() == 0)
                open_workspace("/live")
                page.locator("#live-panel").wait_for(state="visible")
                check("Changing to live clears the CSV error", page.locator("#global-error").is_hidden())
                capabilities = api("/api/live/capabilities")
                page.locator("#live-mode").select_option("packets")
                if not capabilities["packet_capture_available"]:
                    check("Missing packet driver blocks Start with setup guidance", page.locator("#live-start").is_disabled() and page.locator("#packet-setup").is_visible())
                    result["packet_driver_check"] = "Missing capture dependency correctly blocked; capture was not requested"
                else:
                    result["packet_driver_check"] = "Capture dependency available; no capture requested"
                page.locator("#live-mode").select_option("connections")

                page.goto(base, wait_until="networkidle")
                page.emulate_media(reduced_motion="reduce")
                page.evaluate("scrollTo({top: 0, behavior: 'instant'})")
                scene = page.locator(".network-scene")
                stage = page.locator(".network-stage")
                if scene.count() and stage.count():
                    before = scene.evaluate("e => getComputedStyle(e).transform")
                    bounds = stage.bounding_box()
                    if bounds:
                        page.mouse.move(bounds["x"] + bounds["width"] - 5, bounds["y"] + 5)
                    page.wait_for_timeout(150)
                    after = scene.evaluate("e => getComputedStyle(e).transform")
                    check("Reduced motion keeps the decorative scene transform stable", before == after)
                    check("Reduced motion disables scene animation", scene.evaluate("e => getComputedStyle(e).animationName === 'none'"))
                else:
                    check("Reduced motion has no animated decorative scene", page.evaluate("!document.getAnimations().some(a => a.playState === 'running')"))
                check("Reduced motion disables smooth document scrolling", page.evaluate("getComputedStyle(document.documentElement).scrollBehavior === 'auto'"))

                # Deterministic UI failures only; inference still uses the real local API.
                pending_status = []
                page.route("**/api/status", lambda route: pending_status.append(route))
                page.goto(base + "/samples", wait_until="domcontentloaded")
                page.locator('#lifecycle-screen[data-state="loading"]').wait_for()
                check("Initialization shows the custom loading screen", page.locator("#lifecycle-title").inner_text() == "Bringing things into focus.")
                screenshot(page, "loading-desktop")
                page.wait_for_timeout(100)
                for pending in pending_status:
                    pending.continue_()
                page.unroute("**/api/status")
                page.locator("#lifecycle-screen").wait_for(state="hidden")
                check("A ready model dismisses loading and enables its workspace", page.locator("#analyze-button").is_enabled())
                page.route("**/api/status", lambda route: route.fulfill(status=503, json={"error": "UI verification: local status service unavailable"}))
                page.reload(wait_until="domcontentloaded")
                page.locator('#lifecycle-screen[data-state="error"]').wait_for()
                check("A model connection failure shows error without a prediction", page.locator("#report-content").is_hidden() and page.locator("#lifecycle-details").is_visible())
                screenshot(page, "error-desktop")
                page.unroute("**/api/status")
                page.locator("#lifecycle-primary").click()
                page.locator("#lifecycle-screen").wait_for(state="hidden")
                check("Try again recovers the real model connection", page.locator("#analyze-button").is_enabled())
                for state in ["loading", "error", "exit"]:
                    page.goto(base + "/" + state, wait_until="networkidle")
                    page.set_viewport_size({"width": 375, "height": 812})
                    check(f"/{state}: custom lifecycle page is visible", page.locator(f'#lifecycle-screen[data-state="{state}"]').is_visible())
                    no_overflow(page, state + "-mobile")
                    screenshot(page, state + "-mobile")
                page.locator("#lifecycle-primary").click()
                page.locator('#lifecycle-screen[data-state="ended"]').wait_for()
                check("Browser exit confirms monitoring stopped and offers home", not api("/api/live/status")["running"] and page.locator("#lifecycle-home").is_visible())
                response = page.goto(base + "/missing-page", wait_until="networkidle")
                check("Missing pages keep 404 and show the custom error screen", response.status == 404 and page.locator('#lifecycle-screen[data-state="error"]').is_visible() and page.locator("#lifecycle-primary").is_hidden())
                check("Monitoring remains stopped after all UI checks", not api("/api/live/status")["running"])
                check("No capture or language generation request was made", not result["forbidden_actions"])
                check("No external network request was made", not result["external_requests"])
                check("No browser runtime errors occurred", not result["page_errors"])
                result["passed"] = True
            except Exception:
                if "page" in locals():
                    screenshot(page, "failure")
                raise
            finally:
                browser.close()
    except Exception as error:
        result["failure"] = str(error)
    (out / "checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": len(result["checks"]),
                      "failure": result.get("failure"), "report": str(out / "checks.json")}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
