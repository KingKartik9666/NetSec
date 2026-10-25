"""Verify the actual Electron process and owned Rust service on Windows."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

import psutil
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "framework/app/desktop/node_modules/electron/dist/electron.exe")
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts/application/package_root")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/app_electron_react_20261024/desktop_verification")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checks = []

    def check(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))
        (args.output / "report.json").write_text(json.dumps({"checks": checks}, indent=2), encoding="utf-8")
        if not passed:
            raise AssertionError(name)

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = dict(os.environ, NETSEC_ROOT=str(args.root.resolve()))
    env.pop('ELECTRON_RUN_AS_NODE', None)
    command = [str(args.binary.resolve()), f'--remote-debugging-port={port}', '--remote-debugging-address=127.0.0.1', f'--user-data-dir={args.output.resolve() / "profile"}']
    if args.binary.name.lower() == 'electron.exe':
        command.append(str(ROOT / 'framework/app/desktop'))
    process = subprocess.Popen(command, cwd=ROOT, env=env, creationflags=subprocess.CREATE_NO_WINDOW,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    descendants = []
    user = ctypes.windll.user32
    user.IsIconic.argtypes = [wintypes.HWND]
    user.IsZoomed.argtypes = [wintypes.HWND]
    user.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    try:
        with sync_playwright() as playwright:
            browser = None
            deadline = time.monotonic() + 45
            while browser is None and time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Electron exited early: {process.returncode}")
                try:
                    browser = playwright.chromium.connect_over_cdp(f"http://127.0.0.1:{port}", timeout=1500)
                except Exception:
                    time.sleep(.2)
            check("actual Electron Chromium debugging connection", browser is not None)
            page = browser.contexts[0].pages[0]
            page.wait_for_url("http://127.0.0.1:*/", timeout=45000)
            page.wait_for_selector("body.desktop-app", timeout=20000)
            check("authenticated native home rendered", "NetSec" in page.title() and "session=" not in page.url, url=page.url)
            window = ctypes.c_void_p()
            callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

            def find(hwnd, _):
                owner = wintypes.DWORD()
                user.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
                if owner.value == process.pid and user.IsWindowVisible(hwnd):
                    window.value = hwnd
                return True

            user.EnumWindows(callback_type(find), 0)
            check("native Windows handle belongs to Electron process", bool(window.value))
            page.locator("#window-minimize").click()
            page.wait_for_timeout(300)
            check("minimize changes actual Windows state", user.IsIconic(window) != 0)
            user.ShowWindow(window, 9)
            page.locator("#window-maximize").click()
            page.wait_for_timeout(400)
            check("maximize changes actual Windows state", user.IsZoomed(window) != 0)
            page.locator("#window-maximize").click()
            page.wait_for_timeout(300)
            check("restore changes actual Windows state", user.IsZoomed(window) == 0)
            page.locator("#window-close").click()
            page.wait_for_selector("#exit-dialog[open]")
            page.locator("#exit-cancel").click()
            expect(page.locator("#exit-dialog")).not_to_be_visible()
            check("cancel exit preserves owned runtime", process.poll() is None and not page.locator("#exit-dialog").is_visible())
            page.goto(page.url.rstrip("/") + "/models")
            expect(page.locator("#model-catalog input")).to_have_count(10)
            check("all ten models appear in native renderer", page.locator("#model-catalog input").count() == 10)
            page.locator('input[name="analysis-mode"][value="extensive"]').check()
            page.locator("#run-analysis").click()
            expect(page.locator('#job-status')).to_have_text('COMPLETED', timeout=90000)
            page.screenshot(path=str(args.output / "native_models_1440.png"), full_page=True)
            check("native extensive job completes", page.locator('.result-item[data-status="completed"]').count() == 10)
            page.locator('#explain-model').click()
            expect(page.locator('#xai-content')).to_contain_text('SHAP', timeout=180000)
            page.screenshot(path=str(args.output / 'electron_explanation.png'), full_page=True)
            check('React renders actual native explanation', page.locator('.xai-feature-list li').count() > 0)
            check('renderer has no Node or raw IPC', page.evaluate("typeof require === 'undefined' && typeof process === 'undefined' && typeof window.netsecDesktop === 'object'"))
            descendants = psutil.Process(process.pid).children(recursive=True)
            names = [p.name().lower() for p in descendants if p.is_running()]
            check("production process tree has native Rust and no Python or Tauri", "netsec-runtime.exe" in names and not any("python" in n or "tauri" in n for n in names), processes=names)
            old = page.url
            page.evaluate("location.href='https://example.invalid/'")
            page.wait_for_timeout(300)
            check("external top-level navigation is refused", page.url == old)
            # Chromium reports a provisional navigation to CDP even though the
            # Electron will-navigate handler refused it. Clear that test state.
            page.context.new_cdp_session(page).send("Page.stopLoading")
            page.goto(old, wait_until="load")
            expect(page.locator('#window-close')).to_be_visible()
            page.locator("#window-close").click()
            page.wait_for_selector("#exit-dialog[open]")
            page.locator("#exit-confirm").click()
            process.wait(timeout=10)
            check("confirmed close exits native shell", process.returncode == 0)
    except Exception as error:
        checks.append(dict(name="desktop verification completed", passed=False, error=str(error)))
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(p.is_running() for p in descendants):
            time.sleep(.1)
        alive = [p.pid for p in descendants if p.is_running()]
        checks.append(dict(name="owned descendant processes exit", passed=not alive, remaining=alive))
        report = dict(passed=all(c["passed"] for c in checks), checks=checks, root=str(args.root), binary=str(args.binary))
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"passed": report["passed"], "checks": len(checks), "report": str(args.output / "report.json")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
