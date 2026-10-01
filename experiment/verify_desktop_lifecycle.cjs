'use strict';
// Fault injection is confined to the verification app's own runtime.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {_electron} = require(process.env.IMDAI_PLAYWRIGHT || 'playwright');
const root = path.resolve(__dirname, '..');
const out = process.env.IMDAI_VERIFICATION_DIR ? path.resolve(process.env.IMDAI_VERIFICATION_DIR) : path.join(root, 'artifacts/ui_redesign');
const proof = {scope: 'Source Electron startup failure, recovery, owned-service failure and exit', checks: [], errors: [], forbidden_actions: []};
fs.mkdirSync(out, {recursive: true});
const check = (name, value) => {assert.ok(value, name); proof.checks.push(name); console.log('PASS: ' + name);};
(async () => {
  const env = {...process.env, IMDAI_PYTHON: path.join(out, 'missing-runtime', 'python.exe')};
  for (const key of ['ELECTRON_RUN_AS_NODE', 'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV']) delete env[key];
  const app = await _electron.launch({executablePath: path.join(root, 'framework/app/desktop/node_modules/electron/dist/electron.exe'),
    args: [path.join(root, 'framework/app/desktop'), '--user-data-dir=' + path.join(out, 'electron-lifecycle-profile')], cwd: root, env, timeout: 90000});
  let page, origin;
  try {
    page = await app.firstWindow();
    page.setDefaultTimeout(30000);
    page.on('pageerror', error => proof.errors.push(error.message));
    page.on('request', request => {if (request.method() === 'POST' && ['/api/live/start', '/api/assistant/explain'].includes(new URL(request.url()).pathname)) proof.forbidden_actions.push(request.url());});
    await page.locator('body[data-state="error"]').waitFor();
    check('Missing runtime shows an in-app error, without exiting', (await page.locator('#title').textContent()).includes('pause') && page.url().startsWith('file:'));
    check('Startup error includes recovery and local diagnostics', await page.locator('#primary').isVisible() && await page.locator('#details').isVisible());
    await page.screenshot({path: path.join(out, 'desktop-startup-error.png'), fullPage: false, animations: 'disabled'});
    await page.locator('[data-action="close"]').click();
    await page.locator('body[data-state="exit"]').waitFor();
    await page.locator('#cancel').click();
    await page.locator('body[data-state="error"]').waitFor();
    check('Canceling exit preserves the startup error and retry control', await page.locator('#primary').isVisible());
    // Recover with the existing project interpreter; no installation or network.
    await app.evaluate((_electron, python) => {process.env.IMDAI_PYTHON = python;}, path.join(root, '.venv/Scripts/python.exe'));
    await page.locator('#primary').click();
    await page.waitForURL(/^http:\/\/127\.0\.0\.1:/, {timeout: 90000});
    await page.locator('.route-card.route-live').waitFor();
    origin = new URL(page.url()).origin;
    check('Retry starts the actual local service and opens Home', (await page.title()).includes('NetSec'));
    check('Recovery leaves capture stopped', await page.evaluate(async () => !(await (await fetch('/api/live/status')).json()).running));
    await app.evaluate(() => {
      const owned = process._getActiveHandles().find(handle => handle.constructor.name === 'ChildProcess' && handle.spawnargs?.includes('framework.app.desktop_backend'));
      assertOwned(owned);
      function assertOwned(value) {if (!value || value.pid === process.pid) throw new Error('The verification-owned backend was not found');}
      owned.kill();
    });
    await page.waitForURL(url => url.protocol === 'file:', {timeout: 30000});
    await page.locator('body[data-state="error"]').waitFor();
    check('An owned runtime crash opens the themed service error', (await page.locator('#detail').textContent()).includes('Previous observations are no longer current'));
    await page.locator('#primary').click();
    await page.waitForURL(/^http:\/\/127\.0\.0\.1:/, {timeout: 90000});
    await page.locator('.route-card.route-live').waitFor();
    const nextOrigin = new URL(page.url()).origin;
    check('Restart creates a fresh workspace after the crash', nextOrigin !== origin);
    origin = nextOrigin;
    await page.locator('.route-card.route-live').click();
    await page.locator('#lifecycle-screen').waitFor({state: 'hidden'});
    check('Recovered live workspace has no stale verdict or automatic monitoring', await page.evaluate(async () => {const s=await (await fetch('/api/live/status')).json(); return !s.running && !s.multiclass_prediction && !s.model_prediction;}));
    await page.locator('[data-window-action="close"]').click();
    await page.locator('#lifecycle-screen[data-state="exit"]').waitFor();
    check('Final exit uses the app screen with the themed chrome', await page.locator('.window-chrome').isVisible());
    check('Fault/recovery flow has no renderer errors', !proof.errors.length);
    check('Fault/recovery flow never requests capture or LLM generation', !proof.forbidden_actions.length);
    proof.passed = true;
  } catch (error) {
    proof.failure = error.message;
    if (page) await page.screenshot({path: path.join(out, 'desktop-lifecycle-failure.png'), fullPage: false}).catch(() => {});
    throw error;
  } finally {
    const closed = app.waitForEvent('close', {timeout: 20000}).catch(() => {});
    if (page) await page.evaluate(async () => {await window.netsecDesktop.requestClose(); await window.netsecDesktop.confirmClose();}).catch(() => {});
    await closed;
    await app.close().catch(() => {});
    fs.writeFileSync(path.join(out, 'desktop-lifecycle-checks.json'), JSON.stringify(proof, null, 2));
  }
  let stopped = false;
  try {await fetch(origin + '/api/status', {signal: AbortSignal.timeout(2000)});} catch {stopped = true;}
  check('Confirmed exit stops the verification-owned backend', stopped);
  fs.writeFileSync(path.join(out, 'desktop-lifecycle-checks.json'), JSON.stringify(proof, null, 2));
})().catch(error => {console.error(error); process.exitCode = 1;});
