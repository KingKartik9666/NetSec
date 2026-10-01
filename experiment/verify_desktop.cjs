'use strict';
// Executed desktop proof. Set IMDAI_PLAYWRIGHT to the installed verification package.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {_electron} = require(process.env.IMDAI_PLAYWRIGHT || 'playwright');
const root = path.resolve(__dirname, '..');
const out = process.env.IMDAI_VERIFICATION_DIR ? path.resolve(process.env.IMDAI_VERIFICATION_DIR) : path.join(root, 'artifacts/desktop/verification');
fs.mkdirSync(out, {recursive: true});
(async () => {
  const packaged = process.argv.includes('--packaged');
  const full = process.argv.includes('--full');
  const env = {...process.env, IMDAI_PYTHON: packaged ? 'X:/NO-PYTHON-INSTALL/python.exe' : path.join(root, '.venv/Scripts/python.exe')};
  for (const key of ['ELECTRON_RUN_AS_NODE', 'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV']) delete env[key];
  if (packaged) env.PATH = path.join(process.env.WINDIR || 'C:/Windows', 'System32');
  const installDirectory = process.env.IMDAI_INSTALL_DIR || path.join(process.env.LOCALAPPDATA || path.join(require('node:os').homedir(), 'AppData/Local'), 'Programs/NetSec');
  const executablePath = process.env.IMDAI_DESKTOP_EXE || (packaged ? path.join(installDirectory, 'NetSec.exe') : path.join(root, 'framework/app/desktop/node_modules/electron/dist/electron.exe'));
  // Isolate verification from other desktop sessions and their single-instance lock.
  const profileArgument = '--user-data-dir=' + path.join(out, 'electron-profile');
  const app = await _electron.launch({executablePath, args: packaged ? [profileArgument] : [path.join(root, 'framework/app/desktop'), profileArgument], cwd: root, env, timeout: 90000});
  const proof = {packaged, full, executable: executablePath, checks: [], errors: [], live_accuracy_measured: false};
  app.process().stderr.on('data', chunk => {proof.main_diagnostics = ((proof.main_diagnostics || '') + chunk.toString()).slice(-6000);});
  const check = (name, value) => {assert.ok(value, name); proof.checks.push(name); console.log('PASS: ' + name);};
  let origin;
  try {
    const page = await app.firstWindow({timeout: 90000});
    page.setDefaultTimeout(full ? 120000 : 30000);
    page.on('pageerror', e => proof.errors.push(e.message));
    const observedReports = new Map();
    page.on('response', async response => {
      if (new URL(response.url()).pathname !== '/api/live/status' || !response.ok()) return;
      const snapshot = await response.json().catch(() => null);
      if (snapshot?.multiclass_prediction?.observation_id) {
        observedReports.set(snapshot.multiclass_prediction.observation_id, snapshot.multiclass_prediction);
        if (observedReports.size > 20) observedReports.delete(observedReports.keys().next().value);
      }
    });
    await page.waitForURL(/^http:\/\/127\.0\.0\.1:/, {timeout: 90000});
    origin = new URL(page.url()).origin;
    await page.locator('.route-card.route-live').waitFor();
    check('NetSec opens the home page with three workspace links', new URL(page.url()).pathname === '/' && await page.locator('.route-card').count() === 3 && (await page.title()).includes('NetSec'));
    check('Home is navigation, without an automatic analysis', await page.locator('#analyze-button, #live-start, #assistant-button').count() === 0);
    check('Requested header labels and ready badge are removed', await page.locator('.topbar, .breadcrumb, #connection-status').count() === 0);
    const prefs = await app.evaluate(({BrowserWindow}) => {
      const window = BrowserWindow.getAllWindows()[0];
      const p = window.webContents.getLastWebPreferences();
      return {sandbox: p.sandbox, contextIsolation: p.contextIsolation, nodeIntegration: p.nodeIntegration};
    });
    check('Renderer sandbox and isolation enabled', prefs.sandbox && prefs.contextIsolation && !prefs.nodeIntegration);
    const ambient = await page.evaluate(() => ({require: typeof require, process: typeof process}));
    check('Renderer cannot access Node or process', ambient.require === 'undefined' && ambient.process === 'undefined');
    check('File/View application menu is removed', await app.evaluate(({Menu}) => Menu.getApplicationMenu() === null));
    check('Themed native controls are integrated into the app', await page.locator('.window-chrome').isVisible() && await page.locator('.window-actions button').count() === 3);
    check('Preload exposes only fixed window actions', await page.evaluate(() => Object.keys(window.netsecDesktop).sort().join(',') === ['minimize','toggleMaximize','requestClose','confirmClose','cancelClose','retry','getState','onState'].sort().join(',')));
    check('Title bar supports dragging and controls remain clickable', await page.evaluate(() => getComputedStyle(document.querySelector('.window-chrome')).webkitAppRegion === 'drag' && getComputedStyle(document.querySelector('.window-actions')).webkitAppRegion === 'no-drag'));
    await page.locator('[data-window-action="maximize"]').click();
    check('Maximize control changes the native window', await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].isMaximized()));
    await page.getByRole('button', {name: 'Restore window', exact: true}).click();
    check('Restore control returns the native window', !await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].isMaximized()));
    await page.locator('[data-window-action="minimize"]').click();
    check('Minimize control changes the native window', await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].isMinimized()));
    await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].restore());
    const rejected = await fetch(origin + '/api/status');
    check('Separate clients cannot access desktop analysis service', rejected.status === 403);
    await page.screenshot({path: path.join(out, packaged ? 'packaged-home.png' : 'desktop-home.png'), fullPage: true, animations: 'disabled'});
    await page.locator('.route-card.route-live').click();
    await page.waitForURL(url => url.pathname === '/live');
    await page.waitForFunction(() => document.documentElement.dataset.modelState === 'ready');
    check('Live workspace opens through home navigation with saved CPU inference', await page.locator('#live-panel').isVisible());
    const openWorkspace = async (route, panel) => {
      await page.locator(`.studio-nav a[href="${route}"]`).click();
      await page.waitForURL(url => url.pathname === route);
      await page.waitForFunction(() => document.documentElement.dataset.modelState === 'ready');
      await page.locator(panel).waitFor({state: 'visible'});
    };
    proof.model_status = await page.evaluate(async () => (await fetch('/api/status')).json());
    check('Compatibility metadata uses the canonical 170-column schema', proof.model_status.feature_compatibility.length === 170 && proof.model_status.feature_compatibility.filter(feature => feature.live_mapping).length === 7);
    const provider = await page.evaluate(async () => (await fetch('/api/assistant/status')).json());
    check('Tiny offline model available but not loaded', provider.available && !provider.loaded && provider.cpu_threads === 2);
    check('Local knowledge corpus is available without automatic generation', provider.rag.available && provider.rag.chunks > 0 && provider.rag.offline && !provider.automatic_generation && provider.max_generation_attempts === 2);
    const live = await page.evaluate(async () => (await fetch('/api/live/status')).json());
    check('Capture starts stopped with no model verdicts', !live.running && live.model_prediction === null && live.multiclass_prediction === null);
    await page.evaluate(() => document.fonts.ready);
    check('Bundled Inter font loads offline', await page.evaluate(() => document.fonts.check('14px Inter')));
    for (const [route, panel, mode] of [['/csv', '#upload-panel', 'upload'], ['/samples', '#sample-panel', 'sample']]) {
      await openWorkspace(route, panel);
      check(`${route} opens its own source panel through the navigation`, await page.locator('body').getAttribute('data-workspace-mode') === mode && await page.locator(panel).isVisible());
    }
    await openWorkspace('/live', '#live-panel');
    for (const state of ['loading', 'error', 'exit']) {
      await page.goto(origin + '/' + state + '?return=/live');
      await page.locator(`#lifecycle-screen[data-state="${state}"]`).waitFor();
      check(`/${state} is available inside the desktop app`, await page.locator('.window-chrome').isVisible());
      if (state === 'exit') {
        await page.locator('#lifecycle-cancel').click();
        await page.waitForURL(url => url.pathname === '/live');
        await page.locator('#lifecycle-screen').waitFor({state: 'hidden'});
        check('Standalone exit cancel returns to the requested workspace', await page.locator('#live-panel').isVisible());
      }
    }
    if (full) {
      check('Full verification uses an explicitly enabled partial-input model', proof.model_status.partial_input_supported === true);
      await app.evaluate(({BrowserWindow}, target) => {
        global.__imdaiDownloads = [];
        BrowserWindow.getAllWindows()[0].webContents.session.on('will-download', (_event, item) => {
          global.__imdaiDownloads.push({filename:item.getFilename(),url:item.getURL()});
          item.setSavePath(target + '/' + Date.now() + '-' + item.getFilename());
          item.once('done', (_event, state) => {global.__imdaiDownloads.push({state, path:item.getSavePath()});});
        });
      }, out);
      // Electron exposes completed downloads through its Session API. Playwright's
      // page 'download' event does not represent this native desktop lifecycle.
      const saveDownload = async selector => {
        const before = await app.evaluate(() => global.__imdaiDownloads.length);
        await page.locator(selector).click();
        const deadline = Date.now() + 15000;
        while (Date.now() < deadline) {
          const entries = await app.evaluate((_electron, start) => global.__imdaiDownloads.slice(start), before);
          const completed = entries.find(entry => entry.state === 'completed');
          if (completed) return {path: completed.path, text: fs.readFileSync(completed.path, 'utf8')};
          assert.ok(!entries.some(entry => entry.state && entry.state !== 'completed'), 'Native export completes successfully');
          await page.waitForTimeout(100);
        }
        throw new Error('Native report download did not complete');
      };
      const saveReport = async selector => JSON.parse((await saveDownload(selector)).text);
      await openWorkspace('/samples', '#sample-panel');
      const benignSample = await page.locator('#sample-select option').evaluateAll(options => options.find(option => option.textContent.includes('Normal network activity'))?.value);
      assert.ok(benignSample, 'A labelled benign validation sample is available');
      await page.locator('#sample-select').selectOption(benignSample);
      await page.locator('#analyze-button').click();
      await page.locator('.prediction-label').waitFor();
      check('Friendly sample names and real TreeSHAP rendered', (await page.locator('#sample-select option:checked').textContent()).includes('Normal network activity') && await page.locator('#report-content .shap-row').count() > 0);
      const xaiResponse = page.waitForResponse(r => r.url().endsWith('/api/xai'));
      await page.locator('#extra-xai-button').click();
      proof.sample_xai = await (await xaiResponse).json();
      await page.locator('#extra-xai-content').waitFor();
      check('Actual LIME reference and local fidelity rendered', proof.sample_xai.lime.reference_rows === 512 && await page.locator('.lime-row').count() > 0 && (await page.locator('#extra-xai-content').textContent()).includes('fidelity'));
      await page.locator('#extra-xai-tab-1').click();
      check('Native tree statistics show actual input feature count', await page.locator('#extra-xai-view-1 tbody tr').count() === 170);
      await page.locator('#extra-xai-tab-2').click();
      check('Raw parameters and reference values rendered', await page.locator('#extra-xai-view-2 tbody tr').count() === 170);
      const aiResponse = page.waitForResponse(r => r.url().endsWith('/api/assistant/explain'));
      await page.locator('#assistant-button').click();
      const sampleAiResponse = await aiResponse;
      proof.sample_llm = await sampleAiResponse.json();
      assert.ok(sampleAiResponse.ok(), 'Actual sample generation succeeds: ' + (proof.sample_llm.error || sampleAiResponse.status()));
      await page.locator('#assistant-content').waitFor();
      check('Actual local English generation rendered on click', proof.sample_llm.llm_executed && await page.locator('.assistant-response').textContent() === proof.sample_llm.text);
      check('Actual sample generation uses retrieved local sources', proof.sample_llm.rag.used_in_generation && proof.sample_llm.rag.scope === 'csv' && proof.sample_llm.rag.context === proof.sample_llm.rag.passages.map(p => `[${p.id}] ${p.text}`).join('\n') && await page.locator('.rag-source').count() === proof.sample_llm.rag.passages.length);
      check('Generation audit shows bounded successful checks', proof.sample_llm.correction.generation_attempts.length <= 2 && proof.sample_llm.correction.generation_attempts.at(-1).passed && proof.sample_llm.rag.retrieval_attempts.length <= 2);
      const report = await saveReport('#download-report');
      proof.sample_export_keys = Object.keys(report);
      check('Analysis JSON export contains real model evidence', report.explanation.method.includes('TreeSHAP'));
      check('Analysis export preserves RAG sources and correction audit', report.selected_evidence_explanations.local_language_model.rag.context === proof.sample_llm.rag.context && report.selected_evidence_explanations.local_language_model.correction.generation_attempts.at(-1).passed);
      await page.screenshot({path: path.join(out, packaged ? 'packaged-analysis.png' : 'desktop-analysis.png'), fullPage: true});
      const csv = await page.evaluate(async () => (await fetch('/api/example.csv')).text());
      await openWorkspace('/csv', '#upload-panel');
      await page.locator('#assistant-content').waitFor({state: 'hidden'});
      check('Source change clears earlier AI output', !await page.locator('#assistant-content').isVisible());
      check('Source change clears the previous prediction', !await page.locator('#report-content').isVisible());
      await page.locator('#csv-file').setInputFiles({name: 'validation_examples.csv', mimeType: 'text/csv', buffer: Buffer.from(csv)});
      await page.locator('#upload-button').click();
      await page.locator('.batch-row').first().waitFor();
      check('CSV upload classifies eight real validation rows', await page.locator('.batch-row').count() === 8);
      await page.locator('.batch-row').first().click();
      await page.locator('.prediction-label').waitFor();
      check('Uploaded row has exact SHAP', await page.locator('#report-content .shap-row').count() > 0);
      check('CSV report includes all 170 shared feature states', await page.locator('#report-content .compatibility-table tbody tr').count() === 170);
      await page.locator('#csv-file').setInputFiles({name: 'invalid.csv', mimeType: 'text/csv', buffer: Buffer.from('wrong_column\n1\n')});
      await page.locator('#upload-button').click();
      await page.locator('#global-error').waitFor();
      check('Invalid schema clears result and reports error', !await page.locator('#report-content').isVisible());
      await openWorkspace('/live', '#live-panel');
      check('Live page clears the preceding CSV error', !await page.locator('#global-error').isVisible());
      await page.locator('#live-mode').selectOption('packets');
      await page.locator('#live-start').click();
      await page.locator('#live-multiclass-model .top-class').first().waitFor({timeout: 60000});
      await page.locator('#live-packet-model .packet-model-summary').waitFor({timeout: 60000});
      proof.live_capture = await page.evaluate(async () => (await fetch('/api/live/status')).json());
      const multiclass = proof.live_capture.multiclass_prediction;
      check('Actual Npcap capture drives the shared 84-class model on seven observed columns', proof.live_capture.mode === 'packets' && proof.live_capture.metrics.packets_observed > 0 && multiclass?.observation_id && multiclass.features.length === 170 && multiclass.feature_coverage.observed === 7 && multiclass.feature_coverage.missing === 163 && multiclass.prediction.tentative === true);
      check('Live main result shows three friendly tentative categories, model version and all feature states', await page.locator('#live-multiclass-model .top-class').count() === 3 && await page.locator('#live-multiclass-model .compatibility-table tbody tr').count() === 170 && await page.locator('#live-multiclass-model .feature-state.missing').count() === 163 && (await page.locator('#live-multiclass-model').textContent()).includes(multiclass.model.version) && multiclass.prediction.top_classes.every(entry => entry.display_label));
      check('Live TreeSHAP marks missing routing and keeps the binary model separate', (await page.locator('#live-multiclass-model .missing-routing-note').allTextContents()).join(' ').includes('missing-value routing') && multiclass.explanation.all_features.length === 170 && proof.live_capture.model_prediction.observation_id && await page.locator('#live-packet-model .shap-row').count() === 7 && (await page.locator('#live-packet-model h3').first().textContent()).includes('Binary packet comparison'));
      const liveCsvResponse = page.waitForResponse(response => new URL(response.url()).pathname === '/api/live/features.csv' && response.ok());
      const liveCsv = await saveDownload('#download-live-csv');
      const csvResponse = await liveCsvResponse;
      const csvObservationId = new URL(csvResponse.url()).searchParams.get('observation_id');
      const csvObservation = observedReports.get(csvObservationId);
      check('Live CSV export is bound to a displayed multiclass observation', Boolean(csvObservation) && csvObservation.features.length === 170);
      const roundtrip = await page.evaluate(async content => {
        const form = new FormData(); form.append('file', new File([content], 'live_roundtrip.csv', {type: 'text/csv'}));
        const response = await fetch('/api/upload', {method: 'POST', body: form});
        return {status: response.status, data: await response.json()};
      }, liveCsv.text);
      proof.live_csv_roundtrip = {observation_id: csvObservationId, source_model_version: csvObservation.model.version, ...roundtrip};
      check('Downloaded live CSV preserves all 170 measured or missing values and prediction', roundtrip.status === 200 && roundtrip.data.rows.length === 1 && JSON.stringify(roundtrip.data.rows[0].features) === JSON.stringify(csvObservation.features) && roundtrip.data.rows[0].prediction.label === csvObservation.prediction.label && Math.abs(roundtrip.data.rows[0].prediction.score - csvObservation.prediction.score) < 1e-7);
      const liveXai = page.waitForResponse(r => r.url().endsWith('/api/xai'));
      await page.locator('#extra-xai-button').click();
      const liveXaiResponse = await liveXai;
      proof.live_xai = await liveXaiResponse.json();
      check('Live LIME remains bound to the requested 170-feature multiclass observation', liveXaiResponse.ok() && proof.live_xai.source.observation_id === liveXaiResponse.request().postDataJSON().observation_id && proof.live_xai.statistics.features.length === 170 && proof.live_xai.model.version === multiclass.model.version && proof.live_xai.feature_coverage.observed === 7);
      const liveAi = page.waitForResponse(r => r.url().endsWith('/api/assistant/explain'));
      await page.locator('#assistant-button').click();
      const liveAiResponse = await liveAi;
      proof.live_llm = await liveAiResponse.json();
      check('Live English explanation uses actual bound multiclass evidence', liveAiResponse.ok() && proof.live_llm.llm_executed && proof.live_llm.evidence.source.observation_id === liveAiResponse.request().postDataJSON().observation_id && proof.live_llm.evidence.feature_coverage.total === 170 && proof.live_llm.evidence.feature_coverage.observed === 7 && proof.live_llm.evidence.prediction.tentative === true);
      check('Actual live generation retrieves only packet or common knowledge', proof.live_llm.rag.used_in_generation && proof.live_llm.rag.scope === 'packet' && proof.live_llm.rag.passages.every(p => p.scopes.includes('packet') || p.scopes.includes('common')) && proof.live_llm.correction.generation_attempts.at(-1).passed);
      await page.locator('#assistant-content').waitFor();
      proof.live_export = await saveReport('#download-live-report');
      check('Live export records actual observation', proof.live_export.source.synthetic === false && proof.live_export.metrics.packets_observed > 0);
      check('Live export retains the displayed 170-feature multiclass observation alongside the current snapshot', proof.live_export.displayed_multiclass_observation.features.length === 170 && proof.live_export.displayed_multiclass_observation.observation_id && proof.live_export.multiclass_prediction.observation_id);
      check('Live export preserves RAG attached to its original packet window', proof.live_export.selected_evidence_explanations.local_language_model.evidence.source.observation_id === proof.live_llm.evidence.source.observation_id && proof.live_export.selected_evidence_explanations.local_language_model.rag.corpus_sha256 === proof.live_llm.rag.corpus_sha256);
      await page.screenshot({path: path.join(out, packaged ? 'packaged-live.png' : 'desktop-live.png'), fullPage: true});
      await page.locator('#live-stop').click();
      await page.locator('#live-start').waitFor({state: 'visible'});
      await page.waitForFunction(() => !document.getElementById('live-start').disabled);
      check('Stop clears both model verdicts, all generated explanations and live exports', !await page.locator('#assistant-content').isVisible() && !await page.locator('#extra-xai-content').isVisible() && !await page.locator('#live-packet-model .packet-model-summary').count() && !await page.locator('#live-multiclass-model .top-class').count() && await page.locator('#download-live-csv').isDisabled() && await page.locator('#download-live-report').isDisabled());
      await page.locator('.studio-nav a[href="/"]').click();
      await page.locator('.route-card.route-live').waitFor();
      await page.emulateMedia({reducedMotion: 'reduce'});
      check('Reduced motion disables decorative animation', await page.evaluate(() => {
        const style = getComputedStyle(document.querySelector('.network-scene'));
        return style.animationName === 'none' && style.transitionDuration === '0s';
      }));
      await app.evaluate(({BrowserWindow}) => {const w=BrowserWindow.getAllWindows()[0];w.setMinimumSize(300,500);w.setContentSize(390,844);});
      await page.waitForTimeout(200);
      check('Narrow layout has no horizontal page overflow', await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
      await page.screenshot({path: path.join(out, packaged ? 'packaged-narrow.png' : 'desktop-narrow.png'), fullPage: true});
      await page.emulateMedia({reducedMotion: 'no-preference'});
      await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].setContentSize(1440,940));
      await page.evaluate(() => {document.activeElement?.blur(); window.scrollTo({top:0,behavior:'instant'});});
      await page.waitForTimeout(800);
    }
    await page.screenshot({path: path.join(out, packaged ? 'packaged-overview.png' : 'desktop-overview.png'), fullPage: true, animations: 'disabled'});
    const beforeExitURL = page.url();
    await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].close());
    await page.locator('#lifecycle-screen[data-state="exit"]').waitFor();
    check('Native close opens the themed exit screen', (await page.locator('#lifecycle-title').textContent()).includes('Close this workspace'));
    await page.locator('#lifecycle-cancel').click();
    await page.locator('#lifecycle-screen').waitFor({state: 'hidden'});
    check('Cancel preserves the workspace and local service', page.url() === beforeExitURL && await page.evaluate(async () => (await fetch('/api/status')).ok));
    await page.locator('[data-window-action="close"]').click();
    await page.locator('#lifecycle-screen[data-state="exit"]').waitFor();
    await page.screenshot({path: path.join(out, packaged ? 'packaged-exit.png' : 'desktop-exit.png'), fullPage: false, animations: 'disabled'});
    await page.keyboard.press('Escape');
    await page.locator('#lifecycle-screen').waitFor({state: 'hidden'});
    check('Escape cancels the custom title-bar close action', page.url() === beforeExitURL);
    check('No renderer runtime errors', proof.errors.length === 0);
    proof.status = 'passed';
  } catch(error) {
    proof.failure = error.message;
    proof.native_download_events = await app.evaluate(() => global.__imdaiDownloads || []).catch(() => []);
    const page = app.windows()[0];
    if (page) await page.screenshot({path: path.join(out, packaged ? 'packaged-failure.png' : 'desktop-failure.png'), fullPage: true}).catch(() => {});
    throw error;
  } finally {
    const closed = app.waitForEvent('close', {timeout: 20000}).catch(() => {});
    const ownedPage = app.windows()[0];
    if (ownedPage) await ownedPage.evaluate(async () => {await window.netsecDesktop.requestClose(); await window.netsecDesktop.confirmClose();}).catch(() => {});
    await closed;
    await app.close().catch(() => {});
    fs.writeFileSync(path.join(out, packaged ? 'packaged-smoke.json' : 'desktop-smoke.json'), JSON.stringify(proof, null, 2));
  }
  let unreachable = false;
  try {await fetch(origin + '/api/status', {signal: AbortSignal.timeout(2000)});} catch {unreachable = true;}
  check('Owned backend exits when desktop closes', unreachable);
  fs.writeFileSync(path.join(out, packaged ? 'packaged-smoke.json' : 'desktop-smoke.json'), JSON.stringify(proof, null, 2));
  console.log(JSON.stringify({status:proof.status, packaged, checks:proof.checks, errors:proof.errors}, null, 2));
})().catch(error => {console.error(error); process.exitCode = 1;});
