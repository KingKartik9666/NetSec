import {api, post, delay, aborted} from './api.mjs';
import {$, node, errorAt, metadata, download, announce} from './dom.mjs';
import {Generation, terminalStates, eligibleModels, requestedModels, validateRows} from './state.mjs';
import {renderCatalog} from './catalog.mjs';
import {renderResults, renderExplanation, renderEnglish, methods} from './results.mjs';
import {initializeChrome} from './chrome.mjs';

const route = location.pathname.replace(/\/$/, '') || '/';
const analysisRoutes = new Set(['/models', '/samples', '/csv']);
const generation = new Generation(), evidenceGeneration = new Generation();
const state = {catalog: {models: [], schema: {feature_names: [], classes: []}}, selected: new Set(), mode: 'selected',
  samples: [], inputKind: route === '/csv' ? 'csv' : 'sample', csv: '', csvLoading: false, job: null, inputSnapshot: null,
  rowIndex: 0, evidenceModel: '', explanation: null, english: null, records: [], starting: false,
  cancelling: false, xaiBusy: false, englishBusy: false, pollingPaused: false, runtimeFailed: false};

function activeJob() {return state.job && !terminalStates.has(state.job.status);}
function idsForRun() {return requestedModels(state.catalog.models, state.selected, state.mode, state.catalog.schema);}
function modelInfo(id) {return state.catalog.models.find(model => model.id === id);}
function chosenSample() {return state.samples.find(sample => String(sample.id) === $('sample-select').value);}
function hasInput() {return state.inputKind === 'sample' ? Boolean(chosenSample()) : state.inputKind === 'csv' ? Boolean(state.csv) && !state.csvLoading : Boolean($('feature-rows').value.trim());}
function completedModels() {return (state.job?.models || []).filter(model => model.status === 'completed' && model.predictions?.[state.rowIndex]);}
function selectedMethods() {return modelInfo(state.evidenceModel)?.capabilities?.explanations || [];}
function languageReady() {return ['ready', 'available', 'completed'].includes(state.catalog.language?.status);}

function controls() {
  const running = Boolean(activeJob());
  $('run-analysis').disabled = state.runtimeFailed || state.starting || running || !idsForRun().length || !hasInput();
  $('run-analysis').textContent = state.starting ? 'Creating input snapshot…' : running ? 'Analysis in progress…' : state.mode === 'extensive' ? `Run ${idsForRun().length} compatible models ↗` : 'Run selected models ↗';
  $('refresh-catalog').disabled = state.runtimeFailed || running || state.starting;
  $('cancel-job').hidden = !running; $('cancel-job').disabled = state.runtimeFailed || state.cancelling;
  $('cancel-job').textContent = state.cancelling ? 'Requesting cancellation…' : 'Cancel remaining work';
  $('resume-job-poll').hidden = !state.pollingPaused || !running;
  $('export-results').disabled = !state.job;
  $('example-csv').disabled = !chosenSample();
  const completed = completedModels().some(model => model.model_id === state.evidenceModel);
  $('evidence-model').disabled = !completedModels().length;
  $('xai-method').disabled = !completed || state.xaiBusy;
  $('explain-model').disabled = state.runtimeFailed || !completed || state.xaiBusy || !selectedMethods().includes($('xai-method').value);
  $('explain-model').textContent = state.xaiBusy ? 'Calculating model evidence…' : 'Explain this prediction';
  $('explain-english').disabled = state.runtimeFailed || !completed || !languageReady() || modelInfo(state.evidenceModel)?.capabilities?.english !== true || state.englishBusy;
  $('explain-english').textContent = state.englishBusy ? 'Generating local English…' : 'Explain in plain English ↗';
}

function clearEvidence() {
  evidenceGeneration.invalidate(); state.explanation = null; state.english = null; state.xaiBusy = false; state.englishBusy = false;
  $('xai-content').replaceChildren(); $('english-content').replaceChildren();
  $('xai-status').textContent = ''; $('english-status').textContent = 'Generation starts only when requested.';
  errorAt('xai-error'); errorAt('english-error');
}

function invalidateAnalysis() {
  const previous = activeJob() && state.job.id;
  generation.invalidate(); clearEvidence();
  state.job = null; state.inputSnapshot = null; state.rowIndex = 0; state.evidenceModel = ''; state.records = []; state.pollingPaused = false;
  state.csvLoading = false;
  $('job-empty').hidden = false; $('job-content').hidden = true; $('evidence-model').replaceChildren();
  $('evidence-source').textContent = 'Complete a model prediction to inspect its evidence.';
  errorAt('job-error'); errorAt('input-error');
  if (previous) post(`/api/v1/jobs/${encodeURIComponent(previous)}/cancel`, {}).catch(error => {
    // Never publish an obsolete result as current. Cancellation failures remain
    // visible so the user knows the detached server job may still be running.
    errorAt('job-error', `The previous input was detached, but cancellation was not confirmed: ${error.message}`);
  });
  renderEvidenceChoices(); controls();
}

function catalogChanged(id, checked) {
  invalidateAnalysis();
  if (checked) state.selected.add(id); else state.selected.delete(id);
  drawCatalog(); controls();
}
function drawCatalog() {renderCatalog(state.catalog, state.selected, state.mode, catalogChanged);}

async function loadCatalog() {
  errorAt('catalog-error');
  try {
    const catalog = await api('/api/v1/catalog');
    if (!Array.isArray(catalog.models) || !Array.isArray(catalog.schema?.feature_names)) throw new Error('The model catalog has no verified feature schema.');
    state.catalog = catalog; state.runtimeFailed = false;
    $('retry-connection').disabled = false;
    const eligible = eligibleModels(catalog.models, catalog.schema);
    state.selected = new Set([...state.selected].filter(id => eligible.some(model => model.id === id)));
    if (!state.selected.size && eligible.length) state.selected.add(eligible.find(model => model.id === 'xgboost')?.id || eligible[0].id);
    drawCatalog();
    $('language-status').textContent = `${catalog.language?.model || 'Local Qwen'} · ${catalog.language?.status || 'unavailable'}${catalog.language?.reason ? ` · ${catalog.language.reason}` : ''}. Evidence stays on this PC.`;
    $('connection-error').hidden = true;
  } catch (error) {
    errorAt('catalog-error', error); $('connection-message').textContent = `The local runtime could not provide its model catalog: ${error.message}`;
    $('connection-error').hidden = false;
  }
  controls();
}

async function loadSamples() {
  try {
    const response = await api('/api/v1/samples');
    if (!Array.isArray(response.samples)) throw new Error('The runtime returned no validation-sample catalog.');
    state.samples = response.samples;
    $('sample-select').replaceChildren(...state.samples.map(sample => {
      const option = node('option', sample.display_name || sample.label || `Validation observation ${sample.id}`); option.value = String(sample.id); return option;
    }));
    $('sample-select').disabled = !state.samples.length;
    if (!state.samples.length) {$('sample-select').append(node('option', 'No compatible included observations available')); $('sample-select').value = '';}
    sampleDescription();
  } catch (error) {
    $('sample-select').replaceChildren(node('option', 'Samples unavailable'));
    $('sample-select').disabled = true; errorAt('input-error', error);
  }
  controls();
}
function sampleDescription() {
  const sample = chosenSample();
  $('sample-description').textContent = sample
    ? `Source label: ${sample.label} · Source row: ${sample.source_row_id ?? 'unavailable'} · ${sample.origin || 'validation'} observation. This is not an independent test or live capture.`
    : 'No compatible source-validation observation is available.';
}
function showInput(kind) {
  state.inputKind = kind;
  for (const value of ['sample', 'csv', 'rows']) {
    $(`${value}-input`).hidden = kind !== value;
    document.querySelector(`input[name="input-kind"][value="${value}"]`).checked = kind === value;
  }
  controls();
}

function focusModel(id) {
  if (id !== state.evidenceModel) {state.evidenceModel = id; clearEvidence();}
  renderEvidenceChoices();
  if (state.job) renderResults(state.job, state.rowIndex, state.evidenceModel, focusModel);
  controls();
}
function renderEvidenceChoices() {
  const completed = completedModels();
  if (!completed.some(model => model.model_id === state.evidenceModel)) {
    const next = completed[0]?.model_id || '';
    if (next !== state.evidenceModel) {state.evidenceModel = next; clearEvidence();}
  }
  $('evidence-model').replaceChildren(...completed.map(model => {const option = node('option', model.label || model.model_id); option.value = model.model_id; option.selected = model.model_id === state.evidenceModel; return option;}));
  const previousMethod = $('xai-method').value, supported = selectedMethods();
  $('xai-method').replaceChildren(...methods.map(method => {
    const option = node('option', `${method.label}${supported.includes(method.id) ? '' : ' · unavailable'}`);
    option.value = method.id; option.disabled = !supported.includes(method.id); return option;
  }));
  $('xai-method').value = supported.includes(previousMethod) ? previousMethod : methods.find(method => supported.includes(method.id))?.id || '';
  methodDescription();
  $('evidence-source').textContent = state.evidenceModel
    ? `${modelInfo(state.evidenceModel)?.label || state.evidenceModel} · Observation ${state.rowIndex + 1} · Input ${state.job.input?.id || state.job.id}. Explanations stay bound to this exact result.`
    : 'Complete a model prediction to inspect its evidence.';
}
function methodDescription() {
  $('xai-method-note').textContent = methods.find(method => method.id === $('xai-method').value)?.note || 'No supported method is available for this model/reference. Its prediction remains available.';
}

function publishJob(job) {
  if (!job || !Array.isArray(job.models)) throw new Error('The runtime returned an incomplete analysis job.');
  state.job = job;
  const count = job.input?.row_count || Math.max(...job.models.map(model => model.predictions?.length || 0), 1);
  if (state.rowIndex >= count) {state.rowIndex = 0; clearEvidence();}
  const previous = $('result-row').value;
  if ($('result-row').options.length !== count || !previous) {
    $('result-row').replaceChildren(...Array.from({length: count}, (_, index) => {const option = node('option', `Observation ${index + 1}`); option.value = String(index); return option;}));
  }
  $('result-row').value = String(state.rowIndex);
  renderEvidenceChoices(); renderResults(job, state.rowIndex, state.evidenceModel, focusModel); controls();
}

async function pollJob(id, ticket) {
  const controller = generation.controller('poll'); state.pollingPaused = false; controls();
  const deadline = performance.now() + 2 * 60 * 60 * 1000;
  try {
    while (generation.current(ticket) && performance.now() < deadline) {
      const job = await api(`/api/v1/jobs/${encodeURIComponent(id)}`, {signal: controller.signal});
      if (!generation.current(ticket) || state.job?.id !== id) return;
      publishJob(job);
      if (terminalStates.has(job.status)) {announce(`Analysis ${job.status}. ${job.progress?.completed ?? 0} model operations finished.`); return;}
      await delay(document.hidden ? 8000 : 1500, controller.signal);
    }
    if (generation.current(ticket)) throw new Error('Automatic progress checks reached their two-hour limit. The server job remains independent; check progress or cancel it explicitly.');
  } catch (error) {
    if (!generation.current(ticket) || aborted(error)) return;
    state.pollingPaused = true; errorAt('job-error', error); controls();
  }
}

async function runAnalysis() {
  if (state.starting || activeJob()) return;
  errorAt('input-error'); errorAt('job-error');
  let input;
  try {
    if (state.inputKind === 'sample') {const sample = chosenSample(); if (!sample) throw new Error('Choose an available validation sample.'); input = {sample_id: sample.id};}
    else if (state.inputKind === 'csv') {if (!state.csv) throw new Error('Choose a prepared CSV first.'); input = {csv: state.csv};}
    else input = {rows: validateRows(JSON.parse($('feature-rows').value), state.catalog.schema.feature_names.length)};
    if (!idsForRun().length) throw new Error('Choose at least one available compatible model.');
  } catch (error) {errorAt('input-error', error); return;}
  invalidateAnalysis();
  const ticket = generation.value, modelIds = idsForRun();
  const payload = {model_ids: modelIds, mode: state.mode, ...input};
  state.inputSnapshot = {schema: state.catalog.schema, ...input}; state.starting = true; controls();
  try {
    // Retain a late create response long enough to cancel its actual job ID.
    const response = await post('/api/v1/jobs', payload, {timeout: 60000});
    if (!response.id) throw new Error('The runtime did not return an analysis job identity.');
    if (!generation.current(ticket)) {await post(`/api/v1/jobs/${encodeURIComponent(response.id)}/cancel`, {}); return;}
    const initial = response.models ? response : {...response, status: response.status || 'queued', models: modelIds.map(id => ({model_id: id, label: modelInfo(id)?.label || id, status: 'queued'}))};
    publishJob(initial); announce('Analysis input snapshotted. Compatible model results will arrive independently.');
    pollJob(response.id, ticket);
  } catch (error) {if (generation.current(ticket)) errorAt('job-error', error);}
  finally {state.starting = false; controls();}
}

async function cancelJob() {
  if (!activeJob() || state.cancelling) return;
  const id = state.job.id, ticket = generation.value;
  state.cancelling = true; errorAt('job-error'); controls();
  try {
    const job = await post(`/api/v1/jobs/${encodeURIComponent(id)}/cancel`, {});
    if (generation.current(ticket) && state.job?.id === id) {publishJob(job); announce('The runtime returned the cancellation status. Completed model results are retained.');}
  } catch (error) {if (generation.current(ticket)) errorAt('job-error', error);}
  finally {state.cancelling = false; controls();}
}

async function requestEvidence(kind) {
  if (!state.job || !state.evidenceModel) return;
  const ticket = evidenceGeneration.value, controller = evidenceGeneration.controller(kind);
  const binding = {job_id: state.job.id, model_id: state.evidenceModel, row_index: state.rowIndex};
  const isXai = kind === 'xai';
  if (isXai) {state.xaiBusy = true; errorAt('xai-error'); $('xai-status').textContent = 'Running the requested explanation on this model’s actual output…';}
  else {state.englishBusy = true; errorAt('english-error'); $('english-status').textContent = 'Executing the local language model with bounded evidence and grounding checks…';}
  controls();
  try {
    const result = await post(isXai ? '/api/v1/explain' : '/api/v1/english', {...binding,
      ...(isXai ? {method: $('xai-method').value} : state.explanation ? {explanation: state.explanation} : {})}, {signal: controller.signal, timeout: 180000});
    if (!evidenceGeneration.current(ticket)) return;
    if (isXai) {
      if (!result.method) throw new Error('The explanation capability returned no method evidence.');
      state.explanation = result; renderExplanation(result); $('xai-status').textContent = 'Explanation complete for this model, input and row.';
    } else {
      if (typeof result.text !== 'string' || !result.text.trim() || ['unavailable', 'failed'].includes(result.status) || result.llm_executed === false) throw new Error(result.reason || result.error || 'Local language generation returned no verified execution result.');
      state.english = result; renderEnglish(result); $('english-status').textContent = 'Local generation complete. Verify the wording against the original model evidence.';
    }
    state.records.push({...binding, kind, result}); announce(isXai ? 'Requested model explanation completed.' : 'Requested local English generation completed.');
  } catch (error) {
    if (!evidenceGeneration.current(ticket) || aborted(error)) return;
    errorAt(isXai ? 'xai-error' : 'english-error', error);
    $(isXai ? 'xai-status' : 'english-status').textContent = isXai ? 'This explanation failed. Model predictions remain available.' : 'Local generation failed. Review the measured evidence or retry.';
  } finally {
    if (evidenceGeneration.current(ticket)) {if (isXai) state.xaiBusy = false; else state.englishBusy = false; controls();}
  }
}

async function loadSettings() {
  errorAt('settings-error'); $('refresh-settings').disabled = true;
  try {
    const data = await api('/api/v1/settings');
    $('settings-content').replaceChildren(metadata(Object.entries(data).filter(([, value]) => typeof value !== 'object')));
    $('settings-details').textContent = JSON.stringify(data, null, 2);
  } catch (error) {errorAt('settings-error', error);}
  finally {$('refresh-settings').disabled = false;}
}

for (const link of document.querySelectorAll('.studio-nav a')) if (link.getAttribute('href') === route) {link.classList.add('active'); link.setAttribute('aria-current', 'page');}
const view = analysisRoutes.has(route) ? 'analysis-view' : ({'/live': 'live-view', '/settings': 'settings-view', '/help': 'help-view'})[route] || 'home-view';
$(view).hidden = false; document.body.classList.add(view === 'home-view' ? 'home-page' : 'analysis-page');
if (analysisRoutes.has(route)) {$('analysis-title').firstChild.textContent = route === '/csv' ? 'CSV analysis' : route === '/samples' ? 'Validation samples' : 'Model analysis'; document.title = `${route === '/csv' ? 'CSV analysis' : route === '/samples' ? 'Samples' : 'Models'} — NetSec`;}
else document.title = `${route === '/live' ? 'Live screening' : route === '/settings' ? 'Settings' : route === '/help' ? 'Help' : 'Network in focus'} — NetSec`;

document.querySelectorAll('input[name="analysis-mode"]').forEach(input => input.addEventListener('change', () => {invalidateAnalysis(); state.mode = input.value; drawCatalog(); controls();}));
document.querySelectorAll('input[name="input-kind"]').forEach(input => input.addEventListener('change', () => {invalidateAnalysis(); showInput(input.value);}));
$('sample-select').addEventListener('change', () => {invalidateAnalysis(); sampleDescription(); controls();});
$('feature-rows').addEventListener('input', () => {invalidateAnalysis(); controls();});
$('csv-file').addEventListener('change', async () => {
  invalidateAnalysis(); state.csv = ''; const ticket = generation.value, file = $('csv-file').files[0];
  if (!file) {$('csv-file-status').textContent = 'No CSV selected.'; controls(); return;}
  if (file.size > 10 * 1024 * 1024 || !file.name.toLowerCase().endsWith('.csv')) {errorAt('input-error', 'Choose a UTF-8 .csv file no larger than 10 MB.'); controls(); return;}
  state.csvLoading = true; $('csv-file-status').textContent = 'Reading the selected file locally…'; controls();
  try {
    const buffer = await file.arrayBuffer(); const csv = new TextDecoder('utf-8', {fatal: true}).decode(buffer);
    if (!generation.current(ticket)) return;
    state.csv = csv; $('csv-file-status').textContent = `${file.name} · ${file.size.toLocaleString()} bytes · read locally. The runtime validates columns and rows before model execution.`;
  } catch (error) {if (generation.current(ticket)) errorAt('input-error', `CSV could not be read as UTF-8: ${error.message}`);}
  finally {if (generation.current(ticket)) {state.csvLoading = false; controls();}}
});
$('example-csv').addEventListener('click', () => {
  const sample = chosenSample(); if (!sample) return;
  const escaped = name => `"${String(name).replaceAll('"', '""')}"`;
  const csv = `${state.catalog.schema.feature_names.map(escaped).join(',')}\r\n${sample.row.map(value => value === null ? '' : value).join(',')}\r\n`;
  download('netsec-validation-features.csv', csv, 'text/csv;charset=utf-8');
});
$('run-analysis').addEventListener('click', runAnalysis);
$('cancel-job').addEventListener('click', cancelJob);
$('resume-job-poll').addEventListener('click', () => {if (state.job) {errorAt('job-error'); pollJob(state.job.id, generation.value);}});
$('result-row').addEventListener('change', () => {state.rowIndex = Number($('result-row').value); clearEvidence(); publishJob(state.job);});
$('evidence-model').addEventListener('change', () => focusModel($('evidence-model').value));
$('xai-method').addEventListener('change', () => {clearEvidence(); methodDescription(); controls();});
$('explain-model').addEventListener('click', () => requestEvidence('xai'));
$('explain-english').addEventListener('click', () => requestEvidence('english'));
$('refresh-catalog').addEventListener('click', () => {invalidateAnalysis(); loadCatalog();});
$('retry-connection').addEventListener('click', async () => {await loadCatalog(); if (analysisRoutes.has(route)) await loadSamples(); if (route === '/settings') await loadSettings();});
$('refresh-settings').addEventListener('click', loadSettings);
$('export-results').addEventListener('click', () => {
  if (!state.job) return;
  download(`netsec-analysis-${String(state.job.id).replace(/[^a-zA-Z0-9_-]/g, '')}.json`, JSON.stringify({schema_version: 1,
    application: 'NetSec', runtime: state.catalog.runtime, job: state.job, input: state.inputSnapshot,
    catalog_models: state.catalog.models.filter(model => state.job.models.some(result => result.model_id === model.id)), requested_evidence: state.records,
    limitations: ['Predictions require human verification.', 'Model agreement is descriptive, not ground truth.', 'Included samples originate from validation; held-out test scoring remains locked.']}, null, 2));
});
window.addEventListener('pagehide', () => {generation.invalidate(); evidenceGeneration.invalidate();});
window.addEventListener('netsec-runtime-failed', () => {
  // Keep completed evidence exportable. An ended owned process cannot be
  // repaired by publishing an old response or silently replacing a detector.
  generation.invalidate(); evidenceGeneration.invalidate();
  state.runtimeFailed = true; state.starting = false; state.xaiBusy = false;
  state.englishBusy = false; state.cancelling = false; state.csvLoading = false;
  state.pollingPaused = false;
  $('connection-message').textContent = 'The local NetSec service stopped. Close and restart NetSec to continue. You can still export any completed results shown in this workspace.';
  $('connection-error').hidden = false; $('retry-connection').disabled = true;
  if (state.job && !terminalStates.has(state.job.status)) {
    $('job-status').textContent = 'SERVICE STOPPED';
    errorAt('job-error', 'Remaining operations could not be confirmed after the service stopped. Completed predictions are retained with their last recorded status.');
  }
  $('xai-status').textContent = state.explanation ? 'Completed explanation retained. Restart NetSec to request further evidence.' : 'Restart NetSec to request model evidence.';
  $('english-status').textContent = state.english ? 'Completed English explanation retained.' : 'Local generation is unavailable until NetSec is restarted.';
  for (const id of ['live-start', 'live-stop', 'refresh-settings']) $(id).disabled = true;
  controls(); announce('NetSec service stopped. Restart the app; completed results remain exportable.');
});
initializeChrome(); showInput(state.inputKind);
await loadCatalog();
if (analysisRoutes.has(route)) await loadSamples();
if (route === '/settings') await loadSettings();
if (route === '/live') {const {initializeLive} = await import('./live.mjs'); initializeLive({language: state.catalog.language}).catch(error => errorAt('live-error', error));}
