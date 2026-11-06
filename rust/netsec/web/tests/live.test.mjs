import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {caveat, plain, shown, livePrediction, chatSource, scoreKindLabel, verdict, explanations} from '../src/liveView.ts';

const here = path.dirname(fileURLToPath(import.meta.url));
const web = path.join(here, '..'), workspace = path.join(web, '..', '..', '..');
const fixture = JSON.parse(readFileSync(path.join(here, 'fixtures', 'live77_status.json'), 'utf8'));
const bundle = JSON.parse(readFileSync(path.join(workspace, 'artifacts/application/live_models_v1/live77_xgboost/bundle.json'), 'utf8'));
const scored = fixture.scored_recon_scan, skipped = fixture.skipped;

test('the fixture uses the SHA-pinned bundle label and provenance note, and runtime field names', () => {
  const prediction = scored.multiclass_prediction;
  assert.equal(prediction.model_label, bundle.label);
  assert.equal(prediction.provenance_note, bundle.provenance_note);
  assert.equal(prediction.score_kind, bundle.output.score_kind);
  const runtime = ['live/models.rs', 'live.rs'].map(file => readFileSync(path.join(web, '..', 'crates/runtime/src', file), 'utf8')).join('\n');
  for (const key of ['model_label', 'provenance_note', 'score_kind', 'raw_label', 'class_merge', 'merged_source_explanation', 'merge_margins', 'merge_note', 'observation_id', 'binary_prediction', 'multiclass_prediction'])
    assert.match(runtime, new RegExp(`"${key}"`), key);
});

test('only the live77 multiclass verdict is read; a null binary slot is tolerated', () => {
  assert.equal(livePrediction(scored).model_id, 'live77_xgboost');
  assert.equal(livePrediction(fixture.stopped), undefined);
  assert.equal(livePrediction({}), undefined);
  // Even a stray binary value is never rendered: no live77 binary calibration exists.
  assert.equal(livePrediction({binary_prediction: {model_id: 'live_binary_xgboost', label: 'suspicious'}}), undefined);
});

test('the provenance note and model label survive the caveat filter and are exempt from it', () => {
  for (const text of [bundle.label, bundle.provenance_note, scoreKindLabel(bundle.output.score_kind), scored.multiclass_prediction.explanation.merge_note])
    assert.equal(caveat.test(text), false, text);
  assert.deepEqual(plain({provenance_note: bundle.provenance_note}), {provenance_note: bundle.provenance_note});
  // The note is copied from the prediction, never through `shown`, so even caveat wording would stay visible.
  const caveated = 'dataset-trained, live accuracy unverified';
  assert.equal(shown(caveated), undefined);
  assert.equal(verdict({...scored.multiclass_prediction, provenance_note: caveated}).provenance, caveated);
});

test('a completed verdict shows model label, calibrated-after-merge score kind and the merge', () => {
  const value = verdict(scored.multiclass_prediction);
  assert.deepEqual(value, {model: bundle.label, title: 'recon_scan', score: 0.8125, scoreKind: 'calibrated probability after class merge', provenance: bundle.provenance_note,
    raw: 'Raw model argmax: recon_ping-sweep; recon_ping-sweep was merged into recon_scan.'});
  assert.equal(verdict({...scored.multiclass_prediction, raw_label: 'recon_scan'}).raw, undefined);
  assert.equal(scoreKindLabel(undefined), 'model score');
  assert.deepEqual(chatSource(scored.multiclass_prediction), {observation_id: 'w-0001', model_id: 'live77_xgboost'});
});

test('a skipped window has no verdict, no score and no chat binding', () => {
  const value = verdict(skipped.multiclass_prediction);
  assert.equal(value.title, 'Skipped');
  assert.match(value.detail, /Capture loss/);
  assert.equal(value.score, undefined);
  assert.equal(value.provenance, undefined);
  assert.equal(chatSource(skipped.multiclass_prediction), null);
  assert.equal(verdict({available: false, model_executed: false, error: 'bundle missing'}).title, 'Unavailable');
  assert.equal(chatSource(undefined), null);
});

test('a recon_scan verdict explains recon_scan and recon_ping-sweep side by side with the merge note', () => {
  const value = explanations(scored.multiclass_prediction);
  assert.deepEqual(value.items.map(item => [item.method, item.target_label]), [['exact_tree_shap', 'recon_scan'], ['exact_tree_shap', 'recon_ping-sweep']]);
  assert.match(value.note, /merged into recon_scan/);
  const {merged_source_explanation, merge_margins, ...single} = scored.multiclass_prediction;
  const other = explanations({...single, label: 'benign', explanation: {...single.explanation, target_label: 'benign', merge_note: undefined}});
  assert.deepEqual([other.note, other.items.length], [undefined, 1]);
  assert.deepEqual(explanations(skipped.multiclass_prediction).items, []);
});

async function render(load) {
  const {createServer} = await import('vite');
  // router.ts reads window.location at import; nothing else touches the DOM before an effect runs.
  globalThis.window ??= {location: {pathname: '/live', search: '', href: 'http://127.0.0.1/live', origin: 'http://127.0.0.1'}};
  const server = await createServer({root: web, configFile: false, logLevel: 'silent', appType: 'custom', server: {middlewareMode: true, hmr: false, ws: false}, optimizeDeps: {noDiscovery: true, include: []}});
  try {
    const [live, react, dom] = [await server.ssrLoadModule('/src/Live.tsx'), await import('react'), await import('react-dom/server')];
    return load(live, (component, props) => dom.renderToStaticMarkup(react.createElement(component, props)));
  } finally { await server.close(); }
}
const text = html => html.replace(/<[^>]+>/g, ' ').replace(/&#x27;/g, "'").replace(/&amp;/g, '&').replace(/\s+/g, ' ');

test('the Live page renders the live77 verdict with its label, provenance note and both SHAP targets', async () => {
  await render((live, html) => {
    const card = text(html(live.LiveVerdict, {value: scored.multiclass_prediction}));
    for (const expected of [bundle.label, bundle.provenance_note, 'recon_scan', '81.25%', 'calibrated probability after class merge', 'Raw model argmax: recon_ping-sweep'])
      assert.ok(card.includes(expected), `${expected} missing from: ${card}`);
    assert.doesNotMatch(card, /attack probability|Experimental packet-only/i);
    const markup = html(live.LiveExplanations, {value: scored.multiclass_prediction}), xai = text(markup);
    assert.ok(xai.includes(scored.multiclass_prediction.explanation.merge_note), xai);
    // Two method headings (the exact-method label), one per class; the evidence JSON below each is not counted.
    assert.equal(markup.match(/<h4>exact_tree_shap<\/h4>/g)?.length, 2, markup);
    for (const target of ['recon_scan', 'recon_ping-sweep', 'raw class margin, not probability']) assert.ok(xai.includes(target), target);
    const skip = text(html(live.LiveVerdict, {value: skipped.multiclass_prediction}));
    assert.ok(skip.includes('Skipped') && skip.includes('Capture loss'), skip);
    assert.doesNotMatch(skip, /%/);
    assert.match(text(html(live.LiveExplanations, {value: skipped.multiclass_prediction})), /completed measured window/);
  });
});

test('a verdict below the 20% confidence floor shows unfamiliar traffic, not an attack label', () => {
  const low = verdict({available: true, model_executed: true, model_id: 'live77_xgboost', observation_id: 'w', label: 'dos_tcp-flood-port-443', score: 0.087, score_kind: 'calibrated_probability_after_class_merge'});
  assert.equal(low.title, 'Unfamiliar traffic');
  assert.equal(low.unfamiliar, true);
  assert.ok(!low.title.includes('dos') && low.detail.includes('9%'));
  const high = verdict({available: true, model_executed: true, model_id: 'live77_xgboost', observation_id: 'w', label: 'benign', score: 0.95});
  assert.equal(high.title, 'benign');
  assert.equal(verdict({status: 'no_traffic', available: false, model_executed: false}).title, 'No traffic in this window');
});
