import test from 'node:test';
import assert from 'node:assert/strict';
import {badge, mergePending, parseEvents, readEvents, replyMeta, revealFrom, splitFooter, NETSEC_LABEL, CHOOSE_XAI, xaiRequired} from '../src/chatStream.ts';

const sentence = (text, source = 'model', violation = null) => ({text, source, violation});
const view = (revision, sentences, extra = {}) => ({id: 'c1', revision, busy: true, messages: [], pending: {question: 'Why?', sentences}, ...extra});

test('pending sentences are appended in order and never lost to a stale view', () => {
  let context = mergePending(view(1, []), view(1, [sentence('One.')]));
  context = mergePending(context, view(1, [sentence('One.'), sentence('Two.')]));
  assert.deepEqual(context.pending.sentences.map(item => item.text), ['One.', 'Two.']);
  const first = context.pending.sentences[0];
  // A late poll or event with fewer sentences of the same turn keeps what is already shown.
  context = mergePending(context, view(1, [sentence('One.')]));
  assert.deepEqual(context.pending.sentences.map(item => item.text), ['One.', 'Two.']);
  context = mergePending(context, view(1, [sentence('One.'), sentence('Two.'), sentence('Three.')]));
  assert.deepEqual(context.pending.sentences.map(item => item.text), ['One.', 'Two.', 'Three.']);
  assert.equal(context.pending.sentences[0], first, 'already released sentences keep their identity');
});

test('the first view of a context is taken whole, with badges', () => {
  const first = mergePending(null, view(4, [sentence('One.'), sentence('Fixed.', 'template', 'number: 7')]));
  assert.equal(first.revision, 4);
  assert.deepEqual(first.pending.sentences.map(item => item.badge), ['Model', 'Template (model output failed verification)']);
  assert.equal(mergePending(null, {id: 'c1', revision: 0, busy: false, messages: []}).pending, null);
});

test('a revision change drops the pending sentences', () => {
  const streaming = mergePending(view(1, []), view(1, [sentence('One.'), sentence('Two.')]));
  const cleared = mergePending(streaming, {id: 'c1', revision: 2, busy: false, messages: [], pending: null});
  assert.equal(cleared.pending, null);
  assert.equal(cleared.revision, 2);
  const next = mergePending(streaming, view(3, [sentence('New.')]));
  assert.deepEqual(next.pending.sentences.map(item => item.text), ['New.']);
  // An older revision arriving late is ignored.
  assert.equal(mergePending(cleared, view(1, [sentence('Late.')])), cleared);
});

test('a finished turn replaces the pending view, and a stale busy view cannot resurrect it', () => {
  const streaming = mergePending(view(1, []), view(1, [sentence('One.')]));
  const done = mergePending(streaming, {id: 'c1', revision: 1, busy: false, pending: null,
    messages: [{role: 'user', text: 'Why?'}, {role: 'assistant', text: 'One.', reply: {status: 'completed', sentences: [sentence('One.')]}}]});
  assert.equal(done.busy, false);
  assert.equal(done.pending, null);
  assert.equal(done.messages.length, 2);
  assert.equal(mergePending(done, view(1, [sentence('One.')])), done);
});

test('a template badge is set when source is template', () => {
  const merged = mergePending(view(1, []), view(1, [sentence('One.'), sentence('Fixed text.', 'template', 'number: 99 is not in the evidence'),
    sentence('Fixed text.', 'template', 'deadline: answer deadline')]));
  assert.deepEqual(merged.pending.sentences.map(item => item.badge),
    ['Model', 'Template (model output failed verification)', 'Template (model unavailable)']);
  assert.equal(badge(sentence('x', 'template', 'generation_error: server stopped')), 'Template (model unavailable)');
  assert.equal(badge(sentence('x', 'template', 'empty_output: the model wrote no sentence')), 'Template (model unavailable)');
  assert.equal(badge(sentence('x', 'template', 'feature: unknown name')), 'Template (model output failed verification)');
  assert.equal(badge(sentence('x')), 'Model');
});

test('a growing text keeps revealing from the current position and restarts when its prefix changes', () => {
  assert.equal(revealFrom('', 'One.', 0), 0);
  assert.equal(revealFrom('One.', 'One. Two.', 4), 4);
  assert.equal(revealFrom('One.', 'One. Two.', 2), 2);
  assert.equal(revealFrom('One.', 'Two.', 4), 0);
  assert.equal(revealFrom('One. Two.', 'One.', 9), 0);
  // Positions count characters, not UTF-16 units.
  assert.equal(revealFrom('Où', 'Où 😀 ok', 2), 2);
  assert.equal(revealFrom('😀😀', '😀😀 ok', 5), 2);
});

test('server-sent view events are parsed across chunk boundaries', async () => {
  const first = {id: 'c1', revision: 1, busy: true}, second = {id: 'c1', revision: 1, busy: false};
  const wire = `: keep-alive\n\nevent: view\ndata: ${JSON.stringify(first)}\n\nevent: view\r\ndata: ${JSON.stringify(second)}\r\n\r\n`;
  assert.deepEqual(parseEvents(wire), {views: [first, second], rest: ''});
  const cut = wire.length - 7;
  const head = parseEvents(wire.slice(0, cut));
  assert.deepEqual(head.views, [first]);
  assert.deepEqual(parseEvents(head.rest + wire.slice(cut)).views, [second]);
  const encoder = new TextEncoder();
  const body = new ReadableStream({start(controller) {for (let i = 0; i < wire.length; i += 11) controller.enqueue(encoder.encode(wire.slice(i, i + 11))); controller.close();}});
  const seen = [];
  await readEvents(body, value => seen.push(value));
  assert.deepEqual(seen, [first, second]);
});

test('each reply shows its model, device, seconds and check result', () => {
  assert.equal(replyMeta({status: 'completed', model: 'Qwen3-4B', device: 'gpu', llm_executed: true, grounding: {seconds: 2.345, passed: true}}),
    'Qwen3-4B · GPU · 2.3 s · every sentence passed the checks');
  assert.equal(replyMeta({status: 'completed', model: 'Qwen3-4B', device: 'cpu', llm_executed: true, grounding: {seconds: 14, passed: false, template_label: 'Template (model output failed verification)'}}),
    'Qwen3-4B · CPU (slower) · 14.0 s · a sentence failed the checks; NetSec template text follows');
  assert.equal(replyMeta({status: 'completed', model: 'Qwen3-4B', device: null, llm_executed: false, grounding: {seconds: 15.2, passed: false, template_label: 'Template (model unavailable)'}}),
    'Qwen3-4B · 15.2 s · model unavailable; NetSec template text');
  // The first sentence failed: no model text was shown, so no device is claimed.
  assert.equal(replyMeta({status: 'completed', model: 'Qwen3-4B', device: null, llm_executed: false, grounding: {seconds: 1.5, passed: false, template_label: 'Template (model output failed verification)'}}),
    'Qwen3-4B · 1.5 s · the model text failed the checks; NetSec template text');
  assert.equal(replyMeta({status: 'completed', model: 'Qwen3-4B', device: 'gpu', llm_executed: true, grounding: {seconds: 60, passed: false, template_label: 'Template (model unavailable)'}}),
    'Qwen3-4B · GPU · 60.0 s · the model stopped early; NetSec template text follows');
  assert.equal(replyMeta({status: 'out_of_scope', llm_executed: false, grounding: {mode: 'application_policy'}}), '');
});

test('fixed NetSec text (the explanation footer) is labelled NetSec, never Model or Template', () => {
  const footer = {text: 'XAI shows how the model reached its answer, not whether it is correct.', source: 'netsec', part: 'footer', violation: null};
  assert.equal(NETSEC_LABEL, 'NetSec text');
  assert.equal(badge(footer), 'NetSec text');
  const merged = mergePending(null, view(1, [sentence('One.'), footer]));
  assert.deepEqual(merged.pending.sentences.map(item => item.badge), ['Model', 'NetSec text']);
  // The footer is shown apart from the answer, once, after it.
  const answer = [sentence('One.'), sentence('Fixed.', 'template', 'number: 7'), footer];
  assert.deepEqual(splitFooter(answer), {body: answer.slice(0, 2), footer: [footer]});
  assert.deepEqual(splitFooter([sentence('One.')]), {body: [sentence('One.')], footer: []});
});

test('without a selected XAI method the explainer does not start and says so', () => {
  assert.equal(CHOOSE_XAI, 'Choose an XAI method and generate its explanation first; the plain-English explainer only explains a selected XAI result.');
  const base = {id: 'c', revision: 0, messages: []};
  assert.equal(xaiRequired(null), '');
  assert.equal(xaiRequired({...base, xai_method: 'shap', xai_required: null}), '');
  assert.equal(xaiRequired({...base, xai_method: null, xai_required: 'Server text.'}), 'Server text.');
  // A context without the field (or with an unknown method and no message) still never starts the explainer.
  assert.equal(xaiRequired({...base}), CHOOSE_XAI);
  assert.equal(xaiRequired({...base, xai_method: null}), CHOOSE_XAI);
});
