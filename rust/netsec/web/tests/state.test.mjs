import test from 'node:test';
import assert from 'node:assert/strict';
import {Generation, eligibleModels, requestedModels, validateRows, agreement, scoreLabel, rankedFeatures} from '../src/state.ts';

const schema = {feature_names: ['measured', 'indicator']};
const model = (id, extra = {}) => ({id, label: id, status: 'ready', feature_count: 2, capabilities: {prediction: true}, ...extra});

test('extensive mode snapshots all available schema-compatible models only', () => {
  const catalog = [model('a'), model('b'), model('missing', {status: 'unavailable'}), model('live', {feature_count: 7}), model('no-predict', {capabilities: {prediction: false}})];
  assert.deepEqual(eligibleModels(catalog, schema).map(item => item.id), ['a', 'b']);
  assert.deepEqual(requestedModels(catalog, new Set(['a']), 'extensive', schema), ['a', 'b']);
  assert.deepEqual(requestedModels(catalog, new Set(['b', 'missing']), 'selected', schema), ['b']);
});

test('input validation preserves measured zero and missing null without coercion', () => {
  assert.deepEqual(validateRows([0, null], 2), [[0, null]]);
  assert.deepEqual(validateRows([[0, null], [3, 1]], 2), [[0, null], [3, 1]]);
  for (const input of [[true, 1], ['0', 1], [Infinity, 0], [null, null], [1], [], Array.from({length: 501}, () => [1, 0])]) {
    assert.throws(() => validateRows(input, 2));
  }
});

test('late asynchronous responses are invalid after input or model change', () => {
  const generation = new Generation();
  const first = generation.value;
  const old = generation.controller('explanation');
  generation.invalidate();
  assert.equal(generation.current(first), false);
  assert.equal(old.signal.aborted, true);
  const next = generation.controller('explanation');
  const newer = generation.controller('explanation');
  assert.equal(next.signal.aborted, true);
  assert.equal(newer.signal.aborted, false);
});

test('agreement uses completed predictions only and does not create confidence', () => {
  const rows = [
    {status: 'completed', predictions: [{label: 'benign'}]},
    {status: 'completed', predictions: [{label: 'attack'}]},
    {status: 'completed', predictions: [{label: 'attack'}]},
    {status: 'failed', predictions: [{label: 'attack'}]},
    {status: 'running'}
  ];
  assert.deepEqual(agreement(rows, 0), {total: 3, leadingCount: 2, labels: [['attack', 2], ['benign', 1]], ties: false});
  assert.equal(agreement(rows, 1).total, 0);
  assert.equal(agreement(rows.slice(0, 2), 0).ties, true);
  assert.match(scoreLabel('correctness_confidence'), /correctness/i);
  assert.doesNotMatch(scoreLabel('correctness_confidence'), /class probability/i);
});

test('visible explanation ranks the strongest signed contributions without changing complete evidence', () => {
  const evidence = [{feature: 'small', contribution: 0.1}, {feature: 'negative', contribution: -4},
    {feature: 'positive', contribution: 2}, {feature: 'invalid', contribution: null}];
  assert.deepEqual(rankedFeatures(evidence, 2).map(item => item.feature), ['negative', 'positive']);
  assert.deepEqual(evidence.map(item => item.feature), ['small', 'negative', 'positive', 'invalid']);
});
