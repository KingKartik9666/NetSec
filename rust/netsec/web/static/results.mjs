import {$, node, percent, number, details, metadata, warnings} from './dom.mjs';
import {agreement, scoreLabel, rankedFeatures} from './state.mjs';

export function renderResults(job, rowIndex, focused, onFocus) {
  $('job-empty').hidden = true; $('job-content').hidden = false;
  $('job-status').textContent = String(job.status || 'unknown').toUpperCase();
  const total = job.progress?.total ?? job.models?.length ?? 0;
  const completed = job.progress?.completed ?? (job.models || []).filter(model => ['completed', 'failed', 'cancelled', 'incompatible'].includes(model.status)).length;
  $('job-progress').max = Math.max(total, 1); $('job-progress').value = Math.min(completed, total);
  $('progress-label').textContent = `${completed} of ${total} model operations finished`;
  $('job-source').textContent = `Input ${job.input?.id || 'pending identity'} · ${job.input?.row_count || 1} observation(s) · ${job.input?.origin || 'source not yet reported'}. Schema identity is retained in the JSON export.`;
  const tally = agreement(job.models || [], rowIndex);
  $('agreement-note').textContent = tally.total > 1
    ? `${tally.leadingCount} of ${tally.total} completed models share ${tally.ties ? 'a joint leading category' : 'the leading category'}. ${tally.labels.length === 1 ? 'These model predictions agree.' : `${tally.labels.length} different categories are predicted.`} Agreement is descriptive; it is not calibrated confidence or ground truth.`
    : tally.total ? 'One completed prediction is available. More model results may arrive independently.' : 'Completed model predictions will appear independently.';
  $('model-results').replaceChildren(...(job.models || []).map(model => {
    const article = node('article', undefined, `result-item${focused === model.model_id ? ' focused' : ''}`);
    article.dataset.status = model.status;
    const heading = node('div', undefined, 'result-item-heading');
    heading.append(node('h3', model.label || model.model_id), node('span', `${model.status || 'queued'} · ${number((model.elapsed_ms || 0) / 1000, 2)} s`, 'result-status'));
    article.append(heading);
    const prediction = model.status === 'completed' && model.predictions?.[rowIndex];
    if (prediction) {
      article.append(node('h4', prediction.display_label || prediction.label, 'model-result-title'));
      const score = prediction.score == null ? 'Class vote · no supported ensemble class probability' : `${percent(prediction.score)} · ${scoreLabel(prediction.score_kind)}`;
      article.append(node('p', score, 'field-help'));
      if (prediction.attack_probability != null) article.append(node('p', `${percent(prediction.attack_probability)} · calibrated binary attack score`, 'field-help'));
      const alternatives = node('ul', undefined, 'result-alternatives');
      for (const item of prediction.top_classes || []) alternatives.append(node('li', `${item.display_label || item.label} · ${percent(item.score)}`));
      if (alternatives.childNodes.length) article.append(alternatives);
      const inspect = node('button', focused === model.model_id ? 'Inspecting this model ↓' : 'Inspect this model ↓', 'text-button');
      inspect.type = 'button'; inspect.setAttribute('aria-pressed', String(focused === model.model_id));
      inspect.addEventListener('click', () => onFocus(model.model_id)); article.append(inspect);
    } else if (model.error) article.append(node('p', typeof model.error === 'string' ? model.error : JSON.stringify(model.error), 'field-help'));
    else article.append(node('p', model.status === 'completed' ? 'No prediction was returned for this row.' : model.status === 'cancelled' ? 'This operation was cancelled.' : 'Waiting for this model’s own result.', 'field-help'));
    if (model.artifact_sha256) article.append(node('p', `Artifact ${model.artifact_sha256}`, 'result-artifact'));
    return article;
  }));
}

export const methods = [
  {id: 'shap', label: 'SHAP', note: 'Check the exact explained output and units. A branch attribution must be identified separately from a hybrid’s final output.'},
  {id: 'lime', label: 'LIME · local surrogate', note: 'Synthetic perturbations approximate local behavior. Inspect fidelity and the surrogate/model-score gap.'},
  {id: 'anchors', label: 'Anchors · local rules', note: 'Inspect rule precision, coverage and reference distribution. Local rules do not establish causation.'},
  {id: 'counterfactual', label: 'Constrained counterfactuals', note: 'Alternative valid inputs are model probes, not causal remedies or mitigation instructions.'},
  {id: 'permutation', label: 'Grouped permutation importance', note: 'A global reference-data comparison. Related measurements must be permuted together; inspect score changes and reference provenance.'},
  {id: 'ale', label: 'ALE · accumulated local effects', note: 'A global reference-range summary. Inspect bins and support; sparse regions do not justify extrapolation.'}
];

export function renderExplanation(explanation, container = 'xai-content') {
  const fragment = document.createDocumentFragment();
  fragment.append(node('h4', explanation.method_label || explanation.method), metadata([
    ['Target', explanation.target_label], ['Scope', explanation.scope], ['Units', explanation.units]
  ]));
  const state = explanation.details?.state;
  const states = {insufficient_support: 'Insufficient reference support for a reliable anchor. Inspect the measured precision and its lower bound.',
    no_feasible_counterfactual: 'No feasible observed-reference counterfactual was found within this evaluation budget.',
    observed_reference_counterfactual: 'An observed reference changes this model’s decision. It is a model probe, not a causal remedy.',
    unsupported: `This method is unsupported for the current observation/reference.${explanation.details?.reason ? ` ${explanation.details.reason}` : ''}`};
  if (states[state]) fragment.append(node('p', states[state], 'field-help explanation-state'));
  // Counterfactual group differences concern the complete candidate. Repeating
  // its score delta as one bar per changed group would imply local attribution.
  const features = explanation.method === 'counterfactual' ? [] : rankedFeatures(explanation.features || []);
  const maximum = Math.max(...features.map(item => Math.abs(item.contribution || 0)), Number.EPSILON);
  const list = node('ol', undefined, 'xai-feature-list');
  for (const feature of features) {
    const item = node('li');
    const label = node('div', undefined, 'xai-feature-label');
    label.append(node('span', feature.feature), node('span', number(feature.contribution, 6)));
    const track = node('div', undefined, 'xai-track'), bar = node('span', undefined, `xai-bar${feature.contribution < 0 ? ' negative' : ''}`);
    bar.style.width = `${Math.min(Math.abs(feature.contribution || 0) / maximum * 100, 100)}%`;
    const observed = feature.is_group ? `joint group · ${feature.observed_count} of ${feature.group_size} measurements observed` : feature.value == null ? 'missing measurement' : number(feature.value);
    track.append(bar); item.append(label, track, node('p', `Input: ${observed}`, 'field-help')); list.append(item);
  }
  if (features.length) fragment.append(list, node('p', `Showing ${features.length} ranked features. Complete evidence remains in the export.`, 'field-help'));
  if (explanation.quality) fragment.append(details('Quality checks & method settings', explanation.quality));
  fragment.append(warnings(explanation.warnings), details('Explanation evidence & provenance', explanation));
  $(container).replaceChildren(fragment);
}

export function renderEnglish(result, container = 'english-content') {
  const fragment = document.createDocumentFragment();
  fragment.append(node('p', result.text, 'english-text'), metadata([['Model', result.model], ['Provider', result.provider], ['Status', result.status]]));
  fragment.append(warnings(result.warnings), details('Generation & grounding audit', result));
  $(container).replaceChildren(fragment);
}
