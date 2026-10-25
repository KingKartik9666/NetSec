import {$, node} from './dom.mjs';
import {eligibleModels, requestedModels} from './state.mjs';

export function renderCatalog(catalog, selected, mode, onChange) {
  const eligible = new Set(eligibleModels(catalog.models || [], catalog.schema).map(model => model.id));
  $('model-catalog').replaceChildren(...(catalog.models || []).map(model => {
    const label = node('label', undefined, 'model-choice');
    const input = node('input'); input.type = 'checkbox'; input.value = model.id;
    input.checked = eligible.has(model.id) && (mode === 'extensive' || selected.has(model.id));
    input.disabled = !eligible.has(model.id) || mode === 'extensive';
    input.setAttribute('aria-label', `${model.label} · ${model.status || 'availability unknown'}`);
    input.addEventListener('change', () => onChange(model.id, input.checked));
    const body = node('span');
    body.append(node('strong', model.label), node('small', `${model.kind || 'Saved model'} · ${model.feature_count ?? '?'} inputs · ${model.class_count ?? '?'} classes`));
    const status = eligible.has(model.id) ? model.status : model.status === 'ready' ? 'incompatible schema' : model.status;
    body.append(node('span', status || 'Unavailable', `model-availability ${eligible.has(model.id) ? 'ready' : 'unavailable'}`));
    if (model.reason) body.append(node('small', model.reason));
    if (!eligible.has(model.id) && !model.reason) body.append(node('small', 'Prediction is unavailable for the current input schema.'));
    label.append(input, body); return label;
  }));
  const ids = requestedModels(catalog.models || [], selected, mode, catalog.schema);
  $('selection-note').textContent = mode === 'extensive'
    ? `Extensive mode will snapshot ${ids.length} compatible available models. Each result retains its own status and artifact identity.`
    : `${ids.length} model${ids.length === 1 ? '' : 's'} selected. Only compatible measured inputs are accepted.`;
  $('home-model-count').textContent = `${eligible.size} compatible model${eligible.size === 1 ? '' : 's'} available · local inference`;
  $('schema-summary').textContent = `${catalog.schema?.feature_names?.length || 0} ordered features · ${catalog.schema?.classes?.length || 0} categories. Full-input model compatibility is checked independently. Missing measurements are never invented.`;
  $('schema-names').replaceChildren(...(catalog.schema?.feature_names || []).map(name => node('li', name)));
  $('csv-schema-note').textContent = `Use all ${catalog.schema?.feature_names?.length || 0} prepared feature names in the displayed schema. Blank measurements stay missing; 0 means measured zero. The runtime checks each model’s supported input profile.`;
}
