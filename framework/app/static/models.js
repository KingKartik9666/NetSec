"use strict";
// Independent four-family workspace. Text is rendered safely with textContent.
(() => {
  const $ = (id) => document.getElementById(id);
  const families = [
    ["random_forest", "Random Forest", "Fixed reference baseline"],
    ["xgboost", "XGBoost", "Saved trees & evaluated upgrades"],
    ["kan", "KAN", "Genuine spline network"],
    ["hem", "HEM", "Original ensemble components"]
  ];
  const state = {catalog: [], selected: "xgboost", metadata: null, prediction: null,
    samples: [], busy: false, request: 0, explanationRequest: 0, rowIndex: 0};
  const node = (tag, text, className = "") => {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = String(text);
    if (className) element.className = className;
    return element;
  };
  const number = (value, places = 4) => typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString("en-US", {maximumFractionDigits: places}) : "Unavailable";
  const percent = (value) => typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : "Unavailable";
  const label = (id) => families.find(([key]) => key === id)?.[1] || id;
  const fail = (id, error = "") => {$(id).textContent = error; $(id).hidden = !error;};
  const idOf = (entry) => entry.model_id || entry.id || entry.family;
  const capsOf = (metadata) => {
    const caps = metadata?.capabilities || {};
    return Array.isArray(caps) ? Object.fromEntries(caps.map((name) => [name, true])) : caps;
  };
  async function api(path, options = {}) {
    const response = await fetch(path, {credentials: "same-origin", signal: AbortSignal.timeout(180000), ...options});
    const data = await response.json();
    if (!response.ok) {
      const error = new Error(data.error || `Request failed (${response.status})`);
      error.capability = data.capability; error.modelId = data.model_id; error.status = data.status;
      throw error;
    }
    return data;
  }
  const post = (path, payload) => api(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
  function controls() {
    const supported = state.metadata && capsOf(state.metadata).predict !== false;
    $("model-predict-sample").disabled = state.busy || !supported || !$("model-sample").value;
    $("model-predict-row").disabled = state.busy || !supported;
    $("model-predict-csv").disabled = state.busy || !supported || !$("model-csv").files.length;
    $("model-explain").disabled = state.busy || !state.prediction || capsOf(state.metadata).explain === false;
    $("model-explain-method").disabled = state.busy || !state.prediction || capsOf(state.metadata).explain === false;
    $("models-refresh").disabled = state.busy;
    $("model-predict-sample").textContent = state.busy ? "Running selected model…" : "Run selected model ↗";
  }
  function renderCatalog() {
    $("model-selector").replaceChildren(...families.map(([id, title, note]) => {
      const info = state.catalog.find((entry) => idOf(entry) === id) || {};
      const button = node("button", undefined, "model-choice");
      button.type = "button";
      button.dataset.modelId = id;
      button.setAttribute("aria-pressed", String(state.selected === id));
      button.append(node("strong", title), node("small", note), node("span", info.status || "Not loaded", `model-availability ${info.status || ""}`));
      if (info.reason || info.error) button.title = info.reason || info.error;
      button.addEventListener("click", () => chooseModel(id));
      return button;
    }));
  }
  function metadataValue(metadata, key) {
    return metadata[key] ?? metadata.model?.[key] ?? metadata.architecture?.[key];
  }
  function renderMetadata(metadata) {
    const caps = capsOf(metadata);
    const methods = Array.isArray(caps.explain) ? caps.explain : ["native"];
    const methodLabels = {native: "Native model explanation", shap: "SHAP", lime: "LIME approximation", global: "Global tree statistics", ablation: "Feature ablation", occlusion: "Feature occlusion"};
    $("model-explain-method").replaceChildren(...methods.map((method) => {const option = node("option", methodLabels[method] || method); option.value = method; return option;}));
    $("selected-model-description").textContent = metadata.description || metadata.scope || metadata.implementation || "Inspect the recorded architecture, input contract and evaluation limits.";
    $("selected-model-state").textContent = metadata.status || "METADATA VERIFIED";
    $("model-capabilities").replaceChildren(...Object.entries(caps).map(([key, supported]) =>
      node("span", `${key.replaceAll("_", " ")}: ${supported === false ? "unavailable" : typeof supported === "string" ? supported : "supported"}`, `capability-badge${supported === false ? " unsupported" : ""}`)));
    const names = metadata.feature_names || metadata.schema?.feature_names || [];
    const classes = metadata.classes || metadata.schema?.classes || [];
    const values = [
      ["Version", metadataValue(metadata, "version") || metadataValue(metadata, "run_id")],
      ["Dataset", metadata.dataset?.name || metadata.dataset || metadata.dataset_name],
      ["Features", names.length || metadata.features],
      ["Classes", classes.length || metadata.class_count],
      ["Execution", "CPU · offline · no credentials"],
      ["Implementation", metadata.source || metadata.source_repository || metadata.implementation],
      ["Unknown attacks", caps.novelty ? "Supported only under recorded protocol" : "Not evaluated / unavailable"]
    ];
    const details = document.createDocumentFragment();
    values.forEach(([key, value]) => {
      if (value && typeof value === "object") value = JSON.stringify(value);
      details.append(node("dt", key), node("dd", value === null || value === undefined ? "Unavailable" : value));
    });
    $("model-metadata").replaceChildren(details);
    $("model-schema").replaceChildren(node("p", `${names.length} ordered feature names. Only compatible measured inputs are accepted.`, "field-help"),
      node("p", names.join("\n"), "model-schema-list"));
    const limitations = metadata.limitations || metadata.model?.limitations || [];
    $("selected-model-limitations").replaceChildren(...(Array.isArray(limitations) ? limitations : [limitations]).map((item) => node("li", item)));
  }
  async function chooseModel(id) {
    const generation = ++state.request;
    ++state.explanationRequest;
    state.selected = id; state.metadata = null; state.prediction = null; state.busy = true; state.rowIndex = 0;
    $("selected-model-title").textContent = label(id);
    $("selected-model-state").textContent = "CHECKING";
    $("selected-model-description").textContent = "Checking this model independently…";
    $("model-prediction-result").replaceChildren(node("p", "No prediction requested for this model.", "field-help"));
    $("model-explanation-result").replaceChildren();
    ["selected-model-error", "model-prediction-error", "model-explanation-error"].forEach((errorId) => fail(errorId));
    $("model-metadata").replaceChildren(); $("model-capabilities").replaceChildren();
    renderCatalog(); controls();
    try {
      const data = await api(`/api/models/${id}/status`);
      if (generation !== state.request) return;
      state.metadata = data.metadata || data;
      renderMetadata(state.metadata);
      $("models-announcement").textContent = `${label(id)} selected. Model capability status is available.`;
    } catch (error) {
      if (generation !== state.request) return;
      fail("selected-model-error", error.message);
      $("selected-model-state").textContent = (error.status || "UNAVAILABLE").toUpperCase();
      $("selected-model-description").textContent = "This model capability is unavailable. Other models, pages and independent monitoring remain accessible.";
    } finally {
      if (generation === state.request) {state.busy = false; controls();}
    }
  }
  function renderPrediction(data) {
    const prediction = data.predictions?.[state.rowIndex];
    if (!prediction || typeof prediction.label !== "string") throw new Error("The selected model returned no valid prediction.");
    const result = document.createDocumentFragment();
    result.append(node("span", `${label(data.model_id)} · ${data.source?.kind || "Supplied features"}`, "eyebrow"),
      node("h3", prediction.display_label || prediction.label, "model-result-title"));
    result.append(node("p", typeof prediction.score === "number" ? `${percent(prediction.score)} · uncalibrated model score`
      : "This ensemble provides a class vote; no supported probability is available.", "model-result-score"));
    result.append(node("p", "A model hypothesis requires human verification. It does not confirm an attack or prove safety.", "field-help"));
    const values = data.features?.[state.rowIndex] || [], names = data.metadata?.feature_names || [];
    const observed = values.filter((value) => value !== null).length;
    result.append(node("p", `${observed} of ${values.length} inputs measured · Version ${data.metadata?.version || "unavailable"}`, "field-help"));
    if (data.predictions.length > 1) {
      const batchLabel = node("label", `Inspect one of ${data.predictions.length} classified rows`, "field-label"); batchLabel.htmlFor = "model-batch-row";
      const batch = node("select"); batch.id = "model-batch-row";
      data.predictions.forEach((item, index) => {const option = node("option", `Row ${index + 1} · ${item.display_label || item.label}`); option.value = index; option.selected = index === state.rowIndex; batch.append(option);});
      batch.addEventListener("change", () => {state.rowIndex = Number(batch.value); ++state.explanationRequest; fail("model-explanation-error"); $("model-explanation-result").replaceChildren(); renderPrediction(data); controls();});
      result.append(batchLabel, batch);
    }
    const alternatives = prediction.top_classes || [];
    alternatives.forEach((item) => {
      const line = node("div", undefined, "model-top-category");
      line.append(node("span", item.display_label || item.label), node("strong", percent(item.score))); result.append(line);
    });
    if (data.source?.validation_label !== undefined) result.append(node("p", `Source label: ${data.source.validation_label} · ${data.source.validation_label === prediction.label ? "Prediction matches" : "Prediction differs"}`, "field-help"));
    if (prediction.novelty) result.append(node("p", `Unknown-attack status: ${prediction.novelty}`, "field-help"));
    if (names.length === values.length) {
      const coverage = node("details", undefined, "model-json-details"), table = node("table"), body = node("tbody");
      coverage.append(node("summary", "Measured values & missing inputs"));
      names.forEach((name, index) => {const row = node("tr"); row.append(node("td", name), node("td", values[index] === null ? "Missing · unavailable" : number(values[index]))); body.append(row);});
      table.append(body); coverage.append(table); result.append(coverage);
    }
    const details = node("details", undefined, "model-json-details"); details.append(node("summary", "Prediction & source provenance"), node("pre", JSON.stringify(data, null, 2))); result.append(details);
    $("model-prediction-result").replaceChildren(result);
  }
  async function predict(payload, uploadFile = null) {
    const modelId = state.selected;
    const generation = ++state.request;
    ++state.explanationRequest;
    state.busy = true; state.prediction = null; state.rowIndex = 0;
    fail("model-prediction-error"); fail("model-explanation-error");
    $("model-explanation-result").replaceChildren(); controls();
    try {
      let data;
      if (uploadFile) {const form = new FormData(); form.append("file", uploadFile); data = await api(`/api/models/${modelId}/upload`, {method: "POST", body: form});}
      else data = await post(`/api/models/${modelId}/predict`, payload);
      if (generation !== state.request || modelId !== state.selected) return;
      renderPrediction(data); state.prediction = data;
      $("models-announcement").textContent = `${label(modelId)} predicted ${data.predictions[0].label}. Explanation can be requested separately.`;
    } catch (error) {
      if (generation !== state.request) return;
      fail("model-prediction-error", error.message);
      $("model-prediction-result").replaceChildren(node("p", "No prediction was returned. No other model was substituted.", "field-help"));
    } finally {if (generation === state.request) {state.busy = false; controls();}}
  }
  function renderExplanation(data) {
    const raw = data.explanation;
    const explanation = Array.isArray(raw) ? raw[0] : raw;
    if (!explanation || typeof explanation !== "object") throw new Error("The explanation capability returned no evidence.");
    const content = document.createDocumentFragment();
    content.append(node("h3", explanation.method || "Recorded model explanation"));
    if (explanation.scope || explanation.limitations) content.append(node("p", explanation.scope || explanation.limitations, "field-help"));
    const features = explanation.important_features || explanation.features || explanation.contributions || [];
    const weight = (item) => Number(item.shap_value ?? item.contribution ?? item.weight ?? item.importance ?? item.effect ?? 0);
    const maximum = Math.max(0.000001, ...features.map((item) => Math.abs(weight(item))));
    features.slice(0, 20).forEach((feature) => {
      const value = weight(feature);
      const line = node("div", undefined, "research-xai-row");
      const name = node("div", undefined, "feature-name"); name.append(node("strong", feature.feature || feature.name || "Feature"));
      if (feature.is_missing || feature.value === null) name.append(node("small", " · missing-value routing"));
      const track = node("div", undefined, "research-xai-track"), bar = node("div", undefined, `research-xai-bar${value < 0 ? " negative" : ""}`);
      bar.style.width = `${Math.min(100, Math.abs(value) / maximum * 100)}%`; track.append(bar);
      line.append(name, track, node("span", `${value > 0 ? "+" : ""}${number(value)}`, "feature-weight")); content.append(line);
    });
    const details = node("details", undefined, "model-json-details"); details.append(node("summary", "Complete explanation & method provenance"), node("pre", JSON.stringify(data, null, 2))); content.append(details);
    $("model-explanation-result").replaceChildren(content);
  }
  async function explain() {
    if (!state.prediction) return;
    const observation = state.prediction, modelId = state.selected, generation = ++state.explanationRequest;
    $("model-explain").disabled = true; fail("model-explanation-error");
    try {
      const data = await post(`/api/models/${modelId}/explain`, {rows: [observation.features[state.rowIndex]], method: $("model-explain-method").value});
      if (generation !== state.explanationRequest || modelId !== state.selected || observation !== state.prediction) return;
      renderExplanation(data);
    } catch (error) {
      if (generation === state.explanationRequest) fail("model-explanation-error", `${error.message} The existing prediction remains available.`);
    } finally {if (generation === state.explanationRequest) controls();}
  }
  function renderComparison(data) {
    $("model-comparison-note").textContent = data.reason || data.note || data.scope || data.comparison_basis || "Read the recorded evaluation protocol before comparing validation metrics.";
    const table = node("table"), head = node("thead"), heading = node("tr");
    ["Family / version", "Accuracy", "Macro F1", "Latency", "Evaluation conditions"].forEach((title) => heading.append(node("th", title))); head.append(heading); table.append(head);
    const body = node("tbody");
    const rows = data.models || data.rows || data.runs || [];
    families.forEach(([id, title]) => {
      const entry = rows.find((value) => idOf(value) === id) || {}, metrics = entry.metrics || entry.validation_metrics || {};
      const row = node("tr"), identity = node("td", title); identity.append(node("small", entry.version || entry.run_id || "No recorded version"));
      const conditions = node("td", entry.comparable === false ? "Incomparable" : entry.status || "Recorded validation");
      conditions.append(node("small", entry.reason || entry.scope || entry.protocol?.description || "No documented comparison conditions"));
      row.append(identity, node("td", percent(metrics.accuracy)), node("td", number(metrics.macro_f1)),
        node("td", typeof metrics.latency_ms === "number" ? `${number(metrics.latency_ms, 2)} ms` : "Unavailable"), conditions); body.append(row);
    });
    table.append(body); $("model-comparison-table").replaceChildren(table);
    const details = node("details", undefined, "model-json-details"); details.append(node("summary", "Comparison protocol & raw recorded metrics"), node("pre", JSON.stringify(data, null, 2))); $("model-comparison-table").append(details);
  }
  async function refresh() {
    fail("models-catalog-error"); fail("model-comparison-error");
    const results = await Promise.allSettled([api("/api/models"), api("/api/models/comparison"), api("/api/samples")]);
    if (results[0].status === "fulfilled") {state.catalog = results[0].value.models || []; renderCatalog();}
    else fail("models-catalog-error", results[0].reason.message);
    if (results[1].status === "fulfilled") renderComparison(results[1].value);
    else fail("model-comparison-error", results[1].reason.message);
    if (results[2].status === "fulfilled") {
      state.samples = results[2].value.samples || [];
      $("model-sample").replaceChildren(...state.samples.map((sample) => {
        const option = node("option", sample.name || sample.display_name || sample.label); option.value = sample.id; return option;
      }));
      $("model-sample").disabled = !state.samples.length;
      $("model-sample-status").textContent = "The same observation is mapped by verified feature names into each compatible model.";
    } else {
      $("model-sample").replaceChildren(node("option", "Shared samples unavailable")); $("model-sample").disabled = true;
      $("model-sample-status").textContent = "Included samples are unavailable. Independent models can still classify compatible ordered feature values.";
    }
    await chooseModel(state.selected);
  }
  $("model-sample").addEventListener("change", controls);
  $("model-csv").addEventListener("change", controls);
  $("model-predict-csv").addEventListener("click", () => {
    const file = $("model-csv").files[0];
    if (!file) return;
    if (file.size > 10 * 1024 * 1024) {fail("model-prediction-error", "Upload exceeds 10 MB; export a CSV batch of at most 500 rows."); return;}
    predict(null, file);
  });
  $("model-predict-sample").addEventListener("click", () => predict({sample_id: Number($("model-sample").value)}));
  $("model-predict-row").addEventListener("click", () => {
    try {const row = JSON.parse($("model-input-row").value); if (!Array.isArray(row)) throw new Error("Enter one JSON array in the selected model's feature order."); predict({rows: [row]});}
    catch (error) {fail("model-prediction-error", error.message);}
  });
  $("model-explain").addEventListener("click", explain);
  $("models-refresh").addEventListener("click", refresh);
  renderCatalog(); refresh().catch((error) => fail("models-catalog-error", error.message));
})();
