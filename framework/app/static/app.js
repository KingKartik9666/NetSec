"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const state = {status: null, samples: [], rows: [], report: null, busy: false, request: 0, selectedRow: null,
    inputMode: ["live", "sample", "upload"].includes(document.body.dataset.workspaceMode) ? document.body.dataset.workspaceMode : "live", capabilities: null, live: null, liveBusy: false, liveFlight: false, liveTimer: null, liveGeneration: 0, liveStarted: false,
    analysisSource: null, assistant: null, exportingLive: false, exportingLiveCsv: false, liveHistoryKey: null, liveModelKey: null,
    insights: {generation: 0, context: null, xai: null, ai: null, xaiBusy: false, aiBusy: false}};
  const number = (value, digits = 3) => {
    if (value === null || value === undefined) return "missing";
    if (!Number.isFinite(Number(value))) return "unavailable";
    const n = Number(value);
    if (n !== 0 && (Math.abs(n) >= 1000000 || Math.abs(n) < 0.001)) return n.toExponential(3);
    return n.toLocaleString("en-US", {maximumFractionDigits: digits});
  };
  const percent = (value) => value !== null && value !== undefined && Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(2)}%` : "—";
  const signed = (value) => `${Number(value) > 0 ? "+" : ""}${number(value, 4)}`;
  const node = (tag, className = "", content) => {
    const result = document.createElement(tag);
    if (className) result.className = className;
    if (content !== undefined) result.textContent = String(content);
    return result;
  };
  const append = (parent, ...children) => {parent.append(...children); return parent;};
  const motion = {preference: window.matchMedia("(prefers-reduced-motion: reduce)"), transition: null};

  function setupWorkspaceMotion() {
    const visual = document.querySelector(".heading-visual");
    const stage = document.querySelector(".network-stage");
    if (!visual || !stage) return;
    let frame = null;
    let visible = true;
    let pointerX = 0;
    let pointerY = 0;
    const clamp = (value, bound) => Math.max(-bound, Math.min(bound, value));
    const reset = () => {
      if (frame !== null) cancelAnimationFrame(frame);
      frame = null;
      pointerX = 0; pointerY = 0;
      stage.style.setProperty("--pointer-x", "0deg");
      stage.style.setProperty("--pointer-y", "0deg");
      stage.style.setProperty("--scroll-drift", "0px");
    };
    const render = () => {
      frame = null;
      if (motion.preference.matches || document.hidden || !visible) return;
      stage.style.setProperty("--pointer-x", `${pointerX.toFixed(2)}deg`);
      stage.style.setProperty("--pointer-y", `${pointerY.toFixed(2)}deg`);
      stage.style.setProperty("--scroll-drift", `${Math.min(14, Math.max(0, window.scrollY * .035)).toFixed(2)}px`);
    };
    const schedule = () => {
      if (frame === null && !motion.preference.matches && !document.hidden && visible) frame = requestAnimationFrame(render);
    };
    visual.addEventListener("pointermove", (event) => {
      if (event.pointerType === "touch" || motion.preference.matches || !visible) return;
      const bounds = visual.getBoundingClientRect();
      pointerX = clamp((event.clientX - bounds.left - bounds.width / 2) / bounds.width * 24, 12);
      pointerY = clamp((bounds.top + bounds.height / 2 - event.clientY) / bounds.height * 18, 9);
      schedule();
    }, {passive: true});
    visual.addEventListener("pointerleave", () => {pointerX = 0; pointerY = 0; schedule();}, {passive: true});
    window.addEventListener("scroll", schedule, {passive: true});
    document.addEventListener("visibilitychange", () => {if (document.hidden) reset(); else schedule();});
    window.addEventListener("blur", reset);
    motion.preference.addEventListener("change", () => {motion.transition?.skipTransition(); reset();});
    if ("IntersectionObserver" in window) {
      const sceneObserver = new IntersectionObserver(([entry]) => {
        visible = entry.isIntersecting;
        if (visible) schedule(); else reset();
      });
      sceneObserver.observe(stage);
      const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          if (!motion.preference.matches && !document.hidden) {
            entry.target.classList.add("motion-reveal");
            entry.target.addEventListener("animationend", () => entry.target.classList.remove("motion-reveal"), {once: true});
          }
          revealObserver.unobserve(entry.target);
        });
      }, {threshold: .08});
      [$("insights-panel"), $("research-notes")].filter(Boolean).forEach((section) => revealObserver.observe(section));
    }
  }

  async function api(path, options = {}) {
    const response = await fetch(path, {credentials: "same-origin", ...options});
    let data;
    try { data = await response.json(); }
    catch { throw new Error("The app returned an unreadable response. Check the local server and try again."); }
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
    return data;
  }
  function showError(message) {
    $("global-error").textContent = message;
    $("global-error").hidden = !message;
  }
  function refreshControls() {
    const ready = Boolean(state.status) && !state.busy;
    $("sample-select").disabled = !ready || state.samples.length === 0;
    $("analyze-button").disabled = !ready || state.samples.length === 0 || !$("sample-select").value;
    $("csv-file").disabled = !ready;
    $("upload-button").disabled = !ready || !$("csv-file").files.length;
    $("download-report").disabled = state.busy || !state.report;
    for (const button of $("batch-results").querySelectorAll("button")) button.disabled = !ready;
    refreshLiveControls();
    refreshInsightControls();
  }
  function clearReport(loading = false) {
    state.report = null;
    state.analysisSource = null;
    resetInsights();
    $("report-content").hidden = true;
    $("report-content").replaceChildren();
    $("report-empty").hidden = false;
    $("report-empty").classList.toggle("loading", loading);
    $("report-empty").querySelector("h3").textContent = loading ? "Calculating the evidence…" : "Your analysis starts here.";
    $("report-empty").querySelector("p").textContent = loading
      ? "Running the saved model and exact TreeSHAP on the supplied traffic features."
      : "Choose a validation sample or upload traffic features. The model prediction and signed SHAP evidence will appear here.";
    refreshControls();
  }
  function setBusy(busy, message = "") {
    state.busy = busy;
    $("report-panel").setAttribute("aria-busy", String(busy));
    if (message) $("activity-status").textContent = message;
    refreshControls();
  }
  function setupTabs(buttons, panels, onChange = null) {
    const inputTabs = buttons[0]?.id === "live-tab";
    const tablist = buttons[0]?.parentElement;
    let activeIndex = Math.max(0, buttons.findIndex((button) => button.getAttribute("aria-selected") === "true"));
    let activation = 0;
    const updateIndicator = () => {
      if (!inputTabs || !tablist) return;
      const button = buttons[activeIndex];
      tablist.style.setProperty("--tab-left", `${button.offsetLeft}px`);
      tablist.style.setProperty("--tab-width", `${button.offsetWidth}px`);
    };
    const activate = (index, focus = false, withMotion = true) => {
      const direction = index < activeIndex ? -1 : 1;
      const changed = index !== activeIndex;
      const request = ++activation;
      const apply = () => {
        // A superseded transition must not restore an older input source.
        if (request !== activation) return;
        buttons.forEach((button, i) => {
          button.classList.toggle("active", i === index);
          button.setAttribute("aria-selected", String(i === index));
          button.tabIndex = i === index ? 0 : -1;
          panels[i].hidden = i !== index;
        });
        activeIndex = index;
        if (focus) buttons[index].focus();
        if (onChange) onChange(index);
        updateIndicator();
      };
      const animated = withMotion && changed && !motion.preference.matches && !document.hidden;
      if (inputTabs && animated && typeof document.startViewTransition === "function") {
        motion.transition?.skipTransition();
        document.documentElement.dataset.motionDirection = direction < 0 ? "backward" : "forward";
        try {
          const transition = document.startViewTransition(apply);
          motion.transition = transition;
          transition.ready.catch(() => {});
          transition.finished.catch(() => {}).finally(() => {if (motion.transition === transition) motion.transition = null;});
          return;
        } catch { /* The source selection still works when snapshots are unavailable. */ }
      }
      if (inputTabs) motion.transition?.skipTransition();
      apply();
      if (animated && panels[index].animate) {
        panels[index].animate([{opacity: .15, transform: `translateX(${direction * 22}px)`}, {opacity: 1, transform: "translateX(0)"}],
          {duration: inputTabs ? 400 : 260, easing: "cubic-bezier(.22,1,.36,1)"});
      }
    };
    buttons.forEach((button, index) => {
      button.addEventListener("click", () => activate(index));
      button.addEventListener("keydown", (event) => {
        let next;
        if (event.key === "ArrowRight") next = (index + 1) % buttons.length;
        if (event.key === "ArrowLeft") next = (index - 1 + buttons.length) % buttons.length;
        if (event.key === "Home") next = 0;
        if (event.key === "End") next = buttons.length - 1;
        if (next !== undefined) {event.preventDefault(); activate(next, true);}
      });
    });
    if (inputTabs) {
      updateIndicator();
      if ("ResizeObserver" in window) new ResizeObserver(updateIndicator).observe(tablist);
    }
    return activate;
  }

  function evidenceSelection() {
    if (state.inputMode === "live") {
      const live = state.live;
      const report = live?.multiclass_prediction;
      if (!live?.running || live.error || live.state === "error" || !report || report.available === false
          || !Number.isFinite(report.prediction?.score) || typeof report.observation_id !== "string") return null;
      return {key: `live:${report.observation_id}`,
        payload: {kind: "live", observation_id: report.observation_id},
        source: {...report.source, kind: "Actual local partial-input observation", observation_id: report.observation_id,
          observed_at: report.observed_at, interface: live.interface, mode: live.mode, model_version: report.model?.version},
        label: `84-class live observation · ${report.observed_at || report.observation_id.slice(0, 8)} · ${live.interface}`};
    }
    if (!state.report || !state.analysisSource || state.busy) return null;
    const source = state.analysisSource;
    if ((source.kind === "sample") !== (state.inputMode === "sample")) return null;
    return {key: `analysis:${state.request}`, payload: source, source: {...state.report.source},
      label: source.kind === "sample" ? `Validation sample · ${state.report.prediction.display_label || state.report.prediction.label}`
        : `Uploaded CSV row ${state.report.source.row_index ?? ""} · ${state.report.prediction.display_label || state.report.prediction.label}`};
  }
  function resetInsights() {
    state.insights.generation += 1;
    state.insights.context = null;
    state.insights.xai = null;
    state.insights.ai = null;
    state.insights.xaiBusy = false;
    state.insights.aiBusy = false;
    ["extra-xai-content", "assistant-content"].forEach((id) => {$(id).replaceChildren(); $(id).hidden = true;});
    ["extra-xai-error", "assistant-error"].forEach((id) => {$(id).textContent = ""; $(id).hidden = true;});
    $("extra-xai-status").textContent = "Available after a model prediction. Live rules alone have no model attributions.";
    $("assistant-status").textContent = "Generation starts only when requested. Evidence stays on this PC.";
    refreshInsightControls();
  }
  function refreshInsightControls() {
    const selection = evidenceSelection();
    const insights = state.insights;
    $("extra-xai-button").disabled = !selection || insights.xaiBusy || state.liveBusy;
    $("assistant-button").disabled = !selection || !state.assistant?.available || insights.aiBusy || state.liveBusy;
    $("extra-xai-button").setAttribute("aria-busy", String(insights.xaiBusy));
    $("assistant-button").setAttribute("aria-busy", String(insights.aiBusy));
    $("extra-xai-button").textContent = insights.xaiBusy ? "Calculating LIME & statistics…" : "Inspect additional XAI";
    $("assistant-button").querySelector("span").textContent = insights.aiBusy ? "Generating local explanation…" : "Explain in plain English";
    const held = insights.context;
    $("insights-source").textContent = held
      ? `${held.label}${selection?.key !== held.key ? ". This explanation belongs to that selected observation; newer live observations continue above." : ". Evidence is bound to this result."}`
      : selection ? `${selection.label}. Choose an explanation below.`
        : state.inputMode === "live" ? "Wait for a shared-model observation with measured feature values. Rules and the binary comparison do not supply this explanation."
          : "Analyze a sample or an uploaded CSV row to explore its model evidence.";
    $("insights-panel").classList.toggle("has-evidence", Boolean(selection || held));
  }
  function renderAssistantStatus(status) {
    state.assistant = status;
    $("assistant-model-status").textContent = status.available
      ? `${status.model} · ${status.device || "CPU"} · offline · ${status.rag?.chunks || status.rag?.passage_count || "local"} knowledge passages`
      : "Local AI is unavailable in this installation. No template will be substituted for an AI explanation.";
    $("assistant-model-status").classList.toggle("unavailable", !status.available);
    refreshInsightControls();
  }
  function referenceDisclosure(reference) {
    const details = node("details", "xai-reference");
    details.append(node("summary", "", "Training reference & reproducibility"), node("pre", "", JSON.stringify(reference, null, 2)));
    return details;
  }
  function renderLime(lime) {
    const section = node("div", "xai-method-panel");
    section.append(node("p", "explanation-subtitle", `${lime.method} · predicted class “${lime.class_label}”. This is an approximate surrogate, distinct from the exact TreeSHAP evidence above.`));
    const fidelity = node("div", "lime-metrics");
    [["LOCAL FIDELITY R²", number(lime.local_fidelity_r2, 4)], ["MODEL SCORE", percent(lime.model_score)],
      ["SURROGATE SCORE", number(lime.surrogate_prediction, 4)]].forEach(([label, value]) => fidelity.append(append(node("div"), node("span", "", label), node("strong", "", value))));
    section.append(fidelity, node("p", "field-help", "R² describes how closely the local surrogate follows the classifier on its sampled neighborhood. Low or negative values indicate a poor fit; this is not detection accuracy."));
    const conditions = Array.isArray(lime.conditions) ? lime.conditions : [];
    const max = Math.max(...conditions.map((item) => Math.abs(Number(item.weight))), .000001);
    conditions.forEach((item) => {
      const direction = Number(item.weight) >= 0 ? "positive" : "negative";
      const track = node("div", "shap-track");
      track.setAttribute("aria-hidden", "true");
      const bar = node("span", `shap-bar ${direction}`);
      bar.style.width = `${Math.min(50, Math.abs(Number(item.weight)) / max * 50)}%`;
      track.append(bar);
      section.append(append(node("div", "lime-row"), node("span", "lime-condition", item.condition), track,
        node("span", `shap-number ${direction}`, signed(item.weight))));
    });
    section.append(node("p", "xai-method-note", `${number(lime.perturbations, 0)} synthetic perturbations · ${number(lime.reference_rows, 0)} training-reference rows · seed ${lime.seed}. Signed weights describe the surrogate’s output, not SHAP margins.`),
      node("p", "explanation-limitation", lime.limitations));
    return section;
  }
  function statisticsTable(statistics, raw = false) {
    const section = node("div", "xai-method-panel");
    section.append(node("p", "explanation-subtitle", raw
      ? "Actual input values beside frozen training-reference quartiles. These are observations and reference summaries, not model attributions."
      : `${statistics.method}. ${statistics.scope}. Features are sorted by mean split gain.`));
    if (!raw) {
      const parameters = node("div", "model-parameter-grid");
      Object.entries(statistics.model_parameters || {}).forEach(([key, value]) => parameters.append(append(node("div"), node("span", "", key.replaceAll("_", " ")), node("strong", "", value))));
      section.append(parameters);
    }
    const table = node("table", "feature-table extra-feature-table");
    const headings = raw ? ["FEATURE", "INPUT", "REF. MEDIAN", "REF. P25", "REF. P75"] : ["FEATURE", "GAIN", "SPLITS", "COVER"];
    const head = node("tr");
    headings.forEach((label) => {const th = node("th", "", label); th.scope = "col"; head.append(th);});
    table.append(append(node("thead"), head));
    const body = node("tbody");
    const features = [...(statistics.features || [])];
    if (!raw) features.sort((a, b) => Number(b.gain) - Number(a.gain));
    features.forEach((feature) => {
      const cells = raw ? [feature.value, feature.reference_median, feature.reference_p25, feature.reference_p75]
        : [feature.gain, feature.weight, feature.cover];
      body.append(append(node("tr"), node("td", "", feature.feature), ...cells.map((value) => node("td", "", number(value, 5)))));
    });
    table.append(body);
    section.append(append(node("div", "feature-table-wrap extra-table-wrap"), table));
    if (!raw) section.append(node("p", "xai-method-note", Object.entries(statistics.definitions || {}).filter(([key]) => ["gain", "weight", "cover"].includes(key)).map(([key, definition]) => `${key}: ${definition}`).join(". ")));
    return section;
  }
  function renderAdditionalXai(data) {
    if (!data.lime || !data.statistics || !Array.isArray(data.statistics.features)) throw new Error("Additional XAI response is incomplete.");
    const tabs = node("div", "tabs extra-xai-tabs");
    tabs.setAttribute("role", "tablist"); tabs.setAttribute("aria-label", "Additional explanation methods");
    const panels = [renderLime(data.lime), statisticsTable(data.statistics), statisticsTable(data.statistics, true)];
    const buttons = ["LIME", "Tree statistics", "Input values"].map((label, index) => {
      const button = node("button", `tab${index === 0 ? " active" : ""}`, label);
      button.id = `extra-xai-tab-${index}`; button.type = "button";
      button.setAttribute("role", "tab"); button.setAttribute("aria-selected", String(index === 0));
      button.setAttribute("aria-controls", `extra-xai-view-${index}`); button.tabIndex = index === 0 ? 0 : -1;
      panels[index].id = `extra-xai-view-${index}`; panels[index].hidden = index !== 0;
      panels[index].setAttribute("role", "tabpanel"); panels[index].setAttribute("aria-labelledby", button.id);
      tabs.append(button); return button;
    });
    setupTabs(buttons, panels);
    $("extra-xai-content").replaceChildren(tabs, ...panels, referenceDisclosure(data.reference || {}));
    $("extra-xai-content").hidden = false;
  }
  function renderLocalExplanation(data) {
    if (data.llm_executed !== true || typeof data.text !== "string" || !data.text.trim()) throw new Error("The local model returned no verified generated explanation.");
    $("assistant-content").replaceChildren(node("span", "actual-ai-badge", "ACTUAL LOCAL LLM INFERENCE"),
      node("p", "assistant-response", data.text),
      node("p", "assistant-provenance", `${data.model} · ${data.runtime || "Local runtime"} · ${data.device || "CPU"} · ${number(data.elapsed_seconds, 1)} seconds`),
      node("p", "explanation-limitation", data.limitations || "Generated interpretation may be incorrect. Verify it against the measured evidence."));
    if (data.rag) {
      const retrieval = node("section", "rag-evidence");
      retrieval.append(node("span", "eyebrow", "LOCAL KNOWLEDGE SOURCES"),
        node("p", "field-help", data.rag.used_in_generation ? "These passages were retrieved and supplied to the local model. They describe methods and limits; they are not captured traffic." : "No relevant local passage was retrieved. The model used measured evidence only."));
      (data.rag.passages || []).forEach((passage) => {
        const details = node("details", "rag-source");
        details.append(node("summary", "", `${passage.title} · ${passage.id}`), node("p", "", passage.text),
          node("pre", "", JSON.stringify({source: passage.source, scopes: passage.scopes, score: passage.score, sha256: passage.chunk_sha256}, null, 2)));
        retrieval.append(details);
      });
      const attempts = data.correction?.generation_attempts || [];
      retrieval.append(node("p", "field-help", `Retrieval: ${(data.rag.retrieval_attempts || []).length || 1} attempt(s). Generation: ${attempts.length || 1} attempt(s). ${attempts.at(-1)?.passed ? "Grounding checks passed." : "Verify the generated wording."} Checks are heuristic, not proof of factual correctness.`));
      $("assistant-content").append(retrieval);
    }
    $("assistant-content").hidden = false;
  }
  async function requestInsight(kind) {
    const selection = evidenceSelection();
    if (!selection) return;
    const field = kind === "xai" ? "xai" : "ai";
    const prefix = kind === "xai" ? "extra-xai" : "assistant";
    if (state.insights[`${field}Busy`]) return;
    if (state.insights.context?.key !== selection.key) resetInsights();
    const generation = state.insights.generation;
    state.insights.context = selection;
    state.insights[`${field}Busy`] = true;
    state.insights[field] = null;
    $(`${prefix}-content`).replaceChildren(); $(`${prefix}-content`).hidden = true;
    $(`${prefix}-error`).textContent = ""; $(`${prefix}-error`).hidden = true;
    $(`${prefix}-status`).textContent = kind === "xai" ? "Running actual LIME perturbations and reading native tree statistics…"
      : "Retrieving local background, generating an explanation and checking its grounding. Monitoring continues.";
    refreshInsightControls();
    try {
      const data = await api(kind === "xai" ? "/api/xai" : "/api/assistant/explain", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(selection.payload)});
      if (generation !== state.insights.generation) return;
      const returnedSource = kind === "xai" ? data.source : data.evidence?.source;
      if (selection.payload.kind === "live" && returnedSource?.observation_id !== selection.payload.observation_id) throw new Error("The returned explanation does not match the selected packet observation. No result was displayed.");
      if (kind === "xai") renderAdditionalXai(data); else renderLocalExplanation(data);
      state.insights[field] = data;
      $(`${prefix}-status`).textContent = `${kind === "xai" ? "Additional XAI calculated" : "Local explanation generated"} for the selected evidence. Its source is retained in the export.`;
      $("activity-status").textContent = $(`${prefix}-status`).textContent;
    } catch (error) {
      if (generation === state.insights.generation) {
        state.insights[field] = null;
        $(`${prefix}-content`).replaceChildren(); $(`${prefix}-content`).hidden = true;
        $(`${prefix}-error`).textContent = error.message; $(`${prefix}-error`).hidden = false;
        $(`${prefix}-status`).textContent = "Request failed. No substitute explanation was generated.";
      }
    } finally {
      if (generation === state.insights.generation) {state.insights[`${field}Busy`] = false; refreshInsightControls();}
    }
  }
  function exportedInsights() {
    const selected = state.insights.context;
    if (!selected || (!state.insights.xai && !state.insights.ai)) return null;
    return {source: selected.source, evidence_key: selected.key, additional_xai: state.insights.xai,
      local_language_model: state.insights.ai, note: "These explanations belong to the selected evidence source; it may differ from a newer live observation."};
  }
  function downloadJson(value, filename) {
    const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], {type: "application/json"}));
    const link = node("a"); link.href = url; link.download = filename;
    document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function exportLive() {
    if (state.exportingLive || !state.live?.running) return;
    state.exportingLive = true; refreshLiveControls();
    const generation = state.liveGeneration;
    const extras = exportedInsights();
    const displayedObservation = state.live.multiclass_prediction || null;
    try {
      const report = await api("/api/live/report");
      if (generation !== state.liveGeneration || !state.live?.running || state.live.error) return;
      downloadJson({...report, displayed_multiclass_observation: displayedObservation, selected_evidence_explanations: extras}, "netsec-live-report.json");
      $("activity-status").textContent = "Live report exported with observation provenance and available explanations.";
    } catch (error) {if (generation === state.liveGeneration) showError(error.message);}
    finally {state.exportingLive = false; refreshLiveControls();}
  }

  async function exportLiveCsv() {
    const observation = state.live?.multiclass_prediction;
    if (state.exportingLiveCsv || !state.live?.running || !observation?.observation_id || observation.available === false) return;
    const generation = state.liveGeneration;
    state.exportingLiveCsv = true; refreshLiveControls();
    try {
      const response = await fetch(`/api/live/features.csv?observation_id=${encodeURIComponent(observation.observation_id)}`, {credentials: "same-origin"});
      if (!response.ok) {
        let message = "The selected observation is no longer available. Download the latest live observation.";
        try {message = (await response.json()).error || message;} catch { /* Keep the useful default error. */ }
        throw new Error(message);
      }
      const csv = await response.blob();
      if (generation !== state.liveGeneration || !state.live?.running || state.live.error) return;
      const url = URL.createObjectURL(csv);
      const link = node("a"); link.href = url; link.download = `netsec-live-${observation.observation_id}.csv`;
      document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
      $("activity-status").textContent = "Downloaded the selected live observation with all 170 CSV columns. Unavailable values are empty.";
    } catch (error) {if (generation === state.liveGeneration) showError(error.message);}
    finally {state.exportingLiveCsv = false; refreshLiveControls();}
  }

  function renderStatus(data) {
    if (!data.model || !data.execution) throw new Error("Saved-model metadata is incomplete.");
    state.status = data;
    const model = data.model;
    $("model-name").textContent = model.name;
    $("model-dataset").textContent = `${data.dataset} · ${model.version || `seed ${model.seed}`}`;
    $("validation-accuracy").textContent = percent(model.validation_accuracy);
    $("macro-f1").textContent = number(model.macro_f1, 4);
    $("training-rows").textContent = number(model.training_rows, 0);
    const featureCount = Array.isArray(model.features) ? model.features.length : model.features;
    const classCount = Array.isArray(model.classes) ? model.classes.length : model.classes;
    $("class-count").textContent = `${number(classCount, 0)} classes · ${number(featureCount, 0)} features`;
    document.documentElement.dataset.modelState = "ready";
    $("upload-limit").textContent = `Up to ${number(data.max_csv_rows, 0)} rows · ${number(data.max_upload_bytes / 1048576, 1)} MB maximum`;
    $("model-limitations").replaceChildren(...(data.limitations || []).map((limitation) => node("li", "", limitation)));
    const fields = [
      ["Run ID", model.run_id], ["Version", model.version], ["Selection", model.selection_basis], ["Dataset", data.dataset],
      ["Execution", data.execution.device], ["XAI", data.execution.xai], ["Reasoning", data.execution.reasoning],
      ["LLM executed", data.execution.llm_executed ? "Yes" : "No"],
      ["Unknown detection", data.execution.novelty_available ? "Available" : "Not evaluated"],
    ];
    $("provenance-details").replaceChildren(...fields.flatMap(([label, value]) => [node("dt", "", label), node("dd", "", value || "Not specified")]));
    $("csv-schema-help").textContent = data.partial_input_supported === false
      ? "The active original checkpoint requires its full DataSense CSV schema. A partial-input upgrade has not been promoted; partial uploads and 84-class live screening are unavailable with this checkpoint."
      : "Full or partial CSVs use the same 170-column schema as live observations. Each row needs at least one usable measurement in a known column. Omitted columns and blank optional cells stay missing; no new feature columns are invented.";
    $("live-shared-model-note").textContent = data.partial_input_supported === false
      ? "Original checkpoint · partial-input upgrade not active" : "Shared 84-class model · tentative live results";
    $("live-shared-model-description").textContent = data.partial_input_supported === false
      ? "The active original checkpoint accepts full CSV inputs. Live rules and the separate binary packet comparison remain available; they do not replace 84-class inference."
      : "Measured common columns use the CSV schema. Unavailable values stay missing. Live rules and the binary packet model remain separate comparisons.";
    if (data.partial_input_supported === false) $("input-context-note").textContent = "The original CSV checkpoint remains active. Partial-input 84-class inference is unavailable until an evaluated upgrade is promoted. Live rules and the binary model are separate.";
    const compatibility = data.feature_compatibility || data.feature_schema?.feature_compatibility || [];
    if (Array.isArray(compatibility) && compatibility.length) {
      $("upload-compatibility").replaceChildren(renderCompatibility({features: compatibility}, true));
    }
  }
  function renderSamples(data) {
    if (!Array.isArray(data.samples)) throw new Error("The validation sample list is incomplete.");
    state.samples = data.samples;
    const groups = new Map();
    state.samples.forEach((sample) => {
      const label = sample.group || "Traffic examples";
      if (!groups.has(label)) {const group = node("optgroup"); group.label = label; groups.set(label, group);}
      const option = node("option", "", sample.display_name || sample.label);
      option.value = String(sample.id);
      groups.get(label).append(option);
    });
    $("sample-select").replaceChildren(...groups.values());
    if (!state.samples.length) {
      $("sample-select").append(node("option", "", "No validation samples available"));
      $("sample-help").textContent = "Upload a CSV to analyze traffic features.";
    }
  }

  function renderCompatibility(coverage, schemaOnly = false) {
    const features = Array.isArray(coverage.features) ? coverage.features : [];
    const total = coverage.total ?? features.length;
    const observed = coverage.observed ?? features.filter((item) => item.value !== null && item.value !== undefined).length;
    const missing = coverage.missing ?? total - observed;
    const details = node("details", "feature-compatibility");
    const summary = node("summary");
    summary.append(node("span", "", schemaOnly ? `Shared CSV / live feature table · ${total} columns` : `Feature coverage · ${observed} / ${total} observed`),
      node("span", "coverage-summary-note", schemaOnly ? `${features.filter((item) => item.live_mapping).length} mapped from packet counts` : `${missing} missing · inspect values`));
    details.append(summary);
    if (!schemaOnly) {
      const ratio = node("div", "coverage-meter");
      ratio.setAttribute("role", "meter"); ratio.setAttribute("aria-label", "Observed feature coverage");
      ratio.setAttribute("aria-valuemin", "0"); ratio.setAttribute("aria-valuemax", String(total)); ratio.setAttribute("aria-valuenow", String(observed));
      const fill = node("span"); fill.style.width = `${total ? Math.min(100, observed / total * 100) : 0}%`; ratio.append(fill); details.append(ratio);
      details.append(node("p", "coverage-profile", `Input profile: ${String(coverage.profile || "unspecified").replaceAll("_", " ")} · ${coverage.validated_profile === true ? "Source-dataset validation available for this profile" : "This input profile has no reported validation"}.`));
      details.append(node("p", "field-help", `${coverage.validation_scope || "Profile validation describes source-dataset observations."} It does not measure live PC accuracy.`));
    }
    details.append(node("p", "field-help", schemaOnly
      ? "Use these existing column names. Uploads and live downloads share their order. A live mapping means a directly counted packet quantity is available; it does not establish equivalence to the publisher’s extractor."
      : "Missing values remain unavailable to the model. They are not observed zeros or evidence of absent traffic. The live mapping column identifies existing packet-count features only."));
    const table = node("table", "feature-table compatibility-table");
    table.append(node("caption", "sr-only", schemaOnly ? "Canonical 170-column CSV and live input compatibility" : "Current values and missing states in the shared CSV and live feature schema"));
    const heading = node("tr");
    ["EXISTING FEATURE", "VALUE", "STATE", "LIVE MAPPING"].forEach((label) => {const cell = node("th", "", label); cell.scope = "col"; heading.append(cell);});
    table.append(append(node("thead"), heading));
    const body = node("tbody");
    features.forEach((feature) => {
      const isMissing = feature.value === null || feature.value === undefined;
      const row = node("tr", schemaOnly ? "" : isMissing ? "feature-missing" : "feature-measured");
      const label = node("th", "", feature.feature); label.scope = "row";
      row.append(label, node("td", "", schemaOnly ? "—" : number(feature.value, 5)),
        append(node("td"), node("span", `feature-state ${schemaOnly ? "schema" : isMissing ? "missing" : "measured"}`, schemaOnly ? "Awaiting input" : isMissing ? "Missing" : "Measured")),
        node("td", "", feature.live_mapping ? "Packet count" : "Not mapped"));
      body.append(row);
    });
    table.append(body);
    const wrap = append(node("div", "feature-table-wrap compatibility-scroll"), table);
    wrap.tabIndex = 0; wrap.setAttribute("role", "region"); wrap.setAttribute("aria-label", "Scrollable feature compatibility table");
    details.append(wrap);
    return details;
  }
  function renderClassScores(prediction) {
    const classes = node("div", "top-classes");
    classes.append(node("p", "", `${prediction.tentative ? "TOP 3 TENTATIVE CATEGORIES" : "TOP 3 CLOSED-SET CATEGORIES"} · UNCALIBRATED SCORES`));
    (prediction.top_classes || []).slice(0, 3).forEach((entry, index) => {
      classes.append(append(node("div", "top-class"), append(node("span"), node("span", "class-rank", String(index + 1).padStart(2, "0")),
        document.createTextNode(entry.display_label || entry.label)), node("span", "", percent(entry.score))));
    });
    if (!prediction.top_classes?.length) classes.append(node("p", "field-help", "Ranked category scores were not returned."));
    return classes;
  }
  function coverageLabel(coverage) {
    return coverage ? `${coverage.observed} / ${coverage.total} features observed · ${coverage.missing} missing` : "Feature coverage unavailable";
  }
  function renderShap(explanation) {
    const section = node("div", "report-section");
    append(section, append(node("div", "section-caption"), node("h3", "", "What moved this prediction?"), node("span", "", "SIGNED CONTRIBUTIONS")));
    append(section, node("p", "explanation-subtitle", `${explanation.method}. Contributions below explain the raw class margin for “${explanation.display_label || explanation.class_label}”.`));
    const legend = node("div", "shap-legend");
    append(legend, append(node("span"), node("i", "legend-dot negative"), document.createTextNode("Decreases class margin")), append(node("span"), node("i", "legend-dot positive"), document.createTextNode("Increases class margin")));
    section.append(legend);
    const features = explanation.important_features || [];
    if ((explanation.all_features || features).some((feature) => feature.value === null || feature.value === undefined)) {
      section.append(node("p", "missing-routing-note", "Missing-feature SHAP contributions explain the model’s missing-value routing. They are not measured traffic or evidence that a sensor value was zero."));
    }
    const max = Math.max(...features.map((feature) => Math.abs(Number(feature.shap_value))), 0.000001);
    features.forEach((feature) => {
      const direction = Number(feature.shap_value) >= 0 ? "positive" : "negative";
      const row = node("div", "shap-row");
      const missing = feature.value === null || feature.value === undefined;
      const name = append(node("div", "shap-feature"), node("span", "shap-name", feature.feature), node("span", `shap-value${missing ? " missing-input" : ""}`, missing ? "Missing · model routing" : `Measured value: ${number(feature.value, 4)}`));
      const track = node("div", "shap-track");
      track.setAttribute("aria-hidden", "true");
      const bar = node("div", `shap-bar ${direction}`);
      bar.style.width = `${Math.min(50, Math.abs(Number(feature.shap_value)) / max * 50)}%`;
      track.append(bar);
      append(row, name, track, node("span", `shap-number ${direction}`, signed(feature.shap_value)));
      row.setAttribute("aria-label", `${feature.feature}, ${missing ? "missing input, contribution from model missing-value routing" : `value ${number(feature.value, 4)}`}, SHAP contribution ${signed(feature.shap_value)} to raw class margin`);
      section.append(row);
    });
    if (!features.length) section.append(node("p", "explanation-subtitle", "No feature contributions were returned."));
    const proof = node("div", "margin-proof");
    proof.append(node("p", "margin-proof-title", "ADDITIVITY CHECK · ALL FEATURES INCLUDED"));
    const term = (value, label) => append(node("span", "margin-term"), document.createTextNode(number(value, 5)), node("small", "", label));
    append(proof, append(node("div", "margin-equation"), term(explanation.bias, "Base margin"), node("span", "equation-symbol", "+"), term(explanation.contribution_sum, "Σ SHAP values"), node("span", "equation-symbol", "="), term(explanation.margin, "Raw class margin")));
    proof.append(node("p", "margin-proof-error", `Reconstruction error: ${number(explanation.reconstruction_error, 8)} · Contributions are in margin units, not percentage points.`));
    section.append(proof, node("p", "explanation-limitation", explanation.limitations || "TreeSHAP attributes model behavior; it does not establish causation or confirm malicious intent."));
    return section;
  }
  function renderAssessment(reasoning) {
    const section = node("div", "report-section");
    section.append(node("span", "assessment-label", `${reasoning.mode} · ${reasoning.llm_executed ? "LLM executed" : "No LLM inference"}`));
    section.append(node("p", "assessment-text", reasoning.assessment));
    section.append(append(node("div", "section-caption"), node("h3", "", "Suggested verification and response")));
    const actions = node("ol", "action-list");
    actions.append(...(reasoning.actions || []).map((action) => node("li", "", action)));
    section.append(actions, node("p", "uncertainty", reasoning.uncertainty));
    return section;
  }
  function renderFeatures(explanation) {
    const section = node("div", "report-section");
    const features = explanation.all_features || [];
    section.append(node("p", "explanation-subtitle", `All ${features.length} feature attributions for the predicted class. Contributions from missing inputs reflect missing-value routing, not measured traffic.`));
    const table = node("table", "feature-table");
    const headRow = node("tr");
    ["FEATURE", "INPUT VALUE", "SHAP VALUE"].forEach((label) => {const cell = node("th", "", label); cell.scope = "col"; headRow.append(cell);});
    table.append(append(node("thead"), headRow));
    const body = node("tbody");
    features.forEach((feature) => {
      body.append(append(node("tr"), node("td", "", feature.feature), node("td", "", number(feature.value, 5)), node("td", Number(feature.shap_value) >= 0 ? "positive" : "negative", signed(feature.shap_value))));
    });
    table.append(body);
    section.append(append(node("div", "feature-table-wrap"), table));
    return section;
  }
  function renderReport(report) {
    if (!report.prediction || !report.explanation || !report.reasoning || !report.source) throw new Error("The analysis report is incomplete. No result was displayed.");
    const prediction = report.prediction;
    const source = report.source;
    const root = document.createDocumentFragment();
    const summary = node("div", `prediction-summary${prediction.is_attack ? "" : " benign"}`);
    const kind = source.kind === "validation" || source.validation_label !== undefined ? "Validation sample" : "Uploaded features";
    append(summary, append(node("div", "prediction-topline"), node("span", "decision-badge", prediction.tentative ? "Tentative category · partial input" : prediction.is_attack ? "Attack-class prediction" : "Benign-class prediction"), node("span", "source-chip", kind)));
    const displayLabel = prediction.display_label || prediction.label;
    const title = append(node("div"), node("h3", "prediction-label", displayLabel));
    if (source.validation_label !== undefined) title.append(node("p", "ground-truth", `Dataset label: ${source.validation_display_label || source.display_label || source.validation_label} · ${source.validation_label === prediction.label ? "Prediction matches" : "Prediction differs"}`));
    const score = append(node("div", "prediction-score"), node("strong", "", percent(prediction.score)), node("span", "", "Uncalibrated model score"));
    append(summary, append(node("div", "prediction-main"), title, score));
    summary.append(node("p", "prediction-note", `Unknown-attack status: ${prediction.novelty || "not evaluated"}. This prediction requires verification against the underlying traffic.`));
    summary.append(node("p", "coverage-inline", `${coverageLabel(report.feature_coverage)} · Model version: ${report.model?.version || "not provided"}`));
    root.append(summary);
    root.append(renderClassScores(prediction));
    if (report.feature_coverage) root.append(renderCompatibility(report.feature_coverage));

    const tabs = node("div", "tabs report-tabs");
    tabs.setAttribute("role", "tablist");
    tabs.setAttribute("aria-label", "Analysis evidence sections");
    const panels = [renderShap({...report.explanation, display_label: displayLabel}), renderAssessment(report.reasoning), renderFeatures(report.explanation)];
    const buttons = ["TreeSHAP evidence", "Assessment", "All features"].map((label, index) => {
      const button = node("button", `tab${index === 0 ? " active" : ""}`, label);
      button.type = "button"; button.id = `report-tab-${index}`;
      button.setAttribute("role", "tab"); button.setAttribute("aria-controls", `report-view-${index}`);
      button.setAttribute("aria-selected", String(index === 0)); button.tabIndex = index === 0 ? 0 : -1;
      const panel = panels[index]; panel.id = `report-view-${index}`;
      panel.setAttribute("role", "tabpanel"); panel.setAttribute("aria-labelledby", button.id); panel.hidden = index !== 0;
      tabs.append(button); return button;
    });
    setupTabs(buttons, panels);
    root.append(tabs, ...panels);
    const sourceParts = [kind];
    if (source.source_row !== undefined) sourceParts.push(`Source row ${source.source_row}`);
    if (source.row_index !== undefined) sourceParts.push(`CSV row ${source.row_index}`);
    if (report.model?.run_id) sourceParts.push(`Model run: ${report.model.run_id}`);
    root.append(node("div", "report-source", sourceParts.join(" · ")));
    $("report-content").replaceChildren(root);
    $("report-empty").hidden = true;
    $("report-content").hidden = false;
    state.report = report;
    $("activity-status").textContent = `Analysis complete. Predicted class ${displayLabel}.`;
  }

  async function analyze(path, payload) {
    const request = ++state.request;
    showError(""); setBusy(true, "Running inference and computing SHAP evidence."); clearReport(true);
    try {
      const report = await api(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
      if (request === state.request) {
        renderReport(report);
        state.analysisSource = path === "/api/analyze" ? {kind: "sample", sample_id: payload.sample_id}
          : {kind: "features", features: [...payload.features]};
      }
    } catch (error) {
      if (request === state.request) {clearReport(); showError(error.message); $("activity-status").textContent = "Analysis failed. No prediction is displayed.";}
    } finally {if (request === state.request) setBusy(false);}
  }
  function renderBatch(rows) {
    state.rows = rows; state.selectedRow = null;
    const buttons = rows.map((row, index) => {
      const label = row.prediction.display_label || row.prediction.label;
      const button = append(node("button", "batch-row"), node("span", "batch-row-number", `Row ${row.row_index}`), node("span", "batch-row-label", label), node("span", "batch-row-score", percent(row.prediction.score)));
      button.type = "button"; button.setAttribute("aria-pressed", "false");
      button.setAttribute("aria-label", `Explain CSV row ${row.row_index}, predicted ${label}, uncalibrated score ${percent(row.prediction.score)}`);
      button.addEventListener("click", () => {
        state.selectedRow = index;
        for (const [i, item] of [...$("batch-results").children].entries()) item.setAttribute("aria-pressed", String(i === index));
        analyze("/api/explain", {features: row.features, row_index: row.row_index});
      });
      return button;
    });
    $("batch-results").replaceChildren(...buttons);
    $("batch-count").textContent = `${rows.length} rows`;
    $("batch-section").hidden = !rows.length;
  }
  async function upload() {
    const file = $("csv-file").files[0];
    if (!file || !state.status) return;
    showError(""); clearReport(); renderBatch([]);
    if (file.size > state.status.max_upload_bytes) {showError(`The CSV exceeds the ${number(state.status.max_upload_bytes / 1048576, 1)} MB upload limit.`); return;}
    if (!file.name.toLowerCase().endsWith(".csv")) {showError("Choose a file with a .csv extension."); return;}
    const request = ++state.request;
    setBusy(true, "Classifying uploaded CSV rows.");
    const form = new FormData(); form.append("file", file);
    try {
      const data = await api("/api/upload", {method: "POST", body: form});
      if (request !== state.request) return;
      if (!Array.isArray(data.rows)) throw new Error("The CSV response did not contain valid predictions.");
      renderBatch(data.rows);
      $("activity-status").textContent = `Classified ${data.total_rows} CSV rows. Select a row to inspect TreeSHAP evidence.`;
      $("report-empty").querySelector("h3").textContent = "Choose a row to explain.";
      $("report-empty").querySelector("p").textContent = "CSV predictions are ready. Select a row on the left to calculate its exact TreeSHAP contributions.";
    } catch (error) {
      if (request === state.request) {clearReport(); renderBatch([]); showError(error.message);}
    } finally {if (request === state.request) setBusy(false);}
  }

  function refreshLiveControls() {
    const capabilities = state.capabilities;
    const running = Boolean(state.live?.running);
    const mode = $("live-mode").value;
    const available = capabilities && (mode === "packets" ? capabilities.packet_capture_available : capabilities.connections_available);
    $("live-interface").disabled = !capabilities || state.liveBusy || running;
    $("live-mode").disabled = !capabilities || state.liveBusy || running;
    $("live-start").disabled = !available || !$("live-interface").value || running || state.liveBusy;
    $("live-stop").disabled = !running || state.liveBusy;
    $("download-live-report").disabled = !running || Boolean(state.live?.error) || !state.live?.metrics?.packets_observed || state.liveBusy || state.exportingLive;
    $("download-live-report").setAttribute("aria-busy", String(state.exportingLive));
    $("download-live-report").querySelector("span").textContent = state.exportingLive ? "Exporting…" : "Export report";
    const multiclass = state.live?.multiclass_prediction;
    $("download-live-csv").disabled = !running || Boolean(state.live?.error) || !multiclass?.observation_id
      || multiclass.available === false || !Number.isFinite(multiclass.prediction?.score) || state.liveBusy || state.exportingLiveCsv;
    $("download-live-csv").setAttribute("aria-busy", String(state.exportingLiveCsv));
    $("download-live-csv").querySelector("span").textContent = state.exportingLiveCsv ? "Downloading…" : "Download CSV";
    $("live-report-panel").setAttribute("aria-busy", String(state.liveBusy));
    $("live-mode-help").textContent = mode === "packets"
      ? "Inspects observed packet headers on the selected connection. Packet capture requires Npcap."
      : "Uses real operating-system traffic counters and connection information. No packet capture driver is required.";
    $("packet-setup").hidden = !capabilities || mode !== "packets" || capabilities.packet_capture_available;
    $("packet-model-note").hidden = mode !== "packets" || !capabilities;
    if (capabilities) $("packet-model-note").textContent = state.status?.partial_input_supported === false
      ? "The active original checkpoint cannot classify partial live input. The separate binary packet comparison requires a complete 10-second packet window and has no measured live-PC accuracy."
      : "A complete 10-second packet window can supply seven existing count features to the 84-class model. Other inputs remain missing. The separate binary comparison also uses those seven counts. Neither score is live-PC accuracy.";
    if (capabilities && !capabilities.connections_available && mode === "connections") $("live-mode-help").textContent = "Operating-system connection monitoring is unavailable in this environment.";
  }
  function renderCapabilities(data) {
    if (!Array.isArray(data.interfaces)) throw new Error("The local network connection list is incomplete.");
    state.capabilities = data;
    $("live-interface").replaceChildren(...data.interfaces.map((item) => {
      const option = node("option", "", `${item.name}${item.is_up ? "" : " · disconnected"}`);
      option.value = item.id;
      option.disabled = !item.is_up;
      return option;
    }));
    const recommended = data.interfaces.find((item) => item.id === data.recommended_interface && item.is_up)
      || data.interfaces.find((item) => item.is_up);
    if (recommended) $("live-interface").value = recommended.id;
    else {const option = node("option", "", "No active network connection found"); option.value = ""; $("live-interface").replaceChildren(option);}
    $("packet-setup-message").textContent = data.packet_capture_message || "Install Npcap to enable packet inspection, then restart the local app.";
    refreshLiveControls();
  }
  function clearLiveObservations() {
    if (state.inputMode === "live" || state.insights.context?.payload.kind === "live") resetInsights();
    state.liveHistoryKey = null;
    state.liveModelKey = null;
    ["live-packet-rate", "live-byte-rate", "live-connection-count"].forEach((id) => {$(id).textContent = "—";});
    $("live-history").hidden = true;
    $("live-evidence").replaceChildren(node("p", "live-empty-note", "No current evidence is available."));
    $("live-connections").replaceChildren(node("p", "live-empty-note", "No current connection observations are available."));
    $("live-packets-section").hidden = true;
    $("live-packet-model").hidden = true;
    $("live-packet-model").replaceChildren();
    $("live-multiclass-model").replaceChildren(append(node("div", "packet-model-heading"), node("h3", "", "84-class traffic categories"), node("span", "live-rule-badge", "Shared CSV schema")),
      node("p", "packet-model-waiting", "No current shared-model observation is available."));
    delete $("live-multiclass-model").dataset.observationId;
    $("live-observed-count").textContent = "NO CURRENT OBSERVATIONS";
    $("live-connection-caption").textContent = "NO CURRENT OBSERVATIONS";
  }
  function renderLiveFailure(message) {
    state.live = {...state.live, state: "error", error: message};
    clearLiveObservations();
    $("live-status-banner").className = "live-status-banner error";
    $("live-state-label").textContent = "Monitoring status unavailable";
    $("live-state-detail").textContent = message;
    $("live-assessment").className = "live-assessment not-assessed";
    $("live-assessment-label").textContent = "Traffic is not currently assessed";
    $("live-assessment-detail").textContent = "The latest observation could not be verified. Previous healthy-looking results are not displayed.";
    $("live-session-info").textContent = "Connection to local monitoring service failed. Retrying status checks.";
    refreshLiveControls();
    refreshInsightControls();
  }
  function renderLiveHistory(history) {
    const samples = history.slice(-24);
    if (!samples.length) {$("live-history").hidden = true; return;}
    const key = JSON.stringify(samples);
    if (state.liveHistoryKey === key) return;
    state.liveHistoryKey = key;
    const max = Math.max(...samples.map((item) => Number(item.packets_per_second) || 0), 1);
    const chart = node("div", "live-history-bars");
    chart.setAttribute("aria-label", "Recent packet rates; each bar is a real observation");
    samples.forEach((item) => {
      const rate = Math.max(0, Number(item.packets_per_second) || 0);
      const bar = node("span", `live-history-bar${item.status === "suspicious" ? " suspicious" : ""}${rate === 0 ? " zero" : ""}`);
      bar.style.height = `${Math.max(rate > 0 ? 2 : 0, rate / max * 100)}%`;
      bar.title = `${item.time}: ${number(rate, 1)} packets/s, ${number(item.bytes_per_second, 0)} bytes/s`;
      chart.append(bar);
    });
    $("live-history").replaceChildren(append(node("div", "live-history-caption"), node("span", "", "RECENT PACKET RATE"), node("span", "", `Peak ${number(max === 1 && samples.every((item) => !item.packets_per_second) ? 0 : max, 1)} packets/s`)), chart);
    $("live-history").hidden = false;
  }
  function renderLiveConnections(connections) {
    if (!connections.length) {$("live-connections").replaceChildren(node("p", "live-empty-note", "No active remote connections were observed in the latest snapshot.")); return;}
    const list = node("div", "live-connection-list");
    connections.slice(0, 30).forEach((connection) => {
      const row = node("div", "live-connection-row");
      const identity = append(node("div", "live-connection-identity"), node("strong", "", connection.process || "Process unavailable"), node("span", "", `${connection.protocol || "Network"} · ${connection.status || "Status unavailable"}`));
      const address = append(node("div", "live-connection-address"), node("span", "", connection.remote || "No remote destination"), node("small", "", `Local: ${connection.local || "unavailable"}`));
      list.append(append(row, identity, address));
    });
    $("live-connections").replaceChildren(list);
    $("live-connection-caption").textContent = `${connections.length} OBSERVED${connections.length > 30 ? " · FIRST 30 SHOWN" : ""}`;
  }
  function renderLivePackets(packets) {
    $("live-packets-section").hidden = !packets.length;
    if (!packets.length) return;
    const list = node("div", "live-packet-list");
    packets.slice(-12).reverse().forEach((packet) => {
      list.append(append(node("div", "live-packet-row"), node("span", "", packet.protocol || "Packet"), append(node("div"), node("span", "", `${packet.source} → ${packet.destination}`), node("small", "", `${packet.time} · ${number(packet.length, 0)} bytes`))));
    });
    $("live-packets").replaceChildren(list);
  }
  function renderPacketModel(prediction, mode) {
    const panel = $("live-packet-model");
    panel.hidden = mode !== "packets";
    if (mode !== "packets") {panel.replaceChildren(); return;}
    const heading = append(node("div", "packet-model-heading"), node("h3", "", "Binary packet comparison"), node("span", "live-rule-badge", "Separate 2-class model"));
    if (!prediction || prediction.available === false || !Number.isFinite(prediction.score)) {
      panel.replaceChildren(heading, node("p", "packet-model-waiting", prediction?.label || "Waiting for a complete 10-second packet window."), node("p", "explanation-limitation", prediction?.limitations || "No model prediction is available for the current live observation."));
      return;
    }
    const summary = node("div", `packet-model-summary${prediction.is_attack ? " suspicious" : ""}`);
    summary.append(node("strong", "", prediction.label), node("span", "", `${percent(prediction.score)} · uncalibrated model score`));
    const note = node("p", "explanation-limitation", `${prediction.method || "Experimental packet XGBoost"} · ${prediction.window_seconds || 10}-second packet window. This comparison does not supply the 84-class category above. ${prediction.limitations || "This is an experimental classification, not a confirmed incident."}`);
    const parts = [heading, summary, note];
    if (prediction.dataset_validation_accuracy !== undefined) parts.push(node("p", "packet-model-validation", `Source-dataset validation accuracy: ${percent(prediction.dataset_validation_accuracy)}. This is not live-PC accuracy.`));
    if (prediction.explanation) {
      const evidence = node("details", "binary-comparison-evidence");
      evidence.append(node("summary", "", "Inspect the binary comparison’s own TreeSHAP evidence"), renderShap({...prediction.explanation, class_label: prediction.explanation.class_label || "suspicious traffic", limitations: prediction.explanation.limitations || "Exact TreeSHAP explains the experimental packet model’s margin. It does not establish an intrusion or causal traffic behavior."}));
      parts.push(evidence);
    }
    panel.replaceChildren(...parts);
  }
  function renderLiveMulticlass(report, mode) {
    const panel = $("live-multiclass-model");
    const available = report && report.available !== false && Number.isFinite(report.prediction?.score);
    const key = available ? report.observation_id || JSON.stringify(report.prediction) : JSON.stringify([mode, report?.label, report?.reason, report?.message]);
    if (state.liveModelKey === key) return;
    state.liveModelKey = key;
    const coverageOpen = panel.querySelector(".feature-compatibility")?.open || false;
    const shapOpen = panel.querySelector(".multiclass-shap")?.open ?? true;
    const heading = append(node("div", "packet-model-heading"), node("h3", "", "84-class traffic categories"), node("span", "live-rule-badge", "Shared CSV / live model"));
    if (!available) {
      delete panel.dataset.observationId;
      const waiting = report?.reason || report?.message || report?.label || (mode === "packets"
        ? state.status?.partial_input_supported === false
          ? "The active original checkpoint requires full CSV inputs. An upgrade must be promoted and the app restarted before 84-class live screening is available."
          : "Waiting for a complete packet window with usable measurements."
        : "Use Packet inspection to measure the common packet-count columns. Connections mode supplies separate OS observations and rules.");
      panel.replaceChildren(heading, node("p", "packet-model-waiting", waiting),
        node("p", "explanation-limitation", report?.limitations || "No 84-class inference is available for this observation. Missing values are never replaced with invented traffic measurements."));
      return;
    }
    const prediction = report.prediction;
    panel.dataset.observationId = report.observation_id || "";
    const summary = node("div", "multiclass-summary");
    summary.append(node("span", "tentative-label", "TENTATIVE · PARTIAL LIVE INPUT"),
      node("h4", "", prediction.display_label || prediction.label),
      node("p", "", `${percent(prediction.score)} uncalibrated model score · not a confirmed incident`));
    const metadata = node("div", "multiclass-metadata");
    metadata.append(node("strong", "", coverageLabel(report.feature_coverage)),
      node("span", "", `Model version: ${report.model?.version || "not provided"}`),
      node("span", "", `${report.observed_at || "Observation time unavailable"} · ${number(report.observed_window_seconds, 2)}-second window`));
    const parts = [heading, summary, renderClassScores({...prediction, tentative: true}), metadata,
      node("p", "missing-routing-note", "These are tentative matches to the CSV categories using observed common columns. Sparse coverage and source-dataset differences can change the result; live PC accuracy has not been measured.")];
    if (report.feature_coverage) {
      const coverage = renderCompatibility(report.feature_coverage); coverage.open = coverageOpen; parts.push(coverage);
    }
    if (report.explanation) {
      const evidence = node("details", "multiclass-shap"); evidence.open = shapOpen;
      evidence.append(node("summary", "", "TreeSHAP for this 84-class prediction"), renderShap({...report.explanation, display_label: prediction.display_label || prediction.label}));
      parts.push(evidence);
    }
    panel.replaceChildren(...parts);
  }
  function updateLiveMetric(id, value) {
    const element = $(id);
    if (element.textContent === value) return;
    element.textContent = value;
    if (state.inputMode === "live" && !document.hidden && element.animate
        && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      element.animate([{opacity: .35, transform: "translateY(2px)"}, {opacity: 1, transform: "translateY(0)"}],
        {duration: 230, easing: "cubic-bezier(.2,.7,.2,1)"});
    }
  }
  function renderLiveSnapshot(snapshot) {
    if (!snapshot.metrics || !snapshot.assessment || typeof snapshot.running !== "boolean") throw new Error("The monitoring service returned an incomplete observation.");
    state.live = snapshot;
    if (snapshot.running) state.liveStarted = true;
    if (snapshot.state === "error" || snapshot.error) {renderLiveFailure(snapshot.error || "Monitoring stopped because of a capture error."); refreshLiveControls(); return;}
    const running = snapshot.running;
    if (running) {
      if (state.capabilities?.interfaces.some((item) => item.id === snapshot.interface)) $("live-interface").value = snapshot.interface;
      if (["connections", "packets"].includes(snapshot.mode)) $("live-mode").value = snapshot.mode;
    }
    $("live-status-banner").className = `live-status-banner${running ? " running" : ""}`;
    $("live-state-label").textContent = running ? (snapshot.state === "starting" ? "Starting monitoring" : "Monitoring your PC") : (state.liveStarted ? "Monitoring stopped" : "Not started");
    const selected = state.capabilities?.interfaces.find((item) => item.id === snapshot.interface);
    const interfaceName = selected?.name || snapshot.interface || "selected connection";
    $("live-state-detail").textContent = running
      ? `${interfaceName} · ${snapshot.mode === "packets" ? "Packet inspection · 10-second observation windows" : "Connections & traffic · 2-second observation windows"}`
      : "Choose your connection and press Start monitoring. No capture starts automatically.";
    $("live-session-info").textContent = running ? `Observed for ${number(snapshot.elapsed_seconds, 0)} seconds · ${number(snapshot.metrics.packets_observed, 0)} packets counted` : "Monitoring is off. Traffic is not assessed while stopped.";
    $("live-limitations").textContent = `${snapshot.assessment.limitations || "Live rules are experimental and are not a validated intrusion classifier."} Rule evidence is not TreeSHAP. CSV validation accuracy does not apply to live monitoring. No automatic defensive actions are taken.`;
    if (!running) {
      clearLiveObservations();
      $("live-assessment").className = "live-assessment not-assessed";
      $("live-assessment-label").textContent = "Traffic has not been assessed";
      $("live-assessment-detail").textContent = "Start monitoring to collect current observations. A lack of alerts is not proof of safety.";
      refreshLiveControls(); return;
    }
    updateLiveMetric("live-packet-rate", number(snapshot.metrics.packets_per_second, 1));
    updateLiveMetric("live-byte-rate", number(snapshot.metrics.bytes_per_second, 0));
    updateLiveMetric("live-connection-count", number(snapshot.metrics.active_connections, 0));
    const status = snapshot.assessment.status;
    const names = {not_assessed: "Live rules · waiting for observations", idle: "Live rules · no recent traffic activity", benign_looking: "Live rules · no thresholds exceeded", suspicious: "Live rules · suspicious behavior observed"};
    const details = {not_assessed: "Waiting for enough observations to assess behavior.", idle: "The selected connection is idle. Idle traffic is not a safety verdict.", benign_looking: "Current activity looks ordinary under these experimental rules. This does not prove the PC is safe.", suspicious: "One or more observed patterns crossed a rule threshold. Inspect the evidence and underlying traffic before responding."};
    $("live-assessment").className = `live-assessment ${status.replaceAll("_", "-")}`;
    $("live-assessment-label").textContent = names[status] || snapshot.assessment.label || "Not assessed";
    $("live-assessment-detail").textContent = details[status] || "No supported behavior assessment is available.";
    const evidence = snapshot.assessment.evidence || [];
    $("live-evidence").replaceChildren(...(evidence.length ? evidence.map((item) => append(node("div", "live-evidence-row"), node("p", "", item.message), node("small", "", `${item.rule} · observed ${number(item.observed, 3)} · threshold ${number(item.threshold, 3)}`))) : [node("p", "live-empty-note", status === "suspicious" ? "The service returned a rule alert without detailed evidence." : "No rule thresholds were exceeded in the latest observation. This does not establish safety.")]));
    $("live-observed-count").textContent = evidence.length ? `${evidence.length} RULE OBSERVATIONS` : "NO THRESHOLD ALERTS";
    renderLiveHistory(snapshot.history || []);
    renderLiveConnections(snapshot.connections || []);
    renderLivePackets(snapshot.recent_packets || []);
    renderLiveMulticlass(snapshot.multiclass_prediction, snapshot.mode);
    renderPacketModel(snapshot.model_prediction, snapshot.mode);
    refreshLiveControls();
    refreshInsightControls();
  }
  function scheduleLivePoll() {
    if (state.liveTimer) clearTimeout(state.liveTimer);
    state.liveTimer = null;
    if (state.inputMode !== "live" && !state.live?.running) return;
    state.liveTimer = setTimeout(() => {state.liveTimer = null; pollLive();}, 2000);
  }
  async function pollLive() {
    if (state.liveFlight || state.liveBusy) {scheduleLivePoll(); return;}
    if (state.inputMode !== "live" && !state.live?.running) return;
    state.liveFlight = true;
    const generation = state.liveGeneration;
    try {
      const snapshot = await api("/api/live/status");
      if (generation === state.liveGeneration) renderLiveSnapshot(snapshot);
    } catch (error) {if (generation === state.liveGeneration) renderLiveFailure(error.message);}
    finally {state.liveFlight = false; scheduleLivePoll();}
  }
  async function liveAction(action) {
    if (state.liveBusy) return;
    const generation = ++state.liveGeneration;
    state.liveBusy = true;
    resetInsights();
    if (state.liveTimer) {clearTimeout(state.liveTimer); state.liveTimer = null;}
    refreshLiveControls();
    $("live-state-label").textContent = action === "start" ? "Starting monitoring…" : "Stopping monitoring…";
    const payload = action === "start" ? {interface: $("live-interface").value, mode: $("live-mode").value} : {};
    try {
      const snapshot = await api(`/api/live/${action}`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
      if (generation === state.liveGeneration) renderLiveSnapshot(snapshot);
    } catch (error) {if (generation === state.liveGeneration) renderLiveFailure(error.message);}
    finally {state.liveBusy = false; refreshLiveControls(); scheduleLivePoll();}
  }
  function changeInputMode(index) {
    const nextMode = ["live", "sample", "upload"][index];
    if (nextMode !== state.inputMode) {
      state.inputMode = nextMode;
      state.request += 1;
      state.selectedRow = null;
      for (const button of $("batch-results").children) button.setAttribute("aria-pressed", "false");
      if (state.status) showError("");
      setBusy(false);
      clearReport();
    } else resetInsights();
    const live = state.inputMode === "live";
    $("live-report-panel").hidden = !live;
    $("report-panel").hidden = live;
    $("batch-section").hidden = state.inputMode !== "upload" || !state.rows.length;
    $("input-mode-tag").textContent = live ? "THIS PC" : "DATASENSE FEATURES";
    $("input-context-note").textContent = state.status?.partial_input_supported === false
      ? "The original CSV checkpoint remains active. Partial-input 84-class inference is unavailable until an evaluated upgrade is promoted. Live rules and the binary model are separate."
      : live
      ? "Packet mode uses measured common columns in the shared 170-feature schema. The 84-class result is tentative; live rules and the binary packet classifier are separate comparisons."
      : "Full and partial CSVs use the shared XGBoost model and existing feature columns. Missing-input predictions are tentative. TreeSHAP explains the model; unknown-attack detection is not enabled.";
    if (live) pollLive();
    else scheduleLivePoll();
  }
  const selectInputMode = setupTabs([$("live-tab"), $("sample-tab"), $("upload-tab")], [$("live-panel"), $("sample-panel"), $("upload-panel")], changeInputMode);
  const initialModeIndex = ["live", "sample", "upload"].indexOf(document.body.dataset.workspaceMode);
  if (initialModeIndex > 0) selectInputMode(initialModeIndex, false, false);
  $("live-interface").addEventListener("change", () => {resetInsights(); refreshLiveControls();});
  $("live-mode").addEventListener("change", () => {resetInsights(); refreshLiveControls();});
  $("live-start").addEventListener("click", () => liveAction("start"));
  $("live-stop").addEventListener("click", () => liveAction("stop"));
  $("sample-select").addEventListener("change", () => {state.request += 1; clearReport(); setBusy(false);});
  $("analyze-button").addEventListener("click", () => {
    for (const button of $("batch-results").children) button.setAttribute("aria-pressed", "false");
    analyze("/api/analyze", {sample_id: Number($("sample-select").value)});
  });
  $("csv-file").addEventListener("change", () => {
    state.request += 1; clearReport(); renderBatch([]); setBusy(false);
    $("file-name").textContent = $("csv-file").files[0]?.name || "Files are processed for this request and are not saved by the app.";
  });
  $("upload-button").addEventListener("click", upload);
  $("download-report").addEventListener("click", () => {
    if (!state.report) return;
    downloadJson({...state.report, selected_evidence_explanations: exportedInsights()}, "netsec-analysis.json");
  });
  $("download-live-report").addEventListener("click", exportLive);
  $("download-live-csv").addEventListener("click", exportLiveCsv);
  $("extra-xai-button").addEventListener("click", () => requestInsight("xai"));
  $("assistant-button").addEventListener("click", () => requestInsight("ai"));

  async function initialize() {
    const startup = (path) => api(path, {signal: AbortSignal.timeout(15000)});
    const results = await Promise.allSettled([startup("/api/status"), startup("/api/samples"), startup("/api/live/capabilities"), startup("/api/assistant/status")]);
    const errors = [];
    try {
      if (results[0].status === "fulfilled") renderStatus(results[0].value);
      else throw results[0].reason;
    } catch (error) {
      state.status = null;
      $("model-name").textContent = "Model unavailable";
      $("model-dataset").textContent = "Saved artifacts could not be loaded";
      document.documentElement.dataset.modelState = "error";
      errors.push(error.message);
    }
    try {
      if (results[1].status === "fulfilled") renderSamples(results[1].value);
      else throw results[1].reason;
    } catch (error) {
      $("sample-select").replaceChildren(node("option", "", "Validation samples unavailable"));
      errors.push(error.message);
    }
    try {
      if (results[2].status === "fulfilled") renderCapabilities(results[2].value);
      else throw results[2].reason;
    } catch (error) {
      $("live-interface").replaceChildren(node("option", "", "Monitoring connection unavailable"));
      renderLiveFailure(error.message);
    }
    if (results[3].status === "fulfilled") renderAssistantStatus(results[3].value);
    else {
      state.assistant = null;
      $("assistant-model-status").textContent = "Local AI status unavailable. No substitute model or template is used.";
    }
    showError(errors.join(" "));
    refreshControls();
    window.netsecUI?.setState(state.status ? null : "error", errors.join(" "));
    pollLive();
  }
  changeInputMode(["live", "sample", "upload"].indexOf(state.inputMode));
  setupWorkspaceMotion();
  initialize();
})();
