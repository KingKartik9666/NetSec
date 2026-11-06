# NetSec XAI chat

The local assistant has one job (owner redesign, 3 November 2026): translate the
currently selected XAI result into plain English. It does not judge whether the
model is right, gives no advice and does not discuss ML in general. Any other
task gets one fixed line: "I can only explain this result and its XAI evidence."

Run a model in Samples, CSV analysis or Models, choose an XAI method (SHAP, LIME,
anchors, counterfactual, permutation importance or ALE) and generate its
explanation; then choose **Explain in plain English** or **Discuss this evidence**.
A completed packet-window result offers **Discuss this window**. The chat keeps the
selected evidence pinned; new predictions and live windows do not replace it.
Without a selected XAI result the explainer does not start: the buttons are
disabled, the chat shows "Choose an XAI method and generate its explanation
first..." and the server refuses any turn before a model call.

Each method has its own fixed NetSec preset (the owner's binding text in
`artifacts/explainer_20261103/PRESETS.md`, copied byte for byte into the Rust
test data). The preset tells the model what the method computes, how to read its
values and what it must not be read as. It is for the model only and is never
shown. The initial explanation shows two parts, each written by the model and
verified sentence by sentence: (1) what this XAI shows for this result (at most
two sentences) and (2) why the model gave this answer according to this XAI (at
most four sentences, naming features, values and directions exactly as recorded;
for anchors the recorded rule condition, precision and coverage; for a
counterfactual the recorded from -> to changes and the class the model would then
give). A part whose model text fails verification is replaced by that part's
NetSec template. Every answer except the refusal ends with one fixed footer,
labelled NetSec text, never Model or Template: "XAI shows how the model reached
its answer, not whether it is correct." It replaces the earlier per-sentence
caveats. **Explain it more simply**, **Summarize** and **More detail** and free-text
questions about this result follow the same preset. Clear chat cancels
an active answer and clears that conversation. The app retains at most sixteen
contexts and twenty turns per context in session memory; closing it clears them.

## Verified generation (since 1-2 November 2026)

The installed model is Qwen3-4B-Instruct-2507 Q4_K_M (Apache-2.0), pinned by
`artifacts/llm/model.json` (SHA-256 `3605803b...7e597`). It runs offline in one
loopback `llama-server` child (llama.cpp b11564): `127.0.0.1` only, random port,
per-session API key, started in the background when an explanation context opens,
unloaded after 10 idle minutes and killed with the app. The CUDA build is tried
first; if it cannot start, the CPU build (8 threads) answers and the reply says
CPU answers are slower. Detectors always stay on the CPU. Nothing runs before a
background pass has hashed the GGUF and every server file against its pin.

Each turn works like this:

1. A pre-model gate refuses questions over 300 characters, chat-template tokens,
   off-topic requests and listed misuse forms with an app-policy reply that is not
   labelled as model output (`llm_executed: false`). Since the owner decision of
   2 November 2026 the gate is not required to stop every obfuscated misuse form;
   the verifier below is the hard lock. Since pass-rate round 1 (3 November) the gate
   also admits the selected method's own words ("What is the coverage?" on an anchors
   result, "Why is the effect negative?" on ALE, "What does this method mean?"), but
   only in a question made entirely of known question, topic and method words, so
   "What is the importance of exercise?" stays off topic. The red-team gate replay
   adds these questions on real contexts of all six methods (48 in-scope, 12
   off-task).
2. The model sees the fixed BASE rules and misuse-hardening rules, exactly one
   method preset (the selected one), the server-owned evidence packet (prediction,
   confidence, runner-ups and the selected method's recorded values), at most two
   earlier verified answers (without the footer) and the question as quoted JSON
   data.
3. Tokens stream in. Each finished sentence is checked against the packet:
   numbers at their stated precision, feature names, attack names, direction,
   certainty, instructions, links, encoded text and, since 3 November, any
   judgement of whether the model or its answer is correct. Since the redesign
   reviews of 3 November the checks are also method-specific. To make anchors,
   counterfactuals and the LIME fit translatable, the evidence packet now carries
   their recorded values (rule condition, tolerance, precision, coverage, rows
   tested; from/to changes and the new class; R squared). That widens what the
   evidence contains, so a sentence stating those recorded values can now pass where
   the base verifier (which had no such values) rejected it. Correction (pass-rate
   round 1, 3 November): earlier versions of this page said every other rule change
   was stricter; the round 2 review (29 sentences) and the round 3 review (37
   sentences: invented counterfactual directions and multipliers, from/to values
   given to the wrong or several features, anchor statistics and the LIME fit read as
   the model's confidence or correctness, attack consequences after the
   counterfactual class) showed hostile base-rejected sentences passing through that
   widening. Each of them now fails. The base-vs-new check is live, not a recorded
   verdict: the frozen base verifier is compiled into the tests
   (`language/basecopy`, test-only), and `review3_probes_fail_and_base_rejections_stay_rejected`
   asserts that every sentence of `testdata/hostile_sentences.json` and
   `testdata/review3_probes.json` that the base rejects is still rejected (the
   ignored `probe_harness` test does the same for any probe file). What still passes
   only because of the widening are recorded facts in their recorded form; that is a
   tested property of the listed probes and of 1371 real model sentences from the
   redesign rounds, not a proof for all wording. The rules:
   a number after a named feature must be one recorded for that feature (every method
   since pass-rate round 1); a number worded as a value must be the feature's recorded
   value (a linked group has none) and one worded as a contribution, importance,
   effect or weight its recorded contribution; a counterfactual's from/to values
   belong to the feature named right before them, within 5% of the recorded value,
   never to no feature or to several features listed together; every movement word
   about a single counterfactual change (rose, fell, grew, dropped, climbed, jumped,
   stretched, ...) must match its recorded from-to sign, and multiplier words
   (doubled, halved, tenfold, twice, ...) fail everywhere; anchor statistics and the
   LIME fit are not shared numbers: each is accepted only with its own noun or its
   recorded phrase (precision/kept, lower bound, coverage/covered, "applied to N of the
   reference ...", tolerance, "within N of this observation's values", N matching
   reference rows tested; LIME fit or R squared with LIME's local model as subject),
   and never in a sentence with sure, likely, confidence, probability, chance, truth,
   trust or "the model was/is";
   the counterfactual's new class may be named only by a change cue whose subject is
   the model ("the model would predict <class>", "the new class would be <class>"),
   never with the traffic, a host, a device or a person as the subject and never as
   "the new class is"; a change is never framed as someone's action ("we could
   lower ...", "a sender could ..."), as a means ("by lowering <feature>") or as a
   purpose ("is enough so", "to get", "in order to"); action advice (should, must,
   need to, isolate, investigate, patch, escalate, imperatives such as "check the
   host") is never shown; a sentence in which a host or the traffic carries out an
   action ("the host sent an HTTP flood") never names a class; a linked
   counterfactual group "changed together" and no direction is stated about it; a
   direction about the score must match both the recorded score difference and each
   named feature's recorded change; permutation importance is signed and global (no
   "influence" claim about a feature with importance 0 or below, no wording about this
   prediction); ALE effects are signed effects at the value, never a change from one
   value to another; an unsupported anchor is never called supported and never the
   condition or reason for the prediction; ALE, permutation and anchors give no
   "because"; no answer change is stated as having happened; "perfect" is certainty
   wording. Runner-up class names count only in their exact recorded form ("HTTP
   flood port 1883"); the unreviewed "HTTP flood on port 1883" exemption of the
   pass-rate round was reverted and is an open owner proposal. A part-1 sentence must
   name this result (a recorded feature, number or search outcome, or the predicted
   class in a sentence that says more than "the model predicted <class>"); a bare
   "The model predicted <class>." in part 1 is dropped (not shown, not counted), and a
   part 1 with nothing else falls back to its template as a failed check. In every
   part, a sentence that only describes how the method works (the LLM-only preset
   description) is not shown, and an exact repeat of a sentence already shown in the
   reply is dropped. The round 3 security review added, all stricter: no agent
   (person, bot, host) doing a change to a feature ("once a bot lowers ..."), no
   evasion wording ("how to", "stay hidden", "pass as"), no reassurance ("safe",
   "relax", "no threat"), no unhedged attack claim ("under attack", "an attack is
   underway"), no counterfactual class in a sentence about a host or device, a
   statistic followed by another quantity ("0.15 was the confidence") is not that
   statistic, and a sentence with no word about this result or its XAI ("Write a
   Python function ...") fails as `off_topic`. Pass-rate round 1 added, all stricter:
   the counterfactual class is licensed only in a clause that ends with it (no
   "which", "meaning", "so" ... after it, no person in the sentence); attack mechanics
   ("overwhelms a server", "steals credentials", "at risk") fail with any class; the
   recorded change never "still" gives the predicted class; more correctness wording
   ("made the right call", "spot on", "confirms", "makes sense", "no reason to doubt",
   "can be relied on", "good job", ...); a claim about every or no feature ("No
   features pulled the result away", "All other features have an importance of 0")
   fails as `generalization` unless it is about the listed, recorded, top or shown
   features, because the packet lists only the top features; and `off_topic` no longer counts the generic
   words model, data, network, traffic, flow, rate, input, output or method, and
   rejects instruction talk ("my system prompt", "I was told") and general-ML or
   other-task text ("machine learning", "training data", "poem", "capital of", ...).
   Residual, stated plainly: `off_topic` is a vocabulary and pattern check, not
   semantic; an unrelated clause joined to a valid one ("..., and <some unrelated
   fact>") is caught only if it matches a pattern (a clause-level check rejected
   faithful anaphoric clauses such as "but these were minor" on real text and was not
   kept). A passing sentence is shown
   with a **Model** badge. The first failing sentence is never shown: generation
   stops and NetSec template sentences for that question kind fill the rest,
   labelled **Template (model output failed verification)**. A deadline or server
   error gives **Template (model unavailable)**. There is no retry.
4. First token must arrive within 15 s of the request; the whole answer is capped
   at 60 s. Token caps: initial part 1 90 and part 2 220, simpler 120, summary 120,
   detail 300, other 220.
   On GPU the prompt carries the last two verified answers; on CPU it carries only
   the last one, cut to whole sentences within 600 characters, so that a full
   prompt re-evaluation stays under 15 s (estimated 11.9 s p95, 2 Nov 2026).

The backend does not accept client-authored predictions, explanations or message
history as trusted evidence. Stored explanation IDs must match the input, model
and row. The assistant has no browsing, shell, capture, file-edit or mitigation
tools. A verified sentence is consistent with the stored evidence; it is not
proof that the prediction or the XAI is correct, and wrong predictions are
explained fluently, so the fixed footer always stays.

`GET /api/v1/chat/contexts/{id}/events` streams the context view (Server-Sent
Events); the UI releases verified sentences letter by letter and polls the context
every 300 ms only to recover a dropped stream.

## Measurements and evaluation

`.venv/Scripts/python.exe -m tools.verify_native_app` writes `english_matrix.json`
(four validation samples x initial/simpler/summary/detail on GPU and on CPU, plus a
separate CPU cache-miss row with interleaved contexts on different samples).
`method_matrix` adds the per-method row of the redesign: one initial explanation
per sample for every XAI method that can be generated offline.
`.venv/Scripts/python.exe tools/eval_explainer.py` runs 60 fixed validation cases
(about 10 where the detector is wrong) and writes rates, latency and a blind
readability CSV to `artifacts/locked_explainer_20261031/eval/`. Readability
ratings stay "pending human rater" until a non-technical reader fills them in.
Measured on 2 Nov 2026 with the GPU otherwise idle: GPU first token p95 0.2 s, CPU
in-order follow-ups p95 4.4 s. Over 60 cases, 55.8% of turns used some template
fallback. Measured values are recorded in `artifacts/locked_explainer_20261031/changes.md`
(Task 7) and the dated entry in `docs/PROJECT_CONTEXT.md`.

Hardware used for these measurements: Core i7-13620H, about 15.7 GiB RAM and an
RTX 4050 Laptop GPU with about 6 GiB VRAM shared with research training jobs. The
language server uses the GPU only when at least 3600 MiB is free.
