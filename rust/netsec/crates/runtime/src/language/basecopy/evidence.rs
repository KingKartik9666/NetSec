//! Displayable statements come only from server-owned observations and vetted XAI guidance.
use serde::Serialize;
use serde_json::Value;

#[derive(Clone, Debug, Serialize)]
pub struct Point { pub id: String, pub text: String, pub kind: String, pub source: String }
fn point(id: &str, text: impl Into<String>, kind: &str, source: &str) -> Point {
    Point { id: id.into(), text: text.into(), kind: kind.into(), source: source.into() }
}
pub fn points(prediction: &Value, label: &str, explanation: Option<&Value>) -> Vec<Point> {
    let name = prediction["display_label"].as_str().or_else(|| prediction["label"].as_str()).unwrap_or("an unavailable category");
    let mut out = vec![point("P1", format!("{label} predicts {name} for this observation."), "prediction", "saved model result"),
        point("P2", if prediction["score_kind"] == "vote_correctness" {
            "Its confidence estimates the correctness of a hard class vote; it is not an ensemble class probability."
        } else { "The model's probability is an estimate. It is not independent evidence that the predicted category is correct." }, "prediction", "score semantics")];
    if let Some(xai) = explanation {
        let method = xai["method"].as_str().unwrap_or("unknown");
        let units = xai["units"].as_str().unwrap_or("model score");
        out.push(point("X0", format!("The selected explanation uses {method}; its recorded output units are {units}."), "xai", "stored explanation"));
        let mut features = xai["features"].as_array().into_iter().flatten()
            .filter_map(|f| f["contribution"].as_f64().filter(|v| v.is_finite()).map(|v| (v, f))).collect::<Vec<_>>();
        features.sort_by(|(a, _), (b, _)| b.abs().total_cmp(&a.abs()));
        for (i, (value, feature)) in features.into_iter().take(4).enumerate() {
            let name = feature["feature"].as_str().unwrap_or("the recorded feature group");
            let direction = if value > 0. { "positive" } else if value < 0. { "negative" } else { "zero" };
            let text = match method {
                "permutation" => format!("{name} has a displayed {direction} Brier-loss change of {value:.6} on the reference data. This is global importance, not the cause of this individual result."),
                "ale" => format!("The recorded local effect for {name} is {value:.6} {units}. It describes a model response in the reference range, not a causal effect."),
                "anchors" | "counterfactual" => continue,
                _ => format!("{name} has a {direction} contribution of {value:.6} {units} relative to the explanation's reference. A contribution describes this model's explained output, not a proven cause in the network."),
            };
            out.push(point(&format!("X{}", i + 1), text, "xai", "stored feature attribution"));
        }
        if method == "anchors" {
            out.push(point("X1", if xai["details"]["state"] == "supported_anchor" {
                "The anchor has measured reference support. Its precision and coverage apply only within the recorded sampling limits."
            } else { "No sufficiently supported anchor rule was found in the recorded reference search. This does not prove that no useful rule exists." }, "xai", "stored anchor search"));
        }
        if method == "counterfactual" {
            out.push(point("X1", if xai["details"]["decision_change_verified"] == true {
                "An observed reference with the same measured-feature mask changes the model decision. This is a model probe, not a causal remedy or an instruction to change network measurements."
            } else { "No feasible observed-reference decision change was verified within the recorded search budget." }, "xai", "stored counterfactual search"));
        }
        let description = match method {
            "shap" => "SHAP compares the explained model output with a reference. Inspect the recorded units and reconstruction or approximation checks; contributions are not causal effects.",
            "lime" => "LIME fits a local surrogate near the observation. Its fidelity and model-score gap matter: a readable surrogate can still approximate the model poorly.",
            "anchors" => "An anchor is a local rule assessed on reference samples. Precision, coverage and support limit how far the rule can be generalized.",
            "counterfactual" => "A counterfactual asks whether a feasible model input could change its decision. A changed prediction does not establish an effective real-world intervention.",
            "permutation" => "Grouped permutation importance measures reference-data performance changes after related measurements are permuted together. It is global importance, not a local causal explanation.",
            "ale" => "Accumulated local effects summarize model responses across supported reference ranges. Sparse bins and feature dependence limit interpretation.",
            _ => "Read the explanation's scope, units, quality checks and warnings before interpreting it.",
        };
        out.push(point("K1", description, "method", "NetSec XAI guidance"));
    } else {
        out.push(point("X0", "This context has a completed prediction but no stored XAI explanation. Generate an explanation before asking which features influenced it; feature influence cannot be inferred from a class label alone.", "xai", "context availability"));
        out.push(point("K1", "XAI describes how a model responds to its input and reference. Generate SHAP, LIME or another supported method in the analysis workspace to inspect actual feature evidence.", "method", "NetSec XAI guidance"));
    }
    out.extend([
        point("C1", "This is a model prediction requiring human verification. It does not confirm an attack or guarantee that the observed activity is safe.", "limitation", "NetSec evidence policy"),
        point("C2", "Research validation scores do not establish accuracy on this PC or prove that this individual prediction is correct. Agreement between models is descriptive, not ground truth.", "limitation", "NetSec research boundaries"),
        point("C3", "Feature attribution describes model behavior relative to a reference; it cannot establish causation or identify an attacker by itself.", "limitation", "NetSec XAI guidance"),
        point("N1", "First check the selected input, time window and feature coverage. Preserve missing measurements as missing; do not replace them with invented values or measured zeros.", "guidance", "input verification guidance"),
        point("N2", "Compare the displayed influential measurements with their recorded values and reference. Check the explanation's scope and units before interpreting the contribution's direction.", "guidance", "XAI review guidance"),
        point("N3", "Inspect explanation quality, warnings and reference support. If methods or models disagree, review that disagreement rather than treating a majority as verified confidence.", "guidance", "XAI review guidance"),
        point("N4", "Correlate this observation with your own authorized logs and expected activity at the same time. These are suggested human checks; NetSec chat does not block devices, execute commands or change network policy.", "guidance", "human verification guidance"),
    ]);
    out
}
pub fn candidates(all: &[Point], question: &str) -> Vec<Point> {
    let q = question.to_lowercase();
    let prefixes: &[&str] = if ["next", "should i", "what to do", "check", "recommend", "mitigat", "action"].iter().any(|s| q.contains(s)) {
        &["N"]
    } else if ["certain", "confidence", "probab", "trust", "accur", "safe", "confirm", "limit", "caus", "prove"].iter().any(|s| q.contains(s)) {
        &["C", "P2"]
    } else if ["mean", "method", "shap", "lime", "anchor", "counterfactual", "permutation", "ale"].iter().any(|s| q.contains(s)) && !q.contains("feature") {
        &["K", "X0"]
    } else if ["feature", "contribut", "positive", "negative", "measurement", "influenc"].iter().any(|s| q.contains(s)) {
        &["X", "N2"]
    } else { &["P1", "X", "C1"] };
    all.iter().filter(|p| prefixes.iter().any(|s| p.id.starts_with(s))).cloned().collect()
}
pub fn in_scope(question: &str, all: &[Point]) -> bool {
    let q = question.to_lowercase();
    if ["ignore previous", "ignore all", "system prompt", "developer message", "act as", "you are now", "jailbreak", "<|im_", "[inst]"].iter().any(|s| q.contains(s)) { return false; }
    let words = q.split(|c: char| !c.is_alphanumeric()).collect::<Vec<_>>();
    let topics = ["xai", "shap", "lime", "anchors", "anchor", "counterfactual", "permutation", "ale", "netsec", "model", "models", "feature", "features", "evidence", "result", "prediction", "observation", "confidence", "probability", "contribution", "input", "window", "traffic", "attack", "safe", "uncertain", "uncertainty", "causal", "causation", "check", "verify", "verification", "explain", "explanation",
        // Task 5 Ruling 2: context words of follow-ups ("How sure is it?", "Why didn't it pick benign?", "What does raw margin mean as a unit?").
        "class", "classes", "runner", "pick", "picked", "chose", "benign", "margin", "unit", "units", "value", "score", "scores", "label", "detector", "sure", "unsure", "confident"];
    topics.iter().any(|s| words.contains(s)) || ["why", "why this", "how so", "what next", "what should i do", "what is going on", "tell me more", "explain more", "in simpler words", "make it clearer", "continue", "what does that mean"].contains(&q.trim())
        || all.iter().filter(|p| p.kind == "xai").any(|p| p.text.split_whitespace().next().is_some_and(|name| name.len() > 4 && q.contains(&name.to_lowercase())))
}
