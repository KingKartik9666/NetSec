//! Server-owned evidence packet for one prediction and its selected XAI result, and the attack-name lexicon the sentence verifier enforces.
use crate::application::display_label;
use serde_json::Value;
use std::collections::{BTreeMap,BTreeSet};

/// `group_size > 1` marks a linked XAI group: `name` is the family stem, not a member, and the group has no single recorded value.
#[derive(Clone,Debug,PartialEq)]
pub struct Feature { pub name: String, pub value: Option<f64>, pub contribution: f64, pub group_size: usize }
/// The anchor search exactly as recorded: whether its rule reached support, the condition group, its tolerance, precision, precision lower
/// bound, coverage and the number of matching reference rows tested.
#[derive(Clone,Debug,PartialEq)]
pub struct Anchor { pub supported: bool, pub condition: Option<String>, pub tolerance: Option<f64>, pub precision: Option<f64>, pub lower_bound: Option<f64>,
    pub coverage: Option<f64>, pub support: Option<f64> }
/// One recorded counterfactual change: this observation's value (`from`) and the changed reference row's value (`to`); a multi-member group has neither.
#[derive(Clone,Debug,PartialEq)]
pub struct Change { pub name: String, pub group_size: usize, pub from: Option<f64>, pub to: Option<f64> }
/// The counterfactual search exactly as recorded: whether a change was found and verified, the class the model gives after it (a catalog id),
/// the recorded score difference of the whole change, and the changed features.
#[derive(Clone,Debug,PartialEq)]
pub struct Counterfactual { pub found: bool, pub verified: bool, pub new_class: Option<String>, pub score_change: Option<f64>, pub changes: Vec<Change> }
/// Built only from the stored prediction and explanation; `runner_ups` hold class ids (display labels are derived in `prompt_block`).
/// `directional` is false for permutation, ALE and anchors: their values are importance, effect or rule statistics, not a push toward the
/// predicted class. For a counterfactual, `features` are its changed features (value = this observation's value, contribution = the recorded
/// score difference of the whole change).
#[derive(Clone,Debug)]
pub struct Packet { pub model_label: String, pub class_count: usize, pub class_id: String, pub class_label: String, pub confidence: Option<f64>,
    pub runner_ups: Vec<(String, f64)>, pub score_kind: String, pub method: Option<String>, pub units: Option<String>, pub directional: bool,
    pub features: Vec<Feature>, pub anchor: Option<Anchor>, pub counterfactual: Option<Counterfactual>, pub fidelity: Option<f64> }
fn finite(v: &Value) -> Option<f64> { v.as_f64().filter(|v| v.is_finite()) }
/// Generic XAI groups are named `"<first member> / group <n>"`; a single-member group keeps the catalog feature name, a multi-member group
/// keeps its family stem (`network_protocols_all__present_tcp` -> `network_protocols_all`) so no member is credited with the group's contribution.
fn base_name(name: &str, group_size: usize) -> &str {
    let base = name.rsplit_once(" / group ").filter(|(_, n)| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())).map_or(name, |(base, _)| base);
    if group_size > 1 { base.split_once("__").map_or(base, |(stem, _)| stem) } else { base }
}
fn group_size(f: &Value) -> usize { f["group_size"].as_u64().filter(|n| *n > 1).map_or(1, |n| n as usize) }
fn feature(f: &Value) -> Option<Feature> {
    let group_size = group_size(f);
    Some(Feature { name: base_name(f["feature"].as_str()?, group_size).into(), value: finite(&f["value"]), contribution: finite(&f["contribution"])?, group_size })
}
/// At most this many changed features of a counterfactual are carried (the recorded search changes a handful).
const MAX_CHANGES: usize = 8;
fn anchor(xai: &Value) -> Anchor {
    let (q, d) = (&xai["quality"], &xai["details"]);
    Anchor { supported: d["state"] == "supported_anchor", condition: d["condition_group"].as_str().map(String::from), tolerance: finite(&d["normalized_tolerance"]),
        precision: finite(&q["precision"]), lower_bound: finite(&q["precision_lower_bound"]), coverage: finite(&q["reference_coverage"]), support: finite(&q["distinct_reference_support"]) }
}
/// `to` values come from the recorded counterfactual row at the catalog index of a single-member feature; the new class is the catalog class
/// at the recorded candidate index. Without the catalog nothing is looked up, so nothing is invented.
fn counterfactual(xai: &Value, classes: &[String], all_features: &[String]) -> Counterfactual {
    let d = &xai["details"];
    let found = d["state"] == "observed_reference_counterfactual" && d["counterfactual"].is_array();
    let row = d["counterfactual"].as_array();
    let changes = if !found { Vec::new() } else { xai["features"].as_array().into_iter().flatten().filter_map(|f| {
        let size = group_size(f);let name = base_name(f["feature"].as_str()?, size).to_string();
        let to = (size == 1).then(|| all_features.iter().position(|n| *n == name)).flatten().and_then(|i| row.and_then(|r| r.get(i))).and_then(finite);
        Some(Change { from: if size == 1 { finite(&f["value"]) } else { None }, to, group_size: size, name })
    }).take(MAX_CHANGES).collect() };
    let score_change = if found { xai["features"].as_array().into_iter().flatten().find_map(|f| finite(&f["contribution"])) } else { None };
    Counterfactual { found, verified: found && d["decision_change_verified"] == true, score_change, changes,
        new_class: if found { d["candidate_class_index"].as_u64().and_then(|i| classes.get(i as usize)).cloned() } else { None } }
}
/// `build_with` without the catalog (no counterfactual `to` values or new class can be looked up).
pub fn build(prediction: &Value, model_label: &str, class_count: usize, explanation: Option<&Value>) -> Packet {
    build_with(prediction, model_label, class_count, explanation, &[], &[])
}
/// `classes` and `all_features` are the catalog's class ids and feature names, used only to read the recorded counterfactual row and class index.
pub fn build_with(prediction: &Value, model_label: &str, class_count: usize, explanation: Option<&Value>, classes: &[String], all_features: &[String]) -> Packet {
    let class_id = prediction["label"].as_str().unwrap_or("unavailable").to_string();
    let class_label = prediction["display_label"].as_str().map(String::from).unwrap_or_else(|| display_label(&class_id));
    let runner_ups = prediction["top_classes"].as_array().into_iter().flatten()
        .filter_map(|c| Some((c["label"].as_str()?.to_string(), finite(&c["score"])?))).filter(|(id, _)| *id != class_id).take(2).collect();
    let text = |key: &str| explanation.and_then(|x| x[key].as_str()).map(String::from);
    let method = text("method");
    let (anchor, counterfactual) = match (method.as_deref(), explanation) {
        (Some("anchors"), Some(xai)) => (Some(anchor(xai)), None),
        (Some("counterfactual"), Some(xai)) => (None, Some(counterfactual(xai, classes, all_features))),
        _ => (None, None),
    };
    let features = if let Some(cf) = &counterfactual {
        cf.changes.iter().map(|c| Feature { name: c.name.clone(), value: c.from, contribution: cf.score_change.unwrap_or(0.), group_size: c.group_size }).collect()
    } else if anchor.is_some() { Vec::new() } else {
        let mut features = explanation.and_then(|x| x["features"].as_array()).into_iter().flatten().filter_map(feature).collect::<Vec<_>>();
        features.sort_by(|a, b| b.contribution.abs().total_cmp(&a.contribution.abs()));
        features.truncate(5);
        features
    };
    let fidelity = (method.as_deref() == Some("lime")).then(|| explanation.and_then(|x| finite(&x["quality"]["local_fidelity_r2"]))).flatten();
    Packet { model_label: model_label.into(), class_count, confidence: finite(&prediction["score"]), runner_ups,
        score_kind: prediction["score_kind"].as_str().unwrap_or("unknown").into(), directional: !matches!(method.as_deref(), Some("permutation" | "ale" | "anchors")),
        units: text("units"), method, features, anchor, counterfactual, fidelity, class_id, class_label }
}
/// Four significant digits as a plain decimal (never exponent notation), at most 8 decimals; |x| < 1e-8 is written `0`.
pub(crate) fn plain(x: f64) -> String {
    if !x.is_finite() || x.abs() < 1e-8 { return "0".into(); }
    let decimals = (3 - x.abs().log10().floor() as i32).clamp(0, 8) as usize;
    let text = format!("{x:.decimals$}");
    if text.contains('.') { text.trim_end_matches('0').trim_end_matches('.').into() } else { text }
}
pub(crate) fn digit_runs(name: &str) -> Vec<f64> { name.split(|c: char| !c.is_ascii_digit()).filter_map(|d| d.parse::<f64>().ok()).collect() }
impl Packet {
    /// The recorded units in words (`raw_class_margin` -> `raw class margin`).
    pub fn units_text(&self) -> String { self.units.as_deref().unwrap_or("model score").replace('_', " ") }
    pub fn prompt_block(&self) -> String {
        let mut out = vec![format!("Model: {}", self.model_label),
            format!("Classes: the model chose one class out of {} possible classes.", self.class_count),
            format!("Predicted class: {}", self.class_label),
            match self.confidence {
                Some(c) => format!("Confidence: {} ({})", plain(c), if self.score_kind == "vote_correctness" { "estimated correctness of a hard class vote, not a class probability" } else { "the model's own probability estimate, not independent proof" }),
                None => "Confidence: not recorded".into(),
            },
            if self.runner_ups.is_empty() { "Runner-up classes: none recorded".into() } else {
                format!("Runner-up classes (less likely): {}", self.runner_ups.iter().map(|(id, s)| format!("{} (score {})", display_label(id), plain(*s))).collect::<Vec<_>>().join("; "))
            }];
        let units = self.units_text();
        let Some(method) = self.method.as_deref() else {
            out.push("XAI: no stored XAI explanation for this prediction; feature influence is unknown.".into());
            return out.join("\n");
        };
        out.push(format!("XAI method: {method}; units: {units}"));
        if let Some(a) = &self.anchor {
            let num = |x: Option<f64>| x.map_or("not recorded".into(), plain);
            out.push(format!("Anchor search result: {}", if a.supported { "a supported rule was found" } else { "no supported rule was found (the best candidate rule is listed)" }));
            out.push(match &a.condition {
                Some(c) => format!("Rule condition: {c} stays within normalized tolerance {} of this observation's values", num(a.tolerance)),
                None => "Rule condition: none recorded".into(),
            });
            out.push(format!("Precision (share of matching reference rows where the model kept the predicted class): {}", num(a.precision)));
            out.push(format!("Precision lower bound: {}", num(a.lower_bound)));
            out.push(format!("Coverage (share of reference rows that match the condition): {}", num(a.coverage)));
            out.push(format!("Matching reference rows tested: {}", num(a.support)));
            return out.join("\n");
        }
        if let Some(c) = &self.counterfactual {
            if !c.found {
                out.push("Counterfactual search result: no feasible change was found within the recorded search budget".into());
                return out.join("\n");
            }
            out.push(format!("Counterfactual search result: a recorded change to the measured values makes the model give a different answer ({})",
                if c.verified { "decision change verified" } else { "decision change not verified" }));
            out.push(format!("New class after the change: {}", c.new_class.as_deref().map_or("not recorded".into(), display_label)));
            if let Some(s) = c.score_change { out.push(format!("Recorded score difference for the whole change: {} ({units})", plain(s))); }
            out.push("Changed features (from this observation's value to the changed value):".into());
            for change in &c.changes {
                out.push(if change.group_size > 1 { format!("- {} (group of {} linked features): changed together, no single value", change.name, change.group_size) }
                    else { format!("- {}: from {} to {}", change.name, change.from.map_or("missing".into(), plain), change.to.map_or("not recorded".into(), plain)) });
            }
            return out.join("\n");
        }
        let kind = if method == "permutation" { "importance" } else if method == "ale" { "effect" } else { "contribution" };
        if self.features.is_empty() { out.push("Top features: none recorded for this method.".into()); }
        else if self.directional { out.push("Top features by absolute contribution (sign: toward or away from the predicted class):".into()); }
        else { out.push(format!("Top features by absolute {kind} (these values have no direction relative to the predicted class):")); }
        for f in &self.features {
            let (name, value) = if f.group_size > 1 { (format!("{} (group of {} linked features)", f.name, f.group_size), "no single value".into()) }
                else { (f.name.clone(), format!("value {}", f.value.map_or("missing".into(), plain))) };
            let direction = if !self.directional { String::new() } else if f.contribution > 0. { " (toward the predicted class)".into() }
                else if f.contribution < 0. { " (away from the predicted class)".into() } else { " (neutral)".into() };
            out.push(format!("- {name}: {value}, {kind} {}{direction}", plain(f.contribution)));
        }
        if let Some(r2) = self.fidelity { out.push(format!("Local model fit (R squared, how closely the simple local model matches the model near this point): {}", plain(r2))); }
        out.join("\n")
    }
    /// Every number the packet states as a fact of this observation: confidence, runner-up scores, class count, feature values and
    /// contributions (raw, unrounded), group sizes, the digit runs inside packet feature names (`1883` in `network_ports_src__present_1883`,
    /// `6` in `..._dhcpv6`) and the digit runs of the anchor's condition. The counterfactual's `to` values are not facts of this observation,
    /// so they are kept apart in `change_targets` (redesign round 1 review, finding 3). Redesign round 3 (review 2, finding 4): the anchor
    /// statistics and the LIME fit are not shared numbers either; the verifier accepts each only next to its own words.
    pub fn numbers(&self) -> Vec<f64> {
        let condition = self.anchor.iter().flat_map(|a| a.condition.as_deref().map(digit_runs).unwrap_or_default()).collect::<Vec<_>>();
        self.confidence.into_iter().chain(self.runner_ups.iter().map(|r| r.1)).chain([self.class_count as f64])
            .chain(self.features.iter().filter_map(|f| f.value)).chain(self.features.iter().map(|f| f.contribution))
            .chain(self.features.iter().filter(|f| f.group_size > 1).map(|f| f.group_size as f64))
            .chain(self.features.iter().flat_map(|f| digit_runs(&f.name)))
            .chain(condition).collect()
    }
    /// The recorded counterfactual `to` values: the changed reference row's values, stated only as the target of a recorded change.
    pub fn change_targets(&self) -> Vec<f64> { self.counterfactual.iter().flat_map(|c| c.changes.iter().filter_map(|ch| ch.to)).collect() }
    /// The bases of `%` tokens (x100): confidence and runner-up scores (anchor statistics as percentages pair with their own words).
    pub fn percents(&self) -> Vec<f64> {
        self.confidence.into_iter().chain(self.runner_ups.iter().map(|r| r.1)).map(|e| e * 100.).collect()
    }
}

pub struct Lexicon { pub by_class: BTreeMap<String, BTreeSet<String>>, pub all: BTreeSet<String> }
/// Single tokens that are protocols, ports or ordinary words, never attack names on their own.
const GENERIC: [&str; 24] = ["port", "ack", "http", "https", "connect", "data", "web", "benign", "normal", "tcp", "udp", "icmp", "syn", "frag",
    "push", "rst", "fin", "ssh", "telnet", "dictionary", "publish", "blind", "upload", "ip"];
const NOUNS: [&str; 5] = ["flood", "scan", "sweep", "spoofing", "injection"];
/// Always rejected unless a class owns them.
const ALWAYS: [&str; 6] = ["slowloris", "sql injection", "malware", "botnet", "ransomware", "exfiltration"];
fn synonyms(token: &str) -> &'static [&'static str] {
    match token {
        "ddos" | "dos" => &["denial of service"], "bruteforce" => &["brute force", "password guessing"],
        "recon" | "scan" => &["port scan", "scanning", "reconnaissance"], "mqtt" => &["mqtt"], "mitm" => &["man in the middle"], _ => &[],
    }
}
/// Family (`ddos`), rest of the id as a phrase without port/numeric segments (`icmp flood`), final attack noun (`flood`) and synonyms.
fn class_terms(id: &str) -> BTreeSet<String> {
    let id = id.to_lowercase();
    let (family, rest) = id.split_once('_').unwrap_or((&id, ""));
    let words = rest.split('-').filter(|s| !s.is_empty() && *s != "port" && !s.bytes().all(|b| b.is_ascii_digit())).collect::<Vec<_>>();
    let phrase = words.join(" ");
    let candidates = [family, phrase.as_str()].into_iter().chain(words.iter().rev().find(|w| NOUNS.contains(w)).copied())
        .chain([family].into_iter().chain(words.iter().copied()).flat_map(|t| synonyms(t).iter().copied()));
    candidates.filter(|t| !t.is_empty() && (t.contains(' ') || !GENERIC.contains(t))).map(String::from).collect()
}
const SUFFIXES: [&str; 7] = ["", "s", "es", "ed", "ing", "er", "ers"];
/// `stem` starts at a word boundary and ends in one of `suffixes` (optionally after a doubled final letter: `scanned`) and a word boundary.
fn inflected(text: &str, stem: &str, suffixes: &[&str]) -> bool {
    !stem.is_empty() && text.match_indices(stem).any(|(i, _)| {
        let rest = &text[i + stem.len()..];
        let doubled = stem.chars().next_back().and_then(|c| rest.strip_prefix(c)).into_iter().flat_map(|r| SUFFIXES[3..].iter().filter_map(move |s| r.strip_prefix(s)));
        !text[..i].chars().next_back().is_some_and(char::is_alphanumeric)
            && suffixes.iter().filter_map(|s| rest.strip_prefix(s)).chain(doubled).any(|r| !r.chars().next().is_some_and(char::is_alphanumeric))
    })
}
/// Lowercased whole-word (or whole-phrase) match with a strict left boundary (`dos` never matches inside `ddos`) and English inflections
/// on the right: `floods`, `flooding`, `scanned`, `injections`; `-ing` terms also match their stem (`spoofed`), `-e` terms drop it (`brute forced`).
pub fn contains_words(text: &str, phrase: &str) -> bool {
    inflected(text, phrase, &SUFFIXES) || phrase.strip_suffix("ing").is_some_and(|s| s.len() > 2 && inflected(text, s, &SUFFIXES))
        || phrase.strip_suffix('e').is_some_and(|s| s.len() > 2 && inflected(text, s, &SUFFIXES[3..]))
}
pub fn lexicon(classes: &[String]) -> Lexicon {
    let mut by_class = classes.iter().map(|c| (c.clone(), class_terms(c))).collect::<BTreeMap<_, _>>();
    let all = by_class.values().flatten().cloned().chain(ALWAYS.iter().map(|s| s.to_string())).collect::<BTreeSet<_>>();
    // A class also owns every lexicon term inside its own terms (`sql injection blind` owns `sql injection`, `mirai syn flood` owns `syn flood`).
    for terms in by_class.values_mut() {
        let owned = all.iter().filter(|t| terms.iter().any(|own| contains_words(own, t))).cloned().collect::<Vec<_>>();
        terms.extend(owned);
    }
    Lexicon { by_class, all }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn prediction()->Value{json!({"label":"ddos_icmp-flood","display_label":"ddos icmp flood","score":0.91,"score_kind":"probability",
        "top_classes":[{"label":"ddos_icmp-flood","display_label":"ddos icmp flood","score":0.91},{"label":"dos_icmp-flood","score":0.05},{"label":"benign","score":0.02}]})}
    #[test] fn packet_keeps_top_five_by_magnitude(){
        let features=[0.1,-0.9,0.3,0.05,-0.4,0.7,0.2].iter().enumerate().map(|(i,c)|json!({"feature":format!("f{i}"),"value":i as f64,"contribution":c})).collect::<Vec<_>>();
        let p=build(&prediction(),"XGBoost",77,Some(&json!({"method":"shap","units":"raw margin","features":features})));
        assert_eq!(p.features.iter().map(|f|f.contribution).collect::<Vec<_>>(),[-0.9,0.7,-0.4,0.3,0.2]);
        assert_eq!(p.features.iter().map(|f|f.name.as_str()).collect::<Vec<_>>(),["f1","f5","f4","f2","f6"]);
        assert_eq!(p.features[0].value,Some(1.));assert!(p.directional);assert_eq!((p.method.as_deref(),p.units.as_deref()),(Some("shap"),Some("raw margin")));
        assert_eq!(p.numbers(),[0.91,0.05,0.02,77.,1.,5.,4.,2.,6.,-0.9,0.7,-0.4,0.3,0.2,1.,5.,4.,2.,6.]);
    }
    #[test] fn packet_without_xai_has_no_features(){
        let p=build(&prediction(),"XGBoost",77,None);
        assert!(p.features.is_empty());assert!(p.method.is_none());
        assert_eq!((p.class_id.as_str(),p.class_label.as_str(),p.confidence,p.class_count,p.model_label.as_str()),("ddos_icmp-flood","ddos icmp flood",Some(0.91),77,"XGBoost"));
        assert_eq!(p.runner_ups,[("dos_icmp-flood".to_string(),0.05),("benign".to_string(),0.02)]);
        assert_eq!(p.numbers(),[0.91,0.05,0.02,77.]);
        let block=p.prompt_block();assert!(block.contains("no stored XAI"),"{block}");assert!(!block.contains("Caveats")&&!block.contains("human verification"),"no caveat text: {block}");assert!(block.contains("Normal network activity (score 0.02)"),"{block}");assert!(block.contains("out of 77"),"{block}");
    }
    #[test] fn packet_without_probabilities_has_no_runner_ups(){
        let p=build(&json!({"label":"recon_scan","score":0.6,"score_kind":"vote_correctness","top_classes":[]}),"HEM-derived",77,None);
        assert!(p.runner_ups.is_empty());assert_eq!(p.class_label,"recon scan");assert_eq!(p.numbers(),[0.6,77.]);
        assert!(p.prompt_block().contains("hard class vote"));
        let none=build(&json!({"label":"benign","score":null}),"m",2,None);assert_eq!((none.confidence,none.numbers()),(None,vec![2.]));
    }
    #[test] fn packet_strips_group_suffix_skips_non_finite_and_marks_global_methods(){
        let xai=|method:&str|json!({"method":method,"units":"Brier loss increase","features":[{"feature":"flow_duration / group 3","value":1.5,"contribution":0.12},
            {"feature":"proto__tcp / group 17","value":null,"contribution":-0.03},{"feature":"bad / group 4","value":2.,"contribution":null},{"feature":"label only","contribution":"x"}]});
        let p=build(&prediction(),"XGBoost",77,Some(&xai("permutation")));
        assert_eq!(p.features,[Feature{name:"flow_duration".into(),value:Some(1.5),contribution:0.12,group_size:1},Feature{name:"proto__tcp".into(),value:None,contribution:-0.03,group_size:1}]);
        assert!(!p.directional);let block=p.prompt_block();assert!(block.contains("importance"),"{block}");assert!(!block.contains("group"),"{block}");assert!(!block.contains("toward"),"{block}");
        assert!(!build(&prediction(),"m",77,Some(&xai("ale"))).directional);assert!(build(&prediction(),"m",77,Some(&xai("exact_tree_shap"))).directional);
        for method in ["anchors","counterfactual"]{let p=build(&prediction(),"m",77,Some(&xai(method)));assert!(p.features.is_empty(),"{method}");assert_eq!(p.method.as_deref(),Some(method));}
    }
    #[test] fn multi_member_group_is_described_as_a_group(){
        let p=build(&prediction(),"m",77,Some(&json!({"method":"lime","features":[
            {"feature":"network_protocols_all__present_tcp / group 72","value":null,"contribution":0.3,"is_group":true,"group_size":80},
            {"feature":"log_messages_count / group 7","value":null,"contribution":-0.2,"is_group":true,"group_size":2},
            {"feature":"flow_duration / group 3","value":null,"contribution":0.1,"is_group":false,"group_size":1}]})));
        assert_eq!(p.features.iter().map(|f|(f.name.as_str(),f.group_size)).collect::<Vec<_>>(),[("network_protocols_all",80),("log_messages_count",2),("flow_duration",1)]);
        let block=p.prompt_block();
        assert!(block.contains("- network_protocols_all (group of 80 linked features): no single value, contribution 0.3 (toward the predicted class)"),"{block}");
        assert!(block.contains("- log_messages_count (group of 2 linked features): no single value, contribution -0.2 (away from"),"{block}");
        assert!(block.contains("- flow_duration: value missing, contribution 0.1"),"{block}");
        assert!(!block.contains("present_tcp"),"{block}");assert_eq!(block.matches("missing").count(),1,"{block}");
        assert_eq!(p.numbers(),[0.91,0.05,0.02,77.,0.3,-0.2,0.1,80.,2.]);
    }
    #[test] fn feature_name_digits_are_packet_numbers(){
        let p=build(&prediction(),"m",77,Some(&json!({"method":"exact_tree_shap","features":[
            {"feature":"network_ports_src__present_1883","value":1.,"contribution":0.2},{"feature":"network_protocols_all__present_dhcpv6","value":0.,"contribution":-0.1}]})));
        assert!(p.prompt_block().contains("- network_ports_src__present_1883: value 1, contribution 0.2"));
        assert_eq!(p.numbers(),[0.91,0.05,0.02,77.,1.,0.,0.2,-0.1,1883.,6.]);
    }
    #[test] fn prompt_numbers_are_plain_decimals(){
        for (value,text) in [(0.0000123456,"0.00001235"),(1e-9,"0"),(-1e-9,"0"),(0.9731234,"0.9731"),(-0.42,"-0.42"),(12345.6,"12346"),(3.,"3"),(0.5,"0.5"),(1.23e-7,"0.00000012"),(9.99996,"10")]{assert_eq!(plain(value),text,"{value}");}
        let p=build(&prediction(),"m",77,Some(&json!({"method":"shap","features":[{"feature":"a","value":12345678.9,"contribution":0.0000123456}]})));
        let block=p.prompt_block();assert!(block.contains("a: value 12345679, contribution 0.00001235 (toward"),"{block}");
        assert!(!block.chars().zip(block.chars().skip(1)).any(|(a,b)|a.is_ascii_digit()&&(b=='e'||b=='E')),"{block}");
    }
    /// The real per-method explanations of the four matrix samples (language/testdata/matrix_methods.json) with the real 77-class catalog.
    fn real(sample:&str,method:&str)->Packet{
        let dir=std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let read=|p:std::path::PathBuf|serde_json::from_slice::<Value>(&std::fs::read(&p).unwrap()).unwrap();
        let catalog=read(dir.join("../../../../artifacts/application/native_models_v1/catalog.json"));
        let strings=|key:&str|catalog[key].as_array().unwrap().iter().map(|v|v.as_str().unwrap().to_string()).collect::<Vec<_>>();
        let v=read(dir.join("src/language/testdata/matrix_methods.json"));let v=&v[sample];
        build_with(&v["prediction"],"XGBoost",77,Some(&v["explanations"][method]),&strings("classes"),&strings("feature_names"))
    }
    /// Redesign 3 Nov: the anchor search's RECORDED rule condition, precision, lower bound, coverage and support are packet facts.
    #[test] fn anchors_packet_records_rule_condition_precision_and_coverage(){
        let p=real("benign","anchors");let a=p.anchor.as_ref().expect("anchor recorded");
        assert_eq!((a.supported,a.condition.as_deref(),a.tolerance,a.coverage,a.support),(false,Some("log_data-types (linked measurements)"),Some(0.15),Some(0.828125),Some(15.)));
        assert!((a.precision.unwrap()-0.0666667).abs()<1e-6&&(a.lower_bound.unwrap()-0.0118669).abs()<1e-6,"{a:?}");
        let block=p.prompt_block();
        for line in ["Anchor search result: no supported rule was found (the best candidate rule is listed)","Rule condition: log_data-types (linked measurements) stays within normalized tolerance 0.15 of this observation's values",
            "Precision (share of matching reference rows where the model kept the predicted class): 0.06667","Precision lower bound: 0.01187",
            "Coverage (share of reference rows that match the condition): 0.8281","Matching reference rows tested: 15"]{assert!(block.contains(line),"missing {line}:\n{block}");}
        // Review 2 finding 4: anchor statistics are not shared numbers or percents (the verifier pairs each with its own words).
        for n in [0.15,0.828125,15.]{assert!(!p.numbers().contains(&n),"{n}");}
        assert!(!p.percents().iter().any(|x|(x-82.8125).abs()<1e-9||(x-6.666667).abs()<1e-5),"{:?}",p.percents());
        assert!(p.features.is_empty()&&!block.contains("Caveats"),"{block}");
    }
    /// Redesign 3 Nov: the counterfactual's RECORDED changes (from this row's value to the reference row's value), new class and score change.
    #[test] fn counterfactual_packet_records_from_to_changes_and_new_class(){
        let p=real("ddos","counterfactual");let c=p.counterfactual.as_ref().expect("counterfactual recorded");
        assert!(c.found&&c.verified,"{c:?}");assert_eq!(c.new_class.as_deref(),Some("mitm_impersonation"));assert!((c.score_change.unwrap()+0.998104).abs()<1e-6);
        assert_eq!(c.changes.len(),7);
        let interval=&c.changes[2];assert_eq!((interval.name.as_str(),interval.group_size),("log_interval-messages",1));
        assert!((interval.from.unwrap()-192.545456).abs()<1e-5&&(interval.to.unwrap()-96.63636).abs()<1e-5,"{interval:?}");
        assert_eq!((c.changes[4].name.as_str(),c.changes[4].group_size,c.changes[4].from,c.changes[4].to),("Network measurements and indicators",426,None,None));
        let block=p.prompt_block();
        for line in ["Counterfactual search result: a recorded change to the measured values makes the model give a different answer (decision change verified)",
            "New class after the change: mitm impersonation","Recorded score difference for the whole change: -0.9981 (calibrated class probability)",
            "- log_interval-messages: from 192.5 to 96.64","- Network measurements and indicators (group of 426 linked features): changed together, no single value",
            "- listed_src_ports_per_src_ip: from 0.2857 to 0.6667"]{assert!(block.contains(line),"missing {line}:\n{block}");}
        // Review finding 3: a `to` value is a change target, not a value this observation had.
        assert!(p.change_targets().iter().any(|x|(x-96.63636).abs()<1e-4)&&!p.numbers().iter().any(|x|(x-96.63636).abs()<1e-4),"to values are change targets only");
        assert_eq!(p.features.len(),7,"changed features are named packet features");assert!(p.features.iter().all(|f|(f.contribution+0.998104).abs()<1e-6));
        let none=real("scan","counterfactual");let c=none.counterfactual.as_ref().unwrap();assert!(!c.found&&c.changes.is_empty()&&c.new_class.is_none());
        assert!(none.prompt_block().contains("Counterfactual search result: no feasible change was found within the recorded search budget"),"{}",none.prompt_block());
        assert_eq!(real("unknown","counterfactual").counterfactual.unwrap().new_class.as_deref(),Some("benign"));
        // Without the catalog (old callers) no class or to-value is invented.
        let bare=build(&json!({"label":"x"}),"m",2,Some(&json!({"method":"counterfactual","features":[{"feature":"a","value":1.0,"contribution":-0.5}],"details":{"state":"observed_reference_counterfactual","candidate_class_index":1,"decision_change_verified":true,"counterfactual":[2.0]}})));
        let c=bare.counterfactual.unwrap();assert_eq!((c.new_class,c.changes[0].to),(None,None));
    }
    #[test] fn lime_packet_records_local_fit_and_no_caveats(){
        let p=real("benign","lime");assert!((p.fidelity.unwrap()-0.997045).abs()<1e-6);
        let block=p.prompt_block();assert!(block.contains("Local model fit (R squared, how closely the simple local model matches the model near this point): 0.997"),"{block}");
        assert!(!p.numbers().contains(&p.fidelity.unwrap())&&!block.contains("Caveats"),"{block}");
        assert!(real("benign","shap").fidelity.is_none()&&real("benign","shap").anchor.is_none()&&real("benign","shap").counterfactual.is_none());
    }
    #[test] fn lexicon_maps_class_to_hedgeable_terms(){
        let l=lexicon(&["ddos_icmp-flood".into(),"recon_port-scan".into()]);
        let icmp=&l.by_class["ddos_icmp-flood"];
        assert!(icmp.contains("icmp flood")&&icmp.contains("denial of service")&&icmp.contains("ddos")&&icmp.contains("flood"),"{icmp:?}");
        assert!(!icmp.contains("port scan"));assert!(l.all.contains("slowloris"));assert!(l.by_class["recon_port-scan"].contains("port scan"));
    }
    #[test] fn lexicon_drops_ports_and_generic_tokens(){
        let ids=["benign","ddos_http-flood-port-80","dos_syn-flood-port-22","dos_ack-frag-flood-port-1883","web_sql-injection","web_sql-injection-blind","malware_mirai-syn-flood","ddos_slowloris-port-8000","bruteforce_dictionary-ssh","mitm_arp-spoofing"].map(String::from);
        let l=lexicon(&ids);
        assert!(l.by_class["benign"].is_empty());assert!(!l.all.contains("benign"));
        for generic in ["port","http","ack","80","22","1883","port 80","web","ssh","dictionary","connect"]{assert!(!l.all.contains(generic),"{generic}");}
        assert!(l.by_class["ddos_http-flood-port-80"].contains("http flood"));
        let dos=&l.by_class["dos_syn-flood-port-22"];assert!(dos.contains("dos")&&dos.contains("denial of service")&&dos.contains("syn flood")&&!dos.contains("ddos"),"{dos:?}");
        assert!(l.by_class["dos_ack-frag-flood-port-1883"].contains("ack frag flood"));
        let blind=&l.by_class["web_sql-injection-blind"];assert!(blind.contains("sql injection")&&blind.contains("sql injection blind")&&blind.contains("injection"),"{blind:?}");
        let mirai=&l.by_class["malware_mirai-syn-flood"];assert!(mirai.contains("malware")&&mirai.contains("syn flood")&&mirai.contains("mirai syn flood"),"{mirai:?}");
        assert!(l.by_class["ddos_slowloris-port-8000"].contains("slowloris"));
        let brute=&l.by_class["bruteforce_dictionary-ssh"];assert!(brute.contains("brute force")&&brute.contains("password guessing")&&brute.contains("bruteforce"),"{brute:?}");
        for always in ["sql injection","malware","botnet","ransomware","exfiltration"]{assert!(l.all.contains(always));}
        assert!(!l.by_class["mitm_arp-spoofing"].contains("botnet"));
        assert!(l.by_class.values().flatten().all(|t|l.all.contains(t)));
    }
    #[test] fn word_matching_respects_boundaries(){
        assert!(contains_words("consistent with a dos attack","dos"));assert!(!contains_words("consistent with a ddos attack","dos"));
        assert!(contains_words("an icmp flood.","icmp flood"));assert!(!contains_words("x",""));
        for (text,phrase) in [("syn flooding","flood"),("icmp floods","flood"),("consistent with icmp floods.","icmp flood"),("it scans hosts","scan"),("hosts were scanned","scan"),
            ("a port scanner","scan"),("arp spoofed replies","arp spoofing"),("spoofs","spoofing"),("sql injections","injection"),("brute forced logins","brute force"),
            ("brute forcing","brute force"),("password guesses","password guessing"),("sweeps","sweep")]{assert!(contains_words(text,phrase),"{text} / {phrase}");}
        for (text,phrase) in [("a ddos attack","dos"),("ddosed","dos"),("floodlight","flood"),("scant data","scan"),("rescanned","scan"),("injectionx","injection")]{assert!(!contains_words(text,phrase),"{text} / {phrase}");}
    }
}
