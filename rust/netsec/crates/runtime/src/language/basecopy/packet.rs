//! Server-owned evidence packet for one prediction, and the attack-name lexicon the sentence verifier enforces.
use super::evidence;
use crate::application::display_label;
use serde_json::Value;
use std::collections::{BTreeMap,BTreeSet};

/// `group_size > 1` marks a linked XAI group: `name` is the family stem, not a member, and the group has no single recorded value.
#[derive(Clone,Debug,PartialEq)]
pub struct Feature { pub name: String, pub value: Option<f64>, pub contribution: f64, pub group_size: usize }
/// Built only from the stored prediction and explanation; `runner_ups` hold class ids (display labels are derived in `prompt_block`).
/// `directional` is false for permutation and ALE: their values are importance/effect, not a push toward the predicted class.
#[derive(Clone,Debug)]
pub struct Packet { pub model_label: String, pub class_count: usize, pub class_id: String, pub class_label: String, pub confidence: Option<f64>,
    pub runner_ups: Vec<(String, f64)>, pub score_kind: String, pub method: Option<String>, pub units: Option<String>, pub directional: bool,
    pub features: Vec<Feature>, pub caveats: Vec<String> }
fn finite(v: &Value) -> Option<f64> { v.as_f64().filter(|v| v.is_finite()) }
/// Generic XAI groups are named `"<first member> / group <n>"`; a single-member group keeps the catalog feature name, a multi-member group
/// keeps its family stem (`network_protocols_all__present_tcp` -> `network_protocols_all`) so no member is credited with the group's contribution.
fn base_name(name: &str, group_size: usize) -> &str {
    let base = name.rsplit_once(" / group ").filter(|(_, n)| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())).map_or(name, |(base, _)| base);
    if group_size > 1 { base.split_once("__").map_or(base, |(stem, _)| stem) } else { base }
}
fn feature(f: &Value) -> Option<Feature> {
    let group_size = f["group_size"].as_u64().filter(|n| *n > 1).map_or(1, |n| n as usize);
    Some(Feature { name: base_name(f["feature"].as_str()?, group_size).into(), value: finite(&f["value"]), contribution: finite(&f["contribution"])?, group_size })
}
pub fn build(prediction: &Value, model_label: &str, class_count: usize, explanation: Option<&Value>) -> Packet {
    let class_id = prediction["label"].as_str().unwrap_or("unavailable").to_string();
    let class_label = prediction["display_label"].as_str().map(String::from).unwrap_or_else(|| display_label(&class_id));
    let runner_ups = prediction["top_classes"].as_array().into_iter().flatten()
        .filter_map(|c| Some((c["label"].as_str()?.to_string(), finite(&c["score"])?))).filter(|(id, _)| *id != class_id).take(2).collect();
    let text = |key: &str| explanation.and_then(|x| x[key].as_str()).map(String::from);
    let method = text("method");
    let mut features = if matches!(method.as_deref(), Some("anchors" | "counterfactual")) { Vec::new() } else {
        explanation.and_then(|x| x["features"].as_array()).into_iter().flatten().filter_map(feature).collect::<Vec<_>>()
    };
    features.sort_by(|a, b| b.contribution.abs().total_cmp(&a.contribution.abs()));
    features.truncate(5);
    let caveats = evidence::points(prediction, model_label, explanation).into_iter().filter(|p| p.id.starts_with('C')).map(|p| p.text).collect();
    Packet { model_label: model_label.into(), class_count, confidence: finite(&prediction["score"]), runner_ups,
        score_kind: prediction["score_kind"].as_str().unwrap_or("unknown").into(), directional: !matches!(method.as_deref(), Some("permutation" | "ale")),
        units: text("units"), method, features, caveats, class_id, class_label }
}
/// Four significant digits as a plain decimal (never exponent notation), at most 8 decimals; |x| < 1e-8 is written `0`.
pub(crate) fn plain(x: f64) -> String {
    if !x.is_finite() || x.abs() < 1e-8 { return "0".into(); }
    let decimals = (3 - x.abs().log10().floor() as i32).clamp(0, 8) as usize;
    let text = format!("{x:.decimals$}");
    if text.contains('.') { text.trim_end_matches('0').trim_end_matches('.').into() } else { text }
}
impl Packet {
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
        match self.method.as_deref() {
            None => out.push("XAI: no stored XAI explanation for this prediction; feature influence is unknown.".into()),
            Some(method) => {
                out.push(format!("XAI method: {method}; units: {}", self.units.as_deref().unwrap_or("model score")));
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
            }
        }
        out.push("Caveats:".into());
        out.extend(self.caveats.iter().map(|c| format!("- {c}")));
        out.join("\n")
    }
    /// Every number the packet states: confidence, runner-up scores, class count, feature values and contributions (raw, unrounded),
    /// group sizes, and the digit runs inside packet feature names (`1883` in `network_ports_src__present_1883`, `6` in `..._dhcpv6`).
    pub fn numbers(&self) -> Vec<f64> {
        self.confidence.into_iter().chain(self.runner_ups.iter().map(|r| r.1)).chain([self.class_count as f64])
            .chain(self.features.iter().filter_map(|f| f.value)).chain(self.features.iter().map(|f| f.contribution))
            .chain(self.features.iter().filter(|f| f.group_size > 1).map(|f| f.group_size as f64))
            .chain(self.features.iter().flat_map(|f| f.name.split(|c: char| !c.is_ascii_digit()).filter_map(|d| d.parse::<f64>().ok()))).collect()
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

