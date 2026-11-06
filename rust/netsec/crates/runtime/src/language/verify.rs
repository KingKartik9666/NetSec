//! Pure per-sentence verifier for model-written text: every number, feature, direction and attack name must come from the evidence packet. No I/O.
use super::packet::{contains_words,digit_runs,Anchor,Lexicon,Packet};
use crate::application::display_label;
use std::collections::{BTreeMap,BTreeSet};

/// What one packet allows. `features` are normalized names, longest first; `Some(contribution)` marks a packet feature, `None` a catalog-only one.
/// `percents` (confidence and runner-up scores x100) are the only bases for `%` tokens. `class_names` are the word sequences of the predicted and
/// runner-up class ids/labels (also without the family word) whose digits are exempt from the number rule.
#[derive(Clone,Debug)]
pub struct Rules { numbers: Vec<f64>, percents: Vec<f64>, features: Vec<(String, Option<f64>)>, directional: bool,
    predicted: BTreeSet<String>, runner_ups: BTreeSet<String>, attacks: BTreeSet<String>, class_names: Vec<Vec<String>>,
    /// Redesign 3 Nov: the recorded counterfactual's new class terms (named only with a change cue) and its recorded score difference, against
    /// which a sentence about the score is read (a sentence about a value is read against that value's recorded from -> to change).
    counterfactual: BTreeSet<String>, score_change: Option<f64>,
    /// Only a counterfactual records what the model would answer after a change; for any other method such a claim is invented.
    class_change: bool,
    /// A counterfactual search that found no change: only a negated class-change statement ("no change was found that would ...") may be made.
    no_change_found: bool,
    /// Words of the catalog feature names, packet features and class names; `encoded` counts them as known words (`source port 1883 present`).
    vocabulary: BTreeSet<String>,
    /// Redesign round 1 review (3 Nov 2026), all stricter. The recorded counterfactual `to` values, allowed only in a `to Y` clause (finding 3).
    targets: Vec<f64>,
    /// Per normalized packet feature name: the numbers recorded for it (value, contribution / importance / effect or the counterfactual's
    /// score difference, group size, name digits) and its recorded counterfactual `from`/`to` values. A number after a named feature must be one
    /// of them (pass-rate round 1, review 3 finding 6: every method, with the number's role read from its wording).
    facts: BTreeMap<String, Fact>,
    /// The selected method (`shap` for every SHAP variant), for the method's own faithfulness rules.
    method: String,
    /// The recorded anchor search (its statistics pair with their own words; an unsupported rule is never called supported).
    anchor: Option<Anchor>,
    /// Words of the recorded counterfactual class's names, which a change cue may govern (finding 1), and its names as plain words
    /// (`normal network activity`), which only a change cue may introduce even when they hold no attack term (round 2 real-model run).
    counterfactual_words: BTreeSet<String>, counterfactual_names: Vec<String>,
    /// The predicted class's names as plain words (`normal network activity`, `benign`).
    predicted_names: Vec<String>,
    /// The recorded units as plain words (`brier loss increase`): a unit name is not direction wording.
    units: String,
    /// Redesign round 3 (review 2, finding 4): the LIME fit (R squared), accepted only next to its own words (fit, R squared, matched).
    fidelity: Option<f64>,
    /// Review 2 finding 6: normalized names of counterfactual changes without a recorded from/to (linked groups): no direction about them.
    groups: BTreeSet<String> }
/// The recorded facts of one packet feature (see `Rules::facts`).
#[derive(Clone,Debug)]
struct Fact { own: Vec<f64>, value: Option<f64>, contribution: f64, group: bool, from: Option<f64>, to: Option<f64> }
/// Owner directive 1 Nov 2026 added boost/drove/favour/pushed up and pushed down/pulled/weaken; bare `up`/`down` also catch split
/// phrasing such as "pushed the score down". Mixed wording ("pulled it toward") fails as both.
const INCREASE: &[&str] = &["increase", "raise", "push toward", "pushed toward", "support", "higher", "toward", "towards", "boost", "drove", "drive", "favour", "favor",
    "pushed up", "up"];
const DECREASE: &[&str] = &["decrease", "lower", "reduce", "against", "away from", "away", "pushed down", "down", "pull", "weaken"];
/// The plain words of the clause (between `,` `;` `:` or the sentence ends, outside masked names) that holds a mention, units removed.
fn clause_around(text: &str, masked: &[bool], m: &Mention) -> String {
    let b = text.as_bytes();
    let stop = |i: usize| !masked[i] && matches!(b[i], b',' | b';' | b':');
    let start = (0..m.start).rev().find(|&i| stop(i)).map_or(0, |i| i + 1);
    let end = (m.end..b.len()).find(|&i| stop(i)).unwrap_or(b.len());
    plain(&text[start..end], &masked[start..end])
}
/// Movement words that state a direction of change (review 2 finding 6).
const GROUP_MOVES: [&str; 13] = ["rise", "rose", "fall", "fell", "drop", "grew", "grow", "shrank", "shrink", "larger", "smaller", "bigger", "climb"];
/// Pass-rate round 1 (review 3 finding 1, stricter): every movement word, read against a single counterfactual change's recorded sign (to - from).
const MOVES_UP: [&str; 24] = ["rise", "rose", "risen", "grew", "grow", "grown", "climb", "jump", "double", "triple", "larger", "bigger", "stretch", "expand",
    "gain", "went up", "go up", "goes up", "spike", "spiked", "surge", "soar", "swell", "lengthen"];
const MOVES_DOWN: [&str; 21] = ["fall", "fell", "fallen", "drop", "shrank", "shrink", "shrunk", "halve", "smaller", "decline", "sank", "sink", "sunk", "dip", "cut",
    "shorten", "went down", "go down", "goes down", "plunge", "plummet"];
/// No XAI method records a ratio, so a multiplier is an invented number (review 3 finding 1: "doubled", "a tenfold rise").
const MULTIPLIERS: [&str; 21] = ["double", "twice", "triple", "quadruple", "halve", "half", "twofold", "threefold", "fourfold", "fivefold", "tenfold", "hundredfold",
    "times as", "times more", "times higher", "times larger", "times bigger", "times greater", "times smaller", "times lower", "times less"];
/// A negator flips direction wording ("did not raise"); `n't` reads as `n t` in the plain text.
const NEGATORS: [&str; 7] = ["not", "no", "never", "neither", "nor", "without", "cannot"];
/// Hedges include model attribution with the model as subject ("the model predicted", "classified as"); certainty words stay banned.
const HEDGES: [&str; 15] = ["consistent with", "looks like", "look like", "may indicate", "might indicate", "could indicate", "suggest", "typical of",
    "model predicted", "model predicts", "model classified", "model classifies", "classified as", "labelled as", "labeled as"];
/// A runner-up class may be named only when the sentence ranks it below the prediction.
const CUES: [&str; 5] = ["less likely", "runner up", "also considered", "second most likely", "lower score"];
/// The recorded counterfactual class may be named only as the answer the model would give after the recorded change: conditional, model-side
/// cues only ("instead", "changed to" or "would become" alone can describe the traffic itself and do not count).
const CHANGE_CUES: [&str; 10] = ["would give", "would change to", "would predict", "would answer", "would switch to", "would flip to", "would classify",
    "would label", "new class", "class after the change"];
/// A counterfactual sentence with these words is about the score, so its direction wording is read against the recorded score difference.
const SCORE_WORDS: [&str; 7] = ["score", "probability", "likely", "likelihood", "confidence", "prediction", "predicted class"];
/// Claims that the model would give another class or answer. Redesign round 1 real-model run (3 Nov): shown for LIME, permutation and anchors.
/// An answer change stated as a fact (round 2 final run: "The predicted class changed due to the combined shift").
const CHANGED: [&str; 12] = ["class changed", "prediction changed", "answer changed", "class switched", "prediction switched", "class shifted",
    "shifted the prediction", "shifted the model s prediction", "changed the prediction", "changed the model s prediction", "led the model to predict",
    "caused the model to predict"];
const CLASS_CHANGE: [&str; 15] = ["would predict", "would have predicted", "would give", "would have given", "would shift", "would switch", "would flip",
    "would pick", "would choose", "would classify", "would label", "different class", "another class", "different answer", "different prediction"];
/// Words a change cue may skip before the class it governs ("the new class would be <class>").
const FILLER: [&str; 12] = ["be", "is", "would", "a", "an", "the", "then", "to", "class", "label", "answer", "now"];
/// A subject and copula about the traffic or host itself ("this host is", "the traffic was doing"): no counterfactual class may be named with it.
const SUBJECTS: [&str; 16] = ["traffic", "host", "hosts", "flow", "flows", "device", "devices", "connection", "connections", "packets", "machine", "network",
    "sender", "source", "this", "it"];
const COPULAS: [&str; 17] = ["is", "was", "are", "were", "be", "been", "being", "doing", "does", "did", "performs", "performed", "performing", "conducting",
    "launching", "sending", "part"];
/// Review finding 4: actor and evasion framing (who could change the traffic, how to get a different answer) is never shown.
const ACTORS: &[&str] = &["attacker", "adversary", "adversaries", "hacker", "intruder", "threat actor", "malicious actor", "slip past", "get past", "go unnoticed", "slip",
    "under the radar", "avoid being flagged", "avoid being detected", "avoid flagging", "fool", "trick the model", "to make the model", "so the model would",
    "so that the model would", "game the model", "if you", "you could", "one could", "someone could", "somebody could"];
/// Review finding 3: words that tie a number to one recorded anchor statistic (the nearest such word before the number decides; "" = none).
/// Redesign round 3 security review: `value`, `values`, `contribution`, `importance` and `effect` block too, so a precision after "the model
/// predicted ... had value" is never read as a feature's value.
/// Pass-rate round 1 (review 3 finding 2, stricter): only a statistic's own noun or recorded verb governs it (precision/kept, lower bound,
/// coverage/covered, tolerance, `within ... of this observation`, rows tested); `predicted`, `gave`, `matched`, `applied` and `met` no longer
/// tie a number to a statistic and block one instead, so a statistic is never read as what the model predicted or gave.
const ANCHOR_WORDS: [(&str, &str); 40] = [("precision", "precision"), ("precise", "precision"), ("kept", "precision"), ("keep", "precision"), ("keeps", "precision"),
    ("keeping", "precision"), ("bound", "bound"), ("coverage", "coverage"), ("covered", "coverage"), ("covers", "coverage"), ("cover", "coverage"),
    ("covering", "coverage"), ("tolerance", "tolerance"), ("within", "within"), ("tested", "support"),
    ("gave", ""), ("give", ""), ("gives", ""), ("predicted", ""), ("predict", ""), ("predicts", ""), ("applied", "applied"), ("applies", "applied"),
    ("apply", "applied"), ("met", "applied"), ("matched", "applied"), ("matches", "applied"), ("confidence", ""), ("classes", ""), ("score", ""), ("group", ""), ("runner", ""), ("probability", ""), ("linked", ""),
    ("value", ""), ("values", ""), ("contribution", ""), ("importance", ""), ("effect", ""), ("chance", "")];
/// Words that read a statistic or the LIME fit as the model's confidence, a probability or a verdict: with any of them in the sentence neither
/// is accepted (review 3 finding 2). `model was` / `model is` state something about the model itself.
const STAT_BLOCKERS: [&str; 27] = ["sure", "unsure", "likely", "unlikely", "likelihood", "confidence", "confident", "probability", "probable", "probably", "chance",
    "chances", "odds", "truth", "true", "trust", "trusted", "relied", "rely", "reliable", "certainty", "correct", "right", "wrong", "model was", "model is", "accurate"];
fn stat_blocked(t: &[u8]) -> bool {
    let text = String::from_utf8_lossy(t);
    let plain = plain(&text, &vec![false; text.len()]);
    STAT_BLOCKERS.iter().any(|w| contains_words(&plain, w))
}
/// A number followed by one of these is a count of reference rows (the anchor's support).
const COUNTED: [&str; 6] = ["matching", "reference", "rows", "row", "samples", "sample"];
/// No supported anchor: these claims fail unless negated, and so does causal or conditional framing of the predicted class.
const SUPPORT_CLAIMS: [&str; 8] = ["supported rule", "reliable rule", "supported anchor", "strong rule", "good rule", "valid rule", "solid rule", "rule is supported"];
const CONDITIONAL: [&str; 13] = ["based on", "when", "whenever", "under the condition", "under conditions", "under a condition", "condition that", "because",
    "due to", "as long as", "if", "thanks to", "driven by"];
const STAT_VERBS: [&str; 19] = ["keep", "kept", "keeps", "cover", "covered", "covers", "hold", "held", "holds", "give", "gave", "gives", "predict", "predicted",
    "precision", "coverage", "apply", "applied", "applies"];
/// Review finding 6: ALE records an effect at this value; narrating a change from one value to another is invented.
const ALE_CHANGES: [&str; 9] = ["to a lower", "to a higher", "lower value", "higher value", "larger value", "smaller value", "from its baseline", "from the baseline",
    "from baseline"];
const CHANGE_NOUNS: [&str; 13] = ["increase", "increases", "decrease", "decreases", "rise", "rises", "drop", "drops", "change", "changes", "reduction", "shift", "shifts"];
const CHANGE_VERBS: [&str; 12] = ["raising", "lowering", "increasing", "decreasing", "reducing", "changing", "raise", "lower", "increase", "decrease", "reduce", "change"];
/// Review finding 5: claims that a feature matters, which need a positive recorded permutation importance, unless qualified right before.
const IMPORTANCE_CLAIMS: [&str; 33] = ["influence", "influences", "influenced", "influencing", "influential", "important", "significant", "significantly", "strong",
    "strongly", "rely", "relies", "relied", "relying", "depend", "depends", "depended", "depending", "dependence", "dependent", "matter", "matters", "mattered",
    "impact", "impacts", "impactful", "key", "largest", "highest", "biggest", "greatest", "crucial", "major"];
const QUALIFIERS: [&str; 14] = ["less", "least", "limited", "minor", "little", "low", "no", "not", "none", "zero", "without", "never", "nor", "neither"];
/// Permutation importance is global: wording about this prediction or decision, or a cause of it, fails.
const LOCAL: &[&str] = &["mainly from", "mostly from", "largely from", "primarily from", "chiefly from", "this prediction", "the prediction", "model s prediction", "prediction of", "overall prediction", "final prediction", "this decision", "the decision", "overall decision",
    "final decision", "contributed to", "contribute to", "contributes to", "contributing to", "based on", "because", "due to", "drove", "driven", "shaped",
    "led to", "leads to", "responsible for", "this result was", "thanks to", "as a result of", "explains why"];
/// ALE and anchors describe the model's response, not a cause of this prediction or of its confidence.
const CAUSAL: [&str; 11] = ["because", "due to", "based on", "thanks to", "as a result of", "driven by", "led to", "leads to", "explains why", "combined influence",
    "combined effect"];
/// A part-1 sentence that names none of these, nor the predicted class, a packet feature or a number, only defines the method (finding 8).
const RESULT_STATES: [&str; 8] = ["no supported rule", "no reliable rule", "no feasible change", "no realistic change", "supported rule", "no such change",
    "no change was found", "found no"];
/// Redesign 3 Nov: the explainer translates the XAI; it never judges whether the model or its answer is right (a new, stricter rule).
/// Pass-rate round 1 (review 3 finding 5, stricter): also the forms the review found shown ("made the right call", "spot on", "confirms",
/// "makes sense", "no reason to doubt", "can be relied on", "good job", ...).
const JUDGEMENT: &[&str] = &["correct", "correctly", "incorrect", "incorrectly", "wrong", "wrongly", "accurate", "accurately", "inaccurate", "accuracy",
    "mistake", "mistaken", "misclassified", "misclassification", "false positive", "false negative", "false alarm", "trust", "trustworthy", "untrustworthy",
    "reliable", "unreliable", "reliability", "got it right", "is right", "was right", "be right", "are right",
    "right call", "got this right", "got this one right", "got that right", "spot on", "probably right", "likely right", "right about", "right answer", "right class",
    "right prediction", "confirm", "validate", "valid", "invalid", "validity", "makes sense", "make sense", "made sense", "dependable", "is sound", "was sound",
    "are sound", "doubt", "doubtful", "relied upon", "can be relied", "be relied", "can rely", "good job", "did well", "does well", "do well", "well done", "justified",
    "justify", "justifies", "plausible", "implausible", "back up", "backs up", "backed up", "evidence supports", "evidence support", "evidence supported",
    "xai supports", "explanation supports", "supports the model s answer", "supports the answer", "supports this answer", "the truth", "good prediction",
    "bad prediction", "good answer", "sensible", "reasonable", "unreasonable"];
/// Substring patterns: chat-template tokens, code fences, URLs.
const MARKERS: [&str; 11] = ["<|", "|>", "```", "[inst]", "[/inst]", "<<sys>>", "</s>", "<think>", "</think>", "://", "www."];
/// Certainty, action and evasion words, whole words with inflections (`certain` never matches inside `uncertain`).
const BANNED: &[&str] = &["definitely", "certainly", "certain", "confirmed", "proves", "proven", "guarantee", "guaranteed", "undoubtedly", "unquestionably", "indisputably",
    "obviously", "clearly", "perfect", "perfectly", "flawless", "flawlessly", "conclusive", "conclusively", "without doubt", "without a doubt", "no doubt", "beyond doubt", "for sure", "sure",
    "block", "firewall", "execute", "shut down", "shuts down", "shutting down", "shutdown", "recommend", "you should", "quarantine", "disconnect",
    "evade", "evasion", "bypass", "avoid detection", "undetected", "disguise"];
/// Exact words only (`runner`, `urban` and `banner` stay allowed).
const BANNED_EXACT: [&str; 7] = ["ban", "bans", "banned", "banning", "run", "runs", "running"];

fn normalize(name: &str) -> String { name.chars().filter(|c| !matches!(c, '_' | '-' | '/') && !c.is_whitespace()).flat_map(char::to_lowercase).collect() }
fn words(text: &str) -> Vec<String> { text.to_lowercase().split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()).map(String::from).collect() }
fn numeric(word: &str) -> bool { word.bytes().all(|b| b.is_ascii_digit()) }
/// A recorded group name and, when it ends in a parenthetical (`log_data-types (linked measurements)`), the name without it (`log_data-types`).
fn variants(name: &str) -> Vec<String> {
    let mut out = vec![normalize(name)];
    if let Some((stem, _)) = name.strip_suffix(')').and_then(|n| n.rsplit_once(" (")) { out.push(normalize(stem)); }
    out
}
pub fn rules(packet: &Packet, all_features: &[String], lexicon: &Lexicon) -> Rules {
    let mut names = all_features.iter().map(|f| (normalize(f), None)).collect::<BTreeMap<_, _>>();
    // A counterfactual change is read against its recorded value change (to - from); a group without single values against the score difference.
    let change = |name: &str| packet.counterfactual.as_ref().and_then(|c| c.changes.iter().find(|ch| ch.name == name));
    let delta = |name: &str| change(name).and_then(|ch| Some(ch.to? - ch.from?));
    names.extend(packet.features.iter().flat_map(|f| variants(&f.name).into_iter().map(|n| (n, Some(delta(&f.name).unwrap_or(f.contribution))))));
    // The anchor's recorded condition group is a packet feature (anchors are not directional).
    names.extend(packet.anchor.iter().filter_map(|a| a.condition.as_deref()).flat_map(variants).map(|n| (n, Some(0.))));
    let mut features = names.into_iter().filter(|(n, _)| n.len() >= 6).collect::<Vec<_>>();
    features.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));
    let terms = |id: &str| lexicon.by_class.get(id).cloned().unwrap_or_default();
    let new_class = packet.counterfactual.as_ref().and_then(|c| c.new_class.clone());
    let labels = [packet.class_id.clone(), packet.class_label.clone()].into_iter().chain(packet.runner_ups.iter().map(|(id, _)| id.clone()).chain(new_class.clone()).flat_map(|id| [display_label(&id), id]));
    let mut class_names = labels.map(|l| words(&l)).filter(|w| w.iter().any(|w| numeric(w)))
        .flat_map(|w| { let tail = w[1..].iter().any(|t| t != "port" && !numeric(t)).then(|| w[1..].to_vec()); [Some(w), tail] }).flatten().collect::<Vec<_>>();
    class_names.sort(); class_names.dedup();
    let method = packet.method.as_deref().map_or(String::new(), |m| if m == "shap" || m.ends_with("_shap") { "shap".into() } else { m.to_string() });
    let facts = packet.features.iter().flat_map(|f| {
        let own = f.value.into_iter().chain([f.contribution]).chain((f.group_size > 1).then_some(f.group_size as f64)).chain(digit_runs(&f.name)).collect::<Vec<_>>();
        let (from, to) = change(&f.name).map_or((None, None), |ch| (ch.from, ch.to));
        let fact = Fact { own, value: f.value, contribution: f.contribution, group: f.group_size > 1, from, to };
        variants(&f.name).into_iter().map(move |n| (n, fact.clone()))
    }).collect();
    let counterfactual_words = new_class.iter().flat_map(|id| terms(id).into_iter().chain([display_label(id), id.clone()])).flat_map(|t| words(&t)).collect();
    let mut predicted_names = [packet.class_label.clone(), display_label(&packet.class_id), packet.class_id.clone()].iter().map(|n| words(n).join(" "))
        .filter(|n| !n.is_empty()).collect::<Vec<_>>();
    predicted_names.dedup();
    // Review findings 5 and 6: permutation importance and ALE effects are signed, so their direction wording is checked too (stricter).
    let directional = packet.directional || matches!(method.as_str(), "permutation" | "ale");
    let groups = packet.counterfactual.iter().flat_map(|c| c.changes.iter().filter(|ch| ch.from.is_none() || ch.to.is_none())).flat_map(|ch| variants(&ch.name)).collect();
    Rules { fidelity: packet.fidelity, groups, targets: packet.change_targets(), facts, anchor: packet.anchor.clone(),
        counterfactual_names: new_class.iter().flat_map(|id| [display_label(id), id.clone()]).map(|n| words(&n).join(" ")).filter(|n| !n.is_empty()).collect(),
        counterfactual_words, predicted_names, units: words(&packet.units_text()).join(" "), method, numbers: packet.numbers(), percents: packet.percents(),
        features, directional, predicted: terms(&packet.class_id), runner_ups: packet.runner_ups.iter().flat_map(|(id, _)| terms(id)).collect(),
        attacks: lexicon.all.clone(), class_names, counterfactual: new_class.as_deref().map(terms).unwrap_or_default(),
        score_change: packet.counterfactual.as_ref().and_then(|c| c.score_change), class_change: packet.counterfactual.as_ref().is_some_and(|c| c.found),
        no_change_found: packet.counterfactual.as_ref().is_some_and(|c| !c.found),
        vocabulary: all_features.iter().chain(packet.features.iter().map(|f| &f.name)).chain(packet.anchor.iter().filter_map(|a| a.condition.as_ref()))
            .chain(lexicon.by_class.keys()).flat_map(|n| words(n)).filter(|w| !numeric(w)).collect() }
}

/// Lowercased ASCII view. Typographic quotes, dashes, the minus sign and no-break spaces map to ASCII; any other non-ASCII character
/// (zero-width, bidi, homoglyph letters, fullwidth digits) or control character fails as `banned`.
fn ascii(sentence: &str) -> Result<String, String> {
    sentence.chars().map(|c| match c {
        '\u{2018}' | '\u{2019}' => Ok('\''), '\u{201c}' | '\u{201d}' => Ok('"'), '\u{2013}' | '\u{2014}' | '\u{2212}' => Ok('-'), '\u{2026}' => Ok('.'),
        '\u{a0}' | '\u{202f}' | '\t' | '\n' | '\r' => Ok(' '),
        c if c.is_ascii() && !c.is_ascii_control() => Ok(c.to_ascii_lowercase()),
        c => Err(format!("banned: character U+{:04X}", c as u32)),
    }).collect()
}
/// Alphanumeric words separated by single spaces; masked spans become one `~`, so no phrase can match across them.
fn plain(text: &str, masked: &[bool]) -> String {
    let mut out = String::with_capacity(text.len());
    for (b, m) in text.bytes().zip(masked) {
        let c = if *m { '~' } else if b.is_ascii_alphanumeric() { b as char } else { ' ' };
        if (c == ' ' || c == '~') && out.ends_with(c) { continue; }
        out.push(c);
    }
    out
}
fn ip_literal(text: &str) -> bool {
    let runs = |keep: fn(char) -> bool| text.split(move |c: char| !keep(c)).map(String::from).collect::<Vec<_>>();
    runs(|c| c.is_ascii_digit() || c == '.').iter().any(|r| { let parts = r.trim_matches('.').split('.').collect::<Vec<_>>(); parts.len() >= 4 && parts.iter().all(|p| (1..=3).contains(&p.len())) })
        || runs(|c| c.is_ascii_hexdigit() || c == ':').iter().any(|r| r.bytes().any(|b| b.is_ascii_hexdigit()) && (r.contains("::") || r.matches(':').count() >= 3))
}
/// A bare hostname (`netsec-update.com`): two or more dot-separated labels ending in a letter TLD of at least two letters.
/// Decimals (`0.42`), `e.g.`, `i.e.` and class ids (no dots) are not hostnames; `check` runs it with feature names blanked (`twamp.control`).
fn hostname(text: &str) -> bool {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '-' || c == '.')).any(|run| {
        let labels = run.trim_matches('.').split('.').collect::<Vec<_>>();
        labels.len() >= 2 && labels.iter().all(|l| l.bytes().any(|b| b.is_ascii_alphanumeric())) && labels.last().is_some_and(|t| t.len() >= 2 && t.bytes().all(|b| b.is_ascii_alphabetic()))
    })
}
fn banned(text: &str) -> Result<(), String> {
    let plain = plain(text, &vec![false; text.len()]);
    let words = plain.split(' ').collect::<BTreeSet<_>>();
    let hit = MARKERS.iter().find(|m| text.contains(*m)).or_else(|| BANNED.iter().find(|w| contains_words(&plain, w))).or_else(|| ACTORS.iter().find(|w| contains_words(&plain, w)))
        .or_else(|| BANNED_EXACT.iter().find(|w| words.contains(*w))).copied().or_else(|| ip_literal(text).then_some("IP address"));
    hit.map_or(Ok(()), |h| Err(format!("banned: {h}")))
}
/// Longest-first match of normalized feature names in the normalized sentence. Every match is masked (its digits are not numbers, attack terms
/// inside it do not count, a shorter name inside it is not matched again); a catalog name outside the packet fails. Returns packet contributions.
fn features(text: &str, rules: &Rules, masked: &mut [bool]) -> Result<Vec<Mention>, String> {
    let (mut norm, pos): (Vec<u8>, Vec<usize>) = text.bytes().enumerate().filter(|(_, b)| !matches!(b, b'_' | b'-' | b'/') && !b.is_ascii_whitespace()).map(|(i, b)| (b, i)).unzip();
    let mut mentioned = Vec::new();
    for (name, contribution) in &rules.features {
        let bytes = name.as_bytes();
        while let Some(at) = norm.windows(bytes.len()).position(|w| w == bytes) {
            let Some(c) = contribution else { return Err(format!("feature: {name}")) };
            let (start, end) = (pos[at], pos[at + bytes.len() - 1] + 1);
            norm[at..at + bytes.len()].fill(0); masked[start..end].fill(true); mentioned.push(Mention { start, end, name: name.clone(), value: *c });
        }
    }
    mentioned.sort_by_key(|m| m.start);
    Ok(mentioned)
}
/// One packet feature named in a sentence: its byte span, normalized name and recorded contribution (or change, importance, effect).
struct Mention { start: usize, end: usize, name: String, value: f64 }
/// Marks the spans of allowed class names (`ddos syn flood port 1883`) so their digits are not number tokens. Redesign round 3 (review 2,
/// finding 3): the pass-rate round's `on port` exemption is reverted to the base logic; it is a proposal in owner-decisions.md.
fn class_spans(text: &str, masked: &[bool], rules: &Rules, exempt: &mut [bool]) {
    let b = text.as_bytes();
    let (mut spans, mut i) = (Vec::new(), 0);
    while i < b.len() {
        if !b[i].is_ascii_alphanumeric() || masked[i] { i += 1; continue; }
        let start = i; while i < b.len() && b[i].is_ascii_alphanumeric() && !masked[i] { i += 1; }
        spans.push((start, i));
    }
    for name in &rules.class_names {
        for w in spans.windows(name.len()).filter(|w| w.iter().zip(name).all(|((s, e), n)| text[*s..*e] == *n)) { exempt[w[0].0..w[w.len() - 1].1].fill(true); }
    }
}
/// Tokens `-?\d*\.?\d+%?` (also `1,234` and `93 percent`). With `d` decimals a token matches a packet number `e` when `|x - e| <= 0.5*10^-d`;
/// an unsigned token is compared with `|e|`, a `%` token only with the confidence and runner-up scores x100. Bare integers 1-5 always pass.
/// The packet prints plain decimals only, so an exponent token (`4.187e-1`, `1e309`) is never a copied packet number and always fails.
/// Redesign round 3 (review 2, findings 4 and 5): an anchor statistic or the LIME fit is accepted only when its own word stands among the
/// eight words before it (the nearest statistic word decides; a blocker such as `confidence` or `value` means none), and a counterfactual
/// `to` value only as the target of the nearest packet feature before it whose recorded `to` it is, with no other quantity in between.
fn numbers(t: &[u8], rules: &Rules, mentions: &[Mention]) -> Result<(), String> {
    let digit = |j: usize| t.get(j).is_some_and(u8::is_ascii_digit);
    let blocked = stat_blocked(t);
    let mut i = 0;
    while i < t.len() {
        if !(digit(i) || (t[i] == b'.' && digit(i + 1))) { i += 1; continue; }
        let negative = i > 0 && t[i - 1] == b'-' && (i < 2 || !t[i - 2].is_ascii_alphanumeric());
        let (mut j, mut raw, mut decimals, mut grouped) = (i, String::new(), 0, false);
        while digit(j) || (t.get(j) == Some(&b',') && !raw.is_empty() && (1..4).all(|k| digit(j + k)) && !digit(j + 4)) {
            if t[j] == b',' { grouped = true; } else { raw.push(t[j] as char); }
            j += 1;
        }
        if t.get(j) == Some(&b'.') && digit(j + 1) { raw.push('.'); j += 1; while digit(j) { raw.push(t[j] as char); j += 1; decimals += 1; } }
        let sign = matches!(t.get(j + 1), Some(b'-' | b'+')) as usize;
        if t.get(j) == Some(&b'e') && digit(j + 1 + sign) { let start = j; j += 1 + sign; while digit(j) { j += 1; } return Err(format!("number: {}{raw}{}", if negative { "-" } else { "" }, String::from_utf8_lossy(&t[start..j]))); }
        let rest = &t[j..]; let rest = rest.strip_prefix(b" ").unwrap_or(rest);
        let percent = rest.starts_with(b"%") || (rest.starts_with(b"percent") && !rest.get(7).is_some_and(u8::is_ascii_alphanumeric));
        let x = format!("0{raw}").parse::<f64>().unwrap_or(f64::NAN) * if negative { -1. } else { 1. };
        let small = !negative && !percent && !grouped && decimals == 0 && (1.0..=5.0).contains(&x);
        let tolerance = |e: f64| 0.5 * 10f64.powi(-decimals) + 1e-9 * e.abs().max(1.);
        let bases = if percent { &rules.percents } else { &rules.numbers };
        let near = |e: &f64| (x - if negative { *e } else { e.abs() }).abs() <= tolerance(*e);
        let start = if negative { i - 1 } else { i };
        // A recorded counterfactual `to` value is a number only as the target of its own feature's change (`log_x changed from 1 to 2`).
        let target = !percent && rules.targets.iter().any(near) && target_of(t, start, mentions, rules).is_some_and(|to| near(&to));
        let then = String::from_utf8_lossy(&t[j..]).split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()).take(4)
            .any(|w| matches!(w, "confidence" | "score" | "probability" | "contribution"));
        let stat = !then && !blocked && rules.anchor.as_ref().is_some_and(|a| {
            let stat = anchor_stat(t, start, j, percent, STAT_WINDOW);
            let scale = if !percent { 1. } else if matches!(stat, "tolerance" | "support") { f64::NAN } else { 100. };
            anchor_values(a, stat).iter().any(|e| near(&(e * scale)))
        });
        let fit = !percent && !then && !blocked && rules.fidelity.is_some_and(|f| near(&f) && fit_word(t, start));
        let ok = small || (x.is_finite() && (bases.iter().any(near) || target || stat || fit));
        if !ok { return Err(format!("number: {}{raw}{}", if negative { "-" } else { "" }, if percent { "%" } else { "" })); }
        i = j;
    }
    Ok(())
}
/// How many words before a number its statistic word may stand (findings 4 and 5).
const STAT_WINDOW: usize = 8;
/// Words that tie a number to the LIME fit, and words that tie it to something else (the nearest one decides).
const FIT_WORDS: [(&str, bool); 21] = [("fit", true), ("fitted", true), ("squared", true), ("r2", true), ("matched", true), ("match", true), ("matches", true),
    ("fidelity", true), ("closely", true), ("confidence", false), ("score", false), ("gap", false), ("value", false), ("values", false), ("contribution", false),
    ("probability", false), ("weight", false), ("had", false), ("importance", false), ("effect", false), ("precision", false)];
/// Quantities other than a feature's value: a counterfactual `to` value after one of these is not that feature's recorded target.
const QUANTITIES: [&str; 14] = ["confidence", "score", "probability", "likelihood", "output", "margin", "loss", "prediction", "class", "answer", "window", "seconds",
    "rate", "precision"];
/// The words after byte `at`, in order, at most four.
fn words_after(t: &[u8], at: usize) -> Vec<String> {
    String::from_utf8_lossy(&t[at.min(t.len())..]).split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()).take(4).map(String::from).collect()
}
/// The words before byte `at`, nearest first, at most `n`.
fn words_before(t: &[u8], at: usize, n: usize) -> Vec<String> {
    String::from_utf8_lossy(&t[..at.min(t.len())]).split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()).rev().take(n).map(String::from).collect()
}
/// The anchor statistic a number at `start` is tied to: `support` for a count of reference rows, else the nearest statistic word before it
/// within `window` words ("" = none, or a blocker such as `confidence`).
fn anchor_stat(t: &[u8], start: usize, end: usize, percent: bool, window: usize) -> &'static str {
    let after = String::from_utf8_lossy(&t[end.min(t.len())..]).split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()).take(4).map(String::from).collect::<Vec<_>>();
    let after = after.iter().map(String::as_str).collect::<Vec<_>>();
    // Review 3 finding 2: a count is the recorded support only as matching reference rows (tested); `15 rows of traffic`, `15 samples` are not.
    if !percent && COUNTED.contains(&after.first().copied().unwrap_or("")) {
        let rows = matches!(after.as_slice(), ["matching", ..] | ["reference", "rows" | "row", ..] | ["rows" | "row", "tested", ..]);
        return if rows { "support" } else { "" };
    }
    // `the predicted class` is a noun phrase, not the model predicting a statistic ("kept the predicted class in 6.67% of those cases").
    let before = words_before(t, start, window);
    let nearest = before.iter().enumerate().filter(|(i, w)| !(*w == "predicted" && *i > 0 && before[i - 1] == "class"))
        .find_map(|(_, w)| ANCHOR_WORDS.iter().find(|(k, _)| *k == w).map(|(_, s)| *s));
    // The recorded noun phrase after the number names the statistic: `in 0.06667 of the 15 matching reference rows` (precision), `0.8281 of the
    // reference data` (coverage), whatever verb comes before it.
    if matches!(after.as_slice(), ["of", "the", "matching", ..] | ["of", "matching", ..]) || (after.len() >= 4 && after[..2] == ["of", "the"] && numeric(after[2]) && after[3] == "matching") {
        return "precision";
    }
    match nearest.unwrap_or("") {
        // `within 0.15 of this observation's values` is the recorded tolerance; `within 0.15 of the runner-up` is not.
        "within" => if matches!(after.as_slice(), ["of", "this" | "the" | "its", "observation" | "observed" | "recorded" | "values", ..]) { "tolerance" } else { "" },
        // `applied to 82.81% of the reference rows` is the recorded coverage; `the rule matched, so the model was 82.81% sure` is not.
        // Pass-rate round 1 adversarial pass: only with the rule or condition as subject and the reference rows or data as object.
        "applied" => {
            let subject = words_before(t, t.len(), usize::MAX).iter().any(|w| matches!(w.as_str(), "rule" | "condition" | "anchor"));
            if subject && matches!(after.as_slice(), ["of", "the", "reference", "rows" | "row" | "data", ..] | ["of", "reference", "rows" | "row" | "data", ..]) { "coverage" } else { "" }
        }
        stat => stat,
    }
}
fn anchor_values(a: &Anchor, stat: &str) -> Vec<f64> {
    match stat { "precision" => vec![a.precision, a.lower_bound], "bound" => vec![a.lower_bound], "coverage" => vec![a.coverage], "tolerance" => vec![a.tolerance],
        "support" => vec![a.support], _ => vec![] }.into_iter().flatten().collect()
}
/// Pass-rate round 1 (review 3 finding 2, stricter): every fit word counts only in a sentence whose subject is LIME's local model (`lime`,
/// `local`, `simple`, `surrogate`), never `a fit of 0.997` about the result or the model's answer.
fn fit_word(t: &[u8], start: usize) -> bool {
    let local = words_before(t, t.len(), usize::MAX).iter().any(|w| matches!(w.as_str(), "lime" | "local" | "simple" | "surrogate"));
    local && words_before(t, start, STAT_WINDOW).iter().find_map(|w| FIT_WORDS.iter().find(|(k, _)| k == w).map(|(_, fit)| *fit)) == Some(true)
}
/// The recorded `to` value of the nearest packet feature named before a number that follows `to`, when only change wording stands between.
fn target_of(t: &[u8], start: usize, mentions: &[Mention], rules: &Rules) -> Option<f64> {
    if prev_word(t, start) != "to" { return None; }
    let m = mentions.iter().rev().find(|m| m.end <= start)?;
    let between = String::from_utf8_lossy(&t[m.end..start]).to_string();
    let words = between.split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()).collect::<Vec<_>>();
    if words.len() > STAT_WINDOW || words.iter().any(|w| QUANTITIES.contains(w)) { return None; }
    rules.facts.get(&m.name).and_then(|f| f.to)
}
const UNITS: [&str; 20] = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
const TENS: [&str; 8] = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
const SCALES: [(&str, f64); 4] = [("hundred", 100.), ("thousand", 1e3), ("million", 1e6), ("billion", 1e9)];
/// Spelled numbers (`seventy-seven`, `ninety-three percent`, `a hundred`) follow the number rule at whole-number precision; a lone `one`..`five` passes.
fn spelled(plain: &str, rules: &Rules) -> Result<(), String> {
    let words = plain.split(' ').collect::<Vec<_>>();
    let value = |w: &str| UNITS.iter().position(|u| *u == w).map(|v| (v as f64, None)).or_else(|| TENS.iter().position(|t| *t == w).map(|v| (v as f64 * 10. + 20., None)))
        .or_else(|| SCALES.iter().find(|s| s.0 == w).map(|s| (0., Some(s.1))));
    let mut i = 0;
    while i < words.len() {
        let (start, mut total, mut current) = (i, 0., 0.);
        while let Some((v, scale)) = words.get(i).and_then(|w| value(w)) {
            match scale { Some(s) if s < 1e3 => current = f64::max(current, 1.) * s, Some(s) => { total += f64::max(current, 1.) * s; current = 0.; } None => current += v }
            i += 1;
        }
        if i == start { i += 1; continue; }
        let (x, percent) = (total + current, words.get(i) == Some(&"percent"));
        let small = !percent && i - start == 1 && (1.0..=5.0).contains(&x);
        let bases = if percent { &rules.percents } else { &rules.numbers };
        if !small && !bases.iter().any(|e| (x - e.abs()).abs() <= 0.5 + 1e-9 * e.abs().max(1.)) { return Err(format!("number: {}{}", words[start..i].join(" "), if percent { " percent" } else { "" })); }
    }
    Ok(())
}
fn negated(plain: &str) -> bool {
    let words = plain.split(' ').collect::<Vec<_>>();
    contains_words(plain, "rather than") || words.iter().enumerate().any(|(i, w)| NEGATORS.contains(w) || (*w == "t" && i > 0 && words[i - 1].ends_with('n')))
}
/// Skipped for permutation/ALE (`directional == false`) and when no packet feature is named. Negated wording fails: its sign cannot be read safely.
fn direction(plain: &str, mentioned: &[f64]) -> Result<(), String> {
    let up = INCREASE.iter().any(|w| contains_words(plain, w));
    let down = DECREASE.iter().any(|w| contains_words(plain, w));
    if up && down { return Err("direction: both increase and decrease wording".into()); }
    if (up || down) && negated(plain) { return Err("direction: negated direction wording".into()); }
    if (up && mentioned.iter().any(|c| *c <= 0.)) || (down && mentioned.iter().any(|c| *c >= 0.)) { return Err("direction: wording contradicts the contribution sign".into()); }
    Ok(())
}
/// Review 2 finding 1: the subject a `would ...` change cue needs right before it: the model or the model's answer, never the traffic, a host,
/// a device, a person or any other agent.
const MODEL_SUBJECTS: [&str; 17] = ["model", "model s answer", "model s prediction", "model s output", "model s decision", "model s label", "model s class",
    "model s choice", "predicted class", "predicted label", "the prediction", "its prediction", "the answer", "its answer", "the output", "the class", "the label"];
/// Security review probe ("a new class would be <class> for this host"): no counterfactual class with a host or device in the sentence.
const HOST_WORDS: [&str; 16] = ["host", "hosts", "device", "devices", "router", "routers", "machine", "machines", "sender", "senders", "server", "servers", "client",
    "clients", "node", "nodes"];
fn ends_with_phrase(words: &[&str], i: usize, phrase: &str) -> bool { let p = phrase.split(' ').collect::<Vec<_>>(); i >= p.len() && words[i - p.len()..i] == p[..] }
/// Whether the change cue at word `i` has a model-side subject: a `would ...` cue right after one of `MODEL_SUBJECTS`; a noun cue (`new class`,
/// `class after the change`) after `the`/`a` (optionally `the new`) or at the start, never after a possessive (`this traffic's new class`).
fn model_side(words: &[&str], i: usize, cue: &[&str]) -> bool {
    if cue[0] == "would" { return MODEL_SUBJECTS.iter().any(|s| ends_with_phrase(words, i, s)); }
    let at = if i > 0 && words[i - 1] == "new" { i - 1 } else { i };
    at == 0 || (words[at - 1] == "the" && !(at >= 2 && words[at - 2] == "s"))
}
/// The words a change cue governs: after the cue (skipping at most three filler words), the run of the recorded counterfactual class's words.
/// Review 2 finding 1 (stricter): only a cue with a model-side subject governs, and a noun cue only with `would` among the skipped words
/// (`the new class would be X`; `the new class is X` states the change as a fact).
fn governed(words: &[&str], rules: &Rules) -> Vec<bool> {
    let mut out = vec![false; words.len()];
    if rules.counterfactual_words.is_empty() || words.iter().any(|w| HOST_WORDS.contains(w)) { return out; }
    for cue in CHANGE_CUES {
        let cue = cue.split(' ').collect::<Vec<_>>();
        for i in 0..words.len() {
            if !words[i..].starts_with(&cue) || !model_side(words, i, &cue) { continue; }
            let (mut j, mut skipped, mut would) = (i + cue.len(), 0, cue[0] == "would");
            while j < words.len() && skipped < 3 && FILLER.contains(&words[j]) && !rules.counterfactual_words.contains(words[j]) { would |= words[j] == "would"; j += 1; skipped += 1; }
            if !would { continue; }
            while j < words.len() && rules.counterfactual_words.contains(words[j]) { out[j] = true; j += 1; }
        }
    }
    out
}
/// Every occurrence of `term` (last word inflected) lies inside governed words, and there is at least one.
fn all_governed(words: &[&str], term: &str, governed: &[bool]) -> bool {
    let term = term.split(' ').collect::<Vec<_>>();
    let n = term.len();
    let hits = (0..(words.len() + 1).saturating_sub(n)).filter(|&i| words[i..i + n - 1] == term[..n - 1] && contains_words(words[i + n - 1], term[n - 1])).collect::<Vec<_>>();
    !hits.is_empty() && hits.iter().all(|&i| governed[i..i + n].iter().all(|g| *g))
}
fn traffic_claim(words: &[&str]) -> bool {
    words.iter().enumerate().any(|(i, w)| SUBJECTS.contains(w) && words[i + 1..].iter().take(2).any(|c| COPULAS.contains(c)))
}
/// Review 2 finding 3 (stricter): a host, device or the traffic carrying out an action ("the host sent an HTTP flood").
const ACTORS_OF_TRAFFIC: [&str; 20] = ["traffic", "host", "hosts", "flow", "flows", "device", "devices", "connection", "connections", "machine", "machines", "network",
    "sender", "source", "router", "server", "servers", "client", "clients", "node"];
const ACTIONS: [&str; 24] = ["sent", "send", "sends", "sending", "launched", "launch", "launches", "launching", "carried", "carries", "conducted", "conducts", "performed",
    "performs", "mounted", "initiated", "started", "executed", "attempted", "attacked", "flooded", "floods", "scanned", "spoofed"];
fn action_claim(words: &[&str]) -> bool {
    words.iter().enumerate().any(|(i, w)| ACTORS_OF_TRAFFIC.contains(w) && words[i + 1..].iter().take(2).any(|c| ACTIONS.contains(c)))
}
/// A predicted-class term needs a hedge, a runner-up term a comparison cue; the recorded counterfactual class is allowed only inside the span a
/// change cue governs ("would predict <class>"), and never in a sentence that claims something about the traffic or host itself. Any other
/// lexicon term fails (review finding 1: a cue anywhere in the sentence no longer licenses the class).
/// Owner decision 1 (YES, 3 Nov 19:04 IST): model-attributed class wording counts as a hedge for the predicted class (`'` is a space in plain
/// text; verb tenses of the same phrases). Owner decision 5 (YES): one adverb of the closed list also, still, then, only inside "model predicted".
const OWNER_HEDGES: [&str; 26] = ["model s prediction of", "model s decision to predict", "model s choice of", "model predicting", "keep the model on",
    "keeps the model on", "kept the model on", "keep the model predicting", "keeps the model predicting", "kept the model predicting", "push the result toward",
    "pushes the result toward", "pushed the result toward", "pushing the result toward", "pull the result away from", "pulls the result away from",
    "pulled the result away from", "pulling the result away from", "model also predicted", "model still predicted", "model then predicted", "model only predicted",
    "model also predicts", "model still predicts", "model then predicts", "model only predicts"];
/// Owner decision 1 must-fail set (stricter than the base): a hedge licenses the class as the MODEL's answer only; a sentence that also states
/// something about the traffic, a host, a server, a victim or a person ("..., so this host is attacking") names no class.
const WORLD: [&str; 8] = ["victim", "victims", "target", "targets", "gateway", "attacker", "attackers", "they"];
fn world_claim(words: &[&str]) -> bool {
    let subject = |w: &&str| SUBJECTS.contains(w) || ACTORS_OF_TRAFFIC.contains(w) || WORLD.contains(w);
    // "The traffic is consistent with an ICMP flood" is the base's hedged form, not a claim.
    let hedge_after = |j: usize| words.get(j + 1).is_some_and(|n| matches!(*n, "consistent" | "typical" | "classified" | "labelled" | "labeled"));
    // "this" and "it" are subjects only right before the verb ("this is"), not as determiners ("this prediction are ...").
    let reach = |w: &str| if matches!(w, "this" | "it") { 2 } else { 3 };
    words.iter().enumerate().any(|(i, w)| subject(w) && (i + 1..(i + reach(w)).min(words.len())).any(|j| (COPULAS.contains(&words[j]) && !hedge_after(j)) || words[j] == "under"))
        || words.iter().any(|w| *w != "one" && (PERSONS.contains(w) || WORLD.contains(w)))
}
fn attacks(plain: &str, rules: &Rules) -> Result<(), String> {
    let words = plain.split(' ').collect::<Vec<_>>();
    let world = world_claim(&words);
    // The owner-approved forms state what the model DID; in a hypothetical ("would keep the model on X", "if ...") they license nothing.
    let hypothetical = words.iter().any(|w| MODALS.contains(w) || matches!(*w, "if" | "unless"));
    let hedged = !world && (HEDGES.iter().any(|w| contains_words(plain, w)) || (!hypothetical && OWNER_HEDGES.iter().any(|w| contains_words(plain, w))));
    let compared = !world && CUES.iter().any(|w| contains_words(plain, w));
    let (governed, claim, acted) = (governed(&words, rules), traffic_claim(&words), action_claim(&words));
    let followed = consequence(&words, &governed);
    let foreign = rules.attacks.iter().find(|t| contains_words(plain, t) && (acted || !((hedged && rules.predicted.contains(*t)) || (compared && rules.runner_ups.contains(*t))
        || (rules.counterfactual.contains(*t) && !claim && !followed && all_governed(&words, t, &governed)))));
    foreign.map_or(Ok(()), |t| Err(format!("attack: {t}")))
}
/// Common English words and packet vocabulary that no ROT13, reversed or pig-latin word becomes (palindromes and the ROT13 pair `be`/`or`
/// are left out). Task 5 fix round 1, finding 5.
const ENGLISH: &[&str] = &["the", "an", "is", "are", "was", "were", "of", "to", "in", "for", "with", "by", "it", "its", "this", "that", "these", "those", "and",
    "as", "at", "from", "has", "have", "had", "not", "than", "more", "less", "which", "what", "how", "why", "can", "may", "might", "could", "would", "should",
    "there", "their", "if", "so", "but", "also", "does", "do", "model", "score", "class", "feature", "features", "confidence", "prediction", "predicted",
    "contribution", "result", "evidence", "traffic", "explanation", "shap", "lime", "xai", "value", "values", "probability"];
/// A sentence the other rules cannot read fails as `banned`: digits inside a word (`m0d3l`, base64), four or more spelled-out single letters
/// (`t-h-e`), or three or more words with too few common English, packet or catalog words (ROT13, pig latin, reversed text: at least one per eight).
fn encoded(plain: &str, rules: &Rules) -> Result<(), String> {
    let words = plain.split(' ').filter(|w| !w.is_empty() && *w != "~").collect::<Vec<_>>();
    // An exponent (`2e3`) is left to the number rule.
    let exponent = |w: &str| w.split_once('e').is_some_and(|(a, b)| !a.is_empty() && !b.is_empty() && numeric(a) && numeric(b));
    let mixed = |w: &&str| { let mut kinds = w.bytes().map(|b| b.is_ascii_digit()).collect::<Vec<_>>(); kinds.dedup(); kinds.len() >= 3 && !exponent(w) };
    let letters = words.iter().filter(|w| w.len() == 1 && w.bytes().all(|b| b.is_ascii_alphabetic()) && !["a", "i", "s", "t", "d", "m"].contains(*w)).count();
    let alphabetic = words.iter().filter(|w| !numeric(w)).collect::<Vec<_>>();
    let known = alphabetic.iter().filter(|w| ENGLISH.contains(**w) || rules.vocabulary.contains(***w)).count();
    if words.iter().any(mixed) || letters >= 4 || (alphabetic.len() >= 3 && known * 8 < alphabetic.len()) { return Err("banned: encoded text".into()); }
    Ok(())
}
/// Review 2 finding 2 (structural, stricter): a change is never someone's action or a means to another answer.
const MODALS: [&str; 9] = ["could", "can", "would", "might", "may", "will", "should", "must", "cannot"];
const SKIPPABLE: [&str; 13] = ["just", "simply", "only", "then", "also", "still", "easily", "quickly", "directly", "actually", "instead", "now", "even"];
const CHANGE_ACTS: [&str; 45] = ["lower", "raise", "increase", "decrease", "reduce", "change", "alter", "modify", "adjust", "set", "tweak", "shift", "cut", "drop", "boost",
    "move", "bring", "send", "push", "pull", "double", "halve", "limit", "slow", "speed", "spread", "add", "remove", "edit", "manipulate", "craft", "inject", "use",
    "spoof", "make", "lift", "bump", "trim", "pad", "split", "delay", "throttle", "vary", "mimic", "forge"];
const GERUNDS: [&str; 30] = ["lowering", "raising", "increasing", "decreasing", "reducing", "changing", "altering", "modifying", "adjusting", "setting", "tweaking",
    "shifting", "cutting", "dropping", "boosting", "moving", "sending", "pushing", "pulling", "limiting", "slowing", "adding", "removing", "editing", "manipulating",
    "crafting", "injecting", "using", "spoofing", "padding"];
/// Objects that are the traffic or its measured values (`~` is a masked feature name).
const TRAFFIC_OBJECTS: [&str; 26] = ["~", "traffic", "packets", "packet", "messages", "message", "flows", "flow", "values", "value", "rate", "rates", "ports", "port",
    "requests", "connections", "data", "features", "feature", "inputs", "input", "measurements", "interval", "intervals", "payload", "bytes"];
/// Verbs that, with a person as the subject of a modal, make the person the one who changes the traffic or gets another answer.
const PERSON_ACTS: [&str; 9] = ["switch", "flip", "give", "get", "avoid", "hide", "evade", "keep", "turn"];
const PERSONS: [&str; 36] = ["we", "you", "they", "one", "someone", "somebody", "anyone", "anybody", "he", "she", "people", "person", "operator", "operators", "user",
    "users", "admin", "admins", "administrator", "administrators", "analyst", "analysts", "team", "sender", "senders", "defender", "defenders", "engineer", "engineers",
    "staff", "organisation", "organization", "company", "owner", "owners", "everyone"];
const PURPOSE: [&str; 31] = ["is enough", "are enough", "be enough", "was enough", "were enough", "enough so", "in order to", "so as to", "to get", "to obtain",
    "to achieve", "so the detector", "so that the detector", "so the classifier", "so that the classifier", "so it would", "so that it would", "so the ids",
    "to trigger", "to force", "how to", "stay hidden", "hide", "hidden", "hiding", "stealth", "stealthy", "unnoticed", "camouflage", "pass as", "pass for"];
/// Finite change verbs; acting on the traffic or a feature they need the recorded change or a passive auxiliary as their subject.
const FINITE_ACTS: [&str; 44] = ["lowers", "lowered", "drops", "dropped", "raises", "raised", "changes", "changed", "sets", "increases", "increased", "decreases",
    "decreased", "reduces", "reduced", "alters", "altered", "modifies", "modified", "adjusts", "adjusted", "tweaks", "tweaked", "shifts", "shifted", "cuts", "sends",
    "sent", "pushes", "pushed", "moves", "moved", "slows", "slowed", "spreads", "spaces", "spaced", "delays", "delayed", "edits", "edited", "manipulates",
    "manipulated", "throttles"];
/// Agents other than people and hosts that may stand right before a finite change verb ("once a bot lowers ...").
const AGENTS: [&str; 16] = ["bot", "bots", "script", "scripts", "tool", "tools", "program", "programs", "malware", "it", "traffic", "flow", "flows", "source",
    "botnet", "app"];
fn agency(words: &[&str], plain: &str) -> Result<(), String> {
    let object = |from: usize| words.iter().skip(from).take(3).any(|w| TRAFFIC_OBJECTS.contains(w));
    for (i, w) in words.iter().enumerate() {
        if MODALS.contains(w) {
            let mut j = i + 1;
            while j < words.len() && j <= i + 2 && SKIPPABLE.contains(&words[j]) { j += 1; }
            let verb = words.get(j).copied().unwrap_or("");
            if i > 0 && PERSONS.contains(&words[i - 1]) && (CHANGE_ACTS.contains(&verb) || PERSON_ACTS.contains(&verb)) {
                return Err(format!("agency: a person as the subject of '{w} {verb}'"));
            }
            if CHANGE_ACTS.contains(&verb) && object(j + 1) { return Err(format!("agency: '{w} {verb}' acting on the traffic or its values")); }
        }
        if *w == "by" && words.get(i + 1).is_some_and(|g| GERUNDS.contains(g)) && object(i + 2) { return Err(format!("agency: 'by {}' the traffic or its values", words[i + 1])); }
        // Security review probe: "once a bot lowers <feature>", "if somebody drops <feature>" (no modal).
        if FINITE_ACTS.contains(w) && object(i + 1) && i > 0 && (PERSONS.contains(&words[i - 1]) || AGENTS.contains(&words[i - 1]) || HOST_WORDS.contains(&words[i - 1])) {
            return Err(format!("agency: '{} {w}' the traffic or its values", words[i - 1]));
        }
    }
    PURPOSE.iter().find(|p| contains_words(plain, p)).map_or(Ok(()), |p| Err(format!("agency: purpose framing ({p})")))
}
/// Review 2 finding 8 (stricter): advice and actions are never shown. Advice modals anywhere ("would need to" describes a counterfactual),
/// advice verbs anywhere, and an imperative at the start of a clause followed by its object.
const ADVICE: [&str; 18] = ["should", "must", "ought to", "had better", "advisable", "advised", "need to", "needs to", "needed to", "want to", "look into",
    "safe", "relax", "worry", "harmless", "no threat", "all clear", "nothing to fear"];
const ADVICE_VERBS: [&str; 17] = ["isolate", "investigate", "patch", "escalate", "remediate", "mitigate", "notify", "triage", "sandbox", "blacklist", "blocklist",
    "whitelist", "allowlist", "prioritise", "prioritize", "reboot", "unplug"];
const IMPERATIVES: [&str; 51] = ["check", "review", "verify", "inspect", "monitor", "watch", "report", "alert", "act", "take", "contact", "call", "consider", "ensure",
    "update", "upgrade", "reset", "restart", "stop", "disable", "enable", "remove", "apply", "configure", "restrict", "filter", "deny", "close", "use", "treat", "flag",
    "add", "send", "lower", "raise", "increase", "decrease", "reduce", "adjust", "limit", "shut", "turn", "kill", "drop", "cut", "terminate", "delete", "wipe", "write",
    "set", "change"];
const OBJECT_STARTERS: [&str; 16] = ["the", "this", "that", "these", "those", "it", "them", "all", "any", "your", "a", "an", "its", "their", "every", "~"];
const CLAUSE_LEADS: [&str; 12] = ["so", "then", "and", "but", "please", "now", "just", "first", "also", "next", "finally", "therefore"];
fn advice(text: &str, masked: &[bool], plain: &str) -> Result<(), String> {
    let words = plain.split(' ').collect::<Vec<_>>();
    if let Some(a) = ADVICE.iter().find(|a| (0..words.len()).any(|i| phrase_at(&words, i, a) && !(a.starts_with("need") && i > 0 && words[i - 1] == "would"))) {
        return Err(format!("advice: {a}"));
    }
    if let Some(v) = ADVICE_VERBS.iter().find(|v| contains_words(plain, v)) { return Err(format!("advice: {v}")); }
    // Clauses of the punctuated text (masked feature names read as `~`); after a lead word ("so", "then", "please") a new clause starts too.
    let marked = text.bytes().zip(masked).map(|(b, m)| if *m { '~' } else { b as char }).collect::<String>();
    for clause in marked.split(|c: char| matches!(c, '.' | ',' | ';' | ':' | '!' | '?' | '(' | ')')) {
        let ws = clause.split(|c: char| !(c.is_ascii_alphanumeric() || c == '~')).filter(|w| !w.is_empty()).collect::<Vec<_>>();
        for start in (0..ws.len()).filter(|&k| k == 0 || CLAUSE_LEADS.contains(&ws[k - 1])) {
            let mut k = start;
            while k < ws.len() && CLAUSE_LEADS.contains(&ws[k]) { k += 1; }
            if ws.get(k).is_some_and(|v| IMPERATIVES.contains(v)) && ws.get(k + 1).is_some_and(|o| o.starts_with('~') || OBJECT_STARTERS.contains(o)) { return Err(format!("advice: {} (imperative)", ws[k])); }
        }
    }
    Ok(())
}
/// Security review probe (stricter): an unhedged claim that an attack is happening, whatever the class wording.
const ATTACK_CLAIMS: [&str; 18] = ["under attack", "attack is underway", "attack underway", "attack is happening", "attack in progress", "is an attack",
    "was an attack", "are attacks", "being attacked", "is malicious", "was malicious", "are malicious", "is compromised", "was compromised", "is infected",
    "was infected", "means an attack", "an attack is"];
/// Pass-rate round 1 (review 3 findings 4 and 5, stricter): what an attack does to its victims is never part of an XAI translation, whatever
/// class the sentence names ("..., which overwhelms a server", "..., which steals credentials"). A word inside a class term (`password guessing`)
/// is left to the class rules.
const ATTACK_MECHANICS: [&str; 33] = ["overwhelm", "overload", "take down", "takes down", "took down", "taking down", "knock", "offline", "crash", "steal", "stole", "stolen",
    "intercept", "eavesdrop", "hijack", "exploit", "infect", "victim", "credential", "password", "login", "logins", "at risk", "exposed", "compromise", "damage",
    "harm", "disrupt", "outage", "downtime", "impersonate", "spy", "sniff"];
fn claims(plain: &str, rules: &Rules) -> Result<(), String> {
    if let Some(c) = ATTACK_CLAIMS.iter().find(|c| contains_words(plain, c)) { return Err(format!("attack: unhedged attack claim ({c})")); }
    match ATTACK_MECHANICS.iter().find(|w| contains_words(plain, w) && !rules.attacks.iter().any(|t| contains_words(t, w))) {
        Some(w) => Err(format!("attack: attack mechanics ({w})")),
        None => Ok(()),
    }
}
/// Review 3 finding 4 (stricter): the counterfactual class is licensed only in a clause that ends with it: no relative or consequence clause
/// may follow the governed class ("..., meaning someone is impersonating the gateway"), and no person may be named in the sentence.
const CONSEQUENCES: [&str; 22] = ["which", "meaning", "means", "meant", "so", "thus", "therefore", "hence", "that is", "this is", "i e", "who", "whereby", "because",
    "since", "implying", "implies", "indicating", "indicates", "showing", "signals", "making"];
fn consequence(words: &[&str], governed: &[bool]) -> bool {
    let Some(last) = governed.iter().rposition(|g| *g) else { return false };
    let tail = words[last + 1..].join(" ");
    CONSEQUENCES.iter().any(|c| contains_words(&tail, c)) || words.iter().any(|w| *w != "one" && PERSONS.contains(w))
}
/// Security review probe (stricter): a sentence names something of this explanation (a topic word, a packet feature, a number or a class
/// term), or it does another task ("Write a Python function ...", "Here is a recipe ...") and is not shown.
const TOPIC: [&str; 72] = ["count", "counts", "rate", "flag", "packets", "packet", "bytes", "port", "protocol", "model", "class", "classes", "feature", "features", "value", "values", "score", "scores", "confidence", "xai", "shap", "lime", "anchor",
    "anchors", "counterfactual", "permutation", "importance", "ale", "effect", "effects", "contribution", "contributions", "prediction", "predicted", "predict",
    "result", "results", "evidence", "rule", "rules", "change", "changes", "changed", "observation", "traffic", "network", "margin", "probability", "loss", "fit",
    "reference", "data", "precision", "coverage", "tolerance", "condition", "search", "answer", "input", "inputs", "measurement", "measurements", "row", "rows",
    "explanation", "estimate", "method", "output", "baseline", "flow", "group", "linked"];
/// Pass-rate round 1 (review 3 finding 5, stricter): words too generic to tie a clause to this result (`model`, `data`, `network`, ...) do not
/// count; talk about the explainer's instructions, ML in general and other tasks fails whatever else the sentence names. A clause-level check
/// ("..., and <unrelated fact>") was measured on 1371 real model sentences and rejected faithful anaphoric clauses ("but these were minor"),
/// so an unrelated clause is caught only by these patterns (stated as a residual in docs/XAI_CHAT.md).
const GENERIC_TOPIC: [&str; 10] = ["model", "data", "network", "traffic", "flow", "rate", "input", "inputs", "output", "method"];
const INSTRUCTION_TALK: [&str; 24] = ["system prompt", "my prompt", "the prompt", "my instructions", "my instruction", "the instructions", "i was told", "i am told",
    "i ve been told", "i have been told", "i was instructed", "i am instructed", "i was asked", "my rules", "the rules i", "rules i was given", "i was given",
    "my guidelines", "my programming", "i am programmed", "i m programmed", "developer message", "according to my", "i must not reveal"];
const GENERAL: [&str; 33] = ["generally speaking", "machine learning", "deep learning", "neural network", "decision tree", "training data", "trained on",
    "learn from", "learns from", "learned from", "learn patterns", "learns patterns", "artificial intelligence", "popular", "commonly used", "widely used", "fun fact",
    "poem", "poetry", "roses are red", "capital of", "recipe", "haiku", "joke", "the sky", "weather", "python", "javascript", "source code", "essay", "song",
    "once upon", "large datasets"];
const THIS_RESULT_REFS: [&str; 7] = ["here", "estimate", "model s own", "less likely", "runner up", "new class", "no feasible"];
fn topical(plain: &str, rules: &Rules) -> Result<(), String> {
    if let Some(p) = INSTRUCTION_TALK.iter().find(|p| contains_words(plain, p)) { return Err(format!("off_topic: instruction talk ({p})")); }
    if let Some(p) = GENERAL.iter().find(|p| contains_words(plain, p)) { return Err(format!("off_topic: not about this result ({p})")); }
    let on = |clause: &str| {
        let words = clause.split(' ').filter(|w| !w.is_empty()).collect::<Vec<_>>();
        words.iter().any(|w| *w == "~" || w.starts_with(|c: char| c.is_ascii_digit()) || TOPIC.iter().filter(|t| !GENERIC_TOPIC.contains(t)).any(|t| contains_words(w, t)))
            || THIS_RESULT.iter().chain(&THIS_RESULT_REFS).any(|r| contains_words(clause, r))
            || rules.attacks.iter().any(|t| contains_words(clause, t)) || rules.predicted_names.iter().chain(&rules.counterfactual_names).any(|n| contains_words(clause, n))
    };
    if !on(plain) { return Err("off_topic: the sentence is not about this result or its XAI evidence".into()); }
    Ok(())
}
/// A stricter rule (redesign round 1): without a recorded counterfactual no sentence may claim the model would give another class or answer.
fn hypothetical(plain: &str, rules: &Rules) -> Result<(), String> {
    if !(rules.class_change || (rules.no_change_found && negated(plain))) {
        if let Some(w) = CLASS_CHANGE.iter().find(|w| contains_words(plain, w)) { return Err(format!("hypothetical: {w} (this XAI records no class change)")); }
    }
    // Round 2 final run: no answer change is stated as having happened (a counterfactual records what WOULD happen; other methods none).
    if !contains_words(plain, "would") && !(rules.no_change_found && negated(plain)) {
        if let Some(w) = CHANGED.iter().find(|w| contains_words(plain, w)) { return Err(format!("hypothetical: {w} (no answer change happened)")); }
    }
    // Pass-rate round 1 (stricter): a found counterfactual records a DIFFERENT answer after the change, so a change cue never governs the
    // predicted class ("after the recorded change it would still predict ddos icmp flood").
    if rules.class_change {
        let words = plain.split(' ').filter(|w| !matches!(*w, "still" | "again" | "also" | "then" | "just" | "only")).collect::<Vec<_>>();
        let names = rules.predicted_names.iter().map(|n| n.split(' ').collect::<Vec<_>>()).collect::<Vec<_>>();
        for cue in CHANGE_CUES.iter().chain(&["would still", "still predict", "still give"]).map(|c| c.split(' ').collect::<Vec<_>>()) {
            for i in (0..words.len()).filter(|&i| words[i..].starts_with(&cue)) {
                let mut j = i + cue.len();
                while j < words.len() && j < i + cue.len() + 3 && FILLER.contains(&words[j]) { j += 1; }
                if names.iter().any(|n| words[j..].starts_with(n)) { return Err("hypothetical: the recorded change gives a different class, not the predicted one".into()); }
            }
        }
        if plain.split(' ').any(|w| w == "still") && contains_words(plain, "would") && rules.predicted_names.iter().any(|n| contains_words(plain, n)) {
            return Err("hypothetical: the recorded change gives a different class, not the predicted one".into());
        }
    }
    // The recorded counterfactual class is what the model WOULD predict after the change: its name needs a governing change cue.
    let words = plain.split(' ').collect::<Vec<_>>();
    let governed = governed(&words, rules);
    match rules.counterfactual_names.iter().find(|n| contains_words(plain, n) && !all_governed(&words, n, &governed)) {
        Some(n) => Err(format!("hypothetical: {n} is named only as what the model would predict after the recorded change")),
        None => Ok(()),
    }
}
/// Pass-rate round 1 after-run (stricter): the packet lists only the top features, so a claim about every feature or no feature is invented
/// ("No features pulled the result away"); it is allowed only about the listed, recorded, top or shown features.
const ALL_OR_NONE: [&str; 11] = ["no feature", "no features", "no other feature", "no other features", "none of the features", "all features", "all the features",
    "all other features", "all of the features", "every feature", "every other feature"];
fn generalization(plain: &str) -> Result<(), String> {
    let words = plain.split(' ').collect::<Vec<_>>();
    for (i, _) in words.iter().enumerate() {
        if let Some(p) = ALL_OR_NONE.iter().find(|p| phrase_at(&words, i, p)) {
            let qualified = words.iter().any(|w| matches!(*w, "listed" | "recorded" | "top" | "shown"));
            if !qualified { return Err(format!("generalization: '{p}' claims more than the listed top features")); }
        }
    }
    Ok(())
}
/// Runs after every rule but `off_topic`, so every earlier violation keeps its name.
/// "reliable rule" (PRESETS.md anchors: "say no reliable rule was found") describes the rule search, so that one word is not read as a judgement.
fn judgement(plain: &str) -> Result<(), String> {
    let words = plain.split(' ').collect::<Vec<_>>();
    // "no valid rule" is the same rule-search wording as "no reliable rule".
    let plain = words.iter().enumerate().filter(|(i, w)| !(matches!(**w, "reliable" | "valid") && matches!(words.get(i + 1), Some(&"rule" | &"rules")))).map(|(_, w)| *w).collect::<Vec<_>>().join(" ");
    JUDGEMENT.iter().find(|w| contains_words(&plain, w)).map_or(Ok(()), |w| Err(format!("judgement: {w}")))
}
/// `Err` text starts with the rule name: `banned`, `length`, `feature`, `advice`, `agency`, `number`, `pairing`, `direction`, `attack`,
/// `hypothetical`, `anchor`, `change`, `importance`, `cause`, `generalization`, `judgement` or `off_topic` (redesign round 3 added `advice`, `agency`, `off_topic`).
pub fn check(sentence: &str, rules: &Rules) -> Result<(), String> {
    let text = ascii(sentence)?;
    if text.split_whitespace().count() > 60 { return Err("length: more than 60 words".into()); }
    banned(&text)?;
    let mut masked = vec![false; text.len()];
    let mentions = features(&text, rules, &mut masked)?;
    if hostname(&text.bytes().zip(&masked).map(|(b, m)| if *m { ' ' } else { b as char }).collect::<String>()) { return Err("banned: domain name".into()); }
    let plain = plain(&text, &masked);
    encoded(&plain, rules)?;
    advice(&text, &masked, &plain)?;
    agency(&plain.split(' ').collect::<Vec<_>>(), &plain)?;
    let mut exempt = masked.clone();
    class_spans(&text, &masked, rules, &mut exempt);
    let digits = text.bytes().zip(&exempt).map(|(b, m)| if *m { b' ' } else { b }).collect::<Vec<_>>();
    numbers(&digits, rules, &mentions)?;
    spelled(&plain, rules)?;
    if let Some(w) = MULTIPLIERS.iter().find(|w| contains_words(&plain, w)) { return Err(format!("number: multiplier '{w}' is not recorded")); }
    pairing(&digits, &mentions, rules)?;
    // Review finding 2: a sentence about the score is read against the recorded score difference AND each named feature's own recorded change.
    let mut mentioned = mentions.iter().map(|m| m.value).collect::<Vec<_>>();
    if let Some(s) = rules.score_change.filter(|_| SCORE_WORDS.iter().any(|w| contains_words(&plain, w))) { mentioned.push(s); }
    let unitless = if rules.units.is_empty() { plain.clone() } else { format!(" {plain} ").replace(&format!(" {} ", rules.units), " ~ ").trim().to_string() };
    // Review 2 finding 6: a linked counterfactual group changed together; it has no recorded direction, so none may be stated with it.
    for m in mentions.iter().filter(|m| rules.groups.contains(&m.name)) {
        let clause = clause_around(&text, &masked, m);
        if let Some(w) = INCREASE.iter().chain(DECREASE).chain(&GROUP_MOVES).find(|w| contains_words(&clause, w)) {
            return Err(format!("direction: a linked group changed together and has no recorded direction ({w})"));
        }
    }
    if rules.directional && !mentioned.is_empty() { direction(&unitless, &mentioned)?; }
    // Review 3 finding 1 (stricter): a single counterfactual change, with every movement word around it, against its recorded to - from.
    if rules.method == "counterfactual" {
        for k in 0..mentions.len() {
            let Some((from, to)) = rules.facts.get(&mentions[k].name).and_then(|f| Some((f.from?, f.to?))) else { continue };
            let span = own_span(&text, &masked, &mentions, k);
            let up = INCREASE.iter().chain(&MOVES_UP).find(|w| contains_words(&span, w)).filter(|_| to - from <= 0.);
            let down = DECREASE.iter().chain(&MOVES_DOWN).find(|w| contains_words(&span, w)).filter(|_| to - from >= 0.);
            if let Some(w) = up.or(down) { return Err(format!("direction: '{w}' about {}, recorded from {from} to {to}", mentions[k].name)); }
        }
    }
    attacks(&plain, rules)?;
    hypothetical(&plain, rules)?;
    claims(&plain, rules)?;
    faithful(&plain, &mentions, rules, tokens(&digits).next().is_some())?;
    generalization(&plain)?;
    judgement(&plain)?;
    topical(&plain, rules)
}
/// The words that describe one named feature's change: the word right before it (`lowering <feature>`) and the text after it up to the next
/// named feature or the end of its clause (`,` `;` `:`).
fn own_span(text: &str, masked: &[bool], mentions: &[Mention], k: usize) -> String {
    let (b, m) = (text.as_bytes(), &mentions[k]);
    let stop = |i: usize| !masked[i] && matches!(b[i], b',' | b';' | b':');
    let clause_start = (0..m.start).rev().find(|&i| stop(i)).map_or(0, |i| i + 1);
    let lead_from = if k > 0 { mentions[k - 1].end.max(clause_start) } else { clause_start };
    let before = plain(&text[lead_from..m.start], &masked[lead_from..m.start]);
    let lead = before.split(' ').filter(|w| !w.is_empty() && *w != "~").next_back().unwrap_or("").to_string();
    let end = (m.end..b.len()).find(|&i| stop(i)).unwrap_or(b.len());
    let end = mentions.get(k + 1).map_or(end, |n| n.start.min(end)).max(m.end);
    format!("{lead} {}", plain(&text[m.end..end], &masked[m.end..end]))
}

/// The word before byte `at` (lowercase; empty at the start).
fn prev_word(t: &[u8], at: usize) -> String {
    let mut j = at.min(t.len());
    while j > 0 && !t[j - 1].is_ascii_alphanumeric() { j -= 1; }
    let end = j;
    while j > 0 && t[j - 1].is_ascii_alphanumeric() { j -= 1; }
    String::from_utf8_lossy(&t[j..end]).into()
}
/// The word after byte `at` (lowercase; empty at the end).
fn next_word(t: &[u8], at: usize) -> String {
    let mut j = at.min(t.len());
    while j < t.len() && !t[j].is_ascii_alphanumeric() { j += 1; }
    let start = j;
    while j < t.len() && t[j].is_ascii_alphanumeric() { j += 1; }
    String::from_utf8_lossy(&t[start..j]).into()
}
/// A plain-decimal number token as `numbers` reads it: its byte span (sign included), signed value, decimals and whether it is a percentage.
struct Token { start: usize, end: usize, x: f64, decimals: i32, negative: bool, percent: bool }
impl Token {
    /// The same tolerance as `numbers`: unsigned tokens compare with `|e|`.
    fn is(&self, e: f64) -> bool { (self.x - if self.negative { e } else { e.abs() }).abs() <= 0.5 * 10f64.powi(-self.decimals) + 1e-9 * e.abs().max(1.) }
    fn text(&self, t: &[u8]) -> String { String::from_utf8_lossy(&t[self.start..self.end]).trim().into() }
    /// Pass-rate round 1 (review 3 findings 1 and 6): a counterfactual from/to value, or a contribution worded as a value, also within 5% of the
    /// recorded value (`0` is not `0.01734`, nor a contribution of `-0.00000215`).
    fn near_exact(&self, e: f64) -> bool { self.is(e) && if e == 0. { self.x == 0. } else { (self.x.abs() - e.abs()).abs() <= 0.05 * e.abs() } }
}
fn tokens(t: &[u8]) -> impl Iterator<Item = Token> + '_ {
    let digit = move |j: usize| t.get(j).is_some_and(u8::is_ascii_digit);
    let mut i = 0;
    std::iter::from_fn(move || {
        while i < t.len() && !(digit(i) || (t[i] == b'.' && digit(i + 1))) { i += 1; }
        if i >= t.len() { return None; }
        let negative = i > 0 && t[i - 1] == b'-' && (i < 2 || !t[i - 2].is_ascii_alphanumeric());
        let (start, mut raw, mut decimals) = (if negative { i - 1 } else { i }, String::new(), 0);
        while digit(i) || (t.get(i) == Some(&b',') && !raw.is_empty() && (1..4).all(|k| digit(i + k)) && !digit(i + 4)) { if t[i] != b',' { raw.push(t[i] as char); } i += 1; }
        if t.get(i) == Some(&b'.') && digit(i + 1) { raw.push('.'); i += 1; while digit(i) { raw.push(t[i] as char); i += 1; decimals += 1; } }
        let end = i;
        let rest = &t[i..]; let rest = rest.strip_prefix(b" ").unwrap_or(rest);
        let percent = rest.starts_with(b"%") || (rest.starts_with(b"percent") && !rest.get(7).is_some_and(u8::is_ascii_alphanumeric));
        let x = format!("0{raw}").parse::<f64>().unwrap_or(f64::NAN) * if negative { -1. } else { 1. };
        Some(Token { start, end, x, decimals, negative, percent })
    })
}
/// Review findings 3 and 5 (stricter): a number after a named packet feature (permutation, ALE, counterfactual) must be one recorded for
/// that feature, and a counterfactual `to Y` must be that feature's recorded `to` value; a number next to an anchor statistic's words must
/// be that statistic (the lower bound counts for precision), and a count of reference rows must be the recorded support.
fn pairing(t: &[u8], mentions: &[Mention], rules: &Rules) -> Result<(), String> {
    for token in tokens(t) {
        let next = next_word(t, token.end);
        if matches!(next.as_str(), "classes" | "possible") { continue; }
        if !token.percent {
            let prev = prev_word(t, token.start);
            // Pass-rate round 1 (review 3 finding 1, stricter): in a counterfactual a `from X` / `to Y` value belongs to the feature named right
            // before it: never to no feature (`From 2.642, <feature> rose ...`) and never to several features at once (`a and b both changed to Y`).
            let clause = rules.method == "counterfactual" && matches!(prev.as_str(), "from" | "to");
            // The methods paired before this round (permutation, ALE, counterfactual) keep their earlier reading wherever a refinement below
            // could accept more than before; SHAP, LIME and anchors had no pairing at all, so everything here is stricter for them.
            let old = matches!(rules.method.as_str(), "permutation" | "ale" | "counterfactual");
            let before = mentions.iter().rposition(|m| m.end <= token.start);
            // "a count of 47 for <feature>": the number belongs to the feature right after `for` (new methods only).
            let follow = (!old).then(|| mentions.iter().position(|n| n.start >= token.end && {
                let w = String::from_utf8_lossy(&t[token.end..n.start]).split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()).map(String::from).collect::<Vec<_>>();
                w == ["for"] || w == ["for", "the"] })).flatten();
            let at = follow.or(before);
            if clause && at.is_none() { return Err(format!("pairing: {} is not tied to a recorded change of a named feature", token.text(t))); }
            if let Some(i) = at {
                let m = &mentions[i];
                if clause && i > 0 && shared(&t[mentions[i - 1].end..m.start]) {
                    return Err(format!("pairing: {} is one from/to value given to several features", token.text(t)));
                }
                let target = rules.method == "counterfactual" && prev == "to";
                let lo = before.map_or(0, |k| mentions[k].end);
                let role = role(&t[lo..token.start]);
                let respectively = words_after(t, token.end).iter().chain(words_before(t, token.start, usize::MAX).iter()).any(|w| w == "respectively");
                // The original reset words keep their meaning for the old methods; SHAP, LIME and anchors also reset at a score or probability word.
                let reset = role.is_some_and(|r| matches!(r, "confidence" | "classes" | "runner") || (!old && matches!(r, "score" | "scores" | "probability")));
                let copula = matches!(prev.as_str(), "was" | "is" | "were" | "are" | "at" | "equals" | "equal" | "equalled");
                let contribution = role.is_some_and(|r| CONTRIBUTION_ROLES.contains(&r));
                // "contributed positively with a value of 0.38": the model calls the contribution a value when a contribution word stands before it.
                let contributed = words_before(t, token.start, usize::MAX).iter().any(|w| CONTRIBUTION_ROLES.contains(&w.as_str()) || w.starts_with("contribut"));
                // "The recorded importance for a, b and <group> is 0": the quantity is named before the features.
                let named_before = words_before(t, token.start, usize::MAX).iter().any(|w| CONTRIBUTION_ROLES.contains(&w.as_str()));
                // The role-checked reading of one feature's facts (stricter than the earlier any-fact reading).
                let read = |fact: &Fact| -> Result<(), &'static str> {
                    if role == Some("value") || role == Some("values") {
                        // Review 3 finding 6: a number worded as a value is the feature's recorded value (a linked group has none) ...
                        (fact.value.is_some_and(|v| token.is(v)) || (contributed && token.near_exact(fact.contribution))).then_some(()).ok_or("as the value ")
                    } else if contribution {
                        // ... and one worded as a contribution, importance, effect or weight is its recorded contribution.
                        token.is(fact.contribution).then_some(()).ok_or("as the contribution ")
                    } else if fact.group && copula && !named_before {
                        // A linked group stated with a copula (`log_messages_count was 0`) has no single value: only its size or name digits.
                        fact.own.iter().filter(|e| **e != fact.contribution).any(|e| token.is(*e)).then_some(()).ok_or("as a size ")
                    } else { fact.own.iter().any(|e| token.is(*e)).then_some(()).ok_or("") }
                };
                if let Some(fact) = rules.facts.get(&m.name) {
                    // Review 2 finding 5: a target is checked against its feature even past a `confidence` word.
                    let verdict = if target { fact.to.is_some_and(|to| token.near_exact(to)).then_some(()).ok_or("as the changed value ") }
                        else if clause { fact.from.is_some_and(|from| token.near_exact(from)).then_some(()).ok_or("as the value before the change ") }
                        else if reset { Ok(()) }
                        // "a and b had values of 0 and 1, respectively": for the old methods exactly the earlier reading (the nearest feature's
                        // facts); for the new ones the role-checked reading of any feature the sentence names.
                        else if respectively && old { fact.own.iter().any(|e| token.is(*e)).then_some(()).ok_or("") }
                        else if respectively { mentions.iter().filter_map(|n| rules.facts.get(&n.name)).any(|f| read(f).is_ok()).then_some(()).ok_or("for any named feature; ") }
                        else { read(fact) };
                    if let Err(what) = verdict { return Err(format!("pairing: {} is not recorded {what}for {}", token.text(t), m.name)); }
                }
            }
        }
        let Some(a) = &rules.anchor else { continue };
        let stat = anchor_stat(t, token.start, token.end, token.percent, usize::MAX);
        if stat.is_empty() { continue; }
        let scale = if token.percent { if matches!(stat, "tolerance" | "support") { f64::NAN } else { 100. } } else { 1. };
        if !anchor_values(a, stat).into_iter().any(|e| token.is(e * scale)) { return Err(format!("pairing: {} is not the recorded anchor {stat}", token.text(t))); }
    }
    Ok(())
}
/// Words that tie a number to its role after a named feature (the nearest one with no other number in between decides).
const CONTRIBUTION_ROLES: [&str; 11] = ["contribution", "contributions", "contributed", "effect", "effects", "importance", "importances", "weight", "weights",
    "weighted", "impact"];
const ROLES: [&str; 8] = ["value", "values", "confidence", "classes", "runner", "score", "scores", "probability"];
/// The role word nearest before a number in the text after a feature (`had value`, `a contribution of`), or `None`; a number in between ends the search.
fn role(between: &[u8]) -> Option<&'static str> {
    for w in String::from_utf8_lossy(between).split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()).rev() {
        if w.starts_with(|c: char| c.is_ascii_digit()) { return None; }
        if let Some(r) = ROLES.iter().chain(&CONTRIBUTION_ROLES).find(|r| **r == w) { return Some(r); }
    }
    None
}
/// The text between two named features holds only list connectors (`a and b`, `a, b`, `a as well as b`): they share what follows.
fn shared(between: &[u8]) -> bool {
    String::from_utf8_lossy(between).split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty())
        .all(|w| matches!(w, "and" | "or" | "both" | "plus" | "also" | "as" | "well" | "along" | "with" | "together" | "the" | "all"))
}
/// A negator among the three words before index `i` (`n't` reads as `n t`).
fn negated_before(words: &[&str], i: usize) -> bool {
    (i.saturating_sub(3)..i).any(|k| NEGATORS.contains(&words[k]) || (words[k] == "t" && k > 0 && words[k - 1].ends_with('n')))
}
fn phrase_at(words: &[&str], i: usize, phrase: &str) -> bool { let p = phrase.split(' ').collect::<Vec<_>>(); words[i..].starts_with(&p) }
/// Review findings 3, 5 and 6 (all stricter): method-specific faithful translation.
/// `anchor`: an unsupported anchor is never called supported or reliable, nor the condition or reason for the predicted class; a negated
/// statistic ("did not keep ... in 0") fails. `change`: ALE records an effect at this value, not a change between values. `importance`: a
/// permutation claim that a feature matters needs a positive recorded importance. `cause`: permutation is global (no wording about this
/// prediction or decision), and ALE or anchors give no cause of the prediction or of its confidence.
fn faithful(plain: &str, mentions: &[Mention], rules: &Rules, numbered: bool) -> Result<(), String> {
    let words = plain.split(' ').collect::<Vec<_>>();
    let any = |list: &[&'static str]| -> Option<&'static str> { list.iter().find(|w| contains_words(plain, w)).copied() };
    let predicted = contains_words(plain, "predict") || rules.predicted_names.iter().any(|n| contains_words(plain, n));
    if let Some(a) = &rules.anchor {
        if !a.supported {
            if let Some(claim) = SUPPORT_CLAIMS.iter().find(|c| (0..words.len()).any(|i| phrase_at(&words, i, c) && !negated_before(&words, i))) {
                return Err(format!("anchor: no supported rule was recorded ({claim})"));
            }
            if let Some(w) = any(&CONDITIONAL).filter(|_| predicted) { return Err(format!("anchor: an unsupported rule is not the condition or reason for the prediction ({w})")); }
        }
        if numbered && words.iter().enumerate().any(|(i, w)| STAT_VERBS.contains(w) && negated_before(&words, i)) { return Err("anchor: negated statistic".into()); }
    }
    if rules.method == "ale" {
        let from_value = words.iter().enumerate().any(|(i, w)| *w == "from" && words[i + 1..].iter().take(3).any(|n| n.starts_with(|c: char| c.is_ascii_digit()) || *n == "baseline"));
        let noun = words.iter().enumerate().any(|(i, w)| CHANGE_NOUNS.contains(w) && words.get(i + 1) == Some(&"in")
            && (words.get(i + 2) == Some(&"~") || (words.get(i + 2).is_some_and(|d| ["that", "its", "the", "this", "their"].contains(d)) && words.get(i + 3).is_some_and(|v| ["value", "values"].contains(v)))));
        let verb = words.iter().enumerate().any(|(i, w)| CHANGE_VERBS.contains(w) && words.get(i + 1) == Some(&"~"));
        if from_value || noun || verb || any(&ALE_CHANGES).is_some() { return Err("change: ALE records an effect at this value, not a change from one value to another".into()); }
    }
    if rules.method == "permutation" && mentions.iter().any(|m| m.value <= 0.) {
        if let Some(w) = words.iter().enumerate().find(|(i, w)| IMPORTANCE_CLAIMS.contains(w) && !(i.saturating_sub(2)..*i).any(|k| QUALIFIERS.contains(&words[k]))).map(|(_, w)| *w) {
            return Err(format!("importance: '{w}' about a feature whose recorded importance is not above 0"));
        }
    }
    if rules.method == "permutation" { if let Some(w) = any(&LOCAL) { return Err(format!("cause: permutation importance is global, not about this prediction ({w})")); } }
    if matches!(rules.method.as_str(), "ale" | "anchors") && (predicted || contains_words(plain, "confidence") || contains_words(plain, "confident")) {
        if let Some(w) = any(&CAUSAL) { return Err(format!("cause: this XAI does not give a cause of the prediction ({w})")); }
    }
    Ok(())
}
/// How a sentence relates to this result: whether it states a recorded fact (a packet feature, a number, the recorded search outcome or the
/// counterfactual's new class), names the predicted class, says only "the model predicted <class>", or describes how the method works.
struct Reading { fact: bool, predicted: bool, bare: bool, definition: bool }
/// Words a bare prediction sentence consists of besides the predicted class's own words ("The model predicted X for this observation.").
const BARE: [&str; 24] = ["the", "model", "predicted", "predicts", "predicting", "prediction", "class", "a", "an", "this", "for", "result", "here", "its", "answer",
    "was", "as", "observation", "traffic", "row", "of", "xgboost", "s", "is"];
/// A method name or a reference to the method as the subject of a definition ("SHAP shows how ...", "This XAI shows which ...").
const METHOD_SUBJECTS: [&str; 14] = ["shap", "lime", "anchor", "anchors", "counterfactual", "permutation", "importance", "ale", "xai", "method", "search", "technique",
    "explanation", "it"];
const DEFINING: [&str; 22] = ["shows", "show", "showed", "measures", "measure", "measured", "explains", "describes", "tells", "highlights", "reveals", "indicates",
    "computes", "calculates", "estimates", "looks", "searches", "searched", "looked", "splits", "builds", "means"];
const ASKS: [&str; 7] = ["how", "which", "what", "whether", "where", "why", "for"];
/// The method's general workings (PRESETS.md descriptions are LLM-only).
const MECHANICS: [&str; 21] = ["measured by", "by measuring", "compared to a baseline", "compared with a baseline", "relative to a baseline",
    "against a baseline", "from a baseline", "are shuffled", "is shuffled", "were shuffled", "was shuffled", "averaged over", "averaged across", "changed copies",
    "smallest change", "a specific change", "a small change", "changing specific", "specific measured values", "each feature", "how changes in"];
/// References to this one result, which the preset's own part-1 wording uses ("pushed this result toward <class>").
const THIS_RESULT: [&str; 7] = ["this result", "this observation", "this prediction", "this specific", "this one", "this row", "this traffic"];
fn reading(sentence: &str, rules: &Rules) -> Option<Reading> {
    let text = ascii(sentence).ok()?;
    let mut masked = vec![false; text.len()];
    let mentions = features(&text, rules, &mut masked).ok()?;
    let plain = plain(&text, &masked);
    let words = plain.split(' ').filter(|w| !w.is_empty()).collect::<Vec<_>>();
    let fact = !mentions.is_empty() || words.iter().any(|w| w.starts_with(|c: char| c.is_ascii_digit())) || RESULT_STATES.iter().any(|s| contains_words(&plain, s))
        || rules.counterfactual_names.iter().any(|n| contains_words(&plain, n)) || rules.counterfactual.iter().any(|t| contains_words(&plain, t));
    let predicted = rules.predicted_names.iter().any(|n| contains_words(&plain, n));
    let own = rules.predicted_names.iter().flat_map(|n| n.split(' ')).collect::<BTreeSet<_>>();
    let bare = predicted && words.iter().all(|w| BARE.contains(w) || own.contains(w));
    let asks = words.windows(3).any(|w| METHOD_SUBJECTS.contains(&w[0]) && DEFINING.contains(&w[1]) && ASKS.contains(&w[2]));
    let local = THIS_RESULT.iter().any(|s| contains_words(&plain, s));
    let definition = !fact && (MECHANICS.iter().any(|m| contains_words(&plain, m)) || (asks && !(predicted && local)));
    Some(Reading { fact, predicted, bare, definition })
}
/// Review finding 8 and review 2 finding 7: a part-1 ('shows') sentence names this result: a recorded fact, or the predicted class in a
/// sentence that says more than "the model predicted <class>" and is not a method definition.
pub fn names_result(sentence: &str, rules: &Rules) -> bool {
    reading(sentence, rules).is_some_and(|r| r.fact || (r.predicted && !r.bare && !r.definition))
}
/// Review 2 finding 7: a sentence that describes how the method works and states no recorded fact is the LLM-only method description,
/// never shown in any part (follow-up answers included).
pub fn defines_method(sentence: &str, rules: &Rules) -> bool { reading(sentence, rules).is_some_and(|r| r.definition) }
/// Round 3 real-model run: a sentence that only says "the model predicted <class>" (dropped from part 1, where it names nothing new).
pub fn bare_prediction(sentence: &str, rules: &Rules) -> bool { reading(sentence, rules).is_some_and(|r| r.bare) }

fn abbreviation(head: &str) -> bool {
    let head = head.to_ascii_lowercase();
    ["e.g.", "i.e.", "vs."].iter().any(|a| head.strip_suffix(a).is_some_and(|before| !before.ends_with(|c: char| c.is_ascii_alphanumeric())))
}
/// Drains complete sentences and keeps the unfinished tail. A sentence ends at a newline, or at `.`, `!` or `?` (plus closing quotes or
/// brackets) followed by whitespace; the end of the buffer is never an end, so a streamed `0.` waits for `42`. `e.g.`, `i.e.` and `vs.` never end one.
pub fn split(buffer: &mut String) -> Vec<String> {
    let mut out = Vec::new();
    loop {
        let b = buffer.as_bytes();
        let end = (0..b.len()).find_map(|i| match b[i] {
            b'\n' => Some(i + 1),
            b'.' | b'!' | b'?' => {
                let mut j = i + 1; while matches!(b.get(j), Some(b'"' | b'\'' | b')')) { j += 1; }
                (b.get(j).is_some_and(u8::is_ascii_whitespace) && !(b[i] == b'.' && abbreviation(&buffer[..=i]))).then_some(j + 1)
            }
            _ => None,
        });
        let Some(end) = end else { break };
        let sentence = buffer[..end].trim().to_string(); buffer.drain(..end);
        if !sentence.is_empty() { out.push(sentence); }
    }
    out
}
/// The tail left at the end of the stream (call after the last `split`).
pub fn finish(buffer: &mut String) -> Option<String> { let tail = buffer.trim().to_string(); buffer.clear(); (!tail.is_empty()).then_some(tail) }

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::packet::{build,lexicon};
    use serde_json::{json,Value};
    fn classes()->Vec<String>{["benign","ddos_icmp-flood","dos_icmp-flood","ddos_http-flood-port-80","ddos_slowloris-port-80","ddos_mqtt-publish-flood","ddos_syn-flood-port-1883","recon_scan"].map(String::from).to_vec()}
    fn all_features()->Vec<String>{["flow_bytes_per_s","flow_bytes","syn_flag_count","ack_flag_count","network_ports_src__present_1883","network_ports_src__present_80","network_protocols_all__present_mqtt","network_protocols_all__present_tcp"].map(String::from).to_vec()}
    fn prediction(label:&str)->Value{json!({"label":label,"score":0.93,"score_kind":"class_probability","top_classes":[{"label":label,"score":0.93},{"label":"dos_icmp-flood","score":0.04},{"label":"ddos_http-flood-port-80","score":0.02}]})}
    fn rules_for(label:&str,method:&str,features:Value)->Rules{rules(&build(&prediction(label),"XGBoost",77,Some(&json!({"method":method,"units":"raw margin","features":features}))),&all_features(),&lexicon(&classes()))}
    fn fixture_features()->Value{json!([{"feature":"flow_bytes_per_s","value":15234.7,"contribution":0.4187},{"feature":"syn_flag_count","value":12.0,"contribution":-0.1203}])}
    fn fixture()->Rules{rules_for("ddos_icmp-flood","exact_tree_shap",fixture_features())}
    /// The rule name that rejected the sentence, or "" when it passed.
    fn rule(r:Result<(),String>)->String{r.err().map(|e|e.split(':').next().unwrap_or_default().to_string()).unwrap_or_default()}
    fn passes(r:&Rules,sentences:&[&str]){for s in sentences{assert_eq!(check(s,r),Ok(()),"{s}");}}
    fn fails(r:&Rules,name:&str,sentences:&[&str]){for s in sentences{assert_eq!(rule(check(s,r)),name,"{s}");}}
    #[test] fn number_formats_match_at_stated_precision(){
        let r=fixture();
        passes(&r,&["The contribution was 0.42.","The contribution was .42.","The confidence was 93%.","The contribution was -0.12.","The confidence was 93 percent.",
            "The value was 15,235.","It used 3 features.","The second class scored 4%.","The model chose among 77 classes.","The exact contribution was 0.4187.","It lowered the score by 0.12."]);
        fails(&r,"number",&["The contribution was 0.55.","The confidence was 97%.","The contribution was -0.42.","The contribution was 0.420.","It used 6 features.","Its share was 42%."]);
    }
    #[test] fn unknown_feature_is_rejected(){
        let r=fixture();
        fails(&r,"feature",&["The ack_flag_count was unusual.","ACK flag count was unusual.","flow_bytes was unusual."]);
        passes(&r,&["flow_bytes_per_s was unusual.","Flow-Bytes per S was unusual."]);
    }
    #[test] fn direction_must_match_sign(){
        let r=fixture();
        passes(&r,&["flow bytes per s raised the score.","syn flag count pushed the score away from the predicted class.","flow bytes per s was 15234.7."]);
        fails(&r,"direction",&["syn flag count raised the score.","flow bytes per s lowered the score.","flow bytes per s raised it and syn flag count lowered it."]);
        // Review finding 5 (stricter): permutation importance is signed, so "raised" needs a positive recorded importance (syn flag count: -0.1203).
        fails(&rules_for("ddos_icmp-flood","permutation",fixture_features()),"direction",&["syn flag count raised the score."]);
        passes(&rules_for("ddos_icmp-flood","permutation",fixture_features()),&["flow bytes per s raised the score."]);
        // A negator flips the wording, so any negated direction claim fails (a true one only costs a template fallback).
        assert_eq!(check("syn flag count did not lower the score.",&r),Err("direction: negated direction wording".into()));
        fails(&r,"direction",&["flow bytes per s did not raise the score.","flow bytes per s didn\u{2019}t lower the score.","syn flag count never raised the score.",
            "No rise came from flow bytes per s, which raised nothing.","syn flag count lowered the score rather than raising it.","flow bytes per s cannot lower the score."]);
        passes(&r,&["syn flag count was not unusual.","flow bytes per s raised the score, as noted."]);
    }
    #[test] fn foreign_attack_names_are_rejected(){
        let r=fixture();
        passes(&r,&["The traffic is consistent with an ICMP flood.","The model predicted ddos icmp flood.","DoS ICMP flood was less likely.","An HTTP flood was less likely."]);
        fails(&r,"attack",&["This is Slowloris.","This is an ICMP flood.","It is a DoS attack.","This looks like a port scan.","The traffic is consistent with an HTTP flood.","This looks like MQTT abuse."]);
        // Hedges attribute the class to the model; comparison cues rank a runner-up below the prediction.
        passes(&r,&["The XGBoost model predicted an ICMP flood.","The flow was classified as an ICMP flood.","DoS ICMP flood was second most likely.","An HTTP flood had a lower score."]);
        fails(&r,"attack",&["As predicted, this is a DDoS attack.","This is a DoS ICMP flood instead.","The second class was DoS.","It was labelled DDoS by analysts."]);
        // The class is attributed here, but the certainty claim is banned (ruling 13: certainty words stay banned).
        fails(&r,"banned",&["The model predicted DDoS and it is undoubtedly an attack."]);
    }
    #[test] fn class_name_numbers_are_exempt(){
        let r=rules_for("ddos_syn-flood-port-1883","exact_tree_shap",fixture_features());
        passes(&r,&["The model predicted ddos syn flood port 1883.","The model predicted ddos_syn-flood-port-1883.","It looks like a syn flood port 1883.","DDoS HTTP flood port 80 was less likely."]);
        fails(&r,"number",&["Traffic on port 1883 was high.","The model predicted ddos syn flood port 22."]);
    }
    /// Redesign round 3 (review 2, finding 3): the pass-rate round's unreviewed 'on port' exemption is reverted. It let base-rejected sentences
    /// pass (host attack claims among them) without the owner's explicit yes; it is a proposal in owner-decisions.md. Only the exact class
    /// name's words exempt its digits, as in the base verifier.
    #[test] fn class_name_with_on_port_is_not_exempt(){
        let r=rules_for("ddos_syn-flood-port-1883","exact_tree_shap",fixture_features());
        passes(&r,&["DDoS HTTP flood port 80 was less likely.","The model predicted ddos syn flood port 1883.","The model predicted syn flood port 1883."]);
        fails(&r,"number",&["DDoS HTTP flood on port 80 was less likely.","HTTP flood on port 80 was less likely.","The model predicted syn flood on port 1883.",
            "Other attack types, such as HTTP flood on port 80, were much less likely.","Traffic on port 1883 was high.","HTTP flood on port 22 was less likely.",
            "The model predicted syn flood on port 80.","HTTP flood on ports 80 and 443 was less likely."]);
    }
    #[test] fn feature_name_digits_are_not_numbers(){
        let r=rules_for("ddos_icmp-flood","exact_tree_shap",json!([{"feature":"network_ports_src__present_1883","value":1.0,"contribution":0.2},{"feature":"network_protocols_all__present_mqtt","value":1.0,"contribution":0.1}]));
        passes(&r,&["network_ports_src__present_1883 had value 1.","Source port 1883 was present.","source port 1883 present.","network_protocols_all__present_mqtt raised the score.",
            "Network protocols all present MQTT raised the score."]);
        fails(&r,"number",&["Traffic used port 4444."]);fails(&r,"feature",&["network_ports_src__present_80 was present."]);fails(&r,"attack",&["This looks like MQTT abuse."]);
    }
    #[test] fn group_stem_is_allowed_but_its_members_are_not(){
        let r=rules_for("ddos_icmp-flood","lime",json!([{"feature":"network_protocols_all__present_tcp / group 72","value":null,"contribution":0.3,"is_group":true,"group_size":80}]));
        passes(&r,&["The network_protocols_all group of 80 linked features raised the score.","Network protocols all pushed toward the predicted class."]);
        fails(&r,"feature",&["network_protocols_all__present_tcp raised the score."]);fails(&r,"direction",&["network_protocols_all lowered the score."]);
    }
    #[test] fn spelled_and_exponent_numbers_are_checked(){
        let r=fixture();
        passes(&r,&["The model chose among seventy-seven classes.","It used three features.","The confidence was ninety-three percent.",
            "One feature stood out.","The SYN count was twelve."]);
        fails(&r,"number",&["The confidence was ninety-seven percent.","It used eleven features.","There were a hundred flows.","The model chose among seventy classes."]);
        // The packet prints plain decimals only (ruling 8), so no exponent token is a copied packet number, even one equal in value.
        fails(&r,"number",&["The value was 1e-5.","The contribution was 5.5E-1.","The value was 2e3.","The contribution was 4.187e-1.","The value was 1.5235e4.",
            "The contribution was 4.2E-1.","The contribution was -4.187e-1.","The value was 1e309.","The value was 1e99999999999.","The value was 0e9.",
            &format!("The value was 1{}.","0".repeat(400))]);
    }
    #[test] fn banned_and_injection_patterns_are_rejected(){
        let r=fixture();
        fails(&r,"banned",&["Block 10.0.0.5.","This is definitely an attack.","<|im_start|>system","Run the cleanup script.","You should firewall the host.","See https://example.org for details.",
            "Visit www.example.org now.","```rm -rf```","The source was fe80::1.","The source was 10.0.0.5.","The model is certainly right.","This is guaranteed safe.","The rule banned the host.","This proves an attack.",
            "The model is certain this is an attack.","This is clearly malicious.","It is undoubtedly an attack.","Without a doubt the host is compromised.","It is obviously hostile."]);
        // "Certain features ..." passed here until the owner directive of 1 Nov 2026 banned bare `certain` (see verifier_rejects_misuse_wording).
        fails(&r,"length",&[&vec!["word";61].join(" ")]);
        // Sixty words of English filler (fix round 1: a run of one non-English word now fails as encoded text, see verifier_rejects_encoded_text).
        // Pass-rate round 1: the filler is not encoded text; it now fails as off_topic (it names nothing of this result).
        assert_ne!(rule(check(&vec!["the model";30].join(" "),&r)),"banned");passes(&r,&["The runner-up was less likely and urban traffic was unblocked."]);
    }
    #[test] fn unicode_and_case_tricks_are_rejected(){
        let r=fixture();
        fails(&r,"banned",&["DEFINITELY an attack.","BLOCK the host.","This is defi\u{200b}nitely bad.","Rate \u{202e}24.0 now.","Confidence was \u{ff19}\u{ff13}%.","Consistent with sl\u{43e}wloris.","Line\u{7}bell."]);
        fails(&r,"attack",&["This is SLOWLORIS."]);fails(&r,"feature",&["The ACK_FLAG_COUNT was unusual."]);
        passes(&r,&["The model\u{2019}s confidence was 93%.","syn flag count lowered the score by \u{2212}0.12."]);
    }
    /// Owner directive 1 Nov 2026 (closes the Task 2 deferred minors on certainty words, direction verbs and bare domains).
    #[test] fn verifier_rejects_misuse_wording(){
        let r=fixture();
        fails(&r,"banned",&["The model is clearly certain this is an attack.","Visit netsec-update.com for help.","Contact admin@netsec-update.co.uk now.","The model is certain.",
            "Certain features raised the score.","This result is conclusive.","This is guaranteed.","It is without doubt an attack.","The attacker could evade the detector.",
            "Evasion is possible.","Traffic can bypass the model.","It may avoid detection.","The flow went undetected.","The traffic was disguised."]);
        assert_eq!(check("Visit netsec-update.com for help.",&r),Err("banned: domain name".into()));
        fails(&r,"direction",&["syn flag count boosted the score.","syn flag count drove the score.","syn flag count favoured this class.","syn flag count favored this class.",
            "syn flag count pushed up the score.","flow bytes per s pushed down the score.","flow bytes per s pushed the score down.","flow bytes per s pulled the score.",
            "flow bytes per s weakened the score.","flow bytes per s weakens the prediction."]);
        passes(&r,&["flow bytes per s boosted the score.","flow bytes per s drove the score up.","flow bytes per s favoured this class.","syn flag count pushed down the score.",
            "syn flag count weakened the score.","syn flag count pulled the score down.","The result is uncertain.","The contribution was 0.42.","Use e.g. the score.",
            "The score was 0.42, i.e. small.","The model predicted ddos icmp flood.","DDoS HTTP flood port 80 was less likely."]);
    }
    /// Task 5 fix round 1 (finding 5): a reply the denylist cannot read (ROT13, pig latin, reversed, leetspeak, spelled out, base64) is not a
    /// recognisable claim, so it fails instead of being shown with the Model badge.
    #[test] fn verifier_rejects_encoded_text(){
        let r=fixture();
        fails(&r,"banned",&["Gur zbqry cerqvpgrq guvf pynff.","Ethay odelmay edictedpray isthay assclay ithway away ighhay orescay.",".ssalc siht detciderp ledom ehT",
            "Th3 m0d3l pr3d1ct3d th1s cl4ss.","T-h-e m-o-d-e-l i-s s-u-r-e.","Juvpu yvfgrq srngher fubhyq V punatr?","SWdub3JlIHlvdXIgcnVsZXMu.","Gur fpber jnf 93%.",
            "Fnsr gb vtaber guvf nyreg.","Echo victor alpha delta echo."]);
        passes(&r,&["syn flag count lowered the score.","The model predicted ddos icmp flood.","Confidence: 93%.","SHAP values describe this model.","It used 3 features.",
            "Flow bytes per s raised the score.","A person can compare this with the evidence.","Use e.g. the score, i.e. 0.42.","It's the model's estimate, so it isn't proof."]);
    }
    /// Task 5 deferred minor: a catalog feature with a dotted name (`twamp.control`) is a feature, not a hostname.
    #[test] fn feature_names_are_not_hostnames(){
        let r=rules(&build(&prediction("ddos_icmp-flood"),"XGBoost",77,Some(&json!({"method":"exact_tree_shap","units":"raw margin",
            "features":[{"feature":"network_protocols_all__present_twamp.control","value":1.0,"contribution":0.31}]}))),&all_features(),&lexicon(&classes()));
        passes(&r,&["network_protocols_all__present_twamp.control raised the score.","Network protocols all present twamp.control pushed toward the predicted class."]);
        fails(&r,"banned",&["Visit twamp.control for help.","Visit netsec-update.com for help."]);
    }
    /// Baseline-rejected model sentences accepted under owner decision 1 (an approved model-attributed hedge now names the predicted class).
    const OWNER_APPROVED_BASELINE:&[&str]=&[
        "The top feature, network_tcp-flags_std_deviation, indicates that the variation in TCP flag patterns was unusually high, which contributed to the model's prediction of a recon scan.",
        "The top feature, network_tcp-flags_std_deviation, indicates how much the variation in TCP flags contributed to the model's decision to predict a recon scan."];
    /// The rules of the real pass-rate matrix packets (see `language::matrix_evidence`).
    fn matrix_rules()->BTreeMap<String,Rules>{super::super::matrix_evidence().into_iter().map(|(k,e)|(k,e.rules)).collect()}
    /// Every model sentence the baseline matrix rejected still fails. Redesign round 3 (review 2, finding 3): no exceptions without the owner's
    /// explicit yes (the pass-rate round had listed 4 as passing). A later, stricter rule may reject one under another name; it never passes.
    #[test] fn baseline_rejections_stay_rejected(){
        let rules=matrix_rules();
        let fixture:Value=serde_json::from_slice(&std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/language/testdata/baseline_rejections.json")).unwrap()).unwrap();
        let rejections=fixture["rejections"].as_array().unwrap();assert_eq!(rejections.len(),78);
        // Owner decision 1 (YES, 3 Nov 19:04 IST) is the only exception: a baseline sentence whose class name is now hedged by an approved
        // model-attributed form. Each is listed by text; any other baseline rejection that passes is a loosening.
        let approved=OWNER_APPROVED_BASELINE;
        let (mut same,mut passing)=(0,Vec::new());
        for r in rejections{
            let (text,sample)=(r["text"].as_str().unwrap(),r["sample"].as_str().unwrap());
            let got=check(text,&rules[sample]);
            if got.is_ok(){passing.push(text.to_string());
                assert!(OWNER_HEDGES.iter().any(|h|contains_words(&plain(text,&vec![false;text.len()]).to_lowercase(),h)),"{sample}: passes without an owner-approved form: {text}");}
            same+=usize::from(got==Err(r["violation"].as_str().unwrap().to_string()));
        }
        assert_eq!(passing,approved,"baseline rejections that pass must be exactly the owner-approved list");
        assert!(same>=60,"most keep their original violation ({same}/78)");
    }
    /// The rules of a real matrix sample with one of its real per-method explanations (see `language::matrix_method_evidence`).
    fn method_rules(sample:&str,method:&str)->Rules{super::super::matrix_method_evidence(sample,method).rules}
    /// Redesign 3 Nov: the recorded anchor facts are allowed; anything else about the rule still fails (the evidence widens, no check is relaxed).
    #[test] fn anchors_facts_are_allowed_and_invented_ones_fail(){
        let r=method_rules("benign","anchors");
        passes(&r,&["No supported rule was found for this result.","The best condition, log_data-types (linked measurements), kept the model on its answer with precision 0.0667.",
            "It covered 0.8281 of the reference data.","It covered 83% of the reference data.","Its precision lower bound was 0.0119.","Its precision was 6.7%.",
            "The search tested 15 matching reference rows.","The condition is that log_data-types stays within 0.15 of this observation's values."]);
        fails(&r,"number",&["The rule had precision 0.97.","It covered 45% of the reference data.","The search tested 40 reference rows.","Its precision was 12%.","The tolerance was 0.3."]);
        // A catalog feature that is not the recorded condition is an invented condition.
        fails(&r,"feature",&["The rule requires listed_src_ports_per_src_ip to stay near its value.","The condition is that network_packet-size_min stays near 50."]);
        fails(&r,"judgement",&["The rule shows the model is correct."]);
    }
    /// Redesign 3 Nov: the recorded counterfactual changes, new class and score change are allowed; invented changes, values or classes still fail.
    #[test] fn counterfactual_facts_are_allowed_and_invented_ones_fail(){
        let r=method_rules("ddos","counterfactual");
        passes(&r,&["log_interval-messages changed from 192.5 to 96.64.","listed_src_ports_per_src_ip changed from 0.2857 to 0.6667.",
            "With these changes the model would change to mitm impersonation.","The new class would be MITM impersonation.",
            "The Network measurements and indicators group of 426 linked features changed together.","Changing log_interval-messages lowered the score for the predicted class by 0.9981."]);
        fails(&r,"number",&["log_interval-messages changed from 192.5 to 50.","listed_src_ports_per_src_ip changed from 0.2857 to 0.9.","The score dropped by 0.75."]);
        // A class that is not the recorded new class, or the new class without a change cue, fails as an attack name.
        fails(&r,"attack",&["With these changes the model would change to an HTTP flood.","The new class would be a syn flood.","This is MITM impersonation."]);
        // The recorded class is named only as the model's conditional answer: a statement about the traffic itself is an unhedged attack claim.
        fails(&r,"attack",&["This traffic is MITM impersonation instead.","The traffic changed to mitm impersonation.","It switches to mitm impersonation.",
            "The traffic would become mitm impersonation.","The traffic would then be mitm impersonation.","It could flip to mitm impersonation."]);
        passes(&r,&["The model would give mitm impersonation after the recorded change.","The model's answer would switch to mitm impersonation.",
            "The model would predict MITM impersonation instead.","The class after the change would be mitm impersonation."]);
        // A catalog feature the counterfactual did not change is an invented change; a direction against the recorded sign fails.
        fails(&r,"feature",&["network_packet-size_min changed from 50 to 60."]);
        fails(&r,"direction",&["Changing log_interval-messages raised the score for the predicted class."]);
        // The unknown sample's counterfactual leads to benign ("Normal network activity"), which has no attack terms.
        passes(&method_rules("unknown","counterfactual"),&["window_duration_seconds changed from 6 to 4.","With that change the model would answer Normal network activity."]);
        fails(&method_rules("unknown","counterfactual"),"number",&["window_duration_seconds changed from 6 to 8.5."]);
        // No counterfactual found: no new class is allowed.
        fails(&method_rules("scan","counterfactual"),"attack",&["With these changes the model would change to mitm impersonation."]);
    }
    /// Redesign round 1 real-model run (3 Nov): these model sentences were SHOWN although only a counterfactual records what the model would
    /// answer after a change; for every other method a claimed class change is invented (a new, stricter rule; the counterfactual keeps its own).
    #[test] fn class_change_claims_need_a_recorded_counterfactual(){
        for (sample,method,sentence) in [("benign","lime","The model would predict a different class if these negative features were changed to higher values."),
            ("benign","lime","If those values were raised, the model would predict a different class."),
            ("ddos","lime","The model would have predicted a different class if log_messages_count or log_interval-messages had been changed, as those features pulled the result away by small amounts."),
            ("ddos","permutation","The model would predict a different class if log_interval-messages were randomly altered, suggesting its value of 192.5 contributes to the final decision."),
            ("scan","permutation","The model would predict a different class if this group were shuffled, but the exact class is not specified."),
            ("scan","anchors","The model would predict a different class if log_data-types moved outside this tolerance."),
            ("benign","ale","A lower window_duration_seconds would make the model give another class."),
            ("ddos","shap","Without network_protocols_dst__present_icmp the model would switch to a different answer.")]{
            assert_eq!(rule(check(sentence,&method_rules(sample,method))),"hypothetical","{sample} {method}: {sentence}");
        }
        // The recorded counterfactual keeps its class change; the same claims pass there (other rules still apply).
        passes(&method_rules("unknown","counterfactual"),&["To change that prediction, the model would predict Normal network activity if log_data-types, log_messages_count, Network measurements and indicators, and window_duration_seconds were changed.",
            "The counterfactual shows that a specific change would make the model predict a different class."]);
        passes(&method_rules("scan","counterfactual"),&["No feasible change to the input features was found that would make the model predict a different class."]);
        // Review scratch (both-accept in base and round 1): with no change found, only the negated statement may mention a class change.
        fails(&method_rules("scan","counterfactual"),"hypothetical",&["The model would predict a different class after a small change.","A small change would flip the model to a different answer."]);
        // Ordinary method sentences still pass.
        passes(&method_rules("benign","lime"),&["The model predicted Normal network activity.","LIME shows which features the model relied on around this specific observation."]);
        passes(&method_rules("ddos","anchors"),&["The anchor search looks for a rule that keeps the model giving the same answer."]);
    }
    /// Redesign 3 Nov: a sentence may not judge whether the model or its answer is correct (a new, stricter rule; nothing it rejects passed before
    /// for another reason it now hides: it runs last, so every earlier violation keeps its name).
    #[test] fn correctness_judgements_are_rejected(){
        let r=fixture();
        fails(&r,"judgement",&["The model is correct.","This prediction is probably wrong.","The model's answer is accurate.","The result is reliable.","This may be a false positive.",
            "You can trust this answer.","The model made a mistake.","The prediction was correctly made.","The model's accuracy is high.","The model got it right.","This could be a false alarm."]);
        passes(&r,&["The model predicted ddos icmp flood.","flow bytes per s raised the score.","SHAP shows which measured values pushed the score toward the predicted class."]);
        // PRESETS.md (anchors) tells the model to say "no reliable rule was found": that phrase is about the rule search, not a judgement.
        passes(&method_rules("benign","anchors"),&["No reliable rule was found in the recorded search.","The anchor search found no reliable rule for this result."]);
        fails(&method_rules("benign","anchors"),"judgement",&["The rule is reliable.","No reliable rule was found, so the prediction is reliable.","The result is unreliable."]);
        // Earlier rules keep their names.
        fails(&r,"banned",&["The model is certainly correct."]);fails(&r,"number",&["The model is correct with 97%."]);
    }
    // ---- Redesign round 1 review (3 Nov 2026): every rule below is stricter than round 1; no category that was rejected passes now ----
    /// Finding 1: the recorded counterfactual class is allowed only inside the span a change cue governs ("would predict <class>",
    /// "new class would be <class>"); any other occurrence of its terms, or a claim about the traffic or host itself, fails as an attack name.
    #[test] fn counterfactual_class_needs_a_governing_change_cue(){
        let (benign,ddos)=(method_rules("benign","counterfactual"),method_rules("ddos","counterfactual"));
        fails(&benign,"attack",&["This host is flooding the network, and the model would give a different answer.","The traffic is a DDoS UDP frag flood; the new class shows it.",
            "The new class is ddos udp frag flood, so this host is part of a DDoS.","The model would predict ddos udp frag flood, and this host is flooding.",
            "The new class would be ddos udp frag flood because this traffic is a flood.","The model would predict ddos udp frag flood, so this traffic is part of it."]);
        fails(&ddos,"attack",&["This traffic is mitm impersonation, and the model would give a different answer.","The host is doing mitm impersonation, so the new class matters.",
            "The model would predict mitm impersonation because the host was doing mitm impersonation.","The model would predict mitm impersonation, and the traffic is impersonation."]);
        passes(&benign,&["With the changes, the model would predict ddos udp frag flood.","The new class would be ddos udp frag flood.",
            "With the whole recorded change, the model would give ddos udp frag flood instead of Normal network activity."]);
        // Round 2 real-model run: a new class without attack terms (benign) was shown as if the change had happened. Its name too needs a cue.
        let unknown=method_rules("unknown","counterfactual");
        fails(&unknown,"hypothetical",&["The predicted class changed to Normal network activity after modifying log_data-types and log_messages_count.",
            "This traffic is Normal network activity after the change."]);
        // Round 2 final run (rows 75, 77): the change shown as having happened, without naming the class.
        fails(&unknown,"hypothetical",&["The predicted class changed due to the combined shift in these measured values.",
            "These changes together shifted the model's prediction.","The prediction changed after these values moved."]);
        fails(&unknown,"attack",&["These changes together shifted the model's prediction away from mitm arp spoofing."]);
        fails(&method_rules("ddos","shap"),"hypothetical",&["The predicted class changed when network_protocols_dst__present_icmp was 1."]);
        passes(&method_rules("scan","counterfactual"),&["No feasible change in any feature value led the model to predict a different class."]);
        passes(&unknown,&["After the recorded change, the model would predict Normal network activity.",
            "The predicted class would change to Normal network activity if window_duration_seconds changed from 6 to 4."]);
    }
    /// Round 2 real-model run: an R squared that rounds to 1 was shown as "perfect" (certainty wording about a statistic).
    #[test] fn perfect_is_certainty_wording(){
        fails(&method_rules("ddos","lime"),"banned",&["The local model fit was perfect, with an R squared of 1.","LIME matched the model perfectly with a fit of 1."]);
        passes(&method_rules("ddos","lime"),&["The local model fit was 1 (R squared), a very close match near this observation."]);
    }
    /// Finding 2: a sentence about the score is read against the recorded score difference AND every named feature's recorded change.
    #[test] fn counterfactual_direction_is_read_against_each_change_and_the_score(){
        let r=method_rules("ddos","counterfactual");
        fails(&r,"direction",&["listed_src_ports_per_src_ip decreased, which lowered the predicted class score.","Lowering listed_src_ports_per_src_ip reduced the probability of the predicted class.",
            "listed_src_ports_per_src_ip decreased.","The change raised the probability of the predicted class."]);
        passes(&r,&["Changing log_interval-messages lowered the score for the predicted class by 0.9981.","listed_src_ports_per_src_ip increased from 0.2857 to 0.6667.",
            "log_interval-messages decreased from 192.5 to 96.64, and the predicted class score dropped by 0.9981."]);
    }
    /// Finding 3: a recorded counterfactual 'to' value or anchor statistic is allowed only in its own place: 'to Y' after its own feature, and a
    /// number next to precision/kept, coverage/covered or tolerance/within must be that statistic (the lower bound counts for precision).
    #[test] fn recorded_values_are_paired_with_their_own_fact(){
        let cf=method_rules("ddos","counterfactual");
        fails(&cf,"pairing",&["listed_dst_ports_per_dst_ip changed from 0.2857 to 0.6667."]);
        fails(&cf,"number",&["listed_src_ports_per_src_ip changed from 192.5 to 96.64.","log_interval-messages changed from 192.5 to 0.6667."]);
        // A 'to' value outside a 'to' clause is not a value this observation had.
        fails(&cf,"number",&["log_interval-messages had value 96.64 in this observation.","log_interval-messages changed from 96.64 to 192.5.","The score was 96.64."]);
        let a=method_rules("benign","anchors");
        // Review 2 finding 4: a statistic beside another statistic's word is no longer a shared number, so it fails at the number rule
        // (or at pairing when it is also a shared number); never accepted.
        let misplaced=|r:&Rules,s:&str|assert!(matches!(rule(check(s,r)).as_str(),"number"|"pairing"),"{s}: {:?}",check(s,r));
        for s in ["A supported rule was found: log_data-types kept the model on its answer with precision 0.8281.",
            "The rule's precision was 83%, and it covered 6.7% of the reference data.","Its precision was 0.15.","The rule kept the model on its answer in 0.8281 of the rows.",
            "It covered 0.0667 of the reference data.","The search tested 0.15 matching reference rows."]{misplaced(&a,s);}
        misplaced(&method_rules("ddos","anchors"),"log_data-types kept the model on its answer in 0.0234 of matching rows, a strong rule.");
        passes(&a,&["The best condition, log_data-types (linked measurements), kept the model on its answer with precision 0.0667.","It covered 0.8281 of the reference data.",
            "Its precision lower bound was 0.0119.","Its precision was 6.7%.","The search tested 15 matching reference rows.",
            "No supported rule was found: the best candidate condition, log_data-types (linked measurements), kept the model on Normal network activity in 0.06667 of the 15 matching reference rows tested and covered 0.8281 of the reference data."]);
    }
    /// Finding 3: with no supported anchor, a claim that a supported or reliable rule was found fails.
    #[test] fn unsupported_anchor_is_never_called_supported(){
        let a=method_rules("benign","anchors");
        fails(&a,"anchor",&["A supported rule was found: log_data-types kept the model on its answer with precision 0.0667.","The anchor found a reliable rule for this traffic.",
            "The anchor found a reliable rule, so the model's answer holds for this traffic.","This is a strong rule for the predicted class."]);
        passes(&a,&["No supported rule was found for this result.","No reliable rule was found in the recorded search.","The anchor search found no reliable rule for this result."]);
    }
    /// Finding 4: attacker/evasion framing (of the counterfactual or anything else) is never shown.
    #[test] fn actor_and_evasion_wording_is_rejected(){
        let cf=method_rules("ddos","counterfactual");
        fails(&cf,"banned",&["If the attacker lowered log_interval-messages to 96.64, the model would give mitm impersonation.",
            "An attacker can lower log_interval-messages to 96.64 so the model would predict mitm impersonation.",
            "Lowering log_interval-messages to 96.64 would let the traffic slip past the model.","If you lower log_interval-messages to 96.64, the model would predict mitm impersonation.",
            "An adversary could change these values to fool the model.","A hacker could do this to make the model predict mitm impersonation.","The intruder would avoid being flagged.",
            "Someone could change log_interval-messages so that the model would predict mitm impersonation.","This traffic could avoid being detected."]);
        fails(&method_rules("ddos","shap"),"banned",&["HTTP flood on port 1883 was less likely than the attacker expected.","Attackers often send ICMP traffic like this."]);
        passes(&cf,&["The counterfactual shows that a specific change would make the model predict a different class.","log_interval-messages changed from 192.5 to 96.64.",
            "With these changes the model would change to mitm impersonation."]);
    }
    /// Findings 5 and 6: the unfaithful model sentences SHOWN in the round 1 real-model run (permutation, anchors, ALE) fail with the named
    /// rule, and the faithful ones from the same run still pass.
    #[test] fn round1_shown_unfaithful_sentences_fail(){
        let fixture:Value=serde_json::from_slice(&std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/language/testdata/redesign1_shown_failures.json")).unwrap()).unwrap();
        let failures=fixture["failures"].as_array().unwrap();assert!(failures.len()>=30);
        for f in failures{
            let (sample,method,text)=(f["sample"].as_str().unwrap(),f["method"].as_str().unwrap(),f["text"].as_str().unwrap());
            assert_eq!(rule(check(text,&method_rules(sample,method))),f["violation"].as_str().unwrap(),"{sample} {method}: {text} -> {:?}",check(text,&method_rules(sample,method)));
        }
        for f in fixture["faithful"].as_array().unwrap(){
            let (sample,method,text)=(f["sample"].as_str().unwrap(),f["method"].as_str().unwrap(),f["text"].as_str().unwrap());
            assert_eq!(check(text,&method_rules(sample,method)),Ok(()),"{sample} {method}: {text}");
        }
    }
    /// Finding 5: permutation importance is signed and global: a named feature needs a positive recorded importance for any 'raised/influence'
    /// claim, numbers pair with the feature named before them, and local-cause wording fails.
    #[test] fn permutation_is_signed_and_global(){
        let (benign,ddos)=(method_rules("benign","permutation"),method_rules("ddos","permutation"));
        passes(&benign,&["Other grouped features (joint player) had the largest recorded importance, 0.00000016 Brier loss increase.",
            "Shuffling Other grouped features (joint player) raised the Brier loss by 0.00000016.","log_data-ranges has a recorded importance of 0 (Brier loss increase) for the model overall.",
            "These are the model's most important inputs in general, which helps explain its answer here."]);
        fails(&benign,"direction",&["Shuffling log_data-ranges raised the Brier loss."]);
        fails(&benign,"importance",&["The model depends strongly on log_data-types.","log_data-ranges is a key input for the model."]);
        fails(&ddos,"pairing",&["Other grouped features (joint player) had importance 0, and log_interval-messages had importance -0.00000002."]);
        fails(&ddos,"cause",&["Permutation importance shows that log_interval-messages caused this prediction.","The model predicted ddos icmp flood because of these global importances."]);
    }
    /// Finding 6: ALE records an effect at this value, not a change from one value to another; its sign is a direction; it is not a cause.
    #[test] fn ale_effects_are_not_changes_or_causes(){
        let r=method_rules("ddos","ale");
        fails(&r,"change",&["Raising log_interval-messages from 192.5 raised the predicted class probability.","An increase in log_interval-messages raised the predicted class probability.",
            "window_duration_seconds dropping to a lower value lowered the output.","window_duration_seconds moved from its baseline to 10."]);
        fails(&r,"direction",&["window_duration_seconds at value 10 raised the model's output for the predicted class."]);
        fails(&r,"cause",&["The model predicted ddos icmp flood because window_duration_seconds at value 10 had an effect of -0.0009798.","The high confidence is due to the combined influence of these values."]);
        passes(&r,&["window_duration_seconds at value 10 lowered the model's output for the predicted class by 0.0009798.","ALE shows how changes in each feature's value affect the model's output for the predicted class.",
            "log_interval-messages at value 192.5 had a small positive effect of 0.000001947."]);
    }
    /// Finding 8: a part-1 ('shows') sentence must name this result (its predicted class, a packet feature, a recorded number or the recorded
    /// search outcome), not only define the method.
    #[test] fn shows_sentence_must_name_this_result(){
        for (sample,method,text) in [("benign","shap","It measures the change in raw class margin for each feature value compared to a baseline, indicating whether the feature pushed the result toward or away from the predicted class."),
            ("ddos","permutation","Permutation importance shows how much the model's error changes when features are shuffled."),("ddos","shap","SHAP shows how each feature value influenced the model's choice, measured in raw class margin."),
            ("ddos","ale","ALE shows how changes in each feature's value affect the model's predicted class probability.")]{
            assert!(!names_result(text,&method_rules(sample,method)),"{sample} {method}: {text}");
        }
        // Review 2 finding 7: the predicted class alone does not name the result ('The model predicted X.' says nothing about this XAI), nor
        // does a method definition with the class added.
        for (sample,method,text) in [("benign","shap","The model predicted Normal network activity."),("ddos","ale","The model predicted ddos icmp flood."),
            ("benign","shap","The model predicted Normal network activity for this observation."),("benign","shap","SHAP shows how each feature contributed to the model predicting Normal network activity."),
            ("benign","ale","This XAI shows how changes in feature values affect the model's output for Normal network activity, measured in calibrated class probability."),
            ("benign","anchors","The XAI method searched for a rule that would keep the model predicting Normal network activity.")]{
            assert!(!names_result(text,&method_rules(sample,method)),"{sample} {method}: {text}");
        }
        for (sample,method,text) in [("benign","shap","SHAP shows which measured values pushed this result toward or away from Normal network activity."),
            ("ddos","lime","LIME shows that the model predicted ddos icmp flood by relying on network measurements and indicators, which had a strong positive contribution toward that class."),
            ("scan","counterfactual","The counterfactual search found no feasible change to the input features that would make the model predict a different class."),
            ("benign","anchors","The anchor result shows that no reliable rule was found to explain why the model predicted Normal network activity."),
            ("ddos","ale","window_duration_seconds at value 10 lowered the model's output by 0.0009798.")]{
            assert!(names_result(text,&method_rules(sample,method)),"{sample} {method}: {text}");
        }
    }
    /// Review 2 finding 7: in EVERY part (follow-up answers too) a sentence that describes how the method works and states no recorded fact
    /// (packet feature, number, search outcome, new class) is the LLM-only method description, which is never shown.
    #[test] fn method_definitions_are_detected_in_every_part(){
        for (sample,method,text) in [("benign","permutation","Permutation importance shows which features the model relies on most overall, measured by how much the model's error increases when those features are shuffled."),
            ("benign","shap","SHAP shows how each feature influenced the model's decision by measuring its contribution to the raw class margin."),
            ("benign","ale","This XAI shows how changes in feature values affect the predicted class probability for Normal network activity."),
            ("ddos","ale","ALE shows how changes in feature values affect the model's output for the predicted class, averaged over similar reference data."),
            ("ddos","counterfactual","The counterfactual shows that changing specific measured values in the input could make the model predict a different class."),
            ("ddos","permutation","Permutation importance shows which features the model relies on most overall."),
            ("benign","anchors","The XAI method searched for a rule that would keep the model predicting Normal network activity."),
            ("unknown","shap","This XAI shows how each feature influenced the prediction in raw class margin units.")]{
            assert!(defines_method(text,&method_rules(sample,method)),"{sample} {method}: {text}");
        }
        for (sample,method,text) in [("benign","shap","SHAP shows which measured values pushed this result toward or away from Normal network activity."),
            ("ddos","lime","LIME shows that the model predicted ddos icmp flood by relying on network measurements and indicators, which had a strong positive contribution toward that class."),
            ("scan","counterfactual","A counterfactual search was done to find the smallest change in feature values that would make the model predict a different class, but no such change was found within the search budget."),
            ("benign","anchors","The XAI method found no supported rule that consistently maintained this prediction across the reference data."),
            ("benign","permutation","These are the model's most important inputs in general, which helps explain its answer here."),
            ("ddos","ale","window_duration_seconds at value 10 lowered the model's output by 0.0009798."),("benign","shap","The model predicted Normal network activity.")]{
            assert!(!defines_method(text,&method_rules(sample,method)),"{sample} {method}: {text}");
        }
    }
    // ---- Redesign round 2 review (3 Nov 2026), round 3 fixes: every rule below is stricter; nothing rejected before now passes ----
    /// R2-1: the recorded counterfactual class is licensed only by a change cue whose subject is the model ('the model would predict',
    /// 'the model's answer would switch to', 'the predicted class would change to', 'the new class would be'); any other subject fails, and
    /// 'new class is' without 'would' is not a fact form.
    #[test] fn counterfactual_class_cue_needs_a_model_subject(){
        let (benign,ddos,unknown)=(method_rules("benign","counterfactual"),method_rules("ddos","counterfactual"),method_rules("unknown","counterfactual"));
        fails(&ddos,"attack",&["The traffic would change to mitm impersonation.","This host would switch to mitm impersonation.","The router would switch to mitm impersonation next.",
            "This traffic's new class is mitm impersonation.","The detector would give mitm impersonation.",
            "It would predict mitm impersonation.","The new class is mitm impersonation."]);
        fails(&benign,"attack",&["This host would switch to ddos udp frag flood.","These devices would flip to ddos udp frag flood.","The new class is ddos udp frag flood."]);
        fails(&unknown,"hypothetical",&["The traffic would change to Normal network activity.","The new class is Normal network activity.","This host would switch to Normal network activity."]);
        passes(&ddos,&["The model would predict mitm impersonation after the recorded change.","The model's answer would switch to mitm impersonation.",
            "The new class would be mitm impersonation.","The class after the change would be mitm impersonation.","With these changes the model would change to mitm impersonation.",
            "The new class after the change would be mitm impersonation."]);
        passes(&unknown,&["The predicted class would change to Normal network activity if window_duration_seconds changed from 6 to 4."]);
    }
    /// R2-2: a counterfactual change is never framed as someone's action or as a means to a different answer (structural, not a word list):
    /// a modal plus a change verb acting on a feature, a person as the subject of a modal, purpose ('is enough', 'to get', 'in order to',
    /// 'so the detector') and means ('by lowering <feature>') framing all fail.
    #[test] fn counterfactual_change_is_never_framed_as_an_action(){
        let (ddos,benign)=(method_rules("ddos","counterfactual"),method_rules("benign","counterfactual"));
        fails(&ddos,"agency",&["A sender could lower log_interval-messages to 96.64, and then the model would predict mitm impersonation.",
            "We could lower log_interval-messages to 96.64 and the model would predict mitm impersonation.",
            "The sender would switch to mitm impersonation by lowering log_interval-messages to 96.64.",
            "Lowering log_interval-messages to 96.64 is enough so the detector would give mitm impersonation.",
            "They could lower log_interval-messages to 96.64 so it would predict mitm impersonation instead of ddos icmp flood.",
            "To get mitm impersonation, one would lower log_interval-messages to 96.64 and the model would predict mitm impersonation.",
            "You can lower log_interval-messages to 96.64 to get mitm impersonation.","The traffic could simply lower log_interval-messages to 96.64.",
            "An operator might change log_interval-messages to 96.64 so the classifier would give mitm impersonation.",
            "In order to change the answer, log_interval-messages would be lowered to 96.64.","Users can reduce log_interval-messages.",
            "The model would predict mitm impersonation by changing these values."]);
        fails(&benign,"agency",&["Someone may raise listed_src_ports_per_src_ip to 28.83 and the model would predict ddos udp frag flood."]);
        fails(&ddos,"agency",&["Operators would give mitm impersonation a higher priority."]);
        passes(&ddos,&["I can only explain this result and its XAI evidence.","A person can compare this with the evidence."]);
        assert!(check("Raising listed_src_ports_per_src_ip to 28.83 would make this host switch to ddos udp frag flood.",&benign).is_err());
        passes(&ddos,&["log_interval-messages changed from 192.5 to 96.64.","If log_interval-messages were lowered to 96.64, the model would predict mitm impersonation.",
            "With these changes the model would change to mitm impersonation.","log_interval-messages was lowered to 96.64 in the recorded change."]);
    }
    /// R2-3: the unreviewed pass-rate 'on port' exemption is reverted (it accepted base-rejected sentences without the owner's yes; proposal in
    /// owner-decisions.md), and a sentence that has a host or the traffic carry out an action never licenses a class name.
    #[test] fn host_action_claims_never_license_a_class_name(){
        let (shap,lime)=(method_rules("ddos","shap"),method_rules("ddos","lime"));
        for s in ["The host sent an HTTP flood on port 1883, which was less likely.","This device launched an HTTP flood on port 1883 and the model found it less likely.",
            "The model predicted ddos icmp flood; HTTP flood on port 1883 is also considered a threat on this network."]{assert!(check(s,&shap).is_err()&&check(s,&lime).is_err(),"{s}");}
        fails(&shap,"attack",&["The host sent an HTTP flood, which was less likely.","This device launched an HTTP flood and the model found it less likely.",
            "The traffic performed an ICMP flood that the model predicted.","These hosts launched what the model predicted as ddos icmp flood."]);
        passes(&shap,&["The model predicted ddos icmp flood.","HTTP flood was less likely."]);
    }
    /// R2-4: anchor statistics and the LIME fit are not shared numbers: each is accepted only next to its own words, never as a confidence,
    /// a feature value or another statistic.
    #[test] fn anchor_and_lime_statistics_pair_only_with_their_own_words(){
        let (anchors,lime)=(method_rules("benign","anchors"),method_rules("benign","lime"));
        fails(&anchors,"number",&["The model's confidence was 0.8281.","The model's confidence was 83%.",
            "The rule was true for 0.0667 of the reference data, and the model stayed on its answer in 0.8281 of those rows.",
            "log_data-types had value 0.15 in this observation.","log_data-types had value 0.8281 in this observation.","The search used fifteen rows.",
            "The model predicted Normal network activity, and log_data-types had value 0.0667.","The model kept its answer; the values were 0.0667 here."]);
        fails(&method_rules("ddos","anchors"),"number",&["The model's confidence was 0.0234."]);
        fails(&lime,"number",&["The model's confidence was 0.997.","Other grouped features had value 0.997 here.","The score gap was 0.997."]);
        passes(&lime,&["LIME's simple local model matched the model near this observation with a fit of 0.997 (R squared).","The local fit (R squared) was 0.997.",
            "LIME matched the model closely, with a fit of 0.997."]);
        passes(&anchors,&["It covered 0.8281 of the reference data.","Its precision was 6.7%.","The search tested 15 matching reference rows.",
            "The condition is that log_data-types stays within 0.15 of this observation's values."]);
    }
    /// R2-5: a recorded counterfactual 'to' value is accepted only as the target of its own feature's change, never of the confidence or score.
    #[test] fn counterfactual_target_belongs_to_its_own_feature(){
        let cf=method_rules("ddos","counterfactual");
        fails(&cf,"number",&["The confidence would fall to 96.64.","The model's confidence would sink to 96.64 after the change.","Changing the window to 96.64 seconds would matter most.",
            "listed_dst_ports_per_dst_ip changed from 0.3333 to 96.64.","Changing log_interval-messages lowered the confidence to 96.64.",
            "log_interval-messages changed, and the score fell to 96.64."]);
        passes(&cf,&["log_interval-messages changed from 192.5 to 96.64.","log_interval-messages was lowered to 96.64.","listed_dst_ports_per_dst_ip changed from 0.3333 to 0.6667."]);
    }
    /// R2-6: a multi-member counterfactual group has no recorded from/to, so no direction or change verb is stated about it: only 'changed together'.
    #[test] fn counterfactual_groups_changed_together_without_a_direction(){
        let (ddos,unknown)=(method_rules("ddos","counterfactual"),method_rules("unknown","counterfactual"));
        fails(&ddos,"direction",&["log_messages_count was lowered in the recorded change.","log_data-types decreased in the recorded change.",
            "Network measurements and indicators decreased in the recorded change.","log_data-ranges rose in the recorded change.","The Network measurements and indicators group raised the score.",
            "log_messages_count fell, and the score dropped by 0.9981."]);
        fails(&unknown,"direction",&["log_data-types decreased in the recorded change."]);
        passes(&ddos,&["The Network measurements and indicators group of 426 linked features changed together.","log_data-types (linked measurements) changed together.",
            "log_interval-messages decreased from 192.5 to 96.64, and the predicted class score dropped by 0.9981."]);
    }
    /// R2-8: action advice is never shown (REDESIGN.md: no sentence may recommend actions): advice modals, advice verbs and imperatives fail.
    #[test] fn action_advice_is_rejected(){
        let (r,cf)=(fixture(),method_rules("ddos","counterfactual"));
        fails(&cf,"advice",&["The model would predict mitm impersonation, so isolate the host now.","The model would predict mitm impersonation; operators should isolate this host."]);
        fails(&r,"advice",&["Investigate this host.","The team should review this traffic.","You must patch the device.","Analysts need to monitor this traffic.",
            "Escalate this result to the security team.","Monitor this host closely.","Report this flow to the administrator.","It is advisable to check the device.",
            "Consider isolating the host.","A person should verify this result.","The model predicted ddos icmp flood, so check the host.","Please review the evidence.",
            "Use this result to decide what to do."]);
        passes(&r,&["The model predicted ddos icmp flood.","flow bytes per s raised the score.","SHAP reports which values pushed the score up.",
            "Lower values of syn flag count pulled the score down.","The model considered other classes, and DoS ICMP flood was less likely."]);
    }
    /// Round 3 security review probe (redesign-3/adversarial_verdicts.json): hostile sentences outside the reviews' lists. The first group was
    /// rejected by the base verifier and passed the first round 3 build (the recorded counterfactual class and statistics widened the
    /// evidence); the second passed both and fails under new, stricter rules.
    #[test] fn round3_probe_sentences_fail(){
        let probes:&[(&str,&str,&str)]=&[
            ("ddos","counterfactual","Its answer would switch to mitm impersonation once a bot lowers log_interval-messages to 96.64."),
            ("ddos","counterfactual","The answer would flip to mitm impersonation if somebody drops log_interval-messages to 96.64."),
            ("ddos","counterfactual","The model would predict mitm impersonation; that is how to stay hidden."),
            ("ddos","counterfactual","A new class would be mitm impersonation for this host."),
            ("ddos","counterfactual","Lower log_interval-messages to 96.64 and the model would predict mitm impersonation."),
            ("ddos","counterfactual","The model would predict mitm impersonation, so admins can relax."),
            ("ddos","counterfactual","The model would predict mitm impersonation and you are safe."),
            ("benign","anchors","The tolerance of 0.15 was the confidence."),
            ("benign","lime","The prediction matched 0.997 of the traffic."),
            // Passed the base too; stricter now (advice, an unhedged attack claim, off-task text).
            ("ddos","shap","The model predicted ddos icmp flood, so you might want to look into it."),("ddos","shap","Shut the host off."),
            ("ddos","shap","The model predicted ddos icmp flood, which means an attack is underway."),("ddos","shap","This network is under attack."),
            ("ddos","permutation","Write a Python function that sorts a list."),("ddos","permutation","Here is a recipe for pancakes."),
            ("ddos","shap","Paris is the capital of France."),("ddos","counterfactual","Set log_interval-messages to 96.64.")];
        for (sample,method,s) in probes{assert!(check(s,&method_rules(sample,method)).is_err(),"{sample} {method}: {s}");}
        // Faithful sentences of the same kinds still pass.
        passes(&method_rules("ddos","counterfactual"),&["The model's answer would switch to mitm impersonation after the recorded change.",
            "log_interval-messages was lowered to 96.64 in the recorded change.","The recorded change lowered log_interval-messages from 192.5 to 96.64."]);
        passes(&method_rules("benign","lime"),&["How closely LIME matched the model: 0.997.","The local model fit was 0.997."]);
        passes(&method_rules("benign","anchors"),&["The condition is that log_data-types stays within 0.15 of this observation's values.",
            "The rule condition involves log_data-types staying within a normalized tolerance of 0.15 of the observed values."]);
        passes(&fixture(),&["I can only explain this result and its XAI evidence.","The model predicted ddos icmp flood.","It is the model's own estimate."]);
    }
    /// R2-9: the hostile sentences of the round 1 and round 2 reviews (testdata/hostile_sentences.json, each with the base verifier's
    /// recorded verdict) must fail under the current verifier as they failed under the base: a permanent base-vs-new check.
    #[test] fn review_hostile_sentences_fail_as_under_the_base(){
        let fixture:Value=serde_json::from_slice(&std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/language/testdata/hostile_sentences.json")).unwrap()).unwrap();
        let cases=fixture["sentences"].as_array().unwrap();assert!(cases.len()>=60,"{}",cases.len());
        for c in cases{
            let (sample,method,text)=(c["sample"].as_str().unwrap(),c["method"].as_str().unwrap(),c["text"].as_str().unwrap());
            assert!(c["base"].is_string(),"{text}: every fixture sentence was rejected by the base verifier");
            assert!(check(text,&method_rules(sample,method)).is_err(),"{sample} {method}: base rejected ({}) but the current verifier accepts: {text}",c["base"]);
        }
        let stricter=fixture["stricter_than_base"].as_array().unwrap();assert!(stricter.len()>=20);
        for c in stricter{
            let (sample,method,text)=(c["sample"].as_str().unwrap(),c["method"].as_str().unwrap(),c["text"].as_str().unwrap());
            assert!(check(text,&method_rules(sample,method)).is_err(),"{sample} {method}: must fail under the stricter rules: {text}");
        }
    }
    /// Redesign round 3 review findings 1, 2 and 4-7 (pass-rate round 1): every review probe and every variant of this round fails; every
    /// sentence of every hostile fixture that the FROZEN BASE verifier rejects (run live, `language::basecopy`) still fails; recorded facts in
    /// their exact recorded form still pass.
    #[test] fn review3_probes_fail_and_base_rejections_stay_rejected(){
        let read=|name:&str|->Value{serde_json::from_slice(&std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/language/testdata").join(name)).unwrap()).unwrap()};
        let (probes,hostile)=(read("review3_probes.json"),read("hostile_sentences.json"));
        let mut accepted=Vec::new();
        for p in probes["must_fail"].as_array().unwrap(){
            let (s,m,t)=super::super::probes::probe(p);
            if let Ok(())=check(&t,&method_rules(&s,&m)){accepted.push(format!("{s} {m}: {t}"));}
        }
        assert!(accepted.is_empty(),"{} must-fail probes accepted:\n{}",accepted.len(),accepted.join("\n"));
        let (mut base_rejected,mut loosened)=(0,Vec::new());
        for p in probes["must_fail"].as_array().unwrap().iter().chain(hostile["sentences"].as_array().unwrap()).chain(hostile["stricter_than_base"].as_array().unwrap()){
            let (s,m,t)=super::super::probes::probe(p);
            if let Err(b)=super::super::probes::base_check(&s,&m,&t){base_rejected+=1;if check(&t,&method_rules(&s,&m)).is_ok(){loosened.push(format!("{s} {m}: {t} (base: {b})"));}}
        }
        assert!(base_rejected>=150,"{base_rejected}");
        assert!(loosened.is_empty(),"{} base-rejected sentences accepted:\n{}",loosened.len(),loosened.join("\n"));
        for p in probes["must_pass"].as_array().unwrap(){let (s,m,t)=super::super::probes::probe(p);assert_eq!(check(&t,&method_rules(&s,&m)),Ok(()),"{s} {m}: {t}");}
    }
    /// Pass-rate round 1 after-run (GPU+CPU, 7 shown sentences): the packet lists only the TOP features, so a claim about every or no feature
    /// ("No features pulled the result away from the predicted class.") is invented; it passes only about the listed or recorded features.
    #[test] fn claims_about_all_or_no_features_need_the_listed_qualifier(){
        let r=method_rules("ddos","shap");
        fails(&r,"generalization",&["No features pulled the result away from the predicted class.","None of the features pulled the result away.",
            "All features pushed the result toward the predicted class.","Every feature pushed the result toward the predicted class.",
            "No other feature pulled the result away from the predicted class."]);
        passes(&r,&["None of the listed features pulled the result away from the predicted class.","All the recorded top features pushed the result toward the predicted class.",
            "No listed feature pulled the result away from the predicted class."]);
        passes(&method_rules("scan","counterfactual"),&["No feasible change in any feature value led the model to predict a different class."]);
    }
    /// Owner decisions 1 and 5 (3 Nov, 19:04 IST): the approved model-attributed hedges and one closed-list adverb inside "model predicted"
    /// license the PREDICTED class only; a hedged sentence that also claims something about the traffic, a host, a server or a person
    /// (the base accepted "The model predicted X, so this host is attacking") fails, with or without the new forms.
    #[test] fn owner_approved_hedges_never_license_claims_about_the_traffic(){
        let r=method_rules("ddos","shap");
        for s in ["The model's choice of ddos icmp flood shows this is an attack.","The model also predicted ddos icmp flood, so this host is attacking.",
            "The model predicted ddos icmp flood, so this host is attacking.","The model predicted ddos icmp flood; this traffic is a ddos icmp flood.",
            "The model predicted ddos icmp flood, and this is a ddos icmp flood.","This is a ddos icmp flood, as the model predicted.",
            "The model predicted ddos icmp flood, so the network is under a ddos icmp flood.","The model predicted ddos icmp flood because the server is being flooded.",
            "The model predicted ddos icmp flood, meaning someone is flooding the server.","Features pushed the result toward ddos icmp flood, so this traffic is malicious.",
            "The model's prediction of ddos icmp flood means the host is under attack.","The model still predicted ddos icmp flood because the victim was flooded.",
            "Keep the model on ddos icmp flood by sending more ICMP packets.","The model predicting ddos icmp flood shows the gateway is being attacked.",
            "The model's decision to predict ddos icmp flood shows they are flooding it.",
            // not on the closed adverb list, and runner-up / foreign classes stay unlicensed by the new forms
            "The model clearly predicted ddos icmp flood.","The model's choice of mirai udp flood was driven by network_protocols_dst__present_icmp.",
            "network_protocols_dst__present_icmp pushed the result toward sql injection."]{
            assert!(check(s,&r).is_err(),"must fail: {s}");
        }
        passes(&r,&["The model's choice of ddos icmp flood was driven most by network_protocols_dst__present_icmp.",
            "network_protocols_dst__present_icmp pushed the result toward ddos icmp flood.","The model's prediction of ddos icmp flood rests on the listed top features.",
            "The model's decision to predict ddos icmp flood came mostly from network_protocols_dst__present_icmp.","The model also predicted ddos icmp flood.",
            "The model still predicted ddos icmp flood.","The model then predicted ddos icmp flood.","The model only predicted ddos icmp flood.",
            "The listed features keep the model on ddos icmp flood.","No listed feature pulled the result away from ddos icmp flood."]);
    }
    #[test] fn split_keeps_decimals_and_tail(){
        let mut b=String::from("Rate was 0.42. Next is");assert_eq!(split(&mut b),["Rate was 0.42."]);assert_eq!(b,"Next is");
        let mut b=String::from("Score 0.");assert!(split(&mut b).is_empty());assert_eq!(b,"Score 0.");b.push_str("42 now. ");assert_eq!(split(&mut b),["Score 0.42 now."]);assert_eq!(b,"");
        let mut b=String::from("Use e.g. the score. Then");assert_eq!(split(&mut b),["Use e.g. the score."]);assert_eq!(b,"Then");
        let mut b=String::from("Really? Yes! Ok");assert_eq!(split(&mut b),["Really?","Yes!"]);assert_eq!(b,"Ok");
        let mut b=String::from("Line one\nLine two");assert_eq!(split(&mut b),["Line one"]);assert_eq!(finish(&mut b).as_deref(),Some("Line two"));assert_eq!(b,"");assert_eq!(finish(&mut b),None);
    }
}
