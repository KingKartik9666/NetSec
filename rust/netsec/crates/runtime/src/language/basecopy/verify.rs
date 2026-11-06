//! Pure per-sentence verifier for model-written text: every number, feature, direction and attack name must come from the evidence packet. No I/O.
use super::packet::{contains_words,Lexicon,Packet};
use crate::application::display_label;
use std::collections::{BTreeMap,BTreeSet};

/// What one packet allows. `features` are normalized names, longest first; `Some(contribution)` marks a packet feature, `None` a catalog-only one.
/// `percents` (confidence and runner-up scores x100) are the only bases for `%` tokens. `class_names` are the word sequences of the predicted and
/// runner-up class ids/labels (also without the family word) whose digits are exempt from the number rule.
#[derive(Clone,Debug)]
pub struct Rules { numbers: Vec<f64>, percents: Vec<f64>, features: Vec<(String, Option<f64>)>, directional: bool,
    predicted: BTreeSet<String>, runner_ups: BTreeSet<String>, attacks: BTreeSet<String>, class_names: Vec<Vec<String>>,
    /// Words of the catalog feature names, packet features and class names; `encoded` counts them as known words (`source port 1883 present`).
    vocabulary: BTreeSet<String> }
/// Owner directive 1 Nov 2026 added boost/drove/favour/pushed up and pushed down/pulled/weaken; bare `up`/`down` also catch split
/// phrasing such as "pushed the score down". Mixed wording ("pulled it toward") fails as both.
const INCREASE: &[&str] = &["increase", "raise", "push toward", "pushed toward", "support", "higher", "toward", "towards", "boost", "drove", "drive", "favour", "favor",
    "pushed up", "up"];
const DECREASE: &[&str] = &["decrease", "lower", "reduce", "against", "away from", "away", "pushed down", "down", "pull", "weaken"];
/// A negator flips direction wording ("did not raise"); `n't` reads as `n t` in the plain text.
const NEGATORS: [&str; 7] = ["not", "no", "never", "neither", "nor", "without", "cannot"];
/// Hedges include model attribution with the model as subject ("the model predicted", "classified as"); certainty words stay banned.
const HEDGES: [&str; 15] = ["consistent with", "looks like", "look like", "may indicate", "might indicate", "could indicate", "suggest", "typical of",
    "model predicted", "model predicts", "model classified", "model classifies", "classified as", "labelled as", "labeled as"];
/// A runner-up class may be named only when the sentence ranks it below the prediction.
const CUES: [&str; 5] = ["less likely", "runner up", "also considered", "second most likely", "lower score"];
/// Substring patterns: chat-template tokens, code fences, URLs.
const MARKERS: [&str; 11] = ["<|", "|>", "```", "[inst]", "[/inst]", "<<sys>>", "</s>", "<think>", "</think>", "://", "www."];
/// Certainty, action and evasion words, whole words with inflections (`certain` never matches inside `uncertain`).
const BANNED: &[&str] = &["definitely", "certainly", "certain", "confirmed", "proves", "proven", "guarantee", "guaranteed", "undoubtedly", "unquestionably", "indisputably",
    "obviously", "clearly", "conclusive", "conclusively", "without doubt", "without a doubt", "no doubt", "beyond doubt", "for sure",
    "block", "firewall", "execute", "shut down", "shuts down", "shutting down", "shutdown", "recommend", "you should", "quarantine", "disconnect",
    "evade", "evasion", "bypass", "avoid detection", "undetected", "disguise"];
/// Exact words only (`runner`, `urban` and `banner` stay allowed).
const BANNED_EXACT: [&str; 7] = ["ban", "bans", "banned", "banning", "run", "runs", "running"];

fn normalize(name: &str) -> String { name.chars().filter(|c| !matches!(c, '_' | '-' | '/') && !c.is_whitespace()).flat_map(char::to_lowercase).collect() }
fn words(text: &str) -> Vec<String> { text.to_lowercase().split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()).map(String::from).collect() }
fn numeric(word: &str) -> bool { word.bytes().all(|b| b.is_ascii_digit()) }
pub fn rules(packet: &Packet, all_features: &[String], lexicon: &Lexicon) -> Rules {
    let mut names = all_features.iter().map(|f| (normalize(f), None)).collect::<BTreeMap<_, _>>();
    names.extend(packet.features.iter().map(|f| (normalize(&f.name), Some(f.contribution))));
    let mut features = names.into_iter().filter(|(n, _)| n.len() >= 6).collect::<Vec<_>>();
    features.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));
    let terms = |id: &str| lexicon.by_class.get(id).cloned().unwrap_or_default();
    let labels = [packet.class_id.clone(), packet.class_label.clone()].into_iter().chain(packet.runner_ups.iter().flat_map(|(id, _)| [id.clone(), display_label(id)]));
    let mut class_names = labels.map(|l| words(&l)).filter(|w| w.iter().any(|w| numeric(w)))
        .flat_map(|w| { let tail = w[1..].iter().any(|t| t != "port" && !numeric(t)).then(|| w[1..].to_vec()); [Some(w), tail] }).flatten().collect::<Vec<_>>();
    class_names.sort(); class_names.dedup();
    Rules { numbers: packet.numbers(), percents: packet.confidence.into_iter().chain(packet.runner_ups.iter().map(|r| r.1)).map(|e| e * 100.).collect(),
        features, directional: packet.directional, predicted: terms(&packet.class_id), runner_ups: packet.runner_ups.iter().flat_map(|(id, _)| terms(id)).collect(),
        attacks: lexicon.all.clone(), class_names,
        vocabulary: all_features.iter().chain(packet.features.iter().map(|f| &f.name)).chain(lexicon.by_class.keys()).flat_map(|n| words(n)).filter(|w| !numeric(w)).collect() }
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
    let hit = MARKERS.iter().find(|m| text.contains(*m)).or_else(|| BANNED.iter().find(|w| contains_words(&plain, w)))
        .or_else(|| BANNED_EXACT.iter().find(|w| words.contains(*w))).copied().or_else(|| ip_literal(text).then_some("IP address"));
    hit.map_or(Ok(()), |h| Err(format!("banned: {h}")))
}
/// Longest-first match of normalized feature names in the normalized sentence. Every match is masked (its digits are not numbers, attack terms
/// inside it do not count, a shorter name inside it is not matched again); a catalog name outside the packet fails. Returns packet contributions.
fn features(text: &str, rules: &Rules, masked: &mut [bool]) -> Result<Vec<f64>, String> {
    let (mut norm, pos): (Vec<u8>, Vec<usize>) = text.bytes().enumerate().filter(|(_, b)| !matches!(b, b'_' | b'-' | b'/') && !b.is_ascii_whitespace()).map(|(i, b)| (b, i)).unzip();
    let mut mentioned = Vec::new();
    for (name, contribution) in &rules.features {
        let name = name.as_bytes();
        while let Some(at) = norm.windows(name.len()).position(|w| w == name) {
            let Some(c) = contribution else { return Err(format!("feature: {}", String::from_utf8_lossy(name))) };
            norm[at..at + name.len()].fill(0); masked[pos[at]..=pos[at + name.len() - 1]].fill(true); mentioned.push(*c);
        }
    }
    Ok(mentioned)
}
/// Marks the spans of allowed class names (`ddos syn flood port 1883`) so their digits are not number tokens.
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
fn numbers(t: &[u8], rules: &Rules) -> Result<(), String> {
    let digit = |j: usize| t.get(j).is_some_and(u8::is_ascii_digit);
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
        let ok = small || (x.is_finite() && bases.iter().any(|e| (x - if negative { *e } else { e.abs() }).abs() <= tolerance(*e)));
        if !ok { return Err(format!("number: {}{raw}{}", if negative { "-" } else { "" }, if percent { "%" } else { "" })); }
        i = j;
    }
    Ok(())
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
/// A predicted-class term needs a hedge, a runner-up term a comparison cue; any other lexicon term fails.
fn attacks(plain: &str, rules: &Rules) -> Result<(), String> {
    let hedged = HEDGES.iter().any(|w| contains_words(plain, w));
    let compared = CUES.iter().any(|w| contains_words(plain, w));
    let foreign = rules.attacks.iter().find(|t| contains_words(plain, t) && !((hedged && rules.predicted.contains(*t)) || (compared && rules.runner_ups.contains(*t))));
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
/// `Err` text starts with the rule name: `banned`, `length`, `feature`, `number`, `direction` or `attack`.
pub fn check(sentence: &str, rules: &Rules) -> Result<(), String> {
    let text = ascii(sentence)?;
    if text.split_whitespace().count() > 60 { return Err("length: more than 60 words".into()); }
    banned(&text)?;
    let mut masked = vec![false; text.len()];
    let mentioned = features(&text, rules, &mut masked)?;
    if hostname(&text.bytes().zip(&masked).map(|(b, m)| if *m { ' ' } else { b as char }).collect::<String>()) { return Err("banned: domain name".into()); }
    let plain = plain(&text, &masked);
    encoded(&plain, rules)?;
    let mut exempt = masked.clone();
    class_spans(&text, &masked, rules, &mut exempt);
    numbers(&text.bytes().zip(&exempt).map(|(b, m)| if *m { b' ' } else { b }).collect::<Vec<_>>(), rules)?;
    spelled(&plain, rules)?;
    if rules.directional && !mentioned.is_empty() { direction(&plain, &mentioned)?; }
    attacks(&plain, rules)
}

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

