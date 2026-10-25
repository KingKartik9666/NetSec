//! Literal port of the current Python lexical checker, not semantic validation.
use crate::SpikeError;
use regex::Regex;
use serde_json::{json, Value};
use std::collections::HashSet;

pub fn feature_name(raw: &str) -> String {
    raw.strip_prefix("network_")
        .unwrap_or(raw)
        .replace("tcp-flags-", "TCP ")
        .replace(['_', '-'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn normalize(s: &str) -> String {
    s.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn has(s: &str, terms: &[&str]) -> bool {
    terms.iter().any(|term| s.contains(term))
}

fn missing(feature: &Value) -> bool {
    feature["is_missing"] == true
        || feature["observed"] == false
        || feature.get("value").is_some_and(Value::is_null)
}

pub fn check(
    text: &str,
    evidence: &Value,
    passages: &[Value],
    finish: Option<&str>,
) -> Result<Value, SpikeError> {
    let normalized = normalize(text);
    let mut issues = Vec::<String>::new();
    if normalized.is_empty() {
        issues.push("empty_response".into());
    }
    if Regex::new(
        r"\b(?:rewrite the facts|output only|copy every fact|rewritten facts and fixes)\b",
    )?
    .is_match(&normalized)
        || Regex::new(
            r"\b[a-z]+(?:_[a-z]+)*(?:_missing|_changed|_response|_claim)(?:_[a-z]+)*\s*:",
        )?
        .is_match(&normalized)
    {
        issues.push("instruction_or_diagnostic_echo".into());
    }
    if finish == Some("length") {
        issues.push("truncated_response".into());
    }
    if Regex::new(r"\d+(?:\.\d+)?\s*%")?.is_match(text) {
        issues.push("unsupported_percentage".into());
    }
    let prediction = &evidence["prediction"];
    let label = prediction["label"]
        .as_str()
        .ok_or_else(|| SpikeError::Invalid("Missing predicted label".into()))?;
    let low_label = label.to_lowercase();
    let checked_label = low_label
        .strip_prefix("model predicts ")
        .unwrap_or(&low_label)
        .split('(')
        .next()
        .unwrap_or("")
        .trim();
    if !normalized.contains(checked_label) {
        issues.push("predicted_label_missing_or_changed".into());
    }
    let mut hypotheses = Vec::<String>::new();
    if let Some(entries) = prediction["top_classes"].as_array() {
        for entry in entries.iter().take(3) {
            let name = entry["display_label"]
                .as_str()
                .filter(|s| !s.is_empty())
                .or_else(|| entry["label"].as_str())
                .ok_or_else(|| SpikeError::Invalid("Missing hypothesis label".into()))?;
            if !hypotheses.iter().any(|s| s == name) {
                hypotheses.push(name.into());
            }
        }
    }
    if !hypotheses.is_empty() {
        if hypotheses
            .iter()
            .any(|s| !normalized.contains(&normalize(s)))
        {
            issues.push("tentative_category_missing_or_changed".into());
        }
        if hypotheses.len() > 1
            && !has(
                &normalized,
                &["tentative", "hypothes", "alternative", "possible categor"],
            )
        {
            issues.push("tentative_category_warning_missing".into());
        }
    }
    let features = evidence["xai"]["features"]
        .as_array()
        .map(|a| a.iter().take(2).collect::<Vec<_>>())
        .unwrap_or_default();
    let coverage = &evidence["feature_coverage"];
    let mut number_sources = vec![label.to_string()];
    number_sources.extend(hypotheses);
    number_sources.extend(
        features
            .iter()
            .filter_map(|f| f["feature"].as_str())
            .map(feature_name),
    );
    for key in ["observed", "total"] {
        if let Some(v) = coverage.get(key) {
            number_sources.push(v.to_string());
        }
    }
    let numbers = Regex::new(r"\b\d+(?:\.\d+)?\b")?;
    let source = number_sources.join(" ");
    let allowed: HashSet<_> = numbers.find_iter(&source).map(|m| m.as_str()).collect();
    if numbers
        .find_iter(text)
        .any(|m| !allowed.contains(m.as_str()))
    {
        issues.push("unsupported_number".into());
    }
    if coverage["missing"].as_u64().unwrap_or(0) != 0
        && (!has(
            &normalized,
            &[
                "partial",
                "incomplete",
                "missing input",
                "unavailable input",
                "limited input",
            ],
        ) || !has(&normalized, &["limit", "uncertain"]))
    {
        issues.push("partial_input_warning_missing".into());
    }
    if coverage["validated_profile"] == false
        && !has(
            &normalized,
            &["not been validated", "not validated", "unvalidated"],
        )
    {
        issues.push("unvalidated_profile_warning_missing".into());
    }
    if !has(
        &normalized,
        &[
            "verif",
            "does not confirm",
            "not proof",
            "human review",
            "needs checking",
        ],
    ) {
        issues.push("uncertainty_missing".into());
    }
    let incident = Regex::new(
        r"\b(?:confirmed (?:attack|intrusion|incident)|definitely (?:safe|malicious)|guarantees? safety|is safe)\b",
    )?;
    let negation = Regex::new(r"\b(?:not|no|never|cannot|without)\b")?;
    for found in incident.find_iter(&normalized) {
        // Python slices by Unicode codepoints rather than UTF-8 bytes.
        let preceding: String = normalized[..found.start()]
            .chars()
            .rev()
            .take(55)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        if !negation.is_match(&preceding) {
            issues.push("unsupported_incident_or_safety_claim".into());
            break;
        }
    }
    let names = features
        .iter()
        .map(|f| feature_name(f["feature"].as_str().unwrap_or("")).to_lowercase())
        .collect::<Vec<_>>();
    for (feature, name) in features.iter().zip(&names) {
        let raw = feature["feature"]
            .as_str()
            .ok_or_else(|| SpikeError::Invalid("Missing feature name".into()))?;
        let Some(start) = normalized.find(name) else {
            issues.push(format!("feature_name_missing:{raw}"));
            continue;
        };
        let end = names
            .iter()
            .filter(|other| *other != name)
            .filter_map(|other| {
                normalized[start + name.len()..]
                    .find(other)
                    .map(|i| i + start + name.len())
            })
            .min()
            .unwrap_or(normalized.len());
        let clause = normalized[start..end].split('.').next().unwrap_or("");
        if missing(feature) {
            let sentence_start = normalized[..start].rfind('.').map(|i| i + 1).unwrap_or(0);
            let sentence = normalized[sentence_start..end]
                .split('.')
                .next()
                .unwrap_or("");
            if !has(sentence, &["unavailable", "missing", "not observed"])
                || !sentence.contains("default")
            {
                issues.push(format!("missing_input_presented_as_observed:{raw}"));
            }
        }
        let shap = feature["shap_value"]
            .as_f64()
            .ok_or_else(|| SpikeError::Invalid("Missing SHAP value".into()))?;
        let wrong = if shap > 0.0 {
            r"\b(?:oppos(?:ed|es)|decreas(?:ed|es)|lower(?:ed|s))\b"
        } else {
            r"\b(?:support(?:ed|s)|increas(?:ed|es)|rais(?:ed|es))\b"
        };
        if shap != 0.0 && Regex::new(wrong)?.is_match(clause) {
            issues.push(format!("feature_direction_changed:{raw}"));
        }
    }
    if evidence["rules"]["status"] == "benign_looking"
        && prediction["is_attack"] == true
        && (!normalized.contains("rules")
            || !has(
                &normalized,
                &["disagree", "did not", "not flag", "no alert"],
            ))
    {
        issues.push("rule_model_disagreement_missing".into());
    }
    let allowed_ids: HashSet<_> = passages.iter().filter_map(|p| p["id"].as_str()).collect();
    if Regex::new(r"\[([^\]]+)\]")?
        .captures_iter(text)
        .any(|c| c.get(1).is_some_and(|m| !allowed_ids.contains(m.as_str())))
    {
        issues.push("invented_source_identifier".into());
    }
    Ok(json!({"passed": issues.is_empty(), "issues": issues,
        "method": "Deterministic lexical checks of labels, percentages, uncertainty, partial coverage, missing-input routing, directions and source identifiers",
        "limitations": "Heuristic checks can miss paraphrased or semantic errors. Passing is not proof of correctness or citation entailment."}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn phase_zero_grounding_corpus() -> Result<(), SpikeError> {
        let corpus: Value = serde_json::from_str(include_str!(
            "../../../../evaluation/fixtures/rust_migration/v1/rag/grounding.json"
        ))?;
        for case in corpus["cases"]
            .as_array()
            .ok_or_else(|| SpikeError::Invalid("Missing cases".into()))?
        {
            let input = &case["input"];
            let actual = check(
                input["text"].as_str().unwrap_or(""),
                &input["evidence"],
                input["passages"]
                    .as_array()
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
                input["finish_reason"].as_str(),
            )?;
            assert_eq!(actual, case["output"], "{}", case["id"]);
        }
        Ok(())
    }
}
