"""Small, auditable generation checks, not a learned critic or semantic proof."""
import re


def feature_name(raw):
    return " ".join(raw.removeprefix("network_").replace("tcp-flags-", "TCP ").replace("_", " ").replace("-", " ").split())


def feature_is_missing(feature):
    # Legacy evidence without a value/missing field is not assumed unavailable.
    return (feature.get("is_missing") is True or feature.get("observed") is False
            or ("value" in feature and feature["value"] is None))


def hypothesis_labels(evidence):
    """Preserve supplied friendly category labels; never infer alternative attacks."""
    return list(dict.fromkeys(str(entry.get("display_label") or entry["label"])
                             for entry in evidence["prediction"].get("top_classes", [])[:3]))


def check_grounding(text, evidence, passages=(), finish_reason=None):
    normalized = " ".join(text.lower().split())
    issues = []
    if not normalized:
        issues.append("empty_response")
    if (re.search(r"\b(?:rewrite the facts|output only|copy every fact|rewritten facts and fixes)\b", normalized)
            or re.search(r"\b[a-z]+(?:_[a-z]+)*(?:_missing|_changed|_response|_claim)(?:_[a-z]+)*\s*:", normalized)):
        issues.append("instruction_or_diagnostic_echo")
    if finish_reason == "length":
        issues.append("truncated_response")
    if re.search(r"\d+(?:\.\d+)?\s*%", text):
        issues.append("unsupported_percentage")
    label = evidence["prediction"]["label"].lower().removeprefix("model predicts ").split("(")[0].strip()
    if label not in normalized:
        issues.append("predicted_label_missing_or_changed")
    hypotheses = hypothesis_labels(evidence)
    if hypotheses:
        if any(" ".join(name.lower().split()) not in normalized for name in hypotheses):
            issues.append("tentative_category_missing_or_changed")
        if len(hypotheses) > 1 and not any(term in normalized for term in ["tentative", "hypothes", "alternative", "possible categor"]):
            issues.append("tentative_category_warning_missing")
    coverage = evidence.get("feature_coverage") or {}
    # Numerical measurements are deliberately excluded from the small-model
    # prose task. Only digits in supplied labels/names and coverage are allowed.
    number_sources = [evidence["prediction"]["label"], *hypotheses]
    number_sources.extend(feature_name(f["feature"]) for f in evidence.get("xai", {}).get("features", [])[:2])
    number_sources.extend(str(coverage[key]) for key in ("observed", "total") if key in coverage)
    number_pattern = r"\b\d+(?:\.\d+)?\b"
    allowed_numbers = set(re.findall(number_pattern, " ".join(number_sources)))
    if set(re.findall(number_pattern, text)) - allowed_numbers:
        issues.append("unsupported_number")
    if coverage.get("missing", 0):
        incomplete = any(term in normalized for term in ["partial", "incomplete", "missing input", "unavailable input", "limited input"])
        if not incomplete or not any(term in normalized for term in ["limit", "uncertain"]):
            issues.append("partial_input_warning_missing")
    if coverage.get("validated_profile") is False and not any(term in normalized for term in ["not been validated", "not validated", "unvalidated"]):
        issues.append("unvalidated_profile_warning_missing")
    if not any(term in normalized for term in ["verif", "does not confirm", "not proof", "human review", "needs checking"]):
        issues.append("uncertainty_missing")
    for match in re.finditer(r"\b(?:confirmed (?:attack|intrusion|incident)|definitely (?:safe|malicious)|guarantees? safety|is safe)\b", normalized):
        preceding = normalized[max(0, match.start() - 55):match.start()]
        if not re.search(r"\b(?:not|no|never|cannot|without)\b", preceding):
            issues.append("unsupported_incident_or_safety_claim")
            break
    features = evidence.get("xai", {}).get("features", [])[:2]
    names = [feature_name(f["feature"]).lower() for f in features]
    for feature, name in zip(features, names):
        start = normalized.find(name)
        if start < 0:
            issues.append("feature_name_missing:" + feature["feature"])
            continue
        end = min([p for p in [normalized.find(other, start + len(name)) for other in names if other != name] if p >= 0] or [len(normalized)])
        clause = normalized[start:end].split(".")[0]
        if feature_is_missing(feature):
            sentence = normalized[normalized.rfind(".", 0, start) + 1:end].split(".")[0]
            if not any(term in sentence for term in ["unavailable", "missing", "not observed"]) or "default" not in sentence:
                issues.append("missing_input_presented_as_observed:" + feature["feature"])
        wrong = r"\b(?:oppos(?:ed|es)|decreas(?:ed|es)|lower(?:ed|s))\b" if feature["shap_value"] > 0 else r"\b(?:support(?:ed|s)|increas(?:ed|es)|rais(?:ed|es))\b"
        if feature["shap_value"] and re.search(wrong, clause):
            issues.append("feature_direction_changed:" + feature["feature"])
    if (evidence.get("rules") or {}).get("status") == "benign_looking" and evidence["prediction"]["is_attack"]:
        if "rules" not in normalized or not any(term in normalized for term in ["disagree", "did not", "not flag", "no alert"]):
            issues.append("rule_model_disagreement_missing")
    allowed = {str(p.get("id")) for p in passages}
    for citation in re.findall(r"\[([^\]]+)\]", text):
        if citation not in allowed:
            issues.append("invented_source_identifier")
            break
    return {"passed": not issues, "issues": issues,
            "method": "Deterministic lexical checks of labels, percentages, uncertainty, partial coverage, missing-input routing, directions and source identifiers",
            "limitations": "Heuristic checks can miss paraphrased or semantic errors. Passing is not proof of correctness or citation entailment."}
