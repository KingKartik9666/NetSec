"""Deterministic analyst report for real research inference; no LLM execution."""


class ResearchTemplateProvider:
    def generate(self, context: dict) -> dict:
        prediction, explanation = context["prediction"], context["explanation"]
        label = prediction["label"]
        assessment = (f"The trained XGBoost classifier predicts {label}. "
                      f"Its closed-set model score is {prediction['score']:.1%} (uncalibrated). ")
        if prediction["is_attack"]:
            assessment += "This is a model alert requiring verification against the underlying traffic."
            actions = ["Preserve the relevant traffic and device logs; verify the source and time window.",
                       "Compare the feature evidence with expected device behavior before confirming an incident."]
            if "ddos" in label or "dos_" in label:
                actions.append("If flooding is independently confirmed, assess scoped rate limiting and service impact.")
            elif "scan" in label or "recon" in label:
                actions.append("If probing is confirmed, review exposed services and scoped access restrictions.")
            else:
                actions.append("If malicious activity is confirmed, review scoped containment with an operator.")
        else:
            assessment += "A benign prediction does not establish that the device or network is safe."
            actions = ["Continue monitoring and investigate any independent evidence of compromise."]
        strongest = explanation["important_features"][0]
        assessment += (f" The largest displayed absolute SHAP contribution is {strongest['feature']} "
                       f"({strongest['shap_value']:+.3f} raw margin units).")
        coverage = context.get("feature_coverage", {})
        if coverage.get("missing"):
            assessment += (f" Only {coverage['observed']} of {coverage['total']} inputs were measured; "
                           "the displayed categories are tentative hypotheses. " + coverage["validation_scope"])
        if strongest.get("is_missing"):
            assessment += " That feature was unavailable: its contribution describes missing-value routing, not measured traffic."
        return {"mode": "Offline template", "assessment": assessment, "actions": actions,
                "uncertainty": "Validation-selected DataSense model; held-out test and unknown-attack performance remain unevaluated. "
                               "Feature attributions explain model behavior, not malicious intent. No automated network actions are taken.",
                "llm_executed": False}
