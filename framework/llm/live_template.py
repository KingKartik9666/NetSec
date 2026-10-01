"""Deterministic interpretation of live evidence; separate from local LLM inference."""


def describe_observation(snapshot):
    model = snapshot.get("model_prediction")
    rules = snapshot["assessment"]
    if not snapshot["running"] or snapshot.get("error"):
        text = "Monitoring is stopped or unavailable; no current traffic verdict is supplied."
    elif model and model.get("available", True) and "score" in model:
        text = (f"{model['label']}, with an uncalibrated model score of {model['score']:.1%}. "
                f"The separate behavior rules report: {rules['label']}. "
                "These methods may disagree; verify the traffic before confirming an intrusion.")
    else:
        text = f"The behavior rules report: {rules['label']}. No model prediction is available for this observation."
    return {"mode": "Offline evidence template", "llm_executed": False, "assessment": text,
            "actions": ["Review the observed destinations, process names and time window.",
                        "Compare flagged patterns with expected activity before taking action.",
                        "Preserve this report and investigate independent evidence of an incident."],
            "uncertainty": "Live detection accuracy is unvalidated. False alarms are possible. SHAP explains model behavior; rules and model scores do not establish malicious intent. No automatic mitigation executes."}
