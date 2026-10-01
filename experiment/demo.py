"""Teacher demonstration: synthetic features to a grounded offline report."""

import argparse
from collections.abc import Sequence

import pandas as pd
from dotenv import load_dotenv

from framework.data.loader import PROJECT_ROOT, SAMPLE_NAMES, load_demo_data, select_sample, training_data
from framework.detection.demo_detector import DemoDetector
from framework.detection.hem_adapter import HEMAdapter
from framework.llm.provider import choose_provider
from framework.pipeline.security_pipeline import IoTSecurityPipeline
from framework.xai.explainer import FeatureExplainer


def create_pipeline(frame: pd.DataFrame) -> IoTSecurityPipeline:
    """Explicitly select the demo backend; fit only the marked training rows."""
    X, y = training_data(frame)
    detector = DemoDetector().fit(X, y)
    explainer = FeatureExplainer(X, y, detector.classifier.feature_importances_)
    return IoTSecurityPipeline(detector, explainer, choose_provider())


def run_demo(sample_name: str = "ddos") -> dict:
    """Run one held-out sample and print evidence from the actual pipeline."""
    print("=" * 72)
    print("ADAPTIVE IoT SECURITY FRAMEWORK - PROTOTYPE")
    print("=" * 72)
    print("Detector: DemoDetector (DEMONSTRATION ML MODEL; not HEM)")
    print("Reasoning: OFFLINE TEMPLATE PROVIDER (no actual LLM inference)")
    print("Dataset: SYNTHETIC DEMONSTRATION DATA; not research results\n")
    print("[1/5] Loading and validating synthetic IoT traffic...")
    frame = load_demo_data()
    sample = select_sample(frame, sample_name)
    print("[2/5] Fitting the CPU prototype on known training rows...")
    pipeline = create_pipeline(frame)
    result = pipeline.analyze(sample)
    detection, explanation = result["detection"], result["explanation"]
    print("[3/5] Intrusion classification and known/unknown analysis completed.")
    print("[4/5] Feature explanation generated from model and training references.")
    print("[5/5] Offline security reasoning report generated.")
    print("\n" + "-" * 72 + "\nDETECTION RESULT\n" + "-" * 72)
    status = (
        "ANOMALOUS / UNKNOWN - REVIEW REQUIRED" if detection["novelty"] == "unknown"
        else "ATTACK DETECTED (prototype prediction)" if detection["is_attack"]
        else "BENIGN (prototype prediction)"
    )
    print(f"Sample: {sample_name}\nStatus: {status}")
    print(f"Classification: {detection['attack_type']}\nNovelty: {detection['novelty'].upper()}")
    confidence = detection["confidence"]
    print("Confidence: " + (
        "not available for UNKNOWN; no calibrated probability"
        if confidence is None else f"{confidence:.1%} (uncalibrated classifier score)"
    ))
    print(f"Novelty distance: {detection['novelty_score']:.3f}; threshold: {detection['novelty_threshold']:.3f}")
    print("\n" + "-" * 72 + "\nXAI EVIDENCE\n" + "-" * 72)
    print(f"Method: {explanation['method']}")
    for index, feature in enumerate(explanation["important_features"], start=1):
        print(f"{index}. {feature['feature']} = {feature['value']:.4g}; relative score {feature['importance']:.3f}")
    print("\n" + "-" * 72 + "\nSECURITY ANALYSIS\n" + "-" * 72)
    print(result["report"])
    print("\n" + "-" * 72 + "\nRESEARCH STATUS\n" + "-" * 72)
    print("Detector: prototype Random Forest + standardized nearest-neighbor distance.")
    print("HEM: " + HEMAdapter().status()["reason"])
    print("XAI: simplified CLI feature evidence; desktop TreeSHAP/LIME are separate.")
    print("Reasoning: CLI TemplateProvider (no LLM inference or retrieval); optional desktop local LLM with bounded corrective RAG.")
    print("Mitigation: recommendations only; no automated network actions.")
    print("=" * 72)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point with project-root-safe configuration and useful errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", choices=SAMPLE_NAMES, default="ddos")
    args = parser.parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    try:
        run_demo(args.sample)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Demo error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
