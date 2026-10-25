"""Measure Phase 1 seeded Rust LIME against sealed synthetic Python fixtures.

This harness may use Python reference code for comparisons. The Rust executable
does all neighborhood sampling, UBJ model prediction, feature selection and
surrogate fitting itself; Python is never its prediction callback.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "rust" / "spikes" / "lime" / "Cargo.toml"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(binary: Path, request: dict) -> dict:
    result = subprocess.run([str(binary)], input=json.dumps(request, allow_nan=False),
                            encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=90, check=False, cwd=ROOT)
    if result.returncode:
        raise RuntimeError(f"Rust LIME exited {result.returncode}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def feature_names(conditions: list[dict], names: list[str]) -> list[str]:
    result = []
    for condition in conditions:
        if "feature" in condition:
            result.append(condition["feature"])
        else:
            matches = [name for name in names if name in condition["condition"]]
            if len(matches) != 1:
                raise ValueError("Python condition did not identify exactly one frozen feature")
            result.append(matches[0])
    return result


def compare(identifier: str, python: dict, rust: dict, names: list[str]) -> dict:
    py_features = feature_names(python["conditions"], names)
    rs_features = feature_names(rust["conditions"], names)
    metrics = {}
    for k in [1, 3, 5, 12]:
        limit = min(k, len(py_features), len(rs_features))
        overlap = len(set(py_features[:limit]) & set(rs_features[:limit]))
        metrics[f"top_{k}"] = {"compared": limit, "overlap": overlap, "fraction": overlap / limit}
    py_weights = {name: condition["weight"] for name, condition in zip(py_features, python["conditions"])}
    rs_weights = {name: condition["weight"] for name, condition in zip(rs_features, rust["conditions"])}
    # Report effect-aware agreement separately: flat/tiny numerical weights do
    # not constitute a meaningful ranked explanation.
    active_py = {name for name, weight in py_weights.items() if abs(weight) > 1e-8}
    active_rs = {name for name, weight in rs_weights.items() if abs(weight) > 1e-8}
    py_conditions = {name: condition["condition"] for name, condition in zip(py_features, python["conditions"])}
    rs_conditions = {name: condition["condition"] for name, condition in zip(rs_features, rust["conditions"])}
    shared = set(py_features) & set(rs_features)
    return {
        "id": identifier, "top_feature_overlap": metrics,
        "nontrivial_python_features": sorted(active_py), "nontrivial_rust_features": sorted(active_rs),
        "flat_ordering_is_uninformative": not active_py and not active_rs,
        "shared_conditions_equal": all(py_conditions[name] == rs_conditions[name] for name in shared),
        "shared_condition_count": len(shared),
        "same_target_class": python["class_label"] == rust["class_label"],
        "python_model_score": python["model_score"], "rust_model_score": rust["model_score"],
        "anchor_score_abs_error": abs(python["model_score"] - rust["model_score"]),
        "python_local_fidelity_r2": python["local_fidelity_r2"], "rust_local_fidelity_r2": rust["local_fidelity_r2"],
        "r2_abs_difference": abs(python["local_fidelity_r2"] - rust["local_fidelity_r2"]),
        "python_surrogate_prediction": python["surrogate_prediction"], "rust_surrogate_prediction": rust["surrogate_prediction"],
        "surrogate_prediction_abs_difference": abs(python["surrogate_prediction"] - rust["surrogate_prediction"]),
        "python_anchor_surrogate_gap": abs(python["model_score"] - python["surrogate_prediction"]),
        "rust_anchor_surrogate_gap": abs(rust["model_score"] - rust["surrogate_prediction"]),
        "rust_constant_target": rust["constant_target"],
        "observed_count_equal": python["observed_features"] == rust["observed_features"],
        "missing_count_equal": python["missing_features"] == rust["missing_features"],
        "packet_constraints_equal": python["perturbation_constraints"] == rust["perturbation_constraints"],
    }


def same_neighborhood_checks(binary: Path) -> list[dict]:
    """Separate Ridge/selection arithmetic from deliberate RNG differences."""
    import numpy as np
    from lime.lime_base import LimeBase
    rng = np.random.RandomState(20261012)
    cases = []
    for count in [3, 16]:
        data = (rng.uniform(size=(512, count)) > 0.55).astype(float)
        data[0] = 1.0
        labels = 0.2 + 0.25 * data[:, count - 1] - 0.15 * data[:, 0] + 0.10 * data[:, 1]
        if count == 16:
            labels += 0.04 * data[:, 2] * data[:, 3]
        squared = ((data - data[0]) ** 2).sum(axis=1)
        weights = np.exp(-squared / (2 * count * 0.75**2))
        maximum = min(12, count)
        request = {"operation": "fit", "data": data.tolist(), "labels": labels.tolist(), "weights": weights.tolist(), "maximum": maximum}
        reference = LimeBase(lambda _: weights, random_state=42)
        intercept, local_exp, score, local_pred = reference.explain_instance_with_data(
            data, labels[:, None], np.zeros(len(data)), 0, maximum)
        selected_python = reference.feature_selection(data, labels, weights, maximum, "auto").tolist()
        actual = run(binary, request)
        py_coef = dict(local_exp)
        rs_coef = dict(zip(actual["selected_features"], actual["fit"]["coefficients"]))
        cases.append({
            "observed_features": count,
            "selection": "forward_selection" if maximum <= 6 else "highest_weights",
            "same_selected_features": actual["selected_features"] == selected_python,
            "python_selected_features": selected_python, "rust_selected_features": actual["selected_features"],
            "max_coefficient_abs_error": max(abs(py_coef[i] - rs_coef.get(i, 0.0)) for i in py_coef),
            "intercept_abs_error": abs(float(intercept) - actual["fit"]["intercept"]),
            "r2_abs_error": abs(float(score) - actual["fit"]["r2"]),
            "surrogate_prediction_abs_error": abs(float(local_pred[0]) - actual["surrogate_prediction"]),
        })
    return cases


def real_artifact_checks(binary: Path, bundle_path: Path, out: Path) -> dict:
    """Two saved-artifact software checks, with source-validation provenance."""
    import numpy as np
    import xgboost as xgb
    from framework.xai.model_evidence import explain_additional
    bundle_path = bundle_path.resolve()
    bundle = read_json(bundle_path)
    checkpoint = (bundle_path.parent / bundle["checkpoint"]).resolve()
    reference = bundle_path.parent / "xai_reference.npz"
    metadata = read_json(reference.with_suffix(".json"))
    if digest(checkpoint) != bundle["checkpoint_sha256"] or digest(reference) != metadata["sha256"]:
        raise ValueError("Real artifact checksum differs from its saved manifest")
    sample = bundle["samples"][0]
    full = sample["features"]
    packet_names = ["network_packets_all_count"] + [f"network_tcp-flags-{flag}_count" for flag in ["ack", "fin", "psh", "rst", "syn", "urg"]]
    projected = [None] * len(full)
    for name in packet_names:
        index = bundle["feature_names"].index(name)
        projected[index] = full[index]
    cases = [{"id": "saved_full_validation_sample", "values": full},
             {"id": "same_source_seven_count_projection", "values": projected, "packet": True}]
    request = {"operation": "explain", "checkpoint": str(checkpoint), "checkpoint_sha256": bundle["checkpoint_sha256"],
               "reference": str(reference), "feature_names": bundle["feature_names"], "class_names": bundle["classes"], "cases": cases}
    start = time.perf_counter()
    actual = run(binary, request)
    elapsed = time.perf_counter() - start
    write_json(out / "real_rust.json", actual)
    booster = xgb.Booster(params={"device": "cpu", "nthread": 2})
    booster.load_model(str(checkpoint))
    comparisons = []
    reference_outputs = []
    for case, rust in zip(cases, actual["cases"]):
        row = np.asarray([np.nan if value is None else value for value in case["values"]], dtype=np.float32)
        python = explain_additional(booster, row, bundle["feature_names"], bundle["classes"], reference, packet=case.get("packet", False))
        reference_outputs.append({"id": case["id"], "output": python})
        comparisons.append(compare(case["id"], python["lime"], rust["lime"], bundle["feature_names"]))
    write_json(out / "real_python.json", reference_outputs)
    report = {
        "scope": "saved-model execution agreement; source validation sample and its seven-count projection, no new capture or detection accuracy",
        "bundle": str(bundle_path.relative_to(ROOT)), "bundle_sha256": digest(bundle_path),
        "checkpoint_sha256": digest(checkpoint), "reference_sha256": digest(reference),
        "reference_rows": metadata["rows"], "features": len(bundle["feature_names"]), "classes": len(bundle["classes"]),
        "source_sample_id": sample["id"], "source_row": sample["source_row"], "source_validation_label": sample["label"],
        "projection_source": "seven values retained from the same saved source-validation sample; all other inputs unobserved",
        "real_model_file_unchanged": digest(checkpoint) == bundle["checkpoint_sha256"],
        "real_reference_file_unchanged": digest(reference) == metadata["sha256"],
        "two_case_rust_process_seconds": elapsed, "cases": comparisons,
    }
    write_json(out / "real_parity.json", report)
    return report


def main() -> int:
    # These environment changes affect only this comparison process and its
    # children, leaving the independent research workers untouched.
    for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[variable] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path, default=ROOT / "evaluation/fixtures/rust_migration/v1")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/rust_migration/phase1/lime")
    parser.add_argument("--binary", type=Path, help="Use an already built Rust binary")
    parser.add_argument("--real-bundle", type=Path, help="Additionally compare a saved full sample and its canonical seven-count projection")
    args = parser.parse_args()
    fixture = args.fixture_root.resolve()
    out = args.out.resolve()
    if out == fixture or fixture in out.parents:
        raise ValueError("Do not write Phase 1 output inside the sealed Phase 0 corpus")
    out.mkdir(parents=True, exist_ok=True)
    sealed_before = digest(fixture / "manifest.json")
    bundle = read_json(fixture / "models/bundle-v2.json")
    additional = read_json(fixture / "models/additional.json")
    env = os.environ.copy()
    env.update({"CARGO_BUILD_JOBS": "2", "CARGO_TARGET_DIR": str(out / "target"),
                "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "NUMEXPR_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": ""})
    portable = ROOT / "artifacts/rust_migration/phase1/toolchain"
    if (portable / "cargo/bin/cargo.exe").is_file():
        env["RUSTUP_HOME"] = str(portable / "rustup")
        env["CARGO_HOME"] = str(portable / "cargo")
        cargo = portable / "cargo/bin/cargo.exe"
    else:
        cargo = shutil.which("cargo")
    binary = args.binary.resolve() if args.binary else out / "target/release" / ("netsec-lime-spike.exe" if os.name == "nt" else "netsec-lime-spike")
    if not args.binary:
        if not cargo:
            raise RuntimeError("Rust toolchain unavailable; no Python substitute is allowed")
        build = subprocess.run([str(cargo), "build", "--release", "--locked", "--manifest-path", str(MANIFEST)],
                               cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               encoding="utf-8", timeout=600, check=False)
        (out / "release-build.log").write_text(build.stdout, encoding="utf-8")
        if build.returncode:
            raise RuntimeError(f"Rust build failed ({build.returncode}); inspect release-build.log")
    base_request = {"operation": "explain", "checkpoint": str(fixture / "models/model.ubj"),
                    "checkpoint_sha256": bundle["checkpoint_sha256"], "reference": str(fixture / "models/xai_reference.npz"),
                    "feature_names": bundle["feature_names"], "class_names": bundle["classes"]}
    request = dict(base_request, cases=[{"id": case["id"], "values": case["input"], "packet": case["id"] == "packet_counts"} for case in additional["cases"]])
    # Only artifact output contains absolute invocation paths; source uses ROOT.
    write_json(out / "request.json", request)
    started = time.perf_counter()
    results = run(binary, request)
    runtime = time.perf_counter() - started
    write_json(out / "rust_results.json", results)
    repeated = run(binary, request)
    first_evidence = [{"id": c["id"], "lime": c["lime"]} for c in results["cases"]]
    repeat_evidence = [{"id": c["id"], "lime": c["lime"]} for c in repeated["cases"]]
    comparisons = [compare(case["id"], case["output"]["lime"], actual["lime"], bundle["feature_names"])
                   for case, actual in zip(additional["cases"], results["cases"])]
    seeds_request = dict(base_request, cases=[{"id": f"full_seed_{seed}", "values": additional["cases"][0]["input"], "seed": seed} for seed in [42, 43, 44, 45, 46]])
    seed_results = run(binary, seeds_request)
    write_json(out / "seed_sensitivity.json", seed_results)
    same_neighborhood = same_neighborhood_checks(binary)
    write_json(out / "same_neighborhood.json", same_neighborhood)
    # Additional partial numeric/categorical case uses current actual Python
    # implementation for comparison, not a newly invented golden expectation.
    import numpy as np
    import xgboost as xgb
    sys.path.insert(0, str(ROOT))
    from framework.xai.model_evidence import explain_additional
    values = [None] * len(bundle["feature_names"])
    for index in [7, 9, 10]:
        values[index] = additional["cases"][0]["input"][index]
    booster = xgb.Booster(params={"device": "cpu", "nthread": 2})
    booster.load_model(str(fixture / "models/model.ubj"))
    row = np.asarray([np.nan if value is None else value for value in values], dtype=np.float32)
    python_partial = explain_additional(booster, row, bundle["feature_names"], bundle["classes"], fixture / "models/xai_reference.npz")
    partial_result = run(binary, dict(base_request, cases=[{"id": "three_observed_with_missing_categories", "values": values}]))
    write_json(out / "partial_python.json", python_partial)
    write_json(out / "partial_rust.json", partial_result)
    partial_comparison = compare("three_observed_with_missing_categories", python_partial["lime"], partial_result["cases"][0]["lime"], bundle["feature_names"])
    real_artifacts = real_artifact_checks(binary, args.real_bundle, out) if args.real_bundle else {"executed": False, "reason": "No --real-bundle requested"}
    evidence = {
        "scope": "synthetic Phase 1 software parity; no research accuracy, real artifact, live capture or runtime integration",
        "fixture_manifest_sha256": sealed_before,
        "fixture_hashes": {name: digest(fixture / name) for name in ["models/model.ubj", "models/xai_reference.npz", "models/xai_reference.json", "models/additional.json"]},
        "binary_sha256": digest(binary), "binary_bytes": binary.stat().st_size,
        "packages": {name: importlib.metadata.version(name) for name in ["lime", "numpy", "scipy", "scikit-learn", "xgboost"]},
        "seed": 42, "perturbations": 2048, "cpu_threads_max": 2,
        "deterministic_evidence_repeat": first_evidence == repeat_evidence,
        "sealed_manifest_unchanged": digest(fixture / "manifest.json") == sealed_before,
        "three_case_process_seconds": runtime,
        "cases": comparisons, "supplemental_partial": partial_comparison,
        "same_neighborhood": same_neighborhood,
        "full_seed_r2_range": [min(c["lime"]["local_fidelity_r2"] for c in seed_results["cases"]), max(c["lime"]["local_fidelity_r2"] for c in seed_results["cases"])],
        "full_seed_surrogate_range": [min(c["lime"]["surrogate_prediction"] for c in seed_results["cases"]), max(c["lime"]["surrogate_prediction"] for c in seed_results["cases"])],
        "full_seed_top_feature": [c["lime"]["conditions"][0]["feature"] for c in seed_results["cases"]],
        "real_artifacts": real_artifacts,
    }
    write_json(out / "parity.json", evidence)
    required = (evidence["deterministic_evidence_repeat"] and evidence["sealed_manifest_unchanged"]
                and all(c["same_target_class"] and c["anchor_score_abs_error"] <= 1e-5
                        and c["observed_count_equal"] and c["missing_count_equal"] and c["packet_constraints_equal"] and c["shared_conditions_equal"] for c in comparisons + [partial_comparison])
                and all(c["same_selected_features"] and c["max_coefficient_abs_error"] <= 1e-10
                        and c["surrogate_prediction_abs_error"] <= 1e-10 for c in same_neighborhood))
    evidence["technical_checks_passed"] = required
    evidence["decision"] = {
        "algorithm_feasibility": "go" if required else "no-go",
        "drop_in_runtime_replacement": "no-go",
        "reason": "RNG/numeric and exactly-constant R2 deviations require an explicit compatibility decision; full fixture fidelity/anchor disagreement and limited ranking agreement do not establish reliable explanations. Current app remains on Python.",
    }
    write_json(out / "parity.json", evidence)
    print(json.dumps({"technical_checks_passed": required, "parity_report": str(out / "parity.json"), "cases": comparisons, "same_neighborhood": same_neighborhood}, indent=2))
    return 0 if required else 1


if __name__ == "__main__":
    raise SystemExit(main())
