"""Phase 1 local Qwen comparison; never changes runtime or model bytes.

The Python worker executes the current LocalLanguageModel. Rust owns actual
b11564 DLL generation or a native llama-completion child. Evidence is the sealed
synthetic software corpus, not a detection/LLM research quality benchmark.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def system_prompt():
    source = ast.parse((ROOT / "framework/llm/local_model.py").read_text(encoding="utf-8"))
    for node in ast.walk(source):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "system" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Current runtime system prompt not found")


def worker(request_path, output_path):
    from framework.llm.local_model import LocalLanguageModel
    import llama_cpp
    data = json.loads(request_path.read_text(encoding="utf-8"))
    os.environ["IMDAI_LLM_DIR"] = str(Path(data["model"]).parent)
    original = llama_cpp.Llama
    load = {}

    class MeasuredLlama(original):
        def __init__(self, *args, **kwargs):
            before = time.perf_counter()
            super().__init__(*args, **kwargs)
            load["llama_initialization_seconds"] = time.perf_counter() - before

    llama_cpp.Llama = MeasuredLlama
    model = LocalLanguageModel()
    before = time.perf_counter()
    result = model.generate(data["evidence"])
    result["generation_total_seconds"] = time.perf_counter() - before
    result["timing_instrumentation"] = load
    result["grounding"] = result["correction"]["generation_attempts"][-1].copy()
    result["grounding"].pop("attempt", None)
    result["unloaded_after_generation"] = not model.status()["loaded"]
    result["python_binding_version"] = llama_cpp.__version__
    dump(output_path, result)


def execute(command, prefix, timeout=120, allow_failure=False):
    import psutil
    before = time.perf_counter()
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    log_out, log_err = prefix.with_suffix(".stdout.log"), prefix.with_suffix(".stderr.log")
    peaks, pids, process_peak_wsets = [], set(), {}
    with log_out.open("wb") as out, log_err.open("wb") as err:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=out, stderr=err, stdin=subprocess.DEVNULL)
        parent = psutil.Process(process.pid)
        while process.poll() is None:
            processes = [parent]
            try:
                processes.extend(parent.children(recursive=True))
            except psutil.Error:
                pass
            total = 0
            for selected in processes:
                try:
                    memory = selected.memory_info()
                    total += memory.rss
                    pids.add(selected.pid)
                    process_peak_wsets[selected.pid] = max(process_peak_wsets.get(selected.pid, 0), getattr(memory, "peak_wset", memory.rss))
                except psutil.Error:
                    pass
            peaks.append(total)
            if time.perf_counter() - before > timeout:
                for selected in reversed(processes):
                    try:
                        selected.kill()
                    except psutil.Error:
                        pass
                process.wait(timeout=10)
                raise TimeoutError(f"Spike deadline exceeded: {command[0]}")
            time.sleep(0.01)
        elapsed = time.perf_counter() - before
        if process.returncode and not allow_failure:
            raise RuntimeError(f"Spike exited {process.returncode}: {log_err}")
    remaining = [pid for pid in pids if psutil.pid_exists(pid)]
    return {"wall_seconds": elapsed, "returncode": process.returncode, "peak_tree_rss_bytes": max(peaks, default=0),
            "process_peak_working_set_bytes": process_peak_wsets, "sample_interval_seconds": 0.01,
            "sample_count": len(peaks), "observed_process_count": len(pids), "remaining_owned_pids": remaining}


def prepare(output, native_dir):
    from framework.llm.local_model import FILENAME, SHA256, generation_facts
    from framework.rag import LocalKnowledgeBase
    from llama_cpp.llama_chat_format import format_chatml
    model_path = ROOT / "artifacts/llm/qwen" / FILENAME
    if digest(model_path) != SHA256:
        raise ValueError("Local Qwen digest mismatch")
    corpus = json.loads((ROOT / "evaluation/fixtures/rust_migration/v1/rag/grounding.json").read_text(encoding="utf-8"))
    selected = [next(case for case in corpus["cases"] if case["id"] == key) for key in ["faithful", "partial_missing_routing", "rule_disagreement"]]
    results = []
    for case in selected:
        evidence = copy.deepcopy(case["input"]["evidence"])
        retrieval = LocalKnowledgeBase().retrieve(evidence, max_results=2)
        facts = generation_facts(evidence)
        supplied = "LOCAL REFERENCE NOTES:\n" + retrieval["context"] + "\n\nFACT SENTENCES TO JOIN:\n" + "\n".join(facts)
        formatted = format_chatml(messages=[{"role": "system", "content": system_prompt()}, {"role": "user", "content": supplied}]).prompt
        prefix = output / case["id"]
        prompt = prefix.with_suffix(".prompt.txt")
        prompt.write_text(formatted, encoding="utf-8", newline="\n")
        request = prefix.with_suffix(".request.json")
        dump(request, {"model": str(model_path), "native_dir": str(native_dir), "prompt_file": str(prompt),
                       "evidence": evidence, "passages": retrieval["passages"], "timeout_seconds": 90})
        dump(prefix.with_suffix(".prompt-audit.json"), {"facts": facts, "retrieval": retrieval, "prompt_sha256": digest(prompt),
                                                       "evidence_origin": "Sealed Phase 0 synthetic software fixture"})
        results.append((case["id"], request))
    return results


def run(args):
    from framework.rag.grounding import check_grounding
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    native_dir = args.native_dir.resolve()
    cases = prepare(output, native_dir)
    rust = args.rust_binary.resolve()
    results = []
    for case_id, request in ([] if args.checks_only else cases):
        request_data = json.loads(request.read_text(encoding="utf-8"))
        for repeat in range(args.repeats):
            for mode in ["python", "child", "native"]:
                prefix = output / f"{case_id}-{mode}-{repeat + 1}"
                result_file = prefix.with_suffix(".json")
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(request), str(result_file)] if mode == "python" else [str(rust), mode, str(request), str(result_file)]
                measured = execute(command, prefix)
                generated = json.loads(result_file.read_text(encoding="utf-8"))
                expected = check_grounding(generated["text"], request_data["evidence"], request_data["passages"], generated["finish_reason"])
                record = {"case": case_id, "mode": mode, "repeat": repeat + 1, "measurements": measured,
                          "text": generated["text"], "finish_reason": generated["finish_reason"], "python_grounding": expected,
                          "grounding_equal_to_python": generated["grounding"] == expected,
                          "generation": {key: generated[key] for key in ["load_seconds", "llama_reported_load_seconds", "first_token_seconds", "generation_total_seconds", "usage", "unloaded_after_generation", "timing_instrumentation", "child_reaped"] if key in generated}}
                results.append(record)
                dump(output / "measurements.json", {"schema_version": 1, "synthetic_evidence": True, "actual_qwen_inference": True,
                    "settings": {"n_ctx": 2048, "n_threads": 2, "n_threads_batch": 2, "n_batch": 128, "n_ubatch": 128, "n_gpu_layers": 0, "seed": 42, "chat_format": "chatml", "max_tokens": 200, "temperature": 0, "top_p": 1, "repeat_penalty": 1, "flash_attention": False},
                    "model_sha256": digest(Path(request_data["model"])), "native_release": "b11564", "native_files": {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in native_dir.iterdir() if p.suffix in {".dll", ".exe"}},
                    "protected_source_sha256": {name: digest(ROOT / name) for name in ["framework/llm/local_model.py", "framework/rag/grounding.py", "framework/rag/knowledge.json"]},
                    "results": results,
                    "limitations": ["Three synthetic evidence cases; no semantic faithfulness, detector accuracy or real-incident validation", "File/page cache warmed; order fixed Python, child, bindings; single machine under concurrent independent training", "Python native build differs from b11564; child and Rust binding use exactly the same b11564 release", "Peak RSS is 10ms sampled simultaneous sum for owned processes; process peak working sets are OS counters, not unique allocated physical memory"]})
                print(f"{case_id} {mode} run {repeat + 1}: {measured['wall_seconds']:.3f}s, {measured['peak_tree_rss_bytes'] / 2**20:.1f} MiB, grounding {expected['passed']}, parity {record['grounding_equal_to_python']}", flush=True)
    golden = ROOT / "evaluation/fixtures/rust_migration/v1/rag/grounding.json"
    grounding_output = output / "grounding-rust.json"
    execute([str(rust), "grounding", str(golden), str(grounding_output)], output / "grounding-corpus")
    got = json.loads(grounding_output.read_text(encoding="utf-8"))
    expected = json.loads(golden.read_text(encoding="utf-8"))
    equal = [actual["output"] == oracle["output"] for actual, oracle in zip(got["cases"], expected["cases"])]
    dump(output / "grounding-parity.json", {"cases": len(equal), "equal": sum(equal), "passed": all(equal), "source": str(golden.relative_to(ROOT))})
    if not all(equal) or not all(row["grounding_equal_to_python"] for row in results):
        raise ValueError("Rust grounding parity failed")
    refusal_checks = []
    unverified = output / "unverified-small-fixture.gguf"
    unverified.write_bytes(b"not a model: explicit integrity refusal fixture\n")
    for name, mode, change, message in [
        ("owned_child_deadline", "child", {"timeout_seconds": 0}, "Owned llama.cpp process deadline exceeded"),
        ("native_deadline", "native", {"timeout_seconds": 0}, "Native generation deadline exceeded"),
        ("missing_native", "native", {"native_dir": str(output / "absent-native")}, "The system cannot find the path specified"),
        ("missing_weights", "child", {"model": str(output / "absent-model.gguf")}, "The system cannot find the file specified"),
        ("unverified_weights", "native", {"model": str(unverified)}, "Local Qwen weights differ from the verified official model"),
    ]:
        data = json.loads(cases[0][1].read_text(encoding="utf-8"))
        data.update(change)
        request = output / f"{name}.request.json"
        destination = output / f"{name}.response.json"
        dump(request, data)
        measurements = execute([str(rust), mode, str(request), str(destination)], output / name, allow_failure=True)
        diagnostics = (output / name).with_suffix(".stderr.log").read_text(encoding="utf-8")
        passed = measurements["returncode"] != 0 and message in diagnostics and not destination.exists() and not measurements["remaining_owned_pids"]
        refusal_checks.append({"case": name, "passed": passed, "measurements": measurements, "explicit_error": diagnostics.strip(), "substitute_returned": destination.exists()})
    dump(output / "refusal-checks.json", {"cases": refusal_checks, "passed": all(c["passed"] for c in refusal_checks)})
    if not all(c["passed"] for c in refusal_checks):
        raise ValueError("LLM refusal/owned-process cleanup failed")
    if results:
        path = output / "measurements.json"
        recorded = json.loads(path.read_text(encoding="utf-8"))
        recorded["text_byte_equal_by_case"] = {case: len({row["text"] for row in results if row["case"] == case}) == 1 for case, _ in cases}
        recorded["spike_source_sha256"] = {name: digest(ROOT / name) for name in ["tools/spike_llm.py", "rust/spikes/llm/Cargo.lock", "rust/spikes/llm/src/adapter.rs", "rust/spikes/llm/src/grounding.rs", "rust/spikes/llm/src/main.rs", "rust/spikes/llm/src/lib.rs"]}
        recorded["grounding_corpus_cases"] = len(equal)
        recorded["grounding_corpus_equal"] = sum(equal)
        recorded["refusal_checks_passed"] = all(c["passed"] for c in refusal_checks)
        dump(path, recorded)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]), Path(sys.argv[3]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-binary", type=Path, default=ROOT / "rust/spikes/llm/target/release/netsec-llm-spike.exe")
    parser.add_argument("--native-dir", type=Path, default=ROOT / "artifacts/rust_migration/phase1/llm/native-b11564")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/rust_migration/phase1/llm/measurements")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--checks-only", action="store_true", help="Run lexical corpus and refusal checks without repeating measured generation")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error("repeats must be between 1 and 10")
    run(args)


if __name__ == "__main__":
    main()
