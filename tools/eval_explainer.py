"""Report evaluation of the locked XAI explainer on fixed validation cases (plan Task 7 Step 4).

60 fixed cases from the 128 included validation rows (never the locked test partition), about 10 of them rows the
detector gets wrong. Each case gets the initial answer plus the simpler, summary and detail follow-ups through the
real runtime and model. Outputs (artifacts/locked_explainer_20261031/eval/):
  cases.json, turns.json  every case and turn, with each sentence's label and the independent re-check flags
  summary.json            invented-number / invented-feature / wrong-attack rates, template fallback, latency p50/p95
  blind_readability.csv   verified model sentences vs template text (turns without one excluded), sides randomised, ratings empty
  blind_key.json          which side is the model, and the exclusions; ratings are "pending human rater", never filled in here
The re-check is an independent heuristic over the shown model sentences; its flags are for manual review and do
not replace the runtime verifier. No training, packet capture or held-out test data is involved.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import re
import sys
import time

# Runnable as `.venv/Scripts/python.exe tools/eval_explainer.py` (plan command form) as well as with -m.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.verify_native_app import (FIRST_TOKEN_LIMIT, GPU_FREE_MIB, QUESTIONS, RELEASE, ROOT, Evidence, NativeRuntime,  # noqa: E402
                                     chat_turn, gpu_state, llama_servers, percentile, wait_language_verified)

OUTPUT = ROOT / "artifacts/locked_explainer_20261031/eval"
# Mirrors explain::template_ids in runtime/src/language/explain.rs.
# Owner redesign 3 Nov 2026: the initial explanation's two parts fall back to S1 (what the XAI shows) and W1..W6 (why), a follow-up to its kind's list.
WHYS = ["W1", "W2", "W3", "W4", "W5", "W6"]
TEMPLATE_IDS = {"initial": ["S1", *WHYS], "simpler": ["S1", "W1"], "other": ["S1", "W1"], "summary": ["W1"], "detail": ["S1", *WHYS]}
NUMBER = re.compile(r"(?<![\w.])-?(?:\d+(?:\.\d+)?|\.\d+)%?")
# Plan Task 7 Step 4 (amended by Task 7 ruling 2): what the re-check and the runtime verifier cover, and what they do not.
RECHECK_SCOPE = ("The re-check covers only catalog feature names, catalog class (attack) names and numbers; the runtime verifier also checks "
                 "direction where a packet feature is named, and banned words. Neither checks paraphrased claims (a feature described in other "
                 "words, or an effect stated against the wrong class), so 0 flags is not factual correctness. Known shown errors outside this "
                 "scope: inverted-direction sentences such as 'A higher number of source ports and more unique MAC addresses slightly reduced "
                 "the likelihood of an attack.' on a Normal prediction whose features push away from Normal (live verify_native_chat on build "
                 "aa7cba3b; eval cases 3 and 10); routed to Task 8 Step 0.")


def select_cases(samples, predicted, count=60, wrong=10):
    """Fixed, deterministic case list: `wrong` evenly spaced detector errors, the rest one correct row per class in turn."""
    if any(s.get("origin") != "validation" for s in samples):
        raise ValueError("Evaluation cases must come from the included validation rows only")
    order = sorted(range(len(samples)), key=lambda i: int(samples[i]["source_row_id"]))
    errors = [i for i in order if predicted[i] != samples[i]["label"]]
    take = min(wrong, len(errors))
    chosen = [errors[int(j * len(errors) / take)] for j in range(take)] if take else []
    by_class = {}
    for i in order:
        if predicted[i] == samples[i]["label"]:
            by_class.setdefault(samples[i]["label"], []).append(i)
    queues = [by_class[label] for label in sorted(by_class)]
    while len(chosen) < count and any(queues):
        for queue in queues:
            if queue and len(chosen) < count:
                chosen.append(queue.pop(0))
    return [{"index": i, "source_row_id": samples[i]["source_row_id"], "label": samples[i]["label"], "predicted": predicted[i],
             "detector_wrong": predicted[i] != samples[i]["label"], "origin": samples[i]["origin"]} for i in chosen]


def _forms(name):
    lower = name.lower()
    return {lower, re.sub(r"[_\-]+", " ", lower).strip()}


def _find(text, form):
    return [m.span() for m in re.finditer(r"(?<![\w])" + re.escape(form) + r"(?![\w])", text)]


def recheck(sentence, evidence):
    """Independent heuristic re-check of one shown model sentence: catalog features and classes not in the evidence,
    and numbers that match no evidence number at their stated precision. Names are masked before numbers are read."""
    text, flags = sentence.lower(), []
    names = [(form, name, "feature", name in evidence["features"]) for name in evidence["all_features"] | evidence["features"] for form in _forms(name)]
    names += [(form, name, "attack", name in evidence["classes"]) for name in evidence["all_classes"] for form in _forms(name)]
    for form, name, kind, allowed in sorted(names, key=lambda n: -len(n[0])):
        for start, end in _find(text, form):
            if not text[start:end].strip():
                continue
            if not allowed and f"{kind}: {name}" not in flags:
                flags.append(f"{kind}: {name}")
            text = text[:start] + " " * (end - start) + text[end:]
    # As the runtime verifier: digit runs inside the evidence's own feature and class names count as evidence numbers.
    values = [float(v) for v in evidence["numbers"]] + [float(d) for name in evidence["features"] | evidence["classes"] for d in re.findall(r"\d+", name)]
    candidates = values + [abs(v) for v in values] + [v * 100 for v in values] + [abs(v) * 100 for v in values]
    for match in NUMBER.finditer(text):
        token = match.group(0)
        number = float(token.rstrip("%"))
        decimals = len(token.rstrip("%").split(".")[1]) if "." in token else 0
        if not any(abs(c - number) <= 0.5 * 10 ** -decimals + 1e-9 for c in candidates):
            flags.append(f"number: {token}")
    return flags


def template_text(points, kind):
    by_id = {p["id"]: p["text"] for p in points}
    return " ".join(by_id[i] for i in TEMPLATE_IDS[kind] if i in by_id)


BLIND_FIELDS = ("item", "kind", "question", "text_A", "text_B", "rating_A", "rating_B", "preferred", "rater_notes")
BLIND_PAIRING = ("model side = the turn's verified model sentences only (template fill removed); other side = the turn's template baseline. "
                 "Turns with no model sentence, or whose model text equals the template, are excluded and counted. "
                 "Readability ratings say nothing about factual correctness.")


def blind_rows(turns, seed=20261102):
    """Pairs each turn's verified model sentences (never its template fill, which would reveal the side) with its template
    baseline. Turns with no model sentence, or whose model text equals the template, would pair a text with itself and are
    excluded and counted. The model side is balanced and shuffled over the kept items, and the key is separate."""
    pairs, excluded = [], {"no_model_sentence": 0, "identical_to_template": 0}
    for turn in turns:
        model = " ".join(s["text"] for s in turn["sentences"] if s["source"] == "model")
        if not model:
            excluded["no_model_sentence"] += 1
        elif " ".join(model.split()) == " ".join(turn["template"].split()):
            excluded["identical_to_template"] += 1
        else:
            pairs.append((turn, model))
    rng = random.Random(seed)
    sides = (["A", "B"] * math.ceil(len(pairs) / 2))[:len(pairs)]
    rng.shuffle(sides)
    rows, key = [], {}
    for index, ((turn, model), side) in enumerate(zip(pairs, sides)):
        item = f"R{index + 1:03d}"
        other = "B" if side == "A" else "A"
        row = {"item": item, "kind": turn["kind"], "question": turn["question"], f"text_{side}": model, f"text_{other}": turn["template"],
               "rating_A": "", "rating_B": "", "preferred": "", "rater_notes": ""}
        rows.append({k: row[k] for k in BLIND_FIELDS})
        key[item] = {"model_is": side, "case": turn["case"], "kind": turn["kind"], "model_sentences": turn["model_sentences"],
                     "template_sentences_removed": turn["template_sentences"], "rating_status": "pending human rater"}
    return rows, key, excluded


def evidence_for(prediction, explanation, points, catalog):
    """Everything the shown sentences may cite, from server-owned data: prediction, XAI rows, class count and template points."""
    numbers = [prediction.get("score") or 0.0, float(len(catalog["schema"]["classes"]))]
    numbers += [c.get("score") for c in prediction.get("top_classes") or [] if c.get("score") is not None]
    features = set()
    for row in (explanation or {}).get("features") or []:
        name = row["feature"].split(" / group ")[0]
        features.add(name.split("__")[0] if (row.get("group_size") or 1) > 1 else name)
        numbers += [v for v in (row.get("value"), row.get("contribution"), row.get("group_size")) if isinstance(v, (int, float))]
    for point in points:
        numbers += [float(t.rstrip("%")) for t in NUMBER.findall(point["text"])]
    classes = {prediction["label"]} | {c["label"] for c in prediction.get("top_classes") or []}
    return {"numbers": numbers, "features": features, "all_features": set(catalog["schema"]["feature_names"]),
            "classes": classes, "all_classes": set(catalog["schema"]["classes"])}


def summarise(turns, cases, environment, device):
    shown = [s for t in turns for s in t["sentences"] if s["source"] == "model"]
    flags = [f for t in turns for s in t["sentences"] if s["source"] == "model" for f in s["recheck"]]
    rejected = [t for t in turns if t["first_violation"] in ("number", "feature", "attack", "direction", "banned")]
    attempted = len(shown) + len(rejected)
    def rate(n, d):
        return n / d if d else None
    categories = {"number": "invented_number", "feature": "invented_feature", "attack": "wrong_attack"}
    violations = {}
    for t in turns:
        violations[t["first_violation"] or "none"] = violations.get(t["first_violation"] or "none", 0) + 1
    def latency(rows):
        return {"turns": len(rows), "first_token_p50": percentile([t["first_token_seconds"] for t in rows], .5),
                "first_token_p95": percentile([t["first_token_seconds"] for t in rows], .95),
                "total_p50": percentile([t["seconds"] for t in rows], .5), "total_p95": percentile([t["seconds"] for t in rows], .95),
                "first_token_over_15s": sum(1 for t in rows if t["first_token_seconds"] is None or t["first_token_seconds"] > FIRST_TOKEN_LIMIT)}
    all_sentences = [s for t in turns for s in t["sentences"]]
    return {
        "cases": len(cases), "detector_wrong_cases": sum(c["detector_wrong"] for c in cases), "turns": len(turns), "device": device,
        "environment": environment, "partition": "validation (included samples.json rows); locked test partition not read",
        "shown_model_sentences": len(shown), "template_sentences": sum(1 for s in all_sentences if s["source"] == "template"),
        "independent_recheck_of_shown_model_sentences": {
            "method": "heuristic: catalog feature/class names not in the evidence, numbers matching no evidence number at stated precision; flags need manual review",
            "scope": RECHECK_SCOPE,
            **{f"{name}_rate": rate(sum(1 for f in flags if f.startswith(k + ":")), len(shown)) for k, name in categories.items()},
            "flags": [{"case": t["case"], "kind": t["kind"], "text": s["text"], "flags": s["recheck"]} for t in turns for s in t["sentences"] if s["source"] == "model" and s["recheck"]]},
        "runtime_verifier_rejections": {
            "method": "model sentences the runtime verifier stopped (never shown) / model sentences attempted (shown + stopped)",
            **{f"{name}_rate": rate(sum(1 for t in rejected if t["first_violation"] == k), attempted) for k, name in categories.items()},
            "attempted_model_sentences": attempted},
        "template_fallback_rate": rate(sum(1 for t in turns if not t["passed"]), len(turns)),
        "template_sentence_share": rate(sum(1 for s in all_sentences if s["source"] == "template"), len(all_sentences)),
        "first_violation_counts": violations,
        "direction_violations": [t["violation"] for t in turns if t["first_violation"] == "direction"],
        "latency_all": latency(turns), "latency_by_kind": {k: latency([t for t in turns if t["kind"] == k]) for k, _ in QUESTIONS},
        "detector_wrong_subset": {"turns": sum(1 for t in turns if t["detector_wrong"]),
                                  "template_fallback_rate": rate(sum(1 for t in turns if t["detector_wrong"] and not t["passed"]), sum(1 for t in turns if t["detector_wrong"]))},
        "readability": "pending human rater (blind_readability.csv); no rating is produced or inferred by this tool",
    }


def _save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_outputs(output, turns, cases, environment, device, note=None):
    """summary.json, blind_readability.csv and blind_key.json from the recorded turns (no runtime or model needed)."""
    output = Path(output)
    summary = summarise(turns, cases, environment, device)
    rows, key, excluded = blind_rows(turns)
    summary["blind_readability"] = {"items": len(rows), "excluded": excluded, "pairing": BLIND_PAIRING}
    if note:
        summary["note"] = note
    with (output / "blind_readability.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(BLIND_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    _save(output / "blind_key.json", {"pairing": BLIND_PAIRING, "excluded": excluded, "items": key})
    _save(output / "summary.json", summary)
    return summary


def rebuild(output):
    """Re-derives the summary and the blind pair files from a finished run's turns.json/cases.json; report.json is untouched."""
    output = Path(output)
    turns, cases = (json.loads((output / name).read_text(encoding="utf-8")) for name in ("turns.json", "cases.json"))
    previous = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    summary = write_outputs(output, turns, cases, previous["environment"], previous["device"],
                            note="Rebuilt offline from this folder's turns.json and cases.json (Task 7 fix round 1: fair blind pairs, re-check scope); no model run.")
    print(json.dumps({"summary": str(output / "summary.json"), "blind_items": summary["blind_readability"]["items"],
                      "excluded": summary["blind_readability"]["excluded"]}), flush=True)
    return 0


def run(args):
    output = Path(args.output)
    if args.rebuild:
        return rebuild(output)
    evidence = Evidence(output)
    if llama_servers():
        evidence.check("no other llama-server is running (one-server rule)", False, running=llama_servers())
        return 1
    environment = gpu_state()
    device, env = "gpu", {}
    if args.cpu or not (environment.get("available") and environment["memory_free_mib"] >= GPU_FREE_MIB):
        device, env = "cpu", {"NETSEC_LLM_DEVICE": "cpu"}
    turns, cases = [], []
    with NativeRuntime(ROOT, args.binary, evidence, "eval", env=env) as runtime:
        language = wait_language_verified(runtime)
        if not evidence.check("language model files verified by the background hash pass", language.get("integrity") == "verified", language=language):
            return 1
        catalog = runtime.json("/api/v1/catalog")
        samples = runtime.json("/api/v1/samples")["samples"]
        job = runtime.json("/api/v1/jobs", {"model_ids": [args.model], "rows": [s["row"] for s in samples]})
        job, _, _ = runtime.wait_job(job["id"])
        predictions = job["models"][0]["predictions"]
        cases = select_cases(samples, [p["label"] for p in predictions], args.cases, args.wrong)[:args.limit or None]
        evidence.artifact("cases.json", cases)
        for number, case in enumerate(cases):
            binding = {"job_id": job["id"], "model_id": args.model, "row_index": case["index"]}
            xai = runtime.json("/api/v1/explain", dict(binding, method="shap"), timeout=180)
            context = runtime.json("/api/v1/chat/context", dict(binding, explanation_id=xai["explanation_id"]))
            allowed = evidence_for(predictions[case["index"]], xai, context["points"], catalog)
            case["evidence"] = {"numbers": allowed["numbers"], "features": sorted(allowed["features"]), "classes": sorted(allowed["classes"])}
            for kind, body in QUESTIONS:
                t = chat_turn(runtime, context["id"], body)
                for sentence in t["sentences"]:
                    sentence["recheck"] = recheck(sentence["text"], allowed) if sentence["source"] == "model" else []
                turns.append(dict(t, case=number, source_row_id=case["source_row_id"], label=case["label"], predicted=case["predicted"],
                                  detector_wrong=case["detector_wrong"], kind=kind, question=body.get("message", "Explain this result in plain English"),
                                  template=template_text(context["points"], kind)))
            print(json.dumps({"case": number, "row": case["source_row_id"], "wrong": case["detector_wrong"],
                              "passed": [t["passed"] for t in turns[-4:]], "first_token": [t["first_token_seconds"] for t in turns[-4:]]}), flush=True)
            evidence.artifact("turns.json", turns)
    evidence.artifact("cases.json", cases)  # now with each case's allowed evidence, so the re-check can be repeated offline
    summary = write_outputs(output, turns, cases, environment, device)
    evidence.check("every turn executed the local model with labelled sentences", all(t["llm_executed"] and t["labelled"] for t in turns), turns=len(turns))
    evidence.check("no shown model sentence was flagged by the independent re-check", not summary["independent_recheck_of_shown_model_sentences"]["flags"],
                   flags=len(summary["independent_recheck_of_shown_model_sentences"]["flags"]))
    evidence.data["passed"] = all(c["passed"] for c in evidence.data["checks"])
    evidence.save()
    print(json.dumps({"summary": str(output / "summary.json"), "passed": evidence.data["passed"]}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--binary", type=Path, default=RELEASE)
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--cases", type=int, default=60)
    parser.add_argument("--wrong", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases (smoke run)")
    parser.add_argument("--cpu", action="store_true", help="Force the CPU language server (NETSEC_LLM_DEVICE=cpu)")
    parser.add_argument("--rebuild", action="store_true", help="Rewrite summary.json and the blind pair files from --output's recorded turns; starts nothing")
    started = time.monotonic()
    code = run(parser.parse_args())
    print(json.dumps({"seconds": time.monotonic() - started}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
