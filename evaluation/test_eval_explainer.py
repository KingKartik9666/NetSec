"""Offline checks for tools/eval_explainer.py helpers; no runtime, model or held-out data is used."""
from __future__ import annotations

import csv

from tools.eval_explainer import blind_rows, recheck, select_cases, template_text


def sample(i, label):
    return {"source_row_id": str(1000 + i), "label": label, "origin": "validation"}


def test_select_cases_takes_ten_wrong_and_spreads_correct_over_classes():
    labels = ["benign"] * 30 + [f"class_{k}" for k in range(40) for _ in range(2)] + ["recon_scan"] * 18
    samples = [sample(i, label) for i, label in enumerate(labels)]
    predicted = [label if i % 9 else "benign_wrong" for i, label in enumerate(labels)]
    cases = select_cases(samples, predicted)
    assert len(cases) == 60 and len({c["source_row_id"] for c in cases}) == 60
    assert sum(c["detector_wrong"] for c in cases) == 10
    correct = [c for c in cases if not c["detector_wrong"]]
    assert len({c["label"] for c in correct}) >= 40, "one row per class before any class repeats"
    assert cases == select_cases(samples, predicted), "selection is deterministic"
    assert all(c["origin"] == "validation" for c in cases)


def test_select_cases_refuses_non_validation_rows():
    samples = [sample(i, "benign") for i in range(70)]
    samples[3]["origin"] = "test"
    try:
        select_cases(samples, ["benign"] * 70)
    except ValueError as error:
        assert "validation" in str(error)
    else:
        raise AssertionError("held-out rows must never enter the evaluation")


EVIDENCE = {"numbers": [0.93, 0.04, 77.0, 15234.7, 0.4187, -0.1203], "features": {"flow_bytes_per_s", "syn_flag_count"},
            "all_features": {"flow_bytes_per_s", "syn_flag_count", "ack_flag_count", "network_ports_src__present_1883"},
            "classes": {"ddos_icmp-flood", "dos_icmp-flood"}, "all_classes": {"ddos_icmp-flood", "dos_icmp-flood", "recon_scan", "ddos_syn-flood-port-1883"}}


def test_recheck_flags_invented_numbers_features_and_attacks():
    assert recheck("The model predicted ddos icmp flood with a confidence of 93%.", EVIDENCE) == []
    assert recheck("It weighed 77 classes; flow_bytes_per_s was 15234.7 and added 0.42.", EVIDENCE) == []
    assert recheck("The confidence was 0.95.", EVIDENCE) == ["number: 0.95"]
    assert recheck("ack_flag_count raised the score.", EVIDENCE) == ["feature: ack_flag_count"]
    assert recheck("This looks like recon scan traffic.", EVIDENCE) == ["attack: recon_scan"]
    assert recheck("It is not ddos syn flood port 1883.", EVIDENCE) == ["attack: ddos_syn-flood-port-1883"], "digits inside a class name are not numbers"
    # As the runtime verifier: digit runs of the evidence's own feature and class names are evidence numbers (Task 1 FIX, Ruling 12).
    ports = dict(EVIDENCE, features=EVIDENCE["features"] | {"network_ports_src__present_1883"}, classes=EVIDENCE["classes"] | {"dos_syn-flood-port-80"})
    assert recheck("Source port 1883 pushed it toward this class, and flooding port 80 was less likely.", ports) == []
    assert recheck("Source port 1883 pushed it toward this class.", EVIDENCE) == ["number: 1883"]
    assert recheck("Port 4444 was active.", ports) == ["number: 4444"]


def blind_turn(case, model, template, filled=()):
    sentences = [{"text": t, "source": "model"} for t in model] + [{"text": t, "source": "template"} for t in filled]
    return {"case": case, "kind": "initial", "question": "Explain this result in plain English", "text": " ".join(model + list(filled)),
            "sentences": sentences, "template": template, "model_sentences": len(model), "template_sentences": len(filled)}


def test_template_text_and_blind_rows_hide_the_source(tmp_path):
    # Owner redesign 3 Nov: a method preset's part-2 template S1 and part-3 templates W1..W6 (the P/X/C caveat points are gone).
    points = [{"id": i, "text": f"{i} text."} for i in ("S1", "W1", "W2", "W3")]
    assert template_text(points, "initial") == "S1 text. W1 text. W2 text. W3 text."
    assert template_text(points, "summary") == "W1 text."
    assert template_text(points, "simpler") == template_text(points, "other") == "S1 text. W1 text."
    assert template_text(points, "detail") == "S1 text. W1 text. W2 text. W3 text."
    turns = [blind_turn(c, [f"model {c}.", "Second model sentence."], f"template {c}.") for c in range(6)]
    rows, key, excluded = blind_rows(turns, seed=7)
    assert len(rows) == 6 and {r["item"] for r in rows} == set(key) and excluded == {"no_model_sentence": 0, "identical_to_template": 0}
    assert {k["model_is"] for k in key.values()} == {"A", "B"}, "the model side is randomised"
    for row in rows:
        side = key[row["item"]]["model_is"]
        assert row[f"text_{side}"].startswith("model") and row["text_" + ("B" if side == "A" else "A")].startswith("template")
        assert row["rating_A"] == row["rating_B"] == row["preferred"] == "" and "model" not in row["item"]
    path = tmp_path / "blind.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert "Model" not in path.read_text(encoding="utf-8").split("\n")[0]


def test_blind_rows_compare_only_model_sentences_with_the_template():
    """Review fix (Task 7 fix round 1): a fully templated turn would pair the template with itself, and template fill inside the
    'model' side would reveal it, so the model side is the turn's verified model sentences only and such turns are excluded."""
    turns = [blind_turn(0, [], "T0 one. T0 two.", filled=["T0 one.", "T0 two."]),
             blind_turn(1, ["The model said this."], "T1 one. T1 two.", filled=["T1 two."]),
             blind_turn(2, ["T2 one.", "T2 two."], "T2 one.  T2 two."),
             blind_turn(3, ["Plain words here.", "More plain words."], "T3 one.")]
    rows, key, excluded = blind_rows(turns, seed=1)
    assert excluded == {"no_model_sentence": 1, "identical_to_template": 1}
    assert sorted(k["case"] for k in key.values()) == [1, 3]
    assert all(row["text_A"] != row["text_B"] for row in rows), "no item pairs a text with itself"
    for row in rows:
        k = key[row["item"]]
        model, template = row[f"text_{k['model_is']}"], row["text_" + ("B" if k["model_is"] == "A" else "A")]
        assert "T1 two." not in model and "T1" not in model, "template fill never appears on the model side"
        assert template == turns[k["case"]]["template"]
    assert key[next(i for i, k in key.items() if k["case"] == 1)]["template_sentences_removed"] == 1
    assert {k["model_is"] for k in key.values()} == {"A", "B"}, "sides stay balanced over the kept items"


def test_summary_states_the_recheck_scope_and_blind_exclusions(tmp_path):
    import json
    from tools.eval_explainer import write_outputs
    turn = dict(blind_turn(0, ["The model predicted Normal traffic."], "P1 text."), first_violation=None, violation=None, passed=True, detector_wrong=False,
                first_token_seconds=0.1, seconds=1.0, llm_executed=True, labelled=True)
    turn["sentences"][0]["recheck"] = []
    empty = dict(blind_turn(1, [], "P1 text.", filled=["P1 text."]), first_violation="number", violation="number: 9", passed=False, detector_wrong=True,
                 first_token_seconds=0.1, seconds=1.0, llm_executed=True, labelled=True)
    cases = [{"detector_wrong": False}, {"detector_wrong": True}]
    summary = write_outputs(tmp_path, [turn, empty], cases, {"available": False}, "gpu")
    scope = summary["independent_recheck_of_shown_model_sentences"]["scope"]
    assert "paraphrase" in scope and "not factual correctness" in scope
    assert summary["blind_readability"]["items"] == 1 and summary["blind_readability"]["excluded"] == {"no_model_sentence": 1, "identical_to_template": 0}
    key = json.loads((tmp_path / "blind_key.json").read_text(encoding="utf-8"))
    assert key["excluded"] == summary["blind_readability"]["excluded"] and len(key["items"]) == 1 and "model sentences only" in key["pairing"]
    lines = (tmp_path / "blind_readability.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and "P1 text." in lines[1] and "The model predicted Normal traffic." in lines[1]
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == summary
