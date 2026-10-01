"""Completion is exact trial coverage, not just a matching record count."""
from experiment.summarize_full_dataset import has_complete_coverage


def test_completeness_requires_missing_settings_and_selected_repeat_seeds():
    plan = {"configurations": {"xgboost": [{"id": "a"}, {"id": "b"}], "kan": [{"id": "c"}]},
            "search_seed": 42, "repeat_seeds": [123, 2025]}

    def record(name, config, seed, score):
        return {"candidate": name, "config_id": config, "seed": seed, "metrics": {"macro_f1": score}}

    records = [record("xgboost", "a", 42, .7), record("xgboost", "b", 42, .6), record("kan", "c", 42, .5),
               record("xgboost", "a", 123, .68), record("xgboost", "a", 2025, .72),
               record("kan", "c", 123, .52), record("kan", "c", 2025, .51)]
    assert has_complete_coverage(plan, records)
    assert not has_complete_coverage(plan, records[:-1])
    # The count still matches, but a repeated record cannot stand in for a missing trial.
    assert not has_complete_coverage(plan, [*records[:-1], records[0]])
    assert not has_complete_coverage(plan, [*records[:-1], record("kan", "c", 2026, .51)])
    assert not has_complete_coverage(plan, [*records[:3], record("xgboost", "b", 123, .9), *records[4:]])
