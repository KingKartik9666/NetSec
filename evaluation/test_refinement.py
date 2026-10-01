from experiment.refine_candidates import proposals
from experiment.train_candidates import configurations


def test_refinement_is_controlled_before_testing_a_combination():
    for candidate, initial in configurations().items():
        anchor = {"config": initial[0]["parameters"], "config_id": "baseline"}
        specs = proposals(candidate, anchor)
        assert len(specs) == 5
        assert all(len(s["changes"]) == 1 for s in specs[:4])
        assert len(specs[4]["changes"]) == 2
        assert all(s["parameters"][k] == v for s in specs for k, v in s["changes"].items())
