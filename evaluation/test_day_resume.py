"""An explicit resume restores unused active time without rewriting history."""
import pytest


def test_resume_restores_only_remaining_active_time():
    from experiment.resume_all_models_day import continuation_budget
    original={'started_at':100.,'deadline':86500.}
    result=continuation_budget(original,{'requested_at':100.+14*3600},200000.)
    assert result['started_at']==200000.-14*3600
    assert result['deadline']==200000.+10*3600
    assert original=={'started_at':100.,'deadline':86500.}


def test_resume_cannot_silently_create_a_new_day_after_budget_exhaustion():
    from experiment.resume_all_models_day import continuation_budget
    with pytest.raises(TimeoutError):
        continuation_budget({'started_at':100.,'deadline':86500.},{'requested_at':90000.},200000.)
