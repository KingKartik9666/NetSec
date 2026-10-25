"""Windows launcher ancestry must not permanently block original-model CUDA fits."""
from types import SimpleNamespace


def test_controller_launcher_is_not_a_competing_fit(monkeypatch):
    import psutil
    from experiment import train_original_four_compat as adapter
    from experiment import run_xgboost_followup
    processes = [SimpleNamespace(pid=100,info=dict(cmdline=['python.exe','-m','experiment.train_verified_clean','_run']))]
    monkeypatch.setattr(psutil,'process_iter',lambda attrs: processes)
    monkeypatch.setattr(run_xgboost_followup,'memory_available',lambda:dict(physical_gib=8.,commit_gib=12.))
    assert adapter.available_resources(gpu=True)


def test_tabm_worker_blocks_original_cuda_but_allows_memory_guarded_cpu(monkeypatch):
    import psutil
    from experiment import train_original_four_compat as adapter
    from experiment import run_xgboost_followup
    processes = [SimpleNamespace(pid=100,info=dict(cmdline=['python.exe','-m','experiment.train_tabm','--fit','request.json']))]
    monkeypatch.setattr(psutil,'process_iter',lambda attrs: processes)
    monkeypatch.setattr(run_xgboost_followup,'memory_available',lambda:dict(physical_gib=8.,commit_gib=12.))
    assert not adapter.available_resources(gpu=True)
    assert adapter.available_resources(gpu=False)
    monkeypatch.setattr(run_xgboost_followup,'memory_available',lambda:dict(physical_gib=3.,commit_gib=12.))
    assert not adapter.available_resources(gpu=False)
