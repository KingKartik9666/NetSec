"""Suite-wide guard: no CUDA work while a training controller or runner job is alive.

The CUDA tests only skip when no GPU exists, so a plain ``pytest evaluation`` (required after every change) would
start real CUDA fits next to a running training chain (e.g. the detached fix77 chain) and could cost it an attempt
through an out-of-memory error. When ``upgrades2.cpu_only_reason()`` reports a controller/job (or
``IMDAI_TESTS_CPU_ONLY=1``, or the check itself fails), the GPU is hidden before any test module is imported:
``CUDA_VISIBLE_DEVICES`` is set to an empty string (inherited by test subprocesses; torch then reports no CUDA
device, so the existing ``torch.cuda.is_available()`` guards skip) and ``torch.cuda.is_available`` returns False
in this process. Run the full suite again with the GPU after the queue has finished to cover the CUDA tests.
"""
import os

CPU_ONLY_REASON = None


def pytest_configure(config):
    global CPU_ONLY_REASON
    from framework.training.csv_v3 import upgrades2

    CPU_ONLY_REASON = upgrades2.cpu_only_reason()
    if CPU_ONLY_REASON is None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import torch
    except ImportError:
        return
    torch.cuda.is_available = lambda: False


def pytest_report_header(config):
    if CPU_ONLY_REASON is None:
        return "CUDA tests enabled (no training controller or runner job alive)"
    return f"CPU-only test mode: {CPU_ONLY_REASON}; CUDA tests skip (re-run them when the GPU is free)"
