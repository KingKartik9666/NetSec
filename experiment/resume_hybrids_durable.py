"""Versioned Windows JSON-retry adapter; sealed research recipes stay unchanged."""
from __future__ import annotations

from pathlib import Path

from experiment import optimize_hybrids as original
from experiment.datasense_quality import digest
from framework.training.study_supervisor import atomic_write, read

MODULE = 'experiment.resume_hybrids_durable'


def configure():
    record = original.STUDY/'controller_adapters/windows_atomic_json_v1.json'
    adapter = dict(version=1, purpose='Bounded atomic JSON replacement retries for temporary Windows sharing locks',
        source_hashes={str(Path(p).relative_to(original.ROOT).as_posix()):digest(p) for p in (
            Path(__file__), original.ROOT/'framework/training/study_supervisor.py')},
        numerical_or_recipe_changes=False)
    if record.exists() and read(record) != adapter:
        raise ValueError('Recovery adapter changed after pinning')
    atomic_write(record, adapter)
    def write(path, value):
        if Path(path) == original.STUDY/'state.json':
            value = dict(value, recovery_adapter=dict(path=original.relative(record), sha256=digest(record)))
        atomic_write(path, value)
    original.MODULE = MODULE
    original.write = write
    original.runtime.write = write


if __name__ == '__main__':
    configure()
    original.main()
