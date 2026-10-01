"""Concurrent freezes must not replace a backend being verified."""
from pathlib import Path

import pytest

from experiment import build_desktop


def test_build_lock_rejects_overlap_and_releases_after_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(build_desktop, 'ROOT', tmp_path)
    monkeypatch.setattr(build_desktop, 'BUILD', tmp_path / 'artifacts/desktop')
    build_desktop.BUILD.mkdir(parents=True)
    with pytest.raises(ValueError, match='intentional verification failure'):
        with build_desktop.build_lock():
            with pytest.raises(RuntimeError, match='Another desktop build'):
                with build_desktop.build_lock():
                    pytest.fail('Two builds acquired the same output directory')
            raise ValueError('intentional verification failure')
    with build_desktop.build_lock():
        assert (build_desktop.BUILD / 'build.lock').is_file()
