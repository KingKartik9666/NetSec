"""Packaging must reject a frozen app whose model selection is out of date."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


NODE = shutil.which("node") or str(Path(os.environ["USERPROFILE"]) / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe")
HOOK = Path(__file__).resolve().parents[1] / "framework/app/desktop/verify-build.cjs"


@pytest.fixture
def release(tmp_path):
    if not Path(NODE).is_file():
        pytest.skip("Node.js is required for the Electron packaging hook")
    source = tmp_path / "framework/models/xgboost_partial"
    frozen = tmp_path / "backend/_internal/framework/models/xgboost_partial"
    source.mkdir(parents=True)
    frozen.mkdir(parents=True)
    checkpoint = b"frozen checkpoint fixture, not an executable research model"
    sha = hashlib.sha256(checkpoint).hexdigest()
    bundle = {"checkpoint": "model.ubj", "checkpoint_sha256": sha, "promoted": True,
              "promotion": {"improved": True}, "model": {"version": "tested-version"}}
    for directory in [source, frozen]:
        (directory / "bundle.json").write_text(json.dumps(bundle))
        (directory / "model.ubj").write_bytes(checkpoint)
        (directory / "xai_reference.json").write_text('{"source":"training only"}')
        (directory / "xai_reference.npz").write_bytes(b"reference fixture")
    manifest = {"checkpoint_sha256": {"xgboost_partial": sha}}
    return tmp_path, source, frozen, manifest


def verify(release):
    root, _, _, manifest = release
    script = "const hook=require(process.argv[1]);try{hook.verifyModelAssets(process.argv[2],process.argv[3],JSON.parse(process.argv[4]));}catch(e){console.error(e.message);process.exit(1)}"
    return subprocess.run([NODE, "-e", script, str(HOOK), str(root), str(root / "backend"), json.dumps(manifest)], capture_output=True, text=True)


def test_current_model_release_is_accepted(release):
    result = verify(release)
    assert result.returncode == 0, result.stderr


def test_newly_selected_upgrade_cannot_be_omitted(release):
    release[3]["checkpoint_sha256"].clear()
    result = verify(release)
    assert result.returncode != 0
    assert "missing" in result.stderr.lower()


@pytest.mark.parametrize("changed", ["bundle.json", "model.ubj", "xai_reference.json", "xai_reference.npz"])
def test_changed_model_or_explanation_references_require_rebuild(release, changed):
    (release[2] / changed).write_bytes(b"stale artifact")
    result = verify(release)
    assert result.returncode != 0
    assert "stale model" in result.stderr.lower()
