"""Offline checks for tools/stage_native_app.py seal classes; no build, staging or hashing of real assets."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.stage_native_app import seal

CUDA = "artifacts/rust_migration/phase1/llm/native-b11564-cuda"
CPU = "artifacts/rust_migration/phase1/llm/native-b11564"


def test_seal_defers_llm_weights_and_cuda_runtime_to_background(tmp_path):
    expected = {
        "artifacts/llm/Qwen3-4B-Instruct-2507-Q4_K_M.gguf": "background",
        f"{CUDA}/ggml-cuda.dll": "background",
        f"{CUDA}/llama-server.exe": "background",
        "artifacts/llm/model.json": "startup",
        "artifacts/llm/Qwen-Apache-2.0.txt": "startup",
        "licenses/llm-Qwen-Apache-2.0.txt": "startup",
        "artifacts/llm/notes.gguf.txt": "startup",
        f"{CPU}/llama-server.exe": "startup",
        f"{CPU}/ggml-base.dll": "startup",
        f"{CPU}/llama.dll": "startup",
        f"{CUDA}/sub/nested.dll": "startup",
        "bin/netsec-runtime.exe": "startup",
        "framework/rag/knowledge.json": "startup",
        "artifacts/application/native_models_v1/catalog.json": "startup",
        "artifacts/application/native_models_v1/xgboost/model.ubj": "model_load",
    }
    for relative in expected:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_bytes(relative.encode())
    manifest = seal(tmp_path, [Path(p) for p in expected])
    assert {f["path"]: f["verification"] for f in manifest["files"]} == expected
    assert all(f["bytes"] == len(f["path"]) and len(f["sha256"]) == 64 for f in manifest["files"])
    assert json.loads((tmp_path / "runtime_assets.json").read_text(encoding="utf-8")) == manifest


def _write(folder: Path, names):
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(name.encode())


def test_closure_predicate_mirrors_server_closure(tmp_path):
    from tools.stage_native_app import OTHER_TOOLS, closure_names
    entry = ["llama-server.exe", "llama-server-impl.dll", "mtmd.dll", "ggml-cuda.dll", "ggml-rpc.dll", "ggml-cpu-x64.dll", "ggml.dll",
             "LLAMA.DLL", "libomp.dll", "cudart64_13.dll", "llama-server.exe.manifest", "LICENSE-LLVM-OpenMP"]
    excluded = list(OTHER_TOOLS) + ["llama-completion.exe", "llama-cli.exe", "ggml-rpc-server.exe", "release.json", "notes.txt"]
    _write(tmp_path / "cuda", entry + excluded)
    assert closure_names(tmp_path / "cuda") == sorted(entry)
    _write(tmp_path / "no-server", ["ggml.dll"])
    assert closure_names(tmp_path / "no-server") == []
    (tmp_path / "cuda" / "llama-server.exe.local").mkdir()
    try:
        closure_names(tmp_path / "cuda")
    except ValueError as error:
        assert "llama-server.exe.local" in str(error)
    else:
        raise AssertionError("a .local redirection entry must stop staging")


def test_other_tools_match_rust_closure():
    import re
    from tools.stage_native_app import OTHER_TOOLS, ROOT
    source = (ROOT / "rust/netsec/crates/runtime/src/language/server.rs").read_text(encoding="utf-8")
    listed = re.search(r'const OTHER_TOOLS:\[&str;\d+\]=\[([^\]]*)\]', source).group(1)
    assert sorted(OTHER_TOOLS) == sorted(re.findall(r'"([^"]+)"', listed))


def _sealed(manifest: Path, folder: str):
    entries = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    return {e["path"][len(folder) + 1:]: e["verification"] for e in entries if e["path"].startswith(folder + "/") and "/" not in e["path"][len(folder) + 1:]}


def test_llama_folders_stage_and_seal_exactly_the_closure():
    from tools.stage_native_app import DESTINATION, ROOT, closure_names, llama_files
    staged = {}
    for source, relative in llama_files():
        staged.setdefault(relative.parent.as_posix(), set()).add(relative.name)
        assert source == ROOT / relative, relative
    for folder, kind in ((CPU, "startup"), (CUDA, "background")):
        expected = set(closure_names(ROOT / folder))
        assert "llama-server.exe" in expected and "llama-server-impl.dll" in expected and "mtmd.dll" in expected and "ggml-rpc.dll" in expected
        assert not any(n.startswith("llama-completion") for n in expected)
        assert staged[folder] == expected, folder
        assert _sealed(ROOT / "runtime_assets.json", folder) == dict.fromkeys(expected, kind), f"dev seal of {folder}"
        package = DESTINATION / "runtime_assets.json"
        if package.is_file():
            assert set(_sealed(package, folder)) == set(closure_names(DESTINATION / folder)), f"package seal of {folder}"


# Plan step 9: the live77 bundle (owner decision: artifacts/application/live_models_v1/live77_xgboost, bundle.json + model.ubj)
# must seal as startup assets whose hashes equal the Rust loader's pins (live/models.rs); the golden file is not shipped.
LIVE77 = "artifacts/application/live_models_v1/live77_xgboost"
LIVE77_SHIPPED = {f"{LIVE77}/bundle.json", f"{LIVE77}/model.ubj"}


def test_seal_pins_the_live77_bundle_as_startup_assets(tmp_path):
    import shutil
    from tools.stage_native_app import ROOT
    from tools.verify_native_app import rust_pins
    pins = rust_pins()
    (tmp_path / LIVE77).mkdir(parents=True)
    for relative in LIVE77_SHIPPED:
        shutil.copyfile(ROOT / relative, tmp_path / relative)
    records = {f["path"]: f for f in seal(tmp_path, [Path(p) for p in LIVE77_SHIPPED])["files"]}
    assert set(records) == LIVE77_SHIPPED and all(f["verification"] == "startup" for f in records.values())
    assert records[f"{LIVE77}/bundle.json"]["sha256"] == pins["LIVE77_BUNDLE_SHA"]
    bundle = json.loads((ROOT / LIVE77 / "bundle.json").read_text(encoding="utf-8"))
    model = records[f"{LIVE77}/model.ubj"]
    assert model["sha256"] == bundle["checkpoint_sha256"] == pins["LIVE77_CHECKPOINT_SHA"] and model["bytes"] == bundle["bytes"]


RETIRED = ("xgboost_partial", "packet_live")   # stay on disk unused (owner decision); must leave the seal at step 10


def _selection_source():
    import inspect
    from tools.stage_native_app import selected_files
    return inspect.getsource(selected_files)


def _stages_live77(source):
    return "live_models_v1/live77_xgboost" in source and not any(f'"{n}"' in source for n in RETIRED)


def _retired_paths():
    from tools.stage_native_app import ROOT
    paths = set()
    for name in RETIRED:
        bundle = Path("framework/models") / name / "bundle.json"
        checkpoint = json.loads((ROOT / bundle).read_text(encoding="utf-8"))["checkpoint"]
        paths |= {bundle.as_posix(), (ROOT / bundle.parent / checkpoint).resolve().relative_to(ROOT).as_posix()}
    return paths


def test_stage_selects_the_live77_bundle_and_not_the_retired_ones():
    assert _stages_live77(_selection_source())


def test_stage_selection_detector_reads_the_bundle_folder():
    assert not _stages_live77('for name in ("xgboost_partial", "packet_live"):')
    assert not _stages_live77('for folder in ("live_models_v1/live77_xgboost", "packet_live"):')
    assert _stages_live77('bundle = Path("artifacts/application/live_models_v1/live77_xgboost") / "bundle.json"')
    assert _retired_paths() >= {"framework/models/xgboost_partial/bundle.json", "framework/models/packet_live/bundle.json"}


def _check_sealed_live77(manifests, pins, retired):
    """Each manifest holds exactly bundle.json + model.ubj (pinned shas, startup) under LIVE77 and no retired entry."""
    expected = {f"{LIVE77}/bundle.json": pins["LIVE77_BUNDLE_SHA"], f"{LIVE77}/model.ubj": pins["LIVE77_CHECKPOINT_SHA"]}
    for name, files in manifests.items():
        entries = [e for e in files if e["path"].startswith(LIVE77 + "/")]
        assert {e["path"]: e["sha256"] for e in entries} == expected, f"{name}: live77 entries"
        assert all(e["verification"] == "startup" for e in entries), f"{name}: live77 verification"
        assert not {e["path"] for e in files} & retired, f"{name}: retired live bundles still sealed"


def test_sealed_manifest_check_refuses_missing_extra_and_retired_entries():
    pins = dict(LIVE77_BUNDLE_SHA="b" * 64, LIVE77_CHECKPOINT_SHA="c" * 64)
    good = [{"path": f"{LIVE77}/bundle.json", "sha256": "b" * 64, "verification": "startup"},
            {"path": f"{LIVE77}/model.ubj", "sha256": "c" * 64, "verification": "startup"}, {"path": "x", "sha256": "0", "verification": "startup"}]
    retired = {"framework/models/packet_live/bundle.json"}
    _check_sealed_live77({"dev": good}, pins, retired)
    for bad in ([], good[:1], good + [{"path": f"{LIVE77}/golden_live77.json", "sha256": "d", "verification": "startup"}],
                [dict(good[0], sha256="e" * 64), good[1]], [good[0], dict(good[1], verification="model_load")],
                good + [{"path": "framework/models/packet_live/bundle.json", "sha256": "f", "verification": "startup"}]):
        with pytest.raises(AssertionError):
            _check_sealed_live77({"dev": bad}, pins, retired)


def test_sealed_manifests_hold_exactly_the_pinned_live77_files():
    """Keyed on the staging source, not on the manifests: once selected_files() names live77, the dev manifest (and the
    package manifest when it exists) must hold exactly the pinned live77 files and none of the retired bundles."""
    from tools.stage_native_app import DESTINATION, ROOT
    from tools.verify_native_app import rust_pins
    if not _stages_live77(_selection_source()):
        pytest.skip("stage_native_app.selected_files() does not stage live77 yet: plan step 10 (restage and reseal, app chat); "
                    "test_stage_selects_the_live77_bundle_and_not_the_retired_ones is the strict xfail that flips")
    manifests = {"dev runtime_assets.json": ROOT / "runtime_assets.json"}
    if (DESTINATION / "runtime_assets.json").is_file():
        manifests["package runtime_assets.json"] = DESTINATION / "runtime_assets.json"
    assert manifests["dev runtime_assets.json"].is_file(), "dev runtime_assets.json missing: run the step-10 reseal"
    _check_sealed_live77({k: json.loads(p.read_text(encoding="utf-8"))["files"] for k, p in manifests.items()}, rust_pins(), _retired_paths())
