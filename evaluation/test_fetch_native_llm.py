"""Offline checks for tools/fetch_native_llm.py: pinned downloads, CUDA release extraction and the model licence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from tools import fetch_native_llm as fetch


def pin(path: Path) -> dict:
    return {"url": path.as_uri(), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def archive(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as stream:
        for name, data in members.items():
            stream.writestr(name, data)
    return path


def test_download_verifies_size_and_hash(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"pinned bytes")
    good = pin(source)
    fetch.download(good["url"], tmp_path / "good.bin", good["bytes"], good["sha256"], "good.bin")
    assert (tmp_path / "good.bin").read_bytes() == b"pinned bytes"
    with pytest.raises(ValueError, match="SHA-256"):
        fetch.download(good["url"], tmp_path / "bad.bin", good["bytes"], "0" * 64, "bad.bin")
    assert not (tmp_path / "bad.bin").exists() and (tmp_path / "bad.bin.part").exists()


def test_existing_file_that_differs_is_preserved(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"pinned bytes")
    (tmp_path / "target.bin").write_bytes(b"something else")
    good = pin(source)
    with pytest.raises(SystemExit, match="preserved"):
        fetch.download(good["url"], tmp_path / "target.bin", good["bytes"], good["sha256"], "target.bin")
    assert (tmp_path / "target.bin").read_bytes() == b"something else"


def test_extract_rejects_members_outside_the_folder(tmp_path):
    for name in ["../evil.dll", "/abs.dll", "C:/x.dll", "sub/../../evil.dll"]:
        bad = archive(tmp_path / "bad.zip", {name: b"x"})
        with pytest.raises(ValueError, match="outside"):
            fetch.extract(bad, tmp_path / "out")
    assert not (tmp_path / "evil.dll").exists()


def test_cuda_release_is_downloaded_extracted_and_file_hashes_recorded(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    build = archive(tmp_path / "build.zip", {"llama-server.exe": b"server", "ggml-cuda.dll": b"cuda"})
    runtime = archive(tmp_path / "cudart.zip", {"cudart64_13.dll": b"cudart"})
    folder = tmp_path / "native-cuda"
    folder.mkdir()
    assets = [{"name": "build.zip", **pin(build)}, {"name": "cudart.zip", **pin(runtime)}]
    (folder / "release.json").write_text(json.dumps({"release": "b0", "assets": assets}), encoding="utf-8")
    fetch.fetch_cuda(folder, cache)
    release = json.loads((folder / "release.json").read_text(encoding="utf-8"))
    assert release["files"] == {name: hashlib.sha256(data).hexdigest() for name, data in
                                [("cudart64_13.dll", b"cudart"), ("ggml-cuda.dll", b"cuda"), ("llama-server.exe", b"server")]}
    assert (folder / "llama-server.exe").read_bytes() == b"server" and (cache / "build.zip").is_file()
    fetch.fetch_cuda(folder, cache)  # re-run verifies against the recorded hashes
    (folder / "ggml-cuda.dll").write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="ggml-cuda.dll"):
        fetch.fetch_cuda(folder, cache)


def test_cuda_asset_with_wrong_hash_is_not_extracted(tmp_path):
    build = archive(tmp_path / "build.zip", {"llama-server.exe": b"server"})
    folder = tmp_path / "native-cuda"
    folder.mkdir()
    (folder / "release.json").write_text(json.dumps({"assets": [{"name": "build.zip", **pin(build), "sha256": "0" * 64}]}), encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(ValueError, match="SHA-256"):
        fetch.fetch_cuda(folder, cache)
    assert not (folder / "llama-server.exe").exists() and not (cache / "build.zip").exists()


def test_model_and_licence_are_fetched_from_the_manifest(tmp_path):
    weights = tmp_path / "src.gguf"
    weights.write_bytes(b"GGUF weights")
    licence = tmp_path / "LICENSE"
    licence.write_bytes(b"Apache License 2.0")
    folder = tmp_path / "llm"
    folder.mkdir()
    model, terms = pin(weights), pin(licence)
    manifest = {"name": "m", "file": "m.gguf", "prompt_template": "{system}{user}", **model, "license_file": "Qwen-Apache-2.0.txt",
                "license_url": terms["url"], "license_bytes": terms["bytes"], "license_sha256": terms["sha256"]}
    (folder / "model.json").write_text(json.dumps(manifest), encoding="utf-8")
    fetch.fetch_model(folder / "model.json")
    assert (folder / "m.gguf").read_bytes() == b"GGUF weights" and (folder / "Qwen-Apache-2.0.txt").read_bytes() == b"Apache License 2.0"
    fetch.fetch_model(folder / "model.json")  # already present and verified


def test_pinned_manifest_and_cuda_release_match_the_plan():
    if not fetch.MANIFEST.is_file():
        pytest.skip("No language model is selected in this checkout")
    model = json.loads(fetch.MANIFEST.read_text(encoding="utf-8"))
    assert (model["file"], model["bytes"], model["sha256"]) == ("Qwen3-4B-Instruct-2507-Q4_K_M.gguf", 2497281120, "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597")
    assert "/a06e946bb6b655725eafa393f4a9745d460374c9/" in model["url"] and model["url"].startswith("https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/")
    assert (model["license"], model["license_file"], model["device"], model["max_tokens"]) == ("Apache-2.0", "Qwen-Apache-2.0.txt", "gpu", 300)
    assert "{system}" in model["prompt_template"] and "{user}" in model["prompt_template"]
    release = json.loads((fetch.CUDA / "release.json").read_text(encoding="utf-8"))
    assert [a["name"] for a in release["assets"]] == ["llama-b11564-bin-win-cuda-13.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip"]
    assert all(a["url"].startswith("https://github.com/ggml-org/llama.cpp/releases/download/b11564/") and len(a["sha256"]) == 64 for a in release["assets"])


def test_manifest_file_names_cannot_escape_the_llm_folder(tmp_path):
    for key, value in [("file", "../x.gguf"), ("file", "x.bin"), ("license_file", "../LICENSE")]:
        manifest = {"name": "m", "file": "m.gguf", "sha256": "0", "bytes": 1, "url": "file:///x", "prompt_template": "", key: value}
        (tmp_path / "model.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(SystemExit, match="bare"):
            fetch.fetch_model(tmp_path / "model.json")
