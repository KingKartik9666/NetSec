"""Explicitly download the pinned language model, its licence and (with --cuda) the CUDA llama.cpp runtime; never called at app startup.

The manifest artifacts/llm/model.json needs: name, file, sha256, bytes, url, prompt_template ({system} and {user}),
and optionally license_file, license_url, license_bytes, license_sha256.
The CUDA folder's release.json pins each release asset (name, url, bytes, sha256); the first extraction records
the SHA-256 of every extracted file under "files", and later runs verify against it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/llm/model.json"
CUDA = ROOT / "artifacts/rust_migration/phase1/llm/native-b11564-cuda"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified(path: Path, size: int, sha256: str) -> bool:
    return path.is_file() and path.stat().st_size == size and digest(path) == sha256


def download(url: str, target: Path, size: int, sha256: str, label: str) -> None:
    if target.exists():
        if not verified(target, size, sha256):
            raise SystemExit(f"Existing {label} differs from the pin; preserved without overwriting.")
        print(f"Verified existing {label}", flush=True)
        return
    temporary = target.with_name(target.name + ".part")
    # A failed download is retained for diagnosis; never overwrite it silently.
    downloaded = 0
    last_progress = -1
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("xb") as stream:
        while block := response.read(8 * 1024 * 1024):
            stream.write(block)
            downloaded += len(block)
            if downloaded > size:
                raise ValueError(f"Download of {label} exceeded its pinned size")
            progress = downloaded * 10 // size
            if progress != last_progress:
                print(f"Downloaded {downloaded:,} / {size:,} bytes of {label}", flush=True)
                last_progress = progress
    if not verified(temporary, size, sha256):
        raise ValueError(f"Downloaded {label} failed size/SHA-256 verification")
    temporary.rename(target)
    print(f"Verified {label}: {sha256}", flush=True)


def bare(name: str) -> bool:
    return bool(name) and Path(name).name == name and PurePosixPath(name).name == name and ":" not in name and name not in {".", ".."}


def fetch_model(manifest: Path = MANIFEST) -> None:
    if not manifest.is_file():
        raise SystemExit(f"No language model is selected; write {manifest} first.")
    model = json.loads(manifest.read_text(encoding="utf-8"))
    if not bare(model["file"]) or not model["file"].endswith(".gguf"):
        raise SystemExit("Manifest file must be a bare .gguf name")
    if "license_file" in model and not bare(model["license_file"]):
        raise SystemExit("Manifest license_file must be a bare file name")
    download(model["url"], manifest.parent / model["file"], model["bytes"], model["sha256"], model["file"])
    if model.get("license_url"):
        download(model["license_url"], manifest.parent / model["license_file"], model["license_bytes"], model["license_sha256"], model["license_file"])


def extract(archive: Path, destination: Path) -> dict[str, str]:
    """Extract a verified zip without leaving destination; never overwrite a differing file."""
    files = {}
    with zipfile.ZipFile(archive) as stream:
        for member in stream.infolist():
            name = member.filename.replace("\\", "/")
            parts = PurePosixPath(name).parts
            if name.startswith("/") or ":" in name or ".." in parts or not parts:
                raise ValueError(f"Archive member {member.filename!r} would extract outside {destination}")
            if member.is_dir():
                continue
            target = destination.joinpath(*parts)
            data = stream.read(member)
            if target.exists():
                if target.read_bytes() != data:
                    raise SystemExit(f"Existing {name} differs from {archive.name}; preserved without overwriting.")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            files[name] = hashlib.sha256(data).hexdigest()
    return files


def fetch_cuda(folder: Path = CUDA, cache: Path = CUDA.parent) -> None:
    pins = folder / "release.json"
    release = json.loads(pins.read_text(encoding="utf-8"))
    if release.get("files"):
        for name, sha256 in release["files"].items():
            path = folder / name
            if not path.is_file() or digest(path) != sha256:
                raise SystemExit(f"Extracted {name} is missing or differs from release.json; preserved for diagnosis.")
        print(f"Verified {len(release['files'])} extracted CUDA runtime files", flush=True)
        return
    files = {}
    for asset in release["assets"]:
        if not bare(asset["name"]) or not asset["name"].endswith(".zip"):
            raise SystemExit("Release asset names must be bare .zip names")
        archive = cache / asset["name"]
        download(asset["url"], archive, asset["bytes"], asset["sha256"], asset["name"])
        files.update(extract(archive, folder))
    release["files"] = dict(sorted(files.items()))
    pins.write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    print(f"Extracted and recorded {len(files)} CUDA runtime files in {pins.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cuda", action="store_true", help="fetch the pinned llama.cpp CUDA build and cudart instead of the model")
    if parser.parse_args().cuda:
        fetch_cuda()
    else:
        fetch_model()


if __name__ == "__main__":
    main()
