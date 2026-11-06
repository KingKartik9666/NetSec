"""Stage a self-contained native NetSec runtime; never copy Python or research data.

Run after the Rust release build. The manifest records every packaged byte, while
startup verifies shared resources and model loading verifies each selected model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "artifacts/application/package_root"
NATIVE = Path("artifacts/rust_migration/phase1/llm/native-b11564")
CUDA = Path("artifacts/rust_migration/phase1/llm/native-b11564-cuda")


# The other llama.cpp tools' entry DLLs; mirrors OTHER_TOOLS in runtime/src/language/server.rs (a test keeps both equal).
OTHER_TOOLS = ("llama-batched-bench-impl.dll", "llama-bench-impl.dll", "llama-cli-impl.dll", "llama-completion-impl.dll",
               "llama-fit-params-impl.dll", "llama-perplexity-impl.dll", "llama-quantize-impl.dll")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def closure_names(folder: Path) -> list[str]:
    """Exactly the files server.rs closure() pins in a llama folder, plus its license files; empty without llama-server.exe.

    ggml loads every ggml-*.dll beside the server at run time, so the set is the folder listing filtered like closure(),
    never an import walk or a hand list.
    """
    names = []
    for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        lower = path.name.lower()
        if lower.endswith(".local"):
            raise ValueError(f"Language runtime folder holds a DLL redirection entry: {path}")
        if lower == "llama-server.exe" or lower.endswith(".manifest") or lower.startswith("license") or (
                lower.endswith(".dll") and lower not in OTHER_TOOLS):
            if not path.is_file():
                raise ValueError(f"Language runtime entry is not a file: {path}")
            names.append(path.name)
    return sorted(names) if "llama-server.exe" in {n.lower() for n in names} else []


def llama_files():
    """Both llama-server closures: CUDA first (GPU), then the CPU fallback (ruling 39); no other tools or release.json."""
    for folder in (CUDA, NATIVE):
        for name in closure_names(ROOT / folder):
            yield ROOT / folder / name, folder / name


def selected_files():
    yield ROOT / "rust/netsec/target/release/netsec-runtime.exe", Path("bin/netsec-runtime.exe")
    frontend = ROOT / "rust/netsec/web/dist"
    if not (frontend / "index.html").is_file():
        raise ValueError("Build the React renderer first: npm --prefix rust/netsec/web run build")
    for file in frontend.rglob("*"):
        if file.is_file():
            yield file, file.relative_to(ROOT)
    exported = ROOT / "artifacts/application/native_models_v1"
    for file in exported.rglob("*"):
        if file.is_file() and not file.name.startswith("golden"):
            yield file, file.relative_to(ROOT)
    yield ROOT / "framework/rag/knowledge.json", Path("framework/rag/knowledge.json")
    # The language model is optional: stage it only when artifacts/llm/model.json installs one.
    manifest = Path("artifacts/llm/model.json")
    if (ROOT / manifest).is_file():
        model = json.loads((ROOT / manifest).read_text(encoding="utf-8"))
        yield ROOT / manifest, manifest
        yield ROOT / manifest.parent / model["file"], manifest.parent / model["file"]
        if model.get("license_file"):
            yield ROOT / manifest.parent / model["license_file"], Path("licenses") / f"llm-{model['license_file']}"
    for source, name in [(ROOT / "artifacts/rust_migration/phase1/llm/LICENSE-llama-b11564", "llama-MIT.txt"),
                         (ROOT / "rust/spikes/xgboost/LICENSE", "XGBoost-Apache-2.0.txt"),
                         (ROOT / "rust/spikes/xgboost/NOTICE", "XGBoost-NOTICE.txt")]:
        yield source, Path("licenses") / name
    yield from llama_files()
    override = os.environ.get("VCREDIST_ROOT")
    candidates = sorted((Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) /
                         "Microsoft Visual Studio").glob("*/*/VC/Redist/MSVC/*/x64/Microsoft.VC143.CRT"))
    if not override and not candidates:
        raise ValueError("Set VCREDIST_ROOT to the licensed x64 Microsoft.VC143.CRT redistributable directory")
    redist = Path(override) if override else candidates[-1]
    for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        # Both llama-server builds need the CRT on a clean machine; closure() pins these beside each server too.
        yield redist / name, NATIVE / name
        yield redist / name, CUDA / name
        # netsec-runtime.exe has its own loader search directory. A DLL nested
        # beside llama.cpp does not satisfy the Rust executable's direct imports.
        yield redist / name, Path("bin") / name
    yield ROOT / "framework/app/static/fonts/LICENSE.txt", Path("licenses/Inter-OFL.txt")
    for package in ("react", "react-dom"):
        yield ROOT / "rust/netsec/web/node_modules" / package / "LICENSE", Path("licenses") / f"{package}-MIT.txt"
    # The live 77-class model (owner, 3 Nov 2026). The retired xgboost_partial and packet_live bundles stay on disk,
    # unused and unsealed; golden_live77.json is a test fixture and is not shipped.
    bundle = Path("artifacts/application/live_models_v1/live77_xgboost") / "bundle.json"
    yield ROOT / bundle, bundle
    metadata = json.loads((ROOT / bundle).read_text(encoding="utf-8"))
    checkpoint = (ROOT / bundle.parent / metadata["checkpoint"]).resolve()
    if ROOT not in checkpoint.parents:
        raise ValueError("Live checkpoint escapes project root")
    if digest(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Live checkpoint differs from its sealed bundle")
    yield checkpoint, checkpoint.relative_to(ROOT)


def verification(relative: Path) -> str:
    if "native_models_v1" in relative.parts and relative.name not in ("catalog.json", "samples.json", "reference.json"):
        return "model_load"
    # The 2.5 GB GGUF and the CUDA runtime are size-checked at startup and hashed by the language server's
    # background pass before first use (runtime/src/assets.rs background_path accepts only these locations).
    if (relative.parent == Path("artifacts/llm") and relative.suffix == ".gguf") or relative.parent == CUDA:
        return "background"
    return "startup"


def seal(root: Path, files):
    records = []
    for relative in sorted(set(files), key=str):
        path = root / relative
        records.append({"path": relative.as_posix(), "bytes": path.stat().st_size,
                        "sha256": digest(path), "verification": verification(relative)})
    manifest = {"version": 1, "runtime": "rust", "cpu_only": True, "offline": True,
                "files": records, "large_assets_in_git": False}
    (root / "runtime_assets.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal-only", action="store_true", help="Seal current development resources")
    args = parser.parse_args()
    selected = list(selected_files())
    if args.seal_only:
        # Development executable occupies the build output, not packaged bin/.
        paths = [source.relative_to(ROOT) for source, _ in selected if source.is_relative_to(ROOT)]
        manifest = seal(ROOT, paths)
        target = ROOT
    else:
        target = DESTINATION.resolve()
        if target != DESTINATION or ROOT not in target.parents:
            raise ValueError("Staging target is not the designated application package root")
        target.mkdir(parents=True, exist_ok=True)
        previous = target / "runtime_assets.json"
        expected = {relative.as_posix() for _, relative in selected}
        if previous.exists():
            for entry in json.loads(previous.read_text(encoding="utf-8"))["files"]:
                obsolete = (target / entry["path"]).resolve()
                if target not in obsolete.parents:
                    raise ValueError("Previous package manifest escapes staging")
                if entry["path"] not in expected and obsolete.is_file():
                    obsolete.unlink()
        for source, relative in selected:
            output = target / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.exists() or digest(source) != digest(output):
                shutil.copy2(source, output)
        manifest = seal(target, [relative for _, relative in selected])
    print(json.dumps({"root": str(target), "files": len(manifest["files"]),
                      "bytes": sum(f["bytes"] for f in manifest["files"]), "python_runtime": False}))


if __name__ == "__main__":
    main()
