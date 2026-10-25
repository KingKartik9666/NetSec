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


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
    gguf = Path("artifacts/llm/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    yield ROOT / gguf, gguf
    for source, name in [(ROOT / "artifacts/llm/qwen/LICENSE", "Qwen-Apache-2.0.txt"),
                         (ROOT / "artifacts/rust_migration/phase1/llm/LICENSE-llama-b11564", "llama-MIT.txt"),
                         (ROOT / "rust/spikes/xgboost/LICENSE", "XGBoost-Apache-2.0.txt"),
                         (ROOT / "rust/spikes/xgboost/NOTICE", "XGBoost-NOTICE.txt")]:
        yield source, Path("licenses") / name
    # Only completion + its actual CPU dependency closure, no server/GPU/tools.
    native_names = ["llama-completion.exe", "llama-completion-impl.dll", "llama-common.dll",
                    "llama.dll", "ggml.dll", "ggml-base.dll", "libomp.dll", "LICENSE-LLVM-OpenMP"]
    native_names += sorted(f.name for f in (ROOT / NATIVE).glob("ggml-cpu-*.dll"))
    for name in native_names:
        yield ROOT / NATIVE / name, NATIVE / name
    override = os.environ.get("VCREDIST_ROOT")
    candidates = sorted((Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) /
                         "Microsoft Visual Studio").glob("*/*/VC/Redist/MSVC/*/x64/Microsoft.VC143.CRT"))
    if not override and not candidates:
        raise ValueError("Set VCREDIST_ROOT to the licensed x64 Microsoft.VC143.CRT redistributable directory")
    redist = Path(override) if override else candidates[-1]
    for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        yield redist / name, NATIVE / name
        # netsec-runtime.exe has its own loader search directory. A DLL nested
        # beside llama.cpp does not satisfy the Rust executable's direct imports.
        yield redist / name, Path("bin") / name
    yield ROOT / "framework/app/static/fonts/LICENSE.txt", Path("licenses/Inter-OFL.txt")
    for package in ("react", "react-dom"):
        yield ROOT / "rust/netsec/web/node_modules" / package / "LICENSE", Path("licenses") / f"{package}-MIT.txt"
    for name in ("xgboost_partial", "packet_live"):
        bundle = Path("framework/models") / name / "bundle.json"
        yield ROOT / bundle, bundle
        metadata = json.loads((ROOT / bundle).read_text(encoding="utf-8"))
        checkpoint = (ROOT / bundle.parent / metadata["checkpoint"]).resolve()
        if ROOT not in checkpoint.parents:
            raise ValueError("Live checkpoint escapes project root")
        if digest(checkpoint) != metadata["checkpoint_sha256"]:
            raise ValueError("Live checkpoint differs from its sealed bundle")
        yield checkpoint, checkpoint.relative_to(ROOT)


def seal(root: Path, files):
    records = []
    for relative in sorted(set(files), key=str):
        path = root / relative
        records.append({"path": relative.as_posix(), "bytes": path.stat().st_size,
                        "sha256": digest(path),
                        "verification": "model_load" if "native_models_v1" in relative.parts and relative.name not in
                        ("catalog.json", "samples.json", "reference.json") else "startup"})
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
