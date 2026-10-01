"""Freeze the CPU desktop backend on its target OS, then verify it without Python."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import queue
import secrets
import shutil
import subprocess
import stat
import struct
import sys
import threading
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "artifacts/desktop"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime_source_hashes():
    return {path.relative_to(ROOT).as_posix(): digest(path)
            for path in sorted((ROOT / "framework").rglob("*.py"))
            if not {"training", "node_modules", "__pycache__"}.intersection(path.relative_to(ROOT).parts)}


def owned(path):
    """Refuse redirected build directories before tools replace generated files."""
    path = Path(path).absolute()
    if BUILD.resolve() not in path.resolve().parents:
        raise ValueError(f"Build destination leaves the owned build directory: {path}")
    for part in [path, *path.parents]:
        if part == ROOT:
            break
        if part.exists() and (part.is_symlink() or getattr(part.lstat(), "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
            raise ValueError(f"Redirected build destination refused: {part}")
    return path


@contextmanager
def build_lock():
    """OS releases the lock even if a build process crashes; no stale PID cleanup."""
    BUILD.mkdir(parents=True, exist_ok=True)
    with owned(BUILD / "build.lock").open("a+b") as handle:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another desktop build owns the output directory; wait for it to finish") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def stage_assets():
    staging = owned(BUILD / "staging")
    staging.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    for name in ["xgboost_app", "packet_live", "xgboost_partial", "xgboost_partial_enhanced"]:
        source = ROOT / "framework/models" / name
        if name.startswith("xgboost_partial") and not (source / "bundle.json").is_file():
            continue
        target = staging / "framework/models" / name
        target.mkdir(parents=True, exist_ok=True)
        bundle = json.loads((source / "bundle.json").read_text(encoding="utf-8"))
        checkpoint = (source / bundle["checkpoint"]).resolve()
        if digest(checkpoint) != bundle["checkpoint_sha256"]:
            raise ValueError(f"Research checkpoint checksum mismatch: {name}")
        shutil.copy2(checkpoint, target / "model.ubj")
        bundle["checkpoint"] = "model.ubj"
        (target / "bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        for filename in ["xai_reference.npz", "xai_reference.json"]:
            shutil.copy2(source / filename, target / filename)
        checkpoints[name] = bundle["checkpoint_sha256"]
    licenses = staging / "THIRD_PARTY_LICENSES"
    licenses.mkdir(exist_ok=True)
    distributions = []
    for distribution in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
        name, version = distribution.metadata["Name"], distribution.version
        distributions.append({"name": name, "version": version, "homepage": distribution.metadata.get("Home-page"),
                              "project_urls": distribution.metadata.get_all("Project-URL", [])})
        for file in distribution.files or []:
            if any(marker in file.name.lower() for marker in ["license", "copying", "notice"]):
                location = Path(distribution.locate_file(file))
                if location.is_file() and location.stat().st_size < 5_000_000:
                    target = licenses / name / str(file).replace("../", "").replace("..\\", "")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(location, target)
    (licenses / "packages.json").write_text(json.dumps(distributions, indent=2), encoding="utf-8")
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.exists():
        shutil.copy2(python_license, licenses / "PYTHON-LICENSE.txt")
    return staging, checkpoints


def smoke_test(executable):
    """Use an unrelated working directory and remove Python environment hints."""
    token = secrets.token_hex(32)
    environment = dict(os.environ, IMDAI_SESSION_TOKEN=token, IMDAI_LLM_DIR=str(ROOT / "artifacts/llm/qwen"))
    for key in ["PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"]:
        environment.pop(key, None)
    environment["PATH"] = str(Path(os.environ["WINDIR"]) / "System32") if os.name == "nt" else "/usr/bin:/bin"
    records, lines = {}, queue.Queue()
    log = owned(BUILD / "backend-smoke-stderr.log")
    with log.open("w", encoding="utf-8") as errors:
        process = subprocess.Popen([str(executable)], cwd=executable.parent, env=environment, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=errors, text=True, encoding="utf-8")
        threading.Thread(target=lambda: [lines.put(line) for line in process.stdout], daemon=True).start()
        try:
            while True:
                try:
                    message = json.loads(lines.get(timeout=90))
                except queue.Empty:
                    raise RuntimeError(f"Frozen backend startup failed; see {log}")
                if message.get("type") == "ready":
                    break
            origin = f"http://127.0.0.1:{message['port']}"
            def request(route, data=None, authenticated=True):
                headers = {"X-IMDAI-Session": token} if authenticated else {}
                if data is not None:
                    headers["Content-Type"] = "application/json"
                req = urllib.request.Request(origin + route, data=json.dumps(data).encode() if data is not None else None, headers=headers)
                with urllib.request.urlopen(req, timeout=90) as response:
                    return json.load(response)
            try:
                request("/api/status", authenticated=False)
                raise AssertionError("The packaged backend accepted an unauthenticated request")
            except urllib.error.HTTPError as error:
                assert error.code == 403
            records["authentication"] = True
            records["status"] = request("/api/status")
            samples = request("/api/samples")["samples"]
            sample_id = samples[0]["id"]
            records["prediction"] = request("/api/analyze", {"sample_id": sample_id})["prediction"]
            records["lime"] = request("/api/xai", {"kind": "sample", "sample_id": sample_id})["lime"]
            records["live_capabilities"] = request("/api/live/capabilities")
            records["assistant"] = request("/api/assistant/status")
            # Imports and actually executes the native llama.cpp DLL, rather than
            # treating the existence of a module or a model file as inference.
            assert records["assistant"].get("available"), "The standalone build requires the bundled language model and runtime"
            records["language_generation"] = request("/api/assistant/explain", {"kind": "sample", "sample_id": sample_id})
            assert records["language_generation"]["llm_executed"] is True
            assert records["language_generation"]["rag"]["used_in_generation"] is True
            assert records["language_generation"]["correction"]["generation_attempts"][-1]["passed"]
            process.stdin.close()
            assert process.wait(timeout=15) == 0
            records["shutdown_on_parent_eof"] = True
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    (BUILD / "backend-smoke.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def source_archive():
    """Portable build sources + frozen runtime assets; no raw research datasets."""
    metadata = json.loads((ROOT / "framework/app/desktop/package.json").read_text(encoding="utf-8"))
    archive = owned(BUILD / f"releases/{metadata['build']['productName']}-{metadata['version']}-linux-build-source.zip")
    archive.parent.mkdir(parents=True, exist_ok=True)
    included = {}
    for source in (ROOT / "framework").rglob("*.py"):
        relative = source.relative_to(ROOT)
        if not {"node_modules", "__pycache__", "training"}.intersection(relative.parts):
            included[relative.as_posix()] = source
    for directory in ["framework/app/templates", "framework/app/static"]:
        for source in (ROOT / directory).rglob("*"):
            if source.is_file() and "__pycache__" not in source.parts:
                included[source.relative_to(ROOT).as_posix()] = source
    included["framework/rag/knowledge.json"] = ROOT / "framework/rag/knowledge.json"
    desktop = ROOT / "framework/app/desktop"
    for source in desktop.rglob("*"):
        if source.is_file() and not {"node_modules", "__pycache__"}.intersection(source.relative_to(desktop).parts):
            included[source.relative_to(ROOT).as_posix()] = source
    for filename in ["run_desktop.py", "requirements-desktop.txt", "experiment/__init__.py",
                     "experiment/build_desktop.py", "experiment/frozen_code_verification.py", "experiment/setup_local_llm.py", "docs/DESKTOP.md",
                     "docs/PROJECT_CONTEXT.md", "docs/ARCHITECTURE.md", "docs/HEM_INTEGRATION.md", "docs/RAG.md"]:
        included[filename] = ROOT / filename
    for source in (BUILD / "verification/rag_corpus_sources").glob("*"):
        if source.is_file():
            included[source.relative_to(ROOT).as_posix()] = source
    for name in ["xgboost_app", "packet_live", "xgboost_partial", "xgboost_partial_enhanced"]:
        if name.startswith("xgboost_partial") and not (BUILD / "staging/framework/models" / name / "bundle.json").is_file():
            continue
        for filename in ["bundle.json", "model.ubj", "xai_reference.json", "xai_reference.npz"]:
            included[f"framework/models/{name}/{filename}"] = BUILD / "staging/framework/models" / name / filename
    for filename in ["qwen2.5-0.5b-instruct-q4_k_m.gguf", "model.json", "LICENSE"]:
        included[f"artifacts/llm/qwen/{filename}"] = ROOT / "artifacts/llm/qwen" / filename
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
        stream.writestr("README.txt", "IMDAI Linux x64 desktop build sources\n\nRead docs/DESKTOP.md. Build the backend on Linux before creating an AppImage.\nThis archive is source, not a verified Linux binary. The Windows setup EXE is separate.\nPortable checkpoints, training-only XAI references and the small offline LLM are included.\nRaw research datasets and CUDA training environments are not required.\n")
        for relative, source in sorted(included.items()):
            stream.write(source, relative, compress_type=zipfile.ZIP_STORED if source.suffix == ".gguf" else zipfile.ZIP_DEFLATED)
    print(f"Portable Linux build source archive: {archive} ({archive.stat().st_size} bytes)", flush=True)
    return archive


def refresh_assets(executable):
    """Refresh external UI data after visual edits without refreezing Python."""
    files = {}
    for directory in ["templates", "static"]:
        source_directory = ROOT / "framework/app" / directory
        target_directory = owned(executable.parent / "_internal/framework/app" / directory)
        for source in source_directory.rglob("*"):
            if source.is_file() and "__pycache__" not in source.parts:
                relative = source.relative_to(source_directory)
                target = target_directory / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                assert digest(source) == digest(target)
                files[source.relative_to(ROOT).as_posix()] = digest(target)
    manifest_path = executable.parent / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["ui_assets_sha256"] = files
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    if platform.machine().lower() not in {"amd64", "x86_64"} or struct.calcsize("P") != 8:
        raise SystemExit("Build on native x64 with a 64-bit Python interpreter; ARM and 32-bit builds are not supported.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="Create/update the isolated CPU build environment")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--source-archive", action="store_true", help="Package portable build sources and models for a native Linux build")
    parser.add_argument("--refresh-assets", action="store_true", help="Refresh UI files in an existing backend, then repeat its smoke test")
    args = parser.parse_args()
    if args.source_archive:
        with build_lock():
            source_archive()
        return 0
    environment = owned(BUILD / "build-env")
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if args.install:
        if not python.exists():
            subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "--only-binary=llama-cpp-python", "--extra-index-url",
                        "https://abetlen.github.io/llama-cpp-python/whl/cpu", "-r", str(ROOT / "requirements-desktop.txt")], check=True)
    if not python.exists():
        raise SystemExit("Run again with --install to create the isolated CPU build environment.")
    if Path(sys.prefix).resolve() != environment.resolve():
        return subprocess.call([str(python), str(Path(__file__).resolve()), *[arg for arg in sys.argv[1:] if arg != "--install"]])
    with build_lock():
        return build_native(args)


def build_native(args):
    output = owned(BUILD / "backend")
    executable = output / "imdai-backend" / ("imdai-backend.exe" if os.name == "nt" else "imdai-backend")
    if not args.smoke_only and not args.refresh_assets:
        import PyInstaller.__main__
        staging, checkpoints = stage_assets()
        source_hashes = runtime_source_hashes()
        spec = owned(BUILD / "spec")
        spec.mkdir(parents=True, exist_ok=True)
        command = [str(ROOT / "framework/app/desktop_backend.py"), "--name", "imdai-backend", "--onedir", "--console", "--clean",
                   "--noconfirm", "--noupx", "--distpath", str(output), "--workpath", str(owned(BUILD / "pyinstaller-work")),
                   "--specpath", str(spec), "--paths", str(ROOT),
                   "--add-data", f"{staging / 'framework/models'}{os.pathsep}framework/models",
                   "--add-data", f"{staging / 'THIRD_PARTY_LICENSES'}{os.pathsep}THIRD_PARTY_LICENSES",
                   "--add-data", f"{ROOT / 'framework/app/templates'}{os.pathsep}framework/app/templates",
                   "--add-data", f"{ROOT / 'framework/app/static'}{os.pathsep}framework/app/static",
                   "--add-data", f"{ROOT / 'framework/rag/knowledge.json'}{os.pathsep}framework/rag",
                   "--collect-all", "llama_cpp", "--collect-all", "xgboost", "--collect-submodules", "scapy",
                   "--hidden-import", "lime.lime_tabular", "--copy-metadata", "llama-cpp-python",
                   "--exclude-module", "torch", "--exclude-module", "tensorflow", "--exclude-module", "IPython",
                   "--exclude-module", "notebook", "--exclude-module", "tkinter", "--exclude-module", "matplotlib.tests"]
        PyInstaller.__main__.run(command)
        if source_hashes != runtime_source_hashes():
            raise RuntimeError("Runtime source changed during freezing; rebuild from a stable source revision")
        from experiment.frozen_code_verification import verify_frozen_modules
        compiled_hashes = verify_frozen_modules(executable, ROOT)
        manifest = {"platform": sys.platform, "architecture": "x64", "python": sys.version.split()[0],
                    "created_at": datetime.now(timezone.utc).isoformat(), "cpu_only": True,
                    "checkpoint_sha256": checkpoints, "runtime_source_sha256": source_hashes,
                    "compiled_module_sha256": compiled_hashes, "smoke_test_passed": False}
        (executable.parent / "build-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_path = executable.parent / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["smoke_test_passed"] = False
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not args.smoke_only:
        refresh_assets(executable)
    smoke_test(executable)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["smoke_test_passed"] = True
    manifest["executable_sha256"] = digest(executable)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Verified standalone backend: {executable}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
