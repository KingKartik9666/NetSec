"""Check installed runtime bytes and actual Windows known-folder shortcuts."""
import ctypes
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    package = json.loads((ROOT / "framework/app/desktop/package.json").read_text(encoding="utf-8"))
    name = package["build"]["productName"]
    installed = Path(os.environ.get("IMDAI_INSTALL_DIR", str(Path(os.environ["LOCALAPPDATA"]) / "Programs" / name)))
    runtime = installed / "resources/backend"
    manifest = json.loads((runtime / "build-manifest.json").read_text(encoding="utf-8"))
    build_manifest = json.loads((ROOT / "artifacts/desktop/backend/imdai-backend/build-manifest.json").read_text(encoding="utf-8"))
    checks = {relative: sha(ROOT / relative) == sha(runtime / "_internal" / relative)
              for relative in manifest["ui_assets_sha256"]}
    checks["rag_corpus"] = sha(ROOT / "framework/rag/knowledge.json") == sha(runtime / "_internal/framework/rag/knowledge.json")
    for model_name, expected_sha in manifest["checkpoint_sha256"].items():
        checks["checkpoint:" + model_name] = sha(runtime / "_internal/framework/models" / model_name / "model.ubj") == expected_sha
        source = ROOT / "framework/models" / model_name
        frozen = runtime / "_internal/framework/models" / model_name
        source_bundle = json.loads((source / "bundle.json").read_text(encoding="utf-8"))
        installed_bundle = json.loads((frozen / "bundle.json").read_text(encoding="utf-8"))
        checks["model_metadata:" + model_name] = {**source_bundle, "checkpoint": "model.ubj"} == installed_bundle
        for filename in ["xai_reference.npz", "xai_reference.json"]:
            checks[f"{model_name}:{filename}"] = sha(source / filename) == sha(frozen / filename)
    from framework.detection.trained_xgboost import default_bundle
    selected = json.loads(default_bundle(ROOT).read_text(encoding="utf-8"))
    installed_selected = json.loads(default_bundle(runtime / "_internal").read_text(encoding="utf-8"))
    checks["selected_model_version"] = installed_selected["model"].get("version") == selected["model"].get("version")
    checks["selected_model_checksum"] = installed_selected["checkpoint_sha256"] == selected["checkpoint_sha256"]
    checks["desktop_executable"] = (installed / f"{name}.exe").is_file()
    checks["frozen_executable"] = sha(runtime / "imdai-backend.exe") == manifest.get("executable_sha256")
    checks["compiled_runtime_verified"] = (bool(manifest.get("compiled_module_sha256"))
                                            and manifest.get("compiled_module_sha256") == build_manifest.get("compiled_module_sha256"))
    from framework.llm.local_model import FILENAME, SHA256
    checks["bundled_llm"] = sha(installed / "resources/models/qwen" / FILENAME) == SHA256
    checks["embedded_python"] = (runtime / "_internal/python311.dll").is_file()
    checks["no_bundled_npcap"] = not any(p.name.lower() in {"wpcap.dll", "packet.dll", "npcap.sys"} for p in runtime.rglob("*"))
    folder = ctypes.create_unicode_buffer(32768)
    assert ctypes.windll.shell32.SHGetFolderPathW(None, 0x10, None, 0, folder) == 0
    desktop = Path(folder.value)
    checks["desktop_shortcut"] = (desktop / f"{name}.lnk").is_file()
    checks["start_menu_shortcut"] = (Path(os.environ["APPDATA"]) / f"Microsoft/Windows/Start Menu/Programs/{name}.lnk").is_file()
    proof = {"installed_path": str(installed), "checks": checks, "passed": all(checks.values()),
             "manifest": manifest, "desktop_folder": str(desktop), "selected_model": installed_selected["model"]}
    (ROOT / "artifacts/desktop/verification/installed-assets.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
    print(json.dumps(proof, indent=2))
    assert proof["passed"]


if __name__ == "__main__":
    main()
