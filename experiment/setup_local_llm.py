"""Download and verify the official Apache-2.0 Qwen instruction model for local use."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "artifacts/llm/qwen"
FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
REVISION = "9217f5db79a29953eb74d5343926648285ec7e67"
SHA256 = "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
URL = f"https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/{REVISION}/{FILENAME}"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    target = DIRECTORY / FILENAME
    if not target.exists():
        temporary = target.with_suffix(".part")
        print("Downloading official Qwen 0.5B Q4_K_M weights (491 MB).", flush=True)
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("wb") as output:
            size = 0
            while block := response.read(8 * 1024 * 1024):
                output.write(block)
                size += len(block)
                if size % (128 * 1024 * 1024) == 0:
                    print(f"Downloaded {size // 1048576} MiB", flush=True)
        if digest(temporary) != SHA256:
            raise ValueError("Model checksum mismatch; partial download was not promoted")
        temporary.rename(target)
    if digest(target) != SHA256:
        raise ValueError("Local model differs from the pinned official model")
    provenance = {"repository": "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "revision": REVISION,
                  "filename": FILENAME, "sha256": SHA256, "bytes": target.stat().st_size,
                  "license": "Apache-2.0", "url": URL, "runtime": "llama.cpp via llama-cpp-python",
                  "default_device": "CPU", "purpose": "Language generation over detector evidence; not intrusion detection"}
    (DIRECTORY / "model.json").write_text(json.dumps(provenance, indent=2))
    print("Official model checksum verified.", flush=True)


if __name__ == "__main__":
    main()
