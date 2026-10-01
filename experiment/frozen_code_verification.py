"""Verify compiled PyInstaller framework code against the exact current source.

Source-file hashes alone cannot detect stale code in a reused PYZ archive.
This check reads code objects without executing them. Only ``co_filename``
may differ, because PyInstaller relocates source filenames when freezing.
"""
from __future__ import annotations

import hashlib
import json
import marshal
import struct
from pathlib import Path
from types import CodeType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class FrozenCodeMismatch(ValueError):
    """A compiled release module differs from the source it claims to contain."""

    def __init__(self, mismatches: dict[str, dict[str, str]]):
        self.mismatches = mismatches
        descriptions = "; ".join(f"{name}: {detail['reason']}" for name, detail in mismatches.items())
        super().__init__("Frozen framework code does not match source: " + descriptions)


def _constant(value: Any) -> Any:
    """Canonical, typed encoding of Python compiler constants."""
    if isinstance(value, CodeType):
        return _normalized_code(value)
    if value is None:
        return ["none"]
    if value is Ellipsis:
        return ["ellipsis"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["float", struct.pack(">d", value).hex()]
    if isinstance(value, complex):
        return ["complex", struct.pack(">d", value.real).hex(), struct.pack(">d", value.imag).hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, tuple):
        return ["tuple", [_constant(item) for item in value]]
    if isinstance(value, frozenset):
        entries = [_constant(item) for item in value]
        entries.sort(key=lambda item: json.dumps(item, ensure_ascii=True, separators=(",", ":")))
        return ["frozenset", entries]
    raise TypeError(f"Unsupported compiler constant: {type(value).__name__}")


def _normalized_code(code: CodeType) -> dict[str, Any]:
    # Include every CodeType constructor field except the relocated filename.
    # Line and exception tables are deliberately included: edits cannot be
    # ignored merely because they leave similar instruction sequences.
    fields = ("co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals",
              "co_stacksize", "co_flags", "co_code", "co_consts", "co_names",
              "co_varnames", "co_name", "co_qualname", "co_firstlineno",
              "co_linetable", "co_exceptiontable", "co_freevars", "co_cellvars")
    return {field: _constant(getattr(code, field)) for field in fields}


def code_fingerprint(code: CodeType) -> str:
    """SHA256 of exact compiled code, ignoring filenames recursively."""
    if not isinstance(code, CodeType):
        raise TypeError("A compiled Python code object is required")
    canonical = json.dumps(_normalized_code(code), sort_keys=True, ensure_ascii=True,
                           separators=(",", ":")).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _open_archive(executable: Path):
    # PyInstaller is a build dependency, never an app or unit-test dependency.
    from PyInstaller.archive.readers import CArchiveReader
    return CArchiveReader(str(executable))


def verify_frozen_modules(executable: Path | str, project_root: Path | str = ROOT) -> dict[str, str]:
    """Return verified module fingerprints, or reject a stale compiled release.

    Run with the same Python minor version and optimization setting as the
    build. Source compiles with ``dont_inherit=True`` and ``optimize=0``.
    Package ``__init__.py`` modules and the desktop entrypoint are included.
    """
    executable = Path(executable).resolve()
    project_root = Path(project_root).resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"Frozen executable is missing: {executable}")
    archive = _open_archive(executable)
    pyz_names = [name for name in archive.toc if name.startswith("PYZ") and name.endswith(".pyz")]
    if len(pyz_names) != 1:
        raise ValueError("Exactly one embedded PyInstaller PYZ archive is required")
    pyz = archive.open_embedded_archive(pyz_names[0])
    modules = sorted(name for name in pyz.toc if name == "framework" or name.startswith("framework."))
    if not modules:
        raise ValueError("The executable contains no framework modules")
    fingerprints: dict[str, str] = {}
    mismatches: dict[str, dict[str, str]] = {}

    def compare(name: str, source: Path, embedded: CodeType) -> None:
        if not source.is_file():
            mismatches[name] = {"reason": "source is missing", "source": str(source)}
            return
        if not isinstance(embedded, CodeType):
            mismatches[name] = {"reason": "embedded module is not a code object", "source": str(source)}
            return
        current = compile(source.read_bytes(), str(source), "exec", dont_inherit=True, optimize=0)
        source_hash = code_fingerprint(current)
        frozen_hash = code_fingerprint(embedded)
        if source_hash != frozen_hash:
            mismatches[name] = {"reason": "compiled code differs", "source": str(source),
                                "source_sha256": source_hash, "frozen_sha256": frozen_hash}
        else:
            fingerprints[name] = source_hash

    for name in modules:
        relative = Path(*name.split("."))
        is_package = pyz.toc[name][0] == 1
        source = project_root / relative / "__init__.py" if is_package else project_root / relative.with_suffix(".py")
        compare(name, source, pyz.extract(name))
    if "desktop_backend" not in archive.toc:
        mismatches["__entrypoint__.desktop_backend"] = {"reason": "desktop entrypoint is missing"}
    else:
        entry = marshal.loads(archive.extract("desktop_backend"))
        compare("__entrypoint__.desktop_backend", project_root / "framework/app/desktop_backend.py", entry)
    if mismatches:
        raise FrozenCodeMismatch(mismatches)
    return fingerprints


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    arguments = parser.parse_args()
    fingerprints = verify_frozen_modules(arguments.executable, arguments.root)
    print(json.dumps({"status": "passed", "modules": len(fingerprints), "fingerprints": fingerprints}, indent=2))


if __name__ == "__main__":
    main()
