"""Compiled release code must match current source, even when filenames move."""
import marshal
import struct

import pytest

from experiment import frozen_code_verification as frozen


def test_fingerprint_accepts_relocated_filenames_in_nested_functions():
    source = "def outer():\n    def inner():\n        return 'current prompt'\n    return inner\n"
    first = compile(source, "K:/workspace/module.py", "exec", dont_inherit=True, optimize=0)
    relocated = compile(source, "framework/module.py", "exec", dont_inherit=True, optimize=0)
    assert frozen.code_fingerprint(first) == frozen.code_fingerprint(relocated)


@pytest.mark.parametrize("changed", [
    "def explain():\n    return 'old prompt'\n",
    "def explain():\n    return 1 + len('current prompt')\n",
    "\ndef explain():\n    return 'current prompt'\n",
])
def test_fingerprint_rejects_stale_constants_function_code_and_line_metadata(changed):
    current = compile("def explain():\n    return 'current prompt'\n", "source.py", "exec", dont_inherit=True, optimize=0)
    stale = compile(changed, "frozen.py", "exec", dont_inherit=True, optimize=0)
    assert frozen.code_fingerprint(current) != frozen.code_fingerprint(stale)


def test_fingerprint_orders_frozensets_and_preserves_constant_types():
    source = "values = {'one', 'two', 'three'}\nnumber = 1.5 + 2j\n"
    code = compile(source, "first.py", "exec", dont_inherit=True, optimize=0)
    roundtrip = marshal.loads(marshal.dumps(code))
    assert frozen.code_fingerprint(code) == frozen.code_fingerprint(roundtrip)


def test_fingerprint_preserves_nan_payload_bits():
    code = compile("value = 1.0\n", "source.py", "exec", dont_inherit=True, optimize=0)
    first_nan = struct.unpack(">d", bytes.fromhex("7ff8000000000001"))[0]
    second_nan = struct.unpack(">d", bytes.fromhex("7ff8000000000002"))[0]
    first = code.replace(co_consts=(first_nan, None))
    second = code.replace(co_consts=(second_nan, None))
    assert frozen.code_fingerprint(first) != frozen.code_fingerprint(second)


class FakePyz:
    def __init__(self, modules):
        self.modules = modules
        self.toc = {name: (package, 0, 0) for name, (package, _code) in modules.items()}

    def extract(self, name):
        return self.modules[name][1]


class FakeArchive:
    def __init__(self, pyz, entry):
        self.pyz = pyz
        self.entry = entry
        self.toc = {"PYZ.pyz": (), "desktop_backend": ()}

    def open_embedded_archive(self, name):
        assert name == "PYZ.pyz"
        return self.pyz

    def extract(self, name):
        assert name == "desktop_backend"
        return marshal.dumps(self.entry)


def fixture_archive(tmp_path, monkeypatch, *, stale=False, missing_source=False):
    sources = {"framework/__init__.py": '"""Framework package."""\n',
               "framework/app/__init__.py": '"""Application package."""\n',
               "framework/app/server.py": "def explain():\n    return 'current prompt'\n",
               "framework/app/desktop_backend.py": "def main():\n    return 'ready'\n"}
    for name, source in sources.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    modules = {}
    for module, name, package in [("framework", "framework/__init__.py", 1),
                                   ("framework.app", "framework/app/__init__.py", 1),
                                   ("framework.app.server", "framework/app/server.py", 0)]:
        source = sources[name]
        if stale and module == "framework.app.server":
            source = source.replace("current prompt", "old prompt")
        modules[module] = (package, compile(source, name, "exec", dont_inherit=True, optimize=0))
    entry = compile(sources["framework/app/desktop_backend.py"], "desktop_backend.py", "exec", dont_inherit=True, optimize=0)
    archive = FakeArchive(FakePyz(modules), entry)
    monkeypatch.setattr(frozen, "_open_archive", lambda _path: archive)
    executable = tmp_path / "backend.exe"
    executable.write_bytes(b"verification fixture")
    if missing_source:
        (tmp_path / "framework/app/server.py").unlink()
    return executable, archive


def test_verifies_package_init_modules_and_entrypoint(tmp_path, monkeypatch):
    executable, _ = fixture_archive(tmp_path, monkeypatch)
    result = frozen.verify_frozen_modules(executable, tmp_path)
    assert set(result) == {"framework", "framework.app", "framework.app.server", "__entrypoint__.desktop_backend"}
    assert all(len(value) == 64 for value in result.values())


def test_verifier_rejects_stale_embedded_module(tmp_path, monkeypatch):
    executable, _ = fixture_archive(tmp_path, monkeypatch, stale=True)
    with pytest.raises(frozen.FrozenCodeMismatch, match="framework.app.server") as failure:
        frozen.verify_frozen_modules(executable, tmp_path)
    detail = failure.value.mismatches["framework.app.server"]
    assert detail["source_sha256"] != detail["frozen_sha256"]


def test_verifier_rejects_missing_source(tmp_path, monkeypatch):
    executable, _ = fixture_archive(tmp_path, monkeypatch, missing_source=True)
    with pytest.raises(frozen.FrozenCodeMismatch, match="source is missing"):
        frozen.verify_frozen_modules(executable, tmp_path)


def test_verifier_rejects_stale_entrypoint(tmp_path, monkeypatch):
    executable, archive = fixture_archive(tmp_path, monkeypatch)
    archive.entry = compile("def main():\n    return 'old'\n", "desktop_backend.py", "exec", dont_inherit=True, optimize=0)
    with pytest.raises(frozen.FrozenCodeMismatch, match="__entrypoint__.desktop_backend"):
        frozen.verify_frozen_modules(executable, tmp_path)
