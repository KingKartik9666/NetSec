"""Offline checks for the live77 bundle check in tools/verify_native_app.py (plan step 9).

No runtime, packet capture, language server or model inference is started; only the shipped bundle bytes, the Rust
pins and validation-free metadata are read. The held-out test partition is never touched.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tools.verify_native_app import LIVE77, ROOT, live77_bundle, rust_pins


def _copy(tmp_path: Path) -> Path:
    folder = tmp_path / LIVE77
    folder.mkdir(parents=True)
    for name in ("bundle.json", "model.ubj"):
        shutil.copyfile(ROOT / LIVE77 / name, folder / name)
    return folder


def _mutate(tmp_path: Path, change) -> dict:
    """Rewrite the copied bundle and approve its new hash, so each guard is reached independently of the SHA pin."""
    path = _copy(tmp_path) / "bundle.json"
    bundle = json.loads(path.read_text(encoding="utf-8"))
    change(bundle)
    path.write_text(json.dumps(bundle), encoding="utf-8")
    return dict(rust_pins(), LIVE77_BUNDLE_SHA=hashlib.sha256(path.read_bytes()).hexdigest())


def test_rust_pins_are_read_from_the_runtime_source():
    pins = rust_pins()
    assert pins["LIVE77_MODEL_ID"] == "live77_xgboost"
    assert pins["LIVE77_BUNDLE"] == LIVE77.as_posix() + "/bundle.json"
    assert pins["WINDOW_SECONDS"] == "10"  # owner decision 3 Nov 2026 (10 s, not the plan's 1 s)
    for key in ("LIVE77_BUNDLE_SHA", "LIVE77_CHECKPOINT_SHA", "LIVE77_NAMES_SHA", "LIVE77_CLASSES_SHA", "CLASS_MERGE_SPEC_SHA"):
        assert len(pins[key]) == 64 and int(pins[key], 16) >= 0, key
    assert pins["label"] == "Live XGBoost (77-class, 10 s windows)"


def test_shipped_live77_bundle_passes_every_pinned_guard_without_capture():
    result = live77_bundle()
    assert result["passed"], result["failures"]
    assert result["failures"] == [] and result["bundle_sha256"] == rust_pins()["LIVE77_BUNDLE_SHA"]
    assert (result["features"], result["classes"], result["window_seconds"]) == (252, 77, 10)
    assert result["merge"] == "recon_ping-sweep (71) -> recon_scan (70)"
    assert result["label"] == "Live XGBoost (77-class, 10 s windows)"
    assert result["provenance_note"].startswith("Trained on DataSense windows of 1-10 s")
    assert result["packet_capture_started"] is False and result["model_executed"] is False


def test_the_shipped_provenance_note_survives_the_live_caveat_filter():
    result = live77_bundle()
    assert not result["caveat_filter"].search(result["provenance_note"]) and not result["caveat_filter"].search(result["label"])
    assert result["caveat_filter"].search("live accuracy unverified")  # the filter really is the Live page's


def test_bundle_bytes_differing_from_the_rust_pin_are_refused(tmp_path):
    path = _copy(tmp_path) / "bundle.json"
    path.write_text(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=1), encoding="utf-8")
    result = live77_bundle(tmp_path)
    assert not result["passed"] and "bundle SHA256 equals the Rust pin" in result["failures"]


def test_missing_bundle_fails_and_reports_no_substitute(tmp_path):
    result = live77_bundle(tmp_path)
    assert not result["passed"] and result["error"] and result["packet_capture_started"] is False


DECIDED = "label and provenance note equal the owner's decisions file (defaults_chosen_by_app_chat)"
PINNED = "bundle pins the current owner decisions file (sha256)"


def _swap_scan(bundle):
    bundle["classes"][70], bundle["classes"][71] = bundle["classes"][71], bundle["classes"][70]


@pytest.mark.parametrize("change, failure", [
    (_swap_scan, "77 classes in the pinned order, equal to the CSV catalog"),
    (lambda b: b.pop("class_merge"), "class merge ping_sweep_into_recon_scan v1, 71 -> 70, pinned spec"),
    (lambda b: b["class_merge"].update(source_index=70, target_index=71), "class merge ping_sweep_into_recon_scan v1, 71 -> 70, pinned spec"),
    (lambda b: b.update(window_seconds=1), "10 s window"),
    (lambda b: b.update(test_evaluated=True), "test_evaluated and live_validation false"),
    (lambda b: b["output"].update(binary_attack_probability={"kind": "platt"}), "temperature calibration, no logit adjustment, no binary attack probability"),
    (lambda b: b["feature_names"].reverse(), "252 feature names in the pinned order"),
    (lambda b: b.update(provenance_note="Live accuracy unverified."), "label equals the runtime label; label and provenance note survive the Live caveat filter"),
    (lambda b: b.update(label="Experimental packet-only XGBoost"), "label equals the runtime label; label and provenance note survive the Live caveat filter"),
    (lambda b: b.update(checkpoint="../../../../../model.ubj"), "checkpoint inside the bundle folder; bytes and SHA256 match the bundle and the Rust pin"),
    (lambda b: b.update(provenance_note="Trained on DataSense windows; scored on 10 s windows. The held-out test was not used."), DECIDED),
    (lambda b: b["source"].update(decisions_sha256="0" * 64), PINNED),
])
def test_each_guard_refuses_its_mutation(tmp_path, change, failure):
    result = live77_bundle(tmp_path, pins=_mutate(tmp_path, change))
    assert not result["passed"] and failure in result["failures"], result["failures"]


def test_changed_checkpoint_bytes_are_refused(tmp_path):
    with (_copy(tmp_path) / "model.ubj").open("ab") as stream:
        stream.write(b"\0")
    result = live77_bundle(tmp_path)
    assert result["failures"] == ["checkpoint inside the bundle folder; bytes and SHA256 match the bundle and the Rust pin"]


def test_shipped_label_and_note_are_the_owner_decisions_verbatim():
    d = json.loads((ROOT / "artifacts/app_v0_6_20261102/live77_decisions.json").read_text(encoding="utf-8"))["defaults_chosen_by_app_chat"]
    result = live77_bundle()
    assert (result["label"], result["provenance_note"]) == (d["label"], d["provenance_note"])
    assert result["checks"][DECIDED] and result["checks"][PINNED]


def test_evidence_says_the_rust_loader_is_covered_by_cargo_not_by_this_phase():
    coverage = live77_bundle()["rust_loader_coverage"]
    assert "cargo test -p netsec-runtime live::models" in coverage and "not executed by this phase" in coverage


def test_live77_only_run_needs_no_release_executable(tmp_path, monkeypatch):
    import tools.verify_native_app as app
    monkeypatch.setattr(app, "NativeRuntime", lambda *_, **__: (_ for _ in ()).throw(AssertionError("runtime started")))
    monkeypatch.setattr("sys.argv", ["verify_native_app.py", "--phases", "live77", "--binary", str(tmp_path / "absent.exe"), "--output", str(tmp_path / "out")])
    assert app.main() == 0
    report = json.loads((tmp_path / "out/report.json").read_text(encoding="utf-8"))
    assert not (tmp_path / "out/bin/netsec-runtime.exe").exists() and report["passed"]
    assert [c["name"] for c in report["checks"]] == ["live77 bundle SHA, classes and merge load against the Rust pins (offline; no capture)"]


def test_live77_phase_runs_offline_without_starting_the_runtime(tmp_path, monkeypatch):
    import tools.verify_native_app as app
    assert "live77" in app.PHASES

    def refuse(*_, **__):
        raise AssertionError("the live77 phase must not start the native runtime")
    monkeypatch.setattr(app, "NativeRuntime", refuse)
    binary = tmp_path / "netsec-runtime.exe"
    binary.write_bytes(b"stand-in; never executed")
    monkeypatch.setattr("sys.argv", ["verify_native_app.py", "--phases", "live77", "--binary", str(binary), "--output", str(tmp_path / "out")])
    assert app.main() == 0
    report = json.loads((tmp_path / "out/report.json").read_text(encoding="utf-8"))
    check = next(c for c in report["checks"] if c["name"].startswith("live77 bundle"))
    assert check["passed"] and check["packet_capture_started"] is False and report["packet_capture_started"] is False
    assert report["test_arrays_opened"] is False and report["passed"]


def test_estimate_wording_no_longer_names_the_retired_84_or_2_class_live_models():
    source = (ROOT / "tools/verify_native_app.py").read_text(encoding="utf-8")
    assert "84- or 2-class" not in source
