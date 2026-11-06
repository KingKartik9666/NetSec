"""Plan step 5: the native Rust live77 row equals the Python reference on the shared fixtures.

Native checks skip explicitly until the replay example is built with
CARGO_TARGET_DIR=artifacts/app_v0_6_20261102/cargo-target (cargo build -p netsec-runtime --example feature_replay).
No capture, model execution or dataset partition is involved.
"""
import json

import numpy as np
import pytest

from experiment.verify_live77_replay import DEFAULT_BINARY, FIXTURES, compare, expected_vector, request
from framework.training.csv_v3 import live77_reference as ref

FX = json.loads(FIXTURES.read_text(encoding="utf-8"))
MAC = FX["device"]["mac"]


def test_replay_request_carries_contract_names_device_mac_and_nominal_window():
    f = FX["fixtures"][0]
    req = request(f["frames"], MAC, False)
    assert req["live77_names"] == ref.NAMES and len(req["live77_names"]) == 252
    assert req["seconds"] == ref.NOMINAL_SECONDS == 10.0
    assert req["local_mac"] == MAC and req["local_ips"] == [] and req["capture_drops"] == 0
    assert [fr["hex"] for fr in req["frames"]] == [fr["hex"] for fr in f["frames"]]
    assert request(f["frames"], MAC, True)["capture_drops"] == 1


@pytest.mark.parametrize("fixture", FX["fixtures"], ids=lambda f: f["id"])
def test_expected_vectors_expand_to_the_reference(fixture):
    got = ref.frames_vector(fixture["frames"], MAC, capture_drop=fixture.get("capture_drop", False))
    want = expected_vector(fixture["expected"])
    assert (np.isnan(got) == np.isnan(want)).all() and (got[~np.isnan(got)] == want[~np.isnan(want)]).all()


@pytest.mark.skipif(not DEFAULT_BINARY.is_file(), reason=f"native replay example not built: {DEFAULT_BINARY}")
@pytest.mark.parametrize("fixture", FX["fixtures"], ids=lambda f: f["id"])
def test_native_live77_row_matches_reference_and_fixture(fixture):
    result = compare(fixture["frames"], MAC, bool(fixture.get("capture_drop")), DEFAULT_BINARY,
                     expected_vector(fixture["expected"]))
    assert result["passed"], result["failures"]
    assert result["admitted_frames_native"] == (result["admitted_frames_reference"] or result["admitted_frames_native"])
