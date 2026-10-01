"""The experimental packet model rejects incomplete or invented observations."""
import hashlib
import json

import numpy as np
import pytest

from framework.detection.packet_xgboost import (
    CLASSES, FEATURE_NAMES, STATS_KEYS, PacketXGBoost, count_features,
)


def counts(**changes):
    # Synthetic header counts used only as an input-contract test fixture.
    stats = dict(zip(STATS_KEYS, [100, 80, 2, 60, 1, 10, 0]))
    stats.update(window_seconds=10, seconds=10, syn=3)
    stats.update(changes)
    return stats


def test_packet_features_use_total_syn_flags_including_syn_ack():
    values = count_features(counts())
    assert values.dtype == np.float32
    assert values.tolist() == [[100, 80, 2, 60, 1, 10, 0]]


@pytest.mark.parametrize("changes", [
    {"seconds": 15}, {"window_seconds": 2}, {"packets": 0},
    {"ack": 101}, {"fin": -1}, {"rst": .5}, {"urg": True},
    {"syn_all": np.nan}, {"packets": 2**24 + 1},
])
def test_invalid_packet_observations_are_not_classified(changes):
    with pytest.raises(ValueError):
        count_features(counts(**changes))


@pytest.fixture
def packet_model(tmp_path):
    xgb = pytest.importorskip("xgboost")
    # CPU-only tiny unit fixture, not a research training run or accuracy result.
    rng = np.random.default_rng(42)
    X = rng.integers(0, 10, (40, 7)).astype(np.float32)
    X[:, 0] += 20
    y = (X[:, 5] > 4).astype(np.int32)
    model = xgb.train({"objective": "multi:softprob", "num_class": 2,
                       "device": "cpu", "max_depth": 2, "seed": 42, "nthread": 2},
                      xgb.DMatrix(X, label=y), num_boost_round=4)
    checkpoint = tmp_path / "model.ubj"
    model.save_model(checkpoint)
    bundle = {"schema_version": 1, "feature_names": FEATURE_NAMES,
              "stats_keys": STATS_KEYS, "classes": CLASSES,
              "window_seconds": 10, "live_validation": False,
              "test_evaluated": False, "checkpoint": "model.ubj",
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "validation_metrics": {"accuracy": .5}}
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    return PacketXGBoost(path)


def test_packet_prediction_is_separate_cpu_model_with_exact_shap(packet_model):
    result = packet_model.predict(counts())
    explanation = result["explanation"]
    assert result["live_validation"] is False
    assert result["method"] == "Experimental packet XGBoost"
    assert 0 <= result["score"] <= 1
    assert len(explanation["all_features"]) == 7
    assert explanation["method"] == "Exact TreeSHAP (native XGBoost)"
    assert explanation["bias"] + explanation["contribution_sum"] == pytest.approx(explanation["margin"], abs=1e-4)
    assert json.loads(packet_model.booster.save_config())["learner"]["generic_param"]["device"] == "cpu"


def test_packet_checkpoint_tamper_never_becomes_a_normal_verdict(packet_model):
    packet_model.checkpoint.write_bytes(b"invalid checkpoint")
    with pytest.raises(ValueError, match="checksum"):
        PacketXGBoost(packet_model.bundle_path)
