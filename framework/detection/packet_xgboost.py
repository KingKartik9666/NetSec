"""Experimental seven-count packet classifier with CPU inference and TreeSHAP.

This separate model is trained on DataSense ten-second windows. Its measured
dataset validation performance does not establish accuracy on Windows traffic.
Counts describe packet headers; sensor logs and protocol guesses are not used.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Real
from pathlib import Path

import numpy as np

from framework.xai.tree_shap import explain_tree

ROOT = Path(__file__).resolve().parents[2]
FEATURE_NAMES = [
    "network_packets_all_count",
    "network_tcp-flags-ack_count",
    "network_tcp-flags-fin_count",
    "network_tcp-flags-psh_count",
    "network_tcp-flags-rst_count",
    "network_tcp-flags-syn_count",
    "network_tcp-flags-urg_count",
]
STATS_KEYS = ["packets", "ack", "fin", "psh", "rst", "syn_all", "urg"]
WINDOW_SECONDS = 10
CLASSES = ["normal", "suspicious"]
LIMITATIONS = [
    "Experimental binary packet-only model; separate from the 84-class, 170-feature research model.",
    "Validation uses DataSense ten-second windows from the existing within-capture split; test remains locked.",
    "Windows PC traffic and the new live extractor have not been evaluated with labelled live observations.",
    "Counts follow the published feature descriptions; the publisher's exact extraction implementation was unavailable.",
    "Normal and suspicious are model predictions, not incident confirmation or a guarantee of safety.",
    "Scores are uncalibrated. Unknown attacks, encrypted application contents and attack attribution are not evaluated.",
    "TreeSHAP explains the raw class margin, not causation or probability percentage points.",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_features(stats: dict) -> np.ndarray:
    """Validate one completed window. SYN+ACK contributes to syn_all and ack."""
    if not isinstance(stats, dict):
        raise ValueError("Packet statistics must be an object")
    if stats.get("window_seconds", WINDOW_SECONDS) != WINDOW_SECONDS:
        raise ValueError("Packet inference requires a completed ten-second window")
    seconds = stats.get("seconds", WINDOW_SECONDS)
    if (isinstance(seconds, (bool, np.bool_)) or not isinstance(seconds, Real)
            or not np.isfinite(seconds) or abs(float(seconds) - WINDOW_SECONDS) > 1):
        raise ValueError("Captured window must be within one second of the ten-second model window")
    counts = []
    for key in STATS_KEYS:
        value = stats.get(key)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{key} must be a nonnegative packet count")
        number = float(value)
        if not np.isfinite(number) or number < 0 or number != np.floor(number) or number > 2**24:
            raise ValueError(f"{key} must be a finite integer count within float32's exact range")
        counts.append(number)
    if counts[0] == 0:
        raise ValueError("No captured packets: there is no traffic window to classify")
    if any(count > counts[0] for count in counts[1:]):
        raise ValueError("A TCP flag count cannot exceed total captured packets")
    return np.asarray([counts], dtype=np.float32)


class PacketXGBoost:
    """Immutable loaded gbtree model; concurrent prediction does not mutate it."""

    def __init__(self, bundle_path: str | Path = ROOT / "framework/models/packet_live/bundle.json"):
        import xgboost as xgb

        self.bundle_path = Path(bundle_path).resolve()
        if not self.bundle_path.is_file():
            raise FileNotFoundError("Packet model missing. Run: python -m experiment.train_packet_model")
        self.bundle = json.loads(self.bundle_path.read_text(encoding="utf-8"))
        if (self.bundle.get("schema_version") != 1
                or self.bundle.get("feature_names") != FEATURE_NAMES
                or self.bundle.get("stats_keys") != STATS_KEYS
                or self.bundle.get("classes") != CLASSES
                or self.bundle.get("window_seconds") != WINDOW_SECONDS
                or self.bundle.get("live_validation") is not False
                or self.bundle.get("test_evaluated") is not False):
            raise ValueError("Packet bundle does not match the frozen experimental schema")
        self.checkpoint = (self.bundle_path.parent / self.bundle["checkpoint"]).resolve()
        if file_sha256(self.checkpoint) != self.bundle["checkpoint_sha256"]:
            raise ValueError("Packet checkpoint checksum differs from the frozen bundle")
        self.booster = xgb.Booster(model_file=str(self.checkpoint))
        self.booster.set_param({"device": "cpu", "nthread": 4})
        config = json.loads(self.booster.save_config())["learner"]
        if (self.booster.num_features() != len(FEATURE_NAMES)
                or int(config["learner_model_param"]["num_class"]) != len(CLASSES)
                or config["objective"]["name"] != "multi:softprob"
                or config["gradient_booster"]["name"] != "gbtree"
                or config["generic_param"]["device"] != "cpu"):
            raise ValueError("Packet checkpoint and feature/class schema disagree")

    def predict(self, stats: dict) -> dict:
        """Classify genuine completed-window counts; never load demo features."""
        import xgboost as xgb

        values = count_features(stats)
        scores = self.booster.predict(xgb.DMatrix(values, nthread=4), strict_shape=True)[0]
        if scores.shape != (2,) or not np.isfinite(scores).all():
            raise RuntimeError("Unexpected packet-classifier probability output")
        index = int(scores.argmax())
        explanation = explain_tree(self.booster, values, FEATURE_NAMES, CLASSES, CLASSES[index])
        return {
            "label": "Model predicts suspicious traffic" if index else "Model predicts normal traffic",
            "is_attack": bool(index), "score": float(scores[index]),
            "attack_score": float(scores[1]), "method": "Experimental packet XGBoost",
            "window_seconds": WINDOW_SECONDS, "live_validation": False,
            "observed_window_seconds": float(stats.get("seconds", WINDOW_SECONDS)),
            "window_timing_note": "Training windows are exactly ten seconds; capture scheduling jitter is not covered by the dataset validation.",
            "explanation": explanation,
            "dataset_validation_accuracy": float(self.bundle["validation_metrics"]["accuracy"]),
            "limitations": " ".join(LIMITATIONS),
        }
