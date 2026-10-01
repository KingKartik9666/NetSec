"""Load synthetic feature tables without exposing labels to inference."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES = (
    "packet_rate", "byte_rate", "syn_ratio", "connection_rate",
    "unique_destinations", "avg_packet_size",
)
KNOWN_LABELS = ("BENIGN", "DDoS", "PORT_SCAN")
SAMPLE_NAMES = ("benign", "ddos", "scan", "unknown")
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "demo" / "sample_traffic.csv"


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ordered numeric features; reject missing or invalid measurements."""
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing features: {', '.join(sorted(missing))}")
    try:
        values = frame.loc[:, list(FEATURES)].astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Traffic features must be numeric.") from exc
    if values.empty or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Traffic features must be non-empty and finite.")
    if (values < 0).any().any():
        raise ValueError("Traffic features cannot be negative.")
    if (values["syn_ratio"] > 1).any():
        raise ValueError("syn_ratio must be between 0 and 1.")
    return values


def sample_frame(sample: Mapping[str, float]) -> pd.DataFrame:
    """Extract only declared features, excluding metadata and ground truth."""
    return validate_features(pd.DataFrame([dict(sample)]))


def load_demo_data(path: Path | str | None = None) -> pd.DataFrame:
    """Read and validate the demonstration CSV relative to the project root."""
    source = Path(path) if path is not None else DEFAULT_DATASET
    if not source.is_file():
        raise FileNotFoundError(f"Traffic dataset not found: {source}")
    frame = pd.read_csv(source)
    required = {"sample_id", "split", "label", *FEATURES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {', '.join(sorted(missing))}")
    # Replace columns rather than assigning floats into an inferred integer
    # block in place (pandas 3 rejects that dtype-changing .loc assignment).
    frame[list(FEATURES)] = validate_features(frame)
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be present and unique.")
    if not frame["split"].isin(["train", "demo"]).all():
        raise ValueError("split must be train or demo.")
    if not frame["label"].isin([*KNOWN_LABELS, "UNKNOWN"]).all():
        raise ValueError("Dataset contains an unsupported label.")
    training_data(frame)
    return frame


def training_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Use only known, explicitly marked training rows; never demo/UNKNOWN rows."""
    train = frame.loc[frame["split"] == "train"]
    if set(train["label"]) != set(KNOWN_LABELS):
        raise ValueError("Training requires BENIGN, DDoS and PORT_SCAN only.")
    return validate_features(train), train["label"].copy()


def select_sample(frame: pd.DataFrame, name: str) -> dict[str, float]:
    """Select a held-out example and strip its id, split and synthetic label."""
    if name not in SAMPLE_NAMES:
        raise ValueError(f"Unknown demo sample {name!r}; choose {', '.join(SAMPLE_NAMES)}.")
    rows = frame.loc[(frame["split"] == "demo") & (frame["sample_id"] == name)]
    if len(rows) != 1:
        raise ValueError(f"Expected one held-out demo row for {name!r}.")
    return {feature: float(rows.iloc[0][feature]) for feature in FEATURES}
