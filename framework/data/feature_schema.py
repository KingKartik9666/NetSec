"""One measured-input schema for partial CSVs and completed packet windows.

Missing measurements are NaN, including unobserved protocol-presence flags.
An observed empty protocol list is different: its presence flags are zero.
"""
from __future__ import annotations

import ast
import csv
import io

import numpy as np
import pandas as pd

from framework.detection.packet_xgboost import FEATURE_NAMES, STATS_KEYS, count_features

PACKET_COLUMNS = tuple(FEATURE_NAMES)
PROTOCOL_FIELDS = ("network_protocols_all", "network_protocols_src", "network_protocols_dst")


def packet_vector(stats: dict, names: list[str]) -> np.ndarray:
    if any(key not in stats for key in [*STATS_KEYS, "seconds"]):
        raise ValueError("A packet vector requires every measured count and the completed window duration")
    counts = count_features(stats)[0]
    if not set(PACKET_COLUMNS).issubset(names):
        raise ValueError("The checkpoint does not contain the canonical packet count columns")
    values = np.full(len(names), np.nan, dtype=np.float32)
    for name, value in zip(PACKET_COLUMNS, counts):
        values[names.index(name)] = value
    return values


def vector_csv(values, names: list[str]) -> bytes:
    if len(values) != len(names):
        raise ValueError("Feature vector and schema differ")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(names)
    writer.writerow([float(v) if v is not None and np.isfinite(v) else "" for v in values])
    return stream.getvalue().encode("utf-8")


def compatibility(names):
    return [{"feature": name, "live_mapping": name in PACKET_COLUMNS} for name in names]


def feature_coverage(values, bundle: dict) -> dict:
    names = bundle["feature_names"]
    finite = np.isfinite(np.asarray(values, dtype=np.float32))
    observed_names = {name for name, present in zip(names, finite) if present}
    profile = ("full_csv" if finite.all() else "packet_counts_7"
               if observed_names == set(PACKET_COLUMNS) else "partial_custom")
    live_validation = bundle.get("live_profile_validation", {})
    packet_issue = None
    if profile == "packet_counts_7":
        try:
            count_features({key: values[names.index(name)]
                            for key, name in zip(STATS_KEYS, PACKET_COLUMNS)})
        except ValueError as error:
            packet_issue = str(error)
    validated = (profile == "full_csv" or
                 (profile == "packet_counts_7" and packet_issue is None
                  and live_validation.get("validation_rows", 0) > 0))
    return {"observed": int(finite.sum()), "missing": int((~finite).sum()), "total": len(names),
            "profile": profile, "validated_profile": validated,
            "validation_scope": ("Outside the evaluated packet-count profile: " + packet_issue
                                 if packet_issue else "DataSense source-partition validation only; not labelled live-PC accuracy."
                                 if validated else "This combination of observed and missing features has not been validated."),
            "features": [{**entry, "value": float(v) if present else None,
                          "status": "measured" if present else "missing"}
                         for entry, v, present in zip(compatibility(names), values, finite)]}


def transform_partial_frame(frame: pd.DataFrame, bundle: dict) -> np.ndarray:
    names = bundle["feature_names"]
    numeric = set(bundle["numeric_features"])
    known = set(names) | set(PROTOCOL_FIELDS)
    unexpected = [name for name in frame if name not in known and name.startswith(("network_", "log_"))]
    if unexpected:
        raise ValueError("Unrecognized feature columns: " + ", ".join(unexpected[:8]))
    values = np.full((len(frame), len(names)), np.nan, dtype=np.float32)
    for index, name in enumerate(names):
        if name not in frame:
            continue
        column = frame[name].replace(r"^\s*$", np.nan, regex=True)
        numbers = pd.to_numeric(column, errors="raise").to_numpy(dtype=np.float64)
        if np.isinf(numbers).any() or np.any(np.abs(numbers[np.isfinite(numbers)]) > np.finfo(np.float32).max):
            raise ValueError(f"{name} contains an infinite or out-of-range value")
        if name not in numeric and not np.isin(numbers[np.isfinite(numbers)], [0, 1]).all():
            raise ValueError("Prepared protocol flags must be zero, one, or blank when unmeasured")
        values[:, index] = numbers.astype(np.float32)
    for field in PROTOCOL_FIELDS:
        if field not in frame:
            continue
        vocabulary = bundle["protocol_vocabulary"][field]
        columns = [f"{field}__present_{token}" for token in vocabulary + ["UNSEEN"]]
        if any(column in frame for column in columns):
            raise ValueError(f"Use either the raw {field} list or its prepared flags, not both")
        indices = [names.index(column) for column in columns]
        mapping = {token: index for token, index in zip(vocabulary, indices)}
        for row, text in enumerate(frame[field]):
            if pd.isna(text) or (isinstance(text, str) and not text.strip()):
                continue
            if not isinstance(text, str) or len(text) > 8192:
                raise ValueError(f"{field} requires a bounded Python list string")
            try:
                tokens = ast.literal_eval(text)
            except (ValueError, SyntaxError, TypeError, RecursionError) as exc:
                raise ValueError(f"Invalid protocol list in {field}, row {row + 1}") from exc
            if not isinstance(tokens, list) or any(not isinstance(t, str) or len(t) > 80 for t in tokens):
                raise ValueError(f"Invalid protocol list in {field}, row {row + 1}")
            values[row, indices] = 0
            for token in set(tokens):
                values[row, mapping.get(token, indices[-1])] = 1
    if not np.isfinite(values).any(axis=1).all():
        raise ValueError("Each row needs at least one usable measurement with a recognized feature name")
    return values
