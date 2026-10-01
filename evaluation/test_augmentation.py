"""Offline checks for synthetic-training schema safeguards, without GPU training."""
import numpy as np
import pytest

from framework.training.augmentation import project_generated, validate_generated


def test_projection_preserves_missing_protocols_and_packet_count_constraints():
    names = ["network_packets_all_count", "network_tcp-flags-syn_count", "network_packets_src_count",
             "network_packet-size_min", "network_packet-size_avg", "network_packet-size_max",
             "network_packet-size_std_deviation", "network_protocols_all__present_tcp", "network_interval-packets"]
    donors = np.array([[10, 2, 8, 40, 60, 80, 10, 1, 10], [5, np.nan, 5, 20, 30, 40, 5, 0, 10]], dtype=np.float32)
    fake = np.array([[9.2, 100, 30, 80, 400, 20, -5, .7, 3], [3.1, 9, 20, 50, 100, 10, 50, .2, 5]], dtype=np.float32)
    actual = project_generated(fake, donors, names, np.zeros(len(names)), np.full(len(names), 1000))
    assert validate_generated(actual, donors, names)["passed"]
    assert actual[0, 1] <= actual[0, 0] and actual[0, 2] <= actual[0, 0]
    assert np.isnan(actual[1, 1])
    assert np.array_equal(actual[:, -2:], donors[:, -2:])
    assert np.all(actual[:, 3] <= actual[:, 4]) and np.all(actual[:, 4] <= actual[:, 5])


def test_generated_audit_rejects_changed_protocols_and_noninteger_counts():
    names = ["network_packets_all_count", "network_tcp-flags-syn_count", "network_protocols_all__present_tcp"]
    donors = np.array([[5, 1, 1]], dtype=np.float32)
    changed = np.array([[5.2, 8, .7]], dtype=np.float32)
    result = validate_generated(changed, donors, names)
    assert not result["passed"]
    assert any("metadata changed" in v for v in result["violations"])
    assert any("integer count" in v for v in result["violations"])
    assert any("exceeds total" in v for v in result["violations"])


def test_generated_schema_mismatch_fails_clearly():
    with pytest.raises(ValueError, match="schema"):
        validate_generated(np.zeros((1, 2)), np.zeros((1, 3)), ["a", "b"])


def test_gan_resume_rejects_tampered_synthetic_data_without_loading_gpu(tmp_path):
    import hashlib
    import json
    from framework.training.augmentation import array_sha256, file_sha256, gan_request, train_conditional_gan

    X = np.array([[5], [10]], dtype=np.float32)
    y = np.array([0, 1], dtype=np.int32)
    names = ["network_packets_all_count"]
    original = np.array([[5]], dtype=np.float32)
    labels, donors = np.array([0], dtype=np.int32), np.array([0], dtype=np.int64)
    metadata = {"request": gan_request(X, y, names, 42, 1200, 256), "synthetic_rows": 1,
                "synthetic_sha256": hashlib.sha256(original.tobytes() + labels.tobytes()).hexdigest(),
                "donor_indices_sha256": array_sha256(donors)}
    np.save(tmp_path / "synthetic_features.npy", np.array([[-12.5]], dtype=np.float32))
    np.save(tmp_path / "synthetic_labels.npy", labels)
    np.save(tmp_path / "training_donor_indices.npy", donors)
    (tmp_path / "gan.pt").write_bytes(b"unit test artifact; not a fitted model")
    metadata["checkpoint_sha256"] = file_sha256(tmp_path / "gan.pt")
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="synthetic data checksum"):
        train_conditional_gan(X, y, names, tmp_path)
