"""Optional GPU boosters: real checkpoint and metric checks on synthetic fixtures."""
import numpy as np
import pytest

from framework.training.boosters import nvidia_opencl_device, require_lightgbm_gpu


def test_opencl_runtime_proof_rejects_cpu_or_missing_device_evidence():
    for lines in (["This is the CPU trainer"], ["This is the GPU trainer!!"],
                  ["This is the GPU trainer!!", "Using GPU Device: Intel CPU"]):
        with pytest.raises(RuntimeError, match="GPU execution"):
            require_lightgbm_gpu(lines)
    assert "NVIDIA" in require_lightgbm_gpu(["This is the GPU trainer!!", "Using GPU Device: NVIDIA GeForce RTX"])


@pytest.mark.parametrize("candidate", ["catboost", "lightgbm"])
def test_gpu_booster_best_checkpoint_roundtrip_and_absent_validation_class(candidate, tmp_path):
    pytest.importorskip(candidate)
    if candidate == "catboost":
        from catboost.utils import get_gpu_device_count
        if not get_gpu_device_count():
            pytest.skip("Optional CUDA device unavailable")
    else:
        try:
            nvidia_opencl_device()
        except (RuntimeError, OSError):
            pytest.skip("Optional NVIDIA OpenCL device unavailable")
    from framework.training.boosters import predict_booster, train_booster
    from framework.training.trees import _macro_f1
    rng = np.random.default_rng(50)
    X = rng.normal(size=(240, 5)).astype(np.float32)
    y = np.tile(np.arange(4), 60).astype(np.int32)
    X[:, 0] += y * 3
    X[::19, 2] = np.nan
    keep = y[180:] != 3
    V, v = X[180:][keep], y[180:][keep]
    config = {"iterations": 8, "n_estimators": 8, "depth": 3, "num_leaves": 7,
              "max_bin": 31, "min_data_in_leaf": 5, "early_stopping_rounds": 2,
              "weight_strategy": "sqrt_balanced"}
    result = train_booster(candidate, X[:180], y[:180], V, v, 4, config, 42, tmp_path)
    assert result["gpu_training_verified"] and result["training_rows"] == 180
    assert result["validation_macro_f1"] == pytest.approx(_macro_f1(v, result["predictions"], 4))
    np.testing.assert_array_equal(predict_booster(result["checkpoint_path"], V), result["predictions"])
    with pytest.raises(FileExistsError):
        train_booster(candidate, X[:180], y[:180], V, v, 4, config, 42, tmp_path)
