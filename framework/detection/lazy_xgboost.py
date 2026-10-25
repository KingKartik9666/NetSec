"""Legacy workspace facade; model execution stays in its own process."""
from __future__ import annotations


class LazyXGBoostFacade:
    def __init__(self, registry):
        self.registry = registry

    def _snapshot(self):
        return self.registry.call('xgboost', 'snapshot', {})

    @property
    def bundle(self):
        return self._snapshot()['bundle']

    @property
    def samples(self):
        return self._snapshot()['samples']

    @property
    def feature_names(self):
        return self._snapshot()['feature_names']

    @property
    def classes(self):
        return self._snapshot()['classes']

    @property
    def partial_input_supported(self):
        return self.bundle.get('partial_input_supported', False)

    def status(self):
        return self.registry.call('xgboost', 'legacy_status', {})

    def validate_features(self, values):
        import numpy as np
        row = self.registry.call('xgboost', 'validate_features', {'values': values})
        return np.asarray([float('nan') if v is None else v for v in row], dtype=np.float32)

    def predict_batch(self, rows):
        # The preserved workspace explicitly uses its primary checkpoint;
        # research model selection is independent and cannot silently alter it.
        from framework.detection.trained_xgboost import json_features
        return self.registry.call('xgboost', 'legacy_predict', {'rows': [json_features(row) for row in rows]})

    def analyze(self, values, source):
        return self.registry.call('xgboost', 'analyze', {'values': values, 'source': source})

    def explain(self, values):
        return self.registry.call('xgboost', 'legacy_explain', {'values': values})

    def additional_xai(self, values):
        return self.registry.call('xgboost', 'additional_xai', {'values': values})
