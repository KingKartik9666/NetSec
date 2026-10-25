"""Explicit process-fault fixtures; these are not selectable research models."""
import os
import time


class FakeModelAdapter:
    def __init__(self, config, root):
        self.config = config

    def health_check(self):
        return {'status': 'ready', 'reason': 'Test fixture only'}

    def load(self):
        if self.config.get('break_load'):
            raise ImportError('Injected missing dependency')

    def metadata(self):
        return {'feature_names': ['a'], 'classes': ['benign', 'attack'], 'metrics': None}

    def predict(self, rows):
        if rows == [[13]]:
            os._exit(17)
        if rows == [[14]]:
            time.sleep(5)
        if rows == [[15]]:
            raise RuntimeError('Injected inference failure')
        return [{'label': 'benign', 'score': 0.6} for _ in rows]

    def explain(self, rows, method='native'):
        raise ImportError('Injected explanation plugin missing')
