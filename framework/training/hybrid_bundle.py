"""CPU inference for saved research hybrids with exact feature/class alignment."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from experiment.train_verified_clean import read
from framework.training.hybrid_protocol import ROOT, verify_hashes
from framework.training.hybrid_blend import blend
from experiment.datasense_quality import temperature_probabilities


def branch_predict(branch, X):
    """Load a branch once and release it before the next branch is loaded."""
    verify_hashes(branch['assets'])
    family, checkpoint = branch['family'], ROOT/branch['checkpoint']
    if family in ('tabm_ne', 'tabpack'):
        from framework.training.hybrid_neural import load, predict
        model, prep, saved, weights = load(checkpoint)
        p = predict(model, prep, X, family, weights)
        return p
    from experiment.train_csv_models import reload_outputs
    return reload_outputs(family, checkpoint, X)[1]


class HybridPredictor:
    def __init__(self, bundle, *, _verification_only=False):
        self.path = Path(bundle)
        self.bundle = read(self.path)
        allowed = ('completed', 'pending_verification') if _verification_only else ('completed',)
        if self.bundle.get('status') not in allowed:
            raise ValueError('Hybrid bundle is not complete')
        if self.bundle['combiner']['names'] != list(self.bundle['branches']):
            raise ValueError('Bundle branch order differs')
        if self.bundle['combiner']['n_classes'] != len(self.bundle['classes']):
            raise ValueError('Bundle class schema differs')
        if not _verification_only:
            completion_path = self.path.parent/'result.json'
            if not completion_path.exists():
                raise ValueError('Verified hybrid completion record is missing')
            completion = read(completion_path)
            pins = completion.get('artifact_hashes', {})
            if completion.get('status') != 'completed' or not any((ROOT/name).resolve()==self.path.resolve() for name in pins):
                raise ValueError('Invalid hybrid completion record')
            verify_hashes(pins)
        for branch in self.bundle['branches'].values():
            verify_hashes(branch['assets'])
            schema_path = branch.get('schema_manifest')
            if not schema_path or schema_path not in branch['assets']:
                raise ValueError('Branch ordered schema is not pinned')
            schema = read(ROOT/schema_path)
            if any(schema[key] != self.bundle[key] for key in ('feature_names', 'classes')):
                raise ValueError('Bundle and branch ordered schema differ')
            if branch['family'] in ('tabm_ne', 'tabpack'):
                import torch
                saved = torch.load(ROOT/branch['checkpoint'], map_location='cpu', weights_only=True)
                if any(saved[key] != self.bundle[key] for key in ('feature_names', 'classes')):
                    raise ValueError('Neural checkpoint ordered schema differs')
                if saved['preprocessor']['feature_names'] != self.bundle['feature_names']:
                    raise ValueError('Neural preprocessing schema differs')
                del saved

    def predict_proba(self, X, feature_names, calibrated=True):
        X = np.asarray(X, dtype=np.float32)
        if list(feature_names) != self.bundle['feature_names']:
            raise ValueError('Feature names/order differ from the saved hybrid')
        if X.ndim != 2 or X.shape[1] != len(feature_names) or not len(X) or np.isinf(X).any():
            raise ValueError('Invalid raw input shape or infinite values')
        probabilities = np.stack([branch_predict(b, X) for b in self.bundle['branches'].values()])
        p = blend(probabilities, self.bundle['combiner'], list(self.bundle['branches']))
        if calibrated:
            p = temperature_probabilities(p, self.bundle['calibration']['temperature'])
        return p
