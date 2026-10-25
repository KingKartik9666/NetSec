"""CPU-only exported candidate models with calibrated scores and schema pins."""
from pathlib import Path

import numpy as np

from framework.training.hybrid_protocol import ROOT,verify_hashes
from framework.training.study_supervisor import read


class FinalModelPredictor:
    def __init__(self,path,*,_verification_only=False):
        self.path=Path(path);self.bundle=read(path)
        allowed=('completed','pending_verification') if _verification_only else ('completed',)
        if self.bundle.get('status') not in allowed or not self.bundle.get('assets'):
            raise ValueError('Model bundle has not passed completion checks')
        verify_hashes(self.bundle['assets'])

    def predict(self,X,feature_names):
        from framework.training.day_worker import predict_saved
        from framework.training.tabm_calibration import apply_calibration
        if list(feature_names)!=self.bundle['feature_names']:
            raise ValueError('Feature names/order differ from bundle')
        X=np.asarray(X,dtype=np.float32)
        if X.ndim!=2 or X.shape[1]!=len(feature_names) or not len(X) or np.isinf(X).any():
            raise ValueError('Invalid inference matrix')
        record=self.bundle['record'];values=predict_saved(record,X)
        if record['family']=='hem':
            import joblib
            pred,parts=values
            multi,binary,knn=[parts[k] for k in ('multiclass_probabilities','binary_probabilities','knn_predictions')]
            voted=multi[np.arange(len(X)),pred];matching=np.where(pred==0,1-binary,binary)
            features=np.column_stack([np.log(np.clip(voted,1e-12,1)),np.log(np.clip(matching,1e-12,1)),
                (knn==multi.argmax(1)).astype(float)])
            model=joblib.load(ROOT/self.bundle['correctness_calibrator'])
            confidence=model.predict_proba(features)[:,list(model.classes_).index(True)]
            return dict(predictions=pred,probabilities=None,correctness_confidence=confidence,
                confidence_meaning='Correctness of unchanged hard HEM vote, not ensemble class probabilities')
        probabilities,attack=apply_calibration(values,self.bundle['calibration'])
        pred=values.argmax(1)
        np.testing.assert_array_equal(probabilities.argmax(1),pred)
        return dict(predictions=pred,probabilities=probabilities,attack_probability=attack)
