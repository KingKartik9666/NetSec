"""Calibrate unchanged class-decision correctness, separate from class scores.

Uses the same pinned early calibration rows and later assessment. A fixed
regularized Platt recipe is fit once, without assessment-based tuning.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

from experiment.audit_datasense_raw import DEFAULT
from experiment.calibrate_datasense_models import confidence_metrics,partition
from experiment.datasense_quality import write_json


def run():
    fit,evaluate,truth=partition(DEFAULT)
    summary=json.loads((DEFAULT/'calibration/summary.json').read_text())
    rows={}
    for family in ('xgboost','kan','random_forest'):
        entry=summary[family];path=DEFAULT/entry['model_run'];out=DEFAULT/'calibration'/family
        destination=out/'correctness_confidence_result.json'
        if destination.exists():raise FileExistsError(destination)
        probability=np.load(path/'validation_probabilities.npy')
        raw=probability.max(1);correct=probability.argmax(1)==truth
        X=logit(np.clip(raw,1e-6,1-1e-6))[:,None]
        estimator=LogisticRegression(C=1.,random_state=42,max_iter=1000).fit(X[fit],correct[fit])
        if estimator.coef_[0,0]<0:raise ValueError('Learned confidence ordering reversed')
        confidence=estimator.predict_proba(X)[:,1]
        joblib.dump(estimator,out/'decision_confidence_calibrator.joblib')
        np.testing.assert_allclose(joblib.load(out/'decision_confidence_calibrator.joblib').predict_proba(X)[:,1],confidence,atol=0,rtol=0)
        np.save(out/'decision_correctness_confidence.npy',confidence)
        before=confidence_metrics(correct[evaluate],raw[evaluate]);after=confidence_metrics(correct[evaluate],confidence[evaluate])
        result={'family':family,'method':'regularized Platt calibration of logit top-class score',
                'meaning':'Probability the unchanged predicted class is correct; not a multiclass probability distribution.',
                'recipe':{'C':1.,'max_iter':1000,'seed':42},'calibration_rows':len(fit),'assessment_rows':len(evaluate),
                'assessment_before':before,'assessment_after':after,'predictions_unchanged':True,
                'calibrator_reload_verified':True,'assessment_used_for_fitting_or_tuning':False,
                'class_probability_temperature_results_retained':True,
                'research_status':'Exploratory later within-class validation; original validation previously reused.'}
        write_json(destination,result);rows[family]=result
        print(f'{family}: decision-confidence ECE {before["ece_10_equal_width_bins"]:.5f} -> {after["ece_10_equal_width_bins"]:.5f}',flush=True)
    write_json(DEFAULT/'calibration/correctness_confidence_summary.json',rows)


if __name__=='__main__':run()
