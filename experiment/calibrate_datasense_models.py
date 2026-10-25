"""Exploratory calibration on a purged earlier validation portion.

Assess later rows separately. No locked-test access, training-set calibration,
prediction relabeling, or validation-accuracy inflation from temperature scaling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit,softmax

from experiment.audit_datasense_raw import DEFAULT,PARENT
from experiment.datasense_quality import calibration_partition,digest,temperature_probabilities,write_json
from experiment.train_datasense_quality import probability_metrics


def temperature_fit(probabilities,truth):
    def loss(log_t):
        p=temperature_probabilities(probabilities,np.exp(log_t))
        return float(-np.log(np.maximum(p[np.arange(len(truth)),truth],1e-12)).mean())
    fit=minimize_scalar(loss,bounds=(np.log(.1),np.log(10.)),method='bounded')
    if not fit.success:raise RuntimeError('Temperature fit failed')
    return float(np.exp(fit.x))


def confidence_metrics(correct,confidence):
    correct=np.asarray(correct,dtype=bool);confidence=np.asarray(confidence,dtype=float)
    result={'rows':len(correct),'accuracy':float(correct.mean()),'mean_confidence':float(confidence.mean()),
            'correctness_brier_score':float(np.mean((confidence-correct)**2)),
            'correctness_log_loss':float(-np.mean(np.where(correct,np.log(np.clip(confidence,1e-12,1)),np.log(np.clip(1-confidence,1e-12,1)))))}
    bins=[];ece=0.
    for i in range(10):
        low=i/10;high=(i+1)/10
        mask=(confidence>=low)&((confidence<high) if i<9 else (confidence<=1))
        entry={'lower':low,'upper':high,'rows':int(mask.sum()),'accuracy':None,'confidence':None}
        if mask.any():
            entry.update(accuracy=float(correct[mask].mean()),confidence=float(confidence[mask].mean()))
            ece+=float(mask.mean()*abs(entry['accuracy']-entry['confidence']))
        bins.append(entry)
    result['ece_10_equal_width_bins']=ece;result['bins']=bins
    return result


def assessment(probability,truth,manifest):
    result=probability_metrics(truth,probability,manifest['classes'],manifest['benign_index'])
    result['multiclass_brier_score']=float(np.mean(np.sum(probability**2,axis=1)-2*probability[np.arange(len(truth)),truth]+1))
    result['reliability']=confidence_metrics(probability.argmax(1)==truth,probability.max(1))
    return result


def partition(directory):
    out=directory/'calibration';out.mkdir(exist_ok=True)
    base=directory/'04_port_refinement'
    y=np.load(base/'y_validation.npy');start=np.load(base/'window_start_validation.npy');end=np.load(base/'window_end_validation.npy')
    hashes=np.load(PARENT/'prepared/feature_hash_validation.npy')
    codes=calibration_partition(y,start,end,hashes)
    fit=np.flatnonzero(codes==0);evaluate=np.flatnonzero(codes==1)
    if len(fit)<100 or len(evaluate)<100:raise ValueError('Insufficient calibration/evaluation coverage')
    ids=np.load(base/'row_ids_validation.npy')
    partition_arrays={'partition_codes.npy':codes,'calibration_source_row_ids.npy':ids[fit],
                      'assessment_source_row_ids.npy':ids[evaluate]}
    for name,values in partition_arrays.items():
        if (out/name).exists():
            np.testing.assert_array_equal(np.load(out/name),values)
        else:
            np.save(out/name,values)
    if np.intersect1d(ids[fit],ids[evaluate]).size:raise ValueError('Calibration overlap')
    for label in np.intersect1d(y[fit],y[evaluate]):
        if end[fit[y[fit]==label]].max()>=start[evaluate[y[evaluate]==label]].min():raise ValueError('Temporal boundary overlap')
    protocol={'calibration_rows':len(fit),'assessment_rows':len(evaluate),'boundary_or_copy_exclusions':int(np.sum(codes<0)),
        'training_rows':len(np.load(base/'y_train.npy',mmap_mode='r')),'fit_classes':np.unique(y[fit]).tolist(),
        'assessment_classes':np.unique(y[evaluate]).tolist(),'candidate_manifest_sha256':digest(base/'manifest.json'),
        'method':'Class-conditional chronological halves of existing validation; strict touching-window purge; earlier ownership of identical original inputs.',
        'locked_test_loaded':False,'training_data_used_for_calibrator':False,
        'model_selection_warning':'Existing validation influenced prior exploratory experiments. This assessment is not an independent model-selection test.',
        'scope_change':'Separate isolated study; does not change the prior csv_research_v3 training-only OOF calibration policy.',
        'comparison_rule':'Before and after on exactly the same later assessment rows; full-30000 accuracy reported separately.'}
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('Pinned calibration protocol changed')
    else:
        write_json(out/'protocol.json',protocol)
    return fit,evaluate,y


def run(directory=DEFAULT):
    fit,evaluate,y=partition(directory)
    manifest=json.loads((directory/'04_port_refinement/manifest.json').read_text())
    root=directory/'calibration';results={}
    runs={family:directory/'runs'/f'{family}_all_records_s42_04_port_refinement' for family in ('xgboost','kan','hem','random_forest')}
    # Validation-based selection is explicit, prior to fitting the calibrator.
    # Full-feature master stays unchanged; predictors may use its compact subsets.
    choices=[directory/'runs'/f'xgboost_all_records_s42{suffix}' for suffix in ('','_04_port_refinement')]
    choices += [path for path in (directory/'runs').glob('xgboost_all_records_s42_04_port_refinement_d*_r*_cuda') if (path/'result.json').exists()]
    if all((path/'result.json').exists() for path in choices):
        runs['xgboost']=max(choices,key=lambda path:(
            json.loads((path/'result.json').read_text())['metrics']['accuracy'],
            json.loads((path/'result.json').read_text())['metrics']['macro_f1']))
    for family,path in runs.items():
        if not (path/'result.json').exists():
            raise FileNotFoundError(f'Incomplete model: {family}')
        out=root/family
        if out.exists():raise FileExistsError(out)
        out.mkdir()
        source=json.loads((path/'result.json').read_text())
        predictions=np.load(path/'validation_predictions.npy')
        if family!='hem':
            p=np.load(path/'validation_probabilities.npy')
            temperature=temperature_fit(p[fit],y[fit])
            calibrated=temperature_probabilities(p,temperature)
            np.testing.assert_array_equal(calibrated.argmax(1),predictions)
            np.save(out/'calibrated_validation_probabilities.npy',calibrated)
            persisted=np.load(out/'calibrated_validation_probabilities.npy')
            np.testing.assert_array_equal(persisted.argmax(1),predictions)
            np.testing.assert_allclose(persisted,calibrated,atol=0,rtol=0)
            calibration={'method':'scalar temperature scaling','temperature':temperature,'fit_rows':len(fit),
                         'argmax_unchanged':True,'probability_source':'complete classifier'}
            before=assessment(p[evaluate],y[evaluate],manifest);after=assessment(calibrated[evaluate],y[evaluate],manifest)
            after['confidence_note']='Temperature-calibrated probabilities, assessed on later purged validation rows; exploratory within-capture result.'
            write_json(out/'calibrator.json',calibration)
            # Verify the saved parameter reproduces the whole array.
            saved=json.loads((out/'calibrator.json').read_text())
            np.testing.assert_allclose(temperature_probabilities(p,saved['temperature']),calibrated,atol=1e-12)
        else:
            from sklearn.linear_model import LogisticRegression
            with np.load(path/'validation_components.npz') as components:
                multi=softmax(components['multiclass_logits'].astype(float),axis=1)
                binary=expit(components['binary_logits'].astype(float))
                knn=components['knn_predictions']
            mt=temperature_fit(multi[fit],y[fit])
            bp=np.column_stack([1-binary,binary]);binary_y=(y!=manifest['benign_index']).astype(int)
            bt=temperature_fit(bp[fit],binary_y[fit])
            mc=temperature_probabilities(multi,mt);bc=temperature_probabilities(bp,bt)
            chosen=mc[np.arange(len(y)),predictions]
            binary_matching=bc[np.arange(len(y)),(predictions!=manifest['benign_index']).astype(int)]
            features=np.column_stack([np.log(np.clip(chosen,1e-12,1)),
                np.log(np.clip(binary_matching,1e-12,1)),(knn==multi.argmax(1)).astype(float),
                ((binary>=.5)==(predictions!=manifest['benign_index'])).astype(float)])
            correct=predictions==y
            estimator=LogisticRegression(C=1.,random_state=42,max_iter=1000).fit(features[fit],correct[fit])
            confidence=estimator.predict_proba(features)[:,list(estimator.classes_).index(True)]
            joblib.dump(estimator,out/'correctness_calibrator.joblib')
            np.testing.assert_allclose(joblib.load(out/'correctness_calibrator.joblib').predict_proba(features)[:,1],confidence,atol=1e-12)
            np.save(out/'calibrated_correctness_confidence.npy',confidence)
            raw_confidence=.5*(multi[np.arange(len(y)),predictions]+np.where(predictions==manifest['benign_index'],1-binary,binary))
            before=confidence_metrics(correct[evaluate],raw_confidence[evaluate]);after=confidence_metrics(correct[evaluate],confidence[evaluate])
            calibration={'method':'Multiclass/binary component temperatures + regularized logistic correctness calibration',
                'multiclass_temperature':mt,'binary_temperature':bt,'fit_rows':len(fit),'hard_vote_unchanged':True,
                'confidence_meaning':'Estimated probability that the unchanged hard-vote decision is correct; not a multiclass ensemble probability.',
                'before_confidence':'Explicit diagnostic mean of voted-class multiclass probability and matching binary probability; not native HEM probability.',
                'features':['log calibrated voted-class probability','log calibrated matching binary probability','KNN/multiclass agreement','binary/vote agreement']}
            write_json(out/'calibrator.json',calibration)
        entry={'family':family,'calibration':calibration,'assessment_before':before,'assessment_after':after,
               'full_validation_accuracy':source['metrics']['accuracy'],'assessment_rows':len(evaluate),
               'calibration_rows':len(fit),'all_class_decisions_unchanged':True,'calibrator_reload_verified':True,
               'model_result_sha256':digest(path/'result.json'),'locked_test_loaded':False,
               'model_run':path.relative_to(directory).as_posix(),
               'research_status':'Exploratory validation calibration assessment, not independent test or external capture.'}
        write_json(out/'result.json',entry);results[family]=entry
        print(f'{family}: calibration complete, ECE before={before["ece_10_equal_width_bins"]:.5f}, after={after["ece_10_equal_width_bins"]:.5f}',flush=True)
    write_json(root/'summary.json',results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=DEFAULT)
    args=parser.parse_args();run(args.directory)
