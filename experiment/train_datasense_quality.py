"""Fixed validation comparisons of the audited DataSense candidate; CPU only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiment.audit_datasense_raw import DEFAULT,PARENT
from experiment.datasense_quality import digest,verify_probability_parity,weighted_smoothed_derivatives,write_json
from framework.training.tracking import classification_metrics,validate_manifest


def probability_metrics(truth,probabilities,classes,benign):
    predictions=probabilities.argmax(1)
    result=classification_metrics(truth,predictions,classes,benign)
    result['top_three_accuracy']=float(np.mean((np.argsort(-probabilities,axis=1)[:,:3]==truth[:,None]).any(1)))
    confidence=probabilities.max(1)
    result['mean_confidence']=float(confidence.mean())
    result['multiclass_log_loss']=float(-np.log(np.maximum(probabilities[np.arange(len(truth)),truth],1e-12)).mean())
    correct=predictions==truth
    result['mean_confidence_correct']=float(confidence[correct].mean()) if correct.any() else None
    result['mean_confidence_wrong']=float(confidence[~correct].mean()) if (~correct).any() else None
    ece=0.
    for low in np.linspace(0,.9,10):
        mask=(confidence>=low)&(confidence<low+.1 if low<.9 else confidence<=1)
        if mask.any():
            ece += float(mask.mean()*abs(correct[mask].mean()-confidence[mask].mean()))
    result['ece_10_equal_width_bins']=ece
    result['confidence_note']='Uncalibrated model probabilities; confidence is not accuracy.'
    return result


def fit(directory=DEFAULT,arm='observed_signal',seed=42,threads=2,family='xgboost',rounds=79,candidate='03_candidate',device='cpu',depth=None):
    prepared=directory/candidate
    manifest=json.loads((prepared/'manifest.json').read_text())
    if 'model_feature_names' not in manifest:
        raise ValueError('Preparation has not completed')
    validate_manifest(dict(feature_names=manifest['model_feature_names'],classes=manifest['classes'],benign_index=manifest['benign_index']))
    suffix='' if candidate=='03_candidate' else '_'+candidate
    if depth is not None or device!='cpu' or rounds!=79:
        suffix+=f'_d{depth or 6}_r{rounds}_{device}'
    output=directory/'runs'/f'{family}_{arm}_s{seed}{suffix}'
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    truth=np.load(prepared/'y_validation.npy')
    all_y=np.load(prepared/'y_train.npy',mmap_mode='r')
    feature_names=manifest['model_feature_names']
    all_x=np.load(prepared/'X_train.npy',mmap_mode='r')
    Xval=np.load(prepared/'X_validation.npy',mmap_mode='r')
    if arm=='original_control':
        if family!='random_forest':
            raise ValueError('Original XGBoost controls already exist in the parent experiment')
        feature_names=json.loads((directory/'01_raw_audit/scope.json').read_text())['numeric_columns']
        all_x=np.load(PARENT/'prepared/X_train.npy',mmap_mode='r')[:,:len(feature_names)]
        Xval=np.load(PARENT/'prepared/X_validation.npy',mmap_mode='r')[:,:len(feature_names)]
        selected=np.arange(len(all_y))
        weights=np.ones(len(selected),dtype=np.float32)
    elif family=='xgboost':
        selected=np.load(prepared/f'{arm}_representative_indexes.npy')
        weights=np.load(prepared/f'{arm}_multiplicity.npy').astype(np.float32)
    else:
        # Preserve fixed sklearn RF bootstrap: do not change sampling units by grouping.
        selected=np.load(prepared/f'{arm}_selected_indexes.npy')
        weights=np.ones(len(selected),dtype=np.float32)
    Xtrain=np.array(all_x[selected],dtype=np.float32)
    y=np.asarray(all_y[selected])
    if len(np.unique(y))!=len(manifest['classes']):
        raise ValueError('Training arm lost a class')
    if not np.array_equal(truth,np.load(PARENT/'prepared/y_validation.npy')):
        raise ValueError('Validation labels changed')
    request={'family':family,'arm':arm,'seed':seed,'candidate':candidate,'rounds':rounds if family=='xgboost' else None,
             'device':device,'training_groups':len(y),'represented_training_observations':int(weights.sum()),
             'validation_rows':len(truth),'feature_count':Xtrain.shape[1],
             'test_loaded':False,'candidate_manifest_sha256':digest(prepared/'manifest.json'),
             'evaluation':'Fixed rounds; no validation early stopping; all original validation rows; source-validation only.',
             'grouping_note':'Multiplicity preserves full empirical loss. Group-wise subsampling/quantiles may differ from expanded-row fitting.'}
    write_json(output/'request.json',request)
    started=time.monotonic()
    if family=='xgboost':
        import xgboost as xgb
        params=json.loads((PARENT/'runs/cleaned_enriched/request.json').read_text())['parameters']
        params.update(seed=seed,nthread=threads,device=device)
        if depth is not None:params['max_depth']=depth
        if device=='cuda':
            import torch
            if not torch.cuda.is_available():raise RuntimeError('Explicit CUDA unavailable')
        request['parameters']=params
        request['smoothing']=.05
        write_json(output/'request.json',request)
        training=xgb.QuantileDMatrix(Xtrain,label=y,weight=weights,max_bin=params['max_bin'],nthread=threads)
        validation=xgb.DMatrix(Xval,nthread=threads)
        del Xtrain,all_x
        class Progress(xgb.callback.TrainingCallback):
            def after_iteration(self,model,epoch,evals_log):
                if epoch%10==0 or epoch+1==rounds:
                    prediction=model.predict(validation).argmax(1)
                    entry={'round':epoch+1,'rounds':rounds,'elapsed_seconds':time.monotonic()-started,
                           'accuracy':float(np.mean(truth==prediction))}
                    write_json(output/'progress.json',entry)
                    print(f'{family}/{arm} round {epoch+1}/{rounds} accuracy={entry["accuracy"]:.6f} elapsed={entry["elapsed_seconds"]:.1f}s',flush=True)
                return False
        def objective(margins,data):
            return weighted_smoothed_derivatives(margins,y,weights)
        model=xgb.train(params,training,num_boost_round=rounds,obj=objective,callbacks=[Progress()])
        checkpoint=output/'model.ubj'
        model.save_model(checkpoint)
        probability=model.predict(validation)
        reloaded=xgb.Booster(model_file=str(checkpoint))
        reloaded.set_param({'device':'cpu','nthread':threads})
        verify_probability_parity(probability,reloaded.predict(validation),atol=2e-7 if device=='cuda' else 1e-7)
        write_json(output/'model_config.json',json.loads(model.save_config()))
    else:
        if device!='cpu' or depth is not None:
            raise ValueError('Fixed RF parameters and CPU execution cannot change')
        if seed!=42:
            raise ValueError('Fixed conventional RF is seed 42 only')
        from framework.training.fixed_random_forest import train_fixed_random_forest,FIXED_PARAMETERS
        import joblib
        request['parameters']=FIXED_PARAMETERS
        request['grouping_note']='RF receives expanded rows to preserve its frozen bootstrap sampling units.'
        write_json(output/'request.json',request)
        details=train_fixed_random_forest(Xtrain,y,Xval,truth,len(manifest['classes']),{},42,output)
        probability=details['probabilities']
        checkpoint=output/'model.joblib'
        reloaded=joblib.load(checkpoint)
        np.testing.assert_allclose(reloaded.predict_proba(Xval),probability,atol=1e-12,rtol=0)
    metrics=probability_metrics(truth,probability,manifest['classes'],manifest['benign_index'])
    np.save(output/'validation_probabilities.npy',probability)
    np.save(output/'validation_predictions.npy',probability.argmax(1))
    result={'status':'completed',**request,'elapsed_seconds':time.monotonic()-started,
            'metrics':metrics,'checkpoint_sha256':digest(checkpoint),'checkpoint_reload_verified':True,
            'reload_rows':len(truth),'feature_names':feature_names,
            'research_status':'Exploratory within-capture validation. No locked test, live validation or model promotion.'}
    write_json(output/'result.json',result)
    print(json.dumps({k:v for k,v in metrics.items() if k not in ('confusion_matrix','per_class')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=DEFAULT)
    parser.add_argument('--arm',choices=['observed_signal','all_records','original_control'],default='observed_signal')
    parser.add_argument('--family',choices=['xgboost','random_forest'],default='xgboost')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--rounds',type=int,default=79)
    parser.add_argument('--candidate',choices=['03_candidate','04_port_refinement'],default='03_candidate')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--depth',type=int)
    args=parser.parse_args()
    fit(args.directory,args.arm,args.seed,args.threads,args.family,args.rounds,args.candidate,args.device,args.depth)
