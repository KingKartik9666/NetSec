"""Complete saved 160-round XGBoost verification after a one-ULP parity failure.

No refitting or checkpoint modification: compare this same checkpoint on both
devices, require identical decisions, and retain both probability arrays.
"""
from __future__ import annotations

import json
import time

import numpy as np
import xgboost as xgb

from experiment.audit_datasense_raw import DEFAULT
from experiment.datasense_quality import digest,verify_probability_parity,write_json
from experiment.train_datasense_quality import probability_metrics


def run():
    out=DEFAULT/'runs/xgboost_all_records_s42_04_port_refinement_d8_r160_cuda'
    if (out/'result.json').exists():raise FileExistsError('Completed result exists')
    request=json.loads((out/'request.json').read_text());progress=json.loads((out/'progress.json').read_text())
    prepared=DEFAULT/request['candidate'];manifest=json.loads((prepared/'manifest.json').read_text())
    if digest(prepared/'manifest.json')!=request['candidate_manifest_sha256']:raise ValueError('Candidate changed')
    if progress['round']!=request['rounds']:raise ValueError('Fit did not complete all rounds')
    checkpoint=out/'model.ubj';before=digest(checkpoint);started=time.monotonic()
    model=xgb.Booster(model_file=str(checkpoint))
    if model.num_boosted_rounds()!=160:raise ValueError('Unexpected saved round count')
    data=xgb.DMatrix(np.load(prepared/'X_validation.npy',mmap_mode='r'),nthread=2)
    truth=np.load(prepared/'y_validation.npy')
    model.set_param(request['parameters']);model.set_param({'device':'cuda','nthread':2})
    gpu=model.predict(data)
    config=json.loads(model.save_config());write_json(out/'model_config.json',config)
    if not config['learner']['generic_param']['device'].startswith('cuda'):raise ValueError('GPU inference request failed')
    # Separate Booster prevents reuse of the GPU model's prediction cache.
    cpu_model=xgb.Booster(model_file=str(checkpoint))
    cpu_model.set_param({'device':'cpu','nthread':2});cpu=cpu_model.predict(data)
    verify_probability_parity(gpu,cpu,atol=2e-7)
    if float(np.mean(gpu.argmax(1)==truth))!=progress['accuracy']:raise ValueError('Saved-round accuracy differs')
    if digest(checkpoint)!=before:raise ValueError('Recovery altered checkpoint')
    metrics=probability_metrics(truth,cpu,manifest['classes'],manifest['benign_index'])
    np.save(out/'validation_gpu_probabilities.npy',gpu)
    np.save(out/'validation_probabilities.npy',cpu)
    np.save(out/'validation_predictions.npy',cpu.argmax(1))
    evidence={'reason':'Initial 1e-7 CPU/GPU probability threshold failed by float32 rounding; saved checkpoint recovered without fitting.',
              'all_class_decisions_identical':True,'probability_absolute_tolerance':2e-7,
              'maximum_absolute_difference':float(np.max(np.abs(gpu-cpu))),
              'original_checkpoint_sha256':before,'checkpoint_unchanged':True,
              'original_round_count_verified':160,'original_final_accuracy_verified':True}
    write_json(out/'recovery_verification.json',evidence)
    result={**request,'status':'completed','metrics':metrics,'checkpoint_sha256':before,
            'checkpoint_reload_verified':True,'reload_rows':len(truth),'reload_device':'cpu',
            'elapsed_seconds':progress['elapsed_seconds']+time.monotonic()-started,
            'feature_names':manifest['model_feature_names'],'recovery':evidence,
            'research_status':'Exploratory within-capture validation; combined capacity/representation recipe; no locked test or live promotion.'}
    write_json(out/'result.json',result)
    print(json.dumps({'accuracy':metrics['accuracy'],'macro_f1':metrics['macro_f1'],**evidence}))


if __name__=='__main__':run()
