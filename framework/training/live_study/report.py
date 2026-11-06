"""Frozen assessment and inspectable CSV examples; no validation-driven refits."""
from __future__ import annotations

import csv
from pathlib import Path
import time
import numpy as np

from .protocol import ROOT, FEATURES, BRANCHES, HYBRIDS, atomic, read, relative, digest, verify_study
from .worker import predict, verify_result
from .calibration import measure, apply


def assess(study, output):
    study, output = Path(study), Path(output)
    settings, manifest = verify_study(study)
    selected = read(study/'selected.json')
    if set(selected['branches']) != set(BRANCHES) or set(selected.get('hybrids',{})) != set(HYBRIDS):
        raise ValueError('Final assessment requires frozen selections for all families')
    x=np.load(study/'data/X_validation.npy');y=np.load(study/'data/y_validation.npy')
    ids=np.load(study/'data/row_ids_validation.npy');novel=np.load(study/'data/novel_validation.npy')
    records={name:verify_result(ROOT/entry['result']) for name,entry in {**selected['branches'],**selected['hybrids']}.items()}
    results={};exports={}
    review=output/'validation_cases.csv';output.mkdir(parents=True,exist_ok=True)
    with review.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['model','source_row_id','csv_label','prediction','confidence','attack_probability','novel_vector',*FEATURES])
        for name,record in records.items():
            values=predict(record,x);config=record['calibration']
            np.savez_compressed(output/f'{name}_validation_outputs.npz',**values)
            confidence,attack,_=apply(values,config)
            for i in range(len(y)):
                writer.writerow([name,ids[i],manifest['classes'][y[i]],manifest['classes'][values['pred'][i]],
                    float(confidence[i]),float(attack[i]),bool(novel[i]),*map(int,x[i])])
            results[name]=dict(full=measure(values,y,manifest,config),
                novel_vectors=measure({k:v[novel] for k,v in values.items()},y[novel],manifest,config) if novel.any() else None,
                repeated_seeds={})
            for seed in settings['repeat_seeds']:
                repeat=verify_result(study/f'repeats/{name}/seed{seed}/result-link.json')
                replay=predict(repeat,x)
                results[name]['repeated_seeds'][str(seed)]=dict(full=measure(replay,y,manifest,repeat['calibration']),
                    novel_vectors=measure({k:v[novel] for k,v in replay.items()},y[novel],manifest,repeat['calibration']) if novel.any() else None)
            exports[name]=dict(family=name,checkpoint=record['checkpoint'],calibration=relative(ROOT/record['run_directory']/'calibration.json'),
                source_result=relative(ROOT/record['run_directory']/'result.json'),artifact_hashes=record['artifact_hashes'],
                auxiliary=name=='tabpack',backend='CPU Python checkpoint; native Rust export pending',
                deployment_status='research_candidate_not_promoted')
            print(f'ASSESSED {name} CSV accuracy={results[name]["full"]["accuracy"]:.5f}',flush=True)
    result=dict(status='completed',scope='within-capture original CSV validation; never live detection accuracy',
        selected_frozen_before_assessment=True,selection_digest=digest(study/'selected.json'),test_loaded=False,
        rows=len(y),novel_rows=int(novel.sum()),models=results,
        limitation=manifest['warning'],source_extractor_equivalence_verified=False,unlabelled_npcap_accuracy=None)
    atomic(output/'assessment.json',result)
    atomic(study/'exports/manifest.json',dict(feature_names=FEATURES,window_seconds=10,classes=manifest['classes'],
        benign_index=0,models=exports,preprocessing='per-checkpoint fit-only transform',
        zero_traffic_policy='no prediction',invalid_feature_policy='reject',test_loaded=False))
    atomic(output/'result.json',dict(status='completed',phase='assessment',finished_at=time.time(),test_loaded=False,
        artifact_hashes={relative(p):digest(p) for p in [output/'assessment.json',review,study/'exports/manifest.json']+
            list(output.glob('*_outputs.npz'))}))
