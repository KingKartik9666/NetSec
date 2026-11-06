"""Compare whitelisted original CSV rows and existing capture counts locally.

This is a provenance/distribution review, never automatic labelling of traffic.
No test split, payloads, live capture or network requests are used.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from framework.training.live_study.protocol import DEFAULT,FEATURES,atomic,read,digest,relative
from framework.training.live_study.data import valid_counts
from experiment.datasense_quality import selected_rows


def review(study):
    folder=study/'data';manifest=read(folder/'manifest.json')
    x=np.load(folder/'X_train.npy');y=np.load(folder/'y_train.npy');ids=np.load(folder/'row_ids_train.npy')
    roles=np.load(folder/'roles.npy');fit=np.flatnonzero(roles==0)
    source=ROOT/'dataset/combined_dataset.csv'
    classes=np.unique(y[fit]);chosen=classes[np.linspace(0,len(classes)-1,12,dtype=int)]
    examples=[fit[np.flatnonzero(y[fit]==label)[0]] for label in chosen]
    lookup={int(ids[i]):int(i) for i in examples};verified=[]
    for row_id,row in selected_rows(source,np.array(sorted(lookup))):
        i=lookup[row_id];raw=np.array([float(row[name]) for name in FEATURES],dtype=np.float32)
        np.testing.assert_array_equal(raw,x[i])
        assert row['label4']==manifest['classes'][y[i]]
        assert np.datetime64(row['timestamp_end'].removesuffix('Z'),'ns')-np.datetime64(row['timestamp_start'].removesuffix('Z'),'ns')==np.timedelta64(10,'s')
        verified.append(dict(source_row_id=row_id,label=row['label4'],counts=dict(zip(FEATURES,map(int,raw))),exact_match=True))
    # Existing evidence is explicitly historical. It is not a fresh capture.
    captures=[]
    live=ROOT/'artifacts/ui_polish_verification/npcap_live_proof.json'
    if live.exists():
        doc=read(live);prediction=doc['model_prediction']
        values={v['feature']:v['value'] for v in prediction['explanation']['all_features']}
        captures.append(dict(name='historical_Windows_Npcap',source=relative(live),source_sha256=digest(live),
            counts=values,observed_window_seconds=prediction['observed_window_seconds'],ground_truth=None,
            evidence_kind='previous real live Npcap observation; not captured by this review'))
    replay=ROOT/'artifacts/live_verification/pcap_replay_confirmed.json'
    if replay.exists():
        doc=read(replay);stats=doc['stats'];values=[stats[k] for k in ('packets','ack','fin','psh','rst','syn_all','urg')]
        captures.append(dict(name='historical_UQ_IoT_replay',source=relative(replay),source_sha256=digest(replay),
            counts=dict(zip(FEATURES,values)),observed_window_seconds=stats['seconds'],ground_truth=None,
            evidence_kind='previous benign-source PCAP header replay from a different dataset; no Windows live truth'))
    log=np.log1p(x[fit]);q25,q75=np.quantile(log,[.25,.75],axis=0);scale=np.maximum(q75-q25,.1)
    for case in captures:
        row=np.array([case['counts'][name] for name in FEATURES],np.float32)
        assert valid_counts(row[None])[0]
        distances=np.mean(np.abs((log-np.log1p(row))/scale),axis=1)
        closest=np.argsort(distances,kind='stable')[:5]
        case.update(count_contract_valid=True,exact_ten_second_window=case['observed_window_seconds']==10,
            outside_training_ranges=[FEATURES[i] for i in range(7) if row[i]<x[fit,i].min() or row[i]>x[fit,i].max()],
            nearest_fitting_rows=[dict(source_row_id=int(ids[fit[j]]),csv_label=manifest['classes'][y[fit[j]]],
                log_count_robust_l1_distance=float(distances[j]),counts=dict(zip(FEATURES,map(int,x[fit[j]])))) for j in closest],
            interpretation='Similarity is a feature check, not an attack label or detection-accuracy measurement')
    output=study/'reference_review';output.mkdir(parents=True,exist_ok=True)
    result=dict(csv_rows_checked=len(verified),csv_feature_and_label_matches=True,rows=verified,capture_comparisons=captures,
        held_out_test_loaded=False,fresh_capture_performed=False,model_accuracy_measured=False,
        limitation='CSV extractor device/window equivalence remains unverified. Historical capture labels are not ground truth.')
    atomic(output/'comparison.json',result)
    with (output/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['kind','row_or_case','label_or_evidence',*FEATURES])
        for row in verified:writer.writerow(['original CSV',row['source_row_id'],row['label'],*[row['counts'][k] for k in FEATURES]])
        for case in captures:
            writer.writerow(['historical capture',case['name'],case['evidence_kind'],*[case['counts'][k] for k in FEATURES]])
            for row in case['nearest_fitting_rows']:
                writer.writerow(['nearest CSV example',row['source_row_id'],row['csv_label'],*[row['counts'][k] for k in FEATURES]])
    print(f'Checked {len(verified)} original CSV rows, all seven counts and labels match; compared {len(captures)} historical header observations.')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,default=DEFAULT)
    review(parser.parse_args().study.resolve())
