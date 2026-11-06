"""Read only original training arrays; preserve earlier purged role ownership."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from framework.training.live_study.protocol import ROOT,atomic,read,digest,relative,fingerprint,verify
from .schema import core_names,extended_names,derive,WITHHELD,PAYLOAD

BASE=ROOT/'dataset/prepared/datasense_full_train'
PREVIOUS=ROOT/'artifacts/training/live_models_20261025_v1/data'
DEFAULT=ROOT/'artifacts/training/live_features_20261027_v1'


def prepare(output=DEFAULT,extended=False):
    output=Path(output);folder=output/'data';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'manifest.json').exists():
        m=read(folder/'manifest.json');verify(m['array_hashes']);verify(m['input_hashes'])
        if m['extended']!=extended:raise ValueError('Existing profile differs; use a new output revision')
        return m
    source=read(BASE/'manifest.json')
    old=read(PREVIOUS/'manifest.json');verify(old['array_hashes'])
    ids=np.load(BASE/'row_ids_train.npy');admitted=np.load(PREVIOUS/'row_ids_train.npy')
    # Mapping through explicit IDs avoids assuming sorted source row order.
    lookup={int(value):i for i,value in enumerate(ids)}
    positions=np.array([lookup[int(value)] for value in admitted])
    raw=np.load(BASE/'X_train.npy',mmap_mode='r')[positions].astype(np.float32)
    y=np.load(BASE/'y_train.npy')[positions];roles=np.load(PREVIOUS/'roles.npy')
    np.testing.assert_array_equal(y,np.load(PREVIOUS/'y_train.npy'))
    names=extended_names() if extended else core_names()
    start=np.load(BASE/'window_start_train.npy')[positions];end=np.load(BASE/'window_end_train.npy')[positions]
    seconds=(end-start)/1e9
    if np.any(seconds!=10):raise ValueError('Profile needs exact ten-second windows')
    index={name:i for i,name in enumerate(source['feature_names'])};derived=derive(raw,source['feature_names'],seconds)
    x=np.column_stack([raw[:,index[name]] if name in index else derived[name] for name in names]).astype(np.float32)
    if np.isinf(x).any():raise ValueError('Infinite numeric features require explicit handling')
    for name,array in dict(X_train=x,y_train=y,roles=roles,row_ids_train=admitted,
        window_start_train=start,window_end_train=end).items():np.save(folder/f'{name}.npy',array)
    inputs=[BASE/'manifest.json',*[BASE/f'{k}_train.npy' for k in ('X','y','row_ids','window_start','window_end')],
        PREVIOUS/'manifest.json',PREVIOUS/'row_ids_train.npy',PREVIOUS/'y_train.npy',PREVIOUS/'roles.npy']
    m=dict(feature_names=names,classes=source['classes'],benign_index=0,extended=extended,
        profile='broad_header_core_v1' if not extended else 'broad_header_extended_research_v1',
        feature_schema_fingerprint=fingerprint(names),candidate_features=len(names),rows=len(y),fitting_rows=int((roles==0).sum()),
        role_provenance='Preserved previous purged seven-count role mapping for conservative comparable feature screening',
        publisher_equivalence_verified=False,raw_fields_withheld=WITHHELD,
        additional_core_exclusions=PAYLOAD if not extended else {},
        zero_and_missing_policy='Source zeros preserved; missing stays NaN; native absent measurement stays null. Paired exporter parity remains unverified.',
        interval_policy='Capture-order signed deltas preserved; live timestamp precision is a compatibility limitation',
        raw_identifiers_in_model=False,validation_loaded=False,test_loaded=False,
        input_hashes={relative(p):digest(p) for p in inputs},array_hashes={relative(p):digest(p) for p in folder.glob('*.npy')})
    atomic(folder/'manifest.json',m)
    return m
