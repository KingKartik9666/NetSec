"""Verify every exported feature cell and original target/ID by chunked reload."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiment.audit_datasense_raw import DEFAULT,SPLITS
from experiment.datasense_quality import digest,write_json


def run():
    folder=DEFAULT/'04_port_refinement'
    manifest=json.loads((folder/'manifest.json').read_text());names=manifest['feature_names'];results={}
    for split in SPLITS:
        path=folder/f'cleaned_{split}.csv.gz'
        expected=np.load(folder/f'X_full_{split}.npy',mmap_mode='r')
        labels=np.load(folder/f'y_{split}.npy');ids=np.load(folder/f'row_ids_{split}.npy')
        dtype={name:np.float32 for name in names};dtype.update(source_row_id=np.int64,label4=str)
        position=0
        for chunk in pd.read_csv(path,chunksize=4096,dtype=dtype,float_precision='round_trip'):
            end=position+len(chunk)
            np.testing.assert_array_equal(chunk['source_row_id'].to_numpy(),ids[position:end])
            np.testing.assert_array_equal(chunk['label4'].to_numpy(),np.asarray(manifest['classes'])[labels[position:end]])
            np.testing.assert_array_equal(chunk[names].to_numpy(),expected[position:end])
            position=end
        if position!=len(expected):raise ValueError('Exported rows changed')
        results[split]={'rows':position,'full_features':len(names),'all_feature_cells_equal':True,
                        'labels_ids_equal':True,'csv_sha256':digest(path),'csv_bytes':path.stat().st_size}
        print(f'{split}: every exported cell round-trips exactly ({position:,} rows)',flush=True)
    out=DEFAULT/'verification_04_port_refinement';out.mkdir(exist_ok=True)
    write_json(out/'export_verification.json',results)


if __name__=='__main__':run()
