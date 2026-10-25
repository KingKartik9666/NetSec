"""Train-only port-information refinement, preserving the audited master."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil

import numpy as np

from experiment.audit_datasense_raw import DEFAULT,SPLITS,allocate,kind,parsed
from experiment.datasense_quality import binary_information,censored_presence,digest,exact_redundancy,lossless_groups,write_json


FIELDS=('network_ports_src','network_ports_dst')


def run(directory=DEFAULT):
    base=directory/'03_candidate';raw=directory/'01_raw_audit'
    out=directory/'04_port_refinement'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    manifest=json.loads((base/'manifest.json').read_text())
    names=list(manifest['feature_names'])
    y=np.load(base/'y_train.npy')
    signal=np.load(base/'observed_signal_mask_train.npy')
    ids=np.load(base/'row_ids_train.npy')
    classes=np.bincount(y[signal],minlength=len(manifest['classes']))
    counts={field:np.zeros((len(classes),65536),dtype=np.int32) for field in FIELDS}
    write_json(out/'selection_policy.json',{
        'frozen_before_this_candidate_evaluation':True,'fit_partition':'training observed-signal rows only',
        'statistic':'Mutual information of membership in exported list and original target; natural logarithm.',
        'new_tokens_per_direction':64,'minimum_exported_occurrences':10,
        'tie_break':'Ascending integer port','preserve_existing_vocabulary':True,
        'censoring_caveat':'Ranking observes exported occurrence, not true absence. Model values still mark unlisted capped membership unknown.',
        'validation_used_for_vocabulary':False,'test_loaded':False,
        'base_manifest_sha256':digest(base/'manifest.json')})
    with gzip.open(raw/'raw_selected_train.csv.gz','rt',encoding='utf-8',newline='') as stream:
        for i,row in enumerate(csv.DictReader(stream)):
            if int(row['source_row_id'])!=ids[i]:
                raise ValueError('Training row order changed')
            if signal[i]:
                for field in FIELDS:
                    tokens=parsed(row[field],kind(field))[0]
                    if tokens:
                        counts[field][y[i],np.fromiter(tokens,dtype=np.int32)]+=1
            if (i+1)%100000==0:
                print(f'Port vocabulary: {i+1:,}/{len(y):,} training rows',flush=True)
    if i+1!=len(y):
        raise ValueError('Training extraction row count changed')
    vocabulary={}
    with (out/'port_information.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['field','port','exported_occurrences','mutual_information_nats','newly_selected'])
        for field in FIELDS:
            information=binary_information(counts[field],classes)
            frequency=counts[field].sum(axis=0)
            order=np.lexsort((np.arange(65536),-information))
            eligible=[int(port) for port in order if frequency[port]>=10 and f'{field}__present_{port}' not in names]
            vocabulary[field]=eligible[:64]
            names.extend(f'{field}__present_{port}' for port in vocabulary[field])
            selected=set(vocabulary[field])
            writer.writerows((field,p,int(frequency[p]),float(information[p]),int(p in selected)) for p in range(65536))
    write_json(out/'added_port_vocabulary.json',vocabulary)
    del counts
    print('Port vocabulary frozen from training; building all retained rows.',flush=True)
    summary={}
    for split in SPLITS:
        old=np.load(base/f'X_full_{split}.npy',mmap_mode='r')
        ids=np.load(base/f'row_ids_{split}.npy',mmap_mode='r')
        full=allocate(out,f'X_full_{split}','float32',(len(old),len(names)))
        for start in range(0,len(old),4096):
            full[start:start+4096,:old.shape[1]]=old[start:start+4096]
        with gzip.open(raw/f'raw_selected_{split}.csv.gz','rt',encoding='utf-8',newline='') as stream:
            for i,row in enumerate(csv.DictReader(stream)):
                if int(row['source_row_id'])!=ids[i]:
                    raise ValueError('Feature row order changed')
                offset=old.shape[1]
                for field,tokens in vocabulary.items():
                    observed=parsed(row[field],kind(field))[0]
                    full[i,offset:offset+len(tokens)]=censored_presence(observed,tokens,float(row[field+'_count']))[:-1]
                    offset+=len(tokens)
        if i+1!=len(old):
            raise ValueError('Feature extraction row count changed')
        full.flush()
        for prefix in ('y','row_ids','window_start','window_end','observed_signal_mask','unsupported_summary_cells'):
            shutil.copy2(base/f'{prefix}_{split}.npy',out/f'{prefix}_{split}.npy')
        summary[split]={'rows':len(full),'full_features':len(names),'all_rows_retained':True}
        print(f'Port refinement {split}: {len(full):,} rows, {len(names)} complete features',flush=True)
    train=np.load(out/'X_full_train.npy',mmap_mode='r')
    for field,tokens in vocabulary.items():
        for port in tokens:
            column=names.index(f'{field}__present_{port}')
            if np.sum(train[:,column]==1)<10:
                raise ValueError('Selected port membership failed count reconstruction')
    mapping=exact_redundancy(train,names,protected=[n for n in names if n.endswith('__present_UNSEEN')])
    write_json(out/'redundant_columns.json',mapping)
    manifest.update(version=3,feature_names=names,model_feature_indexes=mapping['retained'],
                    model_feature_names=[names[i] for i in mapping['retained']],added_port_vocabulary=vocabulary,
                    port_selection_policy_sha256=digest(out/'selection_policy.json'),
                    base_manifest_sha256=digest(base/'manifest.json'))
    for split in SPLITS:
        full=np.load(out/f'X_full_{split}.npy',mmap_mode='r')
        compact=allocate(out,f'X_{split}','float32',(len(full),len(mapping['retained'])))
        for start in range(0,len(full),4096):
            compact[start:start+4096]=full[start:start+4096,mapping['retained']]
        compact.flush()
        summary[split]['model_features']=compact.shape[1]
        if split=='train':
            for arm in ('all_records','observed_signal'):
                selected=np.load(base/f'{arm}_selected_indexes.npy')
                values=np.asarray(compact[selected])
                first,inverse,multiplicity=lossless_groups(values,y[selected])
                np.save(out/f'{arm}_selected_indexes.npy',selected)
                np.save(out/f'{arm}_representative_indexes.npy',selected[first])
                np.save(out/f'{arm}_inverse.npy',inverse)
                np.save(out/f'{arm}_multiplicity.npy',multiplicity)
                summary[split][arm]={'rows':len(selected),'exact_groups':len(first),
                                     'multiplicity_sum':int(multiplicity.sum()),'reconstruction_verified':True}
                del values
    manifest['artifacts']={p.name:digest(p) for p in out.glob('*.npy')}
    write_json(out/'manifest.json',manifest)
    write_json(out/'preparation_summary.json',summary)
    print('Port refinement complete.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=DEFAULT)
    run(parser.parse_args().directory)
