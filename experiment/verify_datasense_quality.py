"""Independent whole-partition checks and readable audit tables for deep_clean_v2."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from experiment.audit_datasense_raw import DEFAULT,LISTS,PARENT,SPLITS,json_writer
from experiment.datasense_quality import digest,write_json


def meaning(name):
    if name.startswith('label'):
        return ('target/provenance','Never a predictor','Original annotation; never inferred or rewritten')
    if name in ('device_name','device_mac') or name.startswith('timestamp'):
        return ('identity/time provenance','Not a model measurement','Preserved for lineage; excluded from predictors; elapsed duration is separate')
    if name in LISTS:
        return ('observed categorical set','Empty list is observed absence; missing text is invalid','Sets may be capped; absence from a capped list is unknown')
    if name.startswith('log_data-ranges'):
        return ('log value summary','Zero can be measured or unsupported when log count is zero','Mixed log-value semantics; signed minima retained; physical units not independently recoverable')
    if name.endswith('_count'):
        return ('event or distinct-value count','Zero means no recorded events/items','Original reported count retained; list counts stored separately where available')
    if name.startswith('network_time-delta'):
        return ('packet timing summary','Coincident capture times or unsupported summary','Signed values retained; ordering and exact clock behavior cannot be reconstructed')
    if name.endswith('_std_deviation'):
        return ('variability summary','Zero can mean no variation or no supporting observations','Do not delete near-zero-variance columns; rare nonzero values retained')
    if 'interval' in name:
        return ('recorded interval','Zero ambiguous with no or insufficient observations','Exact extraction formula/unit was not published in the examined feature table; no rescaling')
    if name=='network_fragmentation-score':
        return ('fragmentation aggregate','May mean no fragmentation','Formula not independently verified; no guessed conversion')
    if name=='network_fragmented-packets':
        return ('fragmented packet count','No recorded fragmented packets','Keep count; high attack values retained')
    return ('network summary','Zero is context dependent','Keep measured zeros; mask zero summaries only when packet count is zero in the model view')


def run(directory=DEFAULT,candidate_name='03_candidate'):
    raw=directory/'01_raw_audit';candidate=directory/candidate_name
    scope=json.loads((raw/'scan_summary.json').read_text())
    manifest=json.loads((candidate/'manifest.json').read_text())
    mapping=json.loads((candidate/'redundant_columns.json').read_text())
    out=directory/('verification' if candidate_name=='03_candidate' else 'verification_'+candidate_name);out.mkdir(exist_ok=True)
    result={'test_loaded':False,'source_hash_verified':False,'input_pins_verified':True,'partitions':{}}
    for key,want in scope['input_hashes'].items():
        if digest(PARENT/'prepared'/f'{key}.npy')!=want:
            raise ValueError(f'Frozen input changed: {key}')
    for name,want in manifest['artifacts'].items():
        if digest(candidate/name)!=want:
            raise ValueError(f'Candidate bytes changed: {name}')
    profiles=list(csv.DictReader((directory/'02_profile/column_profile.csv').open(encoding='utf-8')))
    profile_lookup={(r['split'],r['column']):r for r in profiles}
    with (out/'field_dictionary.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=json_writer(stream,['field','role','zero_meaning','decision','train_zero_or_empty','validation_zero_or_empty','train_distinct','train_minimum','train_maximum'])
        for name in scope['raw_columns']:
            role,zero,decision=meaning(name)
            train=profile_lookup[('train',name)];val=profile_lookup[('validation',name)]
            writer.writerow(dict(field=name,role=role,zero_meaning=zero,decision=decision,
                train_zero_or_empty=train['zero_or_empty'],validation_zero_or_empty=val['zero_or_empty'],
                train_distinct=train['distinct'],train_minimum=train['minimum'],train_maximum=train['maximum']))
    classes=manifest['classes'];feature_names=manifest['feature_names']
    train_full=np.load(candidate/'X_full_train.npy',mmap_mode='r')
    val_full=np.load(candidate/'X_full_validation.npy',mmap_mode='r')
    with (out/'feature_dictionary.csv').open('w',encoding='utf-8',newline='') as stream:
        columns=['feature','representation','compact_model_retained','redundancy_reference','train_zero',
                 'train_unavailable','train_nonzero_observed','validation_unavailable','minimum_observed','maximum_observed']
        writer=json_writer(stream,columns)
        for i,name in enumerate(feature_names):
            vector=train_full[:,i];finite=np.isfinite(vector)
            reference=mapping['duplicates'].get(name,'')
            if name in mapping['constants']:reference='training constant: '+str(mapping['constants'][name])
            representation=('raw numeric with unsupported-summary mask' if name in scope['numeric_columns'] else
                'categorical membership, potentially censored' if '__present_' in name else
                'quality/provenance of measurement' if name.startswith('quality_') or name.endswith('__capped_list') else
                'documented aggregate-derived feature')
            writer.writerow({'feature':name,'representation':representation,'compact_model_retained':int(i in manifest['model_feature_indexes']),
                'redundancy_reference':reference,'train_zero':int(np.sum(vector==0)),
                'train_unavailable':int(np.sum(~finite)),'train_nonzero_observed':int(np.sum(finite&(vector!=0))),
                'validation_unavailable':int(np.sum(~np.isfinite(val_full[:,i]))),
                'minimum_observed':float(np.min(vector[finite])) if finite.any() else '',
                'maximum_observed':float(np.max(vector[finite])) if finite.any() else ''})
    class_rows=[];pair_rows=[]
    for split in SPLITS:
        y=np.load(candidate/f'y_{split}.npy');ids=np.load(candidate/f'row_ids_{split}.npy')
        original_y=np.load(PARENT/f'prepared/y_{split}.npy')
        original_ids=np.load(PARENT/f'prepared/row_ids_{split}.npy')
        np.testing.assert_array_equal(y,original_y)
        np.testing.assert_array_equal(ids,original_ids)
        full=np.load(candidate/f'X_full_{split}.npy',mmap_mode='r')
        compact=np.load(candidate/f'X_{split}.npy',mmap_mode='r')
        measured=np.load(raw/f'numeric_{split}.npy',mmap_mode='r')
        selected=np.load(candidate/f'observed_signal_mask_{split}.npy')
        durations=(np.load(candidate/f'window_end_{split}.npy')-np.load(candidate/f'window_start_{split}.npy'))/1e9
        values,counts=np.unique(durations,return_counts=True)
        checks={'rows':len(y),'labels_and_ids_unchanged':True,'model_features':compact.shape[1],
                'duration_counts':{str(float(v)):int(n) for v,n in zip(values,counts)},
                'all_rows_retained':len(y)==scope['rows'][split], 'negative_original_cells_retained':True,
                'dropped_column_reconstruction_exceptions':{}}
        for start in range(0,len(y),4096):
            sl=slice(start,start+4096)
            np.testing.assert_array_equal(compact[sl],full[sl,manifest['model_feature_indexes']])
            originals=measured[sl].astype(np.float32)
            current=full[sl,:len(scope['numeric_columns'])]
            negative=originals<0
            np.testing.assert_array_equal(originals[negative],current[negative])
            changed=~((originals==current)|(np.isnan(originals)&np.isnan(current)))
            bits=np.load(candidate/f'unsupported_summary_cells_{split}.npy',mmap_mode='r')[sl]
            np.testing.assert_array_equal(changed,np.unpackbits(bits,axis=1,count=originals.shape[1],bitorder='little').astype(bool))
            if np.isinf(full[sl]).any():
                raise ValueError('Infinite cleaned input')
        for name,other in mapping['duplicates'].items():
            a,b=(full[:,feature_names.index(n)] for n in (name,other))
            differences=int(np.sum(~((a==b)|(np.isnan(a)&np.isnan(b)))))
            pair_rows.append({'split':split,'removed':name,'retained':other,'different_rows':differences})
            if differences:
                checks['dropped_column_reconstruction_exceptions'][name]=differences
            if split=='train' and differences:
                raise ValueError('Redundant training columns were not equal')
        for i,name in enumerate(classes):
            mask=y==i
            class_rows.append({'split':split,'class':name,'rows_preserved':int(mask.sum()),
                              'observed_signal_rows':int(np.sum(mask&selected)),'no_signal_rows':int(np.sum(mask&~selected))})
        if split=='train':
            for arm in ('observed_signal','all_records'):
                first=np.load(candidate/f'{arm}_representative_indexes.npy')
                retained=np.load(candidate/f'{arm}_selected_indexes.npy')
                inverse=np.load(candidate/f'{arm}_inverse.npy')
                counts=np.load(candidate/f'{arm}_multiplicity.npy')
                np.testing.assert_array_equal(np.bincount(inverse,minlength=len(first)),counts)
                for start in range(0,len(retained),4096):
                    sl=slice(start,start+4096)
                    np.testing.assert_array_equal(compact[first[inverse[sl]]],compact[retained[sl]])
                    np.testing.assert_array_equal(y[first[inverse[sl]]],y[retained[sl]])
                checks[arm+'_reconstruction_verified']=True
        result['partitions'][split]=checks
    with (out/'class_retention.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=json_writer(stream,list(class_rows[0]));writer.writerows(class_rows)
    with (out/'duplicate_column_checks.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=json_writer(stream,list(pair_rows[0]));writer.writerows(pair_rows)
    result['source_hash_verified']=digest(PARENT/'source/combined_dataset.csv')==scope['source_sha256']
    if not result['source_hash_verified']:
        raise ValueError('Copied source bytes changed')
    result['candidate_files_verified']=len(manifest['artifacts'])
    result['candidate_manifest_sha256']=digest(candidate/'manifest.json')
    write_json(out/'data_verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='partitions'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=DEFAULT)
    parser.add_argument('--candidate',choices=['03_candidate','04_port_refinement'],default='03_candidate')
    args=parser.parse_args()
    run(args.directory,args.candidate)
