"""Create reversible, semantically explicit DataSense model inputs after audit."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import shutil

import numpy as np

from experiment.audit_datasense_raw import DEFAULT, LISTS, PARENT, SPLITS, allocate, json_writer, kind, parsed
from experiment.datasense_quality import digest, exact_redundancy, lossless_groups, mask_unsupported_statistics, write_json


def build(directory=DEFAULT, parent=PARENT):
    raw = directory / '01_raw_audit'
    profile = directory / '02_profile'
    out = directory / '03_candidate'
    if out.exists():
        raise FileExistsError(f'Refusing overwrite: {out}')
    audit = json.loads((raw / 'scan_summary.json').read_text())
    old = json.loads((parent / 'prepared/manifest.json').read_text())
    prior = json.loads((parent / 'prepared/feature_transform.json').read_text())
    list_counts = json.loads((raw / 'training_list_vocabulary_counts.json').read_text())
    names = old['feature_names'] + prior['extra_feature_names']
    raw_names = audit['numeric_columns']
    lookup = {name:i for i,name in enumerate(names)}
    # Preserve rare protocol/log types; all thresholds/vocabularies use training only.
    extra_tokens = {}
    for field in ('log_data-types', 'network_protocols_all', 'network_protocols_src', 'network_protocols_dst'):
        candidates = sorted(list_counts[field])
        tokens = [t for t in candidates if f'{field}__present_{t}' not in lookup]
        if tokens:
            extra_tokens[field] = tokens
            names.extend(f'{field}__present_{t}' for t in tokens)
        if field == 'log_data-types':
            names.append(f'{field}__present_UNSEEN')
    names += ['quality_no_packets_observed','quality_no_logs_observed','quality_negative_packet_delta']
    for field in LISTS:
        names += [field+'__listed_distinct',field+'__reported_minus_listed',field+'__capped_list']
    names += [f'network_tcp-flags-{flag}__fraction_of_packets' for flag in ('ack','fin','psh','rst','syn','urg')]
    names += ['network_src_packet_fraction','network_dst_packet_fraction',
              'network_packet_size_coefficient_of_variation','network_payload_fraction',
              'network_packets_per_reported_ip','network_packets_per_reported_port']
    if len(names) != len(set(names)):
        raise ValueError('Duplicate generated feature names')
    out.mkdir()
    schema = {
        'version':2,'seed':42,'feature_names':names,'classes':old['classes'],'benign_index':old['benign_index'],
        'target_column':'label4','rows':audit['rows'],'extra_protocol_log_tokens':extra_tokens,
        'numeric_zero_masking':'Only zero summary statistics when supporting event count is zero; raw retained.',
        'censored_list_rule':'Listed distinct >=1000 and recorded count > listed distinct. Unlisted membership becomes NaN.',
        'negative_values':'Preserve; no clipping or invented packet ordering.',
        'redundancy':'Exact training columns only; complete matrix and source lineage retained.',
        'row_grouping':'Exact float32 feature vector plus original target; multiplicity and all source IDs retained.',
        'training_arms':['observed_signal','all_records'],
        'no_signal_rule':'All raw numeric measurements zero and every raw list empty; optional training exclusion only.',
        'test_loaded':False,'data_origin':'User-provided DataSense-format copy; publisher byte identity unverified.',
        'research_policy_frozen_before_fitting':True,
        'source_sha256':audit['source_sha256'],'prior_transform_sha256':digest(parent/'prepared/feature_transform.json'),
        'scan_summary_sha256':digest(raw/'scan_summary.json'),
    }
    write_json(out / 'manifest.json', schema)
    metrics = {}
    for split in SPLITS:
        original = np.load(parent / f'prepared/X_{split}.npy',mmap_mode='r')
        extra = np.load(parent / f'prepared/extra_{split}.npy',mmap_mode='r')
        measured = np.load(raw / f'numeric_{split}.npy',mmap_mode='r')
        lengths = np.load(raw / f'list_lengths_{split}.npy',mmap_mode='r')
        row_ids = np.load(parent / f'prepared/row_ids_{split}.npy',mmap_mode='r')
        X = allocate(out,f'X_full_{split}','float32',(len(original),len(names)))
        changes = allocate(out,f'unsupported_summary_cells_{split}','uint8',(len(X),(len(raw_names)+7)//8))
        counts = Counter()
        idx = {name:i for i,name in enumerate(names)}
        for start in range(0,len(X),4096):
            end = min(start+4096,len(X)); sl=slice(start,end)
            block = np.zeros((end-start,len(names)),dtype=np.float32)
            block[:,:original.shape[1]] = original[sl]
            block[:,original.shape[1]:original.shape[1]+extra.shape[1]] = extra[sl]
            # Verify that we are transforming the selected raw measurements.
            np.testing.assert_allclose(measured[sl].astype(np.float32),original[sl,:len(raw_names)],rtol=1e-6,atol=1e-7,equal_nan=True)
            clean, changed = mask_unsupported_statistics(measured[sl],raw_names)
            block[:,:len(raw_names)] = clean
            changes[sl] = np.packbits(changed,axis=1,bitorder='little')
            counts['unsupported_summary_cells'] += int(changed.sum())
            count_packets = measured[sl,raw_names.index('network_packets_all_count')]
            count_logs = measured[sl,raw_names.index('log_messages_count')]
            block[:,idx['quality_no_packets_observed']] = count_packets==0
            block[:,idx['quality_no_logs_observed']] = count_logs==0
            block[:,idx['quality_negative_packet_delta']] = measured[sl,raw_names.index('network_time-delta_min')]<0
            for j,field in enumerate(LISTS):
                reported = measured[sl,raw_names.index(field+'_count')]
                listed = lengths[sl,j]
                capped = (listed>=1000)&(reported>listed)
                block[:,idx[field+'__listed_distinct']] = np.where(listed>=0,listed,np.nan)
                block[:,idx[field+'__reported_minus_listed']] = np.where(listed>=0,reported-listed,np.nan)
                block[:,idx[field+'__capped_list']] = capped
                counts[field+':capped_rows'] += int(capped.sum())
                columns = [i for i,name in enumerate(names) if name.startswith(field+'__present_')]
                for col in columns:
                    absent = capped & (block[:,col]==0)
                    block[absent,col] = np.nan
                    counts['censored_membership_cells'] += int(absent.sum())
            def ratio(numerator,denominator):
                result=np.full(len(numerator),np.nan,dtype=np.float32)
                np.divide(numerator,denominator,out=result,where=np.isfinite(denominator)&(denominator>0))
                return result
            for flag in ('ack','fin','psh','rst','syn','urg'):
                block[:,idx[f'network_tcp-flags-{flag}__fraction_of_packets']] = ratio(block[:,idx[f'network_tcp-flags-{flag}_count']],count_packets)
            for direction in ('src','dst'):
                block[:,idx[f'network_{direction}_packet_fraction']] = ratio(block[:,idx[f'network_packets_{direction}_count']],count_packets)
            block[:,idx['network_packet_size_coefficient_of_variation']] = ratio(block[:,idx['network_packet-size_std_deviation']],block[:,idx['network_packet-size_avg']])
            block[:,idx['network_payload_fraction']] = ratio(block[:,idx['network_payload-length_avg']],block[:,idx['network_packet-size_avg']])
            block[:,idx['network_packets_per_reported_ip']] = ratio(count_packets,block[:,idx['network_ips_all_count']])
            block[:,idx['network_packets_per_reported_port']] = ratio(count_packets,block[:,idx['network_ports_all_count']])
            if np.isinf(block).any():
                raise ValueError('Derived feature overflow; explicit resolution required')
            X[sl] = block
        changes.flush()
        # Small categorical fields only; large ports and identities stay in raw evidence.
        with gzip.open(raw/f'raw_selected_{split}.csv.gz','rt',encoding='utf-8',newline='') as stream:
            for row_index,row in enumerate(csv.DictReader(stream)):
                if int(row['source_row_id']) != row_ids[row_index]:
                    raise ValueError('Extracted CSV row order changed')
                for field,tokens in extra_tokens.items():
                    observed=set(parsed(row[field],kind(field))[0])
                    for token in tokens:
                        X[row_index,idx[field+'__present_'+token]] = float(token in observed)
                    if field == 'log_data-types':
                        X[row_index,idx[field+'__present_UNSEEN']] = float(bool(observed.difference(tokens)))
        X.flush()
        for prefix in ('y','row_ids','window_start','window_end'):
            shutil.copy2(parent/f'prepared/{prefix}_{split}.npy',out/f'{prefix}_{split}.npy')
        no_signal=np.load(profile/f'no_observed_signal_{split}.npy')
        np.save(out/f'observed_signal_mask_{split}.npy',~no_signal)
        metrics[split]={'rows':len(X),'features':len(names),'no_signal_rows_preserved':int(no_signal.sum()),'changes':dict(counts)}
        print(f'Candidate: {split}, all {len(X):,} records retained, {len(names)} features',flush=True)
    train=np.load(out/'X_full_train.npy',mmap_mode='r')
    # Future unseen membership remains expressible, even when training-constant.
    mapping=exact_redundancy(train,names,protected=[n for n in names if n.endswith('__present_UNSEEN')])
    write_json(out/'redundant_columns.json',mapping)
    schema['model_feature_names']=[names[i] for i in mapping['retained']]
    schema['model_feature_indexes']=mapping['retained']
    for split in SPLITS:
        full=np.load(out/f'X_full_{split}.npy',mmap_mode='r')
        compact=allocate(out,f'X_{split}','float32',(len(full),len(mapping['retained'])))
        for start in range(0,len(full),4096):
            compact[start:start+4096]=full[start:start+4096,mapping['retained']]
        compact.flush()
        metrics[split]['model_features']=compact.shape[1]
        if split=='train':
            y=np.load(out/'y_train.npy',mmap_mode='r')
            for arm in ('all_records','observed_signal'):
                selected=np.arange(len(full)) if arm=='all_records' else np.flatnonzero(np.load(out/'observed_signal_mask_train.npy'))
                selected_X=np.asarray(compact[selected])
                first,inverse,counts=lossless_groups(selected_X,y[selected])
                np.save(out/f'{arm}_representative_indexes.npy',selected[first])
                np.save(out/f'{arm}_selected_indexes.npy',selected)
                np.save(out/f'{arm}_inverse.npy',inverse)
                np.save(out/f'{arm}_multiplicity.npy',counts)
                metrics['train'][arm]={'original_rows':len(selected),'distinct_input_label_pairs':len(first),
                    'redundant_repetitions':len(selected)-len(first),'exact_inverse_verified':True,'multiplicity_sum':int(counts.sum())}
                del selected_X
                print(f'{arm}: {len(selected):,} rows -> {len(first):,} exact input/label groups; counts preserved',flush=True)
    schema['artifacts']={p.name:digest(p) for p in out.glob('*.npy')}
    write_json(out/'manifest.json',schema)
    write_json(out/'preparation_summary.json',metrics)
    print('Reversible candidate complete.',flush=True)


def export(directory=DEFAULT,candidate='03_candidate'):
    out=directory/candidate
    manifest=json.loads((out/'manifest.json').read_text())
    names=manifest['feature_names']
    for split in SPLITS:
        path=out/f'cleaned_{split}.csv.gz'
        if path.exists():
            raise FileExistsError(path)
        X=np.load(out/f'X_full_{split}.npy',mmap_mode='r')
        y=np.load(out/f'y_{split}.npy',mmap_mode='r')
        ids=np.load(out/f'row_ids_{split}.npy',mmap_mode='r')
        with gzip.open(path,'wt',encoding='utf-8',newline='',compresslevel=1) as stream:
            writer=csv.writer(stream,lineterminator='\n')
            writer.writerow(['source_row_id','label4']+names)
            for start in range(0,len(X),4096):
                for i,row in enumerate(X[start:start+4096],start):
                    writer.writerow([int(ids[i]),manifest['classes'][int(y[i])]] + ['' if np.isnan(v) else format(float(v),'.9g') for v in row])
        print(f'Exported {path.name}: {len(X):,} rows',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['build','export'])
    parser.add_argument('--directory',type=Path,default=DEFAULT)
    parser.add_argument('--candidate',choices=['03_candidate','04_port_refinement'],default='03_candidate')
    args=parser.parse_args()
    if args.command=='build':
        build(args.directory)
    else:
        export(args.directory,args.candidate)
