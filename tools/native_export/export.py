"""Export completed research checkpoints; this tool is never an app dependency.

Only the pinned train and validation arrays are accepted. No fitting, downloading,
held-out test access, or modification of sealed research artifacts occurs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'artifacts/training/all_models_wrapup_20261023_v1'
DATA = ROOT / 'dataset/experiments/datasense_verified_clean_20261014_v2/model'
OUTPUT = ROOT / 'artifacts/application/native_models_v1'
LABELS = {'random_forest': 'Random Forest', 'xgboost': 'XGBoost', 'kan': 'KAN',
          'hem': 'HEM-derived', 'tabm_ne': 'TabM-NE', 'tabm_hier': 'TabM-Hier',
          'tabm_hier_ne': 'TabM-Hier-NE', 'tabpack': 'TabPack auxiliary branch',
          'tabm_xgboost': 'TabM + XGBoost', 'tabm_xgboost_kan': 'TabM + XGBoost + KAN',
          'tabpack_xgboost': 'TabPack + XGBoost'}
DTYPES = {'f32': '<f4', 'f64': '<f8', 'i32': '<i4', 'u32': '<u4', 'u8': 'u1'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Application export is an unsealed build output, separate from source study.
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False, separators=(',', ':')), encoding='utf-8')
    temporary.replace(path)


class TensorWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('wb')
        self.tensors = {}

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.stream.close()

    def add(self, name, values, dtype=None):
        if name in self.tensors:
            raise ValueError('Duplicate tensor name')
        a = np.asarray(values)
        if dtype is None:
            if a.dtype.kind == 'b':
                dtype = 'u8'
            elif a.dtype.kind in 'iu':
                if a.size and (a.min() < np.iinfo(np.int32).min or a.max() > np.iinfo(np.int32).max):
                    raise ValueError('Integer tensor outside supported range')
                dtype = 'i32'
            else:
                dtype = 'f64' if a.dtype.itemsize > 4 else 'f32'
        if dtype not in DTYPES:
            raise ValueError('Unsupported tensor dtype')
        a = np.ascontiguousarray(a, dtype=DTYPES[dtype])
        if a.dtype.kind == 'f' and not np.isfinite(a).all():
            raise ValueError('Nonfinite exported weights')
        padding = (-self.stream.tell()) % 8
        self.stream.write(b'\0' * padding)
        descriptor = dict(dtype=dtype, shape=list(a.shape), offset_bytes=self.stream.tell(), length_bytes=a.nbytes)
        self.stream.write(a.tobytes())
        self.tensors[name] = descriptor
        return descriptor


def read_tensor(path, tensor):
    dtype = np.dtype(DTYPES[tensor['dtype']])
    size = math.prod(tensor['shape']) * dtype.itemsize
    offset = tensor['offset_bytes']
    if offset < 0 or size != tensor['length_bytes'] or offset + size > Path(path).stat().st_size:
        raise ValueError('Tensor byte range invalid')
    with Path(path).open('rb') as stream:
        stream.seek(offset)
        return np.frombuffer(stream.read(size), dtype).reshape(tensor['shape'])


def sample_indices(labels, count):
    labels = np.asarray(labels)
    if labels.ndim != 1 or count < 1 or count > len(labels):
        raise ValueError('Invalid sample request')
    classes = np.unique(labels)
    selected_classes = classes if count >= len(classes) else classes[np.linspace(0, len(classes)-1, count, dtype=int)]
    chosen = [int(np.flatnonzero(labels == c)[0]) for c in selected_classes]
    remaining = np.setdiff1d(np.arange(len(labels)), chosen, assume_unique=False)
    if len(chosen) < count:
        chosen += remaining[np.linspace(0, len(remaining)-1, count-len(chosen), dtype=int)].tolist()
    return np.asarray(chosen, dtype=np.int64)


def export_forest(fitted, writer):
    imputer = fitted.named_steps['imputer']
    forest = fitted.named_steps['classifier']
    writer.add('imputer.statistics', imputer.statistics_)
    trees = []
    for index, estimator in enumerate(forest.estimators_):
        tree = estimator.tree_
        prefix = f'trees.{index}'
        for name, array in [('left', tree.children_left), ('right', tree.children_right),
                            ('feature', tree.feature), ('threshold', tree.threshold),
                            ('cover', tree.weighted_n_node_samples)]:
            writer.add(prefix + '.' + name, array)
        leaves = np.flatnonzero(tree.children_left == -1)
        leaf_index = np.full(tree.node_count, -1, dtype=np.int32)
        leaf_index[leaves] = np.arange(len(leaves), dtype=np.int32)
        writer.add(prefix + '.leaf_index', leaf_index)
        values = np.asarray(tree.value[leaves, 0, :], dtype=np.float64)
        values = values / values.sum(axis=1, keepdims=True)
        nonzero = values != 0
        classes = np.nonzero(nonzero)[1].astype(np.uint32)
        pointers = np.concatenate(([0], np.cumsum(nonzero.sum(axis=1)))).astype(np.uint32)
        writer.add(prefix + '.leaf_pointers', pointers, 'u32')
        writer.add(prefix + '.leaf_classes', classes, 'u32')
        writer.add(prefix + '.leaf_probabilities', values[nonzero], 'f64')
        trees.append(dict(prefix=prefix, nodes=tree.node_count, leaves=len(leaves)))
        print(f'EXPORT RF tree {index+1}/{len(forest.estimators_)}', flush=True) if len(forest.estimators_) > 10 else None
    return dict(imputer='imputer.statistics', trees=trees, classes=forest.classes_.astype(int).tolist(),
                threshold_comparison='less_or_equal', aggregation='mean_leaf_probabilities')


def forest_oracle(X, forest, tensors, blob):
    arrays = {name: read_tensor(blob, tensor) for name, tensor in tensors.items()}
    X = np.array(X, dtype=np.float32, copy=True)
    X = np.where(np.isnan(X), arrays[forest['imputer']], X).astype(np.float32)
    output = np.zeros((len(X), len(forest['classes'])), dtype=np.float64)
    for tree in forest['trees']:
        p = tree['prefix'] + '.'
        for row_index, row in enumerate(X):
            index = 0
            while arrays[p+'left'][index] != -1:
                index = arrays[p+'left'][index] if row[arrays[p+'feature'][index]] <= arrays[p+'threshold'][index] else arrays[p+'right'][index]
            leaf = arrays[p+'leaf_index'][index]
            begin, end = arrays[p+'leaf_pointers'][leaf:leaf+2]
            output[row_index, arrays[p+'leaf_classes'][begin:end]] += arrays[p+'leaf_probabilities'][begin:end]
    return output / len(forest['trees'])


def export_pipeline(pipeline, writer):
    steps = []
    for name, step in pipeline.steps:
        cls = type(step).__name__
        if cls == 'SimpleImputer':
            writer.add('prep.imputer.statistics', step.statistics_)
            steps.append(dict(kind='imputer', statistics='prep.imputer.statistics'))
        elif cls == 'FunctionTransformer':
            if step.func.__name__ != 'signed_log1p':
                raise ValueError('Unrecognized preprocessing function')
            steps.append(dict(kind='signed_log1p'))
        elif cls == 'QuantileTransformer':
            writer.add('prep.quantile.quantiles', step.quantiles_.T.copy())
            writer.add('prep.quantile.references', step.references_)
            steps.append(dict(kind='sklearn_quantile', quantiles='prep.quantile.quantiles',
                              references='prep.quantile.references', distribution=step.output_distribution))
        elif cls == 'MinMaxScaler':
            writer.add('prep.minmax.scale', step.scale_)
            writer.add('prep.minmax.min', step.min_)
            steps.append(dict(kind='minmax', scale='prep.minmax.scale', min='prep.minmax.min',
                              range=list(step.feature_range), clip=bool(step.clip)))
        elif cls == 'PCA':
            if step.whiten:
                raise ValueError('Whitened PCA not supported by this frozen export')
            writer.add('prep.pca.components', step.components_)
            writer.add('prep.pca.mean', step.mean_)
            steps.append(dict(kind='pca', components='prep.pca.components', mean='prep.pca.mean',
                              operation='matrix_product_then_subtract_projected_mean'))
        else:
            raise ValueError(f'Unrecognized preprocessing class {cls}')
    return dict(kind='pipeline', steps=steps)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def standalone(family, record, schema, destination):
    import joblib
    import torch
    from framework.training.hybrid_protocol import verify_hashes
    torch.set_num_threads(2)
    folder = destination / family
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / 'model.json'
    checkpoint = ROOT / record['checkpoint']
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old['source']['sha256'] != digest(checkpoint):
            raise ValueError('Existing application export has another source')
        for asset in old['files']:
            if digest(folder / asset['path']) != asset['sha256']:
                raise ValueError('Existing application export asset changed')
        return old
    source_bundle_path = STUDY / f'exports/{family}/bundle.json'
    bundle = json.loads(source_bundle_path.read_text()) if source_bundle_path.exists() else None
    if bundle:
        if bundle['status'] != 'completed':
            raise ValueError('Uncompleted source bundle')
        verify_hashes(bundle['assets'])
    else:
        verify_hashes(record['artifact_hashes'])
    result = dict(version=1, id=family, kind=family, label=LABELS[family],
                  feature_names=schema['feature_names'], classes=schema['classes'], benign_index=0,
                  source=dict(path=relative(checkpoint), sha256=digest(checkpoint)),
                  config=record['config'], calibration=bundle['calibration'] if bundle else None,
                  supports_partial=False, auxiliary=family == 'tabpack', files=[])
    blob = folder / 'weights.bin'
    with TensorWriter(blob) as writer:
        if family == 'random_forest':
            fitted = joblib.load(checkpoint)
            result['forest'] = export_forest(fitted, writer)
            result['preprocessor'] = dict(kind='identity')
            del fitted
        elif family == 'xgboost':
            shutil.copyfile(checkpoint, folder / 'model.ubj')
            result['checkpoint'] = 'model.ubj'
            result['preprocessor'] = dict(kind='identity')
            result['files'].append(dict(path='model.ubj', sha256=digest(folder/'model.ubj'), bytes=checkpoint.stat().st_size))
        else:
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if family in ('kan', 'hem'):
                for model, state in saved['state_dicts'].items():
                    for name, tensor in state.items():
                        writer.add(f'{model}.{name}', tensor.detach().numpy())
                pipeline = joblib.load(checkpoint.parent / 'preprocessing.joblib')
                result['preprocessor'] = export_pipeline(pipeline, writer)
                result['input_size'] = saved['input_size']
                if family == 'hem':
                    with np.load(checkpoint.parent / 'knn_references.npz') as references:
                        writer.add('knn.references', references['X'])
                        writer.add('knn.labels', references['y'])
                    calibrator = joblib.load(ROOT / bundle['correctness_calibrator'])
                    if calibrator.classes_.tolist() != [False, True]:
                        raise ValueError('HEM correctness class order unexpected')
                    result['correctness_calibrator'] = dict(coefficients=calibrator.coef_[0].tolist(),
                                                            intercept=float(calibrator.intercept_[0]))
            else:
                for name, tensor in saved['state_dict'].items():
                    writer.add(name, tensor.detach().numpy())
                result['preprocessor'] = dict(kind='tabm_v2', **saved['preprocessor'])
                result['member_weights'] = saved.get('member_weights')
                result['embedded'] = family in ('tabm_ne', 'tabm_hier_ne')
            del saved
        result['tensors'] = writer.tensors
    result['files'].append(dict(path='weights.bin', sha256=digest(blob), bytes=blob.stat().st_size))
    write_json(manifest_path, result)
    gc.collect()
    return result


def export_hybrid(name, schema, destination):
    from framework.training.hybrid_protocol import verify_hashes
    source = STUDY / f'hybrids/{name}/bundle.json'
    bundle = json.loads(source.read_text())
    completion = json.loads((source.parent/'result.json').read_text())
    if bundle['status'] != 'completed' or completion['status'] != 'completed':
        raise ValueError('Uncompleted hybrid')
    verify_hashes(completion['artifact_hashes'])
    manifest = dict(version=1, id=name, kind='hybrid', label=LABELS[name],
                    feature_names=schema['feature_names'], classes=schema['classes'], benign_index=0,
                    source=dict(path=relative(source), sha256=digest(source)), config={},
                    calibration=bundle['calibration'], supports_partial=False, auxiliary=False,
                    branches=list(bundle['branches']), combiner=bundle['combiner'], files=[], tensors={})
    write_json(destination/name/'model.json', manifest)
    return manifest


def row_records(X, labels, ids, indices, schema):
    output = []
    for index in indices:
        values = [float(x) if np.isfinite(x) else None for x in X[index]]
        output.append(dict(id=str(ids[index]), row_index=int(index), label_index=int(labels[index]),
                           label=schema['classes'][labels[index]], values=values))
    return output


def export_rows(destination, schema):
    metadata = dict(version=1, feature_names=schema['feature_names'], classes=schema['classes'], benign_index=0)
    for split, filename in [('validation', 'samples.json'), ('train', 'reference.json')]:
        X = np.load(DATA/f'X_{split}.npy', mmap_mode='r')
        y = np.load(DATA/f'y_{split}.npy', mmap_mode='r')
        ids = np.load(DATA/f'row_ids_{split}.npy', mmap_mode='r')
        indices = sample_indices(y, 128)
        groups = {}
        for index, name in enumerate(schema['feature_names']):
            # An original field and its derived channels form one perturbation group.
            base = name.split('__', 1)[0]
            groups.setdefault(base, []).append(index)
        write_json(destination/filename, dict(metadata, partition=split,
                  rows=row_records(X, y, ids, indices, schema), feature_groups=list(groups.values()),
                  provenance='Frozen DataSense verified-clean v2; samples are observations, not generated labels.'))


def golden(destination, selected, schema, count, filename='golden.json'):
    from framework.training.day_worker import predict_saved
    from framework.training.tabm_calibration import apply_calibration
    from framework.training.hybrid_blend import blend
    from experiment.datasense_quality import temperature_probabilities
    import joblib
    X = np.load(DATA/'X_validation.npy', mmap_mode='r')
    y = np.load(DATA/'y_validation.npy', mmap_mode='r')
    ids = np.load(DATA/'row_ids_validation.npy', mmap_mode='r')
    indices = sample_indices(y, count)
    sample = np.asarray(X[indices], dtype=np.float32)
    output, raw = {}, {}
    for family, record in selected.items():
        print(f'GOLDEN {family} {count} real validation observations CPU', flush=True)
        prediction = predict_saved(record, sample)
        if family == 'hem':
            pred, parts = prediction
            multi, binary, knn = [parts[k] for k in ('multiclass_probabilities','binary_probabilities','knn_predictions')]
            features = np.column_stack([np.log(np.clip(multi[np.arange(count),pred],1e-12,1)),
                        np.log(np.clip(np.where(pred==0,1-binary,binary),1e-12,1)), (knn==multi.argmax(1)).astype(float)])
            bundle = json.loads((STUDY/'exports/hem/bundle.json').read_text())
            confidence = joblib.load(ROOT/bundle['correctness_calibrator']).predict_proba(features)[:,1]
            output[family] = [dict(predicted=int(pred[i]), probabilities=None, raw_probabilities=None,
                 confidence=float(confidence[i]), attack_probability=None, confidence_kind='vote_correctness') for i in range(count)]
        else:
            raw[family] = prediction
            path = destination/family/'model.json'
            calibration = json.loads(path.read_text())['calibration']
            calibrated, attack = apply_calibration(prediction, calibration) if calibration else (prediction,1-prediction[:,0])
            output[family] = [dict(predicted=int(prediction[i].argmax()), raw_probabilities=prediction[i].astype(float).tolist(),
                 probabilities=calibrated[i].astype(float).tolist(), confidence=float(calibrated[i].max()),
                 attack_probability=float(attack[i]),confidence_kind='class_probability') for i in range(count)]
        gc.collect()
    for name in ('tabm_xgboost','tabm_xgboost_kan','tabpack_xgboost'):
        manifest = json.loads((destination/name/'model.json').read_text())
        combined = blend(np.stack([raw[b] for b in manifest['branches']]),manifest['combiner'],manifest['branches'])
        calibrated, attack = apply_calibration(combined,manifest['calibration'])
        output[name] = [dict(predicted=int(combined[i].argmax()),raw_probabilities=combined[i].astype(float).tolist(),
              probabilities=calibrated[i].tolist(),confidence=float(calibrated[i].max()),
              attack_probability=float(attack[i]),confidence_kind='class_probability') for i in range(count)]
    if Path(filename).name != filename or not filename.endswith('.json'):
        raise ValueError('Oracle filename must be a JSON basename')
    write_json(destination/filename, dict(version=1,partition='validation',test_loaded=False,
               feature_names=schema['feature_names'],classes=schema['classes'],
               rows=row_records(X,y,ids,indices,schema),outputs=output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--golden-rows',type=int,default=16)
    parser.add_argument('--golden-only',action='store_true')
    parser.add_argument('--golden-file',default='golden.json')
    args = parser.parse_args()
    if not (STUDY/'completion.json').is_file():
        raise ValueError('Frozen study completion is required')
    destination = args.output.resolve()
    if ROOT not in destination.parents or STUDY == destination or STUDY in destination.parents:
        raise ValueError('Export must be an isolated new project application artifact directory')
    selected = json.loads((STUDY/'selected.json').read_text())
    schema = json.loads((DATA/'manifest.json').read_text())
    if not args.golden_only:
        models = []
        for family, record in selected.items():
            print('EXPORT', family, flush=True)
            result = standalone(family,record,schema,destination)
            models.append(dict(id=family,label=LABELS[family],kind=family,auxiliary=family=='tabpack',
                 manifest=f'{family}/model.json',sha256=digest(destination/family/'model.json')))
        for name in ('tabm_xgboost','tabm_xgboost_kan','tabpack_xgboost'):
            export_hybrid(name,schema,destination)
            models.append(dict(id=name,label=LABELS[name],kind='hybrid',auxiliary=False,
                 manifest=f'{name}/model.json',sha256=digest(destination/name/'model.json')))
        export_rows(destination,schema)
        write_json(destination/'catalog.json',dict(version=1,feature_names=schema['feature_names'],
                  classes=schema['classes'],benign_index=0,models=models,inference_device='cpu',
                  source_study=relative(STUDY),test_loaded=False))
    golden(destination,selected,schema,args.golden_rows,args.golden_file)
    print('COMPLETE native application export',destination,flush=True)


if __name__ == '__main__':
    main()
