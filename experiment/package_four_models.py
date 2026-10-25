"""Package hash-pinned model manifests; original checkpoints are never rewritten."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil

from experiment.train_four_models import ROOT, STUDY, sha256, verify_common_partitions
from framework.core.config import default_configuration
from framework.training.tracking import write_json

OUTPUT = ROOT / 'framework/models/four_models'


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def pin(paths: list[Path]) -> dict:
    return {relative(path): sha256(path) for path in paths}


def historical(family: str) -> tuple[dict, Path]:
    selected = ROOT / f'artifacts/training/datasense_all_models_full_v1/selected/{family}.json'
    result = json.loads(selected.read_text())['seed_42_record']
    checkpoint = Path(result['checkpoint_path'])
    return result, checkpoint.parent / 'result.json'


def select(family: str) -> tuple[dict | None, Path | None]:
    records = []
    for name in ['control', 'enhanced', 'fixed_baseline']:
        path = STUDY / f'runs/{family}_{name}_s42/result.json'
        if path.is_file():
            result = json.loads(path.read_text())
            if result.get('status') == 'completed':
                if sha256(ROOT / result['checkpoint_path']) != result['checkpoint_sha256']:
                    raise ValueError('Completed result checkpoint integrity failed')
                records.append((result, path))
    # Never promote an enhancement before its matched control completed.
    controls = [r for r in records if r[0]['variant'] != 'enhanced']
    if controls:
        best = controls[0]
        for record in records:
            if record[0]['metrics']['macro_f1'] > best[0]['metrics']['macro_f1'] + 1e-8:
                best = record
        return best
    if family in ['kan', 'hem']:
        return historical(family)
    return None, None


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    numeric = ROOT / 'dataset/prepared/datasense_full_train'
    protocol = ROOT / 'dataset/prepared/datasense_full_protocols'
    import hashlib
    identities = verify_common_partitions(numeric, protocol)
    numeric_manifest = json.loads((numeric / 'manifest.json').read_text())
    protocol_id = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
    limitations = ['Within-capture validation; independent-device/capture generalization not established.',
                   'Held-out test remains locked; unknown-attack detection is not evaluated.',
                   'Unlabelled PC capture proves execution, not live detection accuracy.',
                   'Official UNB identifies DataSense as CIC-IIoT-2025; local CSV download provenance is retained separately.']
    selected = {}
    for family in ['random_forest', 'kan', 'hem']:
        record, source_record = select(family)
        if record is None:
            continue
        checkpoint = ROOT / record['checkpoint_path'] if not Path(record['checkpoint_path']).is_absolute() else Path(record['checkpoint_path'])
        paths = [checkpoint, source_record]
        manifest = {'schema_version': 1, 'model_id': family,
                    'version': 'four-v2-' + checkpoint.parent.name if 'four_models_v2' in checkpoint.parts else 'preserved-' + record['run_id'],
                    'checkpoint': relative(checkpoint), 'source_record': relative(source_record),
                    'feature_names': record.get('feature_names', record.get('features')), 'classes': record['classes'],
                    'benign_index': record.get('benign_index', numeric_manifest['benign_index']), 'config': record['config'],
                    'metrics': record['metrics'], 'protocol_id': protocol_id, 'input_profile': 'full_numeric',
                    'training_rows': record['training_rows'], 'training_seconds': record.get('training_seconds'),
                    'dataset': 'DataSense / CIC-IIoT-2025 dataset family; local CSV',
                    'fresh_retraining': 'four_models_v2' in checkpoint.parts,
                    'limitations': list(limitations), 'test_evaluated': False}
        if family in ['kan', 'hem']:
            preprocessing = checkpoint.parent / 'preprocessing.joblib'
            manifest['preprocessing'] = relative(preprocessing)
            paths.append(preprocessing)
            if family == 'kan':
                paths.append(ROOT / 'external/efficient-kan/src/efficient_kan/kan.py')
                manifest.update(upstream_commit='7b6ce1c87f18c8bc90c208f6b494042344216b11',
                                implementation='Genuine efficient-kan B-spline architecture; train-only scaling',
                                capabilities={'predict': True, 'predict_proba': True, 'explain': ['gradient'], 'partial_inputs': False})
            else:
                paths += [ROOT / f'external/HEM/model_evaluation/{name}.py' for name in ['binary_classifier', 'multiclass_classifier', 'full_model']]
                references = checkpoint.parent / 'knn_references.npz'
                paths.append(references)
                manifest.update(knn_references=relative(references),
                                upstream_commit='8513ad39c904dbe427727ee6f8470ee8a1ce8e0d',
                                implementation='Original HEM component classes and vote; DataSense closed-set input adaptation',
                                capabilities={'predict': True, 'predict_proba': False, 'explain': ['component_diagnostics'], 'partial_inputs': False, 'native_unknown_detection': False})
                manifest['limitations'] += ['Original Kitsune benchmark/unknown-threshold reproduction remains unavailable; no ensemble probabilities are fabricated.']
        else:
            manifest.update(implementation='sklearn RandomForestClassifier: fixed, conventional baseline',
                            capabilities={'predict': True, 'predict_proba': True, 'explain': ['global_importance'], 'partial_inputs': False})
            manifest['limitations'] += ['Base paper RF hyperparameters are undisclosed; selected library defaults are documented, not exact paper reproduction.',
                                        'CPU training is the explicit fixed-RF exception; no RF tuning performed.']
        manifest['artifact_sha256'] = pin(paths)
        write_json(OUTPUT / (family + '.json'), manifest)
        selected[family] = {'version': manifest['version'], 'metrics': manifest['metrics'], 'fresh_retraining': manifest['fresh_retraining']}

    # The current promoted detector remains the explicit live workspace model.
    original_path = ROOT / 'framework/models/xgboost_partial/bundle.json'
    bundle = json.loads(original_path.read_text())
    original_checkpoint = (original_path.parent / bundle['checkpoint']).resolve()
    result, result_path = select('xgboost')
    if result:
        checkpoint = ROOT / result['checkpoint_path']
        bundle = copy.deepcopy(bundle)
        bundle['checkpoint_sha256'] = result['checkpoint_sha256']
        bundle['profile_metrics'] = result['profile_metrics']
        bundle['model'].update(run_id=checkpoint.parent.name, version='four-v2-' + checkpoint.parent.name,
                               validation_accuracy=result['metrics']['accuracy'], macro_f1=result['metrics']['macro_f1'],
                               saved_rounds=result['saved_rounds'],
                               live_validation_accuracy=result['profile_metrics']['packet_counts_7']['accuracy'],
                               live_macro_f1=result['profile_metrics']['packet_counts_7']['macro_f1'],
                               selection_basis='Matched fresh control vs single L2 ablation, full-validation macro F1; current live runtime remains separately pinned.')
        bundle['live_profile_validation'] = result['profile_metrics']['packet_counts_7']
        bundle['training_study'] = relative(STUDY)
        source_records = [result_path]
        fresh = True
    else:
        checkpoint = original_checkpoint
        source_records = [original_path]
        fresh = False
    bundle['checkpoint'] = os.path.relpath(checkpoint, OUTPUT).replace('\\', '/')
    bundle_path = OUTPUT / 'xgboost_bundle.json'
    write_json(bundle_path, bundle)
    for extension in ['npz', 'json']:
        shutil.copyfile(original_path.parent / ('xai_reference.' + extension), OUTPUT / ('xai_reference.' + extension))
    manifest = {'schema_version': 1, 'model_id': 'xgboost', 'version': bundle['model']['version'],
                'checkpoint': relative(checkpoint), 'bundle': relative(bundle_path),
                'feature_names': bundle['feature_names'], 'classes': bundle['classes'], 'benign_index': bundle['benign_index'],
                'config': result['config'] if result else {}, 'metrics': bundle['profile_metrics']['full'],
                'profile_metrics': bundle['profile_metrics'], 'protocol_id': protocol_id, 'input_profile': 'full_csv',
                'fresh_retraining': fresh, 'training_rows': 480405, 'training_seconds': result.get('training_seconds') if result else None,
                'implementation': 'Current depth-six partial-input XGBoost; real CUDA masked-input training',
                'capabilities': {'predict': True, 'predict_proba': True, 'explain': ['native', 'lime', 'global'], 'partial_inputs': True},
                'limitations': limitations, 'test_evaluated': False}
    manifest['artifact_sha256'] = pin([checkpoint, bundle_path, *source_records, OUTPUT / 'xai_reference.npz', OUTPUT / 'xai_reference.json'])
    write_json(OUTPUT / 'xgboost.json', manifest)
    selected['xgboost'] = {'version': manifest['version'], 'metrics': manifest['metrics'], 'fresh_retraining': fresh}
    if not (OUTPUT / 'config.json').exists():
        write_json(OUTPUT / 'config.json', default_configuration())
    write_json(STUDY / 'selected.json', selected)
    print(json.dumps({name: {'version': item['version'], 'accuracy': item['metrics']['accuracy'],
                             'fresh_retraining': item['fresh_retraining']} for name, item in selected.items()}, indent=2))


if __name__ == '__main__':
    main()
