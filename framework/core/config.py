"""Validated local configuration; exactly four research model families."""
from __future__ import annotations

import json
import math
from pathlib import Path

MODEL_IDS = ('random_forest', 'xgboost', 'kan', 'hem')


def default_configuration() -> dict:
    names = {'random_forest': ('Random Forest', 'RandomForestAdapter'),
             'xgboost': ('XGBoost', 'XGBoostModelAdapter'),
             'kan': ('KAN', 'KANAdapter'), 'hem': ('HEM', 'HEMModelAdapter')}
    return {'schema_version': 1, 'request_timeout_seconds': 60,
            'models': {key: {'enabled': True, 'label': label,
                             'adapter': f'framework.detection.adapters.{key}:{adapter}',
                             'manifest': f'framework/models/four_models/{key}.json',
                             'dependencies': ['torch', 'joblib'] if key in ['kan', 'hem'] else
                                             ['sklearn', 'joblib'] if key == 'random_forest' else ['xgboost']}
                       for key, (label, adapter) in names.items()},
            'plugins': {'explanation': True, 'lime': True, 'language': True,
                        'calibration': False, 'conformal': False}}


def load_configuration(path: Path | None = None, configuration: dict | None = None) -> dict:
    value = json.loads(json.dumps(configuration)) if configuration is not None else (
        json.loads(path.read_text(encoding='utf-8')) if path and path.is_file() else default_configuration())
    if value.get('schema_version') != 1 or set(value.get('models', {})) != set(MODEL_IDS):
        raise ValueError('Configuration must contain exactly random_forest, xgboost, kan and hem')
    timeout = value.get('request_timeout_seconds', 60)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0.1 <= timeout <= 300:
        raise ValueError('request_timeout_seconds must be finite and between 0.1 and 300')
    for name, model in value['models'].items():
        if type(model.get('enabled', True)) is not bool:
            raise ValueError(f'{name}.enabled must be boolean')
        adapter = model.get('adapter')
        if not isinstance(adapter, str) or adapter.count(':') != 1:
            raise ValueError(f'{name} requires an adapter module:class')
    plugins = value.get('plugins', {})
    if not isinstance(plugins, dict) or any(type(flag) is not bool for flag in plugins.values()):
        raise ValueError('Plugin activation flags must be booleans')
    return value
