"""Child-process boundary for optional imports and native model execution."""
from __future__ import annotations

import importlib
import logging
from pathlib import Path

OPERATIONS = frozenset({'health', 'metadata', 'predict', 'predict_proba', 'explain',
                        'analyze', 'snapshot', 'additional_xai', 'validate_features', 'legacy_status', 'legacy_predict', 'legacy_explain'})


def model_worker(connection, model_id: str, configuration: dict, root: str) -> None:
    """One worker owns one adapter. No user payload selects an import target."""
    adapter = None
    loaded = False
    try:
        while True:
            try:
                message = connection.recv()
            except EOFError:
                break
            if message.get('operation') == 'close':
                break
            operation, payload = message['operation'], message.get('payload', {})
            try:
                if operation not in OPERATIONS:
                    raise NotImplementedError('Unsupported model operation')
                if adapter is None:
                    module, class_name = configuration['adapter'].split(':')
                    constructor = getattr(importlib.import_module(module), class_name)
                    adapter = constructor(configuration, Path(root))
                if operation in {'predict', 'predict_proba', 'explain'} and not loaded:
                    adapter.load()
                    loaded = True
                if operation == 'health':
                    result = adapter.health_check()
                elif operation == 'metadata':
                    result = adapter.metadata()
                elif operation == 'snapshot':
                    result = adapter.snapshot()
                elif operation == 'legacy_status':
                    result = adapter.legacy_status()
                elif operation == 'legacy_predict':
                    result = adapter.legacy_predict(payload.get('rows'))
                elif operation == 'legacy_explain':
                    result = adapter.legacy_explain(payload.get('values'))
                elif operation in {'predict', 'predict_proba'}:
                    result = getattr(adapter, operation)(payload.get('rows'))
                elif operation == 'explain':
                    result = adapter.explain(payload.get('rows'), payload.get('method', 'native'))
                elif operation == 'analyze':
                    result = adapter.analyze(payload.get('values'), payload.get('source', {}))
                elif operation == 'additional_xai':
                    result = adapter.additional_xai(payload.get('values'))
                else:
                    result = adapter.validate_features(payload.get('values'))
                connection.send({'ok': True, 'result': result})
            except Exception as exc:
                status = ('unavailable' if isinstance(exc, (ImportError, FileNotFoundError, NotImplementedError))
                          else 'incompatible' if isinstance(exc, (ValueError, TypeError)) else 'failed')
                logging.getLogger(__name__).exception('model_capability_failure model=%s capability=%s', model_id, operation)
                connection.send({'ok': False, 'status': status, 'reason': f'{type(exc).__name__}: {exc}'})
    finally:
        connection.close()
