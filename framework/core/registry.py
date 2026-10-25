"""Four independent, lazy model workers with bounded requests and recovery."""
from __future__ import annotations

import atexit
import importlib.util
import json
import logging
import multiprocessing
from pathlib import Path
import threading
import time

from .config import MODEL_IDS, load_configuration
from .contracts import CapabilityError
from .worker import OPERATIONS, model_worker

LOGGER = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, root: Path, config_path: Path | None = None, *, configuration: dict | None = None):
        self.root = Path(root).resolve()
        path = config_path or self.root / 'framework/models/four_models/config.json'
        self.configuration = load_configuration(path, configuration)
        self.timeout_seconds = float(self.configuration.get('request_timeout_seconds', 60))
        self._context = multiprocessing.get_context('spawn')
        self._workers: dict[str, tuple] = {}
        self._locks = {name: threading.Lock() for name in MODEL_IDS}
        self._lifecycle = threading.RLock()
        self._errors: dict[str, dict] = {}
        self._closed = False
        atexit.register(self.close)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _manifest(self, name: str) -> dict:
        configured = self.configuration['models'][name].get('manifest')
        if not configured:
            return {}
        path = (self.root / configured).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Model manifest must stay inside the runtime root')
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.get('model_id') != name:
            raise ValueError('Manifest model identity differs from selection')
        return value

    def describe(self) -> list[dict]:
        """Availability discovery reads manifests; no optional model imports."""
        entries = []
        for name in MODEL_IDS:
            config = self.configuration['models'][name]
            entry = {'model_id': name, 'label': config.get('label', name), 'status': 'ready',
                     'reason': 'Configured; checkpoint loading and integrity are checked on selection',
                     'loaded': self._alive(name),
                     'capabilities': {}, 'metadata': {}}
            try:
                if not config.get('enabled', True):
                    raise FileNotFoundError('This model is disabled in configuration')
                for dependency in config.get('dependencies', []):
                    if importlib.util.find_spec(dependency) is None:
                        raise ImportError(f'Optional dependency missing: {dependency}')
                metadata = self._manifest(name)
                entry['metadata'] = metadata
                entry['capabilities'] = metadata.get('capabilities', {})
                for field in ['checkpoint', 'preprocessing', 'knn_references']:
                    if metadata.get(field) and not (self.root / metadata[field]).is_file():
                        raise FileNotFoundError(f'Missing {name} {field}; restore versioned artifacts')
            except (FileNotFoundError, ImportError) as exc:
                entry.update(status='unavailable', reason=str(exc))
            except Exception as exc:
                entry.update(status='incompatible', reason=f'{type(exc).__name__}: {exc}')
            if name in self._errors:
                entry['last_capability_error'] = self._errors[name]
                if self._errors[name]['capability'] in {'predict', 'predict_proba', 'health', 'metadata', 'snapshot'}:
                    entry.update(status=self._errors[name]['status'], reason=self._errors[name]['error'])
            entries.append(entry)
        return entries

    def _alive(self, name: str) -> bool:
        pair = self._workers.get(name)
        try:
            return bool(pair and pair[0].is_alive())
        except ValueError:  # A concurrent stop closed this process handle.
            return False

    def _stop(self, name: str) -> None:
        with self._lifecycle:
            pair = self._workers.pop(name, None)
        if not pair:
            return
        process, connection = pair
        try:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        finally:
            connection.close()
            if not process.is_alive():
                process.close()

    def call(self, model_id: str, operation: str, payload: dict | None = None):
        """A crash, timeout or import failure affects this request/model only."""
        if model_id not in MODEL_IDS:
            raise CapabilityError(model_id, operation, 'incompatible', 'Select one of the four configured model families')
        if self._closed:
            raise CapabilityError(model_id, operation, 'unavailable', 'The model registry is closed')
        if operation not in OPERATIONS:
            raise CapabilityError(model_id, operation, 'unavailable', 'Unsupported capability')
        config = self.configuration['models'][model_id]
        if not config.get('enabled', True):
            raise CapabilityError(model_id, operation, 'unavailable', 'This model is disabled in configuration')
        flags = self.configuration.get('plugins', {})
        if operation in {'explain', 'additional_xai'} and not flags.get('explanation', True):
            raise CapabilityError(model_id, operation, 'unavailable', 'Explanation capability disabled; raw prediction remains available')
        if (payload or {}).get('method') == 'lime' and not flags.get('lime', True):
            raise CapabilityError(model_id, operation, 'unavailable', 'LIME disabled; raw prediction remains available')
        deadline = time.monotonic() + self.timeout_seconds
        lock = self._locks[model_id]
        if not lock.acquire(timeout=self.timeout_seconds):
            raise CapabilityError(model_id, operation, 'unavailable', 'This model is busy; retry after the current bounded request')
        try:
            with self._lifecycle:
                if self._closed:
                    raise CapabilityError(model_id, operation, 'unavailable', 'The model registry is closed')
                if not self._alive(model_id):
                    self._stop(model_id)
                    parent, child = self._context.Pipe()
                    process = self._context.Process(target=model_worker, args=(child, model_id, {**config, 'plugins': flags}, str(self.root)), daemon=True)
                    process.start()
                    child.close()
                    self._workers[model_id] = (process, parent)
                process, connection = self._workers[model_id]
            connection.send({'operation': operation, 'payload': payload or {}})
            if not connection.poll(max(0, deadline - time.monotonic())):
                self._stop(model_id)
                raise CapabilityError(model_id, operation, 'failed', 'Model worker exceeded the request timeout and was stopped; other models remain available')
            response = connection.recv()
            if not response['ok']:
                raise CapabilityError(model_id, operation, response['status'], response['reason'])
            self._errors.pop(model_id, None)
            return response['result']
        except (EOFError, BrokenPipeError, OSError) as exc:
            self._stop(model_id)
            error = CapabilityError(model_id, operation, 'failed', f'Model worker exited unexpectedly ({type(exc).__name__}); retry to reload this model')
            self._errors[model_id] = error.to_dict()
            LOGGER.error(json.dumps(error.to_dict()))
            raise error from exc
        except CapabilityError as exc:
            self._errors[model_id] = exc.to_dict()
            LOGGER.error(json.dumps(exc.to_dict()))
            raise
        finally:
            lock.release()

    def comparison(self) -> dict:
        rows = []
        for entry in self.describe():
            metadata = entry['metadata']
            metrics = metadata.get('metrics')
            rows.append({'model_id': entry['model_id'], 'label': entry['label'],
                         'status': entry['status'], 'version': metadata.get('version'),
                         'protocol_id': metadata.get('protocol_id'), 'input_profile': metadata.get('input_profile'),
                         'implementation': metadata.get('implementation'), 'metrics': metrics,
                         'metrics_available': isinstance(metrics, dict),
                         'reason': entry['reason'] if not metrics else None})
        protocol_ids = {row['protocol_id'] for row in rows if row['metrics_available']}
        comparable = len(protocol_ids) == 1 and None not in protocol_ids
        return {'rows': rows, 'comparable': comparable,
                'comparison_basis': 'Same source, labels and evaluation row IDs; declared model-specific input adapters' if comparable else
                                    'Incomplete or different evaluation protocols; no cross-protocol ranking',
                'test_evaluated': False, 'live_accuracy_measured': False}

    def close(self) -> None:
        with self._lifecycle:
            self._closed = True
            for name in MODEL_IDS:
                self._stop(name)
        atexit.unregister(self.close)
