"""Native optional language/capture services cannot terminate the HTTP process."""
from __future__ import annotations

import atexit
import multiprocessing
import threading

from .contracts import CapabilityError


def capability_worker(connection, kind: str, root: str, configuration: dict) -> None:
    service = None
    registry = None
    try:
        while True:
            try:
                operation, arguments = connection.recv()
            except EOFError:
                break
            if operation == 'close':
                break
            try:
                if service is None:
                    if kind == 'language':
                        from framework.llm.local_model import LocalLanguageModel
                        service = LocalLanguageModel()
                    elif kind == 'monitor':
                        from pathlib import Path
                        from .registry import ModelRegistry
                        from framework.detection.lazy_xgboost import LazyXGBoostFacade
                        from framework.data.live_monitor import LiveMonitor
                        registry = ModelRegistry(Path(root), configuration=configuration)
                        service = LiveMonitor(detector=LazyXGBoostFacade(registry))
                    else:
                        raise ValueError('Unknown optional service kind')
                result = getattr(service, operation)(*arguments)
                connection.send({'ok': True, 'result': result})
            except Exception as exc:
                connection.send({'ok': False, 'reason': f'{type(exc).__name__}: {exc}',
                                 'status': 'unavailable' if isinstance(exc, (ImportError, FileNotFoundError)) else 'failed'})
    finally:
        if kind == 'monitor' and service is not None:
            service.stop()
        if registry is not None:
            registry.close()
        connection.close()


class _IsolatedService:
    def __init__(self, registry, kind: str):
        self.registry, self.kind = registry, kind
        self._context = multiprocessing.get_context('spawn')
        self._process = None
        self._connection = None
        self._lock = threading.Lock()
        self.timeout_seconds = 60 if kind == 'language' else 15
        atexit.register(self.close)

    def _call(self, operation, *arguments):
        if self.kind == 'language' and not self.registry.configuration.get('plugins', {}).get('language', True):
            raise CapabilityError(self.kind, operation, 'unavailable', 'Local language capability disabled in configuration')
        if not self._lock.acquire(timeout=self.timeout_seconds):
            raise CapabilityError(self.kind, operation, 'unavailable', 'This capability is busy; retry shortly')
        try:
            if self._process is None or not self._process.is_alive():
                self._stop()
                parent, child = self._context.Pipe()
                # Capture owns model children; do not mark its service daemon.
                process = self._context.Process(target=capability_worker,
                    args=(child, self.kind, str(self.registry.root), self.registry.configuration))
                process.start()
                child.close()
                self._process, self._connection = process, parent
            self._connection.send((operation, arguments))
            if not self._connection.poll(self.timeout_seconds):
                self._stop()
                raise CapabilityError(self.kind, operation, 'failed', 'Optional service timed out and was stopped; model research remains available')
            response = self._connection.recv()
            if not response['ok']:
                raise CapabilityError(self.kind, operation, response['status'], response['reason'])
            return response['result']
        except (EOFError, BrokenPipeError, OSError) as exc:
            self._stop()
            raise CapabilityError(self.kind, operation, 'failed', 'Optional service exited; retry to reload only this capability') from exc
        finally:
            self._lock.release()

    def _stop(self):
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2)
            self._connection.close()
            if not self._process.is_alive():
                self._process.close()
            self._process, self._connection = None, None

    def close(self):
        with self._lock:
            if self._process is not None and self._process.is_alive():
                try:
                    self._connection.send(('close', ()))
                    self._process.join(timeout=3)
                except (EOFError, BrokenPipeError, OSError):
                    pass
            self._stop()


class IsolatedLanguageService(_IsolatedService):
    def __init__(self, registry):
        super().__init__(registry, 'language')

    def status(self):
        return self._call('status')

    def generate(self, evidence):
        return self._call('generate', evidence)


class IsolatedMonitorService(_IsolatedService):
    def __init__(self, registry):
        super().__init__(registry, 'monitor')

    def capabilities(self):
        return self._call('capabilities')

    def snapshot(self):
        if self._process is None:
            # Polling an unused monitor must neither load models nor spawn a
            # capture process. Only explicit capabilities/start need it.
            return {'running': False, 'mode': None, 'interface': None,
                    'state': 'stopped', 'error': None, 'elapsed_seconds': 0,
                    'metrics': {'packets_per_second': 0, 'bytes_per_second': 0,
                                'packets_observed': 0, 'active_connections': 0},
                    'assessment': {'status': 'not_assessed', 'evidence': []},
                    'model_prediction': None, 'multiclass_prediction': None,
                    'connections': [], 'recent_packets': [], 'history': []}
        return self._call('snapshot')

    def start(self, interface, mode):
        return self._call('start', interface, mode)

    def stop(self):
        # Stopping an unused capability must not start its process/dependencies.
        if self._process is None:
            return {'running': False, 'state': 'stopped', 'model_prediction': None, 'multiclass_prediction': None}
        return self._call('stop')
