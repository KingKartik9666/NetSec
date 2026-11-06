from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT / 'artifacts/training/live_models_20261025_v1'
BRANCHES = ('random_forest','xgboost','kan','hem','tabm_ne','tabm_hier','tabm_hier_ne','tabpack')
HYBRIDS = {'tabm_xgboost':['tabm_ne','xgboost'],
           'tabm_xgboost_kan':['tabm_ne','xgboost','kan'],
           'tabpack_xgboost':['tabpack','xgboost']}
FEATURES = ['network_packets_all_count', *[f'network_tcp-flags-{flag}_count' for flag in ('ack','fin','psh','rst','syn','urg')]]
ROLES = ('fit','tune','blend','calibration')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        for retry in range(12):
            try:
                os.replace(temporary,path); return
            except PermissionError:
                if retry == 11: raise
                time.sleep(.05*(retry+1))
    finally:
        temporary.unlink(missing_ok=True)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify(pins):
    for name, expected in pins.items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Sealed source/input changed: {name}; create an explicit new study revision')


@contextlib.contextmanager
def exclusive(path):
    """OS-released lock: a stale PID file cannot acquire a second controller."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        if path.stat().st_size == 0: stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream,fcntl.LOCK_UN)


def source_pins():
    # These are code dependencies only. App renderer changes do not stop training.
    paths=list((ROOT/'framework/training').rglob('*.py'))
    paths += list((ROOT/'experiment').glob('*.py'))
    paths += [ROOT/p for p in ('external/efficient-kan/src/efficient_kan/kan.py',
        'external/HEM/model_evaluation/binary_classifier.py', 'external/HEM/model_evaluation/multiclass_classifier.py',
        'external/tabpack/src/project/nn.py','external/tabpack/src/project/optim.py',
        'external/tabpack/src/project/tabpack.py','external/tabpack/src/vendor/muon.py')]
    return {relative(p):digest(p) for p in paths if p.is_file()}


def versions():
    from importlib.metadata import version
    import sys
    return dict(python=sys.version, **{name:version(name) for name in
        ('numpy','scipy','scikit-learn','torch','xgboost','optuna','tabm','rtdl-num-embeddings','joblib')})


def verify_study(study):
    settings = read(Path(study)/'study.json')
    if fingerprint({k:v for k,v in settings.items() if k != 'protocol_fingerprint'}) != settings['protocol_fingerprint']:
        raise ValueError('Sealed study settings changed')
    verify(settings['source_hashes'])
    if versions() != settings['versions']:
        raise ValueError('Training library versions changed; explicit new revision required')
    manifest = read(Path(study)/'data/manifest.json')
    if fingerprint(manifest) != settings['data_manifest_fingerprint']:
        raise ValueError('Data manifest changed')
    verify(manifest['array_hashes']); verify(manifest['input_hashes'])
    return settings, manifest
