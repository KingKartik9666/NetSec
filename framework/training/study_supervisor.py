"""Small durable-study primitives; no ML frameworks in the supervisor process."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import time

FAMILIES = ('xgboost', 'kan', 'hem', 'tabm_ne', 'tabm_hier', 'tabm_hier_ne', 'tabpack')


def atomic_write(path, value):
    """Unique temp files and bounded retries for Windows readers denying delete."""
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False)+'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(12):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 11:
                    raise
                time.sleep(min(.025*2**attempt, .5))
    finally:
        temporary.unlink(missing_ok=True)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def phase_at(protocol, now=None):
    now = time.time() if now is None else now
    if now >= protocol['deadline']:
        return 'expired'
    elapsed = (now-protocol['started_at'])/3600
    return 'search' if elapsed < 14 else 'repeats' if elapsed < 17 else 'oof' if elapsed < 21 else 'finalize'


def next_family(observations):
    """Coverage first; then improvement per second with an exploration bonus."""
    if any(len(observations[f]) < 3 for f in FAMILIES):
        return min(FAMILIES, key=lambda f: (len(observations[f]), FAMILIES.index(f)))
    total = sum(len(rows) for rows in observations.values())
    def priority(family):
        rows = observations[family]
        values = [r['value'] for r in rows if r.get('value') is not None]
        gain = max(values, default=0)-values[0] if values else 0
        seconds = sum(r.get('seconds', 900) for r in rows)/len(rows)
        exploration = .015*math.sqrt(math.log(total+1)/len(rows))
        return (gain+exploration)/max(60, seconds)
    return max(FAMILIES, key=priority)


def process_matches(pid, module, token=None):
    import psutil
    try:
        args = psutil.Process(pid).cmdline()
        return module in args and (token is None or token in args)
    except (psutil.Error, TypeError):
        return False


def terminate_tree(pid):
    import psutil
    try:
        parent = psutil.Process(pid)
        for child in reversed(parent.children(recursive=True)):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
        parent.wait(timeout=15)
    except psutil.NoSuchProcess:
        pass

