"""Read-only combined status of the two seven-family training controllers."""
from __future__ import annotations

import json
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[1]
GROUPS=(('verified_clean_four_models_20261016_v1',('random_forest','xgboost','kan','hem'),32),
        ('tabm_verified_v2_20261016_v1',('tabm_ne','tabm_hier','tabm_hier_ne'),30))


def alive(pid):
    try:
        return bool(pid) and psutil.Process(pid).is_running() and not psutil.Process(pid).status()==psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess,psutil.AccessDenied):
        return False


def status():
    report=dict(dataset='datasense_verified_clean_20261014_v2',features=655,classes=84,
                planned_fits=62,completed_fits=0,models={},test_loaded=False)
    for name,families,planned in GROUPS:
        directory=ROOT/'artifacts/training'/name
        path=directory/'state.json'
        state=json.loads(path.read_text()) if path.exists() else {}
        live=alive(state.get('pid'));active=state.get('active_trial') or {}
        for family in families:
            count=state.get('completed_fits',{}).get(family,0)
            running=active.get('family')==family and alive(active.get('pid'))
            phase=('training' if running else 'queued' if live and not state.get('status','').startswith('completed')
                   else state.get('status','not_started') if live or state.get('status') in ('completed','completed_with_failures','stopped')
                   else 'controller_not_running')
            entry=dict(status=phase,completed_fits=count,controller_alive=live,study=name)
            if running:
                entry.update(recipe=active['recipe'],seed=active['seed'])
            entry['best']=state.get('best',{}).get(family)
            report['models'][family]=entry;report['completed_fits']+=count
    return report


if __name__=='__main__':
    print(json.dumps(status(),indent=2))
