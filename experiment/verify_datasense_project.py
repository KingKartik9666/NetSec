"""Record the project's five required demo commands and complete evaluation."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import subprocess
import sys

from experiment.audit_datasense_raw import DEFAULT,ROOT
from experiment.datasense_quality import write_json


def run(attempt=''):
    base=DEFAULT/'verification'
    if attempt and not attempt.replace('_','').isalnum():raise ValueError('Invalid attempt name')
    out=base/attempt if attempt else base
    out.mkdir(parents=True,exist_ok=True)
    commands=[('demo_default',['run_demo.py'])]
    commands.extend((f'demo_{name}',['run_demo.py','--sample',name]) for name in ('benign','ddos','scan','unknown'))
    commands.append(('evaluation',['-m','pytest','evaluation','-v']))
    results=[]
    for name,args in commands:
        path=out/f'{name}.log'
        if path.exists():raise FileExistsError(path)
        completed=subprocess.run([sys.executable,'-B',*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        path.write_text(completed.stdout,encoding='utf-8')
        results.append({'name':name,'command':[sys.executable,'-B',*args],'exit_code':completed.returncode,'log':str(path.relative_to(DEFAULT))})
        print(f'{name}: exit {completed.returncode}',flush=True)
        if name=='evaluation':print('\n'.join(completed.stdout.splitlines()[-30:]),flush=True)
    summary={'checks':results,'all_passed':all(r['exit_code']==0 for r in results),'attempt':attempt}
    write_json(out/'project_checks.json',summary)
    if attempt:write_json(base/'project_checks.json',summary)
    if not all(r['exit_code']==0 for r in results):raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--attempt',default='')
    run(parser.parse_args().attempt)
