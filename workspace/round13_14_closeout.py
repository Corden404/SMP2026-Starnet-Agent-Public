"""Network-disabled regression and completed-evidence audit. No strategy/API launch."""
from __future__ import annotations
import argparse
import collections
import contextlib
import json
import subprocess
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import round13_crossed as r13
import round14_branches as r14
from round9_lock import ExclusiveBatch
OUT=r13.ROOT/'agent_notes/experiments/round13-14-closeout'


def source_hashes():
    paths=list((r13.ROOT/'tests').glob('test_*.py'))
    paths += [r13.ROOT/'workspace'/n for n in ('round13_crossed.py','round13_stress.py','round13_real.py','round14_branches.py','round15_policy.py','round15_experiment.py','round13_14_closeout.py')]
    return {p.relative_to(r13.ROOT).as_posix():r13.sha(p) for p in paths}


def tests():
    before=source_hashes();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'tests-final.log').open('w',encoding='utf8') as stream,contextlib.ExitStack() as stack:
        stack.enter_context(contextlib.redirect_stdout(stream));stack.enter_context(contextlib.redirect_stderr(stream))
        for name in ('socket.socket.connect','socket.socket.connect_ex','socket.create_connection'):
            stack.enter_context(patch(name,r13.old.denied))
        suite=unittest.defaultTestLoader.discover(str(r13.ROOT/'tests'),pattern='test_*.py')
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    if before!=source_hashes():raise RuntimeError('Test source changed during execution')
    record=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        source_hashes=before,network_disabled=True,originals=r13.preserved())
    r13.save(OUT/'tests-final.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='source_hashes'}),flush=True)
    if not result.wasSuccessful():raise SystemExit(1)


def unexplained_api_records(paths, service):
    expected=set()
    if service:
        prefix='runs/glm-glm-glm-4.6v/'
        expected={prefix+r['case']+'-'+r['variant']+'.json' for r in service['records']}
        expected.add(prefix+'STOP.json')
    return [p for p in paths if p.replace('\\','/') not in expected]


def evidence():
    test=r13.read(OUT/'tests-final.json')
    if test['status']!='PASS' or test['source_hashes']!=source_hashes():raise RuntimeError('Missing or stale test evidence')
    with ExclusiveBatch(r13.OUT/'execution.lock'):a=r13.audit()
    with ExclusiveBatch(r14.OUT/'execution.lock'):b=r14.audit()
    subprocess.run([sys.executable,'-X','utf8','-B',str(r13.ROOT/'workspace/round13_stress.py'),'audit'],check=True,cwd=r13.ROOT)
    stress_audit=r13.read(r13.ROOT/'agent_notes/experiments/round13-balanced-stress/audit.json')
    if b['incomplete'] or stress_audit['incomplete']:raise RuntimeError('Incomplete local experimental evidence')
    service=r13.read(r13.OUT/'api/service-blocked.json') if a['incomplete'] else None
    unexplained=unexplained_api_records(a['incomplete'],service)
    if unexplained:raise RuntimeError('Unexplained incomplete local records: '+repr(unexplained))
    if a['incomplete'] and service['status']!='REAL_VALIDATION_INCOMPLETE_SERVICE_ABORT':
        raise RuntimeError('Missing explicit service-failure evidence')
    main=r13.read(r13.OUT/'summaries/main-fast.json');stress=r13.read(r13.OUT/'summaries/stress-fast.json')
    scans=r13.read(r14.OUT/'summaries/scan.json');feedback=r13.read(r14.OUT/'summaries/feedback.json')
    if (main['n'],stress['n'],scans['states'],feedback['states'])!=(216,54,32,73):raise RuntimeError('Fixed cohort count mismatch')
    subprocess.run([sys.executable,'-X','utf8','-B',str(r13.ROOT/'workspace/round15_experiment.py'),'audit'],check=True,cwd=r13.ROOT)
    v3root=r13.ROOT/'agent_notes/experiments/round15-coupled'
    v3audit=r13.read(v3root/'audit.json')
    if v3audit['incomplete']:raise RuntimeError('V3 local runs incomplete')
    v3dev=r13.read(v3root/'summaries/dev-fast.json')
    if v3dev['n']!=54:raise RuntimeError('V3 development count mismatch')
    usage=collections.Counter();tokens=collections.Counter();statuses=collections.Counter()
    for folder in (r13.OUT,r14.OUT,r13.ROOT/'agent_notes/experiments/round13-balanced-stress',v3root):
        for p in folder.rglob('*.jsonl'):
            for line in p.read_text(encoding='utf8').splitlines():
                if not line.strip():continue
                row=json.loads(line)
                if row.get('event')=='request':usage[row.get('data',{}).get('model','unknown')]+=1
                if row.get('event')=='response':
                    statuses[str(row.get('status'))]+=1
                    try:body=json.loads(row.get('body','{}'))
                    except ValueError:body={}
                    if isinstance(body,dict) and isinstance(body.get('usage'),dict):
                        model=body.get('model','unknown')
                        for key in ('prompt_tokens','completion_tokens','total_tokens'):
                            tokens[model+'/'+key]+=int(body['usage'].get(key,0))
    record=dict(status='LOCAL_COMPLETE_REAL_SERVICE_INCOMPLETE' if service else 'PASS',
        real_validation=service,retained_incomplete_api_records=a['incomplete'],
        tests=test['tests'],full_games=a['games']+stress_audit['games']+v3audit['games'],diagnostic_branches=b['branches'],
        v3_development=r13.read(v3root/'development-decision.json'),
        stress_protocol_sha256=r13.sha(r13.ROOT/'agent_notes/experiments/round13-balanced-stress/protocol.json'),
        max_forward_error=max(a['max_forward_error'],b['max_forward_error'],stress_audit['max_forward_error'],v3audit['max_forward_error']),
        local_decision=r13.read(r13.OUT/'local-decision.json'),api_requests=dict(usage),
        api_tokens=dict(tokens),api_http_statuses=dict(statuses),
        diagnostic_scope='Selected old states, not independent graph means or legal policies',originals=r13.preserved())
    r13.save(OUT/'final-audit.json',record);print(json.dumps(record),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['tests','evidence']);args=parser.parse_args()
    with ExclusiveBatch(OUT/'execution.lock'):
        tests() if args.stage=='tests' else evidence()

if __name__=='__main__':main()
