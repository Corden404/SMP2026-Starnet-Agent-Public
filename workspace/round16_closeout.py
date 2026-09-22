"""Round16 read-only status, network-disabled regression and evidence closeout.

Only writes new round16 audit files. Never starts policy cohorts or paid calls.
Partial cohorts are reported as incomplete, never promoted by partial means.
"""
from __future__ import annotations
import argparse
import collections
import contextlib
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
import round16_common as c
from round9_lock import ExclusiveBatch

OUT=c.OUT/'closeout'
REAL=c.ROOT/'agent_notes/experiments/round16-f1-recovery'


def source_hashes():
    paths=list((c.ROOT/'tests').glob('test_*.py'))
    paths+=list((c.ROOT/'workspace').glob('round16_*.py'))
    return {p.relative_to(c.ROOT).as_posix():c.sha(p) for p in sorted(paths)}


def row_status(folder):
    counts=collections.Counter();active=[]
    for p in sorted(Path(folder).glob('*.json')):
        row=c.read(p);state=row.get('status','UNSPECIFIED');counts[state]+=1
        if state not in ('PASS','REPLAY_VERIFIED'):
            active.append(dict(path=p.name,status=state,age_seconds=round(time.time()-p.stat().st_mtime),
                               states_completed=len(row.get('states',[])),error_type=row.get('error_type')))
    return dict(counts=dict(counts),incomplete=active)


def status():
    result=dict(diagnostic=row_status(c.OUT/'diagnostic/runs'),real=row_status(REAL/'runs'),
        real_blocks=row_status(c.ROOT/'agent_notes/experiments/round16-f1-blocks/runs'),
        pc2={pool:row_status(c.OUT/'pc2/runs'/pool) for pool in ('dev','confirm','stress','integration')},
        original_protection=c.preserved(),partial_scores_not_summarized=True)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def tests():
    before=source_hashes();OUT.mkdir(parents=True,exist_ok=True)
    stamp=time.strftime('%Y%m%d-%H%M%S')
    path=OUT/('tests-'+stamp+'.log')
    with path.open('x',encoding='utf8') as stream,contextlib.ExitStack() as stack:
        stack.enter_context(contextlib.redirect_stdout(stream));stack.enter_context(contextlib.redirect_stderr(stream))
        for name in ('socket.socket.connect','socket.socket.connect_ex','socket.create_connection'):
            stack.enter_context(patch(name,c.old.denied))
        suite=unittest.defaultTestLoader.discover(str(c.ROOT/'tests'),pattern='test_*.py')
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    if before!=source_hashes():raise RuntimeError('Source changed during test execution')
    record=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        source_hashes=before,network_disabled=True,log=path.relative_to(c.ROOT).as_posix(),
        originals=c.preserved())
    c.save(OUT/('tests-'+stamp+'.json'),record,True)
    c.save(OUT/'tests-final.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='source_hashes'}),flush=True)
    if result.testsRun==0 or not result.wasSuccessful():raise SystemExit(1)
    return record


def journal_usage(folder):
    requests=collections.Counter();statuses=collections.Counter();tokens=collections.Counter();uncertain=[]
    for path in sorted(Path(folder).rglob('*.jsonl')):
        for line in path.read_text(encoding='utf8').splitlines():
            if not line.strip():continue
            event=json.loads(line)
            if event.get('event')=='request':requests[event.get('data',{}).get('model','unknown')]+=1
            elif event.get('event')=='response':
                statuses[str(event.get('status'))]+=1
                try:body=json.loads(event.get('body','{}'))
                except (ValueError,TypeError):body={}
                if isinstance(body,dict):
                    for name in ('prompt_tokens','completion_tokens','total_tokens'):
                        tokens[name]+=int(body.get('usage',{}).get(name,0))
            elif event.get('event')=='uncertain':
                uncertain.append(dict(path=path.relative_to(folder).as_posix(),number=event.get('number'),
                                      error_type=event.get('error_type'),may_be_billed=event.get('may_be_billed')))
    return dict(actual_new_requests=dict(requests),http_statuses=dict(statuses),reported_tokens=dict(tokens),
        uncertain=uncertain,usage_is_not_cost=True,replayed_old_responses_not_counted_twice=True)


def audit_real():
    if (c.ROOT/'agent_notes/experiments/round16-f1-blocks/protocol.json').exists():
        return audit_real_blocks()
    import round16_f1_recovery as r
    protocol=r.check();games=0;max_error=0.;missing=[];failed=[];replayed=0;charged=0
    completed=set();paths=[]
    for entry in protocol['seeds']:
        for variant in ('S44','F1'):
            path=REAL/'runs'/(entry['spec']['case']+'-'+variant+'.json')
            if not path.exists():missing.append(path.name);continue
            row=c.read(path)
            if row.get('status')!='PASS':failed.append(dict(path=path.name,status=row.get('status'),error_type=row.get('error_type')));continue
            if row['identity']['entry']!=entry or row['identity']['code']!=protocol['code'] or row['identity']['model_config']!=r.config(r.MODEL):
                raise RuntimeError('Real result identity mismatch')
            max_error=max(max_error,c.audit_game(row));games+=1
            replayed+=row['replayed_responses'];charged+=row['old_uncertain_attempts_charged']
            completed.add((entry['spec']['case'],variant));paths.append(path.relative_to(c.ROOT).as_posix())
    pairs=sum(all((e['spec']['case'],v) in completed for v in ('S44','F1')) for e in protocol['seeds'])
    return dict(status='COMPLETE' if games==96 and not failed and not missing else 'INCOMPLETE',
        planned_graphs=48,completed_games=games,complete_pairs=pairs,failed=failed,missing=missing,
        max_forward_error=max_error,replayed_responses=replayed,old_uncertain_attempts_charged=charged,
        actual_usage=journal_usage(REAL),no_partial_score_conclusion=games!=96,paths=paths)


def combine_usage(folders):
    requests=collections.Counter();statuses=collections.Counter();tokens=collections.Counter();uncertain=[]
    for folder in folders:
        usage=journal_usage(folder)
        requests.update(usage['actual_new_requests']);statuses.update(usage['http_statuses']);tokens.update(usage['reported_tokens'])
        uncertain.extend(dict(item,folder=Path(folder).name) for item in usage['uncertain'])
    return dict(actual_new_requests=dict(requests),http_statuses=dict(statuses),reported_tokens=dict(tokens),
        uncertain=uncertain,usage_is_not_cost=True,replayed_old_responses_not_counted_twice=True)


def audit_real_blocks():
    import round16_f1_blocks as blocks
    p=blocks.check();games=0;max_error=0.;missing=[];failed=[];replayed=0;charged=0;completed=set();paths=[]
    for entry in p['seeds']:
        for variant in ('S44','F1'):
            name=entry['spec']['case']+'-'+variant+'.json';valid=[];errors=[]
            for folder,is_parent in ((blocks.PARENT/'runs',True),(blocks.OUT/'runs',False)):
                path=folder/name
                if not path.exists():continue
                row=c.read(path);blocks.verify_identity(row,entry,variant,is_parent,p)
                if row['status']=='PASS':valid.append((path,row))
                elif is_parent and row['spec']['case']==p['paused']['case'] and variant==p['paused']['variant']:
                    continue
                else:errors.append(dict(path=path.relative_to(c.ROOT).as_posix(),status=row.get('status'),error_type=row.get('error_type')))
            if len(valid)>1:raise RuntimeError('Duplicate completed real game')
            failed.extend(errors)
            if not valid:missing.append(name);continue
            path,row=valid[0];max_error=max(max_error,c.audit_game(row));games+=1
            replayed+=row['replayed_responses'];charged+=row.get('old_uncertain_attempts_charged',0)
            completed.add((entry['spec']['case'],variant));paths.append(path.relative_to(c.ROOT).as_posix())
    pairs=sum(all((e['spec']['case'],v) in completed for v in ('S44','F1')) for e in p['seeds'])
    return dict(status='COMPLETE' if games==96 and not failed and not missing else 'INCOMPLETE',
        planned_graphs=48,completed_games=games,complete_pairs=pairs,failed=failed,missing=missing,
        max_forward_error=max_error,replayed_responses=replayed,old_uncertain_attempts_charged=charged,
        actual_usage=combine_usage((blocks.PARENT,blocks.OUT)),no_partial_score_conclusion=games!=96,paths=paths,
        scheduling_pause=dict(registration='round16-f1-recovery/block-transition-request.json',
            preserved=True,not_a_service_outage=True,paused_prefix=p['paused']),
        model=blocks.MODEL,cohort_unchanged=True)


def required_candidate_cohorts(stages,development):
    required={'dev':162,'integration':18}
    if development.get('eligible'):
        required.update(confirm=432,stress=108)
    complete=bool(development) and all(stages.get(pool,{}).get('counts')=={'PASS':count} for pool,count in required.items())
    return dict(complete=complete,required=required,
        conditional_skips=['confirm','stress'] if development and not development.get('eligible') else [])


def audit_diagnostic():
    import round16_diagnose as d
    rows=row_status(d.OUT/'runs')
    if rows['counts'].get('PASS',0)==54 and len(rows['counts'])==1:
        with ExclusiveBatch(d.OUT/'execution.lock'):
            return d.audit()
    return dict(status='INCOMPLETE',planned_graphs=54,**rows)


def audit_pc2():
    import round16_experiment as e
    if not (e.OUT/'protocol.json').exists():
        decision_path=c.OUT/'diagnostic/summary.json'
        stopped=decision_path.exists() and c.read(decision_path).get('next_branch')=='STOP'
        return dict(status='SKIPPED_DIAGNOSTIC_GATE' if stopped else 'NOT_STARTED_PENDING_DIAGNOSTIC_GATE',
                    required_cohorts=dict(complete=stopped))
    with ExclusiveBatch(e.OUT/'execution.lock'):
        result=e.audit()
    result['stages']={pool:row_status(e.OUT/'runs'/pool) for pool in ('dev','confirm','stress','integration')}
    if (e.OUT/'development-decision.json').exists():
        result['development']=c.read(e.OUT/'development-decision.json')
    result['required_cohorts']=required_candidate_cohorts(result['stages'],result.get('development',{}))
    return result


def check_tests():
    test=c.read(OUT/'tests-final.json')
    if test['status']!='PASS' or test['tests']<=0 or test['source_hashes']!=source_hashes():
        raise RuntimeError('Missing, failing or stale final tests')
    return test


def evidence():
    test=check_tests()
    diagnostic=audit_diagnostic();candidate=audit_pc2();real=audit_real()
    result=dict(status='COMPLETE' if diagnostic['status']=='PASS' and real['status']=='COMPLETE'
                and candidate.get('status') in ('PASS','SKIPPED_DIAGNOSTIC_GATE')
                and candidate.get('required_cohorts',{}).get('complete') else 'PARTIAL_OR_CONDITIONALLY_COMPLETE',
        tests={k:test[k] for k in ('status','tests','failures','errors','skipped','network_disabled','log')},
        diagnostic=diagnostic,pc2=candidate,real=real,originals=c.preserved(),
        not_official_scores=True,uploads=0,git_commits=0,source_hashes=source_hashes())
    c.save(OUT/'final-audit.json',result)
    brief=copy.deepcopy(result);brief.pop('source_hashes',None)
    for key in ('missing','paths'):brief['real'].pop(key,None)
    print(json.dumps(brief,ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['status','tests','evidence']);a=p.parse_args()
    with ExclusiveBatch(OUT/'execution.lock'):
        if a.stage=='status':status()
        elif a.stage=='tests':tests()
        else:evidence()

if __name__=='__main__':main()
