"""Same frozen F1 real48 cohort continued in bounded blocks after operator pause.

The pause is scheduling-only and has an immutable registration. Parent PASS
results are reused verbatim. The single known-not-sent interrupted request is
resumed only after exact replay. Real service failures are never auto-retried.
"""
from __future__ import annotations
import argparse
import collections
import contextlib
import copy
import io
import json
import sys
import time
from pathlib import Path
import round16_common as c
import round16_f1_recovery as parent
from round9_lock import ExclusiveBatch
from experiment_round5 import FiniteEnv
from benchmark import LoggedEnv

PARENT=parent.OUT
OUT=c.ROOT/'agent_notes/experiments/round16-f1-blocks'
DEPENDENCIES=('workspace/round16_f1_blocks.py','tests/test_round16_f1_blocks.py')
MODEL=parent.MODEL


def fingerprint():
    return dict(parent=parent.fingerprint(),blocks=c.fingerprint(DEPENDENCIES))


def inspect_parent():
    original=parent.check()
    request=c.read(PARENT/'block-transition-request.json')
    if request['cause']!='EXECUTION_SCHEDULING_ONLY_NOT_SERVICE_FAILURE_NOT_SCORE_SELECTION':
        raise parent.RecoveryMismatch('Missing registered operator pause')
    rows=[];failed=[]
    for path in sorted((PARENT/'runs').glob('*.json')):
        row=c.read(path)
        if row.get('identity',{}).get('code')!=original['code']:
            raise parent.RecoveryMismatch('Parent result identity drift')
        if row['status']=='PASS':
            c.audit_game(row)
        elif row['status']=='FAILED' and row['error_type']=='ServiceAbort':
            completed,pending=parent.replay_records(path.with_suffix('.http.jsonl'))
            if pending is not None:
                raise parent.RecoveryMismatch('Actual uncertain request is not an operator pause')
            if row.get('replayed_responses',0)!=0:
                raise parent.RecoveryMismatch('Nested historical replay is not supported by this block transition')
            failed.append(dict(path=path.relative_to(c.ROOT).as_posix(),case=row['spec']['case'],variant=row['variant'],
                               completed_responses=len(completed),confirmed_actions=len(row['actions']),confirmed_chains=len(row['chains'])))
        else:
            raise parent.RecoveryMismatch('Parent execution still active or unexpected failure')
        rows.append(path)
    if len(failed)!=1:
        raise parent.RecoveryMismatch('Expected exactly one inspected operator-paused game')
    stopped=c.read(PARENT/'STOP.json')
    if stopped.get('case')!=failed[0]['case'] or stopped.get('variant')!=failed[0]['variant']:
        raise parent.RecoveryMismatch('STOP does not match inspected paused game')
    protected=rows+list((PARENT/'runs').glob('*.jsonl'))+[PARENT/n for n in
        ('protocol.json','preflight.json','health.json','STOP.json','block-transition-request.json')]
    return original,failed[0],{p.relative_to(c.ROOT).as_posix():c.sha(p) for p in protected}


def prepare():
    original,paused,protected=inspect_parent()
    protocol=dict(code=fingerprint(),model=MODEL,model_config=parent.config(MODEL),seeds=original['seeds'],
        planned_graphs=48,protected_parent_files=protected,paused=paused,max_graphs_per_process=12,
        cause='Explicit scheduling-only operator pause, not a service outage. No partial scores consulted.',
        equivalence='Exact S44/F1 package hashes and same original48 graphs/model. Parent complete games retained without rewriting identities; new suffix replays complete parent responses before any new request.',
        limits='Complete each requested fixed block or stop on first error. Never retry a failed block automatically. Old and parent STOP files remain.',
        no_upload=True,no_model_switch=True,no_seed_change=True,no_parameter_change=True)
    c.save(OUT/'protocol.json',protocol,True)
    print(json.dumps(dict(prepared=True,paused=paused,parent_complete_games=len([p for p in (PARENT/'runs').glob('*.json') if c.read(p)['status']=='PASS']),originals=c.preserved())),flush=True)


def check():
    p=c.read(OUT/'protocol.json')
    if p['code']!=fingerprint():raise parent.RecoveryMismatch('Block driver source changed')
    for name,h in p['protected_parent_files'].items():
        if c.sha(c.ROOT/name)!=h:raise parent.RecoveryMismatch('Parent evidence changed: '+name)
    return p


def verify_identity(row,entry,variant,from_parent,protocol):
    expected_code=parent.check()['code'] if from_parent else protocol['code']
    if row['identity']['entry']!=entry or row['identity']['variant']!=variant or row['identity']['code']!=expected_code:
        raise parent.RecoveryMismatch('Result graph/policy fingerprint mismatch')
    if row['identity']['model_config']!=parent.config(MODEL):raise parent.RecoveryMismatch('Mixed real model/config')


def resolved(entry,variant,protocol):
    name=entry['spec']['case']+'-'+variant+'.json'
    candidates=[]
    for folder,old in ((PARENT/'runs',True),(OUT/'runs',False)):
        path=folder/name
        if path.exists():
            row=c.read(path);verify_identity(row,entry,variant,old,protocol)
            if row['status']=='PASS':candidates.append((path,row,old))
            elif not old:
                raise parent.RecoveryMismatch('New incomplete/failed game; no automatic retry')
    if len(candidates)>1:raise parent.RecoveryMismatch('Duplicate completed result')
    return candidates[0] if candidates else None


def replay_one(index,variant,live=False):
    p=check();entry=p['seeds'][index];spec=entry['spec']
    found=resolved(entry,variant,p)
    if found:return found[1]
    path=OUT/('runs' if live else 'preflight')/(spec['case']+'-'+variant+'.json')
    identity=dict(entry=entry,variant=variant,model_config=parent.config(MODEL),code=p['code'])
    if path.exists():
        row=c.read(path)
        if row['identity']!=identity or row['status']!='REPLAY_VERIFIED':
            raise parent.RecoveryMismatch('Existing block replay needs inspection')
        return row
    if live and (OUT/'STOP.json').exists():raise parent.ServiceAbort('New service failure; do not retry')
    old_path=PARENT/'runs'/path.name
    history=c.read(old_path) if old_path.exists() else None
    completed,pending=parent.replay_records(old_path.with_suffix('.http.jsonl')) if history else ([],None)
    if pending is not None:raise parent.RecoveryMismatch('Unexpected transport uncertainty in scheduling pause')
    if history and not (history['status']=='FAILED' and history['error_type']=='ServiceAbort'):
        raise parent.RecoveryMismatch('History is not the inspected operator pause')
    module=parent.load(variant);module.coefficients.cache_clear()
    raw=FiniteEnv(c.generate(spec));logged=LoggedEnv(raw);chains=[];model=None
    llm=parent.ReplayLLM(path.with_suffix('.http.jsonl'),completed,pending,live)
    llm.stop_path=OUT/'STOP.json'
    row=dict(status='STARTED',identity=identity,spec=spec,variant=variant);c.save(path,row)
    started=time.monotonic()
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()) if live else c.offline())
            model=c.old.initialize(module,logged,mode='integration',llm=llm)
            llm.counter=model.llm_counter
            def boundary():
                if history is None:return
                if len(logged.actions)!=len(history['actions']):raise parent.RecoveryMismatch('Paused action count differs')
                for a,b in zip(logged.actions,history['actions']):
                    if a['action']!=b['action'] or list(a['args'])!=list(b['args']) or not c.receipt_equal(a['result'],b['result']):
                        raise parent.RecoveryMismatch('Paused public action/receipt differs')
                if not parent.same_chains(chains,history['chains']):raise parent.RecoveryMismatch('Paused chain history differs')
            llm.boundary=boundary
            for role in ('scout','commander'):
                agent=getattr(model,role);original=agent._run_chain
                def trace(name,data,original=original,role=role):
                    before=copy.deepcopy(data);answer=original(name,data)
                    chains.append(dict(role=role,chain=name,operation=model.operation_count,input=before,output=copy.deepcopy(answer)))
                    return answer
                agent._run_chain=trace
            result=c.finish(raw,logged,model)
        if history and not llm.boundary_checked:raise parent.RecoveryMismatch('Paused prefix boundary not reached')
        if llm.http_request_count>model.llm_counter.call_count or len(raw._scanned)!=44:
            raise parent.RecoveryMismatch('Resource/scan accounting mismatch')
        row.update(result,status='PASS',chains=chains,http_calls=llm.http_request_count,new_http_calls=llm.new_requests,
            replayed_responses=llm.cursor,old_uncertain_attempts_charged=llm.uncertain_charged,
            metadata=llm.response_metadata,feedback_records=getattr(model,'feedback_records',[]),
            elapsed_seconds=time.monotonic()-started)
        c.save(path,row)
    except parent.ReplayComplete:
        row.update(status='REPLAY_VERIFIED',confirmed_operations=len(logged.actions),confirmed_chains=len(chains),
            replayed_responses=llm.cursor,new_http_calls=0)
        c.save(path,row)
    except BaseException as exc:
        row.update(status='FAILED',error_type=type(exc).__name__,actions=logged.actions,chains=chains,
            new_http_calls=llm.new_requests,replayed_responses=llm.cursor,elapsed_seconds=time.monotonic()-started)
        c.save(path,row);c.save(OUT/('STOP.json' if live else 'PREFLIGHT-STOP.json'),dict(case=spec['case'],variant=variant,error_type=type(exc).__name__))
        raise
    return row


def preflight():
    p=check();paused=p['paused']
    index=next(i for i,e in enumerate(p['seeds']) if e['spec']['case']==paused['case'])
    row=replay_one(index,paused['variant'])
    if row['status']!='REPLAY_VERIFIED':raise parent.RecoveryMismatch('Operator-paused prefix did not verify')
    result={k:row[k] for k in ('status','confirmed_operations','confirmed_chains','replayed_responses','new_http_calls')}
    result.update(code=fingerprint(),pause_registration_sha256=c.sha(PARENT/'block-transition-request.json'))
    c.save(OUT/'preflight.json',result,True)
    print(json.dumps({k:v for k,v in result.items() if k!='code'}),flush=True);return result


def run(start,stop):
    p=check();proof=c.read(OUT/'preflight.json')
    if proof['status']!='REPLAY_VERIFIED' or proof['code']!=fingerprint():raise parent.RecoveryMismatch('Missing block preflight')
    if not 0<=start<stop<=48 or stop-start>12:raise ValueError('A process may run at most12 original graph pairs')
    if (OUT/'STOP.json').exists():raise parent.ServiceAbort('Prior service failure requires inspection')
    with ExclusiveBatch(PARENT/'execution.lock'):
        for index in range(start,stop):
            order=('S44','F1') if index%2==0 else ('F1','S44')
            for variant in order:
                row=replay_one(index,variant,True)
                print(json.dumps(dict(index=index,variant=variant,status=row['status'])),flush=True)
    result=dict(status='FIXED_BLOCK_COMPLETE',range=[start,stop],model=MODEL,code=fingerprint())
    c.save(OUT/f'block-{start:02d}-{stop:02d}.json',result,True)
    return result


def audit_complete(require_all=True):
    p=check();rows=[];pairs=[];missing=[];max_error=0.
    for entry in p['seeds']:
        group={}
        for variant in ('S44','F1'):
            found=resolved(entry,variant,p)
            if found is None:missing.append(entry['spec']['case']+'-'+variant);continue
            path,row,old=found;max_error=max(max_error,c.audit_game(row));rows.append((path,row));group[variant]=row
        if len(group)==2:
            pairs.append(dict(case=entry['spec']['case'],family=entry['spec']['family'],delta=group['F1']['final_score']-group['S44']['final_score']))
    if require_all and missing:raise parent.RecoveryMismatch('Original48 cohort incomplete')
    result=dict(status='COMPLETE' if not missing else 'INCOMPLETE',planned_graphs=48,completed_games=len(rows),complete_pairs=len(pairs),missing=missing,
        max_forward_error=max_error,completed_sources={path.relative_to(c.ROOT).as_posix():c.sha(path) for path,row in rows},
        replayed_responses=sum(row['replayed_responses'] for _,row in rows),old_uncertain_attempts_charged=sum(row.get('old_uncertain_attempts_charged',0) for _,row in rows),
        originals=c.preserved())
    if not missing:
        result.update(statistics=c.paired_stats(pairs),pairs=pairs,scope='Same original48 synthetic graphs, glm-4.6v. Scheduling blocks are not extra independent samples. Not Tianchi hidden evaluation.')
    c.save(OUT/'audit.json',result)
    if not missing:c.save(OUT/'summary.json',result,True)
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','completed_sources','missing')}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','preflight','run','summary','audit'])
    p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=12);a=p.parse_args()
    with ExclusiveBatch(OUT/'execution.lock'):
        if a.stage=='prepare':prepare()
        elif a.stage=='preflight':preflight()
        elif a.stage=='run':run(a.start,a.stop)
        else:audit_complete(require_all=a.stage=='summary')

if __name__=='__main__':main()
