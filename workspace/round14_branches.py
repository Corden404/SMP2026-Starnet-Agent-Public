"""All historical V2 decision states and F1 swap opportunities, paired branches.

Diagnostic only. Real synthetic responses live exclusively in FiniteEnv. The
participant sees public receipts; cost-only and eligibility restrictions are
explicit test interventions. This file is never included in a submission.
"""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import statistics
import sys
import time
import types
from unittest.mock import patch

os.environ['ANONYMIZED_TELEMETRY']='False'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import round13_crossed as r13
from round4_seeds import generate
from experiment_round5 import FiniteEnv
from benchmark import LoggedEnv
from round9_lock import ExclusiveBatch

ROOT=r13.ROOT
OUT=ROOT/'agent_notes/experiments/round14-branches'
read,save,sha,digest=r13.read,r13.save,r13.sha,r13.digest
KIND={'scan_node':'scan','communicate':'comm','shield_node':'shield','cut_link':'cut'}
METHOD={v:k for k,v in KIND.items()}
SOURCE_DIRS=dict(scan=ROOT/'agent_notes/experiments/round12-active-scan/runs/dev-fast',
                 feedback=ROOT/'agent_notes/experiments/round11-feedback/runs/dev-fast')


def entries(kind):
    result=[]
    variant='V2' if kind=='scan' else 'F1'
    for path in sorted(SOURCE_DIRS[kind].glob('*-'+variant+'.json')):
        row=read(path)
        if row['status']!='PASS':raise RuntimeError('Historical incomplete record')
        events=row['active_records'] if kind=='scan' else row['feedback_records']
        for event in events:
            if event['event']!=('evaluate' if kind=='scan' else 'proposed'):continue
            result.append(dict(id=f'{kind}-{len(result):03}',case=row['case'],family=row['spec']['family'],
                source=path.relative_to(ROOT).as_posix(),source_sha256=sha(path),
                operation=event['operation'],event=copy.deepcopy(event)))
    if len(result)!=(32 if kind=='scan' else 73):raise RuntimeError('Historical state-count drift')
    return result


def fingerprint():
    return dict(base=r13.fingerprint(),sources={n:sha(ROOT/n) for n in
                ('workspace/round14_branches.py','tests/test_round14.py')})


def prepare():
    r13.check()
    e={k:entries(k) for k in ('scan','feedback')}
    p=dict(entries=e,code=fingerprint(),counts={k:len(x) for k,x in e.items()},workers=2,
        scope='All old development states; not independent confirmation; graph-clustered summaries. No retrospective selection.',
        scan_branches='A no purchase; B charge1 budget+2 operations+1 logical call, disclose nothing; C two real scan receipts but original44 eligibility; D two real scans with new targets eligible. Shared remaining S44 policy. No-sensitive-pair states explicitly skipped.',
        feedback_branches='A shield then first communication; R communication then original shield; F communication then F1 response-conditioned next action. After these2 actions all branches continue exact S44, never recursively apply F1. A/R endpoint equality checked.',
        estimation='Historical predicted gains preserved before evaluation. All negative predicted scan states with a valid pair are evaluated, not only purchases. Evaluation truth is never a pre-action feature.',
        v3_screen='This diagnostic can identify failures and justify implementation, never qualify a submission. Before V3, paired public-state value estimate must pass no-information and shared-policy controls, and corrected decisions must be evaluated on NEW development graphs.',
        no_api=True,no_upload=True,no_old_mutations=True,no_new_training=True)
    save(OUT/'protocol.json',p,True)
    print(json.dumps(dict(prepared=True,counts=p['counts'],originals=r13.preserved())),flush=True)


def check():
    p=read(OUT/'protocol.json')
    if p['code']!=fingerprint():raise RuntimeError('Frozen diagnostic source changed')
    for es in p['entries'].values():
        for e in es:
            if sha(ROOT/e['source'])!=e['source_sha256']:raise RuntimeError('Historical input changed')
    r13.preserved();return p


def force(model,raw,action,reserve=True):
    kind=action['action'];node=action['target_node_1']
    if reserve:model.llm_counter._reserve_call()
    before=None if kind=='scan' else dict(model.local_nodes[node])
    result=model._execute_planned_action(action,raw.get_remaining_budget(),before)
    if not model.action_history[-1]['success']:raise RuntimeError('Forced public action failed')
    return result


def action_from_logged(model,a):
    kind=KIND[a['action']];args=a['args']
    return model._action_dict(kind,args[0],args[1] if kind=='cut' else None)


def restore(row,operation,variant='S44'):
    """Replay only recorded public operations; no policy/LLM decisions in prefix."""
    raw=FiniteEnv(generate(row['spec']));logged=LoggedEnv(raw);module=r13.load(variant)
    module.coefficients.cache_clear();model=r13.old.initialize(module,logged)
    for index,a in enumerate(row['actions'][:operation]):
        if index==44:
            model._should_scan(raw.get_remaining_budget())
        force(model,raw,action_from_logged(model,a),reserve=False)
        if not r13.receipt_equal(logged.actions[-1]['result'],a['result']):raise RuntimeError('Public prefix mismatch')
    if operation>=44:model._should_scan(raw.get_remaining_budget())
    model.llm_counter.call_count=sum(c['operation']<operation for c in row['chains'])
    # No pending F1 action survives a manually replayed prefix.
    if hasattr(model,'_feedback_pending'):model._feedback_pending=None
    return module,raw,logged,model


def public_state(model,raw):
    return dict(nodes=copy.deepcopy(model.local_nodes),edges=sorted(model.local_edges),dead=sorted(model.dead_nodes),
        effects=dict(model.comm_effects),observations=copy.deepcopy(model.persona_observations),
        budget=raw.get_remaining_budget(),operations=model.operation_count,calls=model.llm_counter.call_count,
        unseen_w=model.unseen_w)


def restrict_candidates(module,model,eligible):
    allowed=set(eligible)
    def restricted(self,budget):
        state=self.completed_state()
        slots=min(self.operation_limit-1-self.operation_count,self.llm_counter.safe_limit-self.llm_counter.call_count)
        engine=module.RestrictedRollout(*state,module.SEARCH,slots,eligible=allowed)
        reference=engine.candidates(engine.initial,budget)
        if not reference:return []
        plans=engine.choose(budget,baseline_action=reference[0])
        self.search_records.append(dict(operation=self.operation_count,plans=plans))
        return [plans[0]['first']] if plans else reference
    model._learned_candidates=types.MethodType(restricted,model)


def finish_branch(module,raw,logged,model,operation,charge=None):
    for _ in range(120-model.operation_count):
        if model.step():break
    else:
        if not model.stopped:raise RuntimeError('Branch reached fuse')
    if not (model.operation_count<120 and model.llm_counter.call_count<120):raise RuntimeError('Resource limit')
    charge=charge or dict(budget=0.,operations=0,calls=0)
    expected_budget=100-sum(r13.old.COSTS[a['action']] for a in logged.actions)-charge['budget']
    if abs(expected_budget-raw.get_remaining_budget())>1e-8 or model.operation_count!=len(logged.actions)+charge['operations']:
        raise RuntimeError('Diagnostic charge accounting error')
    result=dict(final_score=raw.evaluate(),remaining_budget=raw.get_remaining_budget(),
        operations=model.operation_count,calls=model.llm_counter.call_count,charge=charge,
        suffix=logged.actions[operation:],stop_reason=model.stop_reason,
        feedback_records=getattr(model,'feedback_records',[]),
        search_truncations=sum(bool(s['plans'][0].get('search_truncated')) for s in model.search_records if s['plans']),
        http_calls=0)
    module.coefficients.cache_clear();return result


def scan_branch(row,entry,branch):
    module,raw,logged,model=restore(row,entry['operation'])
    initial=public_state(model,raw);eligible=set(model.local_nodes);pair=entry['event']['pair']
    if not all(i not in eligible and i not in model.dead_nodes for i in pair) or len(set(pair))!=2:
        raise RuntimeError('Illegal diagnostic scan pair')
    charge=None
    if branch=='B':
        # Explicit test-only bookkeeping: no fabricated public receipt or graph mutation.
        raw._budget-=1.;model.operation_count+=2;model.schedule.time+=2;model.schedule.steps+=2
        model.llm_counter._reserve_call();charge=dict(budget=1.,operations=2,calls=1)
    elif branch in ('C','D'):
        model.llm_counter._reserve_call()
        for node in pair:force(model,raw,model._action_dict('scan',node),reserve=False)
        if branch=='C':restrict_candidates(module,model,eligible)
    result=finish_branch(module,raw,logged,model,entry['operation'],charge)
    result['initial_public_state']=initial
    return result


def feedback_branch(row,entry,branch):
    module,raw,logged,model=restore(row,entry['operation'],'F1')
    initial=public_state(model,raw);event=entry['event'];a=event['shield_node'];b=event['probe_node']
    shield=model._action_dict('shield',a);comm=model._action_dict('comm',b)
    shield.update(cost=5.,gain=0.,roi=0.);comm.update(cost=2.,gain=0.,roi=0.)
    if a==b or b in model.comm_effects or model.local_nodes[b]['comm_left']!=3:raise RuntimeError('Not an eligible first communication')
    if branch=='A':
        force(model,raw,shield);force(model,raw,comm)
    elif branch=='R':
        force(model,raw,comm);force(model,raw,shield)
    else:
        model._feedback_pending=dict(shield=shield,probe_node=b,operation=model.operation_count,ready=False)
        force(model,raw,comm)
        if model.step() and raw.get_remaining_budget()>=2 and not model.stopped:
            raise RuntimeError('Unexpected feedback stop')
    endpoint=public_state(model,raw)
    # Continue exactly the base S44 decision policy, not the F1 wrapper.
    model._feedback_pending=None
    model._learned_candidates=types.MethodType(module._FeedbackBase._learned_candidates,model)
    result=finish_branch(module,raw,logged,model,entry['operation'])
    result.update(initial_public_state=initial,pair_endpoint=endpoint)
    return result


def run_group(kind,index):
    protocol=check();entry=protocol['entries'][kind][index]
    target=OUT/'runs'/kind/(entry['id']+'.json')
    identity=dict(entry=entry,code=protocol['code'])
    if target.exists():
        result=read(target)
        if result['identity']!=identity or result['status'] not in ('PASS','NO_SENSITIVE_PAIR'):
            raise RuntimeError('Incomplete or changed diagnostic result')
        return dict(id=entry['id'],status=result['status'])
    if (OUT/'STOP.json').exists():raise RuntimeError('Previous diagnostic failed')
    row=read(ROOT/entry['source']);result=dict(identity=identity,entry=entry,status='STARTED',branches={})
    save(target,result);started=time.monotonic()
    try:
        if kind=='scan' and 'pair' not in entry['event']:
            result.update(status='NO_SENSITIVE_PAIR',reason=entry['event']['reason'],no_branch_execution=True)
        else:
            branches=['A','B','C','D'] if kind=='scan' else ['A','R','F']
            if index%2:branches.reverse()
            with contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                for name in ('socket.socket.connect','socket.socket.connect_ex','socket.create_connection'):
                    stack.enter_context(patch(name,r13.old.denied))
                for branch in branches:
                    result['branches'][branch]=(scan_branch if kind=='scan' else feedback_branch)(row,entry,branch)
            outputs=result['branches']
            if kind=='scan':
                result['differences']={label:outputs[right]['final_score']-outputs[left]['final_score']
                    for label,left,right in [('resource','A','B'),('information','B','C'),('new_targets','C','D'),('total','A','D')]}
                result['predicted_gain']=entry['event'].get('expected_gain')
                result['historical_purchase']=entry['event'].get('reason')=='purchase'
            else:
                if outputs['A']['pair_endpoint']!=outputs['R']['pair_endpoint']:
                    raise RuntimeError('Commuting endpoint mismatch')
                result['differences']={label:outputs[right]['final_score']-outputs[left]['final_score']
                    for label,left,right in [('reorder','A','R'),('feedback','R','F'),('total','A','F')]}
                result['reorder_suffix_equal']=outputs['A']['suffix'][2:]==outputs['R']['suffix'][2:]
            result['status']='PASS'
        result['elapsed_seconds']=time.monotonic()-started;save(target,result)
    except BaseException as exc:
        result.update(status='FAILED',error_type=type(exc).__name__);save(target,result)
        save(OUT/'STOP.json',dict(id=entry['id'],error_type=type(exc).__name__));raise
    return dict(id=entry['id'],status=result['status'],differences=result.get('differences'))


def summarize(kind):
    p=check();rows=[read(OUT/'runs'/kind/(e['id']+'.json')) for e in p['entries'][kind]]
    if any(r['status'] not in ('PASS','NO_SENSITIVE_PAIR') for r in rows):raise RuntimeError('Incomplete fixed diagnostic')
    valid=[r for r in rows if r['status']=='PASS'];labels=list(valid[0]['differences']) if valid else []
    graphmeans={}
    for case in sorted({r['entry']['case'] for r in valid}):
        group=[r for r in valid if r['entry']['case']==case]
        graphmeans[case]={k:statistics.mean(r['differences'][k] for r in group) for k in labels}
    result=dict(kind=kind,states=len(rows),evaluated_states=len(valid),skipped=len(rows)-len(valid),
        graphs=len(graphmeans),branch_runs=sum(len(r['branches']) for r in rows),
        mean_per_state={k:statistics.mean(r['differences'][k] for r in valid) for k in labels},
        mean_of_graph_means={k:statistics.mean(r[k] for r in graphmeans.values()) for k in labels},
        positive_states={k:sum(r['differences'][k]>1e-8 for r in valid) for k in labels},
        negative_states={k:sum(r['differences'][k]< -1e-8 for r in valid) for k in labels},
        per_graph=graphmeans,records=[dict(id=r['entry']['id'],case=r['entry']['case'],family=r['entry']['family'],
            operation=r['entry']['operation'],differences=r.get('differences'),
            historical_predicted=r.get('predicted_gain'),historical_purchase=r.get('historical_purchase')) for r in valid],
        scope='Historical selected decision states, diagnostic only; states within graph correlated. Not full-policy cohort performance.',
        originals=r13.preserved())
    if kind=='scan':
        predicted=[r for r in valid if r.get('predicted_gain') is not None]
        result['sign_errors']=sum((r['predicted_gain']>0)!=(r['differences']['total']>0) for r in predicted)
        result['predicted_n']=len(predicted)
        result['mae']=statistics.mean(abs(r['predicted_gain']-r['differences']['total']) for r in predicted)
    else:
        result['reorder_max_abs']=max(abs(r['differences']['reorder']) for r in valid)
        result['reorder_suffix_mismatches']=sum(not r['reorder_suffix_equal'] for r in valid)
    save(OUT/'summaries'/(kind+'.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','per_graph')}),flush=True)
    return result


def batch(kind):
    p=check();n=len(p['entries'][kind])
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        pending={};cursor=0;done_count=0
        for _ in range(min(2,n)):
            f=ex.submit(run_group,kind,cursor);pending[f]=cursor;cursor+=1
        while pending:
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                pending.pop(f)
                try:result=f.result()
                except BaseException:
                    for other in pending:other.cancel()
                    raise
                done_count+=1
                if done_count%4==0 or done_count==n:print(json.dumps(dict(kind=kind,completed=done_count,total=n,**result)),flush=True)
                if cursor<n:
                    new=ex.submit(run_group,kind,cursor);pending[new]=cursor;cursor+=1
    return summarize(kind)


def audit():
    p=check();count=0;max_error=0.;incomplete=[]
    for kind,es in p['entries'].items():
        for e in es:
            path=OUT/'runs'/kind/(e['id']+'.json')
            if not path.exists():incomplete.append(e['id']);continue
            result=read(path)
            if result['status']=='NO_SENSITIVE_PAIR':continue
            if result['status']!='PASS':incomplete.append(e['id']);continue
            source=read(ROOT/e['source'])
            for branch,b in result['branches'].items():
                raw=FiniteEnv(generate(source['spec']))
                for a in source['actions'][:e['operation']]:
                    if not r13.receipt_equal(getattr(raw,a['action'])(*a['args']),a['result']):raise RuntimeError('Audit prefix mismatch')
                raw._budget-=b['charge']['budget']
                for a in b['suffix']:
                    if not r13.receipt_equal(getattr(raw,a['action'])(*a['args']),a['result']):raise RuntimeError('Audit suffix mismatch')
                err=abs(raw.evaluate()-b['final_score']);max_error=max(max_error,err)
                if err>1e-8 or raw.get_remaining_budget()!=b['remaining_budget']:raise RuntimeError('Audit result mismatch')
                count+=1
    r=dict(status='PASS' if not incomplete else 'INCOMPLETE',branches=count,max_forward_error=max_error,incomplete=incomplete,originals=r13.preserved())
    save(OUT/'audit.json',r);print(json.dumps(r),flush=True);return r


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','scan','feedback','audit'])
    args=parser.parse_args()
    with ExclusiveBatch(OUT/'execution.lock'):
        if args.stage=='prepare':prepare()
        elif args.stage=='audit':audit()
        else:batch(args.stage)

if __name__=='__main__':main()
