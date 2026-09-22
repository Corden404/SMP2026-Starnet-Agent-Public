"""Prospective action-ranking and candidate-coverage diagnosis on 54 NEW graphs."""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import copy
import json
import statistics
import time
import round16_common as c
from round9_lock import ExclusiveBatch

OUT=c.OUT/'diagnostic'
DEPENDENCIES=('workspace/round16_diagnose.py','tests/test_round16_diagnose.py')


def prepare():
    c.snapshot();c.load()
    rows=[dict(spec=s,sha256=c.digest(c.generate(s))) for s in c.specs('diagnostic')]
    old_hashes=set()
    for name in ('round9','round10-s42','round11-feedback','round12-active-scan','round13-crossed','round13-balanced-stress','round15-coupled'):
        p=c.ROOT/'agent_notes/experiments'/name/'protocol.json'
        for entries in c.read(p).get('seeds',{}).values():
            old_hashes.update(e['sha256'] for e in entries)
    if len({e['sha256'] for e in rows})!=54 or any(e['sha256'] in old_hashes for e in rows):
        raise RuntimeError('Diagnostic seed collision')
    protocol=dict(code=c.fingerprint(DEPENDENCIES),seeds=rows,count=54,workers=2,
        states=['first intervention','first budget<=40','first budget<=20','first budget<=10'],
        candidates='Exact original width2 prefixes plus up to2 positive-ROI candidates per action type, deduplicated. At most6.',
        continuation='One forced public action then byte-identical S44 numerical/CaseVO-fast policy with true simulated receipts; cache cleared for each branch. No hidden state exposed.',
        estimator='Original fixed-mean greedy continuation, complete only within3s per candidate. Truncations retained; no partial values compared.',
        diagnosis_gate='At least5 original-pair sign/ranking errors spanning>=3 graphs =>PC2. Otherwise >=5 states with omitted-candidate true advantage>0.5 spanning>=3 graphs =>coverage. Else stop.',
        inference='States from the same graph are correlated. Oracle selection among forced branches is diagnostic, not deployable gain or global optimal bound.',
        no_api=True,no_upload=True,no_parameter_search=True)
    c.save(OUT/'protocol.json',protocol,True)
    print(json.dumps(dict(prepared=True,graphs=54,originals=c.preserved())),flush=True)


def check():
    p=c.read(OUT/'protocol.json')
    if p['code']!=c.fingerprint(DEPENDENCIES):
        raise RuntimeError('Frozen diagnostic code changed')
    return p


def prefixes(engine,budget):
    candidates=engine.candidates(engine.initial,budget)
    if not candidates:
        return [],[]
    # Exact SEARCH width2, combinations=False candidate order from S44.
    pair=[candidates[0]]
    extras=[]
    for kind in ('comm','shield','cut'):
        extras.extend([a for a in candidates if a['action']==kind][:1])
    extras.extend(candidates[:2])
    seen={engine.key(pair[0])}
    for a in extras:
        if engine.key(a) not in seen:
            pair.append(a);seen.add(engine.key(a))
        if len(pair)>=2:
            break
    pool=list(pair)
    for kind in ('comm','shield','cut'):
        for a in [a for a in candidates if a['action']==kind][:2]:
            if engine.key(a) not in seen:
                pool.append(a);seen.add(engine.key(a))
    return pair,pool


def branch(row,snapshot,action):
    module=c.load();module.coefficients.cache_clear()
    with c.offline():
        _,raw,logged,model=c.restore(row,snapshot['operation'],module)
        if raw.get_remaining_budget()!=snapshot['budget'] or model.llm_counter.call_count!=snapshot['calls']:
            raise RuntimeError('Restored resource state mismatch')
        c.force(model,raw,action)
        result=c.finish(raw,logged,model)
    result['actions']=result['actions'][snapshot['operation']:]
    result.pop('search_records',None)
    result.pop('candidate_records',None)
    module.coefficients.cache_clear()
    return result


def analyze_state(row,snapshot,index):
    module=c.load();module.coefficients.cache_clear()
    with c.offline():
        _,raw,logged,model=c.restore(row,snapshot['operation'],module)
        state=model.completed_state()
        slots=min(model.operation_limit-1-model.operation_count,model.llm_counter.safe_limit-model.llm_counter.call_count)
        engine=module.RestrictedRollout(*state,module.SEARCH,slots,eligible=model.local_nodes)
        pair,pool=prefixes(engine,snapshot['budget'])
        means={i:engine.effect(i,d,state[2],state[3])[0] for i,d in state[0].items()}
        predicted={}
        for action in pool:
            key=repr(engine.key(action))
            try:
                r=engine.continuation(engine.initial,snapshot['budget'],slots,means,False,(action,),time.monotonic()+3.)
                predicted[key]=r['score'] if r is not None else None
            except TimeoutError:
                predicted[key]=None
    results=[]
    order=list(pool)
    if index%2:
        order.reverse()
    for action in order:
        r=branch(row,snapshot,action)
        results.append(dict(action=action,key=list(module.BudgetRollout.key(action)),
            predicted=predicted[repr(module.BudgetRollout.key(action))],result=r))
    lookup={tuple(r['key']):r for r in results}
    pair_rows=[lookup[module.BudgetRollout.key(a)] for a in pair]
    baseline=lookup[module.BudgetRollout.key(snapshot['chosen'])]
    values=[r['result']['final_score'] for r in pair_rows]
    complete=len(pair_rows)==2 and all(r['predicted'] is not None for r in pair_rows)
    dp=(pair_rows[1]['predicted']-pair_rows[0]['predicted']) if complete else None
    dt=(values[1]-values[0]) if len(values)==2 else None
    ranking_error=bool(complete and abs(dp)>1e-8 and abs(dt)>1e-8 and dp*dt<0)
    # A non-greedy first action wins only above S44's original 0.5 prediction margin.
    predicted_index=1 if complete and dp>=.5 else 0
    actual_index=1 if len(values)==2 and dt>.5 else 0
    decision_error=bool(complete and predicted_index!=actual_index)
    best_pair=max(values) if values else baseline['result']['final_score']
    best_pool=max(r['result']['final_score'] for r in results)
    return dict(tags=snapshot['tags'],operation=snapshot['operation'],budget=snapshot['budget'],
        pair=[list(module.BudgetRollout.key(a)) for a in pair],branches=results,
        complete_prediction=complete,predicted_difference=dp,true_difference=dt,
        ranking_error=ranking_error,decision_error=decision_error,
        original_search_truncated=any(p.get('search_truncated',False) for p in snapshot['plans']),
        pair_opportunity=best_pair-baseline['result']['final_score'],
        missed_opportunity=best_pool-best_pair,
        prediction_abs_error=abs(dp-dt) if complete else None,
        baseline_suffix_score_difference=baseline['result']['final_score']-row['final_score'])


def run_graph(index):
    p=check();entry=p['seeds'][index];spec=entry['spec']
    if c.digest(c.generate(spec))!=entry['sha256']:
        raise RuntimeError('Seed hash mismatch')
    path=OUT/'runs'/(spec['case']+'.json')
    identity=dict(entry=entry,code=p['code'])
    if path.exists():
        existing=c.read(path)
        if existing.get('identity')!=identity or existing.get('status')!='PASS':
            raise RuntimeError('Existing incomplete diagnostic needs inspection')
        return dict(case=spec['case'],cached=True)
    if (OUT/'STOP.json').exists():
        raise RuntimeError('Prior diagnostic failure')
    row=dict(status='STARTED',identity=identity,spec=spec,states=[])
    c.save(path,row);started=time.monotonic()
    try:
        base_path=OUT/'baseline'/(spec['case']+'-S44.json')
        if base_path.exists():
            base=c.read(base_path)
            if base.get('identity')!=identity or base.get('status')!='PASS':
                raise RuntimeError('Existing baseline identity mismatch')
        else:
            base=c.run_game(spec,capture=True);base['identity']=identity;c.save(base_path,base,True)
        row['baseline_path']=str(base_path.relative_to(c.ROOT))
        for j,snapshot in enumerate(base['states']):
            row['states'].append(analyze_state(base,snapshot,index+j))
            c.save(path,row)
        row.update(status='PASS',elapsed_seconds=time.monotonic()-started)
        c.save(path,row)
    except BaseException as exc:
        row.update(status='FAILED',error_type=type(exc).__name__);c.save(path,row)
        c.save(OUT/'STOP.json',dict(case=spec['case'],error_type=type(exc).__name__))
        raise
    return dict(case=spec['case'],states=len(row['states']),seconds=row['elapsed_seconds'])


def summarize():
    p=check();rows=[c.read(OUT/'runs'/(e['spec']['case']+'.json')) for e in p['seeds']]
    if any(r['status']!='PASS' for r in rows):
        raise RuntimeError('Incomplete fixed diagnostic')
    states=[(r['spec'],s) for r in rows for s in r['states']]
    errors=[(spec,s) for spec,s in states if s['ranking_error']]
    missed=[(spec,s) for spec,s in states if s['missed_opportunity']>.5]
    pair_gate=len(errors)>=5 and len({s['case'] for s,_ in errors})>=3
    pool_gate=len(missed)>=5 and len({s['case'] for s,_ in missed})>=3
    graph_means=[dict(case=r['spec']['case'],family=r['spec']['family'],
        ranking_errors=sum(s['ranking_error'] for s in r['states']),
        pair_opportunity=statistics.mean(s['pair_opportunity'] for s in r['states']),
        missed_opportunity=statistics.mean(s['missed_opportunity'] for s in r['states'])) for r in rows]
    result=dict(status='PASS',graphs=54,states=len(states),branches=sum(len(s['branches']) for _,s in states),
        ranking_errors=len(errors),ranking_error_graphs=len({s['case'] for s,_ in errors}),
        decision_errors=sum(s['decision_error'] for _,s in states),missed_states=len(missed),
        missed_graphs=len({s['case'] for s,_ in missed}),
        complete_prediction_states=sum(s['complete_prediction'] for _,s in states),
        prediction_mae=statistics.mean(s['prediction_abs_error'] for _,s in states if s['prediction_abs_error'] is not None),
        graph_mean_pair_opportunity=statistics.mean(r['pair_opportunity'] for r in graph_means),
        graph_mean_missed_opportunity=statistics.mean(r['missed_opportunity'] for r in graph_means),
        original_truncated_states=sum(s['original_search_truncated'] for _,s in states),
        baseline_suffix_drift_states=sum(abs(s['baseline_suffix_score_difference'])>1e-8 for _,s in states),
        next_branch='PC2' if pair_gate else 'COVERAGE' if pool_gate else 'STOP',
        scope='Forced-branch diagnostic oracle opportunities, not deployable policy gains. All54 graphs, graph means used.',
        graphs_summary=graph_means,originals=c.preserved())
    c.save(OUT/'summary.json',result,True)
    print(json.dumps({k:v for k,v in result.items() if k!='graphs_summary'}),flush=True)
    return result


def run(start=0,stop=54):
    check();c.preserved()
    if not 0<=start<stop<=54:
        raise ValueError('Invalid fixed block')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        pending={};cursor=start;done_count=0
        while cursor<min(stop,start+2):
            f=executor.submit(run_graph,cursor);pending[f]=cursor;cursor+=1
        while pending:
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                answer=future.result();done_count+=1
                print(json.dumps(dict(completed=done_count,range=[start,stop],**answer)),flush=True)
                if cursor<stop:
                    f=executor.submit(run_graph,cursor);pending[f]=cursor;cursor+=1
    if all((OUT/'runs'/(s['case']+'.json')).exists() for s in c.specs('diagnostic')):
        return summarize()
    return dict(status='FIXED_BLOCK_COMPLETE',range=[start,stop])


def audit():
    p=check();full=branches=0;max_error=0.
    for entry in p['seeds']:
        row=c.read(OUT/'runs'/(entry['spec']['case']+'.json'))
        if row['status']!='PASS':raise RuntimeError('Incomplete audit record')
        base=c.read(c.ROOT/row['baseline_path']);max_error=max(max_error,c.audit_game(base));full+=1
        for state in row['states']:
            for b in state['branches']:
                r=copy.deepcopy(b['result']);r.update(spec=row['spec'],actions=base['actions'][:state['operation']]+r['actions'])
                max_error=max(max_error,c.audit_game(r));branches+=1
    result=dict(status='PASS',full_games=full,branches=branches,max_forward_error=max_error,originals=c.preserved())
    c.save(OUT/'audit.json',result);print(json.dumps(result),flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','summary','audit'])
    p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=54);a=p.parse_args()
    with ExclusiveBatch(OUT/'execution.lock'):
        if a.stage=='prepare':prepare()
        elif a.stage=='run':run(a.start,a.stop)
        elif a.stage=='summary':summarize()
        else:audit()

if __name__=='__main__':main()
