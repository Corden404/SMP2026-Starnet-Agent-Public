"""Historical public-state sensitivity and omitted-cut diagnosis; no policy training."""
import argparse
import collections
import copy
import itertools
import json
import statistics
import time
from round8_common import ROOT, OUT, load_s48, save

M=load_s48()

def historical_states(limit=24):
    # Old round7 development cases only. Never inspect the old sealed final pools.
    folder=ROOT/'agent_notes/experiments/round7/r7-pilot'
    for path in sorted(folder.glob('*-S48.json'))[:limit]:
        row=json.loads(path.read_text(encoding='utf-8'));nodes={};edges=set()
        for a in row['actions']:
            if a['action']!='scan_node':break
            i=a['args'][0];d=a['result']
            nodes[i]={k:d[k] for k in ('w','persona','comm_left')}
            edges.update(tuple(sorted((i,j))) for j in d['neighbors'])
        mean=statistics.mean(d['w'] for d in nodes.values());unknown=set(range(1,51))-set(nodes)
        full=copy.deepcopy(nodes)
        for i in unknown:full[i]=dict(w=mean,persona='unobserved',comm_left=0)
        yield row['case'],(full,edges,{},{}),set(nodes),sorted(unknown),statistics.pstdev(d['w'] for d in nodes.values())

def run(limit=24):
    results=[]
    for case,state,eligible,unknown,sd in historical_states(limit):
        started=time.monotonic()
        engine=M.RestrictedRollout(*state,M.SEARCH,70,eligible=eligible)
        base=engine.candidates(state,76)
        original=engine.choose(76,baseline_action=base[0])[0]
        first=original['first'];initial_mean=state[0][unknown[0]]['w']
        scenario_choices=[]
        for edge_present in (False,True):
            # Symmetric weight scenarios measure sensitivity, not calibrated probabilities.
            for signs in ((0,0),(-1,-1),(1,1),(-1,1),(1,-1)):
                ns=copy.deepcopy(state[0]);es=set(state[1])
                if edge_present:es.add(tuple(unknown))
                for i,sign in zip(unknown,signs):ns[i]['w']=initial_mean+sign*sd
                actions=engine.candidates((ns,es,{},{}),76)
                scenario_choices.append(dict(edge=edge_present,signs=signs,first=engine.key(actions[0]),
                    baseline_action_gain=next(a['gain'] for a in engine.candidates((ns,es,{},{}),76,True) if engine.key(a)==engine.key(first))))
        cut=next((a for a in base if a['action']=='cut'),None)
        cut_result=None
        if cut and engine.key(cut)!=engine.key(first) and original.get('predicted_final') is not None:
            means={i:engine.effect(i,d,{},{} )[0] for i,d in state[0].items()}
            try:
                suffix=engine.continuation(state,76,70,means,False,prefix=(cut,),deadline=time.monotonic()+2.)
                cut_result=dict(action=cut,delta=suffix['score']-original['predicted_final'])
            except TimeoutError:cut_result=dict(timeout=True)
        record=dict(case=case,baseline_action=engine.key(first),baseline_prediction=original.get('predicted_final'),
            unknown=unknown,scenario_choices=scenario_choices,cut=cut_result,seconds=time.monotonic()-started)
        results.append(record)
        print(json.dumps(dict(case=case,cut=cut_result),ensure_ascii=True),flush=True)
    deltas=[r['cut']['delta'] for r in results if r['cut'] and 'delta' in r['cut']]
    summary=dict(scope='Historical round7 pilot public-state diagnosis; not independent outcome evidence',n=len(results),
        uncertain_first_cases=sum(len({tuple(s['first']) for s in r['scenario_choices']})>1 for r in results),
        cut_completed=len(deltas),cut_better_than_half_point=sum(d>.5 for d in deltas),
        cut_mean_predicted_difference=statistics.mean(deltas) if deltas else None,records=results)
    save(OUT/'historical-diagnostics.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},ensure_ascii=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=24);args=ap.parse_args();run(args.limit)
