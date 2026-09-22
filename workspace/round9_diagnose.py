"""S44 public-state unknown-node sensitivity, not an oracle or score-gain claim."""
import collections
import copy
import json
import math
import random
import statistics
from round9_common import OUT, load, save


def main():
    module=load('S44')
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((OUT/'runs/r9-dev-fast').glob('*-S44-r0.json'))]
    chosen=[];counts=collections.Counter()
    for row in rows:
        f=row['spec']['family']
        if row['status']=='PASS' and counts[f]<2:
            chosen.append(row);counts[f]+=1
    if len(chosen)!=12:raise RuntimeError('Expected two completed development graphs per family')
    results=[]
    for row in chosen:
        scan_results=[a['result']['w'] for a in row['actions'] if a['action']=='scan_node']
        unseen_w=statistics.mean(scan_results)
        sd=statistics.pstdev(scan_results)
        for saved in row['states']:
            nodes={int(i):dict(d) for i,d in saved['nodes'].items()}
            eligible=set(nodes)
            unknown=sorted(set(range(1,51))-eligible-set(saved['dead']))
            for i in unknown:nodes[i]=dict(w=unseen_w,persona='unobserved',comm_left=0)
            edges={tuple(e) for e in saved['edges']}
            effects={int(i):v for i,v in saved['effects'].items()}
            observations={p:{int(i):v for i,v in xs.items()} for p,xs in saved['observations'].items()}
            budget=saved['budget']
            engine=module.RestrictedRollout(nodes,edges,effects,observations,module.SEARCH,70,eligible=eligible)
            base=engine.candidates(engine.initial,budget,negative=True)
            if not base:continue
            actual_raw=row['actions'][saved['operation']]
            kinds={'communicate':'comm','shield_node':'shield','cut_link':'cut'}
            if actual_raw['action'] not in kinds:continue
            args=actual_raw['args'];kind=kinds[actual_raw['action']]
            actual=(kind,args[0],args[1] if kind=='cut' else None)
            observed_edges=sum(u in eligible and v in eligible for u,v in edges)
            density=observed_edges/max(1,len(eligible)*(len(eligible)-1)/2)
            possible=[(u,v) for k,u in enumerate(unknown) for v in unknown[k+1:]]
            rng=random.Random('r9-public-'+row['case']+'-'+saved['tag'])
            shuffled=list(possible);rng.shuffle(shuffled)
            # A structural sensitivity stress, not a posterior sample or inferred true topology.
            extra=set(shuffled[:max(1,min(len(shuffled),math.ceil(density*len(shuffled))))])
            records=[]
            for complete_edges in (False,True):
                es=edges|extra if complete_edges else edges
                for pattern in (0,-1,1,-2,2):
                    ns=copy.deepcopy(nodes)
                    for k,i in enumerate(unknown):
                        sign=pattern if abs(pattern)<=1 else (1 if k%2==0 else -1)*(1 if pattern>0 else -1)
                        ns[i]['w']=unseen_w+sign*sd
                    actions=engine.candidates((ns,es,effects,observations),budget,negative=True)
                    actual_action=next((a for a in actions if engine.key(a)==actual),None)
                    records.append(dict(edge_stress=complete_edges,pattern=pattern,
                        top_immediate=engine.key(actions[0]),actual_gain=actual_action['gain'] if actual_action else None))
            result=dict(case=row['case'],family=row['spec']['family'],tag=saved['tag'],unknown=unknown,
                unknown_possible_edges=len(possible),assumed_extra_edges=sorted(extra),
                baseline_top_immediate=engine.key(base[0]),actual_action=actual,
                top_immediate_sensitive=len({tuple(r['top_immediate']) for r in records})>1,
                actual_structure_gain_crosses_zero=kind!='comm' and any(r['actual_gain'] is not None and r['actual_gain']<0 for r in records),
                scenarios=records)
            results.append(result)
        module.coefficients.cache_clear()
    summary=dict(scope='12 S44 development graphs; public snapshots at start/middle/tail; immediate candidate ranking only, not full-policy regret or calibrated probabilities',
        graphs=len(chosen),states=len(results),sensitive_states=sum(r['top_immediate_sensitive'] for r in results),
        sensitive_graphs=len({r['case'] for r in results if r['top_immediate_sensitive']}),
        structure_gain_crosses_zero=sum(r['actual_structure_gain_crosses_zero'] for r in results),
        by_phase={tag:dict(states=sum(r['tag']==tag for r in results),sensitive=sum(r['tag']==tag and r['top_immediate_sensitive'] for r in results)) for tag in ('start','middle','tail')},
        hidden_attributes_used=False,records=results)
    save(OUT/'unknown-sensitivity.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},ensure_ascii=True))

if __name__=='__main__':main()
