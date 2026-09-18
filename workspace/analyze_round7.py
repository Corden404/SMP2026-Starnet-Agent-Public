"""Paired prospective gates. No tuning or rerunning based on individual wins."""
import argparse
import json
import math
import random
import statistics
from experiment_round7 import OUT, specifications, source_hashes
from round6_common import save

def summarize(pool):
    folder=OUT/('r7-'+pool)
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in folder.glob('*.json')]
    base={r['case']:r for r in rows if r['variant']=='D5'}
    result={}
    expected=len(specifications(pool))
    for v in sorted({r['variant'] for r in rows}-{'D5'}):
        rs=sorted([r for r in rows if r['variant']==v and r['case'] in base],key=lambda r:r['case'])
        ds=[r['final_score']-base[r['case']]['final_score'] for r in rs]
        n=len(ds);rng=random.Random(7419)
        boot=sorted(statistics.mean(rng.choices(ds,k=n)) for _ in range(6000))
        families={f:statistics.mean(d for r,d in zip(rs,ds) if r['spec']['family']==f)
                  for f in sorted({r['spec']['family'] for r in rs})}
        wins=sum(d>1e-7 for d in ds);losses=sum(d < -1e-7 for d in ds)
        win_families=len({r['spec']['family'] for r,d in zip(rs,ds) if d>1e-7})
        without_best=statistics.mean(sorted(ds)[:-1]) if n>1 else 0
        tail=statistics.mean(sorted(ds)[:max(1,math.ceil(.1*n))])
        lo,hi=boot[150],boot[5849]
        pilot=statistics.mean(ds)>0 and wins>=3 and win_families>=2 and without_best>0 and tail>=-5
        confirm=lo>0 and tail>=-5
        stress=statistics.mean(ds)>=0 and tail>=-10
        resource_ok=all(r['operation_count']<120 and r['logical_calls']<120 and r['network_attempts']==0 for r in rs)
        result[v]=dict(n=n,expected=expected,complete=n==expected,
            mean=statistics.mean(ds),ci95=[lo,hi],wins=wins,ties=n-wins-losses,losses=losses,
            without_best=without_best,worst_decile=tail,min=min(ds),max=max(ds),families=families,
            mean_seconds=statistics.mean(r['elapsed_seconds'] for r in rs),
            max_seconds=max(r['elapsed_seconds'] for r in rs),resource_ok=resource_ok,
            passed=n==expected and resource_ok and {'pilot':pilot,'confirm':confirm,'stress':stress}[pool],
            changed_action_cases=sum([(a['action'],a['args']) for a in r['actions']]!=
                [(a['action'],a['args']) for a in base[r['case']]['actions']] for r in rs),
            truncations=sum(p.get('plans',[{}])[0].get('search_truncated',False) for r in rs for p in r['search_records'] if p.get('plans')),
            deltas=[dict(case=r['case'],delta=d) for r,d in zip(rs,ds)])
    save(OUT/(pool+'-summary.json'),result)
    print(json.dumps({v:{k:x for k,x in s.items() if k!='deltas'} for v,s in result.items()},ensure_ascii=False,indent=2))
    return result

def freeze(variants):
    for v in variants:
        data=dict(variant=v,source_hashes=source_hashes(v))
        if v=='X1':
            from round7_exchange import LIMITS
            data['limits']=LIMITS
        target=OUT/'candidate-freezes'/(v+'.json')
        if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=data:raise RuntimeError('Candidate drift')
        save(target,data)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pool',default='pilot');ap.add_argument('--freeze',nargs='+')
    args=ap.parse_args()
    if args.freeze:freeze(args.freeze)
    else:summarize(args.pool)
