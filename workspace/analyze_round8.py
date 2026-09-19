"""Seed-level paired summaries; repeat runs are not independent seeds."""
import argparse
import collections
import json
import math
import statistics
import numpy as np
from round8_common import OUT, save

def summarize(pool, mode='fast'):
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in (OUT/('r8-'+pool+'-'+mode)).glob('*.json')]
    rows=[r for r in rows if r.get('status')=='PASS']
    grouped=collections.defaultdict(list)
    for r in rows:grouped[(r['case'],r['variant'])].append(r)
    base={case:rs for (case,v),rs in grouped.items() if v=='S48'}
    output={}
    for v in sorted({r['variant'] for r in rows}-{'S48'}):
        pairs=[]
        for case, bs in sorted(base.items()):
            rs=grouped.get((case,v),[])
            if not rs or {r['repeat'] for r in rs}!={r['repeat'] for r in bs}:continue
            pairs.append((rs[0]['spec']['family'],statistics.mean(r['final_score'] for r in rs)-statistics.mean(r['final_score'] for r in bs),case))
        if not pairs: continue
        deltas=[d for f,d,c in pairs];rng=np.random.default_rng(8192026);n=len(deltas)
        boot=np.zeros(6000)
        for f in sorted({f for f,d,c in pairs}):
            ds=np.array([d for ff,d,c in pairs if f==ff]);boot+=rng.choice(ds,(6000,len(ds)),replace=True).sum(axis=1)/n
        rs=[r for r in rows if r['variant']==v]
        output[v]=dict(n=n,mean=statistics.mean(deltas),ci95=np.quantile(boot,[.025,.975]).tolist(),
            sd=statistics.stdev(deltas) if n>1 else 0,wins=sum(d>1e-7 for d in deltas),ties=sum(abs(d)<=1e-7 for d in deltas),losses=sum(d< -1e-7 for d in deltas),
            without_best_two=statistics.mean(sorted(deltas)[:-2]) if n>2 else None,
            worst_decile=statistics.mean(sorted(deltas)[:max(1,math.ceil(n*.1))]),min=min(deltas),max=max(deltas),
            families={f:statistics.mean(d for ff,d,c in pairs if ff==f) for f in sorted({f for f,d,c in pairs})},
            mean_seconds=statistics.mean(r['elapsed_seconds'] for r in rs),max_seconds=max(r['elapsed_seconds'] for r in rs),
            max_operations=max(r['operation_count'] for r in rs),max_calls=max(r['logical_calls'] for r in rs),
            truncations=sum(bool(p.get('search_truncated')) for r in rs for s in r['search_records'] for p in s['plans'][:1]),
            deltas=[dict(case=c,family=f,delta=d) for f,d,c in pairs])
    save(OUT/(pool+'-'+mode+'-summary.json'),output)
    return output

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pool',default='dev');ap.add_argument('--mode',default='fast');args=ap.parse_args()
    result=summarize(args.pool,args.mode)
    print(json.dumps({v:{k:x for k,x in r.items() if k!='deltas'} for v,r in result.items()},ensure_ascii=True,indent=2))
