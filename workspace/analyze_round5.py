"""Paired development analysis and fixed stage thresholds for round5."""
import argparse
from collections import Counter, defaultdict
import json
import random
import statistics
from pathlib import Path
from round4_common import ROOT, save

OUT=ROOT/'agent_notes/experiments/round5'


def paired(rows, baseline='B0', candidate='F0'):
    index={(r['case'],r['variant']):r for r in rows}
    pairs=[]
    for case in sorted({r['case'] for r in rows}):
        if (case,baseline) not in index or (case,candidate) not in index:continue
        a,b=index[case,baseline],index[case,candidate]
        aa=[(x['action'],x['args']) for x in a['actions']];bb=[(x['action'],x['args']) for x in b['actions']]
        first=next((i for i,(x,y) in enumerate(zip(aa,bb)) if x!=y),min(len(aa),len(bb)) if len(aa)!=len(bb) else None)
        pairs.append({'case':case,'family':a['spec']['family'],'delta':b['final_score']-a['final_score'],'baseline':a['final_score'],
            'candidate':b['final_score'],'first_divergence':first,'baseline_actions':dict(Counter(x[0] for x in aa)),
            'candidate_actions':dict(Counter(x[0] for x in bb)),'baseline_seconds':a['elapsed_seconds'],'candidate_seconds':b['elapsed_seconds']})
    if not pairs:return {'n':0}
    groups=defaultdict(list)
    for p in pairs:groups[p['family']].append(p['delta'])
    rng=random.Random(20260917);boot=[]
    for _ in range(4000):
        sample=[x for values in groups.values() for x in rng.choices(values,k=len(values))]
        boot.append(statistics.mean(sample))
    boot.sort();deltas=[p['delta'] for p in pairs];ordered=sorted(deltas);tail=max(1,int(len(deltas)*.1))
    worst_baseline=sorted(pairs,key=lambda p:p['baseline'])[:tail]
    result={'n':len(pairs),'baseline':baseline,'candidate':candidate,'mean_delta':statistics.mean(deltas),
        'stratified_bootstrap95':[boot[100],boot[3900]],'by_family':{f:statistics.mean(v) for f,v in groups.items()},
        'worst10_percent_paired_delta':statistics.mean(ordered[:tail]),'worst10_percent_baseline_seed_delta':statistics.mean(p['delta'] for p in worst_baseline),
        'without_best_two_mean':statistics.mean(ordered[:-2]) if len(ordered)>2 else None,'divergent_seeds':sum(p['first_divergence'] is not None for p in pairs),
        'pairs':pairs,'scope':'Historical development seeds; deterministic selector; not hidden-test or final-validation evidence'}
    result['development_gate']=result['mean_delta']>=5 and result['stratified_bootstrap95'][0]>0 and result['without_best_two_mean'] is not None and result['without_best_two_mean']>0 and result['worst10_percent_paired_delta']>=-10
    return result


def main(folder):
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in (OUT/folder).glob('*.json')]
    rows=[r for r in rows if 'variant' in r and 'final_score' in r]
    results={v:paired(rows,candidate=v) for v in sorted({r['variant'] for r in rows}-{'B0'})}
    save(OUT/(folder+'-analysis.json'),results)
    print(json.dumps({v:{k:x for k,x in r.items() if k!='pairs'} for v,r in results.items()},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',default='P2');args=p.parse_args();main(args.folder)
