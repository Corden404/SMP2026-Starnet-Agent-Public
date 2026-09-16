"""Analyze recorded results without rerunning or tuning on held-out seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

from experiment_round3 import OUT, SOURCE, VARIANTS, code_hash, save

FAIL_MARKERS=('不确定','无效','不可用','unavailable','Three operations','连续三轮')
DEV_CASES=['dev-002','dev-003','dev-005']
CANDIDATES=['learning','plan','combined','tight','wide','roi_tight']


def load_rows(folder):
    rows=[]
    for p in folder.glob('*.json'):
        r=json.loads(p.read_text(encoding='utf-8-sig'))
        if 'case' in r and 'variant' in r and 'final_score' in r:
            r['result_file']=str(p)
            r['technical_failure']=any(s in r.get('stop_reason','') for s in FAIL_MARKERS)
            rows.append(r)
    return rows


def best_attempts(rows):
    # Earliest technically successful run, never highest score cherry-picking.
    groups={}
    for r in sorted(rows,key=lambda r:r['repeat']):
        key=(r['case'],r['variant'])
        if not r['technical_failure'] and key not in groups:groups[key]=r
    return groups


def paired_stats(pairs):
    deltas=[b['final_score']-a['final_score'] for a,b in pairs]
    rng=random.Random(71234)
    boots=sorted(statistics.mean(rng.choices(deltas,k=len(deltas))) for _ in range(5000))
    return {'n':len(pairs),'baseline_mean':statistics.mean(a['final_score'] for a,b in pairs),
        'candidate_mean':statistics.mean(b['final_score'] for a,b in pairs),
        'paired_delta':statistics.mean(deltas),'bootstrap_95':[boots[125],boots[4875]],
        'wins':sum(d>0 for d in deltas),'losses':sum(d<0 for d in deltas),'ties':sum(d==0 for d in deltas),
        'natural_mean':statistics.mean(b['natural_score'] for a,b in pairs),
        'max_steps':max(b['steps'] for a,b in pairs),'max_llm_calls':max(b['llm_calls'] for a,b in pairs),
        'mean_seconds':statistics.mean(b['elapsed_seconds'] for a,b in pairs),
        'per_case':[{'case':a['case'],'baseline':a['final_score'],'candidate':b['final_score'],'delta':d,
                     'baseline_repeat':a['repeat'],'candidate_repeat':b['repeat']} for (a,b),d in zip(pairs,deltas)]}


def development(select=False):
    rows=load_rows(OUT/'remote')
    groups=best_attempts(rows)
    table={}
    missing=[]
    for variant in CANDIDATES:
        pairs=[]
        for case in DEV_CASES:
            if (case,'score') not in groups or (case,variant) not in groups:
                missing.append((case,variant));continue
            pairs.append((groups[(case,'score')],groups[(case,variant)]))
        if pairs:table[variant]=paired_stats(pairs)
    result={'stage':'real development ablation','comparison_cases':DEV_CASES,'variants':table,
            'technical_failures':[{'case':r['case'],'variant':r['variant'],'repeat':r['repeat'],'reason':r['stop_reason']} for r in rows if r['technical_failure']],
            'missing':missing}
    save(OUT/'development-analysis.json',result)
    print(json.dumps({v:{k:x for k,x in s.items() if k!='per_case'} for v,s in table.items()},indent=2))
    if select:
        if missing:raise RuntimeError('Do not select before all common development runs are complete: '+str(missing))
        winner=max(CANDIDATES,key=lambda v:table[v]['candidate_mean'])
        if table[winner]['paired_delta']<=0:raise RuntimeError('No candidate beats baseline on common development cases')
        selection={'variant':winner,'config':VARIANTS[winner],'source_hash':code_hash(),'selected_at':time.time(),
                   'selected_without_viewing_holdout':True,'development_stats':table[winner],
                   'criterion':'highest real mean over the same three development cases; earliest technically successful attempt',
                   'holdout_cases':[f'holdout-{i:03}' for i in range(20)]}
        path=OUT/'selection.json'
        if path.exists():
            old=json.loads(path.read_text(encoding='utf-8'))
            assert old['variant']==winner and old['source_hash']==selection['source_hash'],'Selection was already frozen'
        else:save(path,selection)
        print('SELECTED '+winner)
    return result


def validation():
    selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    rows=load_rows(OUT/'holdout-remote')
    groups=best_attempts(rows)
    variant=selection['variant']
    pairs=[]
    missing=[]
    for case in selection['holdout_cases']:
        if (case,'score') not in groups or (case,variant) not in groups:
            missing.append(case);continue
        pair=(groups[(case,'score')],groups[(case,variant)])
        assert all(r['source_hash']==selection['source_hash'] for r in pair),'Source drift'
        assert pair[0]['seed_sha256']==pair[1]['seed_sha256'],'Seed mismatch'
        pairs.append(pair)
    if missing:raise RuntimeError('Incomplete validation: '+str(missing))
    stats=paired_stats(pairs)
    stats['variant']=variant
    stats['gate_passed']=stats['paired_delta']>0 and stats['bootstrap_95'][0]>0 and stats['candidate_mean']>=stats['natural_mean']
    stats['technical_failures']=[{'case':r['case'],'variant':r['variant'],'repeat':r['repeat'],'reason':r['stop_reason']} for r in rows if r['technical_failure']]
    stats['source_hash']=selection['source_hash']
    stats['score_type']='official sandbox on held-out synthetic seeds; not Tianchi leaderboard'
    save(OUT/'validation-analysis.json',stats)
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--select',action='store_true')
    p.add_argument('--validate',action='store_true')
    a=p.parse_args()
    validation() if a.validate else development(a.select)
