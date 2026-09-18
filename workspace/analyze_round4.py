"""Predeclared seed-paired analysis; no selection on sealed final results."""
import argparse
from collections import defaultdict
import json
import random
import statistics
import time

from round4_common import OUT, digest, file_hash, save, source_files


def rows_for(batch):
    rows=[]
    for p in (OUT/batch).glob('*-a*.json'):
        row=json.loads(p.read_text(encoding='utf-8'))
        if 'identity' in row and 'variant' in row:rows.append(row)
    return rows


def paired(rows,variant,cases,bootstrap=5000):
    groups=defaultdict(list);failures=[];unresolved=[]
    for row in rows:
        if row['variant'] not in ('B0',variant) or row['case'] not in cases:continue
        if row.get('status')!='completed':
            from remote_round4_batch import transient
            later=[r for r in rows if r.get('status')=='completed' and
                   (r['case'],r['variant'],r.get('repeat',0))==(row['case'],row['variant'],row.get('repeat',0))
                   and r.get('attempt',0)>row.get('attempt',0)
                   and all(r.get('identity',{}).get('definition',{}).get(k)==row.get('identity',{}).get('definition',{}).get(k) for k in ('seed_sha256','model_config','submission_files','harness_files','environment','packages'))]
            resolved=bool(later and (row.get('retryable_infrastructure') and transient(row) or
                                    row.get('status')=='interrupted' and row.get('interrupted_by_batch')))
            fault={'case':row['case'],'variant':row['variant'],'status':row['status'],
                   'error_type':row.get('error_type'),'attempt':row.get('attempt'),
                   'repeat':row.get('repeat',0),'resolved_in_new_session':resolved}
            failures.append(fault)
            if not resolved:unresolved.append(fault)
        else:groups[(row['case'],row['variant'])].append(row)
    data=[];missing=[]
    for case in cases:
        a=groups[(case,'B0')];b=groups[(case,variant)]
        if len({r['repeat'] for r in a})!=len(a) or len({r['repeat'] for r in b})!=len(b):
            raise ValueError('Duplicate completed repetition: '+case)
        if not a or not b:missing.append(case);continue
        if sorted(r['repeat'] for r in a)!=sorted(r['repeat'] for r in b):
            raise ValueError('Unmatched repetitions: '+case)
        for aa,bb in zip(sorted(a,key=lambda r:r['repeat']),sorted(b,key=lambda r:r['repeat'])):
            ia=aa['identity']['definition'];ib=bb['identity']['definition']
            for key in ('seed_sha256','model_config','submission_files','harness_files','environment','packages'):
                if ia[key]!=ib[key]:raise ValueError('Pair identity mismatch '+key+' '+case)
        av=statistics.mean(r['final_score'] for r in a);bv=statistics.mean(r['final_score'] for r in b)
        data.append({'case':case,'baseline':av,'candidate':bv,'delta':bv-av,'spec':b[0]['spec'],
                     'natural':statistics.mean(r['natural_score'] for r in b)})
    if not data:return {'n':0,'missing':missing,'failures':failures,'unresolved_failures':unresolved}
    # Repeat means are one seed observation. Preserve family counts in resampling.
    strata=defaultdict(list)
    for r in data:strata[r['spec']['family']].append(r['delta'])
    rng=random.Random(400917)
    boot=sorted(sum(sum(rng.choices(v,k=len(v))) for v in strata.values())/len(data) for _ in range(bootstrap))
    delta=[r['delta'] for r in data]
    grouped={}
    for factor in ('family','scale','mode','persona_mix'):
        bins=defaultdict(list)
        for r in data:bins[str(r['spec'][factor])].append(r['delta'])
        grouped[factor]={k:{'n':len(v),'mean_delta':statistics.mean(v)} for k,v in bins.items()}
    candidates=[r for r in rows if r.get('status')=='completed' and r['variant']==variant and r['case'] in cases]
    origins=defaultdict(int)
    for r in candidates:
        for e in r.get('chains',[]):origins[e.get('decision_origin','unknown')]+=1
    return {'n':len(data),'variant':variant,'mean_delta':statistics.mean(delta),
            'bootstrap_95':[boot[int(.025*bootstrap)],boot[min(bootstrap-1,int(.975*bootstrap))]],
            'baseline_mean':statistics.mean(r['baseline'] for r in data),
            'candidate_mean':statistics.mean(r['candidate'] for r in data),
            'natural_mean':statistics.mean(r['natural'] for r in data),
            'wins':sum(d>0 for d in delta),'losses':sum(d<0 for d in delta),'ties':sum(d==0 for d in delta),
            'delta_p10':sorted(delta)[int(.1*len(delta))],'worst':min(data,key=lambda r:r['delta']),
            'max_steps':max(r['steps'] for r in candidates),'max_llm_calls':max(r['llm_calls'] for r in candidates),
            'mean_seconds':statistics.mean(r['elapsed_seconds'] for r in candidates),
            'failures':failures,'unresolved_failures':unresolved,'missing':missing,'by_factor':grouped,'decision_origins':dict(origins),
            'per_case':data,'evaluation':candidates[0]['evaluation']}


def analyze(batch,select=False,final=False):
    rows=rows_for(batch)
    protocol=json.loads((OUT/batch/'batch-protocol.json').read_text(encoding='utf-8'))
    cases=protocol['cases'];variants=[v for v in protocol['variants'] if v!='B0']
    report={'batch':batch,'protocol':protocol,'variants':{v:paired(rows,v,cases) for v in variants}}
    save(OUT/batch/'analysis.json',report)
    if select:
        if not protocol['remote'] or not all(c.startswith('select-') for c in cases) or len(cases)!=54:
            raise ValueError('Selection requires all 54 real development seeds')
        if any(s['missing'] or s['unresolved_failures'] for s in report['variants'].values()):
            raise ValueError('Incomplete or failed development cohort')
        best=max(variants,key=lambda v:report['variants'][v]['mean_delta'])
        if report['variants'][best]['mean_delta']<=0:
            report['selection_status']='no_candidate';save(OUT/batch/'analysis.json',report)
            return report
        from experiment_round4 import VARIANTS
        freeze={'variant':best,'runtime':VARIANTS[best],'submission_files':source_files(),
                'harness_files':rows[0]['identity']['definition']['harness_files'],
                'model_config':rows[0]['model_config'],
                'selected_at':time.time(),'development_batch':batch,'development_stats':report['variants'][best],
                'final_cases':[f'final-{i:03}' for i in range(108)],'repeats':1,
                'bootstrap':'5000, family-stratified seed pairs, RNG400917',
                'gate':'mean_delta>0, lower95>0, >=natural, no unresolved failures, below fuse'}
        path=OUT/'selection.json'
        if path.exists():raise RuntimeError('Selection already frozen')
        save(path,freeze)
    if final:
        freeze=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
        if sorted(cases)!=freeze['final_cases'] or variants!=[freeze['variant']] or not protocol['remote']:
            raise ValueError('Final protocol mismatch')
        if freeze['submission_files']!=source_files():raise ValueError('Frozen source changed')
        s=report['variants'][freeze['variant']]
        report['gate_passed']=(not s['missing'] and not s['unresolved_failures'] and s['n']==108 and
            s['mean_delta']>0 and s['bootstrap_95'][0]>0 and s['candidate_mean']>=s['natural_mean'] and
            s['max_steps']<120 and s['max_llm_calls']<120)
        save(OUT/batch/'analysis.json',report)
    print(json.dumps({v:{k:s.get(k) for k in ('n','mean_delta','bootstrap_95','wins','losses','max_steps','max_llm_calls','missing','failures')} for v,s in report['variants'].items()},ensure_ascii=False))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True)
    p.add_argument('--select',action='store_true');p.add_argument('--final',action='store_true')
    a=p.parse_args();analyze(a.batch,a.select,a.final)
