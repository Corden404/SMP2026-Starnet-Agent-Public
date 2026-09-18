"""Statistics only after ORIGINAL cohorts are complete; INFRA attempts never supply scores."""
from collections import defaultdict
import random
import statistics

from recovery_manifest import read


def summarize(records):
    values={}
    for row in records:
        if row['status']!='COMPLETED':continue
        value=read(row['existing_valid'][0]) if row['existing_valid'] else row['result']
        key=(row['kind'],row['case'],row.get('variant'),row.get('point'),row.get('label'))
        if key in values:raise ValueError('Duplicate logical result')
        values[key]=value
    output={'development_status':'INCOMPLETE','counterfactual_status':'INCOMPLETE','model_selection_performed':False}
    development=[(k,v) for k,v in values.items() if k[0]=='development']
    if len(development)==162:
        grouped=defaultdict(dict)
        for key,value in development:grouped[key[1]][key[2]]=value
        if len(grouped)!=54 or any(set(v)!=set(('B0','R1','I1')) for v in grouped.values()):raise ValueError('Original paired cohort changed')
        results={}
        for variant in ('R1','I1'):
            pairs=[];strata=defaultdict(list)
            for case,arms in sorted(grouped.items()):
                a=arms['B0'];b=arms[variant];delta=b['final_score']-a['final_score']
                pair={'case':case,'baseline':a['final_score'],'candidate':b['final_score'],'delta':delta,'spec':b['spec']}
                pairs.append(pair);strata[b['spec']['family']].append(delta)
            rng=random.Random(400917)
            boots=sorted(sum(sum(rng.choices(x,k=len(x))) for x in strata.values())/54 for _ in range(5000))
            results[variant]={'n':54,'mean_delta':statistics.mean(p['delta'] for p in pairs),
                              'bootstrap_95':[boots[125],boots[4875]],'wins':sum(p['delta']>0 for p in pairs),
                              'losses':sum(p['delta']<0 for p in pairs),'baseline_mean':statistics.mean(p['baseline'] for p in pairs),
                              'candidate_mean':statistics.mean(p['candidate'] for p in pairs),
                              'natural_mean':statistics.mean(v[variant]['natural_score'] for v in grouped.values()),
                              'max_steps':max(v[variant]['steps'] for v in grouped.values()),
                              'max_llm_logical_calls':max(v[variant]['llm_calls'] for v in grouped.values()),
                              'worst':min(pairs,key=lambda p:p['delta']),'per_case':pairs,
                              'by_family':{k:{'n':len(v),'mean_delta':statistics.mean(v)} for k,v in strata.items()}}
        output.update(development_status='COMPLETE',development=results)
    counter=[(k,v) for k,v in values.items() if k[0]=='counterfactual']
    if len(counter)==36:
        grouped=defaultdict(dict)
        for key,value in counter:grouped[(key[1],key[3])][key[4]]=value
        if len(grouped)!=12 or any(set(v)!=set(('A','A-repeat','B')) for v in grouped.values()):raise ValueError('Original branch cohort changed')
        output.update(counterfactual_status='COMPLETE',counterfactual=[{'case':key[0],'point':key[1],
            'AA_delta':arms['A-repeat']['final_score']-arms['A']['final_score'],
            'B_minus_A':arms['B']['final_score']-arms['A']['final_score'],
            'proxy_delta_error':(arms['B']['proxy_final']-arms['A']['proxy_final'])-(arms['B']['final_score']-arms['A']['final_score'])}
            for key,arms in sorted(grouped.items())])
    return output
