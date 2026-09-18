"""Execute preregistered cohorts; all results stay outside the submission ZIP."""
import argparse
import json

from analyze_round4 import analyze
from remote_round4_batch import run_batch
from round4_common import OUT, save
from round4_seeds import specs


def screen(batch,workers=3):
    cases=[s['case'] for s in specs() if s['pool'] in ('dev','corr')]
    run_batch(cases,['B0','R1','R2','I1'],batch,workers=workers)
    report=analyze(batch)
    results=dict(report['variants'])
    if any(r.get('missing') or r.get('failures') for r in results.values()):
        raise RuntimeError('Screening incomplete')
    # Combination is conditional on evidence for both independently tested components.
    if results['R2']['mean_delta']>0 and results['I1']['mean_delta']>0:
        combined=batch+'-combined'
        run_batch(cases,['B0','C1'],combined,workers=workers)
        extra=analyze(combined)['variants']['C1']
        if extra['missing'] or extra['failures']:raise RuntimeError('Combination screening incomplete')
        results['C1']=extra
    ranked=sorted(results,key=lambda v:results[v]['mean_delta'],reverse=True)
    selected=[v for v in ranked if results[v]['mean_delta']>0][:2]
    selection={'screen_batch':batch,'selected':selected,
               'criterion':'top two positive proxy paired means; real development must confirm',
               'stats':{v:{k:r[k] for k in ('n','mean_delta','bootstrap_95','wins','losses')} for v,r in results.items()}}
    save(OUT/'screen-selection.json',selection)
    print(json.dumps(selection),flush=True)
    return selection


def development(batch,workers=2):
    selection=json.loads((OUT/'screen-selection.json').read_text(encoding='utf-8'))
    if not selection['selected']:
        save(OUT/'development-status.json',{'status':'no_candidate','retain':'B0','final_set_opened':False})
        return
    cases=[s['case'] for s in specs() if s['pool']=='select']
    run_batch(cases,['B0']+selection['selected'],batch,remote=True,workers=workers)
    analyze(batch,select=True)


def final(batch,workers=2):
    freeze=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    run_batch(freeze['final_cases'],['B0',freeze['variant']],batch,remote=True,
              repeats=freeze['repeats'],workers=workers)
    analyze(batch,final=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['screen','development','final'],required=True)
    p.add_argument('--batch',required=True);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    {'screen':screen,'development':development,'final':final}[a.stage](a.batch,a.workers)
