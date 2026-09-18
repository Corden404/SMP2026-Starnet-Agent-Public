"""Bounded local CPU batch. No sandbox or LLM calls; no final pool access."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import time
from experiment_round5 import run, OUT
from round4_common import save


def one(job):
    case,variant=job
    result=run(case,variant)
    return {k:result[k] for k in ('case','variant','final_score','elapsed_seconds')}


def main(workers):
    validation=json.loads((OUT/'mechanism-validation.json').read_text(encoding='utf-8'))
    if validation['status']!='PASSED':raise RuntimeError('P1 gate not passed')
    jobs=[(f'{pool}-{i:03}',variant) for pool,n in [('dev',108),('corr',72),('select',54)] for i in range(n) for variant in ['B0','F0']]
    timed=[one(job) for job in jobs[:4]]
    save(OUT/'P2-timing.json',{'timing_samples':timed,'remaining_estimate_seconds':sum(x['elapsed_seconds'] for x in timed)/4*(len(jobs)-4)/workers,'workers':workers})
    print(json.dumps({'timing':timed}),flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index,result in enumerate(pool.map(one,jobs[4:])):
            if (index+1)%20==0:print(json.dumps({'completed':index+5,'last':result}),flush=True)
    from analyze_round5 import main as analyze
    analyze('P2')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);args=p.parse_args();main(args.workers)
